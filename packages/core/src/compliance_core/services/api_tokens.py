"""Per-user API tokens for the MCP server.

- A token is ``cwmcp_`` plus 43 base64url characters (256 random bits). Only its SHA-256
  hash is stored; the plaintext is returned once, by :func:`create_token`.
- Scopes: ``read`` is always granted; ``review`` (start runs and rule checks, which spend
  budget) and ``upload`` are explicit opt-ins. A token never raises a role: the service
  layer still checks workspace membership and role on every call, so the effective access
  is role ∩ scopes ∩ workspace allowlist.
- Tokens always expire (at most a year) and can be revoked. :func:`authenticate` gives the
  same error for unknown, malformed, expired and revoked tokens, and is called on every
  MCP tool call, so revocation and expiry take effect immediately.
- The REST API never accepts these tokens (it is cookie-session only).
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_core.db.models import API_TOKEN_SCOPES, ApiToken, User, WorkspaceMember
from compliance_core.services.errors import (
    AuthenticationError,
    InvalidOperationError,
    NotFoundError,
)
from compliance_core.token_schemas import ApiTokenCreate, ApiTokenCreated, ApiTokenRead

TOKEN_PREFIX = "cwmcp_"  # noqa: S105 - a public marker, not a secret
_TOKEN_RE = re.compile(r"^cwmcp_[A-Za-z0-9_-]{43}$")
DISPLAY_PREFIX_CHARS = 14
MAX_ACTIVE_TOKENS = 20
LIST_LIMIT = 100
LAST_USED_RESOLUTION = timedelta(seconds=60)
INVALID_TOKEN = "The API token is invalid, expired or revoked."  # noqa: S105 - a message
TOKEN_NOT_FOUND = "API token not found."  # noqa: S105 - a message


@dataclass(frozen=True)
class TokenPrincipal:
    """Who a valid token acts for, and what it may do (before role checks)."""

    token_id: uuid.UUID
    user_id: uuid.UUID
    scopes: frozenset[str]
    workspace_ids: frozenset[uuid.UUID] | None
    expires_at: datetime

    def allows_workspace(self, workspace_id: uuid.UUID) -> bool:
        return self.workspace_ids is None or workspace_id in self.workspace_ids


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def looks_like_token(raw: str) -> bool:
    return bool(_TOKEN_RE.fullmatch(raw))


def _status(token: ApiToken, now: datetime) -> str:
    if token.revoked_at is not None:
        return "revoked"
    return "active" if token.expires_at > now else "expired"


def _read(token: ApiToken, now: datetime) -> ApiTokenRead:
    return ApiTokenRead(
        id=token.id,
        name=token.name,
        prefix=token.token_prefix,
        scopes=list(token.scopes),  # type: ignore[arg-type]
        workspace_ids=list(token.workspace_ids) if token.workspace_ids is not None else None,
        created_at=token.created_at,
        expires_at=token.expires_at,
        last_used_at=token.last_used_at,
        revoked_at=token.revoked_at,
        status=_status(token, now),  # type: ignore[arg-type]
    )


async def _now(session: AsyncSession) -> datetime:
    # Database time, the same clock the expiry comparisons in SQL use.
    now = await session.scalar(select(func.now()))
    assert isinstance(now, datetime)  # noqa: S101 - now() is never NULL
    return now


async def create_token(
    session: AsyncSession, user_id: uuid.UUID, payload: ApiTokenCreate
) -> ApiTokenCreated:
    # Serialise creation per user (row lock held until commit), so concurrent requests
    # cannot all pass the active-token count below and exceed the cap.
    locked = await session.scalar(select(User.id).where(User.id == user_id).with_for_update())
    if locked is None:
        raise NotFoundError("User not found.")
    active = await session.scalar(
        select(func.count())
        .select_from(ApiToken)
        .where(
            ApiToken.user_id == user_id,
            ApiToken.revoked_at.is_(None),
            ApiToken.expires_at > func.now(),
        )
    )
    if int(active or 0) >= MAX_ACTIVE_TOKENS:
        raise InvalidOperationError(
            f"You already have {MAX_ACTIVE_TOKENS} active API tokens. Revoke one first."
        )
    workspace_ids = None
    if payload.workspace_ids is not None:
        workspace_ids = list(dict.fromkeys(payload.workspace_ids))
        member_of = set(
            await session.scalars(
                select(WorkspaceMember.workspace_id).where(
                    WorkspaceMember.user_id == user_id,
                    WorkspaceMember.workspace_id.in_(workspace_ids),
                )
            )
        )
        if member_of != set(workspace_ids):
            raise NotFoundError("Workspace not found.")
    wanted = set(payload.scopes) | {"read"}
    scopes = [s for s in API_TOKEN_SCOPES if s in wanted]
    raw = TOKEN_PREFIX + secrets.token_urlsafe(32)
    now = await _now(session)
    token = ApiToken(
        user_id=user_id,
        name=payload.name,
        token_prefix=raw[:DISPLAY_PREFIX_CHARS],
        token_hash=hash_token(raw),
        scopes=scopes,
        workspace_ids=workspace_ids,
        created_at=now,
        expires_at=now + timedelta(days=payload.expires_in_days),
    )
    session.add(token)
    await session.flush()
    await session.refresh(token)
    await session.commit()
    return ApiTokenCreated(**_read(token, now).model_dump(), token=raw)


async def list_tokens(session: AsyncSession, user_id: uuid.UUID) -> list[ApiTokenRead]:
    now = await _now(session)
    tokens = await session.scalars(
        select(ApiToken)
        .where(ApiToken.user_id == user_id)
        .order_by(ApiToken.created_at.desc(), ApiToken.id)
        .limit(LIST_LIMIT)
    )
    return [_read(t, now) for t in tokens]


async def revoke_token(
    session: AsyncSession, user_id: uuid.UUID, token_id: uuid.UUID
) -> ApiTokenRead:
    """Revoke the caller's own token (idempotent). Other users' tokens are not found."""
    await session.execute(
        update(ApiToken)
        .where(
            ApiToken.id == token_id,
            ApiToken.user_id == user_id,
            ApiToken.revoked_at.is_(None),
        )
        .values(revoked_at=func.now())
    )
    await session.commit()
    token = await session.scalar(
        select(ApiToken).where(ApiToken.id == token_id, ApiToken.user_id == user_id)
    )
    if token is None:
        raise NotFoundError(TOKEN_NOT_FOUND)
    return _read(token, await _now(session))


async def authenticate(session: AsyncSession, raw: str | None) -> TokenPrincipal:
    """Resolve a presented token to its principal, or raise ``AuthenticationError``.

    Call on every request. The plaintext is never logged or stored; lookups use its hash.
    """
    raw = (raw or "").strip()
    if not looks_like_token(raw):
        raise AuthenticationError(INVALID_TOKEN)
    digest = hash_token(raw)
    token = await session.scalar(
        select(ApiToken).where(
            ApiToken.token_hash == digest,
            ApiToken.revoked_at.is_(None),
            ApiToken.expires_at > func.now(),
        )
    )
    if token is None or not hmac.compare_digest(token.token_hash, digest):
        await session.rollback()
        raise AuthenticationError(INVALID_TOKEN)
    principal = TokenPrincipal(
        token_id=token.id,
        user_id=token.user_id,
        scopes=frozenset(token.scopes),
        workspace_ids=frozenset(token.workspace_ids) if token.workspace_ids is not None else None,
        expires_at=token.expires_at,
    )
    # Coarse last-used tracking (at most one write per token per minute).
    await session.execute(
        update(ApiToken)
        .where(
            ApiToken.id == token.id,
            ApiToken.last_used_at.is_(None)
            | (ApiToken.last_used_at < func.now() - LAST_USED_RESOLUTION),
        )
        .values(last_used_at=func.now())
    )
    await session.commit()
    return principal
