"""Per-process state and the per-call guard every tool and resource goes through.

``McpRuntime.call`` (an async context manager):

1. authenticates the API token **on every call**: over stdio the token from the
   environment, over HTTP the ``Authorization: Bearer`` header of *this* request (never
   anything cached from an earlier request), so revocation, expiry and deletion apply from
   the next call on;
2. checks the token's scope (``read``, ``review`` or ``upload``);
3. yields the principal and a database session for the service layer, which re-checks
   workspace membership and role on every call (a token never raises a role);
4. maps errors: known service errors become MCP errors carrying their code and message;
   anything unexpected is logged (stderr, with a reference) and reported generically.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ResourceError, ResourceNotFoundError, ToolError
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_core.config import Settings
from compliance_core.rule_packs import RulePackValidationError
from compliance_core.runtime import ServiceContext
from compliance_core.services import api_tokens
from compliance_core.services.api_tokens import TokenPrincipal
from compliance_core.services.errors import (
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitedError,
    ServiceError,
)
from compliance_core.uploads import UploadError
from compliance_llm import ProviderResolver, configured_providers
from compliance_mcp_server.config import McpSettings
from compliance_retrieval.config import RetrievalConfig
from compliance_retrieval.service import SearchDependencies

logger = logging.getLogger("compliance_mcp_server")

Transport = Literal["stdio", "http"]
Scope = Literal["read", "review", "upload"]
McpContext = Context[Any, Any]
AUTH_FAILED = (
    "unauthenticated: The API token is missing, invalid, expired or revoked. Create a new "
    "token on the web app's Settings page."
)


@dataclass(slots=True, kw_only=True)
class McpServiceContext(ServiceContext):
    retrieval: RetrievalConfig

    @classmethod
    def create(cls, settings: Settings) -> McpServiceContext:
        return cls(
            **ServiceContext.shared_resources(settings),
            retrieval=RetrievalConfig.from_settings(settings),
        )


@dataclass
class McpRuntime:
    transport: Transport
    settings: Settings
    mcp: McpSettings
    # stdio only: the token from the environment. HTTP never uses it.
    stdio_token: str | None = None
    _ctx: McpServiceContext | None = field(default=None, repr=False)

    @property
    def ctx(self) -> McpServiceContext:
        # Created lazily: building it opens no connection, and tests that never call a
        # tool need no database.
        if self._ctx is None:
            self._ctx = McpServiceContext.create(self.settings)
        return self._ctx

    @property
    def search(self) -> SearchDependencies:
        ctx = self.ctx
        return SearchDependencies(
            session_factory=ctx.session_factory, models=ctx.models, config=ctx.retrieval
        )

    @property
    def resolver(self) -> ProviderResolver:
        # The worker owns provider credentials; this process only knows which providers
        # it was told the worker can use.
        return ProviderResolver(
            self.settings, configured_providers(self.mcp.credentialed_providers)
        )

    async def close(self) -> None:
        if self._ctx is not None:
            await self._ctx.close()
            self._ctx = None

    def presented_token(self, ctx: McpContext | None) -> str | None:
        if self.transport == "stdio":
            return self.stdio_token
        headers = ctx.headers if ctx is not None else None
        if headers is None:
            return None
        scheme, _, value = (headers.get("authorization") or "").partition(" ")
        return value.strip() if scheme.lower() == "bearer" else None

    async def authenticate(self, raw: str | None) -> TokenPrincipal:
        if not api_tokens.looks_like_token((raw or "").strip()):
            raise AuthenticationError(api_tokens.INVALID_TOKEN)  # no database round trip
        async with self.ctx.session_factory() as session:
            return await api_tokens.authenticate(session, raw)

    @asynccontextmanager
    async def call(
        self,
        ctx: McpContext | None,
        scope: Scope,
        *,
        name: str,
        kind: Literal["tool", "resource"] = "tool",
    ) -> AsyncIterator[tuple[TokenPrincipal, AsyncSession]]:
        error: type[ToolError] | type[ResourceError] = (
            ToolError if kind == "tool" else ResourceError
        )
        try:
            principal = await self.authenticate(self.presented_token(ctx))
            if scope not in principal.scopes:
                raise PermissionDeniedError(
                    f"This API token lacks the '{scope}' scope. Create a token with it on the "
                    "web app's Settings page."
                )
            async with self.ctx.session_factory() as session:
                try:
                    yield principal, session
                except BaseException:
                    await session.rollback()
                    raise
        except AuthenticationError as exc:
            raise error(AUTH_FAILED) from exc
        except NotFoundError as exc:
            if kind == "resource":
                raise ResourceNotFoundError(f"{exc.code}: {exc.message}") from exc
            raise ToolError(f"{exc.code}: {exc.message}") from exc
        except RateLimitedError as exc:
            raise error(
                f"{exc.code}: {exc.message} Retry after {exc.retry_after_seconds} s."
            ) from exc
        except RulePackValidationError as exc:
            raise error(f"{exc.code}: {exc.message} {'; '.join(exc.errors[:5])}") from exc
        except ServiceError as exc:
            raise error(f"{exc.code}: {exc.message}") from exc
        except UploadError as exc:
            raise error(f"{exc.code}: {exc}") from exc
        except (ToolError, ResourceError):
            raise
        except Exception as exc:
            reference = uuid.uuid4().hex[:12]
            logger.exception(
                "unexpected MCP error", extra={"mcp_call": name, "error_reference": reference}
            )
            raise error(
                f"internal_error: An unexpected error occurred (reference {reference})."
            ) from exc


def require_workspace(principal: TokenPrincipal, workspace_id: uuid.UUID | None) -> None:
    """The token's workspace allowlist. Outside it, ids look exactly like missing ones."""
    if workspace_id is not None and not principal.allows_workspace(workspace_id):
        raise NotFoundError("Workspace not found.")
