"""MCP API tokens: Settings endpoints (create once, list, revoke) and authentication."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select, update

from compliance_core.db.models import ApiToken
from compliance_core.services import api_tokens
from compliance_core.services.errors import AuthenticationError, InvalidOperationError
from compliance_core.token_schemas import ApiTokenCreate

pytestmark = pytest.mark.integration
URL = "/api/v1/settings/api-tokens"


async def _workspace(client: Any, name: str = "Tokens") -> str:
    response = await client.post("/api/v1/workspaces", json={"name": name})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def test_create_shows_secret_once_with_safe_defaults(clients: Any, api_app: Any) -> None:
    client = await clients.signed_up("tokens@example.com")
    response = await client.post(URL, json={"name": "Claude Desktop"})
    assert response.status_code == 201, response.text
    assert response.headers["cache-control"] == "no-store"
    created = response.json()
    token = created["token"]
    assert token.startswith("cwmcp_")
    assert len(token) == 49
    assert created["prefix"] == token[:14]
    assert created["scopes"] == ["read"]
    assert created["workspace_ids"] is None
    assert created["status"] == "active"
    lifetime = datetime.fromisoformat(created["expires_at"]) - datetime.fromisoformat(
        created["created_at"]
    )
    assert lifetime == timedelta(days=90)

    listed = await client.get(URL)
    assert listed.status_code == 200
    assert listed.headers["cache-control"] == "no-store"
    [item] = listed.json()
    assert "token" not in item
    assert item["id"] == created["id"]
    assert token not in listed.text

    async with api_app.state.ctx.session_factory() as session:
        row = await session.scalar(select(ApiToken).where(ApiToken.id == uuid.UUID(item["id"])))
    assert row is not None
    assert row.token_hash == api_tokens.hash_token(token)
    assert token not in (row.token_hash, row.token_prefix)


async def test_scopes_allowlist_and_validation(clients: Any) -> None:
    client = await clients.signed_up("scopes@example.com")
    other = await clients.signed_up("someone-else@example.com")
    mine = await _workspace(client)
    theirs = await _workspace(other, "Theirs")

    response = await client.post(
        URL,
        json={
            "name": "Reviewer bot",
            "scopes": ["review", "upload"],
            "workspace_ids": [mine],
            "expires_in_days": 7,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["scopes"] == ["read", "review", "upload"]  # read always granted
    assert body["workspace_ids"] == [mine]

    # A workspace the caller is not a member of looks missing.
    response = await client.post(URL, json={"name": "x", "workspace_ids": [theirs]})
    assert response.status_code == 404
    for bad in (
        {"name": "x", "scopes": ["admin"]},
        {"name": "x", "expires_in_days": 0},
        {"name": "x", "expires_in_days": 366},
        {"name": "   "},
        {"name": "x", "workspace_ids": []},
        {"name": "x", "token": "cwmcp_chosen"},
    ):
        assert (await client.post(URL, json=bad)).status_code == 422, bad


async def test_management_needs_session_and_csrf(clients: Any) -> None:
    client = await clients.signed_up("csrf@example.com")
    anonymous = clients.new()
    assert (await anonymous.get(URL)).status_code == 401
    assert (await anonymous.post(URL, json={"name": "x"})).status_code == 401
    response = await client.post(URL, json={"name": "x"}, headers={"X-CSRF-Token": "forged"})
    assert response.status_code == 403
    response = await client.delete(f"{URL}/{uuid.uuid4()}", headers={"X-CSRF-Token": "forged"})
    assert response.status_code == 403


async def test_revoke_is_owner_only_and_idempotent(clients: Any, api_app: Any) -> None:
    alice = await clients.signed_up("alice-tokens@example.com")
    bob = await clients.signed_up("bob-tokens@example.com")
    created = (await alice.post(URL, json={"name": "a"})).json()
    token_id = created["id"]

    assert (await bob.delete(f"{URL}/{token_id}")).status_code == 404
    response = await alice.delete(f"{URL}/{token_id}")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "revoked"
    again = await alice.delete(f"{URL}/{token_id}")
    assert again.status_code == 200
    assert again.json()["revoked_at"] == response.json()["revoked_at"]
    assert (await alice.delete(f"{URL}/{uuid.uuid4()}")).status_code == 404

    async with api_app.state.ctx.session_factory() as session:
        with pytest.raises(AuthenticationError):
            await api_tokens.authenticate(session, created["token"])


async def test_active_token_cap(clients: Any, api_app: Any) -> None:
    client = await clients.signed_up("cap@example.com")
    me = (await client.get("/api/v1/auth/me")).json()
    async with api_app.state.ctx.session_factory() as session:
        for i in range(api_tokens.MAX_ACTIVE_TOKENS):
            session.add(
                ApiToken(
                    user_id=uuid.UUID(me["id"]),
                    name=f"t{i}",
                    token_prefix="cwmcp_x",
                    token_hash=uuid.uuid4().hex * 2,
                    scopes=["read"],
                    expires_at=datetime.now(UTC) + timedelta(days=1),
                )
            )
        await session.commit()
    response = await client.post(URL, json={"name": "one too many"})
    assert response.status_code == 409
    assert "Revoke one first" in response.json()["error"]["message"]


async def test_authenticate_rejects_everything_but_a_live_token(clients: Any, api_app: Any) -> None:
    client = await clients.signed_up("authn@example.com")
    created = (await client.post(URL, json={"name": "a", "scopes": ["review"]})).json()
    raw = created["token"]
    sessions = api_app.state.ctx.session_factory

    async with sessions() as session:
        principal = await api_tokens.authenticate(session, raw)
    assert principal.scopes == frozenset({"read", "review"})
    assert principal.workspace_ids is None
    async with sessions() as session:
        first_use = await session.scalar(
            select(ApiToken.last_used_at).where(ApiToken.id == principal.token_id)
        )
    assert first_use is not None

    unknown = "cwmcp_" + "A" * 43
    for bad in (None, "", raw + "x", raw[:-1], raw.upper(), unknown, f"Bearer {raw}"):
        async with sessions() as session:
            with pytest.raises(AuthenticationError) as info:
                await api_tokens.authenticate(session, bad)
        assert info.value.message == api_tokens.INVALID_TOKEN  # one message for every case

    async with sessions() as session:
        await session.execute(
            update(ApiToken)
            .where(ApiToken.id == principal.token_id)
            .values(created_at=datetime.now(UTC) - timedelta(days=2), expires_at=datetime.now(UTC))
        )
        await session.commit()
    async with sessions() as session:
        with pytest.raises(AuthenticationError):
            await api_tokens.authenticate(session, raw)
    listed = (await client.get(URL)).json()
    assert listed[0]["status"] == "expired"


async def test_rest_api_never_accepts_api_tokens(clients: Any) -> None:
    client = await clients.signed_up("rest@example.com")
    raw = (await client.post(URL, json={"name": "a"})).json()["token"]
    anonymous = clients.new()
    response = await anonymous.get("/api/v1/workspaces", headers={"Authorization": f"Bearer {raw}"})
    assert response.status_code == 401


async def test_active_token_cap_holds_under_concurrent_creation(clients: Any, api_app: Any) -> None:
    client = await clients.signed_up("race@example.com")
    user_id = uuid.UUID((await client.get("/api/v1/auth/me")).json()["id"])
    sessions = api_app.state.ctx.session_factory
    async with sessions() as session:
        for i in range(api_tokens.MAX_ACTIVE_TOKENS - 1):
            session.add(
                ApiToken(
                    user_id=user_id,
                    name=f"t{i}",
                    token_prefix="cwmcp_x",
                    token_hash=uuid.uuid4().hex * 2,
                    scopes=["read"],
                    expires_at=datetime.now(UTC) + timedelta(days=1),
                )
            )
        await session.commit()

    contenders = [sessions() for _ in range(4)]
    for session in contenders:  # warm connections, so the requests really overlap
        await session.execute(select(1))
        await session.rollback()

    async def create(session: Any) -> str:
        try:
            await api_tokens.create_token(session, user_id, ApiTokenCreate(name="concurrent"))
        except InvalidOperationError:
            await session.rollback()
            return "refused"
        return "created"

    try:
        outcomes = await asyncio.gather(*(create(s) for s in contenders))
    finally:
        for session in contenders:
            await session.close()
    assert sorted(outcomes) == ["created", "refused", "refused", "refused"]
    async with sessions() as session:
        active = await session.scalar(
            select(func.count())
            .select_from(ApiToken)
            .where(ApiToken.user_id == user_id, ApiToken.revoked_at.is_(None))
        )
    assert active == api_tokens.MAX_ACTIVE_TOKENS
