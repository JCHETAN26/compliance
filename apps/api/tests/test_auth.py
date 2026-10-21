"""Auth flow tests against real Postgres + Redis through the ASGI app."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from compliance_core.security.tokens import TokenSigner

pytestmark = pytest.mark.integration

PASSWORD = "correct horse battery"


async def test_register_sets_secure_session_cookies(clients: Any) -> None:
    client = clients.new()
    response = await client.post(
        "/api/v1/auth/register", json={"email": "  Alice@Example.COM ", "password": PASSWORD}
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["user"]["email"] == "alice@example.com"
    assert "password" not in response.text
    assert "password_hash" not in response.text
    set_cookies = response.headers.get_list("set-cookie")
    access = next(c for c in set_cookies if c.startswith("cw_access="))
    csrf = next(c for c in set_cookies if c.startswith("cw_csrf="))
    assert "HttpOnly" in access
    assert "SameSite=lax" in access
    assert "Path=/" in access
    assert "HttpOnly" not in csrf
    assert body["csrf_token"] == client.cookies["cw_csrf"]
    assert response.headers["cache-control"] == "no-store"

    me = await client.get("/api/v1/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.com"


async def test_duplicate_email_conflicts_case_insensitively(clients: Any) -> None:
    await clients.signed_up("bob@example.com")
    response = await clients.new().post(
        "/api/v1/auth/register", json={"email": "BOB@example.com", "password": PASSWORD}
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "not-an-email", "password": PASSWORD},
        {"email": "a@example.com", "password": "Zq9!xW"},
        {"email": "a@example.com", "password": "x" * 129},
        {"email": "a@example.com"},
        {"email": "a@example.com", "password": PASSWORD, "role": "admin"},
    ],
)
async def test_register_validation(clients: Any, payload: dict[str, str]) -> None:
    response = await clients.new().post("/api/v1/auth/register", json=payload)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]
    # Submitted values (which may include passwords) are never echoed back.
    assert all("input" not in detail for detail in error["details"])
    if "password" in payload:
        assert payload["password"] not in response.text


async def test_login_logout_and_token_revocation(clients: Any) -> None:
    await clients.signed_up("carol@example.com")
    client = clients.new()
    wrong = await client.post(
        "/api/v1/auth/login", json={"email": "carol@example.com", "password": "nope-nope"}
    )
    assert wrong.status_code == 401
    assert wrong.json()["error"] == {
        "code": "unauthenticated",
        "message": "Invalid email or password.",
    }

    login = await client.post(
        "/api/v1/auth/login", json={"email": "CAROL@example.com", "password": PASSWORD}
    )
    assert login.status_code == 200
    stolen_cookie = client.cookies["cw_access"]
    csrf = login.json()["csrf_token"]

    # Logout without the CSRF header is refused.
    refused = await client.post("/api/v1/auth/logout")
    assert refused.status_code == 403
    assert refused.json()["error"]["message"] == "Missing or invalid CSRF token."

    logout = await client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": csrf})
    assert logout.status_code == 204
    assert "cw_access" not in client.cookies
    assert (await client.get("/api/v1/auth/me")).status_code == 401

    # A copy of the token captured before logout is revoked server-side.
    replay = clients.new()
    replay.cookies.set("cw_access", stolen_cookie)
    assert (await replay.get("/api/v1/auth/me")).status_code == 401


async def test_unknown_email_gets_same_error_as_wrong_password(clients: Any) -> None:
    response = await clients.new().post(
        "/api/v1/auth/login", json={"email": "ghost@example.com", "password": PASSWORD}
    )
    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Invalid email or password."


async def test_failed_logins_are_throttled_per_account(clients: Any) -> None:
    await clients.signed_up("dave@example.com")
    client = clients.new()
    # test_settings sets login_max_failures=3.
    for _ in range(3):
        response = await client.post(
            "/api/v1/auth/login", json={"email": "dave@example.com", "password": "wrong-pass"}
        )
        assert response.status_code == 401
    blocked = await client.post(
        "/api/v1/auth/login", json={"email": "dave@example.com", "password": PASSWORD}
    )
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "rate_limited"
    assert 0 < int(blocked.headers["retry-after"]) <= 60
    # Other accounts are unaffected.
    await clients.signed_up("erin@example.com")
    other = await client.post(
        "/api/v1/auth/login", json={"email": "erin@example.com", "password": PASSWORD}
    )
    assert other.status_code == 200


async def test_successful_login_resets_failure_counter(clients: Any) -> None:
    await clients.signed_up("frank@example.com")
    client = clients.new()
    for _ in range(2):
        await client.post(
            "/api/v1/auth/login", json={"email": "frank@example.com", "password": "wrong-pass"}
        )
    ok = await client.post(
        "/api/v1/auth/login", json={"email": "frank@example.com", "password": PASSWORD}
    )
    assert ok.status_code == 200
    for _ in range(2):
        await client.post(
            "/api/v1/auth/login", json={"email": "frank@example.com", "password": "wrong-pass"}
        )
    again = await client.post(
        "/api/v1/auth/login", json={"email": "frank@example.com", "password": PASSWORD}
    )
    assert again.status_code == 200


async def test_relogin_revokes_previous_session(clients: Any) -> None:
    client = await clients.signed_up("gina@example.com")
    first = client.cookies["cw_access"]
    login = await client.post(
        "/api/v1/auth/login", json={"email": "gina@example.com", "password": PASSWORD}
    )
    assert login.status_code == 200
    assert client.cookies["cw_access"] != first
    replay = clients.new()
    replay.cookies.set("cw_access", first)
    assert (await replay.get("/api/v1/auth/me")).status_code == 401


@pytest.mark.parametrize(
    "cookie",
    ["", "garbage", "a.b.c", "eyJhbGciOiJub25lIn0.eyJzdWIiOiIxIn0."],
)
async def test_invalid_cookies_are_unauthenticated(clients: Any, cookie: str) -> None:
    client = clients.new()
    if cookie:
        client.cookies.set("cw_access", cookie)
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthenticated"


async def test_expired_and_foreign_tokens_rejected(clients: Any, api_app: Any) -> None:
    client = await clients.signed_up("hank@example.com")
    user_id = (await client.get("/api/v1/auth/me")).json()["id"]
    signer: TokenSigner = api_app.state.ctx.signer

    import uuid

    expired = signer.issue(uuid.UUID(user_id), now=datetime.now(UTC) - timedelta(days=2))
    probe = clients.new()
    probe.cookies.set("cw_access", expired.token)
    assert (await probe.get("/api/v1/auth/me")).status_code == 401

    foreign = TokenSigner(
        secret="f" * 48, issuer=signer.issuer, audience=signer.audience, ttl=signer.ttl
    ).issue(uuid.UUID(user_id))
    probe.cookies.set("cw_access", foreign.token)
    assert (await probe.get("/api/v1/auth/me")).status_code == 401


async def test_token_for_deleted_user_rejected(clients: Any, api_app: Any) -> None:
    client = await clients.signed_up("ivy@example.com")
    async with api_app.state.ctx.engine.begin() as conn:
        await conn.exec_driver_sql("DELETE FROM users WHERE email = 'ivy@example.com'")
    assert (await client.get("/api/v1/auth/me")).status_code == 401
