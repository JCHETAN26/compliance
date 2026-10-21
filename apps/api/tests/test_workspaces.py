"""Workspace CRUD and membership authorisation tests."""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.integration


async def _create(client: httpx.AsyncClient, name: str = "Vendor review") -> dict[str, Any]:
    response = await client.post("/api/v1/workspaces", json={"name": name})
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def _user_id(client: httpx.AsyncClient) -> str:
    return str((await client.get("/api/v1/auth/me")).json()["id"])


async def test_requires_authentication(clients: Any) -> None:
    client = clients.new()
    assert (await client.get("/api/v1/workspaces")).status_code == 401
    assert (await client.post("/api/v1/workspaces", json={"name": "x"})).status_code == 401


async def test_create_requires_csrf(clients: Any) -> None:
    client = await clients.signed_up("owner@example.com")
    del client.headers["X-CSRF-Token"]
    response = await client.post("/api/v1/workspaces", json={"name": "No CSRF"})
    assert response.status_code == 403
    client.headers["X-CSRF-Token"] = "forged"
    response = await client.post("/api/v1/workspaces", json={"name": "Forged CSRF"})
    assert response.status_code == 403
    client.headers["X-CSRF-Token"] = client.cookies["cw_csrf"]
    await _create(client, "With CSRF")


async def test_csrf_token_from_other_session_rejected(clients: Any) -> None:
    alice = await clients.signed_up("alice@example.com")
    mallory = await clients.signed_up("mallory@example.com")
    alice.headers["X-CSRF-Token"] = mallory.headers["X-CSRF-Token"]
    response = await alice.post("/api/v1/workspaces", json={"name": "x"})
    assert response.status_code == 403


async def test_create_list_get_rename_delete(clients: Any) -> None:
    client = await clients.signed_up("owner@example.com")
    assert (await client.get("/api/v1/workspaces")).json() == []

    created = await _create(client, "  Vendor review Q3  ")
    assert created["name"] == "Vendor review Q3"
    assert created["role"] == "owner"
    assert created["member_count"] == 1
    assert created["owner_id"] == await _user_id(client)

    second = await _create(client, "Policy audit")
    listed = (await client.get("/api/v1/workspaces")).json()
    assert [w["id"] for w in listed] == [second["id"], created["id"]]

    fetched = await client.get(f"/api/v1/workspaces/{created['id']}")
    assert fetched.status_code == 200
    assert fetched.json() == created

    renamed = await client.patch(f"/api/v1/workspaces/{created['id']}", json={"name": "Renamed"})
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Renamed"

    deleted = await client.delete(f"/api/v1/workspaces/{created['id']}")
    assert deleted.status_code == 204
    assert (await client.get(f"/api/v1/workspaces/{created['id']}")).status_code == 404
    assert [w["id"] for w in (await client.get("/api/v1/workspaces")).json()] == [second["id"]]


@pytest.mark.parametrize("name", ["", "   ", "x" * 121])
async def test_workspace_name_validation(clients: Any, name: str) -> None:
    client = await clients.signed_up("owner@example.com")
    response = await client.post("/api/v1/workspaces", json={"name": name})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_non_members_cannot_see_or_touch_workspace(clients: Any) -> None:
    owner = await clients.signed_up("owner@example.com")
    outsider = await clients.signed_up("outsider@example.com")
    workspace = await _create(owner)
    ws = f"/api/v1/workspaces/{workspace['id']}"

    assert (await outsider.get("/api/v1/workspaces")).json() == []
    for method, url, body in [
        ("GET", ws, None),
        ("PATCH", ws, {"name": "pwned"}),
        ("DELETE", ws, None),
        ("GET", f"{ws}/members", None),
        ("POST", f"{ws}/members", {"email": "outsider@example.com", "role": "reviewer"}),
    ]:
        response = await outsider.request(method, url, json=body)
        assert response.status_code == 404, (method, url, response.text)
        assert response.json()["error"]["message"] == "Workspace not found."

    # Unknown ids look exactly like forbidden ones.
    missing = await outsider.get(f"/api/v1/workspaces/{uuid.uuid4()}")
    assert missing.status_code == 404
    assert (await owner.get(ws)).json()["name"] == workspace["name"]


async def test_malformed_workspace_id(clients: Any) -> None:
    client = await clients.signed_up("owner@example.com")
    response = await client.get("/api/v1/workspaces/not-a-uuid")
    assert response.status_code == 422


async def test_membership_roles_and_permissions(clients: Any) -> None:
    owner = await clients.signed_up("owner@example.com")
    reviewer = await clients.signed_up("reviewer@example.com")
    viewer = await clients.signed_up("viewer@example.com")
    workspace = await _create(owner)
    ws = f"/api/v1/workspaces/{workspace['id']}"

    added = await owner.post(
        f"{ws}/members", json={"email": "Reviewer@Example.com", "role": "reviewer"}
    )
    assert added.status_code == 201, added.text
    assert added.json()["role"] == "reviewer"
    assert added.json()["email"] == "reviewer@example.com"
    added = await owner.post(f"{ws}/members", json={"email": "viewer@example.com"})
    assert added.json()["role"] == "viewer"

    members = (await viewer.get(f"{ws}/members")).json()
    assert [(m["email"], m["role"]) for m in members] == [
        ("owner@example.com", "owner"),
        ("reviewer@example.com", "reviewer"),
        ("viewer@example.com", "viewer"),
    ]
    reviewer_view = (await reviewer.get(ws)).json()
    assert reviewer_view["role"] == "reviewer"
    assert reviewer_view["member_count"] == 3
    assert [w["role"] for w in (await viewer.get("/api/v1/workspaces")).json()] == ["viewer"]

    # Only the owner may rename, delete, or manage members.
    for client in (reviewer, viewer):
        assert (await client.patch(ws, json={"name": "nope"})).status_code == 403
        assert (await client.delete(ws)).status_code == 403
        response = await client.post(f"{ws}/members", json={"email": "owner@example.com"})
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "forbidden"
    viewer_id = await _user_id(viewer)
    reviewer_id = await _user_id(reviewer)
    assert (await reviewer.delete(f"{ws}/members/{viewer_id}")).status_code == 403
    assert (
        await reviewer.patch(f"{ws}/members/{viewer_id}", json={"role": "reviewer"})
    ).status_code == 403

    # Owner can change roles but not to/from owner.
    promoted = await owner.patch(f"{ws}/members/{viewer_id}", json={"role": "reviewer"})
    assert promoted.status_code == 200
    assert promoted.json()["role"] == "reviewer"
    assert (
        await owner.patch(f"{ws}/members/{viewer_id}", json={"role": "owner"})
    ).status_code == 422
    owner_id = await _user_id(owner)
    assert (
        await owner.patch(f"{ws}/members/{owner_id}", json={"role": "viewer"})
    ).status_code == 409

    # Owner removes a member; that member loses access.
    assert (await owner.delete(f"{ws}/members/{reviewer_id}")).status_code == 204
    assert (await reviewer.get(ws)).status_code == 404

    # A member can leave on their own.
    assert (await viewer.delete(f"{ws}/members/{viewer_id}")).status_code == 204
    assert (await viewer.get(ws)).status_code == 404
    assert (await owner.get(ws)).json()["member_count"] == 1


async def test_member_edge_cases(clients: Any) -> None:
    owner = await clients.signed_up("owner@example.com")
    await clients.signed_up("member@example.com")
    workspace = await _create(owner)
    ws = f"/api/v1/workspaces/{workspace['id']}"

    unknown = await owner.post(f"{ws}/members", json={"email": "nobody@example.com"})
    assert unknown.status_code == 404
    assert "No registered user" in unknown.json()["error"]["message"]

    assert (
        await owner.post(f"{ws}/members", json={"email": "member@example.com"})
    ).status_code == 201
    duplicate = await owner.post(f"{ws}/members", json={"email": "member@example.com"})
    assert duplicate.status_code == 409

    as_owner = await owner.post(f"{ws}/members", json={"email": "x@example.com", "role": "owner"})
    assert as_owner.status_code == 422

    owner_id = await _user_id(owner)
    leave = await owner.delete(f"{ws}/members/{owner_id}")
    assert leave.status_code == 409
    assert leave.json()["error"]["code"] == "invalid_operation"

    missing = await owner.delete(f"{ws}/members/{uuid.uuid4()}")
    assert missing.status_code == 404


async def test_deleting_workspace_removes_memberships(clients: Any, api_app: Any) -> None:
    owner = await clients.signed_up("owner@example.com")
    member = await clients.signed_up("member@example.com")
    workspace = await _create(owner)
    ws = f"/api/v1/workspaces/{workspace['id']}"
    await owner.post(f"{ws}/members", json={"email": "member@example.com"})
    assert (await owner.delete(ws)).status_code == 204
    assert (await member.get("/api/v1/workspaces")).json() == []
    async with api_app.state.ctx.engine.connect() as conn:
        remaining = await conn.exec_driver_sql(
            "SELECT count(*) FROM workspace_members WHERE workspace_id = $1",
            (uuid.UUID(workspace["id"]),),
        )
        assert remaining.scalar_one() == 0
