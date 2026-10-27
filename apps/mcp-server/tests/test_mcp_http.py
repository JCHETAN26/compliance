"""Streamable HTTP transport security: bearer-only auth on every request, per-call
revocation/expiry/membership checks inside one client session, scope/role/allowlist
intersection for every id kind (workspace, run, citation, rule pack, resources), no OAuth
advertising, no uploads over HTTP, DNS-rebinding protection and no token in logs."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import httpx2
import pytest
from mcp import Client, MCPError
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextResourceContents
from sqlalchemy import delete, update

from compliance_core.db.models import ApiToken, WorkspaceMember
from compliance_core.services import api_tokens
from compliance_mcp_server import healthcheck
from compliance_mcp_server.server import MCP_PATH
from compliance_mcp_server.tools import tool_names

pytestmark = pytest.mark.integration
HTTP_TOOLS = set(tool_names("http"))
INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "t", "version": "1"},
    },
}
ACCEPT = {"Accept": "application/json, text/event-stream"}


@asynccontextmanager
async def mcp_client(base: str, token: str, **headers: str) -> AsyncIterator[Client]:
    async with (
        httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {token}", **headers}, timeout=60
        ) as http,
        Client(streamable_http_client(base + MCP_PATH, http_client=http)) as client,
    ):
        yield client


async def read_text(client: Client, uri: str) -> str:
    [content] = (await client.read_resource(uri)).contents
    assert isinstance(content, TextResourceContents)
    return content.text


def data(result: Any) -> dict[str, Any]:
    assert not result.is_error, result.content
    if isinstance(result.structured_content, dict):
        return result.structured_content
    payload: dict[str, Any] = json.loads(result.content[0].text)
    return payload


def error_text(result: Any) -> str:
    assert result.is_error, result.structured_content
    return str(result.content[0].text)


async def test_health_and_container_healthcheck(mcp_http: str) -> None:
    async with httpx.AsyncClient(timeout=10) as http:
        health = (await http.get(f"{mcp_http}/health")).json()
    assert health["status"] == "ok"
    assert health["checks"]["database"]["status"] == "ok"
    assert set(health["implemented_tools"]) == HTTP_TOOLS
    assert "upload_document" not in health["implemented_tools"]
    assert health["resource_templates"] == ["rulepack://{id}", "report://{run_id}"]
    assert healthcheck.main([mcp_http]) == 0


async def test_requests_without_a_valid_bearer_token_are_challenged(
    mcp_http: str, mcp_env: Any
) -> None:
    url = mcp_http + MCP_PATH
    token = await mcp_env.token()
    async with httpx.AsyncClient(timeout=10) as http:
        missing = await http.post(url, json=INIT, headers=ACCEPT)
        assert missing.status_code == 401
        assert missing.headers["www-authenticate"] == 'Bearer realm="compliance-mcp"'
        assert "resource_metadata" not in missing.headers["www-authenticate"]
        assert (await http.get(url, headers=ACCEPT)).status_code == 401
        for header in (f"Basic {token}", token, "Bearer", f"Bearer {token}x", "Bearer cwmcp_x"):
            response = await http.post(url, json=INIT, headers=ACCEPT | {"Authorization": header})
            assert response.status_code == 401, header
        # The last character carries only 4 bits (16 values): never "change" it to itself.
        wrong = token[:-1] + ("A" if token[-1] != "A" else "B")
        invalid = await http.post(
            url, json=INIT, headers=ACCEPT | {"Authorization": f"Bearer {wrong}"}
        )
        assert invalid.status_code == 401
        assert invalid.headers["www-authenticate"].endswith('error="invalid_token"')
        assert token not in invalid.text
        # Tokens in URLs are refused outright (they leak into logs and histories).
        for query in (f"access_token={token}", f"token={token}", f"x={token}"):
            response = await http.post(
                f"{url}?{query}", json=INIT, headers=ACCEPT | {"Authorization": f"Bearer {token}"}
            )
            assert response.status_code == 400, query
        # No OAuth discovery is advertised: these are our own tokens, not OAuth.
        for path in (
            "/.well-known/oauth-protected-resource",
            "/.well-known/oauth-protected-resource/mcp",
            "/.well-known/oauth-authorization-server",
            "/authorize",
            "/register",
        ):
            assert (await http.get(mcp_http + path)).status_code == 404, path


async def test_http_offers_no_upload_and_authenticated_reads_work(
    mcp_http: str, mcp_env: Any
) -> None:
    token = await mcp_env.token(("read", "review", "upload"))
    async with mcp_client(mcp_http, token) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert names == HTTP_TOOLS
        listed = data(await client.call_tool("list_workspaces", {}))
        assert [w["id"] for w in listed["workspaces"]] == [str(mcp_env.workspace_id)]
        assert listed["token_scopes"] == ["read", "review", "upload"]
        upload = await client.call_tool(
            "upload_document",
            {"workspace_id": str(mcp_env.workspace_id), "filename": "a.csv", "content_base64": ""},
        )
        assert upload.is_error


async def test_revocation_expiry_and_membership_loss_apply_on_the_next_call(
    mcp_http: str, mcp_env: Any, clients: Any
) -> None:
    sessions = mcp_env.sessions
    ws = {"workspace_id": str(mcp_env.workspace_id)}

    token = await mcp_env.token()
    async with mcp_client(mcp_http, token) as client:
        assert data(await client.call_tool("list_documents", ws))["documents"]
        async with sessions() as session:
            principal = await api_tokens.authenticate(session, token)
            await api_tokens.revoke_token(session, mcp_env.user_id, principal.token_id)
        with pytest.raises(MCPError):
            await client.call_tool("list_documents", ws)

    token = await mcp_env.token()
    async with mcp_client(mcp_http, token) as client:
        assert data(await client.call_tool("list_documents", ws))["documents"]
        async with sessions() as session:
            await session.execute(
                update(ApiToken)
                .where(ApiToken.token_hash == api_tokens.hash_token(token))
                .values(
                    created_at=datetime.now(UTC) - timedelta(days=1),
                    expires_at=datetime.now(UTC) - timedelta(seconds=1),
                )
            )
            await session.commit()
        with pytest.raises(MCPError):
            await client.call_tool("list_documents", ws)

    # A reviewer whose membership is removed mid-session loses access at once.
    reviewer = await clients.signed_up("mcp-reviewer@example.com")
    reviewer_id = uuid.UUID((await reviewer.get("/api/v1/auth/me")).json()["id"])
    added = await mcp_env.client.post(
        f"/api/v1/workspaces/{mcp_env.workspace_id}/members",
        json={"email": "mcp-reviewer@example.com", "role": "reviewer"},
    )
    assert added.status_code == 201, added.text
    token = await mcp_env.token(user_id=reviewer_id)
    async with mcp_client(mcp_http, token) as client:
        assert data(await client.call_tool("list_documents", ws))["documents"]
        async with sessions() as session:
            await session.execute(
                delete(WorkspaceMember).where(
                    WorkspaceMember.workspace_id == mcp_env.workspace_id,
                    WorkspaceMember.user_id == reviewer_id,
                )
            )
            await session.commit()
        assert error_text(await client.call_tool("list_documents", ws)).endswith(
            "not_found: Workspace not found."
        )
        assert data(await client.call_tool("list_workspaces", {}))["workspaces"] == []


async def _completed_review(client: Client, env: Any) -> tuple[str, str, str]:
    """Run nda_standard over the contract; return (run_id, citation_id, pack_id)."""
    packs = data(await client.call_tool("list_rule_packs", {"workspace_id": str(env.workspace_id)}))
    pack = next(p for p in packs["rule_packs"] if p["key"] == "nda_standard")
    started = data(
        await client.call_tool(
            "run_review", {"workspace_id": str(env.workspace_id), "rule_pack_id": pack["id"]}
        )
    )
    assert started["status"] == "queued"
    assert started["llm_is_mock"]
    outcomes = await env.drain()
    assert outcomes == {uuid.UUID(started["run_id"]): "completed"}
    report = data(
        await client.call_tool("get_report", {"run_id": started["run_id"], "format": "json"})
    )
    citation_ids = [c["id"] for f in report["report"]["findings"] for c in f["citations"]]
    assert citation_ids
    return started["run_id"], citation_ids[0], pack["id"]


async def test_scopes_roles_allowlists_and_users_are_intersected(
    mcp_http: str, mcp_env: Any, clients: Any
) -> None:
    owner_review = await mcp_env.token(("read", "review"))
    async with mcp_client(mcp_http, owner_review) as client:
        run_id, citation_id, pack_id = await _completed_review(client, mcp_env)
        assert data(await client.call_tool("get_citation", {"citation_id": citation_id}))
        # A pack in the workspace, for the allowlist checks below.
        imported = await mcp_env.client.post(
            "/api/v1/rule-packs/import",
            json={
                "workspace_id": str(mcp_env.workspace_id),
                "yaml": (
                    "key: house\nname: House\nversion: 1.0.0\nrules:\n  - key: r\n"
                    "    title: R\n    hypothesis_text: Sharing is restricted.\n"
                    "    scope: per_document\n    applies_to_doc_types: [contract]\n"
                    "    severity: low\n"
                ),
            },
        )
        assert imported.status_code == 201, imported.text
        house_id = imported.json()["id"]
        assert "key: house" in await read_text(client, f"rulepack://{house_id}")

    other_ws = await mcp_env.client.post("/api/v1/workspaces", json={"name": "Elsewhere"})
    other_ws_id = other_ws.json()["id"]

    # Read-only token: cannot spend budget.
    read_only = await mcp_env.token()
    async with mcp_client(mcp_http, read_only) as client:
        refused = await client.call_tool(
            "run_review", {"workspace_id": str(mcp_env.workspace_id), "rule_pack_id": pack_id}
        )
        assert "forbidden: This API token lacks the 'review' scope" in error_text(refused)
        assert data(await client.call_tool("get_run_status", {"run_id": run_id}))["status"] == (
            "completed"
        )

    # Allowlisted to the other workspace: the first workspace's ids look missing.
    restricted = await mcp_env.token(("read", "review"), [uuid.UUID(other_ws_id)])
    async with mcp_client(mcp_http, restricted) as client:
        listed = data(await client.call_tool("list_workspaces", {}))
        assert [w["id"] for w in listed["workspaces"]] == [other_ws_id]
        assert listed["token_workspace_restricted"] is True
        ws = {"workspace_id": str(mcp_env.workspace_id)}
        for name, args in (
            ("list_documents", ws),
            ("list_rule_packs", ws),
            ("search_documents", ws | {"query": "confidential"}),
            ("run_review", ws | {"rule_pack_id": pack_id}),
            (
                "check_rule",
                ws | {"document_id": str(next(iter(mcp_env.docs.values()))), "hypothesis": "x"},
            ),
            ("get_run_status", {"run_id": run_id}),
            ("get_report", {"run_id": run_id}),
            ("get_citation", {"citation_id": citation_id}),
        ):
            assert "not_found" in error_text(await client.call_tool(name, args)), name
        for uri in (f"report://{run_id}", f"rulepack://{house_id}"):
            with pytest.raises(MCPError):
                await client.read_resource(uri)
        # Built-in packs are global and stay readable.
        assert "key: nda_standard" in await read_text(client, f"rulepack://{pack_id}")

    # A viewer's 'review' scope does not raise the role.
    viewer = await clients.signed_up("mcp-viewer@example.com")
    viewer_id = uuid.UUID((await viewer.get("/api/v1/auth/me")).json()["id"])
    await mcp_env.client.post(
        f"/api/v1/workspaces/{mcp_env.workspace_id}/members",
        json={"email": "mcp-viewer@example.com", "role": "viewer"},
    )
    viewer_token = await mcp_env.token(("read", "review"), user_id=viewer_id)
    async with mcp_client(mcp_http, viewer_token) as client:
        ws = {"workspace_id": str(mcp_env.workspace_id)}
        refused = await client.call_tool("run_review", ws | {"rule_pack_id": pack_id})
        assert "forbidden: Your role (viewer)" in error_text(refused)
        refused = await client.call_tool(
            "check_rule",
            ws | {"document_id": str(next(iter(mcp_env.docs.values()))), "hypothesis": "x"},
        )
        assert "forbidden: Your role (viewer)" in error_text(refused)
        assert data(await client.call_tool("get_citation", {"citation_id": citation_id}))

    # Another user's token sees none of it.
    stranger = await clients.signed_up("mcp-stranger@example.com")
    stranger_id = uuid.UUID((await stranger.get("/api/v1/auth/me")).json()["id"])
    stranger_token = await mcp_env.token(("read", "review"), user_id=stranger_id)
    async with mcp_client(mcp_http, stranger_token) as client:
        assert data(await client.call_tool("list_workspaces", {}))["workspaces"] == []
        for name, args in (
            ("get_run_status", {"run_id": run_id}),
            ("get_citation", {"citation_id": citation_id}),
            ("get_report", {"run_id": run_id, "format": "markdown"}),
        ):
            assert "not_found" in error_text(await client.call_tool(name, args)), name
        with pytest.raises(MCPError):
            await client.read_resource(f"rulepack://{house_id}")


async def test_untrusted_host_is_rejected_even_with_a_valid_token(
    mcp_http: str, mcp_env: Any
) -> None:
    token = await mcp_env.token()
    async with httpx.AsyncClient(timeout=10) as http:
        response = await http.post(
            mcp_http + MCP_PATH,
            json=INIT,
            headers=ACCEPT | {"Host": "evil.example", "Authorization": f"Bearer {token}"},
        )
    assert response.status_code in {400, 403, 421}


async def test_tokens_never_reach_the_logs(
    mcp_http: str, mcp_env: Any, caplog: pytest.LogCaptureFixture
) -> None:
    token = await mcp_env.token()
    caplog.set_level(logging.DEBUG)
    async with mcp_client(mcp_http, token) as client:
        data(await client.call_tool("list_workspaces", {}))
        await client.call_tool("get_run_status", {"run_id": str(uuid.uuid4())})
    async with httpx.AsyncClient(timeout=10) as http:
        refused = await http.post(
            mcp_http + MCP_PATH, json=INIT, headers=ACCEPT | {"Authorization": f"Bearer {token}x"}
        )
    assert refused.status_code == 401
    assert caplog.records
    for record in caplog.records:
        assert token not in record.getMessage()
        assert token not in repr(record.__dict__)
