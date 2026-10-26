"""Phase 5 end-to-end smoke test: the MCP server against a running stack.

Through the web origin it signs up, creates a workspace, uploads one synthetic NDA and
creates API tokens on the Settings endpoints. Then, as an MCP client:

- streamable HTTP (``--mcp``): unauthenticated requests get 401 + ``WWW-Authenticate:
  Bearer``; the nine HTTP tools (no ``upload_document``) list; workspaces, documents,
  search with citations, rule packs and the ``rulepack://`` resource work; a read-only
  token cannot start a review; ``run_review`` → ``get_run_status`` → ``get_report``
  (Markdown and JSON) → ``report://`` → ``get_citation``; ``check_rule`` returns a verdict
  or a run id; a revoked token fails on the very next call of the same session;
- stdio (``--stdio-command``, optional): the command Claude Desktop would run (for example
  ``docker compose exec -T -e COMPLIANCE_MCP_TOKEN mcp-server python -m
  compliance_mcp_server --transport stdio``) with the token in its environment: all ten
  tools list, ``upload_document`` uploads, and a report is read.

    uv run python scripts/smoke_phase5.py --web http://127.0.0.1:3000 \\
        --mcp http://127.0.0.1:8020 [--stdio-command "..."] [--timeout 1500]

With the default mock LLM this makes no external requests. Against a stack with a real
provider configured, ``run_review`` and ``check_rule`` spend real money: do not run it
there without approval.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import secrets
import shlex
import sys
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import httpx2
from mcp import Client, MCPError, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextResourceContents

from compliance_seed.synthetic import build

PASSWORD = "smoke-phase5-password"  # noqa: S105 - throwaway test account
CONTRACT = "globex_nda.docx"
TERMINAL = {"completed", "failed", "cancelled", "paused", "awaiting_review"}
HTTP_TOOLS = {
    "list_workspaces",
    "list_documents",
    "search_documents",
    "check_rule",
    "run_review",
    "get_run_status",
    "get_report",
    "get_citation",
    "list_rule_packs",
}


class SmokeError(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def note(message: str) -> None:
    print(message)  # noqa: T201


async def step(name: str, fn: Callable[[], Awaitable[None]]) -> None:
    await fn()
    print(f"PASS  {name}")  # noqa: T201


def data(result: Any) -> dict[str, Any]:
    expect(not result.is_error, f"tool error: {result.content}")
    if isinstance(result.structured_content, dict):
        return result.structured_content
    payload: dict[str, Any] = json.loads(result.content[0].text)
    return payload


async def read_text(client: Client, uri: str) -> str:
    [content] = (await client.read_resource(uri)).contents
    expect(isinstance(content, TextResourceContents), f"{uri}: not text")
    assert isinstance(content, TextResourceContents)  # noqa: S101 - narrowing
    return content.text


async def run(args: argparse.Namespace) -> None:
    web = httpx.AsyncClient(base_url=args.web, timeout=120)
    state: dict[str, Any] = {}
    mcp_url = args.mcp.rstrip("/") + "/mcp"

    async def mcp_session(token: str) -> tuple[httpx2.AsyncClient, Client]:
        http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=120)
        return http, Client(streamable_http_client(mcp_url, http_client=http))

    async def setup() -> None:
        email = f"smoke5-{secrets.token_hex(4)}@example.com"
        r = await web.post("/api/v1/auth/register", json={"email": email, "password": PASSWORD})
        expect(r.status_code == 201, f"register: {r.status_code} {r.text}")
        web.headers["X-CSRF-Token"] = r.json()["csrf_token"]
        ws = await web.post("/api/v1/workspaces", json={"name": "Phase 5 smoke"})
        expect(ws.status_code == 201, ws.text)
        state["ws"] = ws.json()["id"]
        r = await web.post(
            f"/api/v1/workspaces/{state['ws']}/documents",
            files=[("files", (CONTRACT, build()[CONTRACT], "application/octet-stream"))],
        )
        expect(r.status_code == 200, f"upload: {r.status_code} {r.text}")
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            [doc] = (await web.get(f"/api/v1/workspaces/{state['ws']}/documents")).json()
            if doc["status"] in ("ready", "failed"):
                expect(doc["status"] == "ready", f"ingestion failed: {doc}")
                state["doc"] = doc["id"]
                return
            await asyncio.sleep(3)
        raise SmokeError("the document did not finish processing")

    async def tokens() -> None:
        for name, scopes in (("read", ["read"]), ("review", ["read", "review"])):
            r = await web.post(
                "/api/v1/settings/api-tokens", json={"name": f"smoke {name}", "scopes": scopes}
            )
            expect(r.status_code == 201, f"token: {r.status_code} {r.text}")
            expect(r.headers.get("cache-control") == "no-store", "token response is cacheable")
            state[f"{name}_token"] = r.json()["token"]
            state[f"{name}_token_id"] = r.json()["id"]
        listed = await web.get("/api/v1/settings/api-tokens")
        expect(state["review_token"] not in listed.text, "token secret listed after creation")

    async def unauthenticated() -> None:
        async with httpx.AsyncClient(timeout=20) as http:
            r = await http.post(
                mcp_url,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={"Accept": "application/json, text/event-stream"},
            )
            health = (await http.get(args.mcp.rstrip("/") + "/health")).json()
        expect(r.status_code == 401, f"unauthenticated MCP request: {r.status_code}")
        expect(r.headers.get("www-authenticate", "").startswith("Bearer"), "no Bearer challenge")
        expect(health["status"] == "ok", f"mcp /health: {health}")
        expect(set(health["implemented_tools"]) == HTTP_TOOLS, "HTTP tool list in /health")

    async def http_reads() -> None:
        http, client = await mcp_session(state["read_token"])
        async with http, client:
            names = {t.name for t in (await client.list_tools()).tools}
            expect(names == HTTP_TOOLS, f"HTTP tools {sorted(names)}")
            workspaces = data(await client.call_tool("list_workspaces", {}))["workspaces"]
            expect(state["ws"] in {w["id"] for w in workspaces}, "workspace not listed")
            ws = {"workspace_id": state["ws"]}
            docs = data(await client.call_tool("list_documents", ws))["documents"]
            expect([d["id"] for d in docs] == [state["doc"]], f"documents {docs}")
            found = data(
                await client.call_tool(
                    "search_documents", ws | {"query": "confidential information", "top_k": 3}
                )
            )["results"]
            expect(bool(found), "search returned nothing")
            cite = found[0]["citation"]
            expect(
                cite["document_id"] == state["doc"] and cite["char_end"] > cite["char_start"],
                f"citation {cite}",
            )
            packs = data(await client.call_tool("list_rule_packs", ws))["rule_packs"]
            nda = next(p for p in packs if p["key"] == "nda_standard")
            state["pack"] = nda["id"]
            expect(
                "key: nda_standard" in await read_text(client, f"rulepack://{nda['id']}"),
                "rulepack resource",
            )
            refused = await client.call_tool("run_review", ws | {"rule_pack_id": nda["id"]})
            message = str(getattr(refused.content[0], "text", ""))
            expect(refused.is_error and "'review' scope" in message, "read-only token reviewed")

    async def http_review() -> None:
        http, client = await mcp_session(state["review_token"])
        async with http, client:
            ws = {"workspace_id": state["ws"]}
            started = data(
                await client.call_tool("run_review", ws | {"rule_pack_id": state["pack"]})
            )
            run_id = started["run_id"]
            note(
                f"      run {run_id} provider={started['llm_provider']} mock={started['llm_is_mock']}"
            )
            deadline = time.monotonic() + args.timeout
            status = started
            while time.monotonic() < deadline:
                status = data(await client.call_tool("get_run_status", {"run_id": run_id}))
                if status["status"] in TERMINAL:
                    break
                await asyncio.sleep(3)
            expect(
                status["status"] == "completed",
                f"run ended {status['status']}: {status.get('error_message')}",
            )
            md = data(await client.call_tool("get_report", {"run_id": run_id}))["report"]
            expect(md.startswith("#"), "markdown report")
            expect(
                await read_text(client, f"report://{run_id}") == md,
                "report:// differs from get_report",
            )
            report = data(
                await client.call_tool("get_report", {"run_id": run_id, "format": "json"})
            )["report"]
            citations = [c["id"] for f in report["findings"] for c in f["citations"]]
            expect(bool(citations), "no citations in the report")
            detail = data(await client.call_tool("get_citation", {"citation_id": citations[0]}))
            expect(
                detail["document_id"] == state["doc"] and bool(detail["quote"]), "citation detail"
            )
            state["run"] = run_id
            check = data(
                await client.call_tool(
                    "check_rule",
                    ws
                    | {
                        "document_id": state["doc"],
                        "hypothesis": "Confidential Information may include verbally conveyed information.",
                        "wait_seconds": min(args.timeout, 120),
                    },
                )
            )
            expect(check["model_supported"] is True, "trained hypothesis not model-supported")
            expect(bool(check["run_id"]), "check_rule returned no run id")
            note(
                f"      check_rule {check['run_id']} status={check['status']} verdict={(check['verdict'] or {}).get('status')}"
            )

    async def revocation() -> None:
        http, client = await mcp_session(state["review_token"])
        async with http, client:
            data(await client.call_tool("list_workspaces", {}))
            r = await web.delete(f"/api/v1/settings/api-tokens/{state['review_token_id']}")
            expect(r.status_code == 200 and r.json()["status"] == "revoked", f"revoke: {r.text}")
            try:
                result = await client.call_tool("list_workspaces", {})
            except MCPError:
                return
            expect(result.is_error, "a revoked token still worked in the same session")

    async def stdio() -> None:
        r = await web.post(
            "/api/v1/settings/api-tokens",
            json={"name": "smoke stdio", "scopes": ["upload"], "workspace_ids": [state["ws"]]},
        )
        expect(r.status_code == 201, r.text)
        argv = shlex.split(args.stdio_command)
        params = StdioServerParameters(
            command=argv[0],
            args=argv[1:],
            env=os.environ | {"COMPLIANCE_MCP_TOKEN": r.json()["token"]},
        )
        async with Client(stdio_client(params), mode="legacy") as client:
            names = {t.name for t in (await client.list_tools()).tools}
            expect(names == HTTP_TOOLS | {"upload_document"}, f"stdio tools {sorted(names)}")
            csv = b"Vendor,Data access\nInitech,High\n" + secrets.token_hex(8).encode() + b",Low\n"
            uploaded = data(
                await client.call_tool(
                    "upload_document",
                    {
                        "workspace_id": state["ws"],
                        "filename": "stdio-register.csv",
                        "content_base64": base64.b64encode(csv).decode(),
                    },
                )
            )
            expect(uploaded["outcome"] == "created", f"stdio upload {uploaded}")
            md = data(await client.call_tool("get_report", {"run_id": state["run"]}))["report"]
            expect(md.startswith("#"), "stdio report")

    async with web:
        await step("sign up, workspace, upload and ingest one NDA (web origin)", setup)
        await step("API tokens created on Settings endpoints (shown once, no-store)", tokens)
        await step(
            "MCP HTTP refuses unauthenticated requests (401 Bearer); /health", unauthenticated
        )
        await step("HTTP read tools, search citations, rule packs, scope enforcement", http_reads)
        await step(
            "HTTP run_review -> status -> report/resource -> citation; check_rule", http_review
        )
        await step("revoked token fails on the next call of an open session", revocation)
        if args.stdio_command:
            await step("stdio: all ten tools, upload_document, report", stdio)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", default="http://127.0.0.1:3000")
    parser.add_argument("--mcp", default="http://127.0.0.1:8020")
    parser.add_argument("--stdio-command", default="", help="command that starts the stdio server")
    parser.add_argument("--timeout", type=int, default=1500, help="seconds to wait for the run")
    args = parser.parse_args()
    try:
        asyncio.run(asyncio.wait_for(run(args), timeout=args.timeout + 1200))
    except SmokeError as exc:
        print(f"FAIL  {exc}", file=sys.stderr)  # noqa: T201
        return 1
    print("Phase 5 smoke: all checks passed")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
