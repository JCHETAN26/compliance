"""The MCP server over stdio, as Claude Desktop runs it: a real subprocess
(``python -m compliance_mcp_server --transport stdio``) with the token in its environment,
the test database/Redis/storage, a real model service on a loopback port, the mock LLM and
the mock rule-check backend. Every tool and both resources are called; review runs are
executed by an in-process stand-in for the worker.

Also checked: stdout carries only protocol messages (no log line corrupts it), the token
never appears on stderr, bad tokens stop the server at start, and revocation applies to the
very next call."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TextIO

import pytest
from mcp import Client, MCPError, StdioServerParameters
from mcp.types import TextResourceContents
from sqlalchemy import func, select

from compliance_core.db.models import LlmCall, ReviewRun, RulePack
from compliance_core.services import api_tokens
from compliance_mcp_server.tools import TOOL_NAMES

pytestmark = pytest.mark.integration
UPLOAD_LIMIT = 4096
TRAINED = "Confidential Information may include verbally conveyed information."
CUSTOM = "The Receiving Party must notify the Disclosing Party of any unauthorised disclosure."
CSV = b"Vendor,Data access,Contract file,Renewal date\nInitech,High,initech_nda.pdf,2027-01-31\n"


def child_env(env: Any, token: str | None) -> dict[str, str]:
    settings = env.settings
    base = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    base.pop("COMPLIANCE_MCP_TOKEN", None)
    base.update(
        APP_ENV="test",
        LOG_LEVEL="INFO",
        DATABASE_URL=settings.database_url,
        REDIS_URL=settings.redis_url,
        STORAGE_DIR=str(settings.storage_dir),
        DATA_DIR=str(settings.data_dir),
        ARTIFACT_CACHE_DIR=str(settings.artifact_cache_dir),
        ARTIFACT_AUTO_DOWNLOAD="false",
        MODEL_SERVICE_URL=settings.model_service_url,
        LLM_PROVIDER="mock",
        REVIEW_RULE_CHECK_BACKEND="mock",
        MAX_UPLOAD_FILE_BYTES=str(settings.max_upload_file_bytes),
        MAX_UPLOAD_REQUEST_BYTES=str(settings.max_upload_request_bytes),
        UPLOAD_RATE_LIMIT_PER_MINUTE="1000",
        MCP_UPLOAD_MAX_BYTES=str(UPLOAD_LIMIT),
        MCP_CHECK_RULE_WAIT_SECONDS="0",
    )
    if token is not None:
        base["COMPLIANCE_MCP_TOKEN"] = token
    return base


def params(env: Any, token: str | None) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "compliance_mcp_server", "--transport", "stdio"],
        env=child_env(env, token),
    )


@asynccontextmanager
async def stdio_client(env: Any, token: str, errlog: TextIO) -> AsyncIterator[Client]:
    from mcp.client.stdio import stdio_client as transport

    # mode="legacy": the initialize handshake Claude Desktop performs.
    async with Client(transport(params(env, token), errlog=errlog), mode="legacy") as client:
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


async def test_stdio_server_calls_every_tool_and_resource(
    mcp_env: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="mcp.client.stdio")
    token = await mcp_env.token(("read", "review", "upload"))
    ws = {"workspace_id": str(mcp_env.workspace_id)}
    contract_id = str(mcp_env.docs["globex_nda.docx"])
    called: set[str] = set()
    stderr_path = tmp_path / "server-stderr.log"
    with stderr_path.open("w", encoding="utf-8") as errlog:
        async with asyncio.timeout(420), stdio_client(mcp_env, token, errlog) as client:
            assert client.server_info is not None
            assert client.server_info.name == "compliance-workspace"
            assert "untrusted" in (client.instructions or "")
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert set(tools) == set(TOOL_NAMES)  # upload_document included over stdio
            assert tools["list_workspaces"].annotations.read_only_hint is True  # type: ignore[union-attr]
            assert tools["run_review"].annotations.read_only_hint is False  # type: ignore[union-attr]
            templates = (await client.list_resource_templates()).resource_templates
            assert {t.uri_template for t in templates} == {"rulepack://{id}", "report://{run_id}"}

            async def call(name: str, args: dict[str, Any]) -> Any:
                called.add(name)
                return await client.call_tool(name, args)

            workspaces = data(await call("list_workspaces", {}))["workspaces"]
            assert [w["id"] for w in workspaces] == [str(mcp_env.workspace_id)]
            assert workspaces[0]["role"] == "owner"

            documents = data(await call("list_documents", ws))["documents"]
            assert [(d["id"], d["status"], d["doc_type"]) for d in documents] == [
                (contract_id, "ready", "contract")
            ]

            encoded = base64.b64encode(CSV).decode()
            upload = ws | {"filename": "../register?.csv", "content_base64": encoded}
            created = data(await call("upload_document", upload))
            assert created["outcome"] == "created"
            assert created["filename"] == "register_.csv"
            assert data(await call("upload_document", upload))["outcome"] == "duplicate"
            assert "invalid_argument" in error_text(
                await call("upload_document", upload | {"content_base64": "bm90IGJhc2U2NA==\n"})
            )
            assert "invalid_argument" in error_text(
                await call("upload_document", upload | {"content_base64": "%%%%"})
            )
            too_big = base64.b64encode(b"x" * (UPLOAD_LIMIT + 1)).decode()
            assert "byte limit" in error_text(
                await call("upload_document", upload | {"content_base64": too_big})
            )
            listed = data(await call("list_documents", ws))["documents"]
            assert {d["id"] for d in listed} == {contract_id, created["document_id"]}

            search = data(
                await call(
                    "search_documents", ws | {"query": "confidential information", "top_k": 3}
                )
            )
            assert 1 <= len(search["results"]) <= 3
            first = search["results"][0]
            assert first["citation"]["document_id"] == contract_id
            assert first["citation"]["char_end"] > first["citation"]["char_start"]
            assert first["text"]
            assert "untrusted" in search["note"]

            packs = data(await call("list_rule_packs", ws))["rule_packs"]
            nda = next(p for p in packs if p["key"] == "nda_standard")
            yaml_text = await read_text(client, f"rulepack://{nda['id']}")
            assert "key: nda_standard" in yaml_text
            assert TRAINED in yaml_text

            # check_rule, returning at once (wait 0): a durable run id to poll.
            pending = data(
                await call("check_rule", ws | {"document_id": contract_id, "hypothesis": CUSTOM})
            )
            assert pending["complete"] is False
            assert pending["status"] == "queued"
            assert pending["model_supported"] is False
            assert pending["verdict"] is None
            assert pending["budget"]["usd_cap"] <= 0.05
            assert pending["budget"]["tokens_cap"] <= 20_000
            custom_run = uuid.UUID(pending["run_id"])
            assert await mcp_env.drain() == {custom_run: "completed"}
            status = data(await call("get_run_status", {"run_id": str(custom_run)}))
            assert status["status"] == "completed"
            assert status["origin"] == "mcp_check_rule"

            # check_rule with a worker running: the verdict comes back within the wait.
            stop = asyncio.Event()
            worker = asyncio.create_task(mcp_env.background_worker(stop))
            try:
                verdict = data(
                    await call(
                        "check_rule",
                        ws
                        | {"document_id": contract_id, "hypothesis": TRAINED, "wait_seconds": 60},
                    )
                )
            finally:
                stop.set()
                await worker
            assert verdict["complete"] is True
            assert verdict["status"] == "completed"
            assert verdict["model_supported"] is True  # verbatim training hypothesis
            finding = verdict["verdict"]
            assert finding["rule"] == TRAINED
            assert finding["status"] in {"satisfied", "violated", "not_mentioned", "needs_review"}
            if finding["status"] != "needs_review":
                assert finding["verified"]
                assert finding["citations"]

            started = data(
                await call(
                    "run_review", ws | {"rule_pack_id": nda["id"], "document_ids": [contract_id]}
                )
            )
            assert started["status"] == "queued"
            assert started["llm_is_mock"] is True
            run_id = uuid.UUID(started["run_id"])
            assert await mcp_env.drain() == {run_id: "completed"}
            status = data(await call("get_run_status", {"run_id": str(run_id)}))
            assert status["status"] == "completed"
            assert sum(status["findings_by_status"].values()) == 17

            markdown = data(await call("get_report", {"run_id": str(run_id)}))
            assert markdown["format"] == "markdown"
            assert markdown["report"].startswith("#")
            report_json = data(await call("get_report", {"run_id": str(run_id), "format": "json"}))
            findings = report_json["report"]["findings"]
            assert len(findings) == 17
            resource = await read_text(client, f"report://{run_id}")
            assert resource == markdown["report"]

            citation_id = next(c["id"] for f in findings for c in f["citations"])
            citation = data(await call("get_citation", {"citation_id": citation_id}))
            assert citation["document_id"] == contract_id
            assert citation["quote"]
            assert citation["current"] == "valid"
            assert "context_before" in citation

    assert called == set(TOOL_NAMES)
    # stdout carried only protocol messages: the client never failed to parse a line.
    assert not [r for r in caplog.records if r.name.startswith("mcp.client.stdio")]
    stderr = stderr_path.read_text("utf-8")
    assert token not in stderr
    # JSON logs went to stderr, including the start-up line written before the SDK
    # diverts fd 1 for the protocol (stdout logging would corrupt the stream there).
    assert '"message": "stdio MCP server starting"' in stderr
    assert '"message": "document uploaded"' in stderr

    async with mcp_env.sessions() as session:
        # The escalated custom check was ledgered like any run; ad-hoc rules create no pack.
        ledgered = await session.scalar(
            select(func.count()).select_from(LlmCall).where(LlmCall.run_id == custom_run)
        )
        assert ledgered
        assert ledgered > 0
        adhoc = await session.scalar(
            select(func.count()).select_from(RulePack).where(RulePack.key == "adhoc_check")
        )
        assert adhoc == 0
        origin = await session.scalar(
            select(ReviewRun.config_json["origin"].astext).where(ReviewRun.id == custom_run)
        )
        assert origin == "mcp_check_rule"


async def _run_server(env: Any, token: str | None) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        *(params(env, token).command, *params(env, token).args),
        env=child_env(env, token),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    async with asyncio.timeout(60):
        stdout, stderr = await process.communicate()
    assert stdout == b""  # nothing but protocol ever goes to stdout
    return int(process.returncode or 0), stderr.decode()


async def test_stdio_refuses_bad_tokens_and_rechecks_every_call(
    mcp_env: Any, tmp_path: Path
) -> None:
    code, stderr = await _run_server(mcp_env, None)
    assert code == 2
    assert "COMPLIANCE_MCP_TOKEN" in stderr
    token = await mcp_env.token()
    for bad in (token[:-1] + ("A" if token[-1] != "A" else "B"), "not-a-token"):
        code, stderr = await _run_server(mcp_env, bad)
        assert code == 2
        assert "invalid, expired or revoked" in stderr
        assert bad not in stderr

    with (tmp_path / "stderr.log").open("w", encoding="utf-8") as errlog:
        async with asyncio.timeout(120), stdio_client(mcp_env, token, errlog) as client:
            assert data(await client.call_tool("list_workspaces", {}))["workspaces"]
            async with mcp_env.sessions() as session:
                principal = await api_tokens.authenticate(session, token)
                await api_tokens.revoke_token(session, mcp_env.user_id, principal.token_id)
            refused = await client.call_tool("list_workspaces", {})
            assert "unauthenticated" in error_text(refused)
            assert token not in error_text(refused)
            with pytest.raises(MCPError, match="unauthenticated"):
                await client.read_resource(f"report://{uuid.uuid4()}")
