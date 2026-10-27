"""MCP server units (no database): registration per transport, annotations, auth before
any work, stderr logging for stdio, settings bounds, the healthcheck and the provider
capability declaration."""

from __future__ import annotations

import logging
import socket
import sys
from pathlib import Path

import pytest
from mcp import Client
from pydantic import ValidationError

from compliance_core.config import AppEnv, Settings
from compliance_core.logs import configure_logging
from compliance_core.services.errors import InvalidOperationError
from compliance_llm import ProviderResolver, configured_providers
from compliance_mcp_server import healthcheck
from compliance_mcp_server.config import McpSettings
from compliance_mcp_server.runtime import McpRuntime, Transport
from compliance_mcp_server.server import INSTRUCTIONS, build_server
from compliance_mcp_server.tools import TOOL_NAMES, tool_names


def _runtime(transport: Transport, token: str | None = None) -> McpRuntime:
    settings = Settings(app_env=AppEnv.TEST, storage_dir=Path("/nonexistent/never-created"))
    return McpRuntime(transport=transport, settings=settings, mcp=McpSettings(), stdio_token=token)


@pytest.mark.parametrize("transport", ["stdio", "http"])
async def test_tools_resources_and_annotations(transport: Transport) -> None:
    async with Client(build_server(_runtime(transport))) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        templates = (await client.list_resource_templates()).resource_templates
        assert client.instructions == INSTRUCTIONS
    assert list(tools) == tool_names(transport)
    assert ("upload_document" in tools) is (transport == "stdio")
    assert set(TOOL_NAMES) - set(tools) == (set() if transport == "stdio" else {"upload_document"})
    assert {t.uri_template for t in templates} == {"rulepack://{id}", "report://{run_id}"}
    for name in ("list_workspaces", "search_documents", "get_report", "get_citation"):
        assert tools[name].annotations.read_only_hint is True  # type: ignore[union-attr]
    for name in ("run_review", "check_rule"):
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.read_only_hint is False
        assert "'review' scope" in (tools[name].description or "")
    schema = tools["search_documents"].input_schema["properties"]
    assert schema["top_k"]["default"] == 8
    assert schema["top_k"]["maximum"] == 50


async def test_calls_without_a_token_fail_before_any_work() -> None:
    # Malformed tokens are refused before the database is touched (storage_dir above does
    # not exist; creating the service context would fail with an internal error instead).
    async with Client(build_server(_runtime("stdio", token="not-a-token"))) as client:
        result = await client.call_tool("list_workspaces", {})
    assert result.is_error
    assert "unauthenticated" in result.content[0].text  # type: ignore[union-attr]


def test_stdio_logging_goes_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        configure_logging("mcp-server", "INFO", stream=sys.stderr)
        logging.getLogger("compliance_mcp_server").info("hello", extra={"tool": "x"})
        captured = capsys.readouterr()
        assert captured.out == ""
        assert '"message": "hello"' in captured.err
        assert '"service": "mcp-server"' in captured.err
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


def test_mcp_settings_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    assert McpSettings().check_rule_budget_usd == 0.05
    assert McpSettings().upload_max_bytes == 10 * 1024 * 1024
    for name, value in (
        ("MCP_UPLOAD_MAX_BYTES", str(51 * 1024 * 1024)),
        ("MCP_CHECK_RULE_BUDGET_USD", "1.5"),
        ("MCP_CHECK_RULE_MAX_WAIT_SECONDS", "301"),
    ):
        monkeypatch.setenv(name, value)
        with pytest.raises(ValidationError):
            McpSettings()
        monkeypatch.delenv(name)


def test_provider_capability_declaration_never_needs_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    declared = configured_providers(" Anthropic , ")
    assert declared("anthropic")
    assert declared("mock")
    assert declared("ollama")
    assert not declared("openai")
    assert not configured_providers("")("anthropic")
    settings = Settings(app_env=AppEnv.TEST, llm_provider="anthropic")
    chosen = ProviderResolver(settings, declared).resolve("anthropic", "")
    assert chosen.provider == "anthropic"
    assert not chosen.is_mock
    with pytest.raises(InvalidOperationError, match="no API key configured"):
        ProviderResolver(settings, configured_providers("")).resolve("anthropic", "")
    # The REST default (environment check) is unchanged.
    with pytest.raises(InvalidOperationError, match="no API key configured"):
        ProviderResolver(settings).resolve("anthropic", "")


def test_healthcheck_fails_when_server_absent(capsys: pytest.CaptureFixture[str]) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert healthcheck.main([f"http://127.0.0.1:{port}"]) == 1
    assert "failed" in capsys.readouterr().err
