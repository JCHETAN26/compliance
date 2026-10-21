"""MCP skeleton tests over in-process, stdio and streamable-HTTP transports."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn
from mcp import Client, StdioServerParameters

from compliance_core.config import Settings, get_settings
from compliance_mcp_server import healthcheck
from compliance_mcp_server.server import INSTRUCTIONS, MCP_PATH, build_http_app, build_server


async def test_in_process_server_exposes_no_tools_yet() -> None:
    async with Client(build_server()) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
    assert tools.tools == []
    assert resources.resources == []


async def test_stdio_transport_handshake() -> None:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "compliance_mcp_server", "--transport", "stdio"],
        env=env,
    )
    async with Client(params, mode="legacy") as client:
        assert client.server_info is not None
        assert client.server_info.name == "compliance-workspace"
        assert client.instructions == INSTRUCTIONS
        assert (await client.list_tools()).tools == []


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def http_server(
    monkeypatch: pytest.MonkeyPatch, test_settings: Settings, migrated_database: str
) -> Iterator[str]:
    monkeypatch.setenv("DATABASE_URL", test_settings.database_url)
    get_settings.cache_clear()
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(build_http_app(), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("MCP HTTP server did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)
    get_settings.cache_clear()


@pytest.mark.integration
def test_http_healthcheck_performs_real_handshake(http_server: str) -> None:
    assert healthcheck.main([http_server]) == 0
    health = httpx.get(f"{http_server}/health").json()
    assert health["status"] == "ok"
    assert health["checks"]["database"]["status"] == "ok"
    assert health["implemented_tools"] == []
    assert health["phase"] == "skeleton"


@pytest.mark.integration
def test_http_rejects_untrusted_host_header(http_server: str) -> None:
    response = httpx.post(
        f"{http_server}{MCP_PATH}",
        headers={"Host": "evil.example", "Content-Type": "application/json"},
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
    )
    assert response.status_code in {400, 403, 421}


def test_healthcheck_fails_when_server_absent() -> None:
    assert healthcheck.main([f"http://127.0.0.1:{_free_port()}"]) == 1
