"""MCP server built on the official ``mcp`` Python SDK (``MCPServer``, formerly FastMCP).

PHASE 0 SKELETON. The server speaks the MCP protocol over stdio and streamable HTTP but
registers no tools or resources yet. ``list_workspaces``, ``search_documents``,
``run_review`` etc. and per-user API-token authentication are Phase 5 of
docs/BUILD_HANDOFF.md. Clients that connect today get an empty tool list and server
instructions stating this, never stubbed tool results.
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from compliance_core.config import get_settings
from compliance_core.db.session import create_engine
from compliance_core.health import database_probe, readiness

SERVER_NAME = "compliance-workspace"
INSTRUCTIONS = (
    "Compliance Workspace MCP server. Phase 0 skeleton: no tools or resources are "
    "implemented yet. Workspace, document search, review and report tools arrive in Phase 5."
)
IMPLEMENTED_TOOLS: list[str] = []
MCP_PATH = "/mcp"


def build_server() -> MCPServer[Any]:
    return MCPServer(name=SERVER_NAME, instructions=INSTRUCTIONS, version="0.1.0")


def _split_env(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


def build_http_app(server: MCPServer[Any] | None = None) -> Starlette:
    server = server or build_server()
    engine_holder: dict[str, AsyncEngine] = {}

    def engine() -> AsyncEngine:
        # Created lazily so the stdio transport and tests need no database.
        if "engine" not in engine_holder:
            engine_holder["engine"] = create_engine(get_settings())
        return engine_holder["engine"]

    async def health(_: Request) -> Response:
        # Tools in Phase 5 call the shared service layer, which needs the database.
        report = await readiness("mcp-server", {"database": database_probe(engine())})
        body = report.model_dump(mode="json") | {
            "transport": "streamable-http",
            "mcp_path": MCP_PATH,
            "implemented_tools": IMPLEMENTED_TOOLS,
            "phase": "skeleton",
        }
        return JSONResponse(body, status_code=200 if report.status == "ok" else 503)

    server.custom_route("/health", methods=["GET"])(health)

    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_split_env("MCP_ALLOWED_HOSTS", "localhost:*,127.0.0.1:*,mcp-server:*"),
        allowed_origins=_split_env("MCP_ALLOWED_ORIGINS", "http://localhost:*,http://127.0.0.1:*"),
    )
    return server.streamable_http_app(
        streamable_http_path=MCP_PATH,
        transport_security=security,
        host="0.0.0.0",  # noqa: S104 - container interface; Host/Origin checks above
    )
