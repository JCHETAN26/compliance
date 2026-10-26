"""``python -m compliance_mcp_server [--transport stdio|http] [--host H] [--port P]``.

stdio needs an API token in ``COMPLIANCE_MCP_TOKEN`` (or a file named by
``COMPLIANCE_MCP_TOKEN_FILE``). It is checked once at start (exit 2 if unusable) and again
on every call. stdout carries only the MCP protocol; all logs go to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import NoReturn

import uvicorn
from sqlalchemy.exc import SQLAlchemyError

from compliance_core.config import get_settings
from compliance_core.logs import configure_logging
from compliance_core.services.errors import AuthenticationError
from compliance_mcp_server.config import get_mcp_settings
from compliance_mcp_server.runtime import McpRuntime
from compliance_mcp_server.server import build_http_app, build_server

logger = logging.getLogger("compliance_mcp_server")
TOKEN_ENV = "COMPLIANCE_MCP_TOKEN"  # noqa: S105 - a variable name, not a secret
TOKEN_FILE_ENV = "COMPLIANCE_MCP_TOKEN_FILE"  # noqa: S105 - a variable name


def _fail(message: str, code: int) -> NoReturn:
    print(f"compliance-mcp-server: {message}", file=sys.stderr)
    raise SystemExit(code)


def stdio_token() -> str:
    token = os.environ.get(TOKEN_ENV, "").strip()
    path = os.environ.get(TOKEN_FILE_ENV, "").strip()
    if not token and path:
        try:
            token = Path(path).read_text("utf-8").strip()
        except OSError:
            _fail(f"cannot read the token file named by {TOKEN_FILE_ENV}.", 2)
    if not token:
        _fail(
            f"set {TOKEN_ENV} (or {TOKEN_FILE_ENV}) to an API token from the web app's "
            "Settings page.",
            2,
        )
    return token


async def run_stdio(runtime: McpRuntime) -> None:
    try:
        try:
            principal = await runtime.authenticate(runtime.stdio_token)
        except AuthenticationError:
            _fail(f"the API token in {TOKEN_ENV} is invalid, expired or revoked.", 2)
        except (OSError, SQLAlchemyError):
            _fail("cannot reach the database (check DATABASE_URL).", 1)
        # Audit trail: which token (id only, never the secret) opened this stdio session.
        logger.info(
            "stdio MCP server starting",
            extra={"token_id": str(principal.token_id), "user_id": str(principal.user_id)},
        )
        await build_server(runtime).run_stdio_async()
    finally:
        await runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compliance Workspace MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", "8020")))
    args = parser.parse_args()
    settings = get_settings()

    if args.transport == "stdio":
        # stdout is the protocol channel: every log line must go to stderr.
        configure_logging("mcp-server", settings.log_level, stream=sys.stderr)
        runtime = McpRuntime(
            transport="stdio",
            settings=settings,
            mcp=get_mcp_settings(),
            stdio_token=stdio_token(),
        )
        asyncio.run(run_stdio(runtime))
        return

    configure_logging("mcp-server", settings.log_level)
    runtime = McpRuntime(transport="http", settings=settings, mcp=get_mcp_settings())
    uvicorn.run(build_http_app(runtime), host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
