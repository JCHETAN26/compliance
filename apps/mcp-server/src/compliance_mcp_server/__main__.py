"""``python -m compliance_mcp_server [--transport stdio|http] [--host H] [--port P]``."""

from __future__ import annotations

import argparse
import asyncio
import os

import uvicorn

from compliance_core.config import get_settings
from compliance_core.logs import configure_logging
from compliance_mcp_server.server import build_http_app, build_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Compliance Workspace MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", "8020")))
    args = parser.parse_args()

    if args.transport == "stdio":
        # stdout carries the protocol; logs go to stderr via the SDK defaults.
        asyncio.run(build_server().run_stdio_async())
        return

    configure_logging("mcp-server", get_settings().log_level)
    uvicorn.run(build_http_app(), host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
