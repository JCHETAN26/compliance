"""Container healthcheck: a real MCP handshake plus the dependency probe.

``python -m compliance_mcp_server.healthcheck [BASE_URL]`` exits 0 only if
1. an MCP client can connect over streamable HTTP and list tools, and
2. ``GET /health`` reports every dependency ``ok``.
"""

from __future__ import annotations

import asyncio
import sys

from mcp import Client

from compliance_core.healthcheck import check
from compliance_mcp_server.server import MCP_PATH


async def _protocol_check(base_url: str) -> int:
    async with asyncio.timeout(8):
        async with Client(base_url + MCP_PATH) as client:
            tools = await client.list_tools()
    return len(tools.tools)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    base_url = (args[0] if args else "http://127.0.0.1:8020").rstrip("/")
    try:
        tool_count = asyncio.run(_protocol_check(base_url))
    except Exception as exc:
        print(f"MCP handshake failed: {type(exc).__name__}: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    ok, message = check(base_url + "/health", [("status", "ok")], timeout=4.0)
    if not ok:
        print(f"/health failed: {message}", file=sys.stderr)  # noqa: T201
        return 1
    print(f"healthy: MCP handshake ok, {tool_count} tools registered")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
