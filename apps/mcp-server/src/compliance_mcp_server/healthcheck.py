"""Container healthcheck for the streamable HTTP transport.

``python -m compliance_mcp_server.healthcheck [BASE_URL]`` exits 0 only if
1. ``GET /health`` reports every dependency ``ok``, and
2. an unauthenticated MCP request is refused with 401 and a ``WWW-Authenticate: Bearer``
   challenge (the endpoint is up and enforcing token auth). A real tool call needs a
   user's API token, which a healthcheck must not hold.
"""

from __future__ import annotations

import sys

import httpx

from compliance_core.healthcheck import check
from compliance_mcp_server.server import MCP_PATH

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "healthcheck", "version": "1"},
    },
}


def _auth_enforced(base_url: str) -> str | None:
    response = httpx.post(
        base_url + MCP_PATH,
        json=INITIALIZE,
        headers={"Accept": "application/json, text/event-stream"},
        timeout=4.0,
    )
    challenge = response.headers.get("www-authenticate", "")
    if response.status_code != 401 or not challenge.startswith("Bearer"):
        return f"expected 401 with a Bearer challenge, got {response.status_code}"
    return None


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    base_url = (args[0] if args else "http://127.0.0.1:8020").rstrip("/")
    ok, message = check(base_url + "/health", [("status", "ok")], timeout=4.0)
    if not ok:
        print(f"/health failed: {message}", file=sys.stderr)  # noqa: T201
        return 1
    try:
        problem = _auth_enforced(base_url)
    except httpx.HTTPError as exc:
        problem = f"{type(exc).__name__}: {exc}"
    if problem:
        print(f"MCP endpoint check failed: {problem}", file=sys.stderr)  # noqa: T201
        return 1
    print("healthy: /health ok, MCP endpoint requires a bearer token")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
