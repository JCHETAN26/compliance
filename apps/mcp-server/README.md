# mcp-server

Built on the official `mcp` Python SDK (v2, `mcp.server.mcpserver.MCPServer`, formerly
FastMCP).

**Phase 0 status: skeleton.** The server speaks MCP over **stdio** and **streamable
HTTP** (`/mcp`), but it registers **no tools or resources**. The workspace, document,
review and report tools, per-user API-token auth and the Claude Desktop setup guide are
Phase 5 of `docs/BUILD_HANDOFF.md`. Clients that connect today see an empty tool list
and server instructions that say exactly this.

```bash
uv run python -m compliance_mcp_server --transport stdio
uv run python -m compliance_mcp_server --transport http --host 127.0.0.1 --port 8020
uv run python -m compliance_mcp_server.healthcheck http://127.0.0.1:8020
```

`GET /health` reports the database probe, the transport and `implemented_tools: []`.
The HTTP transport validates `Host`/`Origin` against `MCP_ALLOWED_HOSTS` and
`MCP_ALLOWED_ORIGINS` (DNS-rebinding protection).
