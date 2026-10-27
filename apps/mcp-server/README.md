# mcp-server

The Compliance Workspace's own MCP server, built on the official `mcp` Python SDK (v2,
`mcp.server.mcpserver.MCPServer`, formerly FastMCP). MCP clients such as Claude Desktop,
Claude Code and Cursor can use it to search documents, check rules, run reviews and read
reports with verifiable citations.

Every tool calls the **same service layer as the REST API**: the same membership and role
checks, upload/search/review-start rate limits, budgets, job queue, citation verifier and
report builder. Review runs execute in the worker exactly like runs started in the web
app.

**Status:** implemented in Phase 5 and verified with a real stdio integration test, HTTP
security tests and an isolated mock preview (`docs/PHASE5_REVIEW.md`). **It has not been
run from Claude Desktop itself.** The configurations below are ready to paste but
unverified on a real Desktop install.

## Tools and resources

| Tool | Scope | What it does |
|---|---|---|
| `list_workspaces()` | read | Workspaces the token can access, your role, document counts |
| `list_documents(workspace_id)` | read | Documents with status, type, pages, OCR |
| `upload_document(workspace_id, filename, content_base64)` | upload | One PDF/DOCX/XLSX/CSV (strict base64, at most `MCP_UPLOAD_MAX_BYTES`, default 10 MiB); ingested in the background. **stdio only** |
| `search_documents(workspace_id, query, top_k=8)` | read | Hybrid search; each hit carries a citation (document, page, character span, chunk) |
| `check_rule(workspace_id, document_id, hypothesis, wait_seconds?)` | review | Checks one hypothesis against one ready document as a small budgeted run. It waits up to `wait_seconds` (default 30, max 120) and returns the verdict, or the `run_id` to poll |
| `run_review(workspace_id, rule_pack_id, document_ids?)` | review | Starts a review (asynchronous) and returns `run_id` |
| `get_run_status(run_id)` | read | Status, counts, budget use, errors |
| `get_report(run_id, format="markdown"\|"json")` | read | The report from verified findings and human decisions |
| `get_citation(citation_id)` | read | Quote, document, page, span, surrounding text, and whether it is still exact |
| `list_rule_packs(workspace_id)` | read | Extra (not in the handoff list): rule pack ids for `run_review` |

| Resource | Content |
|---|---|
| `rulepack://{id}` | The rule pack as YAML (same format as import/export) |
| `report://{run_id}` | The run's latest report as Markdown |

`check_rule` uses the trained model only if the hypothesis is **word for word** one of
the 17 ContractNLI hypotheses it was trained on (see `rule_packs/nda_standard.yaml`).
Anything else, including paraphrases, is decided by the workspace's LLM through the
budget ledger, and its citations are verified like any finding. Each check is a
real, auditable run (marked `origin: mcp_check_rule`). It never pauses for human review,
and its budget is the smaller of the workspace default and `MCP_CHECK_RULE_BUDGET_*`
(default $0.05 / 20,000 tokens).

Document text returned by any tool is **untrusted data** from uploaded files. The server's
instructions tell clients never to follow instructions found inside it.

## API tokens

Create tokens in the web app under **Settings → API tokens for MCP clients**. The secret
is shown once. Only its SHA-256 hash is stored.

- **Scopes:** `read` is always granted. `review` (start reviews and rule checks, which
  **spend LLM budget**) and `upload` are explicit opt-ins. A token never exceeds your
  role: a viewer's token cannot start reviews even with `review`.
- **Workspaces:** all of yours by default, or an allowlist. Anything outside it looks
  "not found", including run, citation and rule-pack ids.
- **Expiry:** 1–365 days, default 90. No token lives forever. At most 20 active tokens per
  user.
- **Checked on every call:** revoking a token, its expiry, or losing workspace membership
  takes effect on the very next tool call, also within an open HTTP session.
- These are this app's own bearer tokens, **not OAuth**. The server advertises no OAuth
  metadata, and clients that only connect through an OAuth flow (Claude Desktop's remote
  "custom connectors") cannot use it. Use stdio from Desktop.

Never put a token in a URL (the HTTP server rejects that), in the repository, or in a
shared shell history.

## Claude Desktop (stdio)

Claude Desktop starts the server as a local process and talks to it over stdin/stdout.
The server needs the stack's database, Redis, document storage and model service, so the
process runs **inside the running `mcp-server` container** via `docker exec`.

Desktop's configuration file on macOS is
`~/Library/Application Support/Claude/claude_desktop_config.json` (Settings → Developer →
Edit Config). Desktop apps do not use your shell's `PATH`, so give absolute paths.
Restart Desktop after editing.

### A. The stack runs in Docker on the same Mac

```json
{
  "mcpServers": {
    "compliance-workspace": {
      "command": "/usr/local/bin/docker",
      "args": [
        "exec", "-i", "-e", "COMPLIANCE_MCP_TOKEN", "compliance-mcp-server-1",
        "python", "-m", "compliance_mcp_server", "--transport", "stdio"
      ],
      "env": { "COMPLIANCE_MCP_TOKEN": "cwmcp_...your token..." }
    }
  }
}
```

`-e COMPLIANCE_MCP_TOKEN` passes the variable by name, so the token is not on a command
line. Check the container name with `docker ps` (`compliance-mcp-server-1` with the
default project name). Run `which docker` in Terminal if `/usr/local/bin/docker` is not
your path.

### B. Desktop on the Mac, stack on a Linux host, over SSH

The token stays on the Linux host; the Mac's config holds no secret.

To create the token from the Mac, forward the web app's port and open its Settings
page (`http://localhost:3000/settings`):

```bash
ssh -N -L 3000:127.0.0.1:3000 you@linux-host     # keep it open; Ctrl-C to stop
```

On the **Linux host** (the account must be allowed to run `docker`). Keep the token file
under your home directory on a filesystem with Unix permissions. Not on an exFAT/FAT
drive: those ignore `chmod`, and the wrapper refuses a file whose mode it cannot trust.

```bash
mkdir -p ~/bin
install -m 0755 apps/mcp-server/scripts/compliance-mcp-stdio.sh ~/bin/compliance-mcp-stdio
install -d -m 0700 ~/.config/compliance-mcp
(umask 077; cat > ~/.config/compliance-mcp/token)   # paste the token, press Enter, then Ctrl-D
```

The wrapper refuses a token file that others can read. It execs
`docker exec -i -e COMPLIANCE_MCP_TOKEN compliance-mcp-server-1 python -m compliance_mcp_server --transport stdio`
and prints nothing to stdout itself. Override the container with
`COMPLIANCE_MCP_CONTAINER` and the file with `COMPLIANCE_MCP_TOKEN_FILE`. Set them in
the remote command if needed, for example
`"COMPLIANCE_MCP_CONTAINER=my-mcp ~/bin/compliance-mcp-stdio"`. Variables set on the Mac
are not forwarded over SSH.

On the **Mac**, first make sure `ssh you@linux-host true` works from Terminal **without
any prompt**: key-based login, the host key already accepted, and the key in the agent or
keychain (`ssh-add --apple-use-keychain ~/.ssh/id_ed25519`). Desktop cannot answer
prompts. Then:

```json
{
  "mcpServers": {
    "compliance-workspace": {
      "command": "/usr/bin/ssh",
      "args": [
        "-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
        "you@linux-host", "~/bin/compliance-mcp-stdio"
      ]
    }
  }
}
```

Quick check from the Mac's Terminal (prints one JSON line; press Ctrl-C to stop):

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"check","version":"0"}}}' \
  | ssh -T -o BatchMode=yes you@linux-host '~/bin/compliance-mcp-stdio'
```

### Acceptance check in Desktop (handoff Phase 5)

In a new chat, ask Claude to:
1. "List my compliance workspaces and the documents in the demo workspace."
2. "Search the demo workspace for clauses on returning or destroying confidential
   information, with citations."
3. "Run the Vendor Confidentiality Review on the demo workspace and show me the report
   when it is done." This needs a token with the `review` scope, and spends the
   workspace's LLM budget if a real provider is configured.

Desktop shows each tool call for approval. The report's findings carry citation ids that
`get_citation` resolves.

## Streamable HTTP (Claude Code, Cursor, scripts)

`POST/GET http://127.0.0.1:8020/mcp` with `Authorization: Bearer <token>` on **every**
request. The endpoint is stateless with JSON responses, and every request is
re-authenticated. Uploads are not available over HTTP.

```bash
claude mcp add --transport http compliance-workspace http://127.0.0.1:8020/mcp \
  --header "Authorization: Bearer cwmcp_...your token..."
```

The port is published on `127.0.0.1` only. Bearer tokens travel in clear over plain
HTTP: put TLS in front before exposing it to any network, and keep `MCP_ALLOWED_HOSTS` /
`MCP_ALLOWED_ORIGINS` (DNS-rebinding protection) accurate. Requests without a valid
token get `401` with `WWW-Authenticate: Bearer realm="compliance-mcp"`. Tokens in query
strings get `400`.

## Configuration

Besides the shared variables (`DATABASE_URL`, `REDIS_URL`, `STORAGE_DIR`,
`MODEL_SERVICE_URL`, `LLM_PROVIDER`, `REVIEW_*`, upload/search rate limits), the server
reads:

| Variable | Default | Meaning |
|---|---|---|
| `COMPLIANCE_MCP_TOKEN` / `COMPLIANCE_MCP_TOKEN_FILE` | – | stdio only: the API token (or a file holding it). Checked at start (exit 2 if unusable) and on every call |
| `MCP_CREDENTIALED_PROVIDERS` | empty | Non-secret: providers whose key the **worker** holds. Compose derives it from the host's `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`; the keys never reach this container |
| `MCP_UPLOAD_MAX_BYTES` | 10485760 | stdio upload limit (never above `MAX_UPLOAD_FILE_BYTES`) |
| `MCP_CHECK_RULE_WAIT_SECONDS` / `_MAX_WAIT_SECONDS` | 30 / 120 | `check_rule` wait before returning the run id |
| `MCP_CHECK_RULE_BUDGET_TOKENS` / `_USD` | 20000 / 0.05 | `check_rule` budget cap (below the workspace default) |
| `MCP_REVIEW_BUDGET_TOKENS` / `_USD` | 400000 / 1.0 | `run_review` budget cap (below the workspace default) |
| `MCP_ALLOWED_HOSTS` / `MCP_ALLOWED_ORIGINS` | localhost | HTTP Host/Origin allowlists |

## Operations and security notes

- stdio writes only the protocol to stdout. Logs are JSON on stderr, including a start-up
  line with the token id (never the token). Unexpected errors return a generic message
  with a reference id. The traceback goes to stderr, and document text is never logged
  at INFO.
- The MCP container has no provider credentials. Starting a review validates the
  provider against `MCP_CREDENTIALED_PROVIDERS`. If that declaration is wrong, the worker
  fails the run with `provider_unavailable` before any provider call.
- Runs that pause for human review (`awaiting_review`) must be resolved in the web app.
- Access to `docker exec` on the host is root-equivalent. For the SSH setup, consider a
  dedicated account.

## Development

```bash
uv run python -m compliance_mcp_server --transport http --host 127.0.0.1 --port 8020
COMPLIANCE_MCP_TOKEN=cwmcp_... uv run python -m compliance_mcp_server --transport stdio
uv run python -m compliance_mcp_server.healthcheck http://127.0.0.1:8020
uv run pytest apps/mcp-server/tests          # stdio integration + HTTP security (needs test DB/Redis)
make smoke-phase5                            # against a running stack (mock LLM by default)
```
