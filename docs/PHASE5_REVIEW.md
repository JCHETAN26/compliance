# Phase 5 review: MCP server, API tokens and Claude Desktop guide

Implementer: Claude Code. Reviewer and acceptance authority: Codex (`SUPERVISION.md`).
This file is the implementer's plan, acceptance checklist and evidence log. Nothing here
accepts the phase. Status values: `todo`, `in progress`, `implemented` (code and tests
written), `verified locally` (named command ran and passed; output recorded below),
`blocked` (with the reason), `device gate` (needs an interactive check in a real client;
cannot be claimed from mock or stdio tests).

Requirements come from `BUILD_HANDOFF.md` §9, §11 (settings API tokens), §12 (Settings
page), §13 and §14 (Phase 5); from the read-only readiness audit (2026-10-26); and from the
manager's Phase 5 decisions (2026-10-26), summarised here:

- Refactor the shared service layer and keep REST behaviour unchanged. MCP must not import
  `compliance_api`.
- Tokens: per-user, opaque, only hashes stored, `read`-only by default, 90-day default
  expiry. Optional workspace allowlist. Effective access is the membership role
  intersected with the token scopes. Tokens can be revoked.
- `check_rule` is an ad-hoc, budgeted one-rule run. It waits a bounded time and returns a
  durable `run_id` while still pending. There are no unledgered LLM calls.
- Stateless HTTP authenticates the bearer token on every request. The stdio env token is
  re-checked on every call. Uploads are stdio only, size-bounded and strict base64.
- Both resources. A token UI to create, list and revoke tokens, showing each token once.
  All nine tools. Guides, plus a real stdio integration test calling every tool and
  resource, and HTTP security tests.
- No OAuth is advertised. No provider keys go into the MCP container: a non-secret
  capability flag is used only to validate review starts, and the worker stays the
  credential owner. REST credential validation is unchanged.
- No new paid calls: tests and preview use the mock LLM. No commits, no live-stack
  changes.

## Acceptance checklist

### Handoff Phase 5 criteria

| # | Criterion | Status |
|---|---|---|
| H1 | All §9 tools: `list_workspaces`, `list_documents`, `upload_document` (stdio only, size limit), `search_documents`, `check_rule`, `run_review`, `get_run_status`, `get_report`, `get_citation` | verified locally (stdio integration test calls all ten; HTTP lists nine; preview smoke) |
| H2 | Resources `rulepack://{id}` (YAML) and `report://{run_id}` (Markdown) | verified locally (stdio test, HTTP allowlist tests, preview smoke) |
| H3 | Per-user API token created on the Settings page; MCP authenticates with it | verified locally (token API tests, Vitest UI tests, preview smoke via Settings endpoints) |
| H4 | MCP tools call the same service layer as REST (no duplicated logic) | implemented; existing REST suites pass unchanged on the refactor (E1) |
| H5 | stdio integration test that runs the server over stdio and calls every tool (and both resources) | verified locally (`apps/mcp-server/tests/test_mcp_stdio.py`) |
| H6 | `apps/mcp-server/README.md` with a ready-to-paste Claude Desktop config | implemented (Desktop configs for local Docker and Mac-to-Linux SSH; not run on a real Desktop) |
| H7 | From Claude Desktop: search documents, run a review, fetch the report | **device gate**: needs an interactive check in Claude Desktop, either on this Linux host (official Linux beta, installed) or on the user's Mac. Not claimable from mock or stdio tests. |

### Shared service layer (WS1)

| # | Item | Status |
|---|---|---|
| S1 | `compliance_core.runtime.ServiceContext`: engine, sessions, Redis, storage, model client, limiters, retrieval config, arq enqueue. The API's `AppContext` extends it with signer, login throttle and revocation list. | implemented, tested |
| S2 | Review start (`services/review_start.py`): role check, workspace defaults, provider/model resolution via an injected resolver (credential check is a parameter), shared review-start limiter, run creation and enqueue. REST passes the real env credential check. | implemented, tested |
| S3 | Upload (`services/upload_ingest.py`): per-file reservation, zip-bomb limits, format detection, registration, enqueue, staging cleanup. Shared by multipart REST and MCP stdio. | implemented, tested |
| S4 | Search (`compliance_retrieval.service.search_for_user`): membership check, rollback, shared search limiter, `run_search`. | implemented, tested |
| S5 | Report (`services/reviews.report_for_run`): presented findings, budget, `build_report`. Markdown rendering is shared. | implemented, tested |
| S6 | Gate: the existing API, search, document and review tests pass unchanged. | verified locally (E1: 105 passed before the MCP work started) |

### Tokens (WS2)

| # | Item | Status |
|---|---|---|
| T1 | Migration `0004` `api_tokens`: user FK cascade, name, display prefix, unique SHA-256 hash, scopes (checked subset of `read`/`review`/`upload`, `read` required), nullable workspace allowlist, `expires_at` NOT NULL, `last_used_at`, `revoked_at` | implemented, tested |
| T2 | Token format `cwmcp_` + 43-char base64url (256-bit). Only the hash is stored. Plaintext is returned once on creation. | implemented, tested |
| T3 | Defaults: scopes `["read"]`, 90 days. Expiry is 1 to 365 days (no non-expiring tokens). At most 20 active tokens per user. Creation is rate-limited. | implemented, tested (cap race fixed: see manager finding) |
| T4 | Allowlisted workspaces must be the caller's at creation. Later membership loss is enforced per call by the services. | implemented, tested |
| T5 | `GET/POST /settings/api-tokens`, `DELETE /settings/api-tokens/{id}`: cookie auth plus CSRF, `Cache-Control: no-store`. The REST API never accepts MCP tokens. | implemented, tested |
| T6 | `authenticate()`: format check before hashing; constant-time hash compare; revoked, expired, unknown and malformed tokens all give the same generic error; `last_used_at` written at most once a minute | implemented, tested |
| T7 | OpenAPI schema and generated TS types regenerated | verified locally (drift check, E5) |

### MCP server (WS3)

| # | Item | Status |
|---|---|---|
| M1 | Every tool and resource call re-authenticates the token (stdio: env token; HTTP: `Authorization: Bearer` on that request). Revocation, expiry and membership loss take effect on the next call, even within an open HTTP session. | verified locally (HTTP: revocation, expiry and membership loss within one client session; stdio: revocation; mutation-checked) |
| M2 | Scopes: `read` for reads and search; `review` (explicit opt-in, spends budget) for `run_review` and `check_rule`; `upload` for `upload_document`. Role checks still apply, so a viewer with `review` is denied. | verified locally (read-only scope refused; viewer with `review` refused by role) |
| M3 | Workspace allowlist enforced for workspace ids, run ids, citation ids, workspace rule packs and both resources. Anything outside the allowlist is "not found". | verified locally (allowlisted token: workspace, run, citation, workspace pack and both resources not found; built-in pack readable; other user sees nothing) |
| M4 | HTTP: stateless, JSON responses. Bearer middleware on `/mcp` returns 401 with `WWW-Authenticate: Bearer realm=...` and no OAuth metadata. Tokens in the query string are rejected. Host/Origin protection is kept. `upload_document` is not registered. `/health` stays open. | verified locally (401 + challenge, no OAuth routes (404), query-string token 400, no upload over HTTP, foreign Host refused with a valid token) |
| M5 | stdio: token from `COMPLIANCE_MCP_TOKEN` (or `COMPLIANCE_MCP_TOKEN_FILE`); fails fast at start if invalid. All logging goes to stderr. A test shows no stdout corruption and no token in stderr. | verified locally (startup refusal exit 2; stdout-only protocol, mutation-checked; token absent from stderr) |
| M6 | Errors: known service errors give `isError` results with code and message. Unexpected errors give a generic message plus a reference id, logged to stderr with the traceback. No document text is logged at INFO. | implemented, tested (generic `internal_error` with reference; known errors keep their code) |
| M7 | `upload_document`: base64 validated strictly, encoded length checked before decoding, `MCP_UPLOAD_MAX_BYTES` (default 10 MiB, never above the REST per-file limit), sanitised filename, shared upload limiter and service | verified locally (strict base64, oversize refused, sanitised filename, duplicate detection) |
| M8 | `check_rule`: one rule against one ready, non-spreadsheet document as an ad-hoc run (`config_json.origin = "mcp_check_rule"`, no rule pack row). `model_supported` only if the hypothesis equals one of the 17 trained hypotheses verbatim; otherwise LLM via the ledger, verifier and citations. Never pauses for review. Budget is min(workspace default, MCP cap). Bounded wait, then the verdict, or `run_id` plus status while pending. | verified locally (pending run id at wait 0, verdict within the wait, verbatim-only model support, LLM path ledgered, no rule-pack row) |
| M9 | `run_review`: shared review start; budget is min(workspace default, MCP cap); optional `document_ids`; returns `run_id`. Runs paused for human review point to the web UI. | verified locally (stdio, HTTP and preview smoke) |
| M10 | Provider capability without keys: `MCP_CREDENTIALED_PROVIDERS` (non-secret, derived in Compose from the host env with `${VAR:+name}`) is used only to validate starts in the MCP process. Credentials stay with the worker, which fails the run with `provider_unavailable` (no spend) if the declaration is wrong. REST validation is unchanged. | implemented, tested (unit test; Compose interpolation checked with `docker compose config` on a copy, keys absent from the MCP service) |
| M11 | Tool annotations (read-only hints; budget-spending tools marked). Server instructions say document text is untrusted data. An extra read tool, `list_rule_packs`, lets clients find a `rule_pack_id`. | implemented, tested |

### Deployment, UI, docs (WS4–WS6)

| # | Item | Status |
|---|---|---|
| D1 | Compose `mcp-server`: model-service URL and dependency, `document-storage` volume, review env without keys, capability flag, stdio upload limit. Healthcheck: `/health` ok, and unauthenticated `/mcp` gives 401 with `WWW-Authenticate: Bearer` | implemented (Compose rendered and checked; healthcheck verified on the preview container; image build not run locally: root disk ~1.8 GB free) |
| D2 | `pyproject`: `mcp>=2.3,<3`, plus retrieval and llm (not agents: no LangGraph in the MCP image); core gains `arq`; `uv.lock` updated consistently | verified locally (`uv lock` diff is these lines only; `uv lock --check --offline` passes; MCP export has no LangGraph/torch) |
| D3 | Web: user Settings page (header link) with create, list and revoke. The token is shown once, with copy and ready config snippets. Vitest coverage and loading, empty and error states. | verified locally (Vitest, lint, typecheck, production build) |
| D4 | `apps/mcp-server/README.md`: tools and resources, tokens and scopes, local Docker stdio config, Mac-to-Linux SSH stdio config (wrapper reads a 0600 token file), HTTP bearer for Claude Code and other clients, limitations | implemented; wrapper script shellchecked and run against the preview container (SSH hop itself not run) |
| D5 | `scripts/smoke_phase5.py` (HTTP plus optional stdio through `docker exec`); Phase 0 smoke, healthcheck and skeleton tests updated; CI runs smoke 5 | verified locally on the preview (smoke 5 all steps); CI step added; hosted CI pending |
| D6 | Root README MCP sections updated (no "no tools yet") | implemented |

## Test plan and environment

- Isolated DB `compliance_phase5_test` on test Postgres 55432 and Redis db 11 (helpers
  `/dev/shm/cw/bin/pyt5.sh`, `pyt_ci5.sh`; copies of the Phase 4 helpers with only the DB
  and Redis changed). The manager's DBs and Redis 12/15 are not touched.
- Scratch, basetemp and stdio storage go under the container `/tmp` tmpfs or `/dev/shm`,
  never the exFAT SSD. Every run is wrapped in `timeout`, and every MCP client call in
  `asyncio.timeout`.
- Mock LLM and mock rule-check backend only. No provider key is read and no paid call is
  made.
- Preview: the isolated preview (API 18000, worker, model service 18010, web 13300) plus
  an MCP container on 18020 runs `smoke_phase5.py`. The live seven containers, their
  `.pth` pin and the model artifacts are not touched.

## Remaining device gate (not yet verified)

"From Claude Desktop: search, run a review, fetch the report" stays **unverified** until
someone runs those three steps interactively in Claude Desktop. That can be done on this
Linux host or on the user's Mac. The
stdio integration test and the `docker exec` stdio smoke exercise the same server
process and transport that Desktop launches. They do not prove Desktop's own client
behaviour, the SSH hop from the Mac, or Desktop's UI.

**Correction (2026-10-26).** Earlier versions of this file said that Claude Desktop runs
only on the user's Mac and that this Linux host cannot run it. That was wrong. There is an
official Claude Desktop for Linux beta
([download](https://claude.com/download),
[Linux docs](https://code.claude.com/docs/en/desktop-linux)), and `dpkg` shows
`claude-desktop` 2.7032.0 installed here at `/usr/bin/claude-desktop`. The manager opened
it normally to a new chat to check readiness, then quit. No sign-in or MCP configuration
was changed and no model prompt was sent, so whether it is signed in is unknown. The
application refused debug flags (this was respected). The test tools could not reach it
through normal accessibility. The Desktop search, review and report check is therefore
still unverified. The Mac SSH handoff below is still a valid option, not the only one.

## Evidence log

All runs used the RAM kit (`/dev/shm/cw/bin`; durable copies in
`../kaggle-recovery/validation-scripts/`) with the repository mounted read-only. Scratch was
on tmpfs. The database was `compliance_phase5_test` on test Postgres 55432, with Redis
db 11; the manager's databases and the live stack were not touched. Logs are in
`../kaggle-recovery/logs/claude-phase5-*`. Every LLM call was to the mock provider; no
provider key was read.

- **E1 — WS1 refactor neutrality.** `pyt5.sh /src/apps/api/tests
  /src/packages/retrieval/tests/test_search_integration.py -x` gave **105 passed**
  (117 s). These are the existing API and search suites, unchanged, run after the shared
  services replaced the route logic.
- **E2 — Token service and API.** `apps/api/tests/test_api_tokens.py` gave **8 passed**,
  including the concurrent cap regression. A mutation without the user-row lock fails it.
- **E3 — MCP HTTP security.** `apps/mcp-server/tests/test_mcp_http.py`: **7 passed**.
- **E4 — MCP stdio integration.** `apps/mcp-server/tests/test_mcp_stdio.py`: **2 passed**.
  It covers all ten tools and both resources over a real subprocess, a start-up refusal
  for missing or invalid tokens, and per-call revocation. Two mutation checks:
  - Caching the principal (no per-call re-check) **fails** the revocation test.
  - Pointing stdio logging at stdout: at first it still passed, because the pinned SDK
    (`mcp.server.stdio`) diverts fd 1 to stderr while it serves. The stdio entry point
    has since gained a start-up log line (token id only), which is written before the
    SDK claims stdout. With it, the mutation **fails** (the client logs a JSON-RPC parse
    error). This is recorded per the manager's logging guidance: the SDK diversion is an
    independent second layer.
- **E5 — Static gates.** `ruff check .` and `ruff format --check .` both pass (275 files);
  `mypy packages apps conftest.py scripts evals ml` reports **no issues in 238 files**.
  The OpenAPI JSON regenerates identically and the TS types match (drift check).
- **E6 — Web.** In the RAM copy (`web.sh`), lint, `tsc` and Vitest (**109 passed**, 18
  files, up from 104) all pass, and `next build` succeeds with the new `/settings` route.
- **E7 — Dependencies.** The `uv lock` diff is only core `arq` and the MCP server's
  retrieval, llm and `mcp>=2.3,<3` lines; `uv lock --check --offline` passes. The MCP
  package export contains `arq`, `mcp` and `httpx2`, and no LangGraph or torch. Rendering
  Compose (a copy of the file, `docker compose config`, with a dummy key in the
  environment) gives `MCP_CREDENTIALED_PROVIDERS=anthropic,`; only `api` and `worker` carry
  key variables. No image was built: the root disk has about 1.8 GB free.
- **E8 — Preview smoke 5.** The isolated preview was restarted with the current tree (DB
  `compliance_preview` migrated to 0004; MCP container `cw-p4-mcp` on 18020; the web
  standalone rebuilt at 13300). `scripts/smoke_phase5.py --web :13300 --mcp :18020
  --stdio-command "python -m compliance_mcp_server --transport stdio"` passed all seven
  steps (`claude-phase5-smoke5-preview.log`). The container healthcheck module passed
  against 18020.
- **E9 — Local-Docker stdio path.** `docker exec -i -e COMPLIANCE_MCP_TOKEN cw-p4-mcp
  python -m compliance_mcp_server --transport stdio`, driven by raw JSON-RPC from the host
  (initialize at 2025-06-18, tools/list, a tool call): 10 tools, every stdout line
  JSON-RPC, the token in neither stdout nor stderr.
- **E10 — SSH wrapper, remote half.** The installed copy of
  `apps/mcp-server/scripts/compliance-mcp-stdio.sh` (shellcheck clean) was launched as the
  remote shell would run the Desktop config's command
  (`COMPLIANCE_MCP_CONTAINER=cw-p4-mcp ~/bin/compliance-mcp-stdio`, with the token in a
  0600 tmpfs file). It was driven by
  `validation-scripts/phase5_stdio_check.py`, a stdlib JSON-RPC client doing a
  Desktop-style handshake. Against the seeded demo workspace (11 documents ready) it ran:
  search with citations, `run_review` (vendor_confidentiality_review; mock;
  completed: 17 satisfied, 6 violated, 38 needs_review; $0), `get_report`, which equals
  `report://`, and `get_citation` (valid). **PASS**, with the token in neither stream
  (`claude-phase5-wrapper-stdio.log`). A token file with mode 0644 is refused (exit 2).
  The check token was revoked afterwards and its file deleted.
- **E11 — Full CI-equivalent application suite.** `PT_WORKDIR=/tmp pyt_ci5.sh -c
  /src/pyproject.toml --rootdir /src` with all eleven testpaths given explicitly (console
  entry point, repo root not on `sys.path`) gave **464 passed in 406 s, exit 0**: the
  Phase 4 baseline of 444 plus 20 new. Log: `claude-phase5-full-ci.log`. The ML suites
  were not rerun: `ml/` does not import any changed module.
- **E12 — Phase 0 smoke (updated MCP check) on the preview.** All ten steps passed,
  including `/mcp` answering 401 with a Bearer challenge (`claude-phase5-smoke0-preview.log`).
- **Not run:** Docker image builds, the hosted CI stack job, and Claude Desktop.

## Device-gate handoff (isolated mock preview)

This is what the user can run from the Mac (one option; see the correction under
"Remaining device gate"). Nothing here touches the original seven
live containers: their MCP container is still the Phase 0 skeleton, so the generic
`docker compose exec mcp-server` configuration does not apply to them yet. The preview
is mock only (mock LLM, mock rule-check backend, no provider key) and lives in RAM
(`/dev/shm`), so it does not survive a reboot.

- Web: `127.0.0.1:13300` (current build, with Settings → API tokens). API: 18000.
  MCP: container `cw-p4-mcp`, HTTP on `127.0.0.1:18020`. DB: `compliance_preview`.
  Redis: db 13.
- Seeded with the handoff's standard demo workspace: `demo@example.com` / `demo1234` (the
  documented demo account), "Demo - Vendor confidentiality review", 11 ready documents.

1. **Mint a token from the Mac.** In Terminal, run
   `ssh -N -L 13300:127.0.0.1:13300 grv22@<linux-host>` and leave it open. Open
   `http://localhost:13300`, sign in as the demo account, go to **Settings → API tokens
   for MCP clients**, create a token with **review** checked, and copy it.
2. **Store it on the Linux host.** Over SSH:
   ```bash
   install -d -m 0700 ~/bin ~/.config/compliance-mcp
   install -m 0755 "/media/grv22/Extreme SSD2/project/compliance/apps/mcp-server/scripts/compliance-mcp-stdio.sh" ~/bin/compliance-mcp-stdio
   (umask 077; cat > ~/.config/compliance-mcp/token)   # paste, Enter, Ctrl-D
   ```
   The token file belongs under the home directory on the root (ext4) filesystem. It must
   never go on the exFAT SSD, which ignores `chmod` (the wrapper would refuse it). It
   must not appear in a command line, a config file or this repository.
3. **Configure Claude Desktop on the Mac** (`~/Library/Application Support/Claude/claude_desktop_config.json`).
   Key-based SSH without prompts must already work (`ssh -o BatchMode=yes
   grv22@<linux-host> true`):
   ```json
   {
     "mcpServers": {
       "compliance-workspace-preview": {
         "command": "/usr/bin/ssh",
         "args": ["-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
                  "grv22@<linux-host>", "COMPLIANCE_MCP_CONTAINER=cw-p4-mcp ~/bin/compliance-mcp-stdio"]
       }
     }
   }
   ```
   Restart Desktop. Then ask it to list the workspaces, search for return/destruction
   clauses with citations, run the Vendor Confidentiality Review, and show the report.

**Not tested here:** the SSH hop from the Mac, Claude Desktop's own client and UI, and
macOS paths and permissions. E10 exercised everything from the remote shell command
inward with the same wrapper file and the same container.

## Manager review findings — 2026-10-26 (resolve through Claude Code)

- **[P2, resolved — see manager re-check below] Active-token limit raced concurrent
  creation.** Codex independently reproduced this in a separate database
  `compliance_codex_phase5_review`, without touching the implementer's database. Starting
  with 19 active tokens, four concurrent `create_token` calls on warmed database
  connections all succeeded, leaving **23** active tokens despite `MAX_ACTIVE_TOKENS = 20`.
  At that time `create_token` counted before inserting without serializing creation for
  the user. The manager asked for the count + insert to be serialized (for example by
  locking the owning User row), the cap retained, a meaningful concurrent regression added
  through Claude Code, and the independent check re-run after the fix. No Codex product
  edits were made. Evidence:
  `../kaggle-recovery/logs/codex-phase5-token-concurrency-warm.log`; isolated diagnostic
  `/tmp/compliance-token-concurrency-check.py`.

  **Implementer response (Claude Code, 2026-10-26): fixed (the re-check was then pending).**
  `create_token` now takes `SELECT … FROM users WHERE id = :user FOR UPDATE` before
  counting active tokens. The row lock is held until the insert commits (or the error
  rolls back), so creation is serialized per user. The cap is unchanged. New regression
  `apps/api/tests/test_api_tokens.py::test_active_token_cap_holds_under_concurrent_creation`
  starts at 19 active tokens and runs four concurrent `create_token` calls on four
  pre-warmed connections. It expects exactly one creation, three refusals and 20 active
  tokens. Results on `compliance_phase5_test`: the token module passes (8 passed).
  Mutation check: the same test, with a copy of `api_tokens.py` without the lock mounted
  read-only over the source, **fails** (`1 failed`). The independent re-run of the
  Codex diagnostic was left to the manager.

  **Manager re-check (Codex): resolved.** The original warmed-connections diagnostic,
  rerun unchanged against the separate review database, now permits exactly one of the
  four concurrent requests, refuses the other three, and leaves exactly 20 active tokens.
  Evidence: `../kaggle-recovery/logs/codex-phase5-token-concurrency-fixed.log`.

### Manager handoff requirement for the device gate

Keep the original live services and Phase 3 source pin untouched. The existing live
MCP container is still the old skeleton, so a generic `docker compose exec mcp-server`
config alone is not a usable handoff for this checkout today. Alongside the general
README configs, prepare and verify a concrete isolated **mock** Phase 5 preview (web
13300, API 18000, separate MCP 18020/container) and an SSH stdio wrapper targeting that
preview container. Document the Mac SSH port-forward for opening the preview Settings
page and minting a token, plus where a private 0600 token file belongs on the Linux
filesystem. No token in argv, logs, source, or committed configs; no paid provider or
production deployment. Actually exercise the same wrapper from this Linux host over
stdio and retain the evidence. Do not claim the Mac SSH hop or Desktop GUI was tested.
The manager can then give the user a concrete three-step Desktop check.

### Manager scope guidance: stdio logging check

A real client successfully parsing every tool/resource response while logs are produced,
plus checking the captured stderr for token leakage, is sufficient integration evidence.
If the pinned SDK itself redirects stdout, a logging-stream mutation may still pass
because of that independent protection. Record that result honestly; do not add artificial
product behavior or spend extra iterations trying to defeat the SDK solely to force that
mutation to fail. Continue with the UI, preview smoke, final gates and handoff.

### Implementer responses to the manager guidance (Claude Code)

- **stdio logging:** recorded in E4. The SDK's fd-1 diversion explains why the first
mutation passed. The start-up log line stays because it is an operational audit record
(token id and user id at session start). It is not a test-only behaviour. No further
iterations were spent on the mutation.
- **Device-gate handoff:** see "Device-gate
handoff (isolated mock preview)" and evidence E8–E10.

## Manager browser finding — mobile Settings overflow (2026-10-26)

- **Finding (Codex, browser check of `54d0647`).** The full browser flow
  (`/tmp/compliance-phase5-browser-check.py`) passed create, copy, read-only and 90-day
  defaults, workspace scope, one-time reveal, reload, no secret in the list, revoke and an
  immediate MCP 401. It failed **only** the mobile document-width check. At a 390 px
  viewport `/settings` had `document.scrollWidth` 655 (669 with a longer token row). The
  `TABLE` and the sr-only "Actions" header `SPAN` ended at 654.5 px, while the card ended at
  374 px (`/tmp/compliance-phase5-layout-check.py`, demo login, read-only; screenshot
  `../kaggle-recovery/logs/codex-phase5-browser-shots/settings-revoked-mobile.png`).

  **Diagnosis (Claude Code, computed styles on the preview).** The table itself was
  clipped correctly by the `overflow-x-auto` wrapper; its bounding box lies beyond the
  viewport only inside the scroller. The page was widened by the sr-only header span:
  Tailwind's `sr-only` is `position: absolute`, its `offsetParent` was `BODY` (no positioned
  ancestor), and the wrapper was `position: static`. An overflow clip applies only to
  descendants whose containing block is the scroller or lies inside it, so the absolutely
  positioned span escaped the clip and extended the document to the table's right edge.

  **Fix.** `TokenTable`'s wrapper in `apps/web/src/components/settings/api-tokens-panel.tsx`
  is now `relative overflow-x-auto` (one class, with a comment). The scroller becomes the
  span's containing block. The header text stays in the accessibility tree unchanged, and
  the table still scrolls horizontally inside the card. No test was added, per the
  manager's scope (low-impact CSS); browser verification plus the existing gates stand in.

  **Verification (2026-10-26).**
  - Before changing the source, adding `relative` to the wrapper in the live page
    gave `scrollWidth` 390.
  - Web gates in the RAM copy (`web.sh run lint|typecheck|test|build`): eslint and
    `tsc` clean, Vitest **109 passed** (18 files), and `next build` succeeded.
  - Only the preview web on 13300 was restarted. It runs from a fresh standalone copy in
    `/dev/shm/cw-preview/web5`, with the previous copy kept as `web5.prev`, using the
    documented `API_INTERNAL_URL=http://127.0.0.1:18000 PORT=13300 HOSTNAME=127.0.0.1 node
    server.js`. Live 3300, the API, the worker and MCP 18020 were not touched.
  - Manager diagnostics, re-run:
    - `/tmp/compliance-phase5-layout-check.py`, unchanged: **width 390, scrollWidth
      390**. The listed "offenders" are now only table cells inside the scroller (the
      table is 644 px wide and clipped by the wrapper).
    - The sr-only span's containing block is now `DIV.relative overflow-x-auto`.
    - Full browser flow: a copy that differs only in the screenshot directory, so the
      manager's failing screenshot is kept. **PASS** for every step, including the
      desktop/mobile layout assertion and no console errors.
    - Screenshots, taken after the token was dismissed and revoked (only the prefix is
      visible):
      `../kaggle-recovery/logs/claude-phase5-browser-shots/settings-revoked-{desktop,mobile}.png`.
    - Log (no token value; `grep cwmcp_` finds nothing):
      `../kaggle-recovery/logs/claude-phase5-mobile-overflow.log`.
  - Mock preview only: no provider key was read and no paid call was made.

  **Pre-existing, not changed (outside Phase 5; for the manager's triage).** The same
  390 px check on the seeded demo workspace found:
  - The documents table (`documents/documents-page.tsx`) has the same escaping sr-only
    "Actions" header: `scrollWidth` **621**. The same one-class fix would apply.
  - `members-panel.tsx` uses the same pattern but does not overflow with the demo's
    single member.
  - The reviews list is **483** for a different reason. The whole workspace-detail column
    is 483 px wide (a grid/flex min-width issue), not an sr-only escape.
