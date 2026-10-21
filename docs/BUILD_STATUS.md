# Build status

Implementer log for `docs/BUILD_HANDOFF.md`. The independent verification ledger is
kept by the supervisor in `docs/SUPERVISION.md`, `docs/PHASE0_REVIEW.md` and
`docs/PHASE1_REVIEW.md`; this file does not restate it as fact. Phase 1 work is
uncommitted; the supervisor commits and pushes it.

| Phase | Status |
|---|---|
| 0 — Skeleton | **Accepted.** Independent local gates passed, and hosted GitHub Actions run 39678264159 passed all three jobs for commit `e6b709e`. |
| 1 — Ingestion + viewer | **Locally accepted by independent verification** (supervisor: `make ci`, production browser walkthrough, live smoke and a fresh-volume auto-seed smoke; see `docs/PHASE1_REVIEW.md`). **Hosted CI has not run Phase 1 yet** (pending the supervisor's push). In particular, the container auto-download path (model and dataset fetched by the worker on a clean start) is unverified until a hosted run, because containers on this host have no internet. |
| 2–6 | Not started. |

## Phase 1 scope delivered

**Storage and database**
- Migration `0002`, reversible, creates:
  - `documents` (status, stage, progress, attempts, generation, warnings, unique
    `(workspace_id, sha256)`);
  - `document_contents` (the canonical text), `pages`, `document_blocks`,
    `table_sheets`, `table_rows`;
  - `chunks`, with `vector(384)` + HNSW cosine, a generated `tsvector` + GIN, and
    offsets, bboxes, sections and `embedding_windows`;
  - `extracted_entities`, which stays empty until Phase 4.
- Alembic's autogenerate compare reports **0 diffs** against the ORM.
- `LocalStorage` uses server-generated keys, confines every path to the storage root,
  and writes atomically via a staging directory.

**Upload API** (`POST /workspaces/{id}/documents`)
- The role check (owner/reviewer) runs before the body is read.
- The multipart body is streamed with per-file, per-request, file-count, field and
  header caps.
- SHA-256 is computed while streaming.
- Content is sniffed: the extension must match, and zips are checked for bombs
  (ratio, total size, entry count), encryption, traversal paths and macros.
- Filenames are sanitised.
- Rate limiting is atomic, per request (`INCR`, before streaming) and per file attempt
  (`INCRBY`, rejected files included).
- Each file gets its own outcome: `created`, `duplicate` (same workspace) or `rejected`.

**Other document endpoints**
- List, get, original file (inline PDF; other formats as attachment + `CSP: sandbox`),
  chunks, pages (+text), blocks, sheets, rows, entities (honest `not_available`),
  delete (owner/reviewer; removes the row, then the file), and retry (failed documents
  only, new generation).
- Workspace deletion also removes its stored files.
- `document_count` was added to workspaces.

**Ingestion** (`compliance_ingestion` + worker job)
- Parsing runs in an isolated child process with a wall-clock timeout and an
  address-space limit (baseline + `INGEST_PARSE_MEMORY_MB`).
- **PDF:** the text layer from per-character boxes, with rotation applied.
- **OCR:** pages with fewer than 25 non-space characters are OCR'd with Tesseract.
  - The raster cap is enforced before `get_pixmap`; a page that won't fit at 100 DPI
    is refused with a visible warning.
  - OCR errors are classified separately as not installed, tesseract error, timeout
    or other.
- **DOCX:** paragraphs, styles, lists and tables (merged cells de-duplicated).
- **XLSX/CSV:**
  - XLSX is read via a file handle, so extension-less storage paths work; CSV
    encoding and dialect are sniffed.
  - A header row is detected even below title rows, and values get a documented
    display form (ISO dates, integral floats, empty cells omitted from row text).
  - Each row becomes one chunk.
- **Chunking**
  - Sentence-preserving and layout-aware (headings, numbered clauses, paragraphs),
    targeting 200–400 tokens with the bge tokenizer and up to 40 tokens of overlap.
    There is no overlap across section breaks, a chunk never ends on a heading, and a
    small tail merges into the previous chunk.
  - **Sentences are never split.** An oversized sentence becomes one whole chunk with
    its true token count.
- **Embeddings:** real local `BAAI/bge-small-en-v1.5` at pinned revision `5c38ec7c…`,
  with the SHA-256 of `model.safetensors` verified at download and at every load, and
  CPU torch.
  - A chunk longer than 510 tokens is embedded in bounded windows. Each window is at
    most 510 tokens, starts on a word, overlaps the previous window by 64 tokens and is
    re-tokenised to verify it fits.
  - The window vectors are averaged, weighted by token count, and renormalised.
  - The source chunk is unchanged, and `embedding_windows` records the window count.
  - There is no fallback embedding.
- **Job correctness**
  - Jobs are keyed by generation.
  - Results are stored in one transaction that re-locks the row, so a deleted or
    superseded document's results are discarded.
  - Stale derived rows are deleted before insert, so retries never duplicate data.
  - Transient errors retry up to `INGEST_MAX_ATTEMPTS`; other errors fail visibly and
    never leave a document stuck in PROCESSING.
  - `_fail` only transitions PROCESSING documents of the same generation.
  - A per-minute sweep re-queues lost uploads and recovers stale PROCESSING documents,
    with atomic status and staleness checks.

**Seed**
- A manifest maps 10 ContractNLI **test-split** NDAs to synthetic vendors:
  - 7 original PDFs;
  - 2 DOCX;
  - 1 genuinely image-only scanned PDF;
  - 1 fictitious vendor with a missing contract.
- The vendor register XLSX has a title row above its header and a Sources sheet.
- Generation is byte-deterministic.
- `run_seed` is idempotent: an existing user's password is never touched, and files are
  deduplicated by SHA-256. Seeding is refused in production.
- The worker auto-seeds only when the demo workspace does not exist yet.
- `make seed` exists, and `make fetch-artifacts` fetches and verifies the model and
  dataset.
- The ContractNLI license was checked before download and recorded in
  `ml/data/LICENSES.md`.

**Web**
- A workspace shell with tabs (Documents / Settings & members).
- Documents page:
  - drag-and-drop or keyboard file picker, one XHR per file with real progress;
  - client prechecks, with the server authoritative;
  - per-file outcomes;
  - a table showing type, pages, OCR flag, live status/stage/progress (polled), errors
    and warnings, with retry/delete for writers;
  - read-only mode for viewers.
- Viewer:
  - PDF via react-pdf, using pdf.js's legacy build (see failures below), with lazy
    page rendering, a text-layer search with escaped HTML and match navigation, and
    exact box overlays.
  - Clicking a chunk scrolls to it and highlights it.
  - `?chunk=` deep links wait for the PDF to load and use scroll-container-relative
    offsets.
  - Spreadsheets render as per-sheet tables, with the selected row highlighted and
    auto-loaded if needed.
  - DOCX shows its extracted body with an exact-range `<mark>`.
  - Sidebar tabs: Chunks, Entities (an honest notice, distinct from load errors) and
    Details (OCR confidence, model, SHA-256, warnings).
  - Download original.

**Compose and CI**
- The worker image includes Tesseract and CPU torch.
- Each image is a single stage with uv and sources bind-mounted and a manifest-only
  dependency layer (no duplicated venv); the web build uses a single builder stage.
- The `document-storage` volume is shared by api and worker; the worker never sees the
  API's secret volume.
- The `artifact-cache` volume can be swapped for a bind mount via
  `ARTIFACT_CACHE_SOURCE`.
- The CI workflow:
  - installs Tesseract;
  - caches and fetches the artifacts;
  - runs the stack with automatic seeding, `compliance_seed --wait`, and the Phase 0
    and Phase 1 smoke tests.

## Supervisor review findings addressed during Phase 1 (`docs/PHASE1_REVIEW.md`)

1. **Oversized sentences were being split.** That code was removed. An oversized
   sentence now stays whole, and the embedder handles long inputs with bounded token
   windows aggregated into one vector. Tests cover:
   - a 600-word sentence kept intact with its exact token count;
   - windows that are bounded, ordered, verbatim, start-to-end and word-aligned;
   - a vector that differs from the first window alone;
   - a real 512-token-plus case on the full-size model.
2. **The OCR raster cap was applied after rendering.** The DPI is now chosen from
   arithmetic before `get_pixmap`, and pages that cannot fit at 100 DPI are refused.
   Tests use a page object whose `get_pixmap` raises if called, plus a real 200×200-inch
   page with a guarded `get_pixmap`. Exception order is corrected (`TesseractError` is a
   `RuntimeError` subclass), with one test per class.
3. **Dataset cache not ignored.** `.cache/`, `*.zip`, `*.bin`, `*.pt` and generated seed
   output are now ignored (`git check-ignore` confirmed).
4. **Multipart headers were buffered without a cap.** Per-header (4 KiB), per-part
   (8 KiB) and header-count caps now apply. python-multipart's own limit is also
   configured and mapped to the same 400. An incremental test streams the header in
   1-, 7- and 4096-byte chunks and asserts parsing stops near the cap.
5. **Stale job transitions.** `_fail` now requires PROCESSING and the same generation,
   and sweep updates re-check status, generation and staleness in their `WHERE`
   clauses. Unexpected exceptions retry or fail visibly. Tests cover:
   - a late failure after success;
   - the sweep racing a completion;
   - an unexpected embedding error;
   - a crash-retry-then-fail sequence.
6. **Upload rate accounting.** Quota is now reserved atomically per request and per
   file attempt (see above). Tests cover rejected files consuming quota (429 with no
   staging leftovers), and concurrent API uploads with exactly the cap succeeding.
7. **PDF deep-link focus.** It now waits for `numPages`, and the scroll offset uses
   bounding rects relative to the scroll container. It was verified in the browser by
   opening a fresh `?chunk=` link to the last page (overlay geometry measured, overlay
   inside the viewport).

## Phase 1 commands run (chronological, with failures)

1. **License check before download:** fetched the ContractNLI project page and read the
   full license section and Terms and Conditions (CC BY 4.0 plus Hitachi terms), then
   wrote `ml/data/LICENSES.md`. After that, `curl` downloaded `contract-nli.zip`
   (65,362,913 B), and its SHA-256 `e03fc77b…` was pinned trust-on-first-use. Also
   read the HF API for `BAAI/bge-small-en-v1.5` (license MIT, revision `5c38ec7c…`,
   safetensors SHA-256 `3c9f3166…`).
2. **Dependencies:** `uv lock` added CPU torch `2.14.1+cpu` from download.pytorch.org;
   the lock contains **0** `nvidia`/`triton`/`cuda` packages. `uv sync --frozen` grew
   the host venv by about 1.04 GB (free root space went from 4.0 GB to 2.99 GB).
3. **Model:** `python -m compliance_ingestion.embeddings fetch --cache-dir .cache/models`
   verified the model. **Failure:** `mkdtemp` left the directory at mode 0700, which the
   container uid could not read. Fixed in code with `chmod 0755`, and the existing
   directory was changed.
4. **Extraction child process:**
   - First run **failed**: `PanicException: global thread pool not initialized`.
     A flat `RLIMIT_AS` of 3 GB broke the Rust thread pools. Fixed by running the child
     single-threaded.
   - Second run **failed**: `malloc failed` while rendering a 9 MB raster. Measured that
     PyMuPDF reserves about 4.2 GB of virtual memory at import (about 150 MB resident).
     The limit is now set after imports as `VmSize + budget`.
5. **Migrations:**
   - First run **failed**: the `Chunk.text` column shadowed SQLAlchemy's `text()` in
     the class body. Fixed by aliasing the import.
   - Second run **failed**: `sa.SchemaItem` is evaluated at runtime. Fixed with
     `from __future__` and an explicit import.
   - The base → head → 0001 → head round trip then passed, and autogenerate compare
     reported **0 diffs**.
6. **Seed generation:** generated twice in a row, all 11 files were **byte-identical**.
   The scanned PDF has 0 text characters and 1 image per page. Extracted with real
   OCR, its 3 pages scored mean confidence 95.3–95.8.
7. **Tests written alongside the code.** Failures along the way:
   - Chunk overlap test: the fixture's sentences started lowercase, which the
     segmenter treats as continuations. Fixed the fixture.
   - Bbox `get_textbox` check: glyph boxes touch on tight 15pt leading. Fixed by
     probing the central band of the rect.
   - Section-label expectation: the test expected a title that the labeller
     deliberately omits. Fixed the expectation.
   - A corrupt DOCX raised `PackageNotFoundError` unmapped. **Fixed in code.**
   - Scanned and heading-only documents produced **0 chunks**: trailing headings were
     carried forward and dropped. **Fixed in code**, with tests.
   - Every XLSX ingestion failed: openpyxl rejects extension-less storage paths.
     **Fixed in code** (file handle), with a test.
   - Seed logging used `extra={"created": …}`, which collides with a reserved
     `LogRecord` attribute. Fixed.
   - Two over-broad assertions in my API tests were tightened, not loosened.
8. **Docker builds** (`DOCKER_BUILD_NETWORK=host`, serial, disk monitored):
   - worker 1.54 GB (1.32 GB dependency layer plus 88 MB Tesseract), then api 202 MB,
     model-service 201 MB, mcp 219 MB, web about 238 MB.
   - Free root space fell to **807 MB**. Recovered using only project-owned
     artifacts:
     - superseded `compliance/*` images, after their containers were replaced;
     - the 59 remaining Phase 0 build-window private BuildKit records, by ID;
     - each Phase 1 build's private records, by ID.
   - Log: `/tmp/cw-disk/phase1-cleanup.log`. Free space ended at 1.7–1.9 GB.
   - The 48 older records from other projects, unrelated images, volumes and global
     caches were not touched.
9. **Stack upgrade** (`WEB_PORT=3300 ARTIFACT_CACHE_SOURCE=./.cache
   ARTIFACT_AUTO_DOWNLOAD=false docker compose up -d --no-build --wait`): all 7
   services healthy. The API applied `0002`, and the worker loaded the verified model
   and auto-seeded. `make seed` exited 0 ("0 file(s) added, 11 already present", all
   11 READY).
10. **Browser walkthrough** (`/tmp/cw-e2e/phase1_ui_check.py`). The first two attempts
    **failed** on my script's locators: an inexact "Ready" text match also hit the
    `ready4s` filename. The third attempt **failed for real**: the PDF viewer crashed
    with `n.toHex is not a function`, because pdf.js 6's modern build needs ES2025
    built-ins that Chrome 139 lacks. Fixed by aliasing `pdfjs-dist` to its legacy build
    and serving the matching legacy worker.

    While debugging I ran `next dev`, which wrote an untracked `apps/web/AGENTS.md`. I
    deleted it and set `agentRules: false`. Two `pkill`/`pgrep -f` commands matched
    their own shell and killed it (exit 144); I then stopped the dev server by PID, and
    no other process was affected. I also fixed sidebar overflow, previews ignoring
    `line-clamp`, and a missing favicon.
11. **Final production run** after rebuilding the web image:
    - `phase1_ui_check.py http://127.0.0.1:3300` exit 0, "console/page errors: none";
    - `scripts/smoke_phase1.py` exit 0, all 3 steps passed;
    - `make smoke WEB_PORT=3300` (Phase 0 regression) exit 0.
12. **Clean-start check** (separate project `compliance-cleanstart`, fresh volumes,
    ports 3310/8101/8111/8121):
    - healthy in 30 s, with all 11 demo documents READY after 52 s and no manual seed;
    - after deleting a demo document, changing the demo password hash and restarting
      worker and api: 10 documents remained and the hash was unchanged, so no
      resurrection and no reset;
    - `down -v` removed only that project's volumes.
13. **Final gate:** `set -o pipefail; make ci > /tmp/cw-p1/gates/make-ci.log`.
    - The first run had exit **2**: mypy flagged untyped PyMuPDF calls in the new smoke
      script. I added a per-file directive.
    - The rerun had **exit 0**: ruff passed with 104 files formatted, mypy strict found
      no issues in 85 files, the OpenAPI schema and types were current, **pytest 197
      passed**, **Vitest 73 passed**, and the Next.js build compiled.

## Phase 1 known issues and limitations

- **Hosted CI has not run Phase 1 yet.** The automatic in-container model/dataset
  download and the first clean CI start are untested here; this host's containers have
  no egress, and its runs used the bind-mounted, pre-verified cache.
- OCR runs only on pages with almost no text layer, so mixed pages are not OCR'd.
  Character boxes inside an OCR word are evenly divided.
- DOCX headers, footers, footnotes and text boxes are not extracted, and DOCX has no
  pages (citations use offsets).
- Document type is a keyword heuristic. Entities are honestly absent until Phase 4.
- The demo set contains no chunk longer than 512 tokens (the maximum is 395), so
  windowing is exercised only by tests, not by the demo data.
- If the API crashes after the DB commit and before file removal, the stored file can
  be orphaned.
- Search in the viewer covers the PDF text layer plus our extracted text; hybrid
  retrieval is Phase 2.
- **Disk:** about 1.7 GB is free on root. Rebuilding the web image needs roughly 1 GB
  transient (`npm ci`) before scoped pruning. The external SSD is unused.

## Phase 1 supervisor validation commands

```bash
cd /home/grv22/projects/compliance
make fetch-artifacts                   # verifies cached model + dataset (no-op if present)
make test-infra-up
set -o pipefail; make ci 2>&1 | tee /tmp/compliance-supervision/phase1-make-ci.log
# Stack (port 3000 taken on this host; daemon has iptables disabled; containers lack egress)
export WEB_PORT=3300 ARTIFACT_CACHE_SOURCE=./.cache ARTIFACT_AUTO_DOWNLOAD=false
DOCKER_BUILD_NETWORK=host docker compose up -d --build --wait   # serial builds advised: see disk note
make seed
env -u PYTHONPATH uv run python scripts/smoke_phase1.py --web http://127.0.0.1:3300
make smoke WEB_PORT=3300
env -u PYTHONPATH /tmp/cw-e2e/.venv/bin/python -I /tmp/cw-e2e/phase1_ui_check.py http://127.0.0.1:3300
# Clean-start (separate project, removed afterwards)
COMPOSE_PROJECT_NAME=compliance-cleanstart WEB_PORT=3310 API_PORT=8101 MODEL_SERVICE_PORT=8111 \
  MCP_PORT=8121 docker compose up -d --no-build --wait   # then: ... docker compose down -v
```

Artifacts:
- browser screenshots: `/tmp/cw-e2e/p1-*.png`;
- browser log: `/tmp/cw-e2e/phase1-ui.log`;
- gate log: `/tmp/cw-p1/gates/make-ci.log`;
- disk actions: `/tmp/cw-disk/phase1-cleanup.log` (ID lists alongside);
- model and dataset: `./.cache` (git-ignored).

---

## Phase 0 scope delivered (accepted)

- **Monorepo** per handoff §4: uv workspace (`packages/*`, `apps/{api,worker,model-service,mcp-server}`)
  plus `apps/web` (npm). `ml/`, `evals/`, `rule_packs/`, `seed/` and
  `packages/{ingestion,retrieval,agents,llm}` are labelled placeholders (`IMPLEMENTED = False`,
  READMEs naming their phase).
- **Database**: Alembic migration `0001` creates `users`, `workspaces`,
  `workspace_members` (enum `owner|reviewer|viewer`) with constraints (lowercase email,
  unique email, name length, FK cascade/restrict) and enables `pgvector`. The API
  container migrates on start, serialised by a Postgres advisory lock.
- **Auth** (`/api/v1/auth/register|login|logout|me`): Argon2id hashes; HS256 JWT in an
  httpOnly SameSite=Lax cookie with `jti/iss/aud/exp/typ`; Redis revocation list
  (logout, and re-login revokes the previous token); session-bound HMAC CSRF token
  required on every cookie-authenticated state change; per-account failed-login
  throttle (429 + Retry-After); the same error for unknown email and wrong password
  (tested), plus a dummy Argon2 verification to even out timing (not measured);
  production refuses a missing/short `JWT_SECRET` or `COOKIE_SECURE=false`.
  In development an empty `JWT_SECRET` produces a random secret persisted (mode 0600)
  in the `api-data` volume.
- **Workspaces** (`GET/POST /workspaces`, `GET/PATCH/DELETE /workspaces/{id}`,
  `GET/POST /workspaces/{id}/members`, plus `PATCH/DELETE /workspaces/{id}/members/{user_id}`):
  the service layer lives in `packages/core` (reusable by MCP). Non-members get 404.
  Only owners rename, delete or manage members. Members can leave. The owner cannot be
  removed or demoted.
- **Errors/observability**: one error envelope; validation errors never echo inputs;
  JSON logs with request id; security headers.
- **OpenAPI → TS**: committed `openapi.json` plus `openapi-typescript` types. A pytest
  test and `make openapi-check` fail on drift.
- **Web** (Next.js 16.4 App Router, TS strict, Tailwind v4, shadcn-style UI,
  TanStack Query, openapi-fetch): sign up, sign in, workspace list (empty, loading and
  error states), create, and a detail page (rename, members add/role/remove/leave,
  delete with typed confirmation), plus sign out. A same-origin `/api/v1/*` route-handler
  proxy forwards an allow-listed set of headers and rejects path traversal. `proxy.ts`
  does the optimistic redirect; the server layout verifies the session with the API.
  Session expiry mid-use redirects to `/login?expired=1`.
- **Worker**: arq with the `ping` diagnostic job only (queue → worker → DB).
- **Future-phase skeletons, honestly labelled**: `model-service` `/v1/check` returns 501
  and `/health` reports `mode=not_implemented`. `mcp-server` (MCP SDK 2.3 `MCPServer`,
  stdio + streamable HTTP with Host/Origin checks) registers zero tools and its
  instructions say so. `make seed/train/eval-*` exit 2 with "not implemented".
- **Compose**: 7 services, each with a behavioural healthcheck (see README table). Host
  ports bind to 127.0.0.1 and are overridable.
- **CI** (`.github/workflows/ci.yml`): python job (ruff, mypy strict, pytest against
  pgvector + Redis service containers), web job (eslint, tsc, vitest, type-drift
  check, next build), and stack job (compose build, `up --wait`, smoke test, logs on
  failure).

## Environment facts that affect verification

- Port 3000 (and 5432) are used by another project on this host. **Run the stack with
  `WEB_PORT=3300`.** API 8000, model-service 8010 and MCP 8020 were free.
- The Docker daemon runs with `"iptables": false` (`/etc/docker/daemon.json`; not
  modified). Containers on the default bridge have no outbound DNS/network, so image
  builds need `DOCKER_BUILD_NETWORK=host`. Runtime containers do not need internet.
- The host exports a ROS `PYTHONPATH`, which makes pytest autoload `launch_testing`
  and crash. The Makefile `unexport`s it; run raw commands with `env -u PYTHONPATH`.
- Host Python is 3.10; uv provides Python 3.11.13 (`.python-version`). Containers use
  `python:3.11-slim-bookworm`.
- Free disk is about 2.3 GB after the build. Nothing outside this project was deleted.

## Commands run (chronological, with outcomes)

1. `uv lock && uv sync` — 91 packages resolved (Python 3.11.13).
   **Failure:** workspace packages created after the first sync had no editable `.pth`
   (`ModuleNotFoundError: compliance_core`). Fixed with
   `uv sync --reinstall-package compliance-core ...` (the other app packages likewise).
2. `uv run pytest packages/core/tests -m "not integration"` — **failed** at first:
   PYTHONPATH pulled in ROS's `launch_testing` pytest plugin (`No module named 'lark'`).
   Rerun with `env -u PYTHONPATH`: 26 passed.
3. `docker compose -f docker-compose.test.yml up -d --wait` → healthy; core suite with
   integration: 29 passed.
4. `env -u PYTHONPATH uv run pytest apps/api -q` — **2 failed, 34 passed**. The two
   failures were bugs in my new test's assertions: `"input"` matched the message
   "Extra inputs are not permitted", and the password `"short"` matched the error type
   `string_too_short`. The test now asserts no `input` key in each detail and uses a
   distinctive password. The production behaviour (inputs never echoed) was unchanged.
   Rerun: all passed.
5. Worker/model-service/MCP tests: 15 passed. ruff flagged ASYNC110 and PT011 in the
   worker test; fixed with a justified noqa and an exact `match=`.
6. `uv run mypy packages apps conftest.py` — 2 errors (asyncpg stubs; a `None` key in
   the alg=none JWT test), fixed. Then: success.
7. `npm install` (apps/web) — **failed** with ERESOLVE: openapi-typescript 7.13
   requires TypeScript ^5, so TS is pinned to ~5.9.3. jsdom 30 needs Node ≥22.22 (host
   has 22.14), so jsdom is ^29.
8. `npx eslint` — failures on `window.location.assign` and then React Compiler hook
   rules in `providers.tsx`. Rewritten as a cache-subscription component using
   `router.replace`. `npx tsc --noEmit` failed once on the injected fetch signature;
   fixed.
9. `npx vitest run` — 53 passed (8 files). `npx next build` — succeeded.
10. `docker compose build` — **failed**: `dns error ... files.pythonhosted.org`
    (iptables-disabled daemon, see above). Added `build.network: ${DOCKER_BUILD_NETWORK:-default}`.
    `DOCKER_BUILD_NETWORK=host docker compose build` — all 5 images built
    (web 235 MB, api 222 MB, model-service 222 MB, mcp-server 245 MB, worker 197 MB).
11. `WEB_PORT=3300 docker compose up -d --wait` — all 7 services healthy.
12. `env -u PYTHONPATH uv run python scripts/smoke_phase0.py --web http://127.0.0.1:3300`
    — 10/10 checks passed (healthz, redirect, sign up through the proxy, server-verified
    shell, CSRF rejection, create/list/detail, membership 404/403, rename/delete, logout
    revocation incl. replayed token, api ready + model-service 501 + MCP health).
13. Browser walkthrough (scratch, not in repo: Playwright 1.64 in `/tmp/cw-e2e/.venv`,
    system Chrome): `env -u PYTHONPATH /tmp/cw-e2e/.venv/bin/python -I /tmp/cw-e2e/ui_check.py http://127.0.0.1:3300`.
    - Attempt 1 **failed**: a strict-mode locator in my script matched two cells. Script bug; fixed with `exact=True`.
    - Attempt 2 **failed** at the final expiry step: a 30s click timeout because the button
      detached. Diagnosis: after sign-in the form calls `router.replace` +
      `router.refresh()`. `wait_for_url` returned before the refresh finished, and the
      script then swapped in a bogus cookie, so the in-flight refresh hit the server layout,
      got 401 from `/auth/me` and correctly redirected to `/login?next=…&expired=1`.
      That is the intended UX, not a focus refetch or query invalidation. The script now
      waits for `networkidle` and tests the two paths separately: (a) reload with a
      bogus cookie redirects server-side to `/login?next=%2Fworkspaces&expired=1`;
      (b) a client mutation after the session dies redirects to `/login?expired=1…`.
    - Attempt 3: **passed**, "console/page errors: none". Screenshots in `/tmp/cw-e2e/*.png`.
14. Healthcheck fault injection: `docker compose stop redis`. The API readiness probe
    returned 503 (`redis` check error), the API container went **unhealthy** after its
    retry window, and the worker crash-looped. After the restart
    (`WEB_PORT=3300 docker compose up -d --wait`) that command exited early ("dependency api
    failed to start") because the API was still unhealthy at that instant. Within one
    interval the API returned to healthy; all 7 were healthy and the smoke test passed again.
15. Session persistence: registered through the web, ran `docker compose restart api`,
    and `/auth/me` returned 200 (the generated secret persisted at
    `/data/secrets/jwt_secret`, `-rw------- app`).
16. Final gates: `make lint typecheck openapi-check test` — ruff clean, 64 files
    formatted, mypy strict clean (54 files), OpenAPI and types up to date, **pytest
    80 passed**, **vitest 53 passed**. `make ci` (adds `next build`) also passed earlier
    with identical code. `make seed` → exit 2 "not implemented".
17. Images were built before the final README edits. The only build-context files
    changed since are `apps/{model-service,mcp-server}/README.md`, so the runtime code
    in the running images is identical to the working tree.

## Blocked / not done

- ~~`.pre-commit-config.yaml` not created~~ **Resolved.** My write was denied by the
  session's sensitive-file permission rule. The supervisor reviewed my proposed content
  and installed it unchanged. I confirmed the installed file is identical to my proposal
  (local hooks: `uv run ruff check --fix`, `uv run ruff format`,
  `uv run mypy packages apps conftest.py scripts`, `npm --prefix apps/web run lint`,
  `npm --prefix apps/web run typecheck`) and ran
  `PRE_COMMIT_HOME=/tmp/cw-disk/pre-commit env -u PYTHONPATH .venv/bin/pre-commit validate-config .pre-commit-config.yaml`
  → exit 0. Hooks were not installed into `.git/hooks`, and the hooks themselves have not
  been run through pre-commit (the same commands pass via `make lint typecheck`).
- **Hosted CI not executed — still pending.** (no push, by instruction). The workflow's actions
  (`actions/checkout@v5`, `actions/setup-node@v5`, `astral-sh/setup-uv@v6`) and the
  GitHub-runner path (default bridge networking, port 3000) are unverified until a run.
- No Playwright E2E in the repo (Phase 6 per handoff). The browser check above is a
  scratch script.

## Known issues / limitations

- Login throttling is per account only. There is no per-IP limit, because the API sits
  behind the web proxy and forwarded IPs are not trusted. Registration is not rate
  limited.
- Add-member reveals to workspace owners whether an email is registered (no invitations).
- No ownership transfer. Access tokens last 8h by default with no refresh token.
- The web `/healthz` checks API liveness, not readiness, so a Redis outage marks only
  `api` (and the worker) unhealthy, not `web`.
- `docker compose up --wait` exits non-zero if a dependency is momentarily unhealthy
  during recovery; rerunning (or waiting one healthcheck interval) resolves it.
- arq 0.28 calls redis-py's deprecated `close()`. The DeprecationWarning is filtered in
  pytest config for the `arq` module only.

## README accuracy check (supervisor follow-up)

I checked the README claims against the tests, Makefile and workflow. Corrections made:

- "and CI" became "a CI workflow, run locally; hosted run pending".
- The logging claim was narrowed to what the code does: JSON access lines with a request
  id, also returned as `X-Request-ID` (tested). Request bodies are never logged; this is
  true by code inspection, with no test.
- Added "registration is not rate limited" to Limitations.
- Documented the optional `uv run pre-commit install`.

Claims confirmed against tests or config:

- Healthcheck behaviour for each of the 7 services (compose file plus route, worker and
  MCP tests).
- Lowercase-email DB constraint (`test_migrations`).
- Token revocation on logout and on re-login, CSRF, account throttle with `Retry-After`,
  404 for non-members, owner-only management, and the error envelope without input
  echo (`apps/api/tests`).
- Proxy header allow-list and path-traversal rejection (vitest).
- OpenAPI/TS drift caught by pytest (JSON) and by the CI web job and `make openapi-check`
  (types).
- `make ci` composition and `make seed` exiting 2.

## Disk: project-only cleanup proposal (nothing deleted)

Root filesystem: about 2.2 GB free (`df -h /`, 2026-10-21 21:20 UTC). I inspected
everything with `docker buildx du --format '{{json .}}'`, `docker images`,
`docker system df -v` and `du -sh`. The analysis files are in `/tmp/cw-disk/`.

**A. Docker build cache from the Phase 0 build. Recommended; about 1.97 GB.**
The BuildKit cache has 162 records. 114 were created on 2026-10-21 between 21:07:35
and 21:09:10 UTC, which is exactly the Phase 0 `docker compose build`. The other 48
(6.75 GB) date from 2026-10-03 and belong to other projects; leave them alone.
Of the Phase 0 records, **80 are private (`Shared=false`) and reclaimable, totalling
1,970 MB**. Shared records (436 MB) back the current `compliance/*` images and free
nothing while those images exist. The largest private records:

| Cache ID | Size | Description |
|---|---|---|
| `lljdktz6185d5d3s3trl63kx6` | 731.2 MB | web `npm ci` layer |
| `tkebzivwaqjb4ufddxud27yjy` | 729.9 MB | web builder `COPY --from=deps node_modules` |
| `8e7iijahlaphwvhmhw9ls2nrx` | 120.8 MB | web `npm run build` |
| `wy7w1e76hvmuxyob7i14dt80t` | 118.0 MB | `uv sync` (one Python service) |
| `753ufd51en3n2yfdz6bo6591a` | 95.9 MB | `uv sync` |
| `knfod57l58p0gm8go2eyqapo8` | 95.8 MB | `uv sync` |
| `myololmyrjkjv51ggjta03bri` | 70.0 MB | `uv sync` |
| (73 more) | 8.6 MB total | context/dockerfile sources, small layers, the `/root/.cache/uv` and `/root/.npm` cache mounts (61 kB and 154 kB) |

Proposed command (IDs only; never `docker builder prune` without filters, which would
also remove the other projects' cache):

```bash
xargs -I{} docker buildx prune --force --filter id={} < /tmp/cw-disk/private_reclaimable_ids.txt
```

<details><summary>All 80 IDs (also in /tmp/cw-disk/private_reclaimable_ids.txt)</summary>

```
0zu0iwo9v1fi5ezc4mui18wgt 4vmx6alztwxqab4teol6ake25 70ht15kkpkyh9en01m32olaj1 bck9xi2xq4oprjocpx44820q2
l7o6amoaq4798zuw4fe8yi2y7 ojn2b40t1hyspvp9qgb667fku p7i4dz669kq0nma23ybk8v7fu tipyt3thrmtyz07y11qskk4jq
vov10ninm5dav4icrl5hzdtna wyta04rbapaw0qqx2vxyb50vt iwj3tanajzp2ap5usprj2tej4 brhurnd1py3ji2fu8bysf9ux0
nzf8h1bqy3rw8a74p3s8j5eiw o3cm3nr7r5h50q5slisno5f2p o76u7v5w1d6t7689vxze5503k jbt9o14isw2kze5n7tv8h82p1
64l49qfyov7g4tn1u49do1hdv mxp04wcl3m93miqukl5ikxuws o3h6p602kuw8k9tfg3j7hnijj s89e6j74it07s1i66ydf55e24
vjtlbc0yucv8xvves66ujr1dj dwgkgtuhku2lbbkxt0rnh7uii ghh32sjj1mxdkiffiq7zc643z j5uubiougkv5rx59542650qtw
jszchmamx8j68jkerzalq4aor wsu2pmpn3vq7zxrzmiepx9myj xrp4froomevv37hys7zpbok4g ye4g8yaaaq3tcbav63i939fsq
d7rd1ttklxf5lgxjkt7j83cxb a2ylhure5vbma43lwsu213a8z a4phiage3ew0b5ddiqwxbbbpq hdalk0mply2zzxfzosde52z3h
i23zvwu15hpy5hq8u16xn3okn l3q7ocp6ok2i33ze88tqyrshq t3jxw6uhz84ksamo3b9us1v44 y5ml6fr8p8gd5de3sp10y89ab
20oj5lx2u6atbynsvz2f0z560 nxeb5qdrrr8a225oxnatryjkl oa8mb0dr2syehg661dtso6fls r4cvvwze9xmp2eeshg875m6hu
smogcags27s7e3o7klxyf3fi7 up3bgvvq6qdh3eqleiej31sta y4byumjcwwuq8sruhiuwn9j4r pp5s2fqnhuy63tce44brhoo1j
0i9iw53ev3l9k1rjwhr47t23k 9nxudilro6k5rirbgmp87g71w gq5vkk78tl0hllorc6xp225xx l72eqgfqximsnxszfz5nbwigw
okn3moug07euec74usdrcdrl2 rm7fu7fb6azuawzuexvxvkev1 z4r7wnghv67ev2bxm9eyma3fm 0q9t3y3thgavbv4h4r2796apu
ajh2jpiag3b7ky2lj85pmg5m0 cz1f3swq8d606od2kxux9n85g kaco9o1ca5d8zvbe04ofkpzxe kppud5biotzu7jaz6qddqmtcc
v8lp5l287sj5l6n0kwassdyl6 z6ejv98lhasgh4zwh09tz9p5f 7rj46i3mejrueuw95tyrgzw3c fh7dym3d15vgye9hw704mse3v
8v7k7p5kidim272yjbtu5deg9 alw3mvb9pula6itypbqroni4x hap1bytbx30djyrq7fgjdqzyl jmxdejo2mvta1vpv7z4ojuwym
jn5ary2ibjfr5ep4uuv80te5y pdnpxswpratujpx15oju2cx7z vt99a4spo41fo10vhkihr54yl wxixemij0eqz9my1mbq0oguy2
z7hqxpl96jopq37okq49piukg 4rzh6wvnjjbvsuzzqx0ccasj7 m4fm5dt3dc068fte416sppkl9 t604x8qby2x4ngecpycfontdx
xqvusdur5opotp705q8vlk4ru myololmyrjkjv51ggjta03bri knfod57l58p0gm8go2eyqapo8 753ufd51en3n2yfdz6bo6591a
wy7w1e76hvmuxyob7i14dt80t 8e7iijahlaphwvhmhw9ls2nrx tkebzivwaqjb4ufddxud27yjy lljdktz6185d5d3s3trl63kx6
```
</details>

Trade-offs and caveats:
- The next `docker compose build` re-runs `npm ci`, `next build` and `uv sync` from
  scratch (network and a few minutes).
- The `--filter id=` form is BuildKit's documented filter, but I have not exercised it
  here because that would delete. Run `docker buildx du` before and after to confirm
  that only these IDs disappear.
- If the list is stale (new builds since 21:20 UTC), regenerate it with
  `docker buildx du --format '{{json .}}' > /tmp/cw-disk/buildx.jsonl && python3 /tmp/cw-disk/analyze.py /tmp/cw-disk/buildx.jsonl`.

**B. Regenerable project files. Optional; about 147 MB.**
- `apps/web/.next` (119 MB): the `next build` output, recreated by every `make ci`.
- `.mypy_cache` (28 MB), `.ruff_cache` (172 kB), `.pytest_cache` (36 kB),
  `apps/web/tsconfig.tsbuildinfo` (160 kB).

**C. Keep for now.**
- `.venv` (220 MB) and `apps/web/node_modules` (796 MB): needed for dev and CI.
- `/tmp/cw-e2e` (140 MB): the Playwright venv used by the supervisor's walkthrough
  command. Delete once that check is no longer needed.
- `compliance_*` volumes (postgres 48.5 MB): running stack data.

**Excluded, not project-owned or shared:**
- The 48 older build-cache records.
- All 9 dangling images (2 weeks to 3 years old; none from this project).
- `pgvector/pgvector:pg16` and `redis:7.4-alpine` (needed by the stack and tests).
- `~/.cache/uv`, `~/.npm` and `~/.cache/ms-playwright` (global caches).
- `/tmp/compliance-supervision` (the supervisor's).
- Everything on `/media/grv22/Extreme SSD2`. It has 58 GB free and could host datasets,
  checkpoints and Docker artifacts from Phase 3 on, but nothing has been written there.

## Decisions affecting later phases (recorded from the user via supervisor)

- Full GPU training runs on **Colab** (`nvidia-smi` fails on this host).
- The real LLM provider for the application, demo and zero-shot baseline is the
  **Anthropic API**. The mock LLM remains for tests and offline development.

## Phase 0 validation commands (historical)

```bash
cd /home/grv22/projects/compliance
# Static + unit/integration (test Postgres/Redis on 127.0.0.1:55432/56379)
make test-infra-up
make lint typecheck openapi-check test build-web     # or: make ci
# Full stack (port 3000 is taken on this host; daemon has iptables disabled)
DOCKER_BUILD_NETWORK=host WEB_PORT=3300 docker compose up -d --build --wait
docker compose ps                                    # expect 7 x (healthy)
make smoke WEB_PORT=3300
# Browser walkthrough (scratch tooling outside the repo)
env -u PYTHONPATH /tmp/cw-e2e/.venv/bin/python -I /tmp/cw-e2e/ui_check.py http://127.0.0.1:3300
# Teardown
docker compose down            # add -v to drop volumes
make test-infra-down
```
