# Compliance Workspace

A full-stack, multi-agent compliance review workspace: upload contracts and vendor
registers, run a versioned rule pack, and get a report whose findings cite the exact
source passage. The full specification is in [`docs/BUILD_HANDOFF.md`](docs/BUILD_HANDOFF.md).

> **Status: Phases 0–5 accepted (see [`docs/SUPERVISION.md`](docs/SUPERVISION.md)).
> Phase 5 (MCP server, API tokens) was accepted by the supervisor on 2026-10-27
> (America/Phoenix) at source head `80432fe`, after hosted CI run 40396544929 passed all
> three jobs and the interactive check in the official
> [Claude Desktop for Linux beta](https://code.claude.com/docs/en/desktop-linux) (2.7032.0)
> passed: search, run a review and fetch the report, against the isolated mock preview
> ([`docs/PHASE5_REVIEW.md`](docs/PHASE5_REVIEW.md)). Phase 4 (agents, live run, report,
> overrides) was accepted earlier after the user-approved real Claude Haiku flagship run
> and passing hosted CI for head `6ea1588`
> ([`docs/PHASE4_REVIEW.md`](docs/PHASE4_REVIEW.md)). Phase 6 has not started; it will
> begin on a separate branch from the accepted Phase 5 commit.**
>
> - **Phase 0** (skeleton, auth, workspaces, compose, CI): accepted, hosted GitHub
>   Actions run 39678264159 (commit `e6b709e`).
> - **Phase 1** (ingestion + viewer): accepted, hosted GitHub Actions run 39687462736
>   (commit `ec6c254`):
>   - authenticated, role-scoped multi-file uploads, validated and limited while streaming;
>   - sandboxed PDF/OCR/DOCX/XLSX/CSV parsing;
>   - sentence-preserving chunking with exact offsets and boxes;
>   - real local `BAAI/bge-small-en-v1.5` embeddings in pgvector, plus full-text vectors;
>   - the documents page and source viewers, and an automatically seeded demo workspace.
> - **Phase 2** (accepted):
>   - hybrid search (pgvector HNSW + Postgres full-text `ts_rank_cd`, run in parallel,
>     fused with weighted reciprocal rank fusion), optional pinned cross-encoder rerank;
>   - `POST /api/v1/workspaces/{id}/search` and a Search tab whose results open the
>     exact source highlight (PDF boxes, spreadsheet rows, DOCX text);
>   - query embeddings served by the model service with the same pinned encoder as
>     ingestion;
>   - `make eval-retrieval`: ContractNLI evidence retrieval with parameters selected
>     only on DEV and frozen before TEST, plus an HNSW sweep (results below).
> - **Phase 3** (implementation, training and evaluation complete; acceptance per
>   [`docs/SUPERVISION.md`](docs/SUPERVISION.md)): the rule-check model pipeline
>   (`ml/`): official ContractNLI data and the 17-rule `nda_standard` pack, a DeBERTa-v3
>   evidence + label cross-encoder with a ModernBERT ablation and an unseen-rule run,
>   majority / TF-IDF / Anthropic zero-shot baselines, calibration, cascade analysis,
>   ONNX fp32 and weight-only quantized ("int8") export with parity, and a
>   micro-batching `/v1/check`. Full GPU training ran on Kaggle; the full Claude Haiku 5.5
>   baseline (all DEV + TEST pairs) and the offline evaluation are complete (results
>   below). The local stack serves the fp32 artifact with the DEV-selected tau 0.3. The
>   quantized file passed full DEV and TEST but is slower than fp32 on CPU, so fp32 stays
>   served (user decision). Hosted CI passed for the final Phase 3 head `ccfff28`
>   (run 40277029790).
> - **Phase 4** (accepted): a LangGraph graph of seven agents
>   (planner, extraction, retrieval, rule check, cross-document reasoning, citation
>   verifier, report writer) run by the worker, with Postgres checkpoints, a budgeted LLM
>   ledger, live agent events over SSE, human review with audited overrides, and the rule
>   pack, review, live-run and report pages. Hosted CI run 40346107207 passed all three
>   jobs for head `6ea1588`. One approved synthetic flagship with Claude Haiku 5.5 and
>   the served fp32 model completed within its $0.25 cap ($0.006062 by the application
>   ledger). See "Reviews (Phase 4)" below.
> - **Phase 5** (accepted): the MCP server
>   ([`apps/mcp-server/README.md`](apps/mcp-server/README.md)) exposes nine tools
>   (plus `list_rule_packs`) and the `rulepack://` and `report://` resources over stdio
>   and streamable HTTP. Every call is authenticated with a per-user API token created on
>   the Settings page, and goes through the same service layer as the REST API. Verified
>   with stdio and HTTP integration tests, an isolated mock preview, hosted CI run
>   40396544929 (`80432fe`), and an interactive Claude Desktop for Linux check: Desktop's
>   native MCP tools searched, ran the mock Vendor Confidentiality Review (run
>   `ed98117f-…`, 61 tasks, $0.00) and fetched the report, and the run matched the
>   database. The Mac-to-Linux SSH path is still untested (not needed for acceptance).
> - **Not implemented yet:** the end-to-end eval suite and dashboard (Phase 6). The
>   panels reserved for them say so explicitly instead of returning fake results.
>
> Verification logs are in [`docs/BUILD_STATUS.md`](docs/BUILD_STATUS.md).

## Architecture (current)

```mermaid
flowchart LR
  browser[Browser] -->|same-origin /api/v1/*| web[web<br/>Next.js App Router]
  web -->|BFF proxy, streamed uploads| api[api<br/>FastAPI]
  api --> pg[(postgres<br/>pgvector + tsvector)]
  api --> redis[(redis<br/>queue, limits)]
  api --> store[(document-storage<br/>volume)]
  worker[worker<br/>arq] --> redis
  worker --> pg
  worker --> store
  worker -->|isolated child process| parse[parse + OCR + chunk]
  worker --> emb[bge-small-en-v1.5<br/>CPU embeddings]
  api -->|query embeddings, rerank| model[model-service<br/>same pinned encoder;<br/>rule check: verified ONNX artifact<br/>or untrained fallback (503)]
  cache[(artifact cache<br/>models + dataset)] --> worker
  cache --> model
  client[MCP clients<br/>Claude Desktop via stdio,<br/>others via HTTP + bearer token] --> mcp
  mcp[mcp-server<br/>tools + resources,<br/>per-user API tokens] --> pg
  mcp -->|query embeddings| model
  mcp -->|enqueue runs| redis
```

| Service | Current behaviour | Healthcheck |
|---|---|---|
| `web` | Auth, workspaces, documents page (drag-and-drop upload with per-file progress, live status), document viewer, search | `/healthz`: Next.js serving **and** API reachable via the proxy upstream |
| `api` | Auth, workspaces, documents (upload, list, metadata, original file, chunks, pages, blocks, sheets/rows, entities status, delete, retry), hybrid search; runs migrations on start | `/api/v1/health/ready`: DB query, Alembic at head, pgvector installed, Redis PING |
| `worker` | `ingest_document` jobs, stale-document sweep, demo seeding on first start | `arq --check` heartbeat; it only heartbeats after the DB is reachable and the pinned model is SHA-256-verified and loaded |
| `model-service` | `/v1/embed` (query/passage embeddings with the pinned ingestion encoder), `/v1/rerank` (only when `RERANK_ENABLED=true`), `/v1/check` (rule-check model from a SHA-256-verified ONNX artifact with dynamic micro-batching; without a verified trained artifact: 503 `rule_check_unavailable`, see [`apps/model-service/README.md`](apps/model-service/README.md)) | `/health`: `embedding=ready` (weights SHA-256-verified and loaded); `rule_check=ready|unavailable` is reported, not required |
| `mcp-server` | MCP tools and resources over streamable HTTP (`/mcp`, bearer token, stateless) and stdio (`docker exec`), backed by the REST service layer; see [`apps/mcp-server/README.md`](apps/mcp-server/README.md) | `/health` DB probe, plus `/mcp` refusing an unauthenticated request with 401 `WWW-Authenticate: Bearer` |
| `postgres` | pgvector/pgvector:pg16 | `pg_isready` + `select 1` |
| `redis` | Redis 7.4, AOF | `redis-cli ping` = `PONG` |

## Quickstart

Requirements: Docker with Compose v2. For development: [uv](https://docs.astral.sh/uv/)
(it installs Python 3.11) and Node.js 22.

```bash
docker compose up --build --wait     # all 7 services, waits until healthy
open http://localhost:3000           # sign in as demo@example.com / demo1234
```

On the first clean start, the worker downloads the pinned embedding model (about 130 MB)
and the ContractNLI archive (65 MB) into the `artifact-cache` volume. It verifies both
against pinned SHA-256s, then seeds and ingests the demo workspace: a vendor register plus
ten NDAs, one of them a scanned image-only PDF. `make seed` re-runs seeding idempotently
and waits for ingestion.

Host-specific notes (all variables are documented in [`.env.example`](.env.example)):

- Ports bind to `127.0.0.1`. If one is taken, override it: `WEB_PORT=3300 docker compose up --build --wait`.
- If the Docker daemon runs with `iptables: false`, builds have no network; set
  `DOCKER_BUILD_NETWORK=host`.
- If containers cannot reach the internet, run `make fetch-artifacts` on the host and
  start with `ARTIFACT_CACHE_SOURCE=./.cache ARTIFACT_AUTO_DOWNLOAD=false`.

## Documents and ingestion (Phase 1)

- **Upload**
  - `POST /api/v1/workspaces/{id}/documents` (multipart field `files`). Only owners and
    reviewers can upload; viewers are read-only.
  - The body is streamed with per-file (50 MB), per-request (200 MB), file-count, header
    and field limits.
  - Content is sniffed: the extension must match, and Office zips are checked for bombs,
    encryption, macros and traversal paths.
  - Filenames are sanitised for display; storage keys are server-generated.
  - Rate limiting reserves quota atomically per request and per file attempt.
  - Identical bytes in the same workspace return the existing document, so SHA-256 is
    idempotent per workspace.
- **Ingestion job**
  - Parsing runs in a separate process with a timeout and a memory limit.
  - **PDF:** the PyMuPDF text layer with a box for every character. Pages with almost no
    text are rasterised (a pixel cap is enforced before rendering) and OCR'd with
    Tesseract; word boxes are mapped back to page points.
  - **DOCX:** body paragraphs, headings, lists and tables, with styles.
  - **XLSX/CSV:** a header row is detected even below title rows. Each sheet is stored as
    structured rows, and each row becomes one searchable chunk.
  - **Chunking** splits on headings, numbered clauses and paragraphs, targeting 200–400
    tokens (counted with the embedding tokenizer) with up to 40 tokens of overlap.
  - **Sentences are never split.** A single sentence longer than 400 tokens becomes one
    oversized chunk. If a chunk exceeds the encoder's 512-token context, it is embedded
    in bounded token windows aligned to word starts, and the windows are averaged back
    into one vector. The chunk text itself is never cut or truncated.
  - Every chunk satisfies `text == canonical_text[char_start:char_end]` and carries
    line-level boxes (PDF).
  - Results are written in one transaction that deletes stale derived rows first and
    re-locks the document. Retries never duplicate data, and a deleted document is
    never resurrected.
- **Storage**
  - `vector(384)` with an HNSW index (cosine distance).
  - A generated `tsvector` with a GIN index.
- **Viewer**
  - PDF via react-pdf, using pdf.js's legacy build so current stable browsers are
    supported. It has a text layer, search, and highlight overlays computed from
    stored boxes.
  - Clicking a chunk scrolls to it and highlights it; `?chunk=<id>` deep links work.
  - Spreadsheets render as tables with the selected row highlighted.
  - DOCX shows its extracted body with the chunk's exact characters marked.
  - The Entities panel says plainly that extraction arrives with the Phase 4 review
    pipeline.
- **Document type** (`contract`/`policy`/`other`) is a keyword heuristic on the
  opening text, not a trained classifier.

## Search (Phase 2)

- **API:** `POST /api/v1/workspaces/{id}/search` with
  `{query, mode: hybrid|vector|keyword, top_k ≤ 50, document_ids?, doc_types?, rerank?}`.
  Any workspace member can search, viewers included; non-members get 404, as do
  document filters naming documents outside the workspace. `GET
  /api/v1/search/capabilities` reports the encoder and whether reranking is available.
- **Retrieval** (`packages/retrieval`):
  - Vector: pgvector HNSW cosine (`ORDER BY embedding <=> q`) with `hnsw.ef_search`
    and `hnsw.iterative_scan = strict_order`, so filtered queries still return k rows
    in exact distance order.
  - Keyword: Postgres full-text search ranked by `ts_rank_cd`. The query's `english`
    lexemes are OR-ed by default (`SEARCH_KEYWORD_MODE=websearch` requires all terms
    and supports quotes and `-`).
  - Both run **in parallel on separate database sessions**, then fuse with weighted
    reciprocal rank fusion, `score = Σ w_m / (k + rank_m)`. Ties break
    deterministically: best single-method rank, then more methods agreeing, then chunk
    id.
  - Every result keeps its per-method rank and native score, the fused score, and the
    optional rerank score.
  - Every query is scoped to the workspace and to READY documents, with optional
    document and type filters. Every value is a bound parameter. Queries are treated
    as data.
- **Results:**
  - Each result is a **stored chunk**: its text plus exact document, page, character,
    bounding-box and sheet/row references. Nothing is a generated snippet.
  - The Search tab links each result to `…/documents/{id}?chunk={chunk_id}`. That
    opens the PDF highlight overlay, the selected spreadsheet row, or the marked DOCX
    text.
- **Query embeddings:**
  - The API calls the model service. The model service runs the same pinned,
    SHA-256-verified `bge-small-en-v1.5` code path as ingestion: passages without a
    prefix, queries with `EMBEDDING_QUERY_INSTRUCTION`, and queries longer than the
    encoder context rejected rather than truncated.
  - Calls have bounded timeouts and retries, and responses are validated (shape,
    dimension, unit norm, finite values).
  - A model-service outage returns 503 with a clear message. Keyword mode keeps
    working.
  - Documents whose stored encoder differs from the query encoder are excluded from
    vector ranking and listed in the response.
- **Reranking** (`RERANK_ENABLED=true`): `cross-encoder/ms-marco-MiniLM-L6-v2` at a
  pinned revision with a verified SHA-256. If it is enabled but missing or unverifiable,
  the model service fails to start. A rerank request to a server without it returns
  409; it never falls back silently.
- **Defaults** (`SEARCH_*` in `.env.example`) were frozen from the DEV split of the
  retrieval evaluation below.

## Retrieval evaluation (Phase 2)

These numbers come from the executed `make eval-retrieval`, which runs `uv run python -m
evals.retrieval.run all`, then `uv run python -m evals.retrieval.sweep`. The JSON
results, plots and provenance are in `evals/results/`; the full method and caveats are
in [`evals/README.md`](evals/README.md).

- **Task:** ContractNLI document-scoped evidence retrieval. The query is the hypothesis,
  the scope is one contract, and the gold set is the annotated evidence spans.
- **Data:** official splits. Parameters were selected only on DEV and frozen (hash
  `6cb1378f…`, 2026-10-21T23:42:44Z) before TEST was first run (23:53:32Z). The final
  full `make eval-retrieval` repeated the same frozen TEST measurement unchanged, to
  verify reproducibility; the metrics were identical. TEST has 1188 pairs, 123
  contracts and 17 unique queries; NotMentioned pairs are excluded.
- **Hit rule:** a chunk covers ≥ 50 % of a gold span.

| TEST | recall@1 | recall@3 | MRR@10 |
|---|---:|---:|---:|
| random ordering (expected) | 0.148 | 0.433 | 0.387 |
| keyword (FTS) | 0.416 | 0.789 | 0.658 |
| vector (pgvector) | 0.455 | 0.811 | 0.692 |
| hybrid (RRF) | 0.432 | 0.816 | 0.679 |
| hybrid + rerank | 0.549 | 0.860 | 0.763 |

- **Hybrid vs single methods:** hybrid significantly beats keyword. It does **not**
  significantly beat vector (recall@3 +0.005, 95 % CI [−0.014, +0.021]), and it is
  lower at recall@1/MRR.
- **Reranking** is the significant gain (MRR +0.084), at p50 ≈ 0.38 s of CPU per query,
  so it is opt-in.
- **Tuning:** DEV tuning gave no TEST improvement over the untuned k=60 fusion. The
  app's `SEARCH_*` defaults are the frozen DEV values (k=5, keyword weight 1.5), and
  `SEARCH_RRF_K=60 SEARCH_KEYWORD_WEIGHT=1.0` restores standard RRF.
- **Short documents:** contracts are short (median 7 chunks), so recall@10 is
  near-trivial; read @1/@3 and MRR.
- **HNSW:** at the production index settings, ANN recall@10 against exact neighbours is
  0.990 at `ef_search=100` (p50 0.47 ms on a 4k-vector corpus).
- **Scope:** this is an isolated text-retrieval eval. It does not cover PDF/OCR
  extraction or report quality.

## Reviews (Phase 4)

A review checks a workspace's documents against a rule pack (**Rule packs** tab:
built-in `nda_standard` and `vendor_confidentiality_review`, YAML import/export, editor).
**Reviews → Start review** queues a run; the worker executes the agent graph:

```mermaid
flowchart LR
  P[planner] --> E[extraction<br/>entities + register links]
  E --> R[retrieval] --> C[rule check<br/>trained model, LLM below tau<br/>or for custom rules]
  C --> V[citation verifier<br/>exact text + semantic support]
  V --> X[cross-document<br/>reasoning] --> V2[citation verifier] --> H{human review<br/>(optional pause)} --> W[report writer]
```

- **Evidence first.** Every finding cites stored text at exact offsets (with PDF line
  boxes). The verifier keeps a finding only if each quote is still exactly at its offsets
  *and* the quotes support the claim (the trained model re-checks model decisions on the
  cited spans alone; an LLM support check covers LLM decisions); otherwise it is
  `needs_review`. `not_mentioned` cannot be quoted, so it is presented as needing review.
  The report states only verified or human-decided findings; recommended actions come from
  the rule pack, not from a model.
- **Cascade.** Model-supported rules use the trained model with the DEV-selected tau 0.3
  (full coverage on DEV/TEST, so escalations are rare); custom rules use the LLM's
  structured verdict, grounded on the span ids it was shown.
- **Untrusted documents.** Document text is delimited data; the LLM has no tools; ids it
  was not shown are rejected; extracted dates/terms are re-read from verbatim quotes.
- **Budget.** Each provider attempt is reserved against the run's token and USD caps
  before it is sent (provable bound, rounded up, serialised per run); unknown outcomes
  stay charged; mock usage is labelled as an estimate.
- **Durability.** Runs hold a fenced lease; checkpoints and memoized steps let a paused,
  failed or interrupted run resume without repeating or re-billing finished work.
- **Live view and report.** The run page streams agent events (SSE with replay) and fills
  a rules × subjects grid; the report has filters, a per-subject view, citations that open
  the highlighted source, audited override decisions, and Markdown/JSON export.

Providers are chosen with `LLM_PROVIDER` (`mock` default, `anthropic`, `openai`,
`ollama`; see `.env.example`). The mock is offline and labelled everywhere. `make
smoke-phase4` runs the synthetic ten-vendor flagship through a running stack.

## MCP server (Phase 5)

Any MCP client can search documents, check rules, run reviews and read reports with
citations. The setup guide, with ready-to-paste Claude Desktop configurations for a local
Docker stack and for a Mac reaching this stack on a Linux host over SSH, is in
[`apps/mcp-server/README.md`](apps/mcp-server/README.md).

1. Sign in, open **Settings** (header), and create an API token. The defaults are read
   only and expire after 90 days. Opt in to `review` (spends LLM budget) or `upload`
   (stdio only) if you need them. Copy the token: it is shown once.
2. Add the server to your client using the guide above. Over stdio, the token goes in the
   client's environment for the server process. Over HTTP, send
   `Authorization: Bearer <token>`, and never put it in the URL.
3. `make smoke-phase5` exercises HTTP and stdio (`docker compose exec`) against a running
   stack with the mock LLM.

## Results: rule-check model (Phase 3)

Official ContractNLI TEST split, all 2,091 pairs, data version `693b452e…`. Every choice
was made on DEV only. Source: `ml/eval/results/` (evaluated 2026-10-25 at `49550f2`;
provenance in each JSON):

| System | Input | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Majority (global) | none | 0.463 | 0.211 |
| Majority per hypothesis | none | 0.659 | 0.617 |
| TF-IDF + logistic regression (C=30 balanced) | BGE top-8 segments | 0.764 | 0.724 |
| Claude Haiku 5.5 zero-shot (effort low) | BGE top-8 segments | 0.739 | 0.703 |
| Trained DeBERTa-v3-base, label head on BGE top-8 | BGE top-8 segments | 0.815 | 0.769 |
| **Trained DeBERTa-v3-base, full pipeline (served)** | all segments | **0.853** | **0.814** |

- Model minus TF-IDF: +0.089 macro-F1 (document-bootstrap 95% CI 0.067–0.111). The
  input scopes differ, so the top-8 row (+0.045, CI 0.020–0.070) is the like-for-like
  comparison.
- Cascade: the DEV rule selected tau 0.3. Coverage is 1.0 on DEV and TEST, so no pair is
  escalated to Haiku and the cascade equals the model. It brings no accuracy gain over the
  model alone. It is also not faster: on the Kaggle CPU bench the model takes 10.0 s per
  pair (p50 of the original dynamic int8 file, since replaced; the latency `cascade.json`
  uses) or 13.4 s (fp32 p50, the served variant), while Haiku averaged 1.16 s per TEST
  request.
- Quantization: the original dynamic int8 export collapsed (TEST macro-F1 0.063) and is
  not used. The repaired weight-only file (8-bit matrix weights, 4-bit embedding, float32
  activations; 213 MB vs 741 MB) scores TEST macro-F1 0.816 against 0.814 for the
  PyTorch run, but is about 20 % slower than fp32 on CPU, so fp32 is served
  ([`docs/INT8_QUANTIZATION.md`](docs/INT8_QUANTIZATION.md)).
- Haiku run: $0.371 in total, including the $0.003 Sonnet diagnostic. All 3,128 pairs
  were answered, with 0 errors.
- Unseen hypotheses (4 held out of training): TEST macro-F1 0.324, against 0.783 for the
  model trained with them.

The handoff's claim that `not_mentioned` dominates is wrong for the official TRAIN split
(entailment 3,530 / not_mentioned 2,820 / contradiction 841); loss weights use the actual
counts. Method, full provenance and caveats: [`ml/README.md`](ml/README.md) and
[`docs/HAIKU_EVALUATION_PLAN.md`](docs/HAIKU_EVALUATION_PLAN.md).

## Development

```bash
make install           # uv sync --frozen + npm ci
make fetch-artifacts   # pinned model + ContractNLI into ./.cache (needed by tests)
make test-infra-up     # throwaway Postgres/pgvector + Redis on 127.0.0.1:55432/56379
make ci                # lint, typecheck, OpenAPI drift check, pytest, vitest, next build
make up && make seed   # full stack + demo data
make smoke smoke-phase1 smoke-phase2 smoke-phase3 WEB_PORT=3000   # end-to-end smoke tests against the stack
make eval-retrieval    # ContractNLI retrieval eval (needs test-infra-up + fetch-artifacts)
make ml-test ml-smoke  # rule-check model unit tests + CPU end-to-end smoke (tiny random model)
make help              # every target
uv run pre-commit install   # optional: ruff, mypy, eslint, tsc hooks on commit
```

- **Python:** a uv workspace (`pyproject.toml` at the root). PyTorch comes from the
  CPU-only index: no CUDA packages are locked or downloaded. Tooling is ruff, mypy
  `--strict` and pytest. Integration tests use real Postgres, Redis, Tesseract and the
  real model, and they **fail** (not skip) if those are unavailable.
- **Web:** Next.js 16 App Router, TypeScript strict, Tailwind v4, shadcn/ui-style
  components, TanStack Query, react-pdf, eslint, Vitest + Testing Library.
- **API types:** `make openapi` regenerates `apps/web/src/lib/api/openapi.json` and the
  `openapi-typescript` types. CI fails if either is stale.
- **Migrations:** Alembic lives in `packages/core/src/compliance_core/db/migrations`.
  `0001` covers identity and workspaces; `0002` covers documents, pages, blocks, sheets,
  rows, chunks and entities. Both are reversible.

## Security model

- **Passwords:** hashed with Argon2id. Emails are normalised to lowercase, and the
  database enforces it.
- **Sessions:** an HS256 JWT in an `httpOnly`, `SameSite=Lax` cookie, which is `Secure`
  when `COOKIE_SECURE=true` (production requires that). Signing out or back in revokes
  the old token in Redis.
- **CSRF:** every cookie-authenticated state change, uploads included, must send a
  session-bound `X-CSRF-Token`.
- **Browser traffic:** goes only to the web origin. The Next.js route handler proxies
  and streams `/api/v1/*` with a header allow-list and rejects path traversal.
- **Scoping:** workspace and document endpoints are scoped to membership. Non-members
  get 404, and viewers cannot upload, delete or retry.
- **Search:** every SQL statement is scoped to the caller's workspace and READY
  documents, and query text is only ever a bound parameter. Document filters outside
  the workspace return 404. Results are rendered as text, never HTML. Search is rate
  limited per user.
- **Original files:** PDFs are served `inline`, everything else as an `attachment` with
  `CSP: sandbox` and `nosniff`. Text from PDFs is HTML-escaped before it reaches the
  pdf.js text layer.
- **Untrusted documents:** parsed in a resource-limited child process. Document text is
  treated as data and never executed.
- **Logs:** JSON lines with request ids. Ingestion logs ids, counts and timings, never
  document contents.

## Data and licenses

ContractNLI (CC BY 4.0, Koreeda & Manning, 2021) provides the demo NDAs, all from its
test split. The license was checked before download and is recorded, with attribution,
in [`ml/data/LICENSES.md`](ml/data/LICENSES.md). `BAAI/bge-small-en-v1.5` is MIT-licensed;
the optional reranker `cross-encoder/ms-marco-MiniLM-L6-v2` is Apache-2.0. The retrieval
eval uses the official ContractNLI splits unchanged and commits only aggregate results.
Raw data, model weights and generated documents are git-ignored and never committed.
The vendor register's access levels and dates are synthetic; see
[`seed/README.md`](seed/README.md).

## Limitations

- No end-to-end evals yet (Phase 6); see the phase plan in the handoff.
- **MCP (Phase 5):** tested over stdio (including `docker exec`) and HTTP with the mock
  LLM, and interactively from Claude Desktop for Linux 2.7032.0 against the mock preview
  only. It has **not** been run from Claude Desktop on a Mac or over the Mac-to-Linux SSH
  hop the guide describes. In that Desktop check, Desktop itself noticed one wrong mock
  citation: a return/destruction finding cited the survival clause (2.3, page 2) of
  `ready4s_mutual_nda.pdf` instead of its return section (8, page 4). The mock citation
  verifier is a lexical heuristic, so "verified" on a mock run means that heuristic
  accepted the passage, not that the citation is semantically correct against a gold
  answer. This case is carried into Phase 6 as a semantic negative example. API tokens
  are this app's own bearer tokens, not OAuth, so Desktop's remote "connector" flow (which expects OAuth) is
  not supported: use stdio. The HTTP endpoint binds to localhost; do not expose it
  without TLS. The MCP container learns which providers the worker holds keys for from a
  non-secret flag; if that is wrong, the worker fails the run without calling the
  provider. Runs paused for human review must be resolved in the web app.
- **Reviews (Phase 4):** verified end to end with the deterministic mock LLM and with one
  approved real-provider run (Claude Haiku 5.5 plus the served fp32 model) on the
  synthetic ten-vendor flagship only. That run is acceptance evidence, not an accuracy
  benchmark on real contracts; 18 of its 58 findings were left `needs_review`, and its
  cost ($0.006062) is the application ledger's figure from provider-reported usage, not
  an Anthropic invoice. The trained model decided every model-supported task at tau 0.3
  with zero escalations, so the run shows no cascade improvement. Only Anthropic was
  exercised live; the OpenAI and Ollama adapters are tested offline only. The mock LLM and
  mock rule-check backend are lexical stand-ins for offline demos, not legal analysis.
  Report export is Markdown/JSON (PDF export is Phase 6). Documents ingested before Phase 4
  highlight citations at chunk level until re-ingested (no stored line geometry).
  Cross-document rules cover the vendor-register scenario only (contract on file,
  required protections, term vs renewal date).
- **Rule-check model:** served as CPU fp32 (ONNX fp32 p50 13.4 s per pair on the Kaggle
  CPU bench); the validated quantized file is smaller but slower on CPU and is not served.
  The repository default `RULE_CHECK_VARIANT=int8` refuses the fp32-only artifact, so
  serving it needs `RULE_CHECK_VARIANT=fp32`. Rules unseen in training score poorly (macro-F1 0.32). Evaluation is on canonical ContractNLI
  text, not PDF/OCR output; the 17 hypotheses are the only model-supported rules (custom
  rules go to the LLM in Phase 4).
- **Search:**
  - Results are whole stored chunks (~200-400 tokens), not sentence-level answers.
    Term marks in the result list are a display aid (word-prefix matches of the
    query), not the ranking signal.
  - Keyword search uses the `english` text-search configuration: no fuzzy matching or
    synonyms, and a query made only of stop words has no keyword candidates.
  - The reranker is off by default; when enabled it scores at most 512 tokens per
    (query, passage) pair on CPU, which adds noticeable latency.
  - If the encoder pin changes, documents embedded by the old encoder are excluded from
    semantic ranking (still keyword-searchable) and listed until they are re-ingested.
  - Search is rate limited per user (`SEARCH_RATE_LIMIT_PER_MINUTE`).
- **OCR:**
  - Tesseract (English) runs only on pages with almost no text layer. Mixed pages
    (some text plus scanned images) are not OCR'd.
  - Pages too large to rasterise at 100 DPI within the 40-megapixel budget are skipped
    with a visible warning.
  - OCR word boxes are divided evenly across characters, so character-level geometry is
    approximate within a word.
- **DOCX:** headers, footers, footnotes, comments and text boxes are not extracted. DOCX
  files have no pages, so their citations use character offsets only.
- **Formats:** encrypted or password-protected PDFs and Office files, macro-enabled
  files, legacy `.doc`/`.xls`, and images are rejected.
- **Highlights on scans:** overlays on OCR'd pages follow Tesseract's line boxes, so a
  skewed scan can be off by a pixel or two.
- **Rate limits:** login throttling is per account only, there is no per-IP limit, and
  registration is not rate limited. Upload limits are per user.
- **Membership:** members must already have an account, so the add-member error
  reveals to owners whether an email is registered. There is no ownership transfer.
- **Deletion:** if the API crashes between the database commit and the file removal,
  the stored original can be orphaned (the row is already gone).
- **Out of scope:** SSO, billing, multi-tenant cloud deployment, real-time co-editing
  and non-English documents (handoff section 16).
