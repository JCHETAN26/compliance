# Compliance Workspace

A full-stack, multi-agent compliance review workspace: upload contracts and vendor
registers, run a versioned rule pack, and get a report whose findings cite the exact
source passage. The full specification is in [`docs/BUILD_HANDOFF.md`](docs/BUILD_HANDOFF.md).

> **Status: Phase 0 and Phase 1 accepted; Phase 2 (hybrid retrieval + search UI +
> retrieval eval) implemented and awaiting independent verification.**
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
> - **Phase 3** (local pipeline implemented and verified; hosted CI pending; acceptance
>   blocked on external Colab training and a real Anthropic baseline): the rule-check
>   model pipeline (`ml/`):
>   official ContractNLI data and the 17-rule `nda_standard` pack, a DeBERTa-v3
>   evidence + label cross-encoder with a ModernBERT ablation and an unseen-rule run,
>   majority / TF-IDF / Anthropic zero-shot baselines, calibration, cascade analysis,
>   ONNX fp32/int8 export with parity, a micro-batching `/v1/check` and a Colab
>   notebook. **No model has been trained yet** (GPU training runs on Colab; see
>   [`ml/notebooks/README.md`](ml/notebooks/README.md)) and the LLM baseline needs an
>   API key, so the model service runs in honest *untrained fallback* mode (`/v1/check`
>   503, no verdicts) and the web app shows a banner.
> - **Not implemented yet:** agents/reviews/reports and entity extraction (Phase 4), MCP
>   tools (Phase 5), the end-to-end eval suite and dashboard (Phase 6). The services and
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
  mcp[mcp-server<br/>MCP SDK, no tools yet] --> pg
```

| Service | Current behaviour | Healthcheck |
|---|---|---|
| `web` | Auth, workspaces, documents page (drag-and-drop upload with per-file progress, live status), document viewer, search | `/healthz`: Next.js serving **and** API reachable via the proxy upstream |
| `api` | Auth, workspaces, documents (upload, list, metadata, original file, chunks, pages, blocks, sheets/rows, entities status, delete, retry), hybrid search; runs migrations on start | `/api/v1/health/ready`: DB query, Alembic at head, pgvector installed, Redis PING |
| `worker` | `ingest_document` jobs, stale-document sweep, demo seeding on first start | `arq --check` heartbeat; it only heartbeats after the DB is reachable and the pinned model is SHA-256-verified and loaded |
| `model-service` | `/v1/embed` (query/passage embeddings with the pinned ingestion encoder), `/v1/rerank` (only when `RERANK_ENABLED=true`), `/v1/check` (rule-check model from a SHA-256-verified ONNX artifact with dynamic micro-batching; without a verified trained artifact: 503 `rule_check_unavailable`, see [`apps/model-service/README.md`](apps/model-service/README.md)) | `/health`: `embedding=ready` (weights SHA-256-verified and loaded); `rule_check=ready|unavailable` is reported, not required |
| `mcp-server` | MCP over streamable HTTP and stdio; **no tools yet** (Phase 5) | Real MCP handshake + `tools/list`, plus DB probe |
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

## Results: rule-check model (Phase 3, partial)

No trained-model or LLM result exists yet. Measured so far on the full official
ContractNLI TEST split (2,091 pairs, `make baselines`, choices made on DEV only):

| Baseline | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority (global) | 0.463 | 0.211 |
| Majority per hypothesis | 0.659 | 0.617 |
| TF-IDF + logistic regression (BGE top-8 candidates; C=30 balanced, DEV-chosen) | 0.763 | 0.723 |

The handoff's claim that `not_mentioned` dominates is wrong for the official TRAIN split
(entailment 3,530 / not_mentioned 2,820 / contradiction 841); loss weights use the actual
counts. Method, provenance and the remaining external steps: [`ml/README.md`](ml/README.md).

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

- No trained rule-check model yet (Colab training pending), no agents, reviews, MCP tools
  or end-to-end evals; see the phase plan in the handoff.
- **Rule-check model (when trained):** dynamic int8 quantization makes a row's output
  depend slightly on the rows batched with it; evaluation is on canonical ContractNLI
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
