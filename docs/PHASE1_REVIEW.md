# Phase 1 independent acceptance checklist

Status: implementation in progress. No Phase 1 acceptance is claimed yet.

## Document ingestion

- Real Postgres migrations for document/page/chunk/structured-row storage, vector
  and full-text indexes; migration/model consistency and downgrade verified.
- Authenticated multi-file upload, bounded streamed body, file signatures and
  container expansion checks, sanitized storage paths, membership/role enforcement.
- SHA256 idempotence is scoped to a workspace; jobs retry safely without duplicate
  chunks/rows and deletion cannot be undone by an outstanding worker job.
- Each supported parser is exercised: PDF layout, genuine image-only PDF OCR,
  DOCX, XLSX, CSV; malformed input produces a stored, visible failure.
- Chunk text equals the canonical source slice at stored offsets. PDF text and OCR
  coordinates fit the source page; multi-line highlighting is visually checked.
- Spreadsheet headers can follow blank/title rows; data remains structured and
  searchable per row. Dates/empty cells have a documented representation.
- Embeddings are real 384-dimensional outputs from the configured local model;
  model identity/revision and runtime failures are explicit. No hash-vector fallback.

## UI and API

- Upload progress and processing/ready/error states reflect real asynchronous jobs.
- Original-file, metadata, chunks, rows and delete access are workspace scoped.
- PDF renders with text search and overlays; clicking a chunk locates its page and
  coordinates. Spreadsheet rows and DOCX text are readable in the viewer.
- Empty/loading/error/keyboard behavior is exercised. Unimplemented agent extraction
  does not show invented entities.

## Reproducible demo and operations

- Dataset license/provenance recorded before sample usage; official test-document
  provenance preserved. Raw data and weights excluded from Git.
- Demo register plus matching NDAs include at least one actual scanned PDF.
- Seed and clean startup repeat without resetting existing user credentials/data.
- CPU-only ML dependencies; disk usage monitored; unrelated artifacts untouched.
- Existing Phase 0 tests remain green; complete local checks and hosted CI pass.
- Browser upload -> ready -> viewer -> exact highlight is independently rerun.

Full trained-model performance, agents, reviews, MCP tools and final evaluations
remain later phases. Phase 1 does not establish any of those results.

## Pending observation during implementation

At the first dataset download, `git check-ignore
.cache/datasets/contractnli/contract-nli.zip` returned nonzero: the new artifact cache
was not yet in `.gitignore`. Resolve this before staging anything. No Phase 1
artifacts have been staged or committed by the supervisor.

## Pending code-review issue: OCR raster cap

Early `ocr.py` computes a pixel-based maximum DPI, then clamps with
`max(72, min(dpi, max_dpi))`. For a sufficiently large valid PDF page, even 72 DPI
exceeds MAX_OCR_PIXELS; the raster call can therefore exceed the stated memory cap.
Reject excessive page dimensions/pixel counts before rasterization, or permit an
appropriately smaller scale with a clear limitation. Add a focused adversarial-page
test that proves get_pixmap is not called with an oversized raster. Recheck final
implementation: this observation was made while code was still being written.

## Pending specification mismatch: oversized sentences

Early `chunking.py` explicitly split oversized sentences at clause punctuation or
whitespace and marked `split_within_sentence`. This contradicts the handoff's
"Never split mid-sentence" requirement. Supervisor interrupted Claude to correct
this before acceptance. Preserve the full source sentence as a chunk even when it
exceeds the approximate token target, and handle bounded encoder context explicitly
(e.g. internal embedding windows with aggregation, without splitting source chunks
or silently truncating). Add meaningful tests for both source preservation and
long-input embedding behavior. Other valid solutions must preserve the requirement.

## Pending code-review issue: multipart header buffering

Early `_StreamingStager.on_header_field/on_header_value` append to bytes without a
header-size cap. A single multipart header can therefore grow up to the 200 MB
request limit, with repeated concatenation and buffering in the API process. Cap
per-header and aggregate part-header bytes while streaming, return a consistent 4xx,
and ensure staging files are cleaned up. Add a focused incremental-header test.
This is a review observation on in-progress code; recheck before acceptance.

## Pending code-review issue: stale-job status transitions

Early worker `_fail` updates any existing document of the same generation, including
one that a concurrent delivery already made READY. Stale sweep updates also compare
only ID/generation after reading a candidate outside the mutation transaction; a
job can finish in between and be needlessly requeued. Compare current status (and
staleness where relevant) atomically, or use an attempt/lease token. Test an old
failure after success and a completion racing the stale sweep. Preserve READY data
and do not reset/requeue a completed generation.

Also verify unexpected embedding/runtime exceptions do not leave PROCESSING stuck
without a visible error until multiple long sweep windows elapse. Expected provider,
model and parser failures need explicit failure/retry behavior and tests.

## Build storage checkpoint before Docker builds

Root is down to about 2.6 GB after CPU dependencies; `.venv` is 1.6 GB. The external
SSD is exFAT (not a drop-in Docker layer filesystem). Do not start a parallel full
Compose build that might exhaust root. Check peak storage first and report a blocker
if needed. Possible project-only remedies: serial image builds with scoped pruning
of newly created private BuildKit records between images; remove only confirmed
obsolete compliance app images after replacing/stopping their containers; clean
regenerable project build outputs. Keep unrelated cache/images/volumes intact.
Do not delete shared uv cache just because files are linked to this venv. The large
Torch CPU library has two hardlinks (cache and venv), so moving only the venv would
not free its root-disk blocks. External storage is not authorized for writes yet.

## Independent first parser/chunker pass

Command run by Codex:

`env -u PYTHONPATH .venv/bin/python -m pytest packages/ingestion/tests/test_parsers.py packages/ingestion/tests/test_sentences_chunking.py -q`

**32 passed in 1.43s.** Includes real Tesseract OCR on an image-only PDF, PDF exact
source-box checks, rotation, DOCX, XLSX/CSV, and oversized sentence preservation.
This is a partial check while implementation continues, not full Phase 1 acceptance.
Artifact cache ignore was rechecked: `.cache/datasets/contractnli/contract-nli.zip`
is now ignored by Git. The initial cache-ignore observation is resolved.

## Independent OCR / embedding / upload-storage pass

The first sandboxed subset run stalled in the subprocess-runner test after 15 tests.
Codex stopped that verifier process only (not Claude or the app). A bounded rerun
of the runner test outside the execution sandbox passed in 2.63s. This suggests an interaction with the sandbox; the exact cause
was not established. Container and hosted CI verification are still required.

Command then run outside the sandbox:

`timeout 180 env -u PYTHONPATH .venv/bin/python -m pytest packages/ingestion/tests/test_ocr_embeddings_runner.py packages/core/tests/test_files_uploads_storage.py -q`

**51 passed in 7.23s.** Covers real pinned 384-d embeddings, short-input equivalence,
long-input windowing, oversized OCR page rejection before rasterization, OCR error
classification, extraction subprocess success/rejection/timeout, streaming upload
limits/header caps, content validation and storage confinement. Together with the
prior 32 parser/chunker tests this is 83 independently passing focused tests.
Full API/worker/container/browser/CI acceptance remains pending.

## Pending code-review issue: upload-rate accounting

Early upload route calls `ensure_allowed` before processing, then `record` only for
accepted files afterward. Fully rejected uploads consume no quota, and simultaneous
requests can all pass the initial check. The handoff requires uploads to be rate
limited. Reserve/record attempts atomically before expensive streaming/validation,
with an explicit request- or file-attempt unit, and test repeated rejected uploads
and concurrent requests against the cap. Do not rely on accepted-file totals alone.

## Independent full Python pass before final UI/build changes

Command run by Codex outside the execution sandbox against isolated project test DB:

`timeout 300 env -u PYTHONPATH .venv/bin/python -m pytest -q`

**194 passed in 28.82s** (exit 0). Log:
`/tmp/compliance-supervision/phase1-python-verify.log`.
This covers Phase 0 regression tests plus Phase 1 parser, embedding, API and worker
integration tests, including real Postgres/pgvector, Redis and OCR/model execution.
Subsequent fixes still require the relevant checks to be rerun. The final UI,
container build, automatic seeding, browser viewer behavior and hosted CI remain
pending; no full Phase 1 acceptance is claimed.

## Pending viewer review: deep-link focus after PDF load

`PdfViewer`'s highlight-scroll effect runs from focusToken/highlight/scrollTo, while
pageRefs are populated only after async PDF `onLoadSuccess` updates numPages. A fresh
`?chunk=...` deep link can run the effect before pageRefs exist, then never retry
when the PDF finishes loading. Test opening a fresh viewer directly to a chunk on a
later page (not just clicking after loading). Ensure focus waits for page geometry
and document readiness, and verify actual viewport placement plus overlay geometry.
Also check the scroll offset's coordinate reference (page.offsetTop vs scroll root).
This is a code-review concern to verify, not yet a reproduced browser failure.

## Independent first frontend pass

`npm run lint && npm run typecheck && npm test` in `apps/web`: **exit 0**,
**73 tests in 13 files passed** (Vitest duration 2.81s). Includes new upload,
DOCX/source highlighting, spreadsheet and geometry/escaping helpers. PDF browser
rendering and direct-to-chunk scroll are still pending browser verification.

## Independent production browser + clean-start acceptance

- `timeout 300 env -u PYTHONPATH .venv/bin/python scripts/smoke_phase1.py --web http://127.0.0.1:3300`:
  passed: 11 seeded READY documents, real OCR, structured register, upload/duplicate/
  invalid-type checks, worker ingestion, exact chunks/boxes, file access, read-only
  viewer role, deletion and original-file removal.
- Copied Claude's final browser script unchanged to a stable supervisor path, then ran
  `timeout 450 env -u PYTHONPATH /tmp/cw-e2e/.venv/bin/python -I /tmp/compliance-supervision/phase1-ui-verify.py http://127.0.0.1:3300`:
  **exit 0**, `console/page errors: none`, `Phase 1 UI check passed`.
  Covers production PDF rendering, OCR-page overlays with measured geometry, fresh
  last-page deep link/viewport placement, text-layer search, spreadsheet row highlight,
  DOCX mark, honest empty entities state, and UI upload -> worker READY -> highlight.
- Visually inspected `/tmp/cw-e2e/p1-scanned-highlight.png`: line-by-line overlays
  align with the selected source text in the scanned original.
- Started a separate project `compliance-phase1-acceptance` using fresh DB/storage
  volumes and host ports 3430/8430/8431/8432, pre-fetched verified artifact bind mount
  (required by this host's container-network restriction), `--no-build --wait`.
  All seven services became healthy without running `make seed`.
- Ran the same Phase 1 smoke script through port 3430: **passed**, all 11 demo docs
  automatically ingested. Logs in `/tmp/compliance-supervision/phase1-clean-*.log`.
- Teardown is scoped to this temporary project with `down -v`; main app data is retained.

Browser issue found by implementer: modern PDF.js required a built-in missing from
Chrome 139. Matching legacy main/worker builds were configured; the final independent
production walkthrough uses that same Chrome and passed. The deep-link concern is
resolved by readiness-aware focus plus the browser test above.

Final post-fix static/test gate and hosted CI still pending.

## Independent final local gate

`make ci > /tmp/compliance-supervision/phase1-final-checks.log 2>&1`: **exit 0**.
Ruff clean, 104 files formatted, strict mypy clean across 85 source files, ESLint,
TypeScript, OpenAPI/type drift checks, **197 pytest passed in 29.39s**, **73 Vitest
passed in 13 files**, and production Next.js build passed.

The new atomic upload-reservation tests are included. The earlier header-buffer,
stale-job transition, and OCR raster-cap concerns are addressed by implementation
and focused tests in this gate.

Remaining small commit hygiene issue: `git diff --check` reports
`apps/web/src/lib/api/queries.ts:127: new blank line at EOF`. Claude should remove
that extra blank line; no additional test is needed for that whitespace-only edit.

Local acceptance is otherwise satisfied. Hosted CI has not yet run on Phase 1.
