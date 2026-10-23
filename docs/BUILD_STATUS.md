# Build status

Implementer log for `docs/BUILD_HANDOFF.md`. The independent verification ledger is
kept by the supervisor in `docs/SUPERVISION.md` and `docs/PHASE0_REVIEW.md` …
`PHASE3_REVIEW.md`; this file does not restate it as fact. Phase 3 work is uncommitted;
the supervisor reviews, commits and pushes it.

| Phase | Status |
|---|---|
| 0 — Skeleton | **Accepted.** Hosted GitHub Actions run 39678264159 passed all jobs for `e6b709e`. |
| 1 — Ingestion + viewer | **Accepted.** Hosted GitHub Actions run 39687462736 passed all jobs for `ec6c254` (auto download, seeding, ingestion, smoke). |
| 2 — Hybrid retrieval, search UI, retrieval eval | **Accepted.** Hosted run 39699615208 passed all jobs for `c61eab2`. |
| 3 — Rule-check model | **Local pipeline implemented and verified; hosted CI pending; acceptance BLOCKED** on external work: full Colab GPU training, the real Anthropic baseline, the resulting cascade/calibration/quantization results on the full splits, and verification of the imported artifact. No trained-model or LLM result exists. See "Phase 3" below. |
| 4–6 | Not started (Phase 4 gated on Phase 3 acceptance). |

## Phase 3 — rule-check model

### Delivered (code, configs, tests, docs)

- **Rule pack** `rule_packs/nda_standard.yaml`: the 17 ContractNLI hypotheses verbatim as
  `model_supported` per-document rules (test checks every text against the dataset).
- **Data** (`ml/data`): official splits unchanged (train 423/7,191, dev 61/1,037, test
  123/2,091 pairs; nothing excluded); ingestion segmentation; overlap-defined gold
  mapping (many-to-many, counted); BGE candidate ranking; deterministic serialization;
  `data_report.json`. **Handoff correction:** TRAIN counts are entailment 3,530 /
  not_mentioned 2,820 / contradiction 841 (not_mentioned does not dominate); loss weights
  come from the actual TRAIN pairs used.
- **Model/training** (`ml/train`): one backbone + evidence head + 3-way label head;
  DeBERTa-v3-base primary, ModernBERT-base ablation, unseen-4-rules run, CPU smoke; all
  backbone/tokenizer files SHA-256 pinned; DEV-only selection (epoch, k, T, thresholds,
  backbone, tau); fp16 skip-aware scheduler; atomic, recoverable, identity-checked
  resume; production refuses subset data/CPU; TEST loaded only after selection is frozen.
- **Evaluation** (`ml/eval`): label/evidence (end-to-end denominators, official-span
  projection, invalid ids as false positives)/calibration/cascade/per-hypothesis/
  bootstrap metrics through one scoring path for every system; fixed-top-8 model variant
  to separate head quality from candidate scope; plots.
- **Baselines** (`ml/baselines`): majority (global, per hypothesis), TF-IDF + logistic
  regression (TRAIN fit, DEV choice), Anthropic zero-shot (structured output, per-prompt
  cache identity, every attempt's usage/latency, lower-bound cost when usage is unknown,
  CLI bounds, dry-run estimate).
- **Export/serving**: ONNX fp32 + dynamic int8 with parity and batch-sensitivity checks,
  parameter-hashed model versions, manifest refresh; `apps/model-service` `/v1/check` with
  a verified artifact, micro-batching, bounded admission (tokenization and rows), size
  and blank-input validation, honest `untrained_fallback` (503) otherwise.
- **UI/API**: `GET /api/v1/models/status` (no paths/reasons exposed) and a site-wide
  banner (`ModelStatusBanner`) with concise availability text.
- **Colab**: `ml/notebooks/colab_train.ipynb` + runbook `ml/notebooks/README.md`; results
  bundle pack/verified import (`ml/colab/bundle.py`).
- **Make/CI**: `ml-data`, `baselines`, `baseline-llm`, `train`, `train-smoke`, `eval-model`,
  `export-model`, `bench-model`, `eval-onnx`, `ml-test`, `ml-smoke`, `install-model`,
  `import-colab-bundle`, `smoke-phase3`; CI runs mypy on `ml`, the ML unit tests, the ML
  CPU smoke and `scripts/smoke_phase3.py`. `uv.lock` and every `pyproject.toml` are
  unchanged (the ML tools are a pinned overlay, `ml/requirements-tools.txt`).

### Measured results (real runs, full official data)

`make baselines`, `data_version 620358b51ef96fac…`, TEST 2,091 pairs, choices on DEV only:

| System | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority (global) | 0.463 | 0.211 |
| Majority per hypothesis | 0.659 | 0.617 |
| TF-IDF + logistic regression (C=30, balanced) | 0.763 | 0.723 (0.722885) |

- Re-running `make baselines` reproduced all six prediction files byte for byte (SHA-256).
- The TF-IDF label C grid was extended to {30, 100} after an earlier run picked its upper
  edge C=10 on DEV (DEV-driven). TEST results have not changed any choice.
- BGE top-8 candidate coverage (pairs with any gold in the top 8): TRAIN .866, DEV .868,
  TEST .837.
- **No trained model, LLM, cascade, calibration, latency or quantization numbers exist
  yet.** Smoke runs use a tiny random backbone; their numbers are meaningless and are never
  written to `ml/eval/results`.

### Commands run and outcomes (with failures)

- **Full data prep** `python -m ml.data.prepare` (CPU, ~25 min; TRAIN ranking 920 s): written
  with the old gzip writer, which embedded random temp names in gzip headers (supervisor
  finding). Fixed the writer and re-serialized with `ml.data.reserialize --verify-ranking dev`:
  rows identical, documents/pairs equal a fresh rebuild, 1,037/1,037 DEV rankings
  re-embedded identically (max score diff 0.0) -> `620358b51ef96fac`; old dir kept as
  `.cache/ml/data/full.pre-reserialize-f93f46478098c871`.
- **Embedding throughput** profiled at ~37 segments/s on this CPU, so the Colab path uses
  the same encoder on GPU (`--device cuda`, recorded).
- **DeBERTa-v3 tokenizer**: conversion needs `sentencepiece`; the converted `tokenizer.json`
  differed from the original SentencePiece ids on 1.25 % of TRAIN spans, 0.36 % after NFKC
  (remaining: byte-fallback private-use bullets). NFKC is now part of the shared encoder;
  every run records the measured rate on TRAIN.
- **ONNX**: the TorchScript exporter needs `onnx`; fp32 parity ~5e-7. Dynamic int8 is
  batch-dependent (activation scales per input tensor): measured and reported as
  `batch_sensitivity`; service tests use a stated int8 tolerance and rebuild the label
  input from the served spans.
- **Service admission**: a 10-request concurrency test hit the new 8-request tokenization
  bound (503) as designed; the test raises the bound explicitly.
- `hf_hub_download` failed after the production `Embedder` set `HF_HUB_OFFLINE=1`
  process-wide; backbone files now download through the project's SHA-256-checked HTTPS
  downloader.
- **Final local gates** (run by the supervisor after the storage recovery, on the final
  sources):
  - `ruff check` + `ruff format --check`: clean (188 files); strict `mypy`: clean (165 files);
    OpenAPI schema and generated TypeScript types current.
  - application `pytest`: 334 passed; ML unit tests: 25 passed; Vitest: 87 passed (banner 4).
  - full ML smoke (`make ml-smoke`, tool overlay): **14 passed in 113 s**, including the fixed
    reserved-name/ancestry test, fp32/int8 service equivalence, interrupted resume ==
    uninterrupted bit-identically, tokenizer cache recovery, bundle round trip and
    malicious/late-invalid bundle refusal, and manifest refresh. Tiny random backbone: no
    performance meaning.
  - These were run as separate static/test steps, not as one literal host `make ci`: the
    host production web build could not be written to the nearly full disk. Instead the
    **unmodified web source was built for production** (`npm ci` + `next build`) inside
    `node:22-alpine` with a RAM-only build directory, and a small runtime image made from
    its outputs is serving the stack (see the storage section). The normal
    `apps/web/Dockerfile` is unchanged; hosted CI builds it with normal disk.
- **Docker**: model-service and API images rebuilt from the final sources (dependency layer
  cached; no lock change); the web image rebuilt as described below; all seven services
  healthy; live `smoke_phase0..3` passed, with the model service in honest untrained fallback.
  Container path: a tiny smoke artifact installed with `make install-model` was loaded by the
  container (uid 10001) through the relative `current` symlink with the explicit smoke
  opt-in and served concurrent, micro-batched `/v1/check` requests (also verified
  independently by the supervisor: 8 concurrent requests, shared batches); it was then
  removed. Browser: smoke and fallback banners verified, no console errors.

### Colab failure on `4cd76d8` (fp16 parameters) and fix

The first production Colab run stopped at its first `scaler.unscale_` with `ValueError:
Attempting to unscale FP16 gradients`; it wrote `run.json` only (no training log, no checkpoint).
**Diagnosis:** transformers 5.20 resolves a missing `dtype` to `"auto"`, i.e. the checkpoint's
dtype (`from_pretrained`) or the config's `dtype` (`from_config`). DeBERTa-v3-base's
`pytorch_model.bin` is fp16 (371 MB for 184 M parameters), so the backbone had fp16 parameters
and GradScaler (correctly) refused them. **Fix:** `build_backbone` and `build_architecture` pass
`dtype=torch.float32`, and `require_float32` stops training (before the optimizer exists) and
export if any floating-point parameter or buffer is not float32. autocast still provides fp16
activations on CUDA; the scaler stays enabled and `allow_fp16` is not used. Export and the CPU
benchmark rebuild the architecture from config, which is now float32 even for a config declaring
float16, so the fp32 ONNX export and its PyTorch parity reference are float32; saved checkpoints
are float32 (~0.74 GB for DeBERTa-v3-base, as in the Drive plan).
**Tests** (`ml/tests/test_fp32_master_weights.py`, CPU): a tiny DeBERTa-v2 checkpoint saved as an
fp16 `.bin` with an fp16 config loads as fp16 with transformers' default (the hazard) and as
float32 through all three project paths, keeping the checkpoint's values; a real, enabled CPU
GradScaler reproduces the exact `ValueError` with fp16 parameters through the production
`optimizer_update`, and with the fixed backbone in the real two-head model under bf16 autocast it
applies the update and advances the schedule. A mutation run (explicit dtype removed) makes both
fix tests fail. ML unit tests 29 passed; ruff/format clean; strict mypy clean (166 files).
**Pending:** GPU verification on Colab (no GPU here); the tool-overlay ML smoke was not re-run for
this change. Recovery steps: `ml/notebooks/README.md` ("Recovering from the fp16 failure").

### First `36eb2eb` Colab run: no resumable checkpoint visible

**Confirmed (user's live inventory):** the `deberta_v3_base` run (Drive mode) reached step ~7,700
in its third epoch in ~83 min on a T4 and was interrupted; the rerun stopped in `recover_last`
with `CheckpointError` (only an incomplete `.last-staging`). Drive was 100 % full (147 MB free of
15 GB); under `checkpoints/` only `best/model.safetensors` (740,088,360 bytes) exists, with no
`COMPLETE`, `state.json` or `trainer.pt`; no training process was running. The visible files do
not permit an exact resume. **Not established:** how the newer checkpoints disappeared
(hypotheses: Trash filling the quota, mount reconciliation after the interrupt). A complete
checkpoint in Drive Trash is possible, not guaranteed; the user has not reported on Trash yet. No
new training is authorized until that decision.

**Changed (notebook, runbook, notebook tests only; training code, configs and the training-code
fingerprint `a74ec481c62e6b5f8d24ce97aded422e61df6febf10f015298ab6ba4e8fdc64d` unchanged):**
`RUN_LOCATION = "local"` by default (runs and checkpoints on the runtime's disk, resumable within
the same runtime; Drive read for bundle/data/baselines and written only by space-checked, verified
backups into new folders; nothing on Drive deleted); `stream()`/`run()` show stdout+stderr live,
log to `RUNS/logs/`, raise on failure, and stop and reap the child on an interrupt or a failing
output relay; the source cell can update an existing checkout from a verified bundle (clean
implementation required). ML unit tests 41 passed (16 notebook tests). The supervisor's operational
`preserve_failed_run.py` (read-only, verified local copy of the failed run) lives outside the
repository. **Pending:** behaviour in a live Colab runtime (display, stop button, Drive free space
through the mount) is unverified.

**Known CI failure, not addressed here:** hosted run 39866518635 failed because concurrent API
startups seeded the Demo workspace twice (duplicate). Follow-up required.

### Storage incident and how it was resolved

The normal web image build needs ~1.1 GB transient (`npm ci` layer) plus build output, more
than this host had free. Two attempts were made with a watchdog that cancels below 200/250 MB
free; both were cancelled (the second on my mistaken assumption that the first attempt's
`npm ci` layer would be reused). BuildKit still holds three **non-reclaimable** records from
those cancelled builds (history entries removed; `prune --filter id=` frees 0 B):
`44m8mc9ug44d6oa18r196a6tk` (1.083 GB), `xzri4ny622zrjgwhhcvkzgvwp` (841 MB),
`re7z3ek7e2zwio1zyt2q4chkt` (61 MB), which is why root has only **~150 MB free**. No shared
Docker restart, broad prune or external-storage write was done; all pre-existing BuildKit
records (including the 48 older ones) are intact. Log: `/tmp/cw-disk/phase3-cleanup.log`.

Recovery (supervisor-run; recipe and scripts authored by me in `/tmp/compliance-supervision`):
`web-ram-build.sh` builds the unmodified `apps/web` source in `node:22-alpine` (pinned
digest) on a 3 GB tmpfs with a 4 GB memory cap, copies only `.next/standalone`,
`.next/static` and `public` to a RAM context, and `Dockerfile.web-runtime` (the existing
runner stage with those three `COPY` sources) turns them into `compliance/web:dev`; no npm
layer touches Docker's disk-backed store. Host web dependencies for the checks are
temporary: `apps/web/node_modules` is a git-ignored symlink to
`/dev/shm/compliance-web-hostdeps` (glibc `npm ci`, regenerable, removed with
`web-host-deps.sh remove`).

**Constraint still open:** importing the Colab results bundle needs free disk equal to the
bundle size plus ~300 MB (the importer checks and refuses otherwise). With ~150 MB free,
space must be made first, e.g. once BuildKit releases the three records above (prune them by
exact id) or by another approved route.

### External steps (exact; nothing faked meanwhile)

1. Supervisor commits/pushes Phase 3 to the private `JCHETAN26/compliance` repository, waits
   for hosted CI, and generates the git bundle plus a notebook copy with `EXPECTED_COMMIT`
   filled in (configuration only; no logic change).
2. User: Colab training per `ml/notebooks/README.md` (upload the bundle to
   `MyDrive/compliance-runs/`, T4 runtime, local run mode; three runs, evaluation, export, CPU
   latency, int8/fp32 TEST accuracy, results bundle downloaded or backed up to Drive).
3. User: Anthropic key as the Colab secret `ANTHROPIC_API_KEY` (or typed into their own shell
   with `read -rs`); dry run first, then DEV and TEST.
4. Free disk headroom, then `make import-colab-bundle BUNDLE=…`, restart model-service,
   `make smoke-phase3` (expects `model_kind=production`); supervisor reviews
   `ml/eval/results/*` and the handoff criterion (model TEST macro-F1 vs TF-IDF 0.722885).

### Phase 3 supervisor validation commands

```bash
cd /home/grv22/projects/compliance
make test-infra-up && make fetch-artifacts
env -u PYTHONPATH uv run ruff check . && env -u PYTHONPATH uv run ruff format --check .
env -u PYTHONPATH uv run mypy packages apps conftest.py scripts evals ml
env -u PYTHONPATH uv run pytest -q && env -u PYTHONPATH uv run python -m pytest -q ml/tests
make ml-smoke                              # ~2-3 min, tool overlay, SMOKE outputs in temp dirs
make openapi-check                         # needs apps/web/node_modules (npm ci)
env -u PYTHONPATH uv run python scripts/smoke_phase3.py --web http://127.0.0.1:3300 --model-service http://127.0.0.1:8010
make baselines                             # reproduces the six prediction files byte for byte
```

## Phase 2 scope delivered (accepted)

**Retrieval** (`packages/retrieval`, shared service layer)
- `sql.py`:
  - vector candidates: pgvector HNSW cosine `ORDER BY embedding <=> q LIMIT k`,
    per-transaction `hnsw.ef_search`, `hnsw.iterative_scan = strict_order` and
    `statement_timeout`, all via `set_config(..., true)` with bound values;
  - keyword candidates: Postgres FTS ranked by `ts_rank_cd`. In `or` mode the query's
    `english` lexemes are OR-ed; `websearch` mode uses `websearch_to_tsquery`. A query
    of only stop words yields no keyword candidates;
  - every statement is scoped to workspace + READY documents (+ optional document
    ids / doc types), with only bound parameters;
  - hydration re-checks the scope; `Explain` is an EXPLAIN construct for tests and
    evals.
- `fusion.py`: weighted RRF `Σ w_m/(k + rank_m)`.
  - Ranks come from native scores, with ties broken by chunk id.
  - A method contributes once per chunk; zero-weight methods are ignored.
  - Fused ties break deterministically by best rank, then method count, then chunk id.
    Scores are rounded to 1e-12 so that mathematically equal sums tie.
  - Per-method rank/score provenance is kept.
- `service.py`:
  - `resolve_scope`: every workspace role may search, viewers included. Foreign or
    unknown document ids return 404.
  - `run_search`: vector (embed → HNSW) and keyword run **concurrently in separate
    sessions** (`asyncio.TaskGroup`). It then fuses or ranks a single method, applies
    the optional rerank (stable sort), hydrates stored spans and returns timings.
  - Model-service errors map to 409 (query too long, rerank disabled) or 503
    (unavailable or invalid response).
  - **Encoder-space consistency:** vector search only matches documents whose stored
    `embedding_model` equals the query encoder's `model@revision[:12]`. Mismatched
    READY documents are excluded and listed in `excluded_from_vector`, and the UI
    shows a re-ingest warning.
- `config.py`: top-k, candidate-k, ef_search, RRF k, weights, keyword mode, rerank
  pool (≥ top-k, ≤ 100) and statement timeout are all validated and configurable via
  `SEARCH_*`. The defaults were **frozen from DEV**: RRF k=5, keyword weight 1.5,
  vector weight 1, `or` mode, BGE query instruction.

**Model service** (query embeddings + optional rerank inside the existing 7 services)
- `POST /v1/embed` (`query` or `passage`) runs the same `Embedder` as ingestion: the
  pinned revision, SHA-256-verified weights, and windowing for passages.
  - Queries are `EMBEDDING_QUERY_INSTRUCTION + query`, never windowed. Overlong
    queries get 422 `query_too_long`.
- `POST /v1/rerank` uses `cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0, pinned
  revision + SHA-256). It loads only when `RERANK_ENABLED=true`.
  - If enabled but unavailable, startup fails.
  - When disabled, `/v1/rerank` returns 409 `rerank_disabled`.
- Single-thread executors per model and a bounded pending queue: overload sheds with
  503.
- `/v1/check` is still an honest 501. `/health` reports `embedding=ready`,
  `rerank=ready|disabled`, `rule_check=not_implemented`.
- **Docker:** the model service shares the worker's dependency layer
  (`DEPS_PACKAGE=compliance-worker`). `SERVICE_PACKAGE` is declared only after the
  deps layer, so CPU PyTorch is stored once.
  - Verified: the model-service build reported the 1.3 GB `uv sync` layer as
    `CACHED`, and free disk was unchanged.
- **Client** (`compliance_core.model_client`):
  - httpx with bounded timeouts, retrying transport errors and 502/503/504 with
    backoff;
  - 4xx error bodies map to request errors and other 5xx to unavailable;
  - typed pydantic contracts reject malformed JSON, missing fields, wrong count or
    dimension, non-unit or non-finite vectors, and unexpected, missing or duplicate
    rerank ids (`ModelServiceProtocolError`).

**API**
- `POST /api/v1/workspaces/{id}/search` is authenticated and sends `Cache-Control:
  no-store`. It applies a per-user rate limit (`SEARCH_RATE_LIMIT_PER_MINUTE`, atomic
  INCR). `GET /api/v1/search/capabilities` reports the encoder and rerank availability.
  `SearchUnavailableError` maps to 503.
- The OpenAPI schema and TS types are regenerated (`make openapi-check` is clean).

**Web**
- A Search tab (`/workspaces/{id}/search`):
  - URL-synced query/mode/top-k/document/type/rerank, so the page is shareable and
    back navigation restores it;
  - capabilities-driven rerank toggle;
  - result cards with filename, page/sheet-row/section location, stored text with
    term marks rendered as React text (never HTML), provenance badges (semantic
    rank·cosine, keyword rank·ts_rank_cd, RRF, rerank) and an expand toggle;
  - loading, empty, error (503/429/404) and encoder-mismatch states, and an
    `aria-live` status;
  - keyboard operable and responsive.
- Each result links to `…/documents/{id}?chunk={chunk_id}`, which opens the existing
  viewers' exact highlight.

**Retrieval evaluation** (`evals/retrieval`, `make eval-retrieval`; details in
`evals/README.md`, results in "Phase 2 retrieval evaluation results" below).

## Supervisor review items addressed (`docs/PHASE2_REVIEW.md`)

- **Model-client contracts:**
  - typed response validation (count, dimension, norm, finite values, rerank ids);
  - 5xx is unavailable, distinct from invalid user input;
  - tested with `MockTransport` (19 tests).
- **Encoder-space consistency:** a SQL filter on the stored encoder label, plus an
  explicit `excluded_from_vector` report and a UI warning. Tested by relabelling a
  document.
- **Storage:** a shared deps layer, so torch is stored once (see above). Disk was
  managed per the procedure below.
- **Benchmark interpretability:**
  - recall@1/3/5/10, hit@k and MRR@10, with an exact random-ordering baseline;
  - the chunks-per-document distribution and the share of pairs whose document has
    ≤ k chunks;
  - an explicit statement that this is document-scoped retrieval, not
    workspace-global.
- **Metric definitions:**
  - a per-chunk hit criterion (one chunk covers ≥ 50 % of a gold span's
    characters), with boundary tests and an exhaustive-permutation test of the random
    baseline;
  - 17 unique queries and pair counts reported;
  - memoised query embeddings disclosed in the latency scope;
  - the duplicate test document is kept in an overflow workspace and searched in its
    own workspace (`documents_in_overflow_workspaces: 1`).
- **Eval DB safety:**
  - strict name regex `compliance_eval(_[a-z0-9_]{1,30})?`, validated inside
    `recreate_database`/`stored_fingerprint` (not via assert);
  - an ownership marker written immediately after creation;
  - an existing database without the marker is never dropped;
  - a per-database advisory lock (`EvalLock`) for every run;
  - integration tests prove refusal happens before any connection or DROP for
    malicious names and unmanaged databases.
  - One-time action: the first, aborted run had created `compliance_eval` before the
    marker existed (0 committed documents). I dropped it manually; the drop is
    logged in `/tmp/cw-eval/actions.log`.
- **Segmentation:**
  - ContractNLI keeps wrapped PDF lines. Consecutive lines now form one paragraph
    block until a line ends with `.;:!?` (optionally followed by closing quotes or
    brackets). A heading-like line is its own block only when no paragraph is open.
  - Offsets are unchanged, and the rule never looks at gold.
  - Tested with a wrapped single sentence longer than `max_tokens`. The corpus
    fingerprint covers `data.py`, so stale chunks were rebuilt.
  - After the fix: 0 unreachable gold spans in DEV and TEST.

## Phase 2 retrieval evaluation results

The final run of record is `make eval-retrieval` (exit 0). It ran `uv run python -m
evals.retrieval.run all`, then `uv run python -m evals.retrieval.sweep`.

- **Provenance** (all three files): git HEAD `ec6c254b2fed4b33645aa060142ccdb4856efb2e`,
  dirty (48 paths, the uncommitted Phase 2 work); code fingerprint
  `ac6c9f740b87a2ad576d510bed5424e0cf3e0a3c679b62de7fad8abd0d183852`; seed 20261021.
- **Frozen config:** hash `6cb1378fdec77f44581e184ff11c5072d7feead6fbb63019191b488c5f491516`.
  - Parameters were selected only on DEV. The original freeze, before any TEST run,
    was at 2026-10-21T23:42:44Z (copy: `/tmp/cw-eval/frozen_config.run2.json`).
  - The first TEST run was at 23:53:32Z.
  - The final full `make eval-retrieval` re-ran DEV, which reselected the same
    config with the same hash, so the canonical file's timestamp reads 00:03:45Z.
  - It then repeated the TEST measurement unchanged to verify reproducibility. The
    TEST metrics were identical; only latencies differ.
  - No parameter was revised from TEST.
- **App defaults:** the application's `SEARCH_*` defaults are the frozen DEV values
  (k=5, keyword weight 1.5). They are included in the fingerprinted code, and
  `SEARCH_RRF_K=60 SEARCH_KEYWORD_WEIGHT=1.0` restores standard RRF.
- **Full method and honest reading:** `evals/README.md`.

| TEST (1188 pairs / 123 docs / 17 unique queries) | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| random (expected) | 0.148 | 0.433 | 0.671 | 0.919 | 0.387 |
| keyword | 0.416 | 0.789 | 0.918 | 0.989 | 0.658 |
| vector | 0.455 | 0.811 | 0.934 | 0.991 | 0.692 |
| hybrid (DEV-tuned: k=5, kw 1.5) | 0.432 | 0.816 | 0.935 | 0.992 | 0.679 |
| hybrid untuned (k=60, 1:1) | 0.452 | 0.816 | 0.937 | 0.991 | 0.695 |
| hybrid + rerank | 0.549 | 0.860 | 0.955 | 0.996 | 0.763 |

**Comparisons** (document-clustered paired bootstrap, 95 % CI):

| Comparison | recall@3 | MRR |
|---|---|---|
| hybrid − keyword | +0.028 [0.014, 0.042] | +0.021 [0.012, 0.029] |
| hybrid − vector | +0.005 [−0.014, 0.021] | −0.013 [−0.032, 0.007] |
| rerank − hybrid | +0.044 [0.021, 0.069] | +0.084 [0.065, 0.103] |

**Assessment:**
- **Handoff criterion** ("hybrid beats at least one single method"): met against
  keyword only, not significantly against vector.
- **DEV tuning** gained nothing on TEST relative to untuned RRF.
- **Interpretability:** median 7 chunks per document, and 72 % of TEST pairs have
  ≤ 10 chunks, so recall@10 is near-trivial. There are 0 unreachable gold spans.
- **Latency** (p50/p95 total, in-process, memoised query embeddings): hybrid
  4.4/6.0 ms; rerank adds p50 379 ms / p95 1259 ms on CPU.

**HNSW sweep:**
- **Setup:** 4047 train+dev vectors and 300 queries. Index use was verified per
  setting with EXPLAIN, with 20 warm-ups each. The exact SQL scan agrees 100 % with
  numpy.
- **Production index (m=16, ef_construction=64):** ANN recall@10 is 0.990 at
  ef_search=100, p50 0.47 ms.
- **Skipped:** m=32/ef_construction=32 is invalid in pgvector, so it is skipped and
  recorded.

**Artifacts:**
- `evals/results/retrieval_dev.json`, `retrieval_test.json`, `hnsw_sweep.json`;
- `retrieval_dev_grid.png`, `retrieval_test_methods.png`, `hnsw_sweep.png`;
- `evals/retrieval/frozen_config.json`.

## Phase 2 commands run (with failures and fixes)

- **Unit tests:**
  - `pytest packages/retrieval/tests packages/core/tests/test_model_client.py`
    passed 36.
  - A tie test exposed float-sum ties (1/3+1/3 vs 1/2+1/6), so fused scores are now
    rounded in the sort key.
- **`pytest apps/model-service/tests`:** the first run failed because FastAPI rejects
  `Model | JSONResponse` return types. I switched to a typed `ServiceError` handler;
  19 passed with the real models.
- **`pytest packages/retrieval/tests`:** 40 passed (real Postgres, pinned
  encoder/reranker in-process).
- **`pytest apps/api/tests/test_search.py`:** 17 passed. I fixed two test mistakes
  first: the rows field is `row_number`, and there is no `/content` endpoint.
- **`make openapi`:** regenerated the schema and types; the drift test had failed
  before regeneration.
- **Docker:** builds ran serially with `DOCKER_BUILD_NETWORK=host`, the stack with
  `WEB_PORT=3300 ARTIFACT_CACHE_SOURCE=./.cache ARTIFACT_AUTO_DOWNLOAD=false`.
  - The images were built twice: after the API work, and again after the `uv.lock`
    change (matplotlib, dev group).
  - Both times the model-service build reported the 1.3 GB deps layer `CACHED`.
- **Smokes:** `smoke_phase0.py`, `smoke_phase1.py` and the new `smoke_phase2.py`
  (`--web http://127.0.0.1:3300 --model-service http://127.0.0.1:8010`) all passed
  on the final images. Log: `/tmp/cw-eval/final-smokes.log`.
- **Browser walkthrough** `/tmp/cw-e2e/phase2_ui_check.py`: passed. Log
  `/tmp/cw-e2e/phase2-ui.log`, screenshots `/tmp/cw-e2e/p2-*.png`.
  - Run 1 failed on an ambiguous text locator in my script.
  - Run 2 failed because it started while model-service was still restarting from
    run 1's outage step. The 503 error UI rendered as designed.
  - Run 3 passed, as did the final run on the final images.
  - The screenshots showed the Mode select overlapping its neighbour, so the
    selects are now full width.
- **Eval:**
  - The first DEV run was aborted to apply the supervisor's segmentation and DB-safety
    findings. Stopping it with `pkill -f` killed its own shell (exit 144, the known
    pattern), but the eval did stop.
  - The sweep failed twice before succeeding: pgvector `Vector` objects needed
    `.to_numpy()`, and pgvector rejects `ef_construction < 2m`, so that combination
    is now skipped and recorded.
  - The final `make eval-retrieval` exited 0. Log: `/tmp/cw-eval/make-eval-final.log`.
- **`make ci`:** exit 0 (lint, typecheck incl. `evals`, openapi-check, **304 pytest**,
  **83 vitest**, next build). Log: `/tmp/cw-eval/make-ci-1.log`. Only docs changed
  afterwards.
- **Disk:** only project-owned images and build records were removed, each after
  its containers were replaced; IDs and container logs were saved first.
  - Records: `/tmp/cw-disk/phase1-cleanup.log`, `p2-*` and `p2-r2-*` ID lists, logs
    in `/tmp/cw-disk/p2-logs*/`.
  - The 48 older other-project build records are preserved.
  - Free space: 1.6 GB at the end; the minimum was 409 MB during the final web build.

**Final review follow-ups** (tests, CI and docs only; no numerical rerun; the code
fingerprint still computes to `ac6c9f74…`):
- **Hosted mypy:** now `uv run mypy packages apps conftest.py scripts evals`, the same
  scope as `make typecheck`.
- **New test** `evals/retrieval/tests/test_sweep_grid.py`. It checks:
  - the grid has exactly one invalid combination, (32, 32), and contains the
    production index;
  - the committed `hnsw_sweep.json` covers exactly the 66 valid
    (m, ef_construction, ef_search) rows, all HNSW-planned, with (32, 32) listed as
    skipped;
  - `ef_search` ≥ k;
  - against real pgvector, in a rolled-back temporary table, that
    `ef_construction < 2m` is rejected and the 2m boundary accepted. This proves the
    skip rule matches the engine.
  - `sweep.py` is unchanged.
- **Docs:** replaced "measured once on TEST" with the DEV-only selection, the
  pre-TEST freeze evidence and the unchanged reproducibility rerun. Clarified the
  component comparison and the app defaults.

## Phase 2 known issues and limitations

- **Hybrid vs vector:** hybrid is not significantly better than vector-only on TEST,
  and the DEV-tuned fusion parameters did not beat untuned RRF on TEST (see results).
- **Document-scoped queries:** these use Postgres's exact sort over the document's
  rows; the planner does not choose HNSW at that selectivity. Workspace-wide queries
  over large corpora use the index (iterative scan); that path is covered by the
  EXPLAIN test and the sweep.
- **The eval is text-only:** canonical ContractNLI text, with no PDF/OCR noise. Short
  contracts make @10 near-trivial.
- **Reranker:**
  - The reranker (MS MARCO-trained, Apache-2.0 weights) is off by default; its CPU
    cost is p50 ≈ 0.38 s per query.
  - MS MARCO's own data terms are non-commercial. The model card licenses the weights
    as Apache-2.0, and the dataset is not used here; worth a legal check before
    commercial use.
- **Search UI:** at desktop width the Mode select truncates "Hybrid (semantic +
  keyword)". It is readable and has no overlap, but it is cosmetic and unfixed.
- **Term marks** in results are a display aid (word-prefix matches), not the ranking
  signal.
- **Hosted CI:**
  - Hosted CI has not run Phase 2.
  - The CI artifact cache key includes `config.py`, so the first hosted run downloads
    the reranker (~91 MB) once.
  - The CI stack still runs with reranking disabled. Rerank is exercised with real
    weights in pytest.
- **Disk:** the root disk is ~99 % full (1.6 GB free). Another full image rebuild
  needs the remove-old-images-first procedure above.

## Phase 2 supervisor validation commands

```bash
cd /home/grv22/projects/compliance
set -o pipefail
make test-infra-up                       # 55432 / 56379
make fetch-artifacts                     # verifies encoder, reranker (91 MB), ContractNLI
make ci                                  # lint, typecheck (incl. evals), openapi, pytest, vitest, build
env -u PYTHONPATH uv run pytest -q packages/retrieval/tests apps/api/tests/test_search.py \
  apps/model-service/tests packages/core/tests/test_model_client.py evals/retrieval/tests
env -u PYTHONPATH uv run pytest -q evals/retrieval/tests    # 24 (metrics, DB safety, sweep grid)
make eval-retrieval                      # ~15 min first time (embeds 607 docs), reuses DB later
# Same frozen hash expected: 6cb1378fdec77f44581e184ff11c5072d7feead6fbb63019191b488c5f491516
DOCKER_BUILD_NETWORK=host docker compose build    # serial; see disk procedure above
WEB_PORT=3300 ARTIFACT_CACHE_SOURCE=./.cache ARTIFACT_AUTO_DOWNLOAD=false docker compose up -d --wait
env -u PYTHONPATH uv run python scripts/smoke_phase0.py --web http://127.0.0.1:3300
env -u PYTHONPATH uv run python scripts/smoke_phase1.py --web http://127.0.0.1:3300
env -u PYTHONPATH uv run python scripts/smoke_phase2.py --web http://127.0.0.1:3300 --model-service http://127.0.0.1:8010
env -u PYTHONPATH /tmp/cw-e2e/.venv/bin/python -I /tmp/cw-e2e/phase2_ui_check.py http://127.0.0.1:3300
```

`phase2_ui_check.py` stops and restarts only `compliance-model-service-1` to exercise
the outage path.

## Phase 1 scope delivered (accepted)

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
