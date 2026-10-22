# Phase 2 independent acceptance checklist

Status: implementation in progress. Phase 2 is not accepted yet.

## Retrieval correctness and access

- Actual local pinned query embeddings, PostgreSQL HNSW and full-text queries;
  semantic/keyword queries execute in parallel using safe separate sessions.
- Workspace membership and all document filters prevent cross-workspace leakage.
- RRF ranks, weights and ties are defined and tested. No fabricated score labels.
- Every result identifies stored source text and exact viewer coordinates/offsets.
- Optional reranking is explicit and fails clearly when enabled but unavailable.
- Shared inference changes preserve Phase 1 ingestion, long-input handling and the
  honest absence of a trained rule-check model. No hidden remote embedding API.
- Search UI handles empty/loading/error states and opens exact source highlights,
  including spreadsheet rows and PDF chunks on later pages.

## Evaluation

- Official ContractNLI splits retained; development selection separated from final
  test measurement. No test-driven tuning or cherry-picked favorable gold set.
- Real embeddings, keyword search, vector search and fusion; meaningful evidence
  recall@k/MRR definitions and no-evidence exclusions/counts documented.
- HNSW sweep uses actual ANN plans and exact-neighbor reference, with warmup and
  latency scope. ANN recall is distinguished from evidence recall.
- Benchmarks use isolated DB/schema/data and cannot drop live/test indexes or data.
- JSON/plots/configs/provenance and exact commands correspond to real runs. Derived
  README numbers match result files; no raw datasets or model weights in Git.
- Hybrid improvement target is assessed honestly, even if it is not achieved.

## Regression and release gate

- Existing Phase 0/1 checks stay green, including automatic seed and source viewer.
- Independent targeted/full static/tests, real API/browser path and Docker checks.
- Disk usage bounded; unrelated caches/images/volumes and global config preserved.
- Full hosted CI passes before Phase 3 can begin.

Later model training, agents, review reports, MCP tools and final eval/dashboard
remain out of scope for this phase and are not complete.

## Independent benchmark input inventory

Read the cached official JSON files directly with Python zipfile/json, without
altering them. Observed:

| Official split | Documents | Hypotheses | Pairs with annotated evidence |
|---|---:|---:|---:|
| train | 423 | 17 | 4371 |
| dev | 61 | 17 | 614 |
| test | 123 | 17 | 1188 |

Document ID sets are pairwise disjoint. These are input counts, not retrieval/model
performance results. The evaluator should disclose any subset and no-evidence
exclusions and must not tune its selected configuration on the test split.

## Storage/inference design checkpoint

Root has only ~1.6 GB free. Adding ingestion's CPU ML dependency set to model-service
while retaining it in worker can duplicate a ~1.3 GB dependency layer even if both
read one weights volume. Before building, verify whether heavy dependency layers
are truly shared (different SERVICE_PACKAGE build args can invalidate otherwise
similar layers). A common dependency/runtime image for Python roles, or moving
inference ownership behind a shared client, may avoid disk duplication. Preserve
all Phase 1 behaviors and accurate untrained rule-check metadata. Do not begin a
build that will exhaust disk; provide a concrete remedy if needed.

## Pending model-client contract review

Early `model_client.py` decodes success JSON into unchecked dictionaries, converts
vectors/scores directly, and treats every non-retryable HTTP error (including 500)
as a request error. Ensure malformed JSON, missing fields, wrong vector count/dim,
non-finite values and unexpected rerank IDs/scores fail clearly as upstream/service
errors, not uncaught KeyError/JSONDecodeError/SQL errors or fabricated rankings.
Use validated typed response contracts and meaningful tests. Separate service 5xx
failures from invalid user queries.

Search must also enforce encoder-space consistency: document vectors from one model
revision cannot silently be compared with queries from another 384-d encoder or
revision. Verify returned model/revision against stored embedding metadata and give
an actionable re-ingestion/config error (or explicit safe filtering) for mismatches.
Default settings matching is not sufficient after an inference-service upgrade.

## Independent first unit-contract pass

`timeout 120 env -u PYTHONPATH .venv/bin/python -m pytest packages/retrieval/tests/test_fusion_config.py packages/core/tests/test_model_client.py -q`

**36 passed in 0.05s** (exit 0). Covers weighted RRF/config edge cases plus typed
upstream response validation, retry/error behavior, invalid vectors and rerank IDs.
This is a unit subset (mock transport for client failure tests), not actual-model,
SQL retrieval, browser, benchmark, or Phase 2 acceptance.

## Independent retrieval/model/API integration subset

`timeout 300 env -u PYTHONPATH .venv/bin/python -m pytest packages/retrieval/tests packages/core/tests/test_model_client.py apps/api/tests/test_search.py apps/model-service/tests -q`

**95 passed in 43.78s** (exit 0). Log:
`/tmp/compliance-supervision/phase2-retrieval-verify.log`.
Tests exercise real pinned encoder/reranker via the in-process HTTP service,
Postgres/pgvector and FTS, source-reference integrity, workspace/document/type filters,
viewer permissions, HNSW query planning, encoder mismatch reporting, model outages,
optional reranking and API rate limiting. Mock transport is used only in deliberate
client failure/contract tests. Live Docker HTTP, browser and real dataset benchmark
checks are still pending, as is the final full regression gate.

## Benchmark interpretability checkpoint

For per-document ContractNLI retrieval, many NDAs may have fewer chunks than the
requested k; recall@8 can then be near-trivial. Report useful smaller cutoffs
(e.g. @1/@3 alongside @8), chunk-count distribution and/or fraction with <=k chunks,
and state that scope explicitly. A high @8 is not evidence of workspace-global
search quality. Global ANN-neighbor recall remains a separate efficiency check.
Keep model/metric choices honest; do not alter held-out gold to improve scores.

## Independent live-stack smoke

`timeout 180 env -u PYTHONPATH .venv/bin/python scripts/smoke_phase2.py --web http://127.0.0.1:3300 --model-service http://127.0.0.1:8010`

**Exit 0**, all five groups passed: live pinned encoder with rule-check still 501;
hybrid vectors+FTS returning exact stored chunks; structured spreadsheet references
and deep links; vector/keyword modes and filters/validation; workspace isolation and
viewer search permission. Log: `/tmp/compliance-supervision/phase2-live-smoke.log`.

`docker compose ps` independently confirmed all seven healthy. `/metadata` identifies
BGE full revision and dimension 384; reranking is disabled by default on the main
stack (tested enabled using actual models in the integration subset). Browser
interaction and benchmark result verification still pending.

## Evaluation metric/data review checkpoints

- If partial span coverage is pooled across overlapping chunks, use a union of
  ranges rather than double-counting characters. A clearly defined per-chunk hit
  criterion is also valid. Define the gold-hit threshold and MRR relevance rule
  explicitly; test boundary/overlap cases.
- ContractNLI has only 17 unique hypothesis texts repeated across document pairs.
  Report unique query count and measured query/pair counts; cached/precomputed query
  embeddings must be disclosed in latency scope rather than counted as live inference.
- Preserve any exact duplicate official document texts (the loader observed one in
  test). If overflow workspaces are needed for truthful SHA dedup constraints, verify
  each query uses that document's actual workspace and still evaluates every selected
  official document. ANN corpus scope/reference must remain explicit.


## Pending high-priority benchmark DB safety issue (resolve before acceptance)

Early `evals/retrieval/corpus.py` accepts any database name starting with
`compliance_eval` and interpolates it inside quoted DROP/CREATE DATABASE SQL without
escaping embedded quotes. A crafted env URL can escape the identifier despite that
prefix check. Validate a strict safe identifier (including length) or quote it
correctly, and apply the validation inside the destructive function itself (not an
optimizable assert that assumes a caller already checked).

Also refuse to drop an existing database lacking an evaluator ownership marker.
A matching prefix alone is not proof that the database was created by this tool.
Fingerprint mismatch must never authorize destruction of an unmanaged DB. Add
focused malicious-name and unmanaged-sentinel DB tests, proving refusal happens
before destructive SQL. Handle concurrent eval runs with an explicit lock or safe
separate per-run DB identity; do not let one sweep reset another's database/indexes.
Default port55432 is isolated here, but the EVAL_DATABASE_URL override must be safe.

## Pending canonical-text segmentation concern

`data.line_blocks` currently makes every nonblank newline-delimited line a paragraph.
ContractNLI text contains wrapped PDF lines; those line breaks are not necessarily
paragraph or sentence boundaries. The production chunker treats each block as a
sentence-segmentation boundary, so a long sentence wrapped across many short lines
can be cut mid-sentence at a size/paragraph break despite the Phase 1 invariant.

Verify with a wrapped single sentence exceeding the chunk target (no blank lines,
period only at the end). Group wrapped lines into real paragraphs/clauses/headings
without changing canonical offsets, or otherwise preserve sentence boundaries in
this benchmark input. Do not use gold evidence selections to tune segmentation.
Add a focused test and ensure the corpus fingerprint invalidates stale chunks.
This is a source-review concern, not a claim that completed metrics are wrong; no
TEST scores have been inspected by the supervisor.

## Independent production search browser verification

Copied Claude's finalized browser script unchanged to a stable supervisor path:

`timeout 450 env -u PYTHONPATH /tmp/cw-e2e/.venv/bin/python -I /tmp/compliance-supervision/phase2-ui-verify.py http://127.0.0.1:3300`

**Exit 0.** Passed keyboard-opened PDF result on page2 with measured source-box
alignment/viewport placement, back navigation restoring query/results, exact selected
spreadsheet row, keyword/type filters and empty state, 390px mobile layout without
horizontal overflow. Controlled outage stopped only this project's model-service:
hybrid showed a clear 503 state while keyword mode still answered. The script
restored the service in finally, waited for health, then checked mobile results.
Two expected 503 console messages occurred only during that deliberate outage;
no other console or page errors were accepted.

Independently confirmed all seven containers healthy afterward. Visually inspected
`/tmp/cw-e2e/p2-results.png`. Log:
`/tmp/compliance-supervision/phase2-ui-verify.log`.
Benchmark correctness/results, final full gates and hosted CI remain pending.

## Independent evaluator-safety and metric tests

`timeout 180 env -u PYTHONPATH .venv/bin/python -m pytest evals/retrieval/tests -q`

**19 passed in 1.22s** (exit 0). Includes malicious/invalid database identifiers
refused before a connection, unmanaged database with sentinel preserved, owned DB
lifecycle, concurrent-run lock refusal/release, metric checks against exact random
permutation expectations, wrapped oversized-sentence preservation and official DEV
gold-offset integrity. The earlier benchmark-guard and wrapped-line concerns are
resolved in code plus these tests. No held-out TEST performance result accepted yet.

Corrected DEV run is active with ~4 CPU threads. Its log is piped through buffered
`grep -v`, so the log remains empty until flush; process CPU/time confirmed progress.
Do not mistake that buffering for successful completion or invent intermediate scores.

## Independent full application gate

`make ci > /tmp/compliance-supervision/phase2-full-checks.log 2>&1`: **exit 0**.
Ruff clean, 130 files formatted, strict mypy clean across 110 source files, ESLint,
TypeScript, current OpenAPI/generated types, **304 pytest passed in 95.86s**, **83
Vitest passed in 14 files**, and production Next.js build passed. The pytest duration
includes concurrent corpus-embedding load; it is not a performance benchmark.
`git diff --check` passed. Official retrieval results/plots and ANN sweep remain in
progress, and hosted CI remains pending before Phase 2 acceptance.

## Pending HNSW sweep grid issue — confirmed against PostgreSQL

The Cartesian grid includes `m=32, ef_construction=32`. A transaction-scoped temporary
table probe on the isolated test server returned:

`ERROR: ef_construction must be greater than or equal to 2 * m`

The temporary transaction/session rolled back; no persistent data changed. Skip and
report invalid combinations (or declare a valid grid) before a long sweep, and add a
small grid-validation test. Do not fail halfway through after six valid builds.

The sampled DEV-span query lengths were independently checked using the real pinned
tokenizer and seeded sample: all 283 fit, maximum 207 tokens including the default
query instruction, zero exceed 510. No extra query filtering is needed for this
fixed sample; disclose query construction as implemented.

Final CI consistency check: local `make typecheck` includes `evals`, while the current
GitHub Actions mypy command still omits it. Keep the intended static-check scope
consistent before the Phase 2 push (this is configuration, no extra product tests).

## Benchmark reporting clarification

The current runner tunes instruction/keyword mode/RRF weights jointly for hybrid,
then evaluates the component methods with that shared selected configuration. State
that clearly as a component comparison. If DEV's best standalone vector instruction
or keyword mode differs, disclose the difference (or add independently DEV-selected
standalone results) so readers do not infer those are separately optimized baselines.
Do not select or revise any parameters from TEST outcomes.

## Independent held-out base-metric recomputation

Re-ran the existing evaluator against all 1,188 official TEST pairs, using the frozen
config and cached real corpus, without overwriting canonical result files. Keyword,
vector and hybrid metric dictionaries match `retrieval_test.json` exactly. Log:
`/tmp/compliance-supervision/phase2-base-metric-recheck.log` (exit 0).
Hybrid recall@3 = .8160 vs vector .8109; the document-bootstrap interval includes
zero, so the report must not claim robust superiority over vector. Reranking's
reported recall@3 is .8603; this independent recomputation did not repeat reranking.

Final pending reminders: hosted mypy still omitted `evals` when independently read;
resolve before push. Add/verify the requested small HNSW grid-validity check. The
final benchmark command is running again; preserve its actual provenance and report
which completed run supplies canonical results. Earlier TEST values are not tuning
feedback; keep the frozen configuration unchanged.

Documentation precision: the final full-command reproducibility rerun repeats the
same frozen TEST measurement; use wording such as "parameters selected only on DEV;
TEST rerun unchanged to verify reproducibility" instead of the literal "measured
once" where applicable. Preserve evidence of the original pre-TEST freeze (hash
6cb1378fdec77f44581e184ff11c5072d7feead6fbb63019191b488c5f491516,
2026-10-21T23:42:44 UTC) even if the all-command rewrites its timestamp. No TEST-based
selection or parameter revision is authorized.

## Complete benchmark command and figure verification

Claude's final `make eval-retrieval` completed **EXIT 0** in
`/tmp/cw-eval/make-eval-final.log`. Independently read canonical DEV/TEST/sweep JSON:
all record actual HEAD ec6c254b2fed4b33645aa060142ccdb4856efb2e, dirty=true,
source fingerprint ac6c9f740b87a2ad576d510bed5424e0cf3e0a3c679b62de7fad8abd0d183852.
TEST config hash remains unchanged and scores reproduce the earlier result.
HNSW sweep contains 66 measurements over 4,047 train+dev vectors and 300 queries;
invalid m=32/ef_construction=32 is explicitly skipped. Independently viewed the
new sweep PNG: shared y axis and comparable x axes are now correct. Numerical ANN
results can vary with graph construction/timing; do not present reruns as exact
latency reproducibility. The main evidence metrics reproduced exactly.

Final doc read: `evals/README.md` says "tuned values are kept as defaults". Clarify
that this means frozen benchmark configuration only: the app retains k=60/equal
weights by default. Do not imply the application was switched to tuned defaults.

Correction to supervisor's preceding defaults note: that note relied on an earlier
snapshot. A fresh source read now shows Settings.search_rrf_k=5 and
search_keyword_weight=1.5; RetrievalConfig defaults match. The implementation did
adopt the DEV-selected config before the final benchmark (covered by its source
fingerprint). Keep these actual DEV-selected defaults; do not revert them based on
TEST. Document the app and benchmark consistently with the current code. The
k60/equal configuration is now a reference baseline, not the app default. The
supervisor ledger's earlier wording about defaults is superseded by this correction.

## Final local release gate

After Claude's final CI/documentation/grid-test fixes, Codex independently ran:
- Evaluator subset: **24 passed in 2.14s**, including actual pgvector invalid/boundary
  settings in rolled-back temporary tables and canonical sweep/grid agreement.
- Final rebuilt-image Phase 2 smoke: **exit 0, all five groups passed**.
- Final complete `make ci`: **exit 0**, 309 pytest in 68.60s, 83 Vitest, all
  static/schema checks and production web build passed.

Logs: `/tmp/compliance-supervision/phase2-final-eval-tests.log`,
`phase2-final-image-smoke.log`, `phase2-release-checks.log`.
Hosted mypy includes evals. Documentation records pre-TEST frozen hash/time and the
unchanged reproducibility rerun. DEV-selected app defaults are verified against
actual Settings/RetrievalConfig and were present before the canonical result run.
No remaining local acceptance finding; hosted CI is required before Phase 3.
