# Phase 3 independent acceptance checklist

Status: started with Claude after Phase 2 accepted. No trained-model claim yet.
User choices: full GPU training on Colab; real LLM baseline via Anthropic API.

## Data and inference validity

- Official ContractNLI train/dev/test preserved, 17 mapped hypotheses/rules; licenses
  verified before data use. No raw data or model weights committed.
- Actual TRAIN class counts used for weights (Entailment3530, NotMentioned2820,
  Contradiction841). The handoff's majority-class assumption is corrected explicitly.
- Shared canonical segmentation and source offsets. Gold evidence only supplies
  training/evaluation targets, never model/retrieval inference input.
- All label baselines use the same official TEST pairs and same retrieved input;
  exclusions are explicit and cannot quietly remove NotMentioned pairs.
- DEV-only hyperparameters, temperature, evidence/decision thresholds and model
  selection. TEST cannot choose checkpoints or reported variant. Unseen-hypothesis
  experiment completely excludes those rules from training and calibration choices.

## Model, results and serving

- Real evidence and three-way label prediction, required backbone and documented
  ablation. CPU smoke artifacts clearly distinguished from full trained artifacts.
- Majority, TF-IDF/logreg, real Anthropic zero-shot, own trained model; official
  full label/evidence metrics, calibration, generalization and cascade accounting.
- Real export and quantization parity, accuracy changes, model sizes, measured CPU
  latency for PyTorch/ONNX fp32/int8 on identified hardware and consistent workloads.
- Dynamic batching correctness, bounded queue/cancellation/timeouts, validated model
  contracts, exact cited source mapping, concurrency and failure states.
- Untrained/missing/incompatible artifacts never serve fabricated decisions or
  advertise a trained-model capability. Hashes/configs/seeds/code identity actual.

## Reproducibility and regression

- Colab notebook runs the same versioned scripts on T4, securely accesses private
  repo and returns verifiable artifacts/results without printing or storing tokens.
- `make train`, `eval-model`, `export-model`; bounded CPU smoke in hosted CI.
- App/ingestion/search behavior remains working; shared ML layers and disk limits
  respected. No broad cleanup or external SSD writes without authorization.
- Independent meaningful local checks plus hosted CI before phase acceptance.
- External Colab training and real API credentials may require user action. Prepare
  all runnable code and concrete instructions/artifacts before requesting it. Never
  substitute smoke/mock numbers to pass the full-training or real-provider gate.

Phase 4 agents remain gated on this phase's verified acceptance.

## Colab runtime and artifact checkpoint

The T4 path must support interruption/resume and separate primary, ablation and
unseen-rule runs; do not assume unlimited continuous GPU runtime. Record candidate
sampling/truncation explicitly, preserve train-only sampling and measure any evidence
coverage ceiling. DEV evaluation and calibration must reflect the same inference
pipeline, with no gold-span shortcut. Keep host downloads bounded: full fp32 exports
and their CPU benchmarks can run in Colab, with only the verified int8 deployment
artifact/results returned here if needed. No large local download without a concrete
size plan. Notebook source must not depend on uncommitted files only present on this
host; provide an actual versioned checkout or a reproducible source bundle.

Segmentation review: canonical ingestion spans need not match official ContractNLI
annotation units exactly. Define evidence overlap/relevance and evaluation units
clearly, and test containment/boundary cases. Use TRAIN/DEV only to choose mapping,
retrieval candidates and thresholds. Inspection of TEST input integrity/statistics
must not become feedback for changing this mapping or hyperparameters.

## Plan review

The implementation plan is acceptable for local work: pinned ML tool overlay avoids
invalidating the shared 1.3GB dependency layer; full pretrained/training weights
stay on Colab; CPU smoke is explicitly random/tiny and never production evidence.
No authorization for actual full training/API calls without their required external
configuration is implied by a successful smoke.

Check the planned shared-input rule carefully: majority, TF-IDF, real LLM and own
model evaluate identical official label pairs; input-bearing methods get the same
non-gold top-8 candidates. Report the candidate evidence coverage ceiling rather
than crediting retrieval misses to the learned head alone. Evidence head top-k must
be its own predicted ranking at inference, never gold selection. Held-out-rule
training weights and preprocessing fitting use seen TRAIN rules only.

The handoff also requires a UI banner when no trained artifact is available, not
only metadata/503. Include an honest visible model-status state while preserving
working search in fallback mode. Smoke-only artifact loading needs an explicit
opt-in and must keep metadata/UI marked smoke; smoke output cannot masquerade as
production-trained or satisfy this phase's acceptance.

Artifact size note: dynamic MatMul int8 quantization may leave DeBERTa's large
128k-vocabulary embedding table in fp32. Do not assume the complete deployment
bundle is ~200MB from parameter count alone. Measure actual quantized size and
avoid simultaneously extracting, copying and retaining duplicate archives on this
host. Full fp32 and int8 CPU benchmarks can remain in Colab if storage requires it.

## Early batching source-review findings (implementation still underway)

`MicroBatcher.close()` currently cancels `_collect` while it may be awaiting an
active executor forward. Active batch futures are neither in `_carry` nor `_queue`,
so draining those does not resolve them; `infer()` can remain pending forever.
Track/fail active futures on shutdown and test close during a deliberately blocked
forward (release the test worker afterward so it cannot leak a thread).

Cancellation also decrements `_in_flight` in `infer` immediately, while canceled
chunks and their row lists may remain queued behind a slow forward. Repeated
admit/cancel cycles can then accumulate more queued row memory than max_queue_rows.
Bound actual retained pending/active rows, release capacity when chunks are removed
or otherwise remove canceled chunks promptly, and test repeated cancellations while
the first forward is blocked. An API request timeout alone does not fix retained
queue admission.

`InferenceSpec.temperature` currently allows positive infinity (Pydantic float gt=0
alone is insufficient). Require finite numeric inference config and preserve the
existing good finite-logit validation. Check the batcher's standalone output contract
also enforces exactly three label columns, even though OnnxBackend already does.

Evidence metric denominator: if the learned/baseline heads only see top-8 retrieved
segments, report end-to-end evidence recall/AP against all relevant document spans,
including gold missed by the candidate stage. A sklearn AP computed only over the
candidate labels is conditional head quality and must be labeled separately; it
cannot replace full evidence mAP. No-evidence pairs need an explicit convention and
counts, while remaining present in label accuracy/macro-F1.

## Pre-tokenization admission gap

Current `/v1/check` bypasses Runtime.slots and calls engine.check directly; the
engine submits tokenization to a one-worker ThreadPoolExecutor before batcher
admission. That executor queue is unbounded. CheckRequest currently permits 512 ×
20,000-character spans (~10MB per request), and full token lists are built before
per-row truncation. Concurrent requests can therefore queue/allocate unbounded
memory even with the repaired inference row limit. Add bounded admission covering
queued tokenization and a sensible aggregate input-size budget, not only individual
string/list caps. Reject overload clearly; test blocked tokenization and repeated
requests/cancellation without allowing retained executor work to exceed the bound.
Document any service request-timeout behavior and keep client timeouts meaningful.

Preparation throughput: full CPU candidate embedding is nontrivial (~42k segments).
Profile before an unnecessarily repeated host run. A cached processed dataset with
verified hashes can be prepared once and shared by every baseline/model. If adding a
Colab GPU preparation option, preserve the pinned embedding semantics, record device
and numeric scope, and freeze one candidate artifact for all compared methods rather
than silently preparing different retrieval inputs for each method.

Candidate-scope clarification after reading prepare.py: it now says TF-IDF/LLM see
BGE top-8 while the own model scores every segment. That follows the handoff's
learned evidence-retrieval design, but differs from the earlier shared-top-8 plan.
Make this explicit before experiments: a gain in the full pipeline cannot be
attributed solely to the classifier when candidate scopes differ. Keep TF-IDF and
zero-shot LLM on identical retrieved spans as required. Consider a fixed-BGE-top-8
own-model comparison to isolate label/evidence-head quality, alongside the intended
all-span learned-retrieval pipeline. Do not quietly describe all methods as having
identical input spans if they do not. Decide/freeze protocol using TRAIN/DEV only.

## Confirmed gzip reproducibility defect

`write_jsonl_gz` sets mtime=0 but omits filename when constructing gzip.GzipFile
around a named temporary raw file. Python embeds that random temporary basename
in the gzip header. Independent `od -An -tc -N90` of the smoke document file shows
FNAME flag 0x08 and `.documents_train.jsonl.gz.0gxzn37e` in its header. Thus equal
rows give different file hashes/data_version across runs despite the comment.
Pass an explicit stable/empty gzip filename and add a two-output hash equality
regression check, including runs targeting distinct temporary basenames. Rebuild
smoke data after fixing this before it becomes a checkpoint/input fingerprint.

Full data prep has now started in the background. It imported the old gzip writer;
editing io.py in parallel will not change that function in the already-running
process. Do not accept old randomly named gzip output as deterministic or silently
stamp it with a fingerprint of changed runtime code. Preserve expensive computed
rankings if possible, but regenerate deterministic serialized files and manifest/
data_version under the fixed writer before training, with truthful provenance for
that conversion. Avoid re-embedding the entire corpus merely to change gzip headers.
Read these recent review additions before finalizing full data preparation.

Converted-tokenizer cache integrity: backbone.py validates pinned source files, but
an existing `rulecheck-tokenizer.json` bypasses conversion/parity checks completely.
A stale/corrupt derived tokenizer can therefore change training IDs while source
pins still pass. Validate the cached derivative against its recorded hash and source/
converter identity (or regenerate/check it), and test corruption recovery/refusal.
Keep training/export/serving on the same verified IDs.

## Independent data inventory

Executed the new canonical loader against the verified original archive without
running training or altering candidates. **Exit 0**: train423docs/7191pairs,
dev61/1037, test123/2091; document ID sets disjoint; class counts agree with raw
inventory; zero evidence-bearing pairs without mapped canonical segments. Log:
`/tmp/compliance-supervision/phase3-data-inventory.log`. This verifies data integrity,
not classifier/evidence performance or Phase 3 acceptance.

Training source review: in fp16, GradScaler can skip optimizer.step on overflow but
train.py currently advances scheduler and optimizer-step counter unconditionally.
Keep scheduler/step semantics truthful and correct when a scaled update is skipped;
add a focused overflow-path check without requiring real CUDA if feasible.

Checkpoint interruption window: _save_last renames last→.last-old then staging→last.
A disconnect between renames leaves only .last-old, but current resume looks only at
last/state.json. Recover the validated prior checkpoint (or use an atomic pointer
scheme) instead of silently restarting. Test that interrupted promotion layout and
an uninterrupted-versus-resumed smoke run, including RNG/optimizer/scheduler state.

LLM baseline source review: read_cache keeps only the latest attempt per pair, and
to_predictions totals only those latest responses. A billed failed/stop-reason
response followed by a successful rerun disappears from cost accounting. Preserve
all known attempt usage/cost separately from final prediction results; report
transport failures/SDK retries with unknown usage honestly (known cost is a lower
bound if usage cannot be recovered). Aggregate known latency/attempt counts too.

Validate CLI bounds before paid calls: --concurrency=0 currently deadlocks, while
--limit=0 falls through to the full dataset and negative limits silently slice most
of it. Reject nonpositive supplied limits/concurrency clearly. Cache identity should
also hash the actual per-pair rendered prompt (template edits must not silently reuse
old answers merely because PROMPT_VERSION wasn't manually bumped).

LLM evidence validity: parse_answer currently ignores fabricated excerpt IDs and
then scores only retained IDs. Count invalid claimed IDs as evidence false positives
(or explicitly invalidate the evidence answer and report that convention); simply
removing them improves measured precision. Label correctness can remain a separate
metric, but hallucinated citations must remain visible in evidence quality.

While adding aggregate input/admission validation, reject whitespace-only hypothesis
and span text too; min_length=1 currently permits an inference request with no
usable rule/evidence text. Preserve original text for offsets; validation need not
silently strip or normalize canonical source text.

## Independent encoding/batching subset

`timeout 180 env -u PYTHONPATH .venv/bin/python -m pytest apps/model-service/tests/test_rule_check_encoding.py apps/model-service/tests/test_rule_check_batching.py -q`

**12 passed in 0.67s**, exit 0. Controlled backend tests cover concurrent result
alignment, split request size, bounded admission, malformed/non-finite outputs,
shutdown failing active+queued futures, and canceled work retaining capacity until
actually released. The original active-shutdown/canceled-row bugs are resolved in
source plus this focused verification. Pre-tokenization admission and other later
review findings still need their own checks. Log:
`/tmp/compliance-supervision/phase3-encoding-batching-check.log`.

Production/smoke provenance guard: TrainConfig prevents random backbones/max_steps
from being labeled production, but train.run currently permits a production config
pointing at the smoke dataset. Refuse production training/export/serving promotion
from subset data (while preserving the legitimate full-data unseen-rule experiment).
Do not rely only on evaluate.py refusing the standard results output directory.

Training isolation: train.run currently calls load_dataset with default all splits,
which parses TEST labels before model selection despite config.py's "TEST is never
read here" comment. Load TRAIN/DEV for training/selection, then load TEST only after
the selected checkpoint/k are frozen if the CLI generates final predictions there.
Checksum verification of input files is fine; test targets must not be training data.

## Independent first ML unit subset

`timeout 180 env -u PYTHONPATH .venv/bin/python -m pytest ml/tests -q`:
**16 passed in 3.52s**, exit 0. Data/rule-pack integrity, metric calculations and
training-example construction are covered; no full training/performance claim.
Log `/tmp/compliance-supervision/phase3-ml-unit-check.log`. Later review fixes may
need additional targeted tests before this becomes a final gate.

Resume identity: train.run compares prior config_hash, but also verify the original
processed data_version and derived tokenizer hash before loading a checkpoint.
Otherwise a regenerated/changed dataset at the same configured path can silently
resume an old run. Record and check relevant training-code identity, or explicitly
require a new run directory for incompatible code changes. A resumed session's new
provenance should supplement, not overwrite, the original run's identity.

## Independent combined review-fix subset

`timeout 180 env -u PYTHONPATH .venv/bin/python -m pytest ml/tests apps/model-service/tests/test_rule_check_encoding.py apps/model-service/tests/test_rule_check_batching.py apps/model-service/tests/test_rule_check_service.py -q`

**39 passed in 11.27s**, exit 0. Includes the repaired gzip hash invariance, data/
training inputs/metrics, skipped-fp16 update behavior, checkpoint recovery layouts,
known-attempt LLM accounting and invalid-evidence penalties, artifact contracts,
batching shutdown/cancellation, actual tokenization-work admission under blocking/
cancellation, aggregate input rejection and nonblank input validation. Model logits
are deliberately deterministic/fake in contract tests; this does not replace the
real train/export/ONNX service smoke or full Colab training.
Log `/tmp/compliance-supervision/phase3-review-integration-check.log`.

Remaining integration reminder: Docker compose's model-service healthcheck still
expects rule_check=not_implemented, while the new health schema uses unavailable/
ready. Audit/update current smoke/health contracts and generated API types before
rebuilding; preserve historical phase evidence rather than rewriting old results.
Validate the actual container path for loading an imported deployment artifact, not
only the host Python/in-process test path.

Colab GPU environment integration: after installing CUDA torch over this project's
CPU-locked torch, a later `uv run --frozen` (including one hidden inside make targets)
can resync/reinstall the CPU build. Notebook commands must preserve the CUDA runtime
(e.g. explicit no-sync or a separate correctly resolved ML environment) while running
the same versioned scripts and pinned tool dependencies. Verify torch.cuda and its
build immediately before training, and test/inspect every nested command path; a
one-time successful CUDA check before a resync is insufficient.

Int8 smoke review: batch-dependent activation scaling is a legitimate numerical
property; the added export batch-sensitivity measurement is good. Keep tolerances
measured/documented. In the service smoke, do not skip all label/confidence checking
when near-tied evidence scores select different spans. Rebuild the offline label
input from the spans the service actually chose and compare that distribution within
measured tolerance; always verify label/maximum confidence match the returned
probabilities. This preserves meaningful two-pass contract coverage for every case
without falsely demanding exact top-k equality from a random tiny int8 model.

## High-priority bundle importer path/overwrite safety finding

Tar member checks are good, but metadata later constructs filesystem destinations:
- install_artifact uses MODELS / manifest.model_version; model_version currently has
  no safe-name constraint. `../../...` can escape MODELS, and existing target is then
  recursively deleted.
- _install uses CACHE/data/data_manifest['name'] and baselines/version[:16] without
  validating a plain allowed name / 64-hex data_version. A valid-hash bundle can move,
  replace or delete paths outside its intended cache subtree through these fields.

Validate/restrict every metadata-derived path component and enforce resolved path
containment before any mutation (including public install_artifact). Preflight all
staged dataset/run/result/artifact contracts before moving or overwriting existing
local data/results; artifact validation currently happens only after results moves.
Never delete an existing target/aside merely because the incoming metadata collides;
reuse verified identical content or preserve a unique backup. Add malicious-version/
name and late-invalid-artifact tests with local sentinel data proving no mutation
outside scope and no partial overwrite on failed preflight. Avoid extracting an
unbounded-size/count archive on this nearly full disk; enforce a concrete cap/budget.

## Notebook first-pass findings

- Default source Option A runs `git bundle verify` before cloning, from /content
  outside a Git repository; that command requires repository context. Use a valid
  verification/clone sequence and actually test the default bundle path from a
  fresh directory. Exact commit + clean checkout checks remain necessary.
- Quantized TEST accuracy is required by this phase, but RUN_ONNX_ACCURACY defaults
  false. Make the required measurement part of the complete path (or explicitly
  label the default run incomplete and give the exact completion step).
- The artifact is exported before the optional real LLM/cascade evaluation. When
  RUN_LLM=true, evaluate regenerates final tau/inference metadata afterward, but the
  artifact bundled at the end is still the earlier export with provisional tau=None.
  Refresh/re-export the artifact after final DEV cascade selection, keeping weight/
  benchmark provenance accurate and metadata synchronized. Do not report a finalized
  cascade while serving an artifact with stale parameters. Consider parameter hashes
  in artifact version identity so changed calibration/cascade isn't the same version.

Importer follow-up: safe regex alone still permits artifact model_version="current",
which collides with the active pointer. install_artifact can rename that pointer,
install a nonempty directory at current, then fail replacing it with its own symlink.
Reject reserved artifact version names before mutation and test preservation of an
existing current pointer. Also reject non-identical source/target directory ancestry
for public install-artifact so a mistaken parent source cannot be copied recursively
into its own descendant. These checks belong in preflight and the public helper.

## Independent live rebuilt-service regression

Independently checked Docker: all seven services healthy after API/model-service
rebuild. Actual model-service /health reports embedding=ready,
rule_check=unavailable, rule_check_mode=untrained_fallback (no production-trained
artifact installed). Then ran the live Phase 2 smoke: **exit 0, all five groups
passed**, preserving real hybrid/keyword/vector search, exact stored source refs,
filters, spreadsheet links and workspace isolation. Log:
`/tmp/compliance-supervision/phase3-live-retrieval-regression.log`.
This does not yet verify the new web image/banner or a trained artifact in Docker.

## First independent full-gate attempt during Claude quota reset

`make ci` exited 2 at Ruff format-check: the final newly appended reserved-name/
ancestry smoke test in ml/smoke/tests/test_smoke_pipeline.py is unformatted. Ruff
lint itself passed. No implementation edits made by supervisor; Claude must finish
that test and formatting on reset (reported 19:20 Phoenix, current ~19:05–19:10).
The ancestry test's final conditional expression also appeared unfinished and needs
review rather than a cosmetic formatter-only pass.

Separate independent full type/schema/backend/web/build regression checks and the
real train/export/ONNX service smoke are running, with logs
phase3-first-regression-checks.log and phase3-tools-independent.log under
/tmp/compliance-supervision. Their results are not yet accepted. All seven live
services independently healthy again after the temporary smoke artifact was removed;
main model-service is back to genuine untrained_fallback.

Visual banner review: independently viewed /tmp/cw-ml/banner-fallback.png. Layout is
clean, but the text repeats "No trained rule-check model" twice and exposes fallback
implementation wording. Use one concise user-facing availability message. Before
Phase 4 exists, avoid implying an operational LLM review path already handles checks;
the banner should describe actual current availability. Keep tests aligned with that
meaning, not duplicated exact phrasing.

Second independent gate attempt: `make typecheck openapi-check test build-web`
stopped at strict mypy: ml/colab/bundle.py:117, `"object" has no attribute "values"`.
Log phase3-first-regression-checks.log. Supervisor then started the unaffected
`make openapi-check test build-web` separately, preserving the type failure as open.
No supervisor implementation edits or auto-formatting workaround.

## Independent real pipeline/tools smoke (first pass)

`timeout 1200 env -u PYTHONPATH uv run --frozen --with-requirements ml/requirements-tools.txt python -m pytest -q -p no:cacheprovider ml/smoke/tests --basetemp /tmp/compliance-supervision/phase3-tools-temp`

**13 passed, 1 failed in 116.35s**. Real tiny train/export/quantization, fp32/int8
service consistency, exact interrupted/resumed result identity, changed-identity
refusal, subset/production refusal, tokenizer cache recovery, bundle round-trip and
malicious/late-invalid preflight protection, and metadata refresh passed.
The unfinished final ancestry test expects a refusal for a path that resolves to
exactly the installed source/target (the idempotent case), so no exception is correct.
Construct an actual non-identical ancestor/descendant valid-artifact source and
preserve an explicit idempotent-success check instead. Also format that test. Do not
weaken the existing ancestry protection to match the mistaken test.
Log `/tmp/compliance-supervision/phase3-tools-independent.log`; supervisor owns its
temporary outputs under phase3-tools-temp. Final whole-suite rerun still required.

## Independent application regression and real Docker inference

- `make openapi-check test build-web`: **exit 0**, 334 application pytest tests,
  25 ML unit tests, 87 Vitest tests, schema/type generation check and production web
  build passed. This deliberately does not erase the open Ruff-format/mypy findings.
- Independently generated a fresh tiny smoke checkpoint using the actual train →
  classic baseline → evaluate → export CLI sequence in supervisor-owned temp space.
- Started only `compliance-phase3-verify`, the rebuilt model-service image with its
  default nonroot user, explicit smoke opt-in and a read-only verified int8 artifact
  bind mount, on isolated port18010. Eight concurrent actual HTTP /v1/check requests
  passed kind/variant, finite normalized probabilities, label/maximum-confidence,
  source-ID and shared-batch checks. Two multi-request batches, 32 rows, no failures.
  The verifier container was removed in a finally/trap; the main stack wasn't changed.
- Independently confirmed the main seven containers all healthy and model-service
  restored/unaffected as untrained_fallback. A sandbox-only localhost curl couldn't
  reach host networking; elevated host-network checks passed.

Logs: phase3-first-app-regressions.log,
phase3-container-e2e/build.log and phase3-container-e2e/http-check.log under
/tmp/compliance-supervision. These prove the actual tiny artifact's end-to-end and
container serving path, not full trained-model or real Anthropic performance.

## Independent full-label baseline metric recomputation

Read all saved baseline TEST predictions for data_version 620358b51ef96fac and the
original official TEST labels. All methods have exactly 2,091 unique matching pair
IDs, with no dropped NotMentioned cases. Independently recomputed accuracy/macro-F1
using sklearn and compared to the custom implementation: exact agreement.

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| global majority | .462936 | .210962 |
| per-hypothesis majority | .659493 | .616779 |
| TF-IDF/logreg | .763271 | .722885 |

Log phase3-baseline-recompute.log. Final report currently selects C=30, balanced,
from DEV; earlier logged runs selected C=10 at their smaller grid boundary. Document
any grid expansion as DEV-driven, retain actual run provenance, and do not alter
baseline choices based on the TEST scores now inspected. "TEST predicted once"
wording is inaccurate across reproducibility/regeneration runs; state DEV-only
selection and unchanged/no-TEST-feedback protocol instead. These are real classical
baselines, not evidence that the required trained model or LLM baseline is complete.

## Supervisor storage action for final web build

All host web tests/build already passed. At ~02:30 UTC, to avoid touching shared
Docker state or external storage, supervisor removes only regenerable project
apps/web/node_modules and apps/web/.next (not source/lockfiles). The running Docker
web image is unaffected and Docker build context ignores both directories. Restore
host dependencies with npm ci after final image/old-private-cache cleanup, then
run any required final web checks. No archive needed: package-lock pins dependencies
and normal approved network/package caches remain available. Do not interpret the
missing host node_modules as an implementation failure.

Storage sequencing clarification: host node_modules was intentionally removed by
supervisor after all web checks passed, to give the container build headroom. Do not
reinstall it before the image build if that reintroduces the same disk shortage.
A brief local web-only outage is authorized for this reversible rebuild: after
checking that no other project's containers reference the exact old compliance web
image, stop/remove only this project's web container and retire that old image if
needed, build/start the new one, reclaim exact new private build records, then npm ci
restores host dependencies. Leave API/data/other projects running. No shared daemon
restart, broad prune, or external-storage write is authorized/needed for this path.

## Supervisor owns final Docker web rebuild now

Do not run a concurrent web build/recreate or restore host npm dependencies until
supervisor reports completion; continue tests/documentation. Supervisor verified the
actual running web container is 4d1a206e5383 (project=compliance, working directory
this repo), image d79adc3748601f430221ad3e8703cc6ea04a6c73a6840d7f6a1d41a4fdc84bee,
used by no other container. Will retire only those exact IDs for brief local web-only
replacement. This is management work, with all application code still Claude-authored.

## Final local readiness gate — passed, full Phase 3 acceptance still pending

- Final lint/format: clean, 188 files formatted; strict mypy: 165 files clean.
- Schema and generated TypeScript types current.
- Independent final tests: **334 app pytest + 25 ML unit + 87 Vitest passed**.
- Independent whole real ML smoke: **14 passed in 113.26s**. The earlier wrong
  ancestry test is corrected and all importer, resume, quantization/service and
  metadata-refresh checks now pass.
- Final web source production build succeeded in an isolated Node22-alpine builder
  with bounded 3GB tmpfs/4GB memory. Claude-authored runtime-only recipe retained the
  original runner contract; image 3c66029c43ed8685bdd42c49c168cc7d6a2244abd1b139adeec1b11d15015367
  is running. Only this project's old web container/image were retired; other services
  stayed running. No shared Docker restart or external storage used.
- Rebuilt final API/model-service source layers (heavy deps reused), restored all
  seven healthy services; Phase0/1/2/3 live smokes passed.
- Independent browser on final Docker web/API: concise actual fallback banner,
  desktop+390px mobile checked, no console/page errors. Final API image synchronization
  resolved an earlier stale runtime message. Main service honestly remains untrained.

Logs in /tmp/compliance-supervision: phase3-final-release-checks.log,
phase3-final-ml-smoke.log, phase3-web-ram-build.log, phase3-final-python-images.log,
phase3-final-live-smokes.log, phase3-final-banner.log; screenshots phase3-final-banner*.png.

These are split local release gates, not a claim that disk-backed `make ci` with its
normal Docker builder ran here: equivalent fresh production build ran in RAM, and
hosted CI must exercise the normal repository path. Temporary host node_modules is
an ignored symlink into /dev/shm/compliance-web-hostdeps; normal lockfile install can
restore it when disk headroom returns. Only three exact new unshared runtime-build
records were pruned, recovering69.69MB. Cancelled project BuildKit records remain
pinned (~2GB), so model import still needs adequate free disk; no broad cleanup.

No production trained weights, real Anthropic baseline or full model/calibration/
quantization/cascade performance is complete. Colab+API execution and result import
remain required; no Phase4 work is authorized until those results are verified.

## Hosted pipeline verification and final Colab resume guard

Hosted run39715788502 for84ecada2cf106b26df92b9efc0b025294b42cfe6 completed
**success in all three jobs**: Python (including full real CPU ML smoke), web,
and normal Docker build/clean startup/automatic seed/Phase0–3 smoke. This verifies
the ordinary repository Docker path on a host with sufficient disk, supplementing
the local RAM build route.

A final notebook-only fix allows generated ml/eval/results/ outputs when re-running
source checkout in the same Colab runtime, while still refusing tracked/untracked
implementation changes elsewhere and requiring the exact expected commit. No model,
training, metrics or provider logic changed. Codex independently ran the actual
notebook guard's temporary-Git tests: **4 passed**, exit0, log
/tmp/compliance-supervision/phase3-notebook-final-check.log. Full ML unit suite is
now26 tests (previously25); the added test is the source guard. Final handoff commit
and its CI outcome are recorded in .cache/colab/HANDOFF.json and GitHub Actions.

Full GPU training, real Anthropic evaluation/cascade and trained-artifact import
remain external and unaccepted. The pending Docker-restart question has not granted
permission to interrupt unrelated projects.

### Colab FP16 failure: independent fix review

The first full Colab run at `4cd76d8` failed on the first optimizer update with
`Attempting to unscale FP16 gradients`. The user's traceback and Drive inventory
show a run record and empty training log, with no checkpoint. Claude reproduced
transformers 5 loading FP16 checkpoint parameters by default, explicitly selected
FP32 parameters in pretrained/random/config-only loaders, and added early dtype
checks to training and export. AMP and GradScaler remain enabled for CUDA.

Supervisor reviewed the implementation and real regression tests: default FP16
loading and the exact enabled-GradScaler exception are reproduced; loading the
same weights in FP32 preserves their values and permits a real optimizer update.
Independent ML unit plus full pipeline smoke: **43 passed** (29 unit + 14 smoke,
119.23 seconds). This includes actual tiny training, resume identity checks,
FP32/int8 ONNX export/parity and service inference. Ruff and format checks passed.
The initial independent smoke invocation lacked the pinned ML tool overlay; after
restoring it, the complete suite passed. No full production GPU result is claimed:
Colab must still demonstrate successful CUDA updates and complete all runs.
