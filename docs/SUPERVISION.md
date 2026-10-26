# Independent build supervision

The user assigned all implementation to Claude Code and management, review, and
verification to Codex. The authoritative requirements are in `BUILD_HANDOFF.md`.

## Phase acceptance ledger

| Phase | Implementation | Independently verified acceptance |
|---|---|---|
| 0 — Skeleton | Complete | Accepted: local gates and hosted CI passed |
| 1 — Ingestion and viewer | Complete | Accepted: local gates and hosted CI passed |
| 2 — Retrieval | Complete | Accepted: local gates and hosted CI passed |
| 3 — Trained model | Full Kaggle training done; full Haiku baseline (all 1037 DEV + 2091 TEST, $0.3713 incl. diagnostic) and offline evaluation done (TEST macro-F1 0.8138 vs TF-IDF 0.7244 vs Haiku 0.7034; tau 0.3, coverage 1.0); recovered fp32 served locally with the evaluated tau 0.3 (smoke OK); repaired int8 C1 passes full DEV and TEST (0.8156); hosted CI passed (`f0548aa`) ; int8 decision: keep fp32 served | Not accepted; pending final review |
| 4 — Agents and review UI | Not started | Pending |
| 5 — MCP | Not started | Pending |
| 6 — Evaluation and polish | Not started | Pending |

No phase is accepted solely on the implementer's report. Checks must be run and
results recorded. Mock, synthetic, and smoke results do not establish performance
on official datasets, real-provider behavior, or Claude Desktop interoperability.

## Initial environment observations

- Docker daemon and Compose are available outside the execution sandbox.
- Host Python is 3.10.12; the application requires Python 3.11+ in containers.
- Node 22.14.0 and npm 10.9.2 are available.
- `nvidia-smi` fails to communicate with the NVIDIA driver; GPU training is not
  currently verified as available.
- Initial free disk: approximately 7.4 GB. Monitor build and model storage needs;
  do not delete unrelated files, images, or volumes.
- Local Git identity: `JCHETAN26`, account-specific GitHub noreply email.
- Remote: private `JCHETAN26/compliance` repository.
- No AI coauthor trailers. Do not publish credentials or raw training datasets.

## User-confirmed choices

- Full GPU training: Colab when the training phase is ready.
- Real application LLM provider and zero-shot baseline: Anthropic API.
- Claude Code Max remains the implementation tool; it is not the application's
  Anthropic API credential or a substitute for real-provider validation.

## Independent Phase 0 checks — first backend pass

Commands run by Codex after Claude reported the backend ready:

- `env -u PYTHONPATH .venv/bin/python -m pytest -q`: **80 passed in 5.97s**.
  Run outside the sandbox against the isolated project's Postgres/Redis test stack.
- `env -u PYTHONPATH .venv/bin/ruff check .`: passed.
- `.venv/bin/ruff format --check .`: 55 files already formatted.
- `env -u PYTHONPATH .venv/bin/mypy packages apps/api apps/worker apps/model-service apps/mcp-server`:
  passed, 52 source files.

These checks do not yet accept Phase 0: the complete container build, web checks,
browser sign-up/workspace flow, and hosted CI remain pending.

## Independent Phase 0 checks — first frontend pass

- `npm run lint && npm run typecheck && npm test` in `apps/web`: passed,
  **53 tests in 8 files** (1.83s).
- An earlier typecheck during new test creation failed on the injected fetch
  function's TypeScript signature. Claude corrected the signature; the combined
  check above then passed. No implementation edits were made by Codex.

## Independent Phase 0 checks — running containers and browser

- `docker compose ps --format json`: all seven services running and healthy.
- App host port is 3300 (`WEB_PORT=3300`); the normal default remains 3000.
- Stable browser walkthrough command in `PHASE0_REVIEW.md`: passed with no console
  or page errors. Covers signup, keyboard submission, workspace CRUD, membership,
  backend error display, signout/login, and server/client session expiry.
- Workspace-detail screenshot visually inspected by Codex.
- First independent browser attempt overlapped with test edits and timed out;
  fixed-copy rerun succeeded, as recorded in the review file.

## Reviewed configuration installation

Claude authored `.pre-commit-config.yaml` but its noninteractive file tool requested
an interactive permission for that sensitive filename. Codex reviewed the proposed
content (local uv/npm quality hooks), copied the exact unmodified Claude-authored
artifact from the session log into the repository, and validated it with:

`PRE_COMMIT_HOME=/tmp/compliance-supervision/pre-commit .venv/bin/pre-commit validate-config`

Passed. A first validation attempt used the tool's default cache location, which
is read-only in the sandbox; the /tmp cache override resolved that environment
failure. No product implementation was authored by Codex.

## Independent Phase 0 final local gate

- `make ci` (log: `/tmp/compliance-supervision/phase0-final-checks.log`): **exit 0**.
  Includes Ruff lint and format, strict mypy (including scripts and conftest), ESLint,
  TypeScript, OpenAPI schema/type drift checks, 80 pytest tests, 53 Vitest tests,
  and the Next.js production build.
- `make smoke WEB_PORT=3300`: **exit 0, all 10 checks passed**, including
  browser-facing proxy auth, CSRF, workspace/membership controls, token revocation,
  and API/model/MCP health behavior.
- Redis recovery independently confirmed: all seven containers healthy, and
  `curl -fsS http://127.0.0.1:3300/healthz` returned web/API status `ok`.

Local Phase 0 acceptance is complete. Hosted CI is still required before starting
Phase 1 under the handoff's sequential gates.

## Phase 0 commit and hosted CI

- Commit `e6b709e944208a5a411fd3e6127054dd0b1da3ae` pushed to `main` on the private
  `JCHETAN26/compliance` repository. Author confirmed as `JCHETAN26` with the
  account-specific noreply email; no coauthor trailer.
- Hosted CI run: https://github.com/JCHETAN26/compliance/actions/runs/39678264159
  Python and web jobs passed; Docker build/smoke job pending at time of this entry.

## Scoped cache cleanup

Validated the 80 proposed BuildKit record IDs against fresh daemon metadata:
all private, reclaimable, and created during this project's Phase 0 build window.
Pruned using individual `--filter id=<verified-id>` arguments only. Free disk rose
from roughly 2.2 GB to 4.0 GB. 30 records were removed; other selected records remain
because of cache dependencies. All 48 older cache records were independently
confirmed preserved. No images, containers, volumes, source, or other-project
artifacts were removed. All seven application services remain healthy.

## Phase 0 accepted

Hosted run `39678264159` completed successfully: Python, web, and Docker build +
Compose smoke-test jobs all passed for commit `e6b709e`. Phase 1 is authorized to
begin. Full project acceptance remains pending Phases 1–6.

## Independent dataset-source check before Phase 1 sample data

Checked the publishers' pages on 2026-10-21 before using dataset content:

- ContractNLI: https://stanfordnlp.github.io/contract-nli/ — publisher identifies
  CC BY 4.0, 607 NDAs, 17 hypotheses, official document spans and annotations.
- CUAD: https://www.atticusprojectai.org/legal/ — publisher lists CUAD under
  CC BY 4.0. CUAD usage is a later-phase requirement.

The implementer must still record provenance, attribution and the distributed
license terms in `ml/data/LICENSES.md` before downloads/use. Keep raw data out of Git.

## Phase 1 local acceptance

See `PHASE1_REVIEW.md` for independent checks and corrections. Full final `make ci`
passed with 197 Python tests and 73 frontend tests, plus lint, strict typing, schema
drift checks and production build. Production browser and fresh-volume automatic
seed smoke both passed. The temporary acceptance project was removed; the main app
and data are retained. Hosted CI and final commit hygiene remain pending before
Phase 2 can begin. No full-project completion is claimed.

## Phase 1 commit and hosted CI

Commit `ec6c254b2fed4b33645aa060142ccdb4856efb2e` pushed to `main`, author JCHETAN26
with the account-specific GitHub noreply email, no coauthor trailer. Worktree clean
at push. Original handoff hash still matches the user attachment exactly.

Run: https://github.com/JCHETAN26/compliance/actions/runs/39687462736
Python and web jobs passed; Docker/clean startup/automatic artifact download/seed/
smoke job pending at this entry. Phase 2 remains unstarted until the full run passes.

## Phase 1 accepted

Hosted run `39687462736` completed successfully for `ec6c254`: Python, web, and
Docker build/automatic artifact downloads/clean seed/ingestion/smoke jobs all passed.
Phase 2 is authorized to begin. The full project remains incomplete (Phases 2–6).

## Independent Phase 2 verification (release pending)

- Fusion/client unit subset: 36 passed.
- Actual-model, PostgreSQL/FTS retrieval, model-service and search API subset: 95
  passed in 43.78s. Isolation, model protocol failures, encoder mismatch, reranking
  and exact stored source references checked.
- Live Docker smoke: all five groups passed. Seven services healthy afterward.
- Browser search flow passed: measured page-2 PDF highlight, spreadsheet row,
  keyboard navigation, restored URL state, filters/empty state, mobile layout,
  model outage causing hybrid 503 while keyword remains available, service restored.
- Evaluator safety/metric subset: 19 passed. Unsafe identifiers and unmanaged DBs
  preserved; evaluator ownership/concurrency lock checked; canonical wrapped long
  sentences and gold offsets checked; metric random expectation checked exactly.
- Full `make ci`: exit 0, 304 pytest, 83 Vitest, Ruff, strict mypy, ESLint, TypeScript,
  schema drift and production web build passed. See `PHASE2_REVIEW.md` for commands.
- Independent rerun of all 1,188 official TEST pairs reproduced saved keyword,
  vector and hybrid metric dictionaries exactly using the unchanged frozen config.
- Additional evaluation Ruff/mypy check passed (12 source files).

Current observed TEST evidence recall@3: keyword .7885, vector .8109, hybrid .8160,
hybrid+rerank .8603. Hybrid-versus-vector document-bootstrap interval includes zero;
there is no statistically robust improvement claim over vector. Hybrid improves
keyword. Rerank results are from actual benchmark execution, not the separate base
metric recomputation. These are document-scoped evidence metrics; random recall@10
is already .9187 because many documents have fewer than ten chunks. Query timings
are warm in-process/localhost measurements with 17 memoized query embeddings and
must not be described as cold HTTP or production latency.

The final complete benchmark command exited 0 and reproduced the evidence metrics.
Canonical DEV/TEST/sweep source fingerprint is ac6c9f740b87a2ad576d510bed5424e0cf3e0a3c679b62de7fad8abd0d183852.
Figures were independently inspected, including shared ANN comparison axes.
The app adopted DEV-selected k=5 / keyword weight=1.5 before the final run; the
k60/equal configuration is a reference baseline. This corrects an earlier supervisor
snapshot assumption. Hosted CI is still pending. Phase 2 remains unaccepted and
Phase 3 has not started.

Final Phase 2 local release gate passed after all review fixes: independent full
`make ci` exit 0, 24 evaluator tests exit 0, final rebuilt-image smoke all five
groups passed. Evidence is in `PHASE2_REVIEW.md` and
`/tmp/compliance-supervision/phase2-release-checks.log`. Phase 2 is locally verified;
hosted CI remains required before accepting the phase and starting Phase 3.

## Phase 2 accepted

Commit c61eab2eb60772442b5857a16bc45252ec9c9b91, authored and committed by
JCHETAN26, pushed to the private repository. Hosted GitHub Actions run
[39699615208](https://github.com/JCHETAN26/compliance/actions/runs/39699615208)
completed **success** for all three jobs: Python, web, and clean Docker stack.
Build, seven service health checks, automatic artifact downloads/demo ingestion and
Phase 0/1/2 smoke steps all passed. Combined with the independent local verification
and actual official-dataset results, Phase 2 is accepted. Phase 3 may now begin.

## Phase 3 local pipeline readiness

Claude authored the data/training/baseline/calibration/export/serving pipeline,
Colab notebook, artifact handoff, UI status and test/CI integration. Independent
final local gates passed: 334 application tests, 25 ML unit tests, 87 web tests,
14 real train/export/ONNX-service smoke tests, lint/format, strict mypy165files and
schema checks. A production build of the unmodified web source ran in bounded RAM
to avoid pinned cancelled BuildKit layers; final web/API/model-service images are
running and all seven services healthy. Actual desktop/mobile browser checks and
Phase0–3 live smokes passed. Details/limitations are in PHASE3_REVIEW.md.

Classical baseline predictions cover all2091officialTESTpairs; independent sklearn
recomputation confirms TF-IDF/logreg macro-F1 .722885, per-hypothesis majority .616779,
global majority .210962. All choices are DEV-only; none of these establish trained
model superiority or real-provider performance.

Full Phase3 is **not accepted**: Colab full GPU training/ablation/unseen-rule runs,
real Anthropic baseline/cascade, full quantization results and independent imported
artifact verification remain. Phase4 remains gated. Hosted pipeline run39715788502 for84ecada passed all three jobs, including
normal Docker build and the real CPU ML smoke. A final notebook-only resume-guard
fix passed four independent tests; no training/metric logic changed. Exact handoff
commit/latest CI are tracked in .cache/colab/HANDOFF.json and GitHub Actions. Host disk remains constrained by ~2GB of this
project's cancelled BuildKit records; unrelated projects/caches were preserved.

## Phase 3 Kaggle recovery status (2026-10-25)

This supersedes the "Colab ... pending" wording above. Full GPU training ran on Kaggle,
not Colab. The fp32 DeBERTa artifact was recovered
(`rule-check-deberta_v3_base-0e28e064-e9245464-pf1e1842f`). The live worker and
model-service were recreated with the no-sudo recovery kit. All seven local services are
healthy, and the model-service serves that artifact as production fp32 with tau `None`.
One real Anthropic request passed. Hosted CI passed for `dcc1e2f` (run 40207127703).
Details: `KAGGLE_LOCAL_VALIDATION.md`. Phase 3 is still **not accepted**. Still
outstanding: the full LLM baseline (1037 DEV + 2091 TEST), the cascade tau, int8 and
final review.

## Phase 3 Haiku baseline and final evaluation (2026-10-25)

This supersedes the "still outstanding" list above.

- The approved full Haiku baseline exited 0. All 1037 DEV and 2091 TEST pairs were
  answered, with 0 errors and 0 attempts missing usage. Total cost was $0.371269985,
  including the Sonnet diagnostic.
- The independent offline `ml.eval.evaluate` run at `49550f2` also exited 0. TEST
  macro-F1: trained model 0.8138, TF-IDF 0.7244, Haiku 0.7034.
- The DEV-selected tau is 0.3 with coverage 1.0, so nothing escalates. The cascade claims
  no accuracy gain over the model. It is not faster than Haiku either (about 10–13 s per
  pair against 1.16 s).
- The aggregate results are in `ml/eval/results/`. Records: `HAIKU_EVALUATION_PLAN.md` and
  `KAGGLE_LOCAL_VALIDATION.md`.

Phase 3 is still **not accepted**. Pending:

- final review. Done so far: the int8 decision (keep fp32; the repaired int8 file passes on
  DEV and TEST but is slower on CPU, see `INT8_QUANTIZATION.md`). fp32 with tau 0.3 is
  served, and hosted CI passed for `f0548aa`.

Phases 4–6 remain pending and gated.
