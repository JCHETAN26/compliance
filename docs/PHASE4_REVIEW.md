# Phase 4 review: agents, live run, report and overrides

Implementer: Claude Code. Reviewer and acceptance authority: Codex (`SUPERVISION.md`).
This file is the implementer's plan, acceptance checklist and evidence log. Nothing here
accepts the phase. Status values: `todo`, `in progress`, `implemented` (code and tests
written), `verified locally` (named command ran and passed; output recorded below),
`blocked` (with the reason), `not authorized` (needs manager approval).

Requirements come from `BUILD_HANDOFF.md` §5, §8, §11, §12 and §14 (Phase 4) and from the
manager's Phase 4 instructions (2026-10-25).

**Outcome (2026-10-26): Phase 4 accepted by the supervisor** (`SUPERVISION.md`, "Phase 4
approved real-provider acceptance") after the approved real Haiku flagship ("Real-provider
flagship evidence" below) and hosted CI run 40346107207 (all three jobs passed, head
`6ea1588`). Entries below that say acceptance, hosted CI or the real run is pending are
kept as history of the state before that run.

## Acceptance checklist

### Handoff Phase 4 criteria

| # | Criterion | Status |
|---|---|---|
| H1 | Flagship vendor demo runs end to end with the mock LLM | verified locally (mock LLM; tests, preview smoke, browser) |
| H2 | Flagship vendor demo runs end to end with a real LLM | verified (approved Haiku run `abdc0cff-0431-4394-94f8-4070b2ae4dad`, manager-run and independently checked; see "Real-provider flagship evidence") |
| H3 | Every finding has a verified citation or is `needs_review` | verified locally |
| H4 | Prompt-injection regression test passes | verified locally |

### Graph and agents

| # | Item | Status |
|---|---|---|
| G1 | LangGraph `StateGraph` with typed shared state (`run_id`, `workspace_id`, `rule_pack`, `documents`, `extracted_entities`, `tasks`, `findings`, `citations`, `budget`, `errors`) | implemented, tested |
| G2 | Seven agents as graph nodes: planner, extraction, retrieval, rule check, cross-document reasoning, citation verifier, report writer | implemented, tested |
| G3 | Parallel fan-out (`Send`) bounded by `max_concurrency`; per-node retry policy with backoff | implemented, tested |
| G4 | Checkpoints persisted in Postgres (custom saver on the app database), resumable after crash, failure, pause or interrupt | implemented, tested |
| G5 | Human-in-the-loop: optional `interrupt()` on `needs_review`; override, then resume | implemented, tested (API + browser) |
| G6 | Every agent step emits persisted `agent_events` (start, tool call, result, error, tokens, cost) | implemented, tested |
| G7 | Versioned prompt files; prompt, model and template versions recorded on findings and LLM calls | implemented, tested |

### Rule checking and evidence

| # | Item | Status |
|---|---|---|
| R1 | `model_supported` rules use the trained model client; DEV-selected tau 0.3 (from the model artifact, overridable per workspace); confidence below tau escalates | implemented; mock-backend tests + bounded real fp32 run |
| R2 | Unsupported rules and true escalations use the provider's structured verdict | implemented, tested |
| R3 | Structured outputs grounded against the span IDs actually presented; unknown IDs rejected | implemented, tested |
| R4 | Citation verifier: exact stored text at stored offsets **and** semantic support (model for model-decided findings, LLM support check otherwise); failure gives `needs_review` | implemented, tested |
| R5 | No citation (for example `not_mentioned`) means `needs_review`; the machine label is kept separately | implemented, tested |
| R6 | Extraction entities cite a chunk with an exact quote; dates/terms are parsed from the quote, never taken on trust | implemented, tested |
| R7 | Register rows link to documents conservatively (exact or high-margin fuzzy filename, unique party match); ambiguous gives `needs_review` | implemented, tested |
| R8 | Cross-document reasoning: high-access vendors need a contract with the required protections; contract term must cover the renewal date; missing, expired and ambiguous cases tested | implemented, tested |
| R9 | Report states only verified findings, or explicitly marks items for human review; no invented dates, clauses or citations | implemented, tested |
| R10 | No cascade-improvement claim: tau 0.3 gives full coverage on DEV/TEST; escalations are not forced | implemented (0 escalations at tau 0.3 in the real-model run) |

### Safety and correctness

| # | Item | Status |
|---|---|---|
| S1 | Tenant, role and CSRF checks on every new route, including SSE and overrides; no IDOR or cross-workspace citation leaks (tests) | implemented, tested |
| S2 | Document text is untrusted, delimited data; no tools, shell commands or URL fetches are driven by document content | implemented, tested |
| S3 | Stable per-run event cursor (`seq`) with JSON pagination and SSE `Last-Event-ID` replay/reconnect | implemented, tested (real SSE route) |
| S4 | Long checks run in the worker; HTTP routes only enqueue | implemented |
| S5 | Duplicate ARQ jobs and resumes are safe: per-run lease with fencing token, idempotent job ids, memoized steps; a killed worker is recovered once its lease expires; overrides and resumes serialize on the run lock | implemented, tested |
| S6 | Resume skips successful steps without rebilling (step memo + checkpoint pending writes) | implemented, tested |
| S7 | Budget: durable conservative reservation before every provider request (serialized per run); measured usage after; unknown attempts keep their reservation; mock usage labelled `mock_estimate` | implemented, tested |
| S8 | Providers through env: Anthropic, OpenAI, Ollama, mock. Plain mock reads no key and calls no external LLM | implemented, tested (no real provider called) |
| S9 | Secrets never in the database, checkpoints, events, logs, frontend or Git; LangSmith tracing forced off | implemented, tested (sentinel secret) |
| S10 | Human overrides are audited rows; the original machine finding is never rewritten | implemented, tested |
| S11 | Unsupported or missing-evidence findings are never marked compliant to finish the demo | implemented, tested |

### API and UI

| # | Item | Status |
|---|---|---|
| A1 | Rule packs: list, get, create, update, YAML import/export; built-in `nda_standard` and `vendor_confidentiality_review` | implemented, tested |
| A2 | Reviews: start, list, get, control (pause/cancel/resume), events (SSE + paginated JSON), tasks grid, findings, overrides, report (md/json), citations | implemented, tested |
| A3 | Review settings per workspace (provider/model among env-configured providers, tau, budget caps, pause-on-review, concurrency); global read-only `/settings/llm` | implemented, tested |
| A4 | OpenAPI schema and generated TypeScript types updated | verified locally |
| A5 | Web: rule packs page with model-support badge and editor; new review; live run (timeline, progress grid, counters, pause/resume/cancel); report (filters, per-subject view, citations that open the highlighted viewer, override with comment, export) | implemented, tested (Vitest + browser) |
| A6 | Browser flagship run against isolated test infrastructure | verified locally (isolated preview stack) |

### Quality gates

| # | Gate | Status |
|---|---|---|
| Q1 | Ruff lint + format | verified locally |
| Q2 | mypy strict (`packages apps conftest.py scripts evals ml seed`) | verified locally |
| Q3 | ESLint + `tsc --noEmit` + Vitest | verified locally |
| Q4 | OpenAPI drift check | verified locally |
| Q5 | Full application pytest (existing suites plus new Phase 4 suites) | verified locally |
| Q6 | Next.js production build | verified locally |
| Q7 | Bounded real-model verification (fp32 service, a few pairs) | verified locally (bounded) |

## Design

### Packages

- `compliance_llm`: provider abstraction (`anthropic`, `openai`, `ollama`, `mock`) over raw
  HTTP with timeouts; pricing table; structured JSON-schema outputs; usage reporting. No
  provider SDKs, so no new heavy dependencies.
- `compliance_core`: ORM models and migration `0003`, rule-pack schema/validation/YAML,
  services for rule packs, reviews (runs, tasks, findings, citations, overrides, events,
  reports), review settings, and the pure report renderer used by the API and the agents.
- `compliance_agents`: LangGraph graph, the seven agents, the Postgres checkpoint saver,
  the budgeted LLM gateway, the rule-check model gateway (service client and test mock),
  prompts, and the run executor (lease, heartbeat, pause/cancel, interrupt/resume).
- `compliance_worker`: `run_review` ARQ job and a stale-run sweeper cron.
- `compliance_api`: new routes; the API only enqueues work.

### Graph

```
planner -> extraction (Send per contract) + register linking
        -> dispatch -> retrieval (Send per task) -> rule check (Send per task)
        -> cross-document reasoning -> citation verifier (Send per finding)
        -> human review gate (interrupt when enabled and needs_review exists)
        -> report writer -> END
```

Large data (document text, spans, quotes) stays in Postgres; graph state carries ids,
statuses and small summaries, so checkpoints stay small. Checkpoints use `durability="sync"`.

### Budget ledger

`llm_calls` rows are written before every provider request (`reserved`) inside a
transaction that locks the run row and checks `spent + reserved + new <= cap` for both
tokens and USD. After the response the row is settled with measured usage
(`succeeded`); transport errors after sending become `unknown` and keep the full
reservation; definite failures before sending are released. Parallel tasks therefore
cannot overspend. Mock calls record `usage_source = mock_estimate` and cost 0.

### Durability and idempotency

- Run lease (`lease_token`, `lease_expires_at`) acquired atomically by the job; heartbeats
  renew it; every event/finding write is fenced on the token.
- ARQ job ids are `review:<run>:<generation>`; duplicates are no-ops.
- `step_results` memoizes each provider/model step by a deterministic key, so a step
  re-executed after a crash returns the stored result without a new request.
- Findings are unique per `(run, task_key)`; events dedupe on an optional key.

### Interpretation choices (flagged for review)

- `decided_by` adds `system` for deterministic cross-document reasoning; the handoff
  lists model | llm | human.
- `not_mentioned` has no positive evidence to cite, so the final status is `needs_review`
  with `machine_status = not_mentioned` (criterion H3).
- "Contract must not have expired relative to the renewal date" is read as: the contract
  end date (stated, or effective date plus a stated term) must be on or after the
  register's renewal date. Missing or unparseable dates give `needs_review`.
- The report writer is deterministic (template over verified findings) so it cannot state
  unbacked facts; remediation text comes from the rule pack, not from a model.
- `/settings/llm` is global and read-only (providers come from env); editable defaults are
  per workspace at `/workspaces/{id}/review-settings`.
- Report export: Markdown and JSON in Phase 4; PDF export is left to Phase 6 ("report
  export" is listed there).

## Real-provider flagship evidence

The user approved one synthetic Haiku flagship capped at $0.25. Codex ran the plan below
once on 2026-10-26; the record of acceptance is `SUPERVISION.md`. Facts as independently
checked by the manager:

- **Run** `abdc0cff-0431-4394-94f8-4070b2ae4dad`: completed; Anthropic
  `claude-haiku-5-5`, live fp32 model service, concurrency 2, caps **$0.25 and 400,000
  tokens**.
- **Provider usage** (database accounting): 27 successful non-mock attempts (9
  extraction, 9 rule check, 9 citation verifier); 26,618 input + 6,778 output = **33,396
  tokens**; **$0.006062** in the application ledger from provider-reported usage (not an
  Anthropic invoice); zero retries, unknown outcomes or errors. No extra real reviews or
  baseline reruns.
- **Findings:** 58 in total: 37 satisfied and 3 violated, all 40 with verified
  citations, plus 18 `needs_review`. The trained model decided 27 tasks at tau 0.3 with
  zero escalations, so no cascade improvement follows.
- **Smoke checks, all passed:** OCR; missing-contract detection; report grounding and
  Markdown export; citation, file and cross-user isolation; all seven agents' events and
  SSE replay.
- **Injection:** the Hooli fixture did not make its vendor compliant; return/destruction
  and required protections stayed `needs_review`.
- **Environment after the run:** the paid-provider preview containers were removed and
  the mock preview restored with no Anthropic key configured. The original production
  containers and the served fp32 model were not changed.
- **Logs** (outside the checkout): `kaggle-recovery/logs/codex-phase4-haiku-approved.log`,
  `kaggle-recovery/logs/codex-phase4-haiku-evidence.json`.

These are synthetic acceptance results, not an accuracy benchmark on real contracts. The
measured usage is within the plan's estimate (about 15k–40k input, 3k–8k output tokens,
$0.002–$0.01).

## Real-provider flagship plan (historical: prepared before approval, run once as above)

At the time of writing this was not authorized: no Anthropic/OpenAI/Ollama request and no
key read had been made. The mock flagship was reviewable, so this was the plan submitted
for approval.

**Scope.** One review run of the synthetic ten-vendor flagship (`compliance_seed.synthetic`:
invented companies and clauses, no customer or dataset text) with
`vendor_confidentiality_review`, Anthropic `claude-haiku-5-5` for LLM steps and the
trained fp32 model (live model service, HTTP only) for model-supported rules, on the
isolated preview stack (ports 18000/13300, DB `compliance_preview`).

**Expected requests.** The mock run of the same corpus made 27 LLM steps (9 extractions,
9 custom-rule verdicts, 9 support checks). With the real model, support checks of
model-decided findings use the model, so at most 27 successful requests are expected;
with the 3-attempt limit this normal path permits up to 81 attempts. Model-service
failures or inputs the model cannot accept can add fallback verdict/support calls, so
81 is not a global request ceiling. Escalations below tau 0.3 are expected to be none.
The enforced limits are the run's USD and token caps, including all retries/fallbacks.

**Cost estimate.** Built-in Haiku prices $0.10 / $0.50 per MTok (cache write $0.125,
read $0.01). The mock's character-based estimate for the 27 steps was 15,388 tokens;
allowing for structured-output framing, expect roughly 15k–40k input and 3k–8k output
tokens: **about $0.002–$0.01**. The ledger's provable per-attempt bound (UTF-8 body bytes
+ 4,096 tokens at the cache-write rate, plus 1,024 output tokens) is about $0.0013 for
these prompts, so the normal 81-attempt path stays under $0.11. **Hard caps for the run: $0.25 and
400,000 tokens**, enforced before each request; measured usage above a reservation stops
the run.

**Commands (manager-run; the key is typed into the shell, passed by name, never echoed,
stored or logged):**

```bash
cd "/media/grv22/Extreme SSD2/project/compliance"
/dev/shm/cw/bin/preview.sh down
read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
PREVIEW_LLM_PROVIDER=anthropic PREVIEW_RULE_CHECK_BACKEND=service \
  PREVIEW_MODEL_SERVICE_URL=http://127.0.0.1:8010 /dev/shm/cw/bin/preview.sh up
unset ANTHROPIC_API_KEY
# preview web on 13300 (same as the mock run; see "Preview stack" below), then:
timeout 3600 /dev/shm/cw/bin/crun4.sh --network host -- timeout 3500 \
  python /src/scripts/smoke_phase4.py --web http://127.0.0.1:13300 --timeout 3000 \
  --budget-usd 0.25 --budget-tokens 400000 --max-concurrency 2
/dev/shm/cw/bin/preview.sh down
```

The smoke prints the run's measured tokens, cost and unknown attempts. Outcomes that
depend on the LLM (the custom third-party rule, extraction) may differ from the mock
run; the acceptance invariants (verified citations or needs_review, report states only
verified facts, missing contract flagged) are what the smoke asserts.

## Reproducing the local checks

Helper scripts live in `/dev/shm/cw/bin` (tmpfs) with durable copies in
`../kaggle-recovery/validation-scripts/` (`crun4.sh`, `zz_phase4.pth`, `pyt.sh`, `web.sh`,
`preview.sh`, `phase4-req.txt`, `phase4_ui_check.py`, `real_model_check.py`). `crun4.sh` is
`crun.sh` plus a pinned overlay of the 21 packages `uv.lock` added for LangGraph:

```bash
cp ../kaggle-recovery/validation-scripts/{crun4.sh,zz_phase4.pth,pyt.sh,web.sh,preview.sh} /dev/shm/cw/bin/
UV_CACHE_DIR=/dev/shm/p4/uvcache uv pip install --target /dev/shm/cw/overlay4 --no-deps \
  --python-version 3.11 --python-platform x86_64-manylinux_2_36 \
  -r ../kaggle-recovery/validation-scripts/phase4-req.txt
# lint + types
/dev/shm/cw/bin/crun4.sh --network none -- sh -c 'ruff check --no-cache . && ruff format --no-cache --check . && python -m mypy --cache-dir /tmp/mypy packages apps conftest.py scripts evals ml seed'
# all application tests (test Postgres 55432 / Redis 56379, artifact cache read-only)
TP=$(python3 -c 'import tomllib;print(" ".join("/src/"+p for p in tomllib.load(open("pyproject.toml","rb"))["tool"]["pytest"]["ini_options"]["testpaths"]))')
PT_TIMEOUT=2700 PT_INNER=2600 /dev/shm/cw/bin/pyt.sh $TP
# CI-like: console entry point, repo root not on sys.path (pyt.sh's .pth adds /src and can
# hide import errors). Collection exactly as `uv run pytest`, then a writable-cwd full run:
/dev/shm/cw/bin/pyt_ci.sh --collect-only -q
PT_WORKDIR=/tmp PT_TIMEOUT=3000 PT_INNER=2900 /dev/shm/cw/bin/pyt_ci.sh -c /src/pyproject.toml --rootdir /src -q $TP
# web (RAM copy of apps/web; node_modules never on the SSD)
/dev/shm/cw/bin/web.sh install && for s in lint typecheck test build; do /dev/shm/cw/bin/web.sh run $s; done
```

**Preview stack** (does not touch the live services):

```bash
/dev/shm/cw/bin/preview.sh up        # API 18000, worker, model service 18010 (embeddings only)
cd /dev/shm/cw-web/web/.next/standalone && cp -r ../static .next/static && cp -r ../../public public && \
  API_INTERNAL_URL=http://127.0.0.1:18000 PORT=13300 HOSTNAME=127.0.0.1 nohup node server.js &
/dev/shm/cw/bin/crun4.sh --network host -- python /src/scripts/smoke_phase4.py --web http://127.0.0.1:13300
PYTHONPATH=/dev/shm/cw-e2e/site python3.11 phase4_ui_check.py http://127.0.0.1:13300 FILES_DIR SHOTS_DIR
/dev/shm/cw/bin/preview.sh down
```

## Evidence log

Entries are appended as gates run. Real-provider evidence and final acceptance are in
"Real-provider flagship evidence" above and in `SUPERVISION.md`; "pending" statements
below are historical. The final head `6ea1588` passed hosted CI run 40346107207 (all
three jobs).

Independent manager checks (2026-10-26, working tree; at that time final acceptance was
pending):

- **Hosted CI passed for final product source `081089b`:** run
  [40343922911](https://github.com/JCHETAN26/compliance/actions/runs/40343922911), all
  three jobs green. Python: 444 application tests; 116 ML tests and one optional
  Anthropic-SDK skip; 15 CPU training/export/service smoke tests. Web: lint, types,
  104 Vitest tests, generated schema and production build. Docker: image build,
  clean startup, seed, and Phase 0–4 smoke. The clean CI stack has no trained artifact
  and used 72 mock fallback calls; its 58 findings were 40 satisfied, 5 violated and
  13 needs_review, cost $0, unknown calls 0. This differs from the local mock-model
  preview's 27 mock LLM calls and is not real-provider evidence. The complete local
  ML suite passed all 117 tests with the optional SDK present. This final record
  supersedes earlier entries that say hosted verification was pending.
- Final mock browser workflow independently rerun by Codex: all steps passed with no
  console/page errors. Log: `kaggle-recovery/logs/codex-phase4-browser.log`; screenshots:
  `kaggle-recovery/logs/codex-phase4-browser-shots/`.
- CI-style console entry point, repo root absent from injected paths: **444 tests
  collected** after Claude removed direct conftest imports. Log:
  `kaggle-recovery/logs/codex-phase4-ci-collection.log`. Initial hosted CI run
  `40341052392` failed those imports; the full corrected run is still pending here.
- Corrected console-entry-point full run: **443 passed**, one readiness test failed only
  because the verification checkout is mounted read-only and default `.data` was absent.
  Independent rerun of the entire readiness/API health file with temporary writable data:
  **6 passed in 3.44s** (`codex-phase4-ci-readiness.log`). Hosted CI uses a writable
  checkout; the next hosted run is required before claiming a clean final CI result.
- Full application suite against isolated DB `compliance_codex_phase4_test` and Redis
  index 12: **436 passed in 337.31s**. Late concurrency/packaging corrections were still
  under review and require their focused rerun. Log:
  `kaggle-recovery/logs/codex-phase4-full-application.log`.
- Late crash-recovery, override/resume locking, and negative verifier cases independently
  rerun: **7 passed in 42.92s**. Log:
  `kaggle-recovery/logs/codex-phase4-review-regressions.log`.
- Offline container with `--network none`: provider adapters/accounting, rule-pack
  validation, and agent helper suites: **45 passed in 2.04s**. Log:
  `kaggle-recovery/logs/codex-phase4-offline-units.log`.
- Complete ML unit suite independently rerun with networking disabled and read-only
  mounts of `/usr/bin/git` and `/usr/lib/git-core`: **117 passed in 25.73s** (one existing
  Torch deprecation warning). This resolves the implementer's three Git-missing container
  failures; no ML test remains blocked on that environment issue. Log:
  `kaggle-recovery/logs/codex-phase4-ml-tests.log`.
- Inspected the browser's PDF citation screenshot: copied quote and measured PDF line
  highlights align. Timeline readability findings remain recorded below.
- All seven original live services healthy after accepted Phase 3 source was pinned;
  live web `/healthz` reports API healthy. Phase 4 preview is separate.

Implementer runs (Claude Code, 2026-10-26, final working tree; logs under `/dev/shm/cw-e2e`):

- `crun4.sh --network none -- ruff check . && ruff format --check . && mypy packages apps
  conftest.py scripts evals ml seed`: all checks passed; 259 files formatted; **mypy: no
  issues in 230 source files**.
- Full application suite (all `testpaths`, real pinned embedder, Tesseract, test
  Postgres/Redis), `pyt.sh`: **436 passed in 337 s** (log `pytest-final.log`). New suites:
  `packages/llm/tests` (provider shapes, error classes, strict usage, cache pricing,
  reservation bound, mock reads no key/opens no socket), `packages/core/tests/test_rule_packs.py`,
  `packages/agents/tests` (units, flagship, injection, durability/budget/lease/pause/cancel,
  review flow: interrupt + override, source change/deletion incl. mid-run, escalation,
  mis-roled facts, model-answer validation), `apps/api/tests/test_reviews_api.py` (roles,
  CSRF, non-member 404s, cross-workspace citation, control, real SSE route, rule packs,
  settings), `apps/worker/tests/test_reviews_job.py`.
- ML unit tests: the ML code is unchanged in Phase 4; the evidence of record is the
  manager's run above (**117 passed** with `/usr/bin/git` and `/usr/lib/git-core` mounted
  read-only). There is no Git blocker.
- Web (RAM copy of `apps/web`, `web.sh`): ESLint clean, `tsc --noEmit` clean, **Vitest 104
  passed**, `openapi-typescript` output identical to the committed `schema.d.ts`, `next build`
  succeeded (16 routes incl. rule-packs, reviews, run, report).
- Preview stack (`preview.sh`: test Postgres DB `compliance_preview`, Redis DB 13, model
  service 18010 in untrained-fallback mode for embeddings, API 18000, worker, standalone web
  13300; mock LLM and mock rule-check backend): `scripts/smoke_phase4.py --web
  http://127.0.0.1:13300` **all 7 checks passed**: 10 files ingested (scanned PDF OCR'd),
  run completed, **58 findings (40 satisfied, 5 violated, 13 needs review)**, 27 LLM calls,
  15,388 mock-estimated tokens, $0, 0 unknown attempts; SSE replay after `Last-Event-ID`.
- Browser (`phase4_ui_check.py`, Playwright 1.64, system Chrome, headless): sign up →
  workspace → upload 10 files → rule packs (model/LLM/cross-document badges) → review with
  pause-on-review → live grid/timeline → awaiting review → audited decision in the report →
  resume → completed → PDF citation opened with line highlight → per-subject/risk tabs →
  mobile width: **all steps passed, no console or page errors**. Screenshots in
  `/dev/shm/cw-e2e/shots` (inspected: run grid/timeline with readable labels, decision with
  reasoning steps, exact PDF line highlight).
- Bounded real-model run (`real_model_check.py`, preview DB, live fp32 model service over
  HTTP only, mock LLM, 3 documents): model `rule-check-deberta_v3_base-0e28e064-e9245464-paa1268cf`
  fp32, **tau 0.3 read from the artifact**, 6 full-document model decisions (confidence
  0.907–0.928), 5 verified by the model re-checking only the cited spans, the
  `not_mentioned` one `needs_review`, **0 escalations**, 4.9 s.
- Live services were not restarted, rebuilt or reconfigured. No provider key was read.

Implementer runs, second gate (Claude Code, 2026-10-26, after the two late concurrency
findings and the verifier negatives; logs copied to `kaggle-recovery/logs/claude-phase4-*gate2*`):

- `crun4.sh --network none -- ruff check . && ruff format --check .`: all checks passed,
  260 files formatted; `mypy packages apps conftest.py scripts evals ml seed`: **no issues
  in 231 source files**.
- New regressions, `pyt.sh` (test DB `compliance_test`, Redis 15): **7 passed in 42.65 s**
  (`apps/worker/tests/test_reviews_job.py` incl. the new stale in-progress sweeper test,
  `test_durability.py::test_killed_worker_is_recovered_promptly_under_fencing`,
  `test_review_flow.py::test_override_locks_the_run_first_and_loses_to_a_concurrent_resume`,
  `test_verifier.py` (2 tests)).
- Each new test was also run against the pre-fix code (the old file mounted read-only over
  the fixed one in the container) and **fails there**: sweeper (unit and end to end)
  `{'requeued': 0} != {'requeued': 1}`; override with the old order "override locked the
  finding before the run"; override locking the run but reading a cached status (no
  `populate_existing`) "DID NOT RAISE InvalidOperationError"; verifier with overlap instead
  of containment `('satisfied', True) != ('needs_review', False)`; citation `verified`
  ignoring `supported` `(False, True) != (False, False)`.
- Affected suites (`packages/agents/tests`, `apps/api/tests/test_reviews_api.py`,
  `apps/worker/tests`): **86 passed in 244.87 s**. The full application suite was not
  rerun by the implementer this gate (the manager's independent full run is in progress
  on its own DB/Redis); ML tests were not rerun (unchanged code).
- Preview refreshed (API/worker/model service restarted on the new code; web unchanged)
  and `smoke_phase4.py --web http://127.0.0.1:13300`: **all 7 checks passed**, 58
  findings (40 satisfied, 5 violated, 13 needs review), 27 mock calls, $0, 0 unknown.
  The preview worker's `cron:requeue_stale_reviews` ran each minute with `{'requeued': 0}`.
- Live services were not restarted (uptimes unchanged); no provider key was read and no
  external LLM request was made.

Implementer runs, packaging correction (Claude Code, 2026-10-26, after the manager's Docker
packaging blocker; log `kaggle-recovery/logs/claude-phase4-packaging.log`, scripts
`kaggle-recovery/validation-scripts/{context-probe.Dockerfile,verify_wheel.py}`; all build
and install output in `/dev/shm/pk`, nothing on the root disk except a few-KB BuildKit
context):

- Blocker reproduced with the real `.dockerignore` and tree: a `FROM scratch` probe
  (`COPY rule_packs`, `COPY packages/core`, `--output type=local`) failed with
  `"/rule_packs": not found`. Building the core wheel from a context without `rule_packs`
  makes hatchling raise `FileNotFoundError` for the force-include, so the services'
  `uv sync --no-editable` image step would have failed before the packs could ship.
- Fix: `rule_packs` removed from `.dockerignore` (the only change outside tests and this
  document; `packages/core/pyproject.toml` and the loader are unchanged). The dependency
  layer is built from the manifests-only stage, so a pack edit only rebuilds the final
  workspace-install layer.
- After the fix the same probe exports `rule_packs/{README.md,nda_standard.yaml,vendor_confidentiality_review.yaml}`.
  `uv build --offline --wheel` of `packages/core` **from that exported Docker context**
  produced a wheel containing `compliance_core/builtin_rule_packs/` with both YAML packs
  (plus README). A non-editable install straight from the directory (the build path
  `uv sync --no-editable` uses) also carries them.
- Outside the checkout: wheel installed with `--no-deps --target` into a RAM directory,
  then run in a `--network none` container with no `/src` mount, cwd `/tmp`, `python -I`
  (`verify_wheel.py`): `compliance_core` imported from the install, `builtin_pack_dir()` =
  the installed `builtin_rule_packs`, both files byte-identical (SHA-256) to the repository,
  `load_builtin_specs()` (parse + `validate_spec`, including the trained-hypothesis check)
  accepted `nda_standard` v1.0.0 (17 rules, 17 model-supported) and
  `vendor_confidentiality_review` v1.0.0 (7 rules, 3 model-supported), no model-support
  errors, YAML round trip stable: **OK**.
- Regression `packages/core/tests/test_packaging.py` (no DB, no network, no build): every
  wheel `force-include` source of every workspace member must exist and no file under it
  may be excluded by `.dockerignore` (Docker rules: last match wins, `!` re-includes, a
  match on a parent excludes, `**`); the matcher is checked against known exclusions; and
  the force-included directory must be the `compliance_core/builtin_rule_packs` that
  `builtin_pack_dir()` reads and ship the same YAML files the checkout loads. With the
  pre-fix `.dockerignore` mounted over the fixed one it **fails** (`.dockerignore excludes
  force-included rule_packs/README.md`); fixed: `test_packaging.py` + `test_rule_packs.py`
  **11 passed in 0.19 s** (`--noconftest`, `--network none`).
- `crun4.sh --network none`: `ruff check .` and `ruff format --check .` passed (261 files);
  **mypy: no issues in 232 source files**.
- Not run by the implementer: a full Docker image build/compose smoke (left to hosted CI,
  root disk ~2 GB free) and any DB-backed test (the manager is rerunning the second-gate
  cases on `compliance_test`/Redis 15). Live services were not restarted; no key was read
  and no LLM/provider request was made.
- Remaining limitation: building a wheel *via an sdist* (`uv build` without `--wheel`, or
  `pip` from an sdist) still fails, because the sdist does not contain `../../rule_packs`.
  Nothing in the repository, Dockerfile or CI builds core from an sdist; publishing an
  sdist would need the packs moved into the package or added to the sdist target.

Hosted CI failure and fix (Claude Code, 2026-10-26; working tree on top of `ae782cb`, logs
under `/dev/shm/cw/logs/claude-ci-collect-fix-*`, copied to `kaggle-recovery/logs/`):

- **Failure.** Hosted CI run `40341052392` at `ae782cb`: Ruff and mypy passed, the web job
  passed, and `uv run pytest` stopped at **collection** with three
  `ModuleNotFoundError: No module named 'conftest'` errors —
  `apps/api/tests/test_documents.py:18` (`from conftest import TEST_MAX_FILE_BYTES`),
  `evals/retrieval/tests/test_corpus_safety.py:11` and `test_sweep_grid.py:14`
  (`from conftest import TEST_DATABASE_URL`). Because the Python job failed, the Docker
  build/compose smoke did not run.
- **Cause (reproduced).** pytest 9.1.1 registers the root `conftest.py` as
  `sys.modules["conftest"]`, and before importing any other non-package conftest it
  deletes that entry (`_pytest/config/__init__.py`, `_importconftest`). Phase 4 added the
  first nested one, `packages/agents/tests/conftest.py`. Under `--import-mode=importlib` it
  is imported as `packages.agents.tests.conftest`, so no module named `conftest` remains.
  Initial conftests are loaded for every `testpaths` entry, so `--ignore` does not help:
  `test_documents.py` collected together with `packages/agents/tests` fails, while it collects
  on its own. `uv run pytest` runs a console entry point, so `sys.path[0]` is the venv
  `bin` directory and the repo root is not importable either. My local runs missed it
  because `zz_compliance_src.pth` puts `/src` (the repo root) on `sys.path`, so
  `/src/conftest.py` was found as a top-level module (`pyt.sh` collects the same set without
  error).
- **CI-like runner** `pyt_ci.sh` (`/dev/shm/cw/bin`, durable copy in
  `../kaggle-recovery/validation-scripts/` with `ci/`): runs pytest's real console entry
  point (`_pytest.config:_console_main`, as in `entry_points.txt`) from `/ci`. The
  `.pth` adds only the ten package `src` directories, never the repo root, and the default
  cwd is the repo root, as in CI. `ci/syspath_probe` printed the container's `sys.path`:
  repo root absent. On `ae782cb` it reproduces CI exactly: **414 collected, 3 errors**,
  the same three `ModuleNotFoundError`s.
- **Fix.** The three modules no longer import `conftest`. They request the existing
  session `test_settings` fixture (built from the same `TEST_DATABASE_URL` /
  `TEST_MAX_FILE_BYTES`): `test_size_limits` uses `test_settings.max_upload_file_bytes`;
  `test_skip_rule_matches_pgvector` uses `test_settings.database_url`; and
  `test_corpus_safety.py` gets a `server` URL fixture from `test_settings.database_url`.
  Its helpers take that URL, and its three copied `asyncpg.connect` blocks are now one
  `connect()` helper. Assertions are unchanged; import mode, `testpaths` and the
  conftests are unchanged; nothing is skipped.
- **Guard.** Ruff `TID251` (banned-api) bans importing `conftest`, with a message
  pointing to fixtures (`pyproject.toml`). Run on the `ae782cb` versions of the three files,
  it reports each import, so a regression would fail the Ruff step, before tests run.
- **Validation (fixed tree).**
  - CI-like collection (cwd repo root, no args, so `testpaths`): **444 collected, 0 errors**.
  - Changed modules (`test_documents.py`, `evals/retrieval/tests`) on `compliance_test`
    / Redis 15: **37 passed in 6.22 s**.
  - Ruff 0.17.1 (the `uv.lock` version): `ruff check .` and `ruff format --check .`
    passed (261 files).
  - mypy, exact CI arguments (`packages apps conftest.py scripts evals ml`): **no issues
    in 225 source files**; with `seed` added: **no issues in 232**.
  - Full suite, first CI-like run (cwd `/src`): 443 passed, 1 failed. The failure is
    `test_health_and_errors.py::test_readiness_fails_when_dependencies_are_down`
    (`OSError: Read-only file system: '/src/.data'`). That test, unchanged in this fix,
    leaves `storage_dir` at its default `./.data/storage` relative to the cwd. My
    container mounts the source read-only; CI's checkout is writable (`.data/` is
    gitignored). It is a runner artifact, not a CI failure.
  - Full suite, rerun with a writable cwd (`PT_WORKDIR=/tmp`, `-c /src/pyproject.toml
    --rootdir /src`, explicit `testpaths`; the console entry point never adds the cwd to
    `sys.path`). A first attempt (3 failed, 1 error: a `TRUNCATE` deadlock and "cached
    statement plan is invalid due to a database schema change") ran 07:51–07:57 local, while
    the manager's readiness rerun (`codex-phase4-ci-readiness.log`, written 07:52) was using
    the shared `compliance_test`. Its `migrated_database` fixture downgrades and re-upgrades
    the schema, which matches both errors (cause inferred from timing and error signatures).
    The attempt is discarded as contaminated, not counted. Repeated on a
    quiet DB at committed `081089b` with a clean tree: **444 passed in 358.32 s**
    (`claude-ci-collect-fix-full3.log`).
- Not run here: hosted CI (manager pushes), Docker build/compose smoke, ML suites and web
  checks (unchanged by this fix). No provider call, key read, live restart, source-pin
  change, commit or push.

## Manager review checks (Codex)

These are verification requirements, not conclusions that a defect exists yet.

**Concrete review findings to resolve before any real-provider run:**

- `compliance_llm/pricing.py` currently estimates reservations as characters / 3 + 64.
  That is an average-language estimate, not a hard token bound (Unicode and schema/framing
  overhead can exceed it). Reuse the conservative UTF-8-body bound and overhead approach
  already verified in the Phase 3 budget runner, or use a provably safe provider-specific
  bound. Test Unicode, large schemas, maximum output, and a request near the run cap.
- `http_providers.py` currently turns missing/malformed usage into zero with `_int`.
  A successful HTTP response without trustworthy usage must retain an unknown/full
  reservation, not silently count as free. Reject booleans, negative and missing counts.
  Preserve cache creation/read components and their correct prices when present; do not
  price cache writes as ordinary input. Cover these cases with fake transport responses.

- The ORM uses six-decimal USD columns returned as floats. Budget enforcement must
  conservatively round reservations/charges up at the storage boundary or use exact
  money arithmetic. Add a concurrent near-cap test with fractional-microdollar usage;
  rounding must never free budget that was actually consumed.
- Verify deleting or changing a cited source after verification, including during a run
  and before reading a saved report. A missing citation must not leave a finding presented
  as currently verified; retain a clearly historical record or downgrade to needs_review.
- Custom rules must not gain trained-model support merely by setting model_supported=true.
  Verify support against the known trained hypotheses; unseen rules need the LLM path.
- Exercise fan-out with several tasks, one failing branch and resume. The final report
  must wait for all branches, contain unique findings, and avoid rebilling successful steps.
- `run_read()` currently aggregates `Finding.status` directly, while the findings/task
  views use effective statuses after overrides and current-source validation. Ensure the
  live run counters and report totals do not contradict those effective findings after
  an override or a source change/deletion; label any original-machine totals explicitly.
- Model gateway validation: reject contradictory label/status pairs, invalid probability
  dictionaries (missing/extra labels, NaN, negative/out-of-range or non-normalized values),
  inconsistent confidence, duplicate evidence IDs, and missing/invalid metadata tau.
  A malformed model response must not become a verified or high-confidence finding.
- Extraction review: `_check_value()` currently proves only that a date/duration occurs
  in the quote, not that it denotes the claimed role. An effective date must not be
  accepted as expiry, or a confidentiality survival period as the agreement term. Require
  role-specific support or abstain; test these swaps in cross-document term decisions.
  Also render `{entity_types}` in the extraction system prompt (currently passed through
  literally), and check the promised party/file contradiction handling in `link_subject()`.
- Verifier `_exact()` currently accepts any positive overlap with the cited chunk.
  Require the cited range to be contained in that chunk, as its contract says, and test
  a quote that extends outside it. Keep citation-level `verified` truthful when an LLM
  support check endorses only a subset of quote IDs (not every exact quote is supported).
- Runner error handling: `_safe_message()` currently truncates raw exception text and
  `logger.exception` prints the traceback. SQLAlchemy errors can include bound document
  text; other exceptions can include credentials. Persist/log sanitized error codes and
  exception types, not arbitrary exception strings/SQL parameters; test sentinel secrets.
  Explicitly close the graph stream on pause/cancel and verify no background branch
  continues provider work after the runner releases its lease.
- SSE route: `session.rollback()` expires ORM instances, but the route then reads
  `run.workspace_id`. Capture needed scalar IDs before rollback, and exercise the real
  streaming route (not only the event serialization helper) to catch MissingGreenlet.
- Mid-run source changes: cross-document `_entity_citation()` currently slices current
  text at the extracted entity's old offsets without comparing the stored quote/value.
  Reject stale entities before reasoning; otherwise a formerly extracted date can be
  asserted using a newly copied, unrelated exact quote. Test mutation between extraction
  and cross-document reasoning, not only deletion after completion.
- The current flagship test describes six register rows (`16 + 4 + 4 + 6` findings).
  The handoff flagship specifies ten vendors. Expand the actual seeded/browser fixture
  and its assertions to ten; do not mark the ten-vendor criterion met with a smaller demo.
- Role-check follow-up: generic words such as `period`, `until`, or `expires` alone still
  accept “The confidentiality obligation continues for a period of three years” as the
  agreement term, or “The offer expires on January 1, 2099” as agreement expiry. Bind the
  date/duration to the agreement's role or abstain, and cover both examples in tests.
- Frontend event hook: `useRunEvents` currently retains `events` and `cursor` when
  `runId` changes. Reset both for a new run and test a mounted component navigating from
  a high-sequence run to a fresh run (including late responses from the old run). Ensure
  active-to-stopped transitions also leave a truthful connection-status badge.
- Independent visual review: PDF quote and geometry highlight align. The run timeline
  currently shows raw `cd|rule_key|row:<UUID>` identifiers as its main text. Render readable
  subject/rule/status labels in the primary timeline; keep IDs only in optional details.
  Also make the global model-unavailable banner compatible with explicitly enabled mock
  reviews (it currently says automated checks are unavailable while the mock run succeeds).
- Worker crash recovery: `requeue_stale_reviews()` skips an expired-lease run while its
  original ARQ job still reports `in_progress`. A killed worker can leave that lock until
  the long job timeout. Expired run leases should permit a fenced recovery job even in
  that case. Add a sweeper test with an expired running lease and a stale in-progress
  ARQ job, proving timely re-enqueue rather than waiting for the job timeout.
- Override/resume concurrency: `override_finding()` locks the finding, reads the run
  without locking it, then `append_event()` locks the run. Worker transactions lock the
  run first. Use consistent run-first locking and re-check the active status under that
  lock, so a concurrent resume cannot admit an override after execution starts or create
  opposite lock order. Add a concurrent resume/override regression test.
- **Docker packaging blocker found by manager:** `.dockerignore` still excludes the
  entire `rule_packs` directory, but `packages/core/pyproject.toml` now force-includes
  `../../rule_packs` into the core wheel. Docker's full-source bind therefore cannot
  build that wheel or ship the packs. Include the required rule-pack assets in the
  Docker build context and verify installed-wheel loading outside the source checkout.
  Hosted Docker build/smoke must pass; do not restart live services for this validation.

Operational note from manager (2026-10-26): live recovered worker/model-service Python
imports are now pinned through their existing `.pth` bind to the accepted `ccfff28`
source snapshot under `/cache/runtime-phase3/<full-commit>/`. Their previous active-checkout
bind made fresh worker health probes import unfinished Phase 4 code and fail on missing
LangGraph; the old worker process itself continued running. Health check now passes, no
restart occurred. Preserve the pin; isolated tests use their own overlay/path files.
See `SUPERVISION.md` for the operational record before planning any live deployment.

## Implementer responses to the manager review checks (Claude)

Each item is implemented and covered by a test (file in brackets).

**Correction (2026-10-26, second gate).** My earlier statement that every review item was
addressed was wrong: the last two findings (worker crash recovery with a stale ARQ
in-progress lock, and override/resume lock order) were still open in the source, and the
verifier bullet pointed only to the happy-path flagship test. Both findings are now fixed,
the verifier has negative regressions, and each new test was shown to fail on the pre-fix
code (evidence log, second gate).

**Correction (2026-10-26, packaging).** The second-gate wording above was still too broad:
it covered the concurrency findings but not the manager's later Docker packaging blocker
(`.dockerignore` excluded `rule_packs` while the core wheel force-includes it), which was
open in the source at that time. It is now fixed and verified (last bullet below; evidence
log, packaging correction). With that, every manager finding listed above has an
implementer response; hosted Docker build/smoke is still required to confirm the image.

- **Reservation bound** — reuses the Phase 3 rule: UTF-8 JSON body bytes + 4,096 framing
  tokens priced at the highest input-side rate, plus the full output allowance; requests
  above 100k bound tokens are refused; measured usage above a reservation raises
  `budget_invariant` and stops the run. [`test_providers.py::test_reservation_is_a_provable_upper_bound`,
  `test_durability.py::test_usage_above_reservation_stops_the_run`]
- **Untrustworthy usage** — missing, boolean, negative, non-integer or 1-hour-cache usage
  gives `usage=None`; the attempt keeps its full reservation (`usage_source=none`). Cache
  writes/reads are stored separately and priced at cache rates.
  [`test_untrustworthy_usage_is_reported_as_missing`, `test_cache_components_are_priced_at_cache_rates`,
  `test_unknown_and_usageless_attempts_keep_their_reservation`]
- **Money** — integer micro-dollars everywhere; every conversion rounds up; reservations are
  serialised by the run row lock. [`test_money_rounds_up_and_is_exact`,
  `test_parallel_reservations_never_exceed_the_cap`, `test_budget_cap_with_fractional_prices_under_parallelism`]
- **Deleted/changed sources** — citations keep `document_filename` and survive deletion
  (`SET NULL`); every read re-checks quotes at their offsets, so changed or deleted sources
  present the finding as `needs_review` (`source_changed`), also in run totals and the
  report; mid-run changes make stale entities unusable. [`test_changed_or_deleted_sources_are_no_longer_presented_as_verified`,
  `test_source_changed_mid_run_does_not_lend_new_text_to_old_facts`]
- **Custom rules and model support** — server-side verbatim check against the trained
  hypotheses. [`test_rule_packs.py`, `test_rule_pack_api`]
- **Failing branch + resume** — one branch fails; resume finishes the rest, each step is
  billed once, findings are unique, the report waits for all branches.
  [`test_failing_branch_then_resume_without_rebilling`]
- **Run counters** — `findings_by_status` now uses the effective statuses (overrides,
  re-checked sources); machine and post-verification totals are reported separately and
  labelled. [`test_interrupt_override_and_resume`, `test_changed_or_deleted_sources...`]
- **Model gateway validation** — label/status agreement, exactly three finite normalised
  probabilities, confidence equals the label's probability and is the maximum, unique and
  known evidence ids, production kind, valid metadata tau (else the model is not used).
  [`test_malformed_model_answers_are_rejected`, `test_nan_probabilities_are_rejected`,
  `test_model_without_valid_tau_is_not_used`]
- **Extraction roles** — role cues plus agreement binding: an offer's expiry or a
  confidentiality/survival period is rejected; effective and expiry dates cannot be swapped;
  `{entity_types}` is rendered in the system prompt (unfilled placeholders raise); the
  register link is `ambiguous` when the listed file names other parties and another document
  names the vendor; merely similar file names are never linked. [`test_units.py`,
  `test_mis_roled_dates_are_not_used_for_the_term_decision`, `test_register_linking_is_conservative`]
- **Verifier** — the cited range must be contained in its chunk; citation-level `verified`
  requires the citation itself to be judged supporting. Negative regressions: an evidence
  quote stretched 40 characters past its chunk (still the document's exact text, so only
  the containment check can catch it; injected while the run is paused after rule check)
  makes the finding `needs_review` with "not contained in the cited chunk", the citation
  `exact_match`/`verified` false, while an untouched finding still verifies; an LLM support
  check that endorses only `Q1` of two exact quotes leaves the finding verified but marks
  only the endorsed citation `supported`/`verified` (DB rows and API).
  [`test_verifier.py`; happy path in `test_flagship.py`]
- **Runner errors** — persisted/logged errors carry a code and the exception type (own
  messages only for our own exception classes); logs contain code locations, never
  messages or tracebacks with values; a sentinel secret raised by a provider appears in no
  run row, event or log. A control watcher cancels in-flight work on cancel; the stream is
  closed explicitly and no provider call starts after the job returns.
  [`test_failing_branch_then_resume_without_rebilling`, `test_cancel_interrupts_inflight_provider_calls`]
- **SSE route** — scalar ids are captured before releasing the session; the real streaming
  route is exercised with `Last-Event-ID` and a live event. [`test_events_pagination_and_real_sse_stream`]
- **Ten vendors** — the synthetic flagship has ten register vendors (nine contract files,
  one missing, a versioned ambiguous pair, an expired term, an explicit expiry date, the
  scanned PDF and the injection NDA); tests, smoke and browser use all ten (58 findings).
- **Event hook** — state is scoped to the run id (fresh list and cursor, late old-run
  responses dropped) and the status becomes `idle` when streaming stops.
  [`use-run-events.test.tsx`]
- **Timeline and banner** — the timeline shows "rule — subject · status · verified" with raw
  ids only in the tooltip; the model banner explains what reviews use instead
  (`review_rule_check_backend`), so it no longer claims checks are unavailable when the mock
  backend is enabled. [`reviews.test.tsx`, `model-status-banner.test.tsx`, `test_messages_reflect_the_review_backend`]
- **Worker crash recovery** — for a `running` run whose lease has expired, the sweeper no
  longer treats the original job's ARQ `in_progress` lock as pending (a killed worker
  leaves it until the job timeout); only a job still `queued`/`deferred` defers recovery.
  The recovery job id is per run and minute, and it is counted only when actually
  enqueued. Safety comes from the lease: the recovery job can only claim an expired lease,
  and the dead holder's token can no longer renew or write. Queued runs keep the old rule.
  Tests: stale in-progress lock with an expired lease is re-enqueued at once, a merely
  queued original is left to start, no duplicate within the minute
  [`test_reviews_job.py::test_sweeper_recovers_an_expired_lease_despite_a_stale_in_progress_job`];
  end to end, a worker cancelled inside a provider request (lease, ARQ lock and its in-flight
  attempt left behind), lease expired, sweeper enqueues, the recovery job's
  `run_review` completes the run, the dead token cannot renew, no attempt stays
  `reserved`, no step is billed twice
  [`test_durability.py::test_killed_worker_is_recovered_promptly_under_fencing`].
- **Override/resume concurrency** — `override_finding()` now locks the run first (like the
  worker, `control_run()` and `append_event()`), then the finding, and checks the active
  status under the run lock; locked reads use `populate_existing` so the status is the
  one read under the lock, not a cached one. Test: (1) while another transaction holds the
  run lock, a waiting override holds no finding lock (that transaction takes the finding
  with `NOWAIT`), so no opposite lock order; (2) a real `control_run(resume)` that gets the
  run lock first wins and the override queued behind it is refused (`InvalidOperationError`, HTTP 409), no audit
  row is written, and events are ordered override → resume
  [`test_review_flow.py::test_override_locks_the_run_first_and_loses_to_a_concurrent_resume`].
- **Docker packaging** — `rule_packs` is no longer excluded from the Docker build context,
  so the core wheel's `../../rule_packs` force-include resolves in the image build. Verified
  with the real `.dockerignore` (context probe), a wheel built offline from the exported
  Docker context, and that wheel installed and loaded outside the checkout (both packs
  byte-identical, parsed and validated, including the trained-hypothesis check). Regression:
  force-included files must survive `.dockerignore` and match the packs the loader reads; it
  fails on the pre-fix file [`packages/core/tests/test_packaging.py`]. Hosted CI Docker
  build/compose smoke (which seeds the built-in packs) remains the image-level check; an
  sdist-based core build is still unsupported (see evidence, packaging correction).
- **Hosted CI collection failure** (run `40341052392`, `ae782cb`) — three test modules
  imported constants with `from conftest import …`. That only worked while the root conftest
  was the only one and the repo root was on `sys.path`; the new
  `packages/agents/tests/conftest.py` evicts `sys.modules["conftest"]`, and `uv run pytest`
  does not put the root on the path. The modules now use the `test_settings` fixture, and
  Ruff `TID251` bans `conftest` imports. Validated with a CI-like console-entry runner with
  no repo root on `sys.path` (evidence log, hosted CI failure and fix). Green hosted CI,
  including the Docker smoke it skipped, is still required.
