# Kaggle recovery — Phase 3 local validation (2026-10-25)

Follows `KAGGLE_RECOVERY_FIXES.md`. Local only: no provider/API call, no key read, no
retraining, upload, notebook change, migration of a live DB, deploy, commit or push by the
validator. The live `compliance-*` stack was inspected read-only and not changed. Every
result below was produced on the final source (after the seed change), unless marked
otherwise.

Validated source: during this run the tracked changes were committed outside this session
as `0152768` ("Guard Phase 3 recovery and serialize demo seeding", branch
`fix/phase3-kaggle-recovery`, pushed). The worktree equals that commit (`git diff HEAD`
empty), and no source was edited after the gates below. They therefore apply to `0152768`.
The supervisor subsequently opened draft PR
https://github.com/JCHETAN26/compliance/pull/1. Hosted CI run
https://github.com/JCHETAN26/compliance/actions/runs/40206201837 passed all three jobs
(Python, web, Docker build/clean-stack smoke) for `0152768`.

## Independent supervisor verification

On `0152768`, Codex reran the seeding and model-service suites against isolated test
Postgres/Redis: **44 passed in 42.07 s**. Log:
`../kaggle-recovery/logs/codex-seed-model-tests.log`.

An independent HTTP check sent eight concurrent requests to the isolated production-fp32
service: all returned 200 with valid labels and source span IDs, multi-request batch
counts increased, and blank/duplicate-span inputs returned 422. The absent cascade tau
was correctly represented by `escalate: null`. Record:
`../kaggle-recovery/logs/codex-http-check.json`.

These checks do not establish real Anthropic baseline or cascade performance. The
approved one-request provider diagnostic still awaits a locally configured private key.

## Seed concurrency fix (`seed/src/compliance_seed/job.py`)

Problem: the worker's automatic `seed_demo` job and `python -m compliance_seed` (CI,
`make seed`) can run at once on a clean start. Each did check-then-create with commits in
between, so the second seed got `ConflictError` on the demo user, or, if the user already
existed, created a second demo workspace.

Fix: `run_seed` takes `pg_advisory_xact_lock(SEED_ADVISORY_LOCK_ID = 7_362_918_402)` in a
separate transaction that touches no table, and holds it while `_seed_database` runs. The
services commit after each step, so the next seed sees the committed user, workspace and
documents. The lock ends with that transaction (on success or error). It uses one extra
pooled connection per seed (pool size 10).

Test: `seed/tests/test_seed.py::test_concurrent_seeds_create_one_demo_workspace[False|True]`
runs two `run_seed` calls concurrently. A rendezvous wrapper on `register_user` and
`create_workspace` holds each caller up to 2 s until both arrive. That makes the race
deterministic without the lock; with the lock, the timeout releases the single caller.
It asserts 1 user / 1 workspace / 11 documents, one `workspace_created`, the same document
ids for both seeds, and that each new document is queued exactly once.

| Check | Result |
|---|---|
| Seed tests, fixed source | **7 passed** (concurrent cases 4.9 s / 2.9 s) |
| Same tests with the lock line removed (mutation; repo untouched, mutated file mounted over `/src`) | **2 failed** as intended: `ConflictError` (user absent); `[True, True]` workspaces created (user present) |
| Same race before the fix (previous session) | failed the same way: `seed-race-before-fix.log` |
| 3 concurrent real `python -m compliance_seed` processes, isolated test DB | all exit 0; users 1, workspaces 1, members 1, documents 11 (11 distinct SHA-256), 11 ingest jobs + queue key in Redis db 14; tables truncated and db 14 flushed afterward |

## Gates run on the final source

| Gate | Result | Log (`../kaggle-recovery/logs/phase3-local/`) |
|---|---|---|
| `ruff check` / `ruff format --check` (0.17.1) | all checks passed; 200 files formatted | `lint-type-final.log` |
| `mypy --strict` (2.4.0), `make typecheck` set (`packages apps conftest.py scripts evals ml`) | no issues, 172 files | same |
| same plus `seed` (the Makefile omits it) | no issues, 178 files | same |
| Application suite (all `testpaths`, real pinned models, test Postgres/Redis) | **336 passed** in 145 s (334 before + 2 new) | `pytest-app-final.log` |
| `ml/tests` in container | 113 passed; 3 failed only because the image has no `git` | `pytest-ml-final.log` |
| the 2 git-dependent modules on the host | 44 passed (includes those 3) | `pytest-ml-git-host-final.log` |
| ML CPU smoke (real torch → ONNX → service) | **14 passed** in 136 s | `pytest-ml-smoke-final.log` |
| `git diff --check` | clean | — |

## Recovered fp32 artifact over HTTP

- Pinned downloads are in the SSD `.cache` (git-ignored): `models/BAAI--bge-small-en-v1.5--5c38ec7c405e`
  (140 MB), `models/cross-encoder--ms-marco-MiniLM-L6-v2--233902d25c44` (94 MB),
  `datasets/contractnli/contract-nli.zip`, `ml/` (DeBERTa tokenizer, smoke). Re-verified
  **offline** (`--network none`) by the project's own `fetch` commands:
  all three report "verified" (`cache-verify-offline.log`).
- `.cache/models/rule-check/current` is an exFAT copy (no symlinks) of
  `kaggle-recovery/fp32-recovered/deberta_v3_base`. SHA-256 is identical for all 4 files, and
  the recovered source is unchanged (`recovered-now.sha`, `current-now.sha`). `load_artifact`:
  fp32 loads; int8 is refused ("no int8 ONNX file … set RULE_CHECK_VARIANT=fp32").
- Fresh isolated service `cw-ms-fp32-check` (127.0.0.1:18010, host network, read-only
  source and cache, `RULE_CHECK_VARIANT=fp32`, 4 threads, batch 64 rows / 5 ms). `/health`:
  rule_check ready, mode model. `/metadata`: version
  `rule-check-deberta_v3_base-0e28e064-e9245464-pf1e1842f`, production, fp32, tau **null**.
- `http_check.py`, 8 seeded DEV pairs (12–65 spans each):
  - Labels and evidence sets vs the training run's saved predictions: **1.0 / 1.0**.
  - The same 8 requests concurrently: identical labels, evidence and probabilities
    (max diff 0.0).
  - Batcher: 24 batches, 624 rows, max 64 rows, **4 multi-request batches**, 0 rejected,
    0 failed.
  - Error contract: duplicate span ids 422, blank text 422, oversize 413, small request 200.
  - Latency, CPU fp32: 1.6–17.8 s per request. Sequential total 78 s; concurrent wall time
    94 s. Batching merges requests but gives no throughput gain on this CPU.
  - Result file: `http_check-final.json`; the previous session's run gave the same result
    (`http_check.json`).

## Live stack: worker crash loop (diagnosed, NOT changed)

The live stack was started from the old checkout `/home/grv22/projects/compliance`, which no
longer exists. Docker re-created its bind source `/home/grv22/projects/compliance/.cache` as
an empty root-owned directory.

- `compliance-worker-1` (`unless-stopped`) has crash-looped 1,419 times. The error is
  `ModelUnavailableError: BAAI/bge-small-en-v1.5@5c38ec7… is not in /cache/models`
  (`ARTIFACT_AUTO_DOWNLOAD=false`).
- `compliance-model-service-1` has the same mount. It is healthy only because it loaded the
  models before the files went away, and **will fail the same way on its next restart**.

Safe recovery (needs sudo and the user's approval). It is reversible, uses no root-disk
space and changes no container or compose file. It has not been tested here:

```bash
sudo mount --bind -o ro "/media/grv22/Extreme SSD2/project/compliance/.cache" \
  /home/grv22/projects/compliance/.cache
# the worker's next automatic restart picks it up; check with
docker logs --tail 20 compliance-worker-1
# undo: sudo umount /home/grv22/projects/compliance/.cache   (not persistent across reboot)
```

Caveats:
- When the live model-service restarts with this mount, `current` is the fp32-only artifact.
  The stack default `RULE_CHECK_VARIANT=int8` then refuses it, so rule-check is unavailable
  until the stack is recreated with `RULE_CHECK_VARIANT=fp32`. That is a stack change for
  Phase 4.
- The worker has `SEED_DEMO=true`. If the live DB has no demo workspace it will seed (the
  dataset is in the cache).
- Not recommended now: recreating the worker from the SSD checkout
  (`docker compose -p compliance … up -d --no-deps --no-build worker` with
  `ARTIFACT_CACHE_SOURCE=./.cache`). That changes the live stack and carries drift from the
  old checkout's compose/.env.
- Copying only the 140 MB embedding model into the old path also works. It costs root disk
  space (~900 MB free).

## Unfinished blockers (Phase 3 is not accepted yet)

1. LLM baseline: the cause of the 400s is unknown. Codex runs the separately approved
   one-call diagnostic after the user configures the key. Then a fix and a full DEV/TEST run.
2. Cascade tau: re-run `ml.eval.evaluate` after a usable baseline. Until then no tau is
   served (verified above).
3. int8 is still collapsed. Repair it (choose on DEV) or adopt fp32 as the CPU build. CPU
   fp32 latency (above) matters for Phase 4.
4. Live worker recovery: awaiting approval (above).
5. Hosted CI now passes, including OpenAPI drift checking and the web production build
   (see supervisor record above). Real-provider acceptance remains outstanding.

## Reproducing (exact commands)

Environment: helper scripts are saved in `../kaggle-recovery/validation-scripts/`. The live
copies are in `/dev/shm/cw/bin` (tmpfs; lost on reboot). `crun.sh` runs a throwaway
`compliance/model-service:dev` container with:
- the SSD source mounted read-only on `PYTHONPATH`;
- a lock-pinned dev overlay (ruff, mypy, pytest, …) at `/dev/shm/cw/overlay`;
- a 3 GB tmpfs `/tmp`, 8 GB memory and 512 pids.

Rebuild the overlay if `/dev/shm` was cleared (≈208 MB):

```bash
mkdir -p /dev/shm/cw/bin && cp ../kaggle-recovery/validation-scripts/{crun.sh,zz_compliance_src.pth,http_check.py,seed_race_procs.sh} /dev/shm/cw/bin/
UV_CACHE_DIR=/dev/shm/cw/uvcache timeout 300 uv pip install --target /dev/shm/cw/overlay --no-deps \
  --python-version 3.11 --python-platform x86_64-manylinux_2_36 \
  -r ../kaggle-recovery/validation-scripts/overlay-req.txt; rm -rf /dev/shm/cw/uvcache
```

From the repo root (`SRC` = its absolute path), with the isolated test services
(`docker compose -f docker-compose.test.yml up -d --wait`, ports 55432/56379):

```bash
set -o pipefail; SRC="$PWD"; C=/dev/shm/cw/bin/crun.sh
T="-e TEST_ARTIFACT_CACHE_DIR=/cache -e TEST_DATABASE_URL=postgresql+asyncpg://compliance:compliance@127.0.0.1:55432/compliance_test -e TEST_REDIS_URL=redis://127.0.0.1:56379/15"
PYT="python -m pytest -c /src/pyproject.toml --rootdir /src -p no:cacheprovider --basetemp=/tmp/pt -q"
# lint + types
timeout 1300 $C --network none -- sh -c 'ruff check --no-cache .; ruff format --no-cache --check .; python -m mypy --cache-dir /tmp/mypy packages apps conftest.py scripts evals ml seed'
# seed tests, then the no-lock mutation (expect 2 failed)
timeout 900 $C --network host -w /tmp -v "$SRC/.cache":/cache:ro $T -- timeout 850 $PYT /src/seed/tests
python3 - seed/src/compliance_seed/job.py /dev/shm/cw/job_nolock.py <<'EOF'
import sys
s = open(sys.argv[1]).read()
a = '''            await lock_session.execute(
                text("SELECT pg_advisory_xact_lock(:lock_id)"), {"lock_id": SEED_ADVISORY_LOCK_ID}
            )
'''
assert s.count(a) == 1
open(sys.argv[2], "w").write(s.replace(a, ""))
EOF
timeout 900 $C --network host -w /tmp -v "$SRC/.cache":/cache:ro -v /dev/shm/cw/job_nolock.py:/src/seed/src/compliance_seed/job.py:ro $T -- timeout 850 $PYT /src/seed/tests -k concurrent
# 3 real CLI seed processes against the test DB (truncates the test tables before and after)
timeout 600 $C --network host -w /tmp -v "$SRC/.cache":/cache:ro -v /dev/shm/cw/bin/seed_race_procs.sh:/tmp/seed_race_procs.sh:ro -- timeout 550 sh /tmp/seed_race_procs.sh 3
# application suite (testpaths from pyproject.toml)
TP=$(~/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/bin/python3.11 -c 'import tomllib,sys;print(" ".join("/src/"+p for p in tomllib.load(open(sys.argv[1],"rb"))["tool"]["pytest"]["ini_options"]["testpaths"]))' pyproject.toml)
timeout 2700 $C --network host -w /tmp -v "$SRC/.cache":/cache:ro $T -- timeout 2600 $PYT $TP
# ML unit tests (git-dependent ones: run on the host as in KAGGLE_RECOVERY_FIXES.md, TMPDIR on /dev/shm)
timeout 1300 $C --network none -w /tmp -v "$SRC/.cache":/cache:ro -e TEST_ARTIFACT_CACHE_DIR=/cache -- timeout 1200 $PYT /src/ml/tests
# ML CPU smoke (writes tokenizer files into the SSD .cache/ml)
timeout 2000 $C --network host -w /tmp -v "$SRC/.cache":/src/.cache -e HF_HOME=/tmp/hf -- timeout 1900 $PYT /src/ml/smoke/tests
# fp32 service + HTTP check
timeout 60 $C -d --name cw-ms-fp32-check --network host --log-driver json-file --log-opt max-size=5m --log-opt max-file=1 \
  -v "$SRC/.cache":/cache:ro -e APP_ENV=development -e ARTIFACT_AUTO_DOWNLOAD=false -e RULE_CHECK_VARIANT=fp32 \
  -e RULE_CHECK_THREADS=4 -e RULE_CHECK_MAX_BATCH_ROWS=64 -e RULE_CHECK_MAX_WAIT_MS=5 -e EMBEDDING_THREADS=2 -e RERANK_ENABLED=false \
  -e DATABASE_URL=postgresql+asyncpg://compliance:compliance@127.0.0.1:55432/compliance_test -e REDIS_URL=redis://127.0.0.1:56379/14 \
  -- env ARTIFACT_CACHE_DIR=/cache STORAGE_DIR=/tmp/storage DATA_DIR=/tmp/data \
  uvicorn --factory compliance_model_service.main:create_app --host 127.0.0.1 --port 18010 --no-access-log
B="$SRC/../kaggle-recovery/claude-tmp/bundle-data"
timeout 2400 $C --network host -v "$B":/bundle-data:ro -v /dev/shm/cw/bin:/cwbin:ro -v /dev/shm/cw/logs:/cwlogs -- \
  timeout 2300 python /cwbin/http_check.py http://127.0.0.1:18010 /bundle-data/data \
  /bundle-data/runs/deberta_v3_base/predictions_dev.jsonl.gz 8 /cwlogs/http_check-final.json
docker rm -f cw-ms-fp32-check   # cleanup
```

Note: `crun.sh` unsets `ARTIFACT_CACHE_DIR`, so commands without `env ARTIFACT_CACHE_DIR=…`
after `--` read the cache as `/src/.cache`. That is the same SSD directory, read-only.

## Runtime recovery and real provider check (2026-10-25, later)

This supersedes the "NOT changed" live-stack section and blocker 4 above.

- Runtime recovery was applied by Codex with the no-sudo kit
  `../kaggle-recovery/runtime-recovery/` (log `../kaggle-recovery/logs/runtime-recovery-apply.log`).
  The worker and the model-service were recreated, using the baseline image IDs, `/cache` and
  `/src` read-only from the SSD, and `RULE_CHECK_VARIANT=fp32`. Restart these two services only
  through `cd ../kaggle-recovery/runtime-recovery && python3 -I rr.py compose ...`. A plain
  `docker compose up` reapplies the int8/auto-download defaults. Result: all 7 local
  services are healthy, and worker and model-service have 0 restarts. `/metadata` reports
  `rule-check-deberta_v3_base-0e28e064-e9245464-pf1e1842f`, production, fp32, tau `None`.
- Live HTTP check (`../kaggle-recovery/logs/codex-live-recovery-check.json`): 4 concurrent real
  requests, all answered (`entailment`), `escalate: null`, 0.24 s. Batcher: 2 multi-request
  batches, 8 rows, max 4, 0 rejected/failed.
- Hosted CI for `dcc1e2f`: run 40207127703, success.
- One real Anthropic request (`../kaggle-recovery/provider-diagnostic/result.json`; run once,
  `REQUEST_STARTED` prevents reruns). Pair `dev:3:nda-1`, `claude-sonnet-5-5`, effort low,
  `max_tokens` 4096, SDK 2.1.0, official endpoint, 0 retries, 1 attempt, request hash
  `c8be3bb4…`, data `693b452e…`.
  - Outcome: HTTP success, `end_turn`, 2.39 s. Request `req_011CgPDTv6uV55FbtP3mg6N3`.
  - Answer: valid JSON. Label `entailment` = gold. Evidence S1, S2 (segments 16, 17; 16 is
    gold). No invalid ids.
  - Usage: input 730, cache write 603, cache read 0, output 27. Cost $0.003237 at the pinned
    pricing ($2/$10/$2.50/$0.10 per MTok, retrieved 2026-10-21).
  - The current production request works. This record does not identify the cause of the
    earlier HTTP 400s in the Kaggle run, and one request is not a full-baseline acceptance.
- Codex's rough full-run estimate (`../kaggle-recovery/logs/full-baseline-cost-estimate.json`):
  3128 pairs, about $8.3–25.8 depending on output length. It is not a bound.
- Prepared, NOT run: a budget-bounded full DEV+TEST runner, `../kaggle-recovery/provider-baseline/`
  (README with the invocation and the offline evaluation steps that follow). It has a $30
  cumulative hard cap with a durable reservation ledger and uses the production
  `run_pairs`/`save_system` path. 9 offline fake-client checks passed.

Phase 3 remains unaccepted. Outstanding: the full LLM baseline (1037 DEV + 2091 TEST), then
`ml.eval.evaluate` for the DEV-selected cascade tau and serving it, int8 (or adopting fp32), and
final review.
