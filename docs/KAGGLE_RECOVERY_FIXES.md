# Kaggle recovery fixes — 2026-10-25 (uncommitted, for review)

Follows `KAGGLE_RECOVERY_STATUS.md` (that file is unchanged). No training, provider/API
call, upload, remote-notebook change, deploy, commit or push was made. The three completed
runs were not retrained.

## What is confirmed and what is not

| Finding | Status |
|---|---|
| All 3,128 LLM requests (1,037 DEV, 2,091 TEST) got HTTP 400 | Confirmed (saved metadata/logs) |
| The rejection is systemic, not per prompt | Confirmed: every request across varied prompts, mean latency 0.30 s (fast server-side rejection) |
| The request was sent (not a local SDK error) | Confirmed: `anthropic==2.1.0` knows `output_config.effort`/`format` and `claude-sonnet-5-5`; an unknown keyword would be a local `TypeError`, not HTTP 400 |
| **Why** the API rejected it | **Not established.** The body was discarded. Hypotheses, unranked by evidence: account state (e.g. credit balance/workspace restriction, also reported as 400 `invalid_request_error`), or a parameter/model combination rejected by the API (`effort`, JSON-schema output, prompt-caching block). The next bounded run's `error_detail` will say which |
| Cascade tau 0.3 is invalid | Confirmed: selected against LLM-only DEV accuracy 0.0; the artifact manifest had been refreshed to serve it (`…-pf1e1842f` → `…-paa1268cf`, only tau changed) |
| int8 export collapsed | Confirmed: export DEV parity label agreement 0.083, top-k 0.0 (recorded in the manifest); TEST macro-F1 .063 |
| fp32 export is faithful | Confirmed: parity max 1.2e-5; TEST macro-F1 .8128 vs .8138 PyTorch; local spot check below |
| Root cause of the int8 collapse | Hypothesis only: dynamic quantization of DeBERTa-v3 (MatMul + Gather/embedding). Not investigated; any repair must choose on DEV |

## Changes

- `ml/baselines/llm.py`: each rejected attempt stores a sanitized `error_detail` (status, API
  error type, message truncated to 300 chars with `sk-ant-…`, bearer and `x-api-key` values
  redacted, request id; never the prompt or headers). The first pair is a one-request
  preflight; a non-retryable 4xx there, or 10 consecutive ones later, stops all remaining
  requests (exit 3, no predictions written, attempts kept in the cache). Retryable statuses
  (408/409/429/5xx/529) and isolated per-pair rejections do not stop the run. The summary
  lists distinct `error_diagnostics`.
- `ml/eval/cascade.py`, `ml/eval/evaluate.py`: `llm_usability` (≥ 95 % answered on each
  split). An unusable LLM baseline is excluded from the table, comparisons and cascade, with
  the reason in `baseline_notes`; tau falls back to the labelled provisional model-only
  rule. `select_tau` raises `UnusableLLMBaselineError` for LLM-only DEV accuracy outside
  (0, 1]. `inference_params.json` records `llm_baseline_usable`; `cascade.json` records
  `llm_baseline`.
- `ml/export/serving.py` (new; moved out of `export.py` so recovery needs no torch):
  a tau is served only when `tau_provisional` is false **and** `llm_baseline_usable` is
  true. Older parameter files (the recovered one) therefore serve no tau.
- `rule_check/artifact.py`: production int8 is refused for serving unless the manifest's
  export parity shows label agreement ≥ 0.9 and top-k agreement ≥ 0.75 (smoke artifacts are
  not gated). The gate fails closed: a missing, non-numeric (string, `None`), boolean, NaN,
  infinite or out-of-[0, 1] agreement value is refused as "incomplete or malformed"; an int
  `1` is accepted. The error names `RULE_CHECK_VARIANT=fp32` when an fp32 file exists. New `verify_artifact` checks integrity only (any present
  variant) for tools that measure or move artifacts.
- `ml/export/export.py`: reports `int8_ok`; warns loudly on failure; refuses `--drop-fp32`
  for production when int8 fails. Post-write/refresh verification uses `verify_artifact`.
  `eval_onnx.py` and `bench.py` likewise (they must still be able to measure a bad int8).
- `ml/export/recover_fp32.py` (new): fp32-only artifact from a verified export with
  copies (never links), after checking recorded fp32 parity and the full-split ONNX
  evaluation (same model version or an earlier one of the same files in the manifest's
  refresh history; fp32 macro-F1 drop ≤ 0.01). Keeps tau only with valid `--params`.
  Malformed evidence raises `RecoveryError` before anything is written: a missing,
  non-numeric, boolean, NaN, infinite or overflowing (huge int) fp32 parity difference, a
  negative one, a macro-F1 outside [0, 1], or an evaluation file that is unreadable, not a
  JSON object, has a non-string `model_version` or lacks the score sections. Source
  untouched. It writes into a fresh uniquely named staging directory
  (`tempfile.mkdtemp`, `.<out>.partial-*`) and removes only that one on failure; an existing
  directory such as the old fixed `.<out>.partial` name is never reused or deleted.
- `ml/colab/bundle.py`: install/preflight use `verify_artifact`, log servable variants, and
  fall back to a verified copy for `current` where symlinks fail (exFAT); a previous copy is
  renamed aside, not deleted. The new link/copy is staged in a fresh uniquely named
  `.current-new-*` directory (`mkdtemp`); a pre-existing directory of the old fixed
  `.current-new-<pid>` name is left untouched.
- Tests: `ml/tests/test_recovery_guards.py` (23 test functions; three are parametrized over
  malformed values; staging preservation is checked for both recovery and install). Fixture
  updates: the service test's
  production manifest gains a passing parity record; the smoke refresh test sets
  `llm_baseline_usable=True`. `ml/README.md` documents the guards.

## Checks run (exact outcomes)

Environment: uv-managed CPython 3.11.13 (already on the host) with a pip `--target`
directory on the SSD (`kaggle-recovery/env/site311` plus `site311-pure.zip`; venvs and uv's
cache need symlinks, which exFAT refuses). No torch/transformers.

**Test scratch must not be on the SSD.** exFAT ignores `chmod` and refuses symlinks, so
permission/symlink-sensitive tests misbehave there: `test_log_failure_stops_the_child_and_raises`
relies on `chmod(0500)` to stop an endlessly-printing child, which never stops on exFAT.
Run pytest with the source/env on the SSD but `TMPDIR` and `--basetemp` on Linux `/tmp`
(or `/dev/shm`), under an overall `timeout`.

- **Withdrawn:** an earlier version of this file claimed "73 passed" for the six modules
  below. That claim has no saved log and does not hold: test runs using SSD scratch were
  stopped by the reviewer because of runaway output, and the partial log
  (`kaggle-recovery/logs/codex-independent-tests.log`) shows failures. That was the
  environment, not the implementation (see next item).
- **Independent run (Codex), after the malformed-value and staging corrections:**
  `ml/tests/test_recovery_guards.py`, `test_metrics.py`, `test_training_inputs.py`,
  `test_io.py`, `test_kaggle_notebook.py`, `test_notebook.py` with Linux `/tmp` scratch →
  **102 passed in 15.52 s**. Log: `kaggle-recovery/logs/codex-independent-tests-linux-tmp.log`
  (the log holds pytest's output only, not the command line). This includes the real
  `anthropic==2.1.0` `BadRequestError` test.
- Mutation check (earlier pass, before the corrections; not re-run): with HEAD's `llm.py`
  6 of the new tests failed; with HEAD's `cascade.py`+`evaluate.py` 3 failed.
- Lint: `ruff check`, `ruff format --check` (0.17.1) and `mypy --strict` (2.4.0) were run in
  the earlier pass (reported clean apart from missing `torch`/`fastapi`/`sklearn` stubs) but
  **not re-run after the corrections**: ruff and mypy are no longer installed in the env and
  were not reinstalled. As a stand-in, all 13 changed Python files compile (no syntax errors,
  checked without writing bytecode); lines over 100 characters are only unsplittable
  strings/comments, and the repo's ruff config ignores E501.

The following real-artifact checks ran in the earlier pass, before the corrections, and were
not re-run (no further artifact writes or inference were made):

- Real recovered artifact: `load_artifact(int8)` refused ("label agreement 0.083 (min 0.9),
  top-k agreement 0.000 …; set RULE_CHECK_VARIANT=fp32"); `load_artifact(fp32)` loads; the
  recovered `inference_params.json` serves tau None.
- `python -m ml.export.recover_fp32 --source …/artifact-deberta_v3_base --out
  kaggle-recovery/fp32-recovered/deberta_v3_base --quantization-eval
  …/quantization_eval.json --params <bundle results/inference_params.json>` → version
  `rule-check-deberta_v3_base-0e28e064-e9245464-pf1e1842f`, fp32 only, tau None, fp32
  SHA-256 `8e32d88a…4a74` identical; source artifact hashes re-verified unchanged.
- fp32 spot check (`kaggle-recovery/claude-tmp/scripts/fp32_spot_check.py`, onnxruntime
  1.31.0, 24 seeded DEV pairs): label agreement with the training run's saved predictions
  **1.0**, max label-logit difference 0.0042 (training used fp16 autocast), 119.6 s
  (~5 s/pair, 8 threads). Log: `kaggle-recovery/logs/fp32_spot_check.log`.

Not run: model-service/API test suites and the CPU smoke pipeline (need torch, fastapi,
Postgres/Redis and the downloaded embedding model); `make ci`; the service over HTTP with
the recovered artifact.

## Remaining blockers for Phase 3 acceptance

1. LLM baseline: needs a bounded `--limit 1` run (paid, user decision) to capture the
   rejection's `error_detail`, then a fix and a full DEV/TEST run. The Kaggle attempt cache
   (all 400s) holds no usable answers; reruns re-request every pair.
2. Cascade: re-run `ml.eval.evaluate` after a usable LLM baseline, then refresh the manifest.
   Until then serve no tau (the recovered fp32 artifact does this).
3. Serving: install the fp32 artifact (`python -m ml.colab.bundle install-artifact
   --artifact-dir kaggle-recovery/fp32-recovered/deberta_v3_base`; on exFAT `current`
   becomes a ~750 MB copy) and set `RULE_CHECK_VARIANT=fp32`; then verify `/v1/check` in
   the full stack. Service CPU latency is ~5 s/pair locally, which matters for Phase 4.
4. int8: repair (e.g. exclude embedding Gather or quantize statically, choosing on DEV) or
   document fp32 as the CPU build; the handoff asks for int8 numbers, which now exist and
   are honest failures.
5. Full local gates (`make ci`, smoke) in a proper environment, with test scratch on a
   Linux filesystem; re-run `ruff`/`mypy` on the changed files (not re-run after the
   corrections, tools absent); independent review of the final diff.

## Space notes

The SSD is exFAT with 1 MiB clusters: every small file uses ≥ 1 MiB. The test environment
is 7.5 GB on disk (2 GB of data). Large pure-Python packages are in `site311-pure.zip`
(directory entries included so namespace packages import). Delete `kaggle-recovery/env`
when done; recreate with `python3.11 -m pip install --target …` plus the zip step. Keep
pytest scratch (`TMPDIR`, `--basetemp`) on `/tmp`, not on the SSD (see Checks run).
Recovery evidence (`inspection/`, `verification.json`, bundle) is untouched.

## Independent supervisor completion checks

Codex independently ran the six lightweight test modules with Linux `/tmp` scratch:
**102 passed in 15.52 s**. Afterward Claude made only two nonbehavioral lint changes
in the regression test file (a raw regex literal and formatting).

Codex installed pinned Ruff 0.17.1 in an isolated temporary tool directory and checked
the final 13 changed Python files: **lint passed; all 13 files formatted**.
`git diff --check` passed. Log: `kaggle-recovery/logs/codex-independent-lint.log`.
This supersedes the earlier statement that final Ruff verification was unavailable;
full strict mypy, application/integration suites and hosted CI remain outstanding.

The real recovered fp32 artifact was independently loaded and hash-verified, its
invalid cascade tau is absent, the original collapsed int8 artifact is refused,
and malformed quality values are refused. Record:
`kaggle-recovery/logs/codex-artifact-check.json`. No live service was deployed,
no model was retrained and no real-provider baseline call was made.
