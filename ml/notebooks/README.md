# Colab runbook: training the rule-check model (Phase 3 external step)

`colab_train.ipynb` runs the repository's own scripts on a Colab T4 GPU. It adds no modelling code.
This page lists everything a person has to do, in order. Nothing here requires pasting a token or
API key into a chat, a notebook cell, a file or a command line.

## 0. Preconditions

- The Phase 3 code is **committed** to the private `JCHETAN26/compliance` repository (the
  supervisor reviews, commits and pushes it, and prepares the git bundle and a copy of this
  notebook with `EXPECTED_COMMIT` filled in). Colab must train exactly that commit; the notebook
  refuses any other or a dirty checkout.
- A Google account with Colab and Google Drive. Local mode (default) needs Drive space only for the
  optional backups (about 4 GB for all three runs, the artifact and the bundle; see the space plan).
- Optional, for the real LLM baseline: an Anthropic API key with billing enabled.

## 1. Get the source to Colab (choose one)

**Option A — git bundle (recommended, no token at all).** On the development machine:

```bash
cd /home/grv22/projects/compliance
git bundle create /tmp/compliance.bundle HEAD main   # ~1 MB; contains committed history only
```

Upload `compliance.bundle` to Google Drive as `MyDrive/compliance-runs/compliance.bundle`
(drive.google.com -> New -> File upload), then delete `/tmp/compliance.bundle`. The notebook verifies
the bundle inside a fresh empty repository, clones it and checks out `EXPECTED_COMMIT`. (This exact
sequence was exercised locally from a fresh directory.)

**Option B — the private GitHub repository through a Colab secret.**

1. GitHub -> Settings -> Developer settings -> Personal access tokens -> **Fine-grained tokens** ->
   Generate: *Repository access* = only `JCHETAN26/compliance`; *Permissions* = Contents: **Read-only**
   (nothing else); a short expiration (e.g. 7 days).
2. In Colab: the key icon (Secrets) in the left bar -> Add new secret -> name `GITHUB_TOKEN`, paste the
   token **there** (never in a cell) -> enable "Notebook access" for this notebook.
3. In the notebook set `USE_OPTION_B_GITHUB = True`.

The token reaches `git` only via the environment of a temporary `GIT_ASKPASS` helper that is deleted
after the clone; it is never printed, never part of a URL and never written to `.git/config`. Revoke the
token on GitHub when training is done.

## 2. Run the notebook

1. Open `ml/notebooks/colab_train.ipynb` in Colab (File -> Upload notebook, from your checkout or from
   the GitHub web UI). Runtime -> Change runtime type -> **T4 GPU**.
2. Set `EXPECTED_COMMIT` (full hash) in section 2 (Source code).
3. Keep `RUN_LOCATION = "local"` (section 1, the default); see the space plan. Run all cells. Steps: GPU + Drive -> source -> environment (`uv sync --frozen`, then CUDA torch
   2.14.1+cu126 replaces the locked CPU build; `assert_cuda()` re-checks before every GPU step) ->
   pinned artifacts -> data (prepared once on GPU, kept on Drive so every session has the same
   `data_version`) and classic baselines -> three training runs -> evaluation, export, CPU latency and
   int8/fp32 TEST accuracy -> optional LLM baseline -> final evaluation and manifest refresh -> bundle.
   Every command's output (stdout and stderr) appears live in its cell and is appended to a log under
   `RUNS/logs/` whose path is printed first; a failing command stops the cell with its exit code. The
   stop button interrupts the command itself (SIGINT, then SIGTERM, then SIGKILL) and waits for it, so
   no orphaned process keeps writing.
4. **Interruptions.** Local files live as long as the Colab **runtime (VM)**: a browser disconnect
   followed by a reconnect to the same runtime keeps them; a deleted, reset or expired runtime does
   not (the [Colab FAQ](https://research.google.com/colaboratory/faq.html) notes that VMs have a
   limited lifetime). After a reconnect or kernel restart in the same runtime, rerun sections 1–5,
   then the interrupted cell. The source check still requires the exact commit and an unchanged
   implementation, but ignores `ml/eval/results/`, which the pipeline itself rewrites
   (`data_report.json`) and fills. Any other changed or untracked file stops the notebook.
   - `RUN_LOCATION = "local"`: in the same runtime a rerun resumes from `checkpoints/last`. Once the
     runtime is deleted, reset or expired, an unfinished run **starts again from step 0**. Backups of
     completed runs (below) have no `checkpoints/last` and are never restored automatically.
   - `RUN_LOCATION = "drive"` (legacy): a resume across runtimes needs a complete
     `checkpoints/last` to have actually reached Drive, which nothing verifies; free space is
     necessary, not sufficient, and the space checks are no durability guarantee (see the
     `36eb2eb` incident below). A different data version, tokenizer or training code is refused.
   - **Training a new commit in the same runtime (Option A):** replace `compliance.bundle` on Drive,
     set the new `EXPECTED_COMMIT`, then rerun from section 2. The source cell verifies the bundle,
     fetches it into the existing checkout (`FETCH_HEAD` only: no branch update, reset or
     working-tree change) and checks out the commit only if the implementation is clean; git itself
     refuses to overwrite a changed generated file. Then rerun sections 3–5 (the lock may differ).
     With Option B an existing checkout that lacks the commit is refused with instructions (use
     Option A, or delete the runtime and start over).
5. Measured on a T4 (first `36eb2eb` run): about 83 minutes to optimizer step 7,700, in the third epoch
   of `deberta_v3_base`. Each run logs steps and elapsed time.

### Space plan

`checkpoints/best` of `deberta_v3_base` measured 740,088,360 bytes; the other sizes are estimates from
parameter counts.

| Item | Approximate size |
|---|---:|
| DeBERTa-v3-base run: `checkpoints/best` (fp32 weights) | 0.74 GB |
| DeBERTa-v3-base run: `checkpoints/last` (weights + AdamW state), rewritten every 500 steps | 2.2 GB |
| ModernBERT-base run (best + last) | 2.4 GB |
| Second DeBERTa run (unseen hypotheses) | 3.0 GB |
| ONNX fp32 export | 0.74 GB |
| ONNX int8 export + tokenizer (also in the bundle) | measured at export (`sizes_bytes`) |
| Data, baselines, predictions, results | < 0.1 GB |

**Local mode (default).** Local disk needs about 10 GB free per training run (`train()` checks this
and stops otherwise; nothing is deleted automatically) — a fresh T4 runtime has far more. Drive only
receives verified copies in new folders under `MyDrive/compliance-runs/backups/`: each completed run
without `checkpoints/last` (~0.8 GB per run), the baselines if computed here, the artifact folder and
the results bundle. `backup_to_drive` first checks Drive's free space (copy size + 1 GB) and otherwise
prints `SKIPPED` and copies nothing; it never deletes, overwrites or reuses anything on Drive. Its
read-back check verifies the write through the mount, not the upload to Google, so confirm the
folder on drive.google.com. If a backup is skipped, download the file from the Colab file browser
before ending the session (always do this for the results bundle).

**Drive mode (legacy).** Every checkpoint rotation writes ~2.2 GB and deletes the previous one.
Files deleted through the Colab mount are expected to go to Drive Trash, which counts against the
quota, so a 15 GB Drive can fill within a few rotations; a full Drive makes writes fail. Ample free
space is still no guarantee that a checkpoint reached Drive.

### The first `36eb2eb` run: no resumable checkpoint visible

Confirmed facts: the first `deberta_v3_base` run of `36eb2eb` (Drive mode) reached step ~7,700 in its
third epoch and was then interrupted; the rerun stopped with `CheckpointError: ... only incomplete
checkpoints ['.last-staging']`. The user's live inventory showed Drive 100 % full (147 MB free of
15 GB) and, under `checkpoints/`, only `best/model.safetensors` (740,088,360 bytes): no `COMPLETE`,
`state.json` or `trainer.pt` anywhere. No training process was running. `recover_last` refused to
restart silently, as designed.

Not established: how the newer checkpoints disappeared. Hypotheses consistent with the facts are
Trash filling the quota so that later checkpoint writes never reached Drive, and the mount
reconciling pending renames/deletes after the interrupt; neither is proven, and the interrupt's role
is unknown. The [Colab FAQ](https://research.google.com/colaboratory/faq.html) warns that
interrupted Drive operations can lose data.

- **The currently visible files do not permit an exact resume** (weights only: no optimizer,
  scheduler, scaler, RNG or position). Do not warm-start from `best`, create `COMPLETE`/`state.json`
  or edit `run.json`.
- Preserve the folder first with the supervisor's operational `preserve_failed_run.py`
  (`!python3 <path to the script>`): a read-only, SHA-256-verified copy into a new local folder.
  It is lost when the runtime is deleted, reset or expires; download what you want to keep.
- A complete checkpoint may still exist in Drive Trash (one folder with `model.safetensors`,
  `trainer.pt`, `state.json` and `COMPLETE`); this is possible, not guaranteed. Check before
  emptying Trash; Trash search is limited, so "not found" is not proof either.
- If nothing complete is found, the decision is a fresh retrain of `deberta_v3_base` in local mode
  (a new run directory under `/content/compliance-runs`). The failed Drive folder stays where it is.

### Recovering from the fp16 failure of source `4cd76d8`

The first production run on `4cd76d8` stopped at its first optimizer step with
`ValueError: Attempting to unscale FP16 gradients`. Cause: transformers 5 loads checkpoints
with `dtype="auto"`, and DeBERTa-v3-base's `pytorch_model.bin` is stored in fp16, so the model
had fp16 parameters. Fixed in a later commit: backbones are always built in float32 and
training refuses to start otherwise; mixed precision comes only from autocast. That failed run
wrote `run.json` but no training log and no checkpoint, so nothing is lost. To recover:

1. Use the new commit, bundle and notebook copy the supervisor provides.
2. Runtime -> **Disconnect and delete runtime** (the old checkout in `/content/compliance` does
   not contain the new commit).
3. In Google Drive, move the failed run folder `MyDrive/compliance-runs/deberta_v3_base` to
   `MyDrive/compliance-runs/archive/deberta_v3_base-failed-4cd76d8-fp16` (keep it; do not delete
   it). Leave `data-full/` and `baselines/` exactly where they are: the restored data keeps the
   same `data_version`.
4. Replace `MyDrive/compliance-runs/compliance.bundle` with the new bundle, then run all cells.

Re-using the old run folder is refused by design (`ResumeIdentityError`: the training code
changed); do not edit `run.json` or work around that check. GPU verification of the fix is still
pending: the first `step 1` and, after 500 steps, the first `checkpoints/last` in the new run
folder show the run has passed the point where `4cd76d8` failed.

**Optional: recover in the same runtime** (keeps the CUDA environment, pinned weights and
prepared data; the steps above remain the default). The supervisor provides
`recover_fp16.py`, an operational script that is not part of the repository. Set `NEW_COMMIT` in
it, upload it next to the new bundle in `MyDrive/compliance-runs/`, and run it in a new cell:
`!python /content/drive/MyDrive/compliance-runs/recover_fp16.py`.

- It first checks everything: the checkout is exactly `4cd76d8` (or already the new commit) with
  no implementation changes outside `ml/eval/results/`; the bundle verifies and contains the new
  commit, which descends from `4cd76d8`; and the `deberta_v3_base` folder is the step-1 failure
  (not complete, started from `4cd76d8`, no checkpoint entries, empty or missing training log).
  Anything else stops it without changes.
- Only then does it check out the new commit and rename the failed folder to a unique
  `deberta_v3_base.failed-fp16-4cd76d8-<timestamp>` sibling. It deletes nothing and never touches
  `data-full/`, `baselines/` or the environment. Running it again changes nothing.
- Afterwards do **not** rerun the source cell (it still holds the old `EXPECTED_COMMIT`). Rerun the
  **first training cell** and then the following cells; `PY`, `RUNS`, `run()` and `assert_cuda()`
  stay valid, and each `run()` starts a fresh process with the fixed code. For any later full
  restart, use the new ready notebook with the new `EXPECTED_COMMIT`.

## 3. Optional: the Anthropic zero-shot baseline

Either in Colab (section 8, `RUN_LLM = True`, key stored as the Colab secret `ANTHROPIC_API_KEY`), or
locally after importing the bundle:

```bash
read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY   # typed into your own terminal, not echoed
make baseline-llm SPLIT=dev LLM_ARGS=--dry-run          # prompt sample + rough cost estimate, no API call
make baseline-llm SPLIT=dev && make baseline-llm SPLIT=test
unset ANTHROPIC_API_KEY
```

Default model `claude-sonnet-5-5`, effort `low`, over the same top-8 candidates as TF-IDF; prices in
`ml/baselines/anthropic_pricing.yaml` (re-check before running). Interrupted runs resume from the
response cache without paying twice; every attempt's usage and latency are recorded.

Running it in Colab (section 8) is simpler: section 9 then re-evaluates and refreshes the artifact. If it is
run locally after the import instead, finish with (no weights needed; the imported run records suffice):

```bash
make eval-model RUN_DIR=.cache/ml/runs/deberta_v3_base ABLATION_RUN_DIR=.cache/ml/runs/modernbert_base \
     UNSEEN_RUN_DIR=.cache/ml/runs/deberta_v3_base_unseen4 EVAL_ARGS="--bench ml/eval/results/latency_colab_cpu.json"
make export-model RUN_DIR=<run_dir named in ml/eval/results/inference_params.json> \
     ARTIFACT_DIR=.cache/models/rule-check/<installed version> EXPORT_ARGS=--refresh-manifest
make install-model ARTIFACT_DIR=.cache/models/rule-check/<installed version>   # new version name
```

(the refresh rewrites only the manifest; `install-model` then installs it under its new, parameter-
hashed version and repoints `current`; the old directory can be deleted afterwards).

## 4. Bring the results back

1. Download `results-bundle-<commit12>.tar` (local mode: `/content/compliance-runs/` via the Colab file
   browser, or its backup folder under `MyDrive/compliance-runs/backups/`; Drive mode:
   `MyDrive/compliance-runs/`). It holds the processed data,
   run records and DEV/TEST predictions, baselines, `ml/eval/results/*`, and the int8 artifact; no
   training weights. Check free disk first (`df -h /`): import needs the bundle size plus ~300 MB
   and refuses otherwise. On the current development host (~150 MB free while three cancelled
   BuildKit records stay pinned) space has to be made first; see `docs/BUILD_STATUS.md`.
2. `make import-colab-bundle BUNDLE=~/Downloads/results-bundle-<commit12>.tar` — verifies every
   SHA-256, refuses unsafe paths/names and over-budget archives, validates everything before changing
   anything, keeps any differing local data/results as backups, and installs the artifact as
   `.cache/models/rule-check/current`.
3. Delete the downloaded tar. Restart the model service (`docker compose up -d model-service`) and run
   `make smoke-phase3` (expects `model_kind=production`). The web banner disappears once a trained
   model serves.
4. The supervisor reviews `ml/eval/results/*` (handoff criterion: model TEST macro-F1 above the
   TF-IDF baseline, reported honestly either way) and the provenance in each file.
