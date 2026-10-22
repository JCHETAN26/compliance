# Colab runbook: training the rule-check model (Phase 3 external step)

`colab_train.ipynb` runs the repository's own scripts on a Colab T4 GPU. It adds no modelling code.
This page lists everything a person has to do, in order. Nothing here requires pasting a token or
API key into a chat, a notebook cell, a file or a command line.

## 0. Preconditions

- The Phase 3 code is **committed** to the private `JCHETAN26/compliance` repository (the
  supervisor reviews, commits and pushes it, and prepares the git bundle and a copy of this
  notebook with `EXPECTED_COMMIT` filled in). Colab must train exactly that commit; the notebook
  refuses any other or a dirty checkout.
- A Google account with Colab and Google Drive (about **10 GB** free on Drive, see the space plan).
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
2. Set `EXPECTED_COMMIT` (full hash) in cell 2.
3. Run all cells. Steps: GPU + Drive -> source -> environment (`uv sync --frozen`, then CUDA torch
   2.14.1+cu126 replaces the locked CPU build; `assert_cuda()` re-checks before every GPU step) ->
   pinned artifacts -> data (prepared once on GPU, kept on Drive so every session has the same
   `data_version`) and classic baselines -> three training runs -> evaluation, export, CPU latency and
   int8/fp32 TEST accuracy -> optional LLM baseline -> final evaluation and manifest refresh -> bundle.
4. **Disconnects** are expected on Colab: rerun cells 1–5, then the interrupted cell. Training resumes
   from `checkpoints/last` (a different data version, tokenizer or training code is refused).
5. Training time on a T4 has not been measured yet; each run logs steps and elapsed time.

### Drive space plan (estimates from parameter counts, not yet measured)

| Item | Approximate size |
|---|---:|
| DeBERTa-v3-base run: `checkpoints/best` (fp32 weights) | 0.74 GB |
| DeBERTa-v3-base run: `checkpoints/last` (weights + AdamW state) | 2.2 GB |
| ModernBERT-base run (best + last) | 2.4 GB |
| Second DeBERTa run (unseen hypotheses) | 3.0 GB |
| ONNX fp32 export (kept on Drive only) | 0.74 GB |
| ONNX int8 export + tokenizer (also in the bundle) | measured at export (`sizes_bytes`) |
| Data, baselines, predictions, results | < 0.1 GB |

After a run reports `run complete`, its `checkpoints/last` may be deleted to save about 2 GB
(`checkpoints/best` must stay until export is done).

## 3. Optional: the Anthropic zero-shot baseline

Either in Colab (cell 8, `RUN_LLM = True`, key stored as the Colab secret `ANTHROPIC_API_KEY`), or
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

Running it in Colab (cell 8) is simpler: cell 9 then re-evaluates and refreshes the artifact. If it is
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

1. Download `MyDrive/compliance-runs/results-bundle-<commit12>.tar` (it holds the processed data,
   run records and DEV/TEST predictions, baselines, `ml/eval/results/*`, and the int8 artifact; no
   training weights). Check free disk first (`df -h /`): import needs the bundle size plus ~300 MB
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
