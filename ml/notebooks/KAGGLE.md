# Kaggle runbook: training the rule-check model (Phase 3 external step)

`kaggle_train.ipynb` runs the same pipeline as `colab_train.ipynb` (same scripts, configs, data
protocol and DEV-only selections) on a Kaggle GPU, with Kaggle paths. It adds no modelling code.
The Colab notebook and its runbook (`README.md`) are unchanged. Nothing here asks you to paste a
token or API key into a chat, a cell, a file or a command line.

**Status.** Not yet run on Kaggle. Local checks covered the notebook's own code: bundle
discovery, exact-commit checkout, path layout, output budget, the import of previous outputs and
the child-process handling. No GPU, Kaggle account or Kaggle API was used to write it. Phase 3
stays unaccepted until the full results bundle is imported and independently verified.

## Sources

- Kaggle CLI docs: [kernels](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md),
  [kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md),
  [datasets](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md). Checked on
  2026-10-23. They say: `NvidiaTeslaT4` is "GPU T4 ×2, default GPU"; secrets are defined on
  kaggle.com only and read in a notebook with `UserSecretsClient` from the pre-installed
  `kaggle_secrets`; `kernels push` uploads and then runs the notebook (`--no-run` is the
  equivalent of Quick Save); `is_private` defaults to true; datasets are private unless `--public`
  is given.
- Kaggle Notebooks docs, <https://www.kaggle.com/docs/notebooks>. The page renders with
  JavaScript and could not be fetched from the development host, so read it yourself before you
  start. This runbook assumes two things it says: `/kaggle/working` is saved as a version's output,
  up to **20 GB**, and `/kaggle/temp` is scratch space that is not saved. The notebook does not
  rely on `/kaggle/temp` existing; it falls back to `/tmp`.
- Not established from any official source checked: the session run-time limit, the weekly GPU
  quota, how idle interactive sessions behave, the "Persistence" setting, whether a version that
  fails or times out keeps its output, and limits on the number of output files. The notebook does
  not depend on any of these. Read the current values in the Kaggle UI.

## 0. Preconditions

- A Kaggle account that can use **GPU** and **Internet** in notebooks. Kaggle may ask you to verify
  the account (e.g. by phone) first. You check GPU availability and the remaining weekly quota
  under your profile/settings; this runbook promises no number of free hours.
- The reviewed commit and two files from the supervisor: `compliance.bundle` (a `git bundle` of
  the private repository, committed history only, about 1 MB) and `kaggle_train-ready.ipynb` (this
  notebook with only `EXPECTED_COMMIT` filled in). The notebook refuses any other commit and any
  changed source.
- Optional, for the real LLM baseline: an Anthropic API key with billing enabled. Use it only after
  you have decided to pay for those calls.

## 1. The source as a private dataset (no GitHub token needed)

1. kaggle.com -> **Datasets** -> **New Dataset**. Upload `compliance.bundle`, name the dataset
   e.g. `compliance-source`, and keep it **Private**. Never make it public: it contains the private
   repository's history. Do not add anything else to it.
2. For a new commit later, create a **new version** of the same dataset that replaces the bundle.
   Do not add a second dataset that also contains a bundle. If two bundles are visible, the
   notebook stops and lists both.

Optional CLI (only after `kaggle` is installed and authenticated on your own machine): put the
bundle alone in a folder, run `kaggle datasets init -p <folder>`, edit `dataset-metadata.json`
(title, `id` = `<your username>/compliance-source`), then run `kaggle datasets create -p <folder>`.
Do **not** pass `--public`; datasets are private by default. The CLI is not needed for anything
else here.

## 2. The notebook

1. kaggle.com -> **Code** -> **New Notebook** -> File -> **Import Notebook**, and choose
   `kaggle_train-ready.ipynb`. Keep the notebook **private**: the output of a public notebook is
   public, and this output holds prepared data, predictions and model files.
2. Notebook settings (right-hand panel): **Accelerator: GPU T4 x2**. The pipeline uses one GPU,
   `cuda:0`; the notebook sets `CUDA_VISIBLE_DEVICES=0`, so the second T4 stays idle and gives no
   speed-up. Turn **Internet on**: it is needed for the pinned Python packages, CUDA PyTorch, the
   pinned model and data downloads and, if enabled, the API.
3. **Add Input** -> your private `compliance-source` dataset. Kaggle mounts it read-only under
   `/kaggle/input/`. The notebook searches there for the single `*.bundle` file (up to 4 folders
   deep, symlinks followed). `SOURCE_BUNDLE` can name one exact path under `/kaggle/input`
   instead.
4. Settings cell (section 0). The ready copy has `EXPECTED_COMMIT`. Leave the defaults:
   `SOURCE_BUNDLE = None`, `IMPORT_PREVIOUS_OUTPUT = None`, all three `TRAIN_RUNS`,
   `RUN_LLM = False`, `OUTPUT_LIMIT_GB = 20`.

What the cells do: GPU and path check -> verified bundle and exact commit -> runtime helpers and
an output preflight -> environment -> pinned artifacts -> optional import of a previous output
-> prepared data and classic baselines -> three training runs -> evaluation, ONNX export, CPU
latency and int8/fp32 TEST accuracy -> optional LLM baseline -> final evaluation and manifest
refresh -> results bundle and output report. Every command's output is shown live in its cell
and appended to a log under `compliance-runs/logs/`; the log path is printed first. A failing
command stops the cell with its exit code. The stop button interrupts the command itself
(SIGINT, then SIGTERM, then SIGKILL) and waits for it to end.

**Environment.** `uv 0.8.22` is installed with the kernel's own `python -m pip install --target`
into scratch, so Kaggle's Python environment is not changed. `uv sync --frozen` builds the
project's Python 3.11 `.venv` from `uv.lock`; uv downloads Python 3.11 into scratch if the image
lacks it. The same torch version is then reinstalled as the CUDA 12.6 build (`2.14.1+cu126`),
and the ML tool pins are added. All later steps call `.venv/bin/python` directly, never
`uv run`/`make`, which would re-sync the CPU build. `assert_cuda()` checks Python 3.11,
`2.14.1+cu126` and a real CUDA operation before each GPU step.

### Running it: interactive session or Save & Run All

- **Interactive** (Run All in the editor) is useful for checking the first cells: the GPU, the
  source and the environment. Files exist only while the session runs.
- **Save Version -> Save & Run All** runs the notebook from top to bottom in a fresh session. It
  stores `/kaggle/working` (up to 20 GB) as that version's output, which you can download or
  attach to a later session. Use this for the long runs.
- Kaggle limits how long one session or version may run. One `deberta_v3_base` run on a Colab T4
  took about 83 minutes to reach optimizer step 7,700, in its third epoch; Kaggle's T4 has not
  been measured. If all three runs plus export might not fit within the limit you see, split the
  work: set `TRAIN_RUNS` to one or two runs for a version, then continue in the next version (§3).
  Evaluation, export and the bundle start only when all three runs are complete; until then the
  notebook reports this and skips those cells without an error.

## 3. Resuming and continuing

- **Same session:** rerun the interrupted training cell (after rerunning sections 0–4 if the
  kernel restarted). `ml.train.train` resumes from `compliance-runs/<run>/checkpoints/last`.
- **Next session or version, from a saved output.** Attach the previous version's output as an
  input (Add Input -> your notebook's output) and set `IMPORT_PREVIOUS_OUTPUT` to its
  `compliance-runs` folder under `/kaggle/input`, or to the folder that contains it. The input is
  only read. These entries are **copied** into the new `compliance-runs`, each one SHA-256
  verified, and never over an entry that already exists:
  - `data-full` (the prepared data; the same `data_version` is reused) and `baselines`, if their
    data version matches;
  - `llm-cache` (answered API calls are not paid for twice);
  - **complete** runs with the current training-code fingerprint and data version, without their
    2.2 GB resume checkpoint, which a complete run never needs;
  - **unfinished** runs only if they hold a complete resume checkpoint (`COMPLETE` marker). The
    training script then checks the config, data version, tokenizer and training code again
    before it resumes.

  Weights alone (`checkpoints/best` without a complete `checkpoints/last`) are **never** resumed
  or used as a warm start: that run is not imported and starts fresh. Logs, exports and old
  bundles are not imported. Section 6 prints the outcome for every entry, and the preflight
  (section 3) shows each run's state: `absent`, `complete`, `resumable`, `restart` or `blocked`.
- **Persistence setting.** If your notebook's settings offer to keep files between sessions and
  you enable it, `compliance-runs` may still be there in the next session. That has not been
  verified here. Rely only on what the preflight report shows; the training script's identity
  checks still apply.
- Nothing becomes durable automatically. A file outlives its session only if it is in a saved
  version's output or you have downloaded it.

### The failed Colab run

Leave Google Drive and the failed Colab run folder as they are; nothing here reads from or writes
to Drive. That run left only `checkpoints/best` (weights, no optimizer or position), so it cannot
be resumed and is not imported: start fresh on Kaggle. Import a Colab folder only if a **complete**
checkpoint was found for it (and copied into a Kaggle dataset). Even then, the fingerprint and
data-version checks above apply.

## 4. Space plan

`compliance-runs` (in `/kaggle/working`) is what a version saves; Kaggle's limit is 20 GB, which
the notebook reads as 20 × 10⁹ bytes. Expected contents with all three runs:

| Item | Approximate size |
|---|---:|
| `deberta_v3_base` (best 0.74 GB measured + `checkpoints/last` 2.2 GB) | 3.0 GB |
| `modernbert_base` (best + last, estimated) | 2.4 GB |
| `deberta_v3_base_unseen4` (estimated) | 3.0 GB |
| `artifact-<run>`: ONNX fp32 + int8 + tokenizer | ~1 GB |
| results bundle (int8 artifact, data, predictions, results; at most 1.2 GB) | < 1.2 GB |
| prepared data, baselines, logs, results copy, LLM cache | < 0.1 GB |
| **Total** | **~10–11 GB** |

- The preflight (section 3) and the final cell print two separate figures. The first is the
  saved-output size: `compliance-runs` against the limit, with its file count and each entry. The
  second is the space actually free on the working and scratch filesystems; it can be larger than
  the limit and says nothing about what a version keeps.
- Before each training run the notebook requires 10 GiB actually free (checkpoint promotion
  temporarily holds two resume checkpoints). It also requires that `compliance-runs` plus the
  run's expected size stays within the limit; a resumed run counts with what it already holds.
  Export, the bundle and imports are checked the same way. A failed check stops the cell and
  **deletes nothing**.
- Each unfinished run's `checkpoints/last` stays in the output, so a later version can resume
  it. Complete runs keep theirs too (nothing is deleted automatically); imports leave it out.
- Not in the output (scratch, `/kaggle/temp` or `/tmp`): the source checkout, `.venv` (several GB
  with CUDA), uv, pip, Hugging Face and torch caches, the downloaded backbones, BGE encoder and
  ContractNLI archive. The notebook needs about 15 GiB free there. If scratch and
  `/kaggle/working` share a filesystem, the report says so.
- Nothing in the output contains a secret. Logs hold command lines and program output only, never
  the environment, and the API key is never printed.

## 5. Optional: the Anthropic zero-shot baseline (real, billed API calls)

Default `RUN_LLM = False`. To run it on Kaggle, only once the cost is authorized:

1. Create the private notebook first. Then, in the editor: **Add-ons -> Secrets** -> add
   `ANTHROPIC_API_KEY` (the value is entered there only) and attach it to this notebook. The
   Kaggle CLI cannot create or attach secrets.
2. Set `RUN_LLM = True`. Section 3 checks the secret **before** the environment, data
   preparation and hours of GPU training. It uses Kaggle Secrets only; there is no API call and
   no charge. A missing, empty or unattached secret stops the notebook there with these setup
   steps. The value, its length and the lookup error's text are never shown, and the value is not
   kept.
3. Section 10 runs the dry runs, which print a prompt sample and a rough cost estimate. It then
   reads the secret again (`UserSecretsClient().get_secret`) into the child process's environment
   only and calls the API for DEV and TEST. The prices are in
   `ml/baselines/anthropic_pricing.yaml`; check them before running.
4. The response cache lives directly in `compliance-runs/llm-cache`. The script's own cache path
   in the scratch checkout (`.cache/ml/llm`) is a link to it, and the script flushes after every
   recorded attempt, so each billed attempt is in the output folder as soon as it is written. That
   means it is in a saved version, or in the files on disk if the session dies. A rerun, including
   one in a later session with the output imported, does not pay twice for pairs already answered.
   If an earlier real cache folder exists at that path, it is merged in append-only (a file is
   only extended, never shortened; diverging files stop the notebook) and moved aside in scratch,
   never deleted. A link pointing anywhere else is refused. The cache holds the recorded response
   fields and usage, never the key.
5. Detach or delete the secret when you are done.

Alternatively, run it locally after importing the bundle, as in `README.md` §3. The latency file
from Kaggle is `ml/eval/results/latency_kaggle_cpu.json`; use it in place of
`latency_colab_cpu.json` in those commands.

## 6. Bring the results back

1. Download `compliance-runs/results-bundle-<commit12>-<time>.tar` from the notebook's output.
   That is the Output panel in the editor, or the saved version's Output tab. Optional CLI:
   `kaggle kernels output <user>/<notebook> -p <dir> --file-pattern 'results-bundle-.*'`.
2. Then follow `README.md` §4 exactly: check free disk, run
   `make import-colab-bundle BUNDLE=<path>`, restart the model service and run
   `make smoke-phase3`. The importer works the same way for a Kaggle bundle.
3. The supervisor reviews `ml/eval/results/*` and the provenance (the source commit is recorded
   in the bundle manifest).

## 7. Optional: CLI kernel metadata template

`kaggle-kernel-metadata.template.json` is a template for `kaggle kernels push`. It sets a private
notebook, GPU, Internet, `machine_shape: NvidiaTeslaT4` and the source dataset. Replace
`KAGGLE_USERNAME` with your username, put it next to the ready notebook as
`kernel-metadata.json`, and run `kaggle kernels push -p <folder>`. That uploads the notebook and
starts a run (`--no-run` uploads without running). Secrets cannot be set from the CLI. The web UI
does everything above without the CLI.

## Limitations

- Not executed on Kaggle yet. The first real session must show `assert_cuda()` passing, the first
  training steps and, after 500 steps, a complete `checkpoints/last`.
- The Kaggle T4 run time is unmeasured, and the session and quota limits are not built in.
- No automatic durability: copy or download what you want to keep.
- One GPU only. Data parallelism over both T4s would change the training protocol and is out of
  scope.
