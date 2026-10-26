# Rule-check quantization — diagnosis and candidate repair (2026-10-25)

Follows `KAGGLE_RECOVERY_FIXES.md` (int8 collapse: export DEV parity label agreement 0.083,
top-k 0.0; TEST macro-F1 .063 vs fp32 .813). Offline only: no retraining, no provider call,
no live-stack change. Every choice below was made on DEV; TEST was not run.

## Cause

The old exporter called `quantize_dynamic(weight_type=QInt8)`: int8 weights for every
MatMul and the embedding Gather, and **one uint8 scale per activation tensor**, computed at
run time. DeBERTa-v3-base has activation outlier channels. Measured on fp32 at the input of
each weight MatMul (32 DEV rows, `ranges.json`):

| MatMul input | max abs | std | uint8 step / std |
|---|---|---|---|
| `layer.1/output/dense` (FFN out, after GELU) | 100.8 | 0.25 | 1.55 |
| `layer.0/output/dense` | 100.8 | 0.35 | 1.12 |
| `layer.3/intermediate/dense` (FFN in) | 208.2 | 1.00 | 0.87 |
| median over all 72 weight MatMuls | | | 0.12 |

A step larger than the signal spread rounds almost every value of the other channels to
the same level, so the FFNs lose their input. That is a property of per-tensor activation
scales, not of a flag value. ORT's documented non-VNNI remedy (`reduce_range`, MatMul only,
per-channel) does not fix it (trial A). This CPU is AVX2 without VNNI, the case where ORT
documents U8S8 saturation, so saturation may add to the error but is not the main cause.

## Trials (fixed DEV sample)

Sample: the first 8 pairs of the exporter's own seeded DEV parity sample (`SEED` 20261021),
399 evidence rows plus the label rows. Reference: ONNX fp32 from the recovered artifact
(SHA-256 `8e32d88a…4a74`; recorded parity vs PyTorch fp32 max |Δ| 1.2e-5). Metrics follow
`ml.export.export.parity`; "end-to-end" also uses each model's own top-k. Times are
for the same work at 8 threads (fp32: 35.9 s). Gate = the service's
`int8_quality_problem` (label ≥ 0.9, top-k ≥ 0.75).

| Trial | Scheme | Size | Time | max \|Δ\| evid. | mean \|Δ\| evid. | max \|Δ\| label | label agr. | top-k agr. | end-to-end | corr | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | dynamic, MatMul+Gather, QInt8 | 245 MB | 24.0 s | 8.98 | 8.28 | 6.94 | 0.25 | 0.00 | 0.25 | 0.07 | fail |
| A | dynamic, MatMul only, per-channel, reduce_range | 540 MB | 26.4 s | 6.04 | 1.89 | 3.55 | 1.00 | 0.25 | 0.75 | 0.82 | fail |
| B | weight-only 8-bit MatMul (block 128, sym.), fp32 embedding | 545 MB | 43.1 s | 0.136 | 0.008 | 0.014 | 1.00 | 0.875 | 1.00 | 1.0000 | pass |
| **C1** | B + 4-bit embedding (block 32, sym.) | **213 MB** | 43.1 s | 0.270 | 0.016 | 0.030 | 1.00 | 0.875 | 1.00 | 0.9999 | pass |
| C2 | C1 + `accuracy_level` 4 (int8 products) | 213 MB | 43.4 s | 0.370 | 0.029 | 0.046 | 1.00 | 0.875 | 1.00 | 0.9998 | pass |

The original collapse reproduces locally (all 8 pairs got the same label). The one top-k
miss in B/C1/C2 is the same pair. Weight-only quantization keeps activations in float32,
so the outliers no longer matter.

**Choice: C1** (best size at near-B parity). C2 is no faster on this AVX2 CPU and slightly
worse. B is the pure-8-bit fallback if a 4-bit embedding table is not acceptable under the
"int8" name.

**Cost:** weight-only quantization dequantizes weights for a float32 product. On this CPU
it is about 20 % *slower* than fp32 (43 s vs 36 s). The old dynamic file was faster but
wrong. The gain is size: 213 MB vs 741 MB on disk and in RAM. CPU latency stays the Phase 4
issue already recorded for fp32.

## Full DEV check (C1)

`python -m ml.export.eval_onnx --split dev` on a measurement-only candidate artifact (the
recovered manifest with the int8 slot pointing at C1 and no fp32 file; `make_candidate.py`).
All 1037 DEV pairs (53,992 evidence rows), the artifact's DEV-fitted parameters (k 4,
temperature 2.73, threshold 0.87), 8 threads. Reference: the training run's own DEV
predictions (PyTorch, CUDA fp16 autocast), scored with the same parameters.

| DEV, all pairs | macro-F1 | accuracy |
|---|---|---|
| PyTorch (training run) | 0.8020 | 0.8486 |
| ONNX int8 C1 | **0.7989** (−0.0031) | 0.8467 (−0.0019) |

Label agreement with the training run: **0.9961** (the old int8 got 0.140 on TEST). The drop is
within the ≤ 0.01 macro-F1 tolerance that `recover_fp32` applies to fp32. Wall time: 4265 s
(4.1 s/pair), part of it alongside the smoke run. Batch sensitivity (64 sample rows, batched vs
one at a time): max |Δ| **0.0** (old int8: 0.21), as expected with float32 activations. Result:
`../kaggle-recovery/logs/int8-investigation/dev_eval_C1.json`.

TEST was not run. The repair passes the DEV check, and the final TEST evaluation
(`eval_onnx --split test`) is proposed next. No production artifact with the new int8 file has
been built or installed. The training checkpoint is not available locally, so the exporter's
PyTorch parity step cannot be re-run here.

## Exporter change

- `ml/export/export.py`: `quantize_int8` uses `MatMulNBitsQuantizer` in two passes
  (MatMul 8-bit, then Gather 4-bit; `INT8_SCHEME`), and the scheme is recorded in the parity
  report as `int8.quantization`. Re-running it on the recovered fp32 file gives a file
  byte-identical to the evaluated candidate (SHA-256 `98aae3ba…db82e`). The serving gate
  is unchanged. ORT warns that it raises the model to opset 21 / IR 10 for int4. The file
  loads and runs in the pinned onnxruntime 1.31.0.
- `ml/smoke/tests/test_int8_quantization.py`: a tiny graph with the same structure (Gather
  → MatMul → Relu with one channel ×300 → MatMul with that channel's weights ÷300). The
  exporter's scheme: relative error 0.099 (mostly the 4-bit random-Gaussian embedding). Old
  dynamic: 1.03. Trial-A flags: 0.78. Bounds 0.15 and 0.5. With the old `quantize_int8`
  mounted over the source, the test fails (1.03 ≥ 0.15).

## Checks run

- `ml/smoke/tests` in the model-service image (real torch → ONNX → int8 → service):
  **15 passed** in 372 s (14 existing + the new test). Warnings: three pre-existing torch
  deprecations, and ORT's `invalid value encountered in divide` for all-zero embedding
  blocks (scale 0, mapped to 0 by its own `np.where`). Log:
  `../kaggle-recovery/logs/int8-investigation/ml-smoke-int8.log`.
- `ruff check`, `ruff format --check` and `mypy --strict` on the changed Python files: clean.
- Codex independently ran the new outlier regression in an isolated offline container:
  **1 passed** in 16.97 s. Log: `../kaggle-recovery/logs/codex-int8-regression.log`.
  Full-DEV validation was then completed (above); no production replacement has been accepted.

## Artifacts

Nothing original was modified: the hashes of the Kaggle artifact, the recovered fp32
artifact and `.cache/models/rule-check/current` were recorded before the work
(`originals-before.sha`) and re-checked after it. Evidence: `../kaggle-recovery/logs/int8-investigation/`
(trials, ranges, sample, full-DEV result). Measurement-only candidate (verified; not for
serving or installing): `../kaggle-recovery/int8-candidate/C1/`. Scripts: `../kaggle-recovery/claude-tmp/scripts/int8_diag.py`,
`make_candidate.py`, `batch_sens.py`, `toy_probe.py`.
