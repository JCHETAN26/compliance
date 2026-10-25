# ml/ — the rule-check model (Phase 3)

The trained model behind `POST /v1/check`: given a rule hypothesis and a contract's spans it
predicts **entailment / contradiction / not_mentioned** (app status satisfied / violated /
not_mentioned), the evidence spans and a calibrated confidence. Everything here is code and
configuration; datasets, checkpoints and exported weights are never committed.

**Status.** The local pipeline is implemented and verified (unit tests, the full CPU smoke
with a tiny *random* backbone, live Docker serving in fallback and with a smoke artifact);
hosted CI is pending. **Phase 3 acceptance is blocked** on external work: no production model
has been trained (the DeBERTa-v3 runs need a GPU, see the Colab runbook
[`notebooks/README.md`](notebooks/README.md)), the zero-shot baseline needs an Anthropic API key,
and the cascade, calibration and quantization results on the full splits and the import of a
trained artifact follow from those. Until a verified production artifact is installed, the model
service runs in untrained fallback mode and the web app shows a banner.

## Layout

| Path | What |
|---|---|
| `data/contractnli.py`, `data/prepare.py`, `data/store.py` | Official splits, ingestion segmentation, gold mapping, BGE candidate ranking, processed-data store |
| `data/reserialize.py` | Rewrites a processed dataset with the deterministic writer (no re-embedding), with verification |
| `train/` | Config schema, pinned backbone/tokenizer fetch, two-head model, TRAIN sampling, resumable training loop, run configs |
| `eval/` | Shared inference (`infer.py`), metrics, calibration, cascade, scoring, evaluation report, plots |
| `baselines/` | Majority, TF-IDF + logistic regression, Anthropic zero-shot (`llm.py`, pricing table) |
| `export/` | ONNX fp32 + int8 export with parity (`export.py`), CPU latency (`bench.py`), ONNX accuracy (`eval_onnx.py`) |
| `colab/bundle.py` | Results bundle Colab -> repository (pack / verified import / artifact install) |
| `notebooks/colab_train.ipynb` | Thin Colab wrapper over the same scripts |
| `notebooks/kaggle_train.ipynb`, `notebooks/kaggle_runtime.py`, `notebooks/KAGGLE.md` | Thin Kaggle wrapper over the same scripts, its stdlib runtime helpers and runbook |
| `requirements-tools.txt` | Pinned tool overlay (onnx, onnxscript, sentencepiece, anthropic) |
| `tests/` (base env), `smoke/tests/` (tool overlay) | Unit tests; end-to-end CPU smoke |

The serving-side code shared with training (input encoding, label order, top-k, artifact
manifest, batching, ONNX backend) lives in `apps/model-service/src/compliance_model_service/rule_check/`,
so served and evaluated predictions are computed by the same functions.

## Dependencies and disk

`uv.lock` and every `pyproject.toml` are deliberately unchanged in this phase: a change would
invalidate the ~1.3 GB Docker dependency layer shared by the worker and model service, which
this host cannot afford to rebuild. ONNX Runtime and `tokenizers` are already locked. The ML
tools run as a pinned overlay: `uv run --frozen --with-requirements ml/requirements-tools.txt …`
(the Make targets do this). Colab uses the same lock plus CUDA torch of the same version.

## Data (official splits, no gold at inference)

- **ContractNLI** (CC BY 4.0, see `data/LICENSES.md`): train 423 documents / 7,191 pairs, dev 61 /
  1,037, test 123 / 2,091; 17 hypotheses; every pair kept, document ids disjoint.
- **Correction to the handoff:** §7.3 says `not_mentioned` dominates. Observed TRAIN counts are
  entailment 3,530, not_mentioned 2,820, contradiction 841. Labels are unchanged; the label loss
  uses inverse-frequency weights computed from the actual TRAIN pairs used (0.679 / 0.850 / 2.850
  for the full TRAIN set).
- **Spans** = the application's ingestion segmentation (Phase 2 `line_blocks` + the ingestion
  sentence splitter), as `[start, end)` offsets into the unchanged text.
- **Gold evidence** (targets/evaluation only): a segment is gold iff it shares a non-whitespace
  character with a gold official span. The relation is many-to-many (416 TRAIN official spans
  overlap more than one segment; many segments contain several official spans); both directions
  are counted in `ml/eval/results/data_report.json`.
- **Candidates:** every pair's segments are ranked by the pinned BGE encoder (production query
  instruction and windowing). The top 8 are the fixed input of the TF-IDF and LLM baselines; the
  learned model scores every segment. Coverage ceiling of the top 8 (pairs with any gold in the top
  8): TRAIN 0.866, DEV 0.868, TEST 0.837 (reported, not tuned on).
- The processed full dataset in use has `data_version 620358b51ef96fac…`. It was re-serialized from
  the original preparation (`f93f4647…`) after a gzip-header determinism fix: rows identical,
  documents/pairs equal a fresh rebuild, and all 1,037 DEV rankings re-embedded exactly.

## Model

One backbone with two heads on the first-token representation:

1. **evidence head**: `[CLS] hypothesis [SEP] span [SEP]` -> evidence logit, for every span;
2. **label head**: `[CLS] hypothesis [SEP] s1 [SEP] … sk [SEP]` over the evidence head's own
   top-k spans in document order -> 3 logits; temperature-scaled softmax gives the confidence.

Training (TRAIN only, per epoch, seeded): gold segments plus sampled hard (BGE top-16 non-gold)
and random negatives for the evidence head; one label example per pair whose spans are the gold
segments (one may be dropped) filled from the BGE ranking (NotMentioned pairs: the BGE top-k).
Configs (`train/configs/`): `deberta_v3_base` (primary), `modernbert_base` (pure backbone swap
ablation), `deberta_v3_base_unseen4` (nda-1, nda-8, nda-16, nda-18 held out of TRAIN and of every
DEV selection; drawn with `random.Random(20261021)` before training), `smoke` (CPU only).

**DEV-only choices:** checkpoint/epoch and label top-k (DEV macro-F1 over seen hypotheses),
temperature (DEV NLL), evidence threshold (DEV segment F1), serving backbone (DEV score of the
primary vs the ablation), cascade tau. TEST is predicted only after the selection is frozen and
is never used to change a choice.

Engineering safeguards: parameters are always float32 (transformers 5 would otherwise load
DeBERTa-v3's fp16 checkpoint as fp16 parameters, which broke the first Colab run of `4cd76d8`;
mixed precision comes only from autocast, and training and export refuse non-float32 models);
fp16 skipped updates don't advance the LR schedule; checkpoints are
promoted atomically and recovered after an interrupted promotion; a resume must match the data
version, derived tokenizer hash and training-code fingerprint; production configs refuse subset
data and CPU; the derived DeBERTa tokenizer is verified against its pinned sources and converter
(NFKC normalization applied, measured against the original SentencePiece model on TRAIN text).

## Evaluation (`ml.eval.evaluate`)

Every system is scored by the same code on the same 2,091 TEST pairs:

- label accuracy, macro-F1 (document-bootstrap 95 % CI), per-class P/R/F1, confusion matrix,
  per-hypothesis scores; unanswered pairs count as wrong;
- evidence on gold entailment/contradiction pairs with **end-to-end denominators** (all gold units
  of the document): segment P/R/F1, mAP, P@k, and the same projected onto official spans; invalid
  claimed evidence ids count as false positives;
- calibration: ECE before/after DEV temperature scaling, reliability diagram;
- cascade: coverage/accuracy curves, DEV-selected tau, LLM cost per document and latency;
- the unseen-hypothesis experiment and the backbone ablation.

Candidate scopes differ (stated in `baselines_comparison.json`): TF-IDF and LLM read the BGE top 8;
the model scores all segments. A `rule_check_model_top8_candidates` variant (same weights, label
top-k restricted to the BGE top 8) isolates head quality.

### Classic baselines (measured)

`make baselines` on the full official data (`data_version 620358b51ef96fac`), TEST, 2,091 pairs:

| System | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority (global TRAIN label: entailment) | 0.463 | 0.211 |
| Majority per hypothesis | 0.659 | 0.617 |
| TF-IDF + logistic regression (C=30, balanced; chosen on DEV) | 0.763 | 0.723 (0.722885) |

These were also recomputed independently with scikit-learn by the supervisor (exact agreement).
The TF-IDF label grid was extended to C ∈ {30, 100} after an earlier run selected its upper edge
(C=10) on DEV; that extension was DEV-driven. No model or LLM result exists yet.

## Export and serving

`make export-model` exports the selected checkpoint to ONNX fp32 (TorchScript exporter, opset 17)
and int8 (ONNX Runtime dynamic quantization), checks parity against PyTorch on seeded DEV pairs
(fails above 1e-3 for fp32), measures int8 **batch sensitivity** (dynamic int8 activation scales
are per input tensor, so int8 outputs depend slightly on co-batched rows) and writes a manifest
with SHA-256 of every file. The model version includes a hash of the served parameters; after a
later DEV selection (e.g. the final cascade tau) `--refresh-manifest` updates parameters without
re-exporting weights. `make install-model` / `make import-colab-bundle` verify and install it as
`.cache/models/rule-check/current`; the model service loads it (int8 by default, fp32 with
`RULE_CHECK_VARIANT=fp32`), refuses smoke artifacts unless `RULE_CHECK_ALLOW_SMOKE=true`, and
micro-batches rows across concurrent requests.

Guards from the first production run (int8 collapsed; every LLM request was rejected):

- **int8 quality gate.** A production int8 file is refused for serving unless the export's DEV
  parity shows label agreement >= 0.9 and top-k agreement >= 0.75 with PyTorch; the error names
  `RULE_CHECK_VARIANT=fp32` when fp32 exists. The export reports `int8_ok` and refuses
  `--drop-fp32` when int8 fails.
- **fp32 recovery.** `python -m ml.export.recover_fp32 --source <artifact> --out <new dir>
  --quantization-eval <quantization_eval.json> [--params <inference_params.json>]` writes a
  verified fp32-only artifact (copies, never links) after checking recorded fp32 parity and
  full-split ONNX fidelity; the source is untouched.
- **Cascade tau.** An LLM baseline answering < 95 % of pairs on a split is excluded (not a
  baseline, not cascade input) and `select_tau` refuses LLM-only DEV accuracy 0. The exporter
  serves a tau only when `inference_params.json` records `llm_baseline_usable: true`.
- **LLM request errors.** Each rejected attempt stores a sanitized `error_detail` (status, API
  error type, truncated message with credentials redacted, request id). A one-request preflight
  and a 10-consecutive-rejection breaker stop the run (exit 3, no predictions written).
- Where symlinks are unsupported (exFAT), installation makes `current` a verified copy.

## Make targets

```bash
make ml-data             # official splits -> .cache/ml/data/full (~20 min CPU; done once)
make baselines           # majority + TF-IDF on the prepared data
make baseline-llm SPLIT=dev LLM_ARGS=--dry-run   # prompts + rough cost; then without --dry-run (needs ANTHROPIC_API_KEY)
make train CONFIG=ml/train/configs/deberta_v3_base.yaml RUN_DIR=<dir>   # CUDA only
make eval-model RUN_DIR=<primary> ABLATION_RUN_DIR=<modernbert> UNSEEN_RUN_DIR=<unseen>
make export-model RUN_DIR=<chosen> ARTIFACT_DIR=<dir>
make bench-model / make eval-onnx                # CPU latency / ONNX TEST accuracy
make ml-test                                     # unit tests (base env)
make ml-smoke                                    # CPU end-to-end smoke (~2-3 min; SMOKE outputs only)
make import-colab-bundle BUNDLE=<tar>            # verified import of Colab results + artifact
```

## Limitations (current)

- No production training run or real Anthropic result exists yet (external steps).
- Dynamic int8 is batch-dependent; static calibrated quantization would remove that.
- The evaluation is on canonical ContractNLI text, not PDF/OCR output.
- CUAD is not used in this phase (the clause-extraction evaluation belongs to Phase 4).
