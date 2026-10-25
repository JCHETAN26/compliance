# Kaggle recovery checkpoint — 2026-10-24

Read directly from the authenticated Kaggle account `cjitendr`. This supersedes
older statements that no production Kaggle training has completed; it does not
declare Phase 3 accepted.

## Remote state

| Notebook | Latest status observed |
|---|---|
| `cjitendr/compliance-rule-check-training` | ERROR |
| `cjitendr/compliance-rule-check-recovery` | COMPLETE; empty output listing |
| `cjitendr/compliance-training-fresh` | COMPLETE; saved training, evaluation and export outputs |

The completed private notebook is
https://www.kaggle.com/code/cjitendr/compliance-training-fresh.
It used private dataset `cjitendr/compliance-training-source`, source commit
`556eeb55b76cb25c5b6c91c20d52df19a57794db`, and data version
`693b452eef54fdd46c84125278db3c430fbb68f1ed85f098edb5ab4e1675a70f`.
All three run records (`deberta_v3_base`, `modernbert_base`,
`deberta_v3_base_unseen4`) say `production` and `complete`, with training-code
fingerprint `a74ec481c62e6b5f8d24ce97aded422e61df6febf10f015298ab6ba4e8fdc64d`.
No training was restarted during recovery.

## Recovered files and checks

Local recovery root, outside the Git checkout:
`/media/grv22/Extreme SSD2/project/kaggle-recovery/`.

- `inspection/fresh/compliance-runs/` holds downloaded run records, reports,
  LLM execution logs, the complete ONNX artifact including fp32, and results bundle.
- `notebook/` holds the downloaded notebook source and metadata.
- `fresh-execution-log.json` holds the saved remote execution log.
- `verification.json` holds independent prediction-score recomputation and
  verification results.

Bundle: `results-bundle-556eeb55b76c-20261024T105807492057Z.tar`,
278,353,920 bytes; SHA-256:
`7d473a586b4590babb9dc488b70ca842745c89a14afd8ce902de6a4273e31cf0`.
All 66 manifest-listed members passed size and SHA-256 verification. Archive
membership was checked for duplicates, unexpected members, traversal paths and
non-regular files. The separately downloaded tokenizer, int8 and fp32 ONNX files
also passed their artifact-manifest size and SHA-256 checks.

The bundle is preserved, not imported or activated in a running service.
Full training checkpoints remain in Kaggle's saved output; they are deliberately
excluded from the results bundle. The recovery download did not fetch them.

## Results and remaining acceptance blockers

| System | TEST accuracy | TEST macro-F1 |
|---|---:|---:|
| TF-IDF | 0.764228 | 0.724374 |
| Trained DeBERTa | 0.852702 | 0.813778 |
| ONNX fp32 | 0.851746 | 0.812763 |
| ONNX int8 | 0.105213 | 0.063465 |

TF-IDF and trained DeBERTa scores were independently recomputed from all 2,091
saved TEST predictions with a standard-library confusion-matrix calculation;
both exactly match the saved reports. ONNX scores above are read from the saved
quantization evaluation, not rerun locally. Model versus TF-IDF clears the
reported handoff performance criterion. This alone does not complete acceptance.

1. **Int8 quality is unacceptable.** The service defaults to int8; do not activate
   that default using this artifact. The fp32 export retained quality and is now
   downloaded and hash-verified. Continue with fp32 service validation, or repair
   quantization using DEV for choices and keep TEST for frozen evaluation.
2. **The real LLM baseline failed.** All 1,037 DEV and 2,091 TEST requests returned
   HTTP 400, with zero usable answers. The implementation records the status and
   exception class but omits the response body, so the precise request rejection
   is not established. Fix/diagnose the request with a bounded check before any
   full paid rerun; preserve the existing attempt cache on Kaggle.
3. **Cascade selection is invalid as evidence of a working LLM cascade.** Its
   selection used LLM-only DEV accuracy 0.0 and selected tau 0.3, giving 100% model
   coverage. Recompute after a usable LLM baseline; refresh inference parameters
   and artifact manifest afterward.
4. **Local service integration remains unverified.** This is a fresh clone with
   no project environment or running stack established by this recovery. The
   project requires Python >=3.11; the shell's default `python` is 3.10. The SSD
   filesystem rejected symlink creation during tool relocation; account for that
   when setting up a virtual environment or the importer's `current` symlink.

Do not retrain the three completed models merely to continue evaluation/export.
Do not repeat the full LLM baseline until its HTTP 400 failure has been diagnosed.
Phase 3 remains pending these acceptance checks; Phase 4 is not started here.

Downloads were moved to the project SSD after the host root filesystem reported
no free space. No unrelated files were removed, no remote notebook was modified,
and no new training or provider API call was launched.
