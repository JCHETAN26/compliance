# ml/ — rule-check model (NOT IMPLEMENTED YET)

Planned for **Phase 3** of `docs/BUILD_HANDOFF.md`: ContractNLI/CUAD download scripts
(`data/`), baselines and training (`train/`), evaluation and calibration (`eval/`),
ONNX export + int8 quantization (`export/`), and a Colab notebook (`notebooks/`).

Nothing here is implemented in Phase 0: no datasets, weights or results exist yet.
`make train`, `make eval-model` and `make export-model` exit with a "not implemented"
error. Raw data and model artifacts must never be committed (see `.gitignore`).
