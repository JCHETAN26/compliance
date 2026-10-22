"""Accuracy of the exported ONNX variants on full official splits (quantization effect).

    uv run --frozen --with-requirements ml/requirements-tools.txt python -m ml.export.eval_onnx \
        --artifact-dir <artifact> --run-dir <run> --split test --out <json>

Runs the full inference pipeline with ONNX Runtime on CPU for each variant present in
the artifact (int8, and fp32 if kept), converts with the artifact's DEV-fitted
parameters (k, temperature, evidence threshold) and scores with ``ml.eval.scoring``.
The training run's own predictions (PyTorch, CUDA fp16 autocast) are scored with the
same parameters for reference, so the table shows the accuracy change from export and
from int8 quantization.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from compliance_model_service.rule_check.artifact import load_artifact
from compliance_model_service.rule_check.encoding import InputEncoder, sigmoid, softmax
from compliance_model_service.rule_check.labels import LABELS
from compliance_model_service.rule_check.onnx_backend import OnnxBackend
from ml.common import provenance
from ml.common.io import write_json
from ml.data.store import load_dataset
from ml.eval import scoring
from ml.eval.infer import PairPrediction, TokenCache, load_predictions, predict_pairs
from ml.eval.systems import SystemPrediction


def log(message: str) -> None:
    print(message, flush=True)  # noqa: T201


def convert(
    preds: list[PairPrediction], k: int, temperature: float, threshold: float
) -> list[SystemPrediction]:
    rows = []
    for p in preds:
        probs = softmax(np.array(p.label_logits[k]), temperature)
        scores = [float(x) for x in sigmoid(np.array(p.evidence_logits))]
        rows.append(
            SystemPrediction(
                p.pair_id,
                LABELS[int(np.argmax(probs))],
                probs=[float(x) for x in probs],
                evidence_scores=scores,
                evidence_set=[i for i, s in enumerate(scores) if s >= threshold],
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score ONNX fp32/int8 on an official split")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    artifact = load_artifact(args.artifact_dir, variant="int8", allow_smoke=True)
    manifest = artifact.manifest
    inf = manifest.inference
    data_dir = args.data_dir or provenance.REPO / ".cache" / "ml" / "data" / (
        "full" if manifest.kind == "production" else "smoke"
    )
    dataset = load_dataset(data_dir, splits=(args.split,))
    if dataset.data_version != manifest.provenance.get("data_version"):
        raise SystemExit("data version differs from the artifact's")
    encoder = InputEncoder.from_file(artifact.tokenizer_path, manifest.encoder.to_config())
    tokens = TokenCache(dataset, encoder)
    pairs = dataset.pairs[args.split]
    results: dict[str, Any] = {}
    reference = load_predictions(args.run_dir / f"predictions_{args.split}.jsonl.gz")
    by_id = {p.pair_id: p for p in reference}
    ordered = [by_id[p.pair_id] for p in pairs]
    if inf.top_k not in ordered[0].label_logits:
        raise SystemExit(f"run predictions lack k={inf.top_k}")
    results["pytorch_training_run"] = {
        "note": "the training run's predictions (CUDA, fp16 autocast) with the artifact's parameters",
        "scores": scoring.score(
            dataset,
            pairs,
            convert(ordered, inf.top_k, inf.temperature, inf.evidence_threshold),
            bootstrap=False,
        ),
    }
    agreement: dict[str, float] = {}
    for variant, entry in (
        ("onnx_int8", manifest.files.onnx_int8),
        ("onnx_fp32", manifest.files.onnx_fp32),
    ):
        if entry is None or not (args.artifact_dir / entry.path).is_file():
            results[variant] = "not in the artifact"
            continue
        log(f"{variant}: full {args.split} inference on CPU ...")
        backend = OnnxBackend(args.artifact_dir / entry.path, encoder, threads=args.threads)
        preds = predict_pairs(
            pairs, tokens, backend, top_k_values=[inf.top_k], chunk_rows=4096, progress=log
        )
        converted = convert(preds, inf.top_k, inf.temperature, inf.evidence_threshold)
        results[variant] = {"scores": scoring.score(dataset, pairs, converted, bootstrap=False)}
        ref = convert(ordered, inf.top_k, inf.temperature, inf.evidence_threshold)
        agreement[variant] = float(
            np.mean([a.label == b.label for a, b in zip(converted, ref, strict=True)])
        )
    summary = {
        name: {
            "macro_f1": r["scores"]["label"]["macro_f1"],
            "accuracy": r["scores"]["label"]["accuracy"],
        }
        for name, r in results.items()
        if isinstance(r, dict)
    }
    write_json(
        args.out,
        {
            "scope": "SMOKE artifact"
            if manifest.kind == "smoke"
            else f"official {args.split} split, all pairs",
            "model_version": manifest.model_version,
            "split": args.split,
            "summary": summary,
            "label_agreement_with_training_run": agreement,
            "results": results,
            "provenance": provenance.collect(sys.argv, seed=None),
        },
    )
    log(f"wrote {args.out}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
