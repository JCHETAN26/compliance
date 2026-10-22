"""Export the DEV-selected checkpoint to ONNX fp32 + dynamic int8, check parity, pack.

    uv run --frozen --with-requirements ml/requirements-tools.txt python -m ml.export.export \
        --run-dir <run> --params ml/eval/results/inference_params.json --out <artifact dir>

1. Rebuild the backbone architecture from its pinned config (no pretrained download),
   load ``checkpoints/best/model.safetensors`` and export ``input_ids, attention_mask ->
   evidence_logits, label_logits`` with dynamic batch and sequence axes
   (``torch.onnx.export``, TorchScript exporter, opset 17) -> ``model.fp32.onnx``.
2. ``onnxruntime.quantization.quantize_dynamic`` (QInt8 weights; MatMul/Gather, so the
   embedding table is quantized too) -> ``model.int8.onnx``.
3. Parity on a seeded sample of DEV pairs through the full pipeline (every segment's
   evidence row plus the label row of PyTorch's own top-k): max |delta logit| of ONNX
   fp32 and int8 against PyTorch fp32 on CPU, label-argmax and top-k agreement. The
   export fails if ONNX fp32 deviates by more than ``FP32_TOLERANCE``; int8 deviations
   are reported (their effect on accuracy is measured by ``ml.export.eval_onnx``).
   Dynamic int8 quantizes activations with a scale computed over the whole input
   tensor, so an int8 row's output depends slightly on the other rows batched with it;
   ``batch_sensitivity`` measures this (batched vs one row at a time).
4. ``manifest.json`` (see ``compliance_model_service.rule_check.artifact``) with SHA-256
   of every file, the DEV-fitted inference parameters and provenance. A provisional
   (model-only) tau is NOT written as the serving tau.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from compliance_model_service.rule_check.artifact import (
    ArtifactFiles,
    EncoderSpec,
    InferenceSpec,
    Manifest,
    describe_file,
    load_artifact,
)
from compliance_model_service.rule_check.encoding import InputEncoder, sigmoid, top_k_in_order
from compliance_model_service.rule_check.labels import LABELS
from compliance_model_service.rule_check.onnx_backend import OnnxBackend
from ml.common import provenance
from ml.common.io import read_json, write_json
from ml.data.store import load_dataset
from ml.eval.infer import TokenCache, TorchForward
from ml.train.backbone import build_architecture, fetch_backbone, require_float32
from ml.train.config import TrainConfig
from ml.train.model import RuleCheckModel
from ml.train.train import _load_weights

FP32_TOLERANCE = 1e-3
OPSET = 17
PARITY_PAIRS = 24
SEED = 20261021


def log(message: str) -> None:
    print(message, flush=True)  # noqa: T201


def load_model(run_dir: Path) -> tuple[RuleCheckModel, TrainConfig, InputEncoder]:
    run = read_json(run_dir / "run.json")
    if run.get("status") != "complete":
        raise SystemExit(f"{run_dir} is not a complete training run")
    cfg = TrainConfig.model_validate(run["provenance"]["config"])
    for name, rel in (
        ("tokenizer_sha256", "tokenizer.json"),
        ("best_weights_sha256", "checkpoints/best/model.safetensors"),
    ):
        if provenance.file_sha256(run_dir / rel) != run[name]:
            raise SystemExit(f"{run_dir / rel} does not match the run's recorded {name}")
    if cfg.kind == "production":
        manifest = read_json(provenance.REPO / cfg.data.data_dir / "manifest.json")
        if manifest.get("name") != "full" or manifest.get("data_version") != run["data_version"]:
            raise SystemExit("a production artifact needs the run's full official data version")
    files = fetch_backbone(cfg.backbone, need_weights=False)
    model = RuleCheckModel(build_architecture(cfg.backbone, files), dropout=cfg.training.dropout)
    _load_weights(model, run_dir / "checkpoints" / "best" / "model.safetensors")
    require_float32(model)  # the fp32 ONNX export and parity reference must be float32
    model.eval()
    encoder = InputEncoder.from_file(run_dir / "tokenizer.json", cfg.encoder.to_config())
    return model, cfg, encoder


def export_fp32(model: RuleCheckModel, encoder: InputEncoder, path: Path) -> None:
    import torch

    ids, mask = encoder.pad([[encoder.cls_id, 100, 200, encoder.sep_id, 300, encoder.sep_id]] * 2)
    torch.onnx.export(
        model,
        (torch.from_numpy(ids), torch.from_numpy(mask)),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["evidence_logits", "label_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "evidence_logits": {0: "batch"},
            "label_logits": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
    )


def quantize_int8(fp32: Path, int8: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore[import-untyped]

    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)


def parity(
    run_dir: Path, cfg: TrainConfig, model: RuleCheckModel, encoder: InputEncoder, k: int, out: Path
) -> dict[str, Any]:
    dataset = load_dataset(provenance.REPO / cfg.data.data_dir, splits=("dev",))
    tokens = TokenCache(dataset, encoder)
    pairs = random.Random(SEED).sample(  # noqa: S311 - reproducible sampling
        dataset.pairs["dev"], min(PARITY_PAIRS, len(dataset.pairs["dev"]))
    )
    torch_fwd = TorchForward(
        model, encoder, device="cpu", fp16=False, max_tokens=cfg.inference.max_tokens_per_forward
    )
    rows: list[list[int]] = []
    spans: list[tuple[int, int]] = []
    for pair in pairs:
        hyp = tokens.hypotheses[pair.key]
        start = len(rows)
        rows.extend(encoder.evidence_input(hyp, s).ids for s in tokens.segments("dev", pair.doc_id))
        spans.append((start, len(rows)))
    ev_ref, _ = torch_fwd(rows)
    label_rows = []
    topk_ref = []
    for pair, (a, b) in zip(pairs, spans, strict=True):
        chosen = top_k_in_order([float(x) for x in sigmoid(ev_ref[a:b])], min(k, b - a))
        topk_ref.append(chosen)
        segs = tokens.segments("dev", pair.doc_id)
        label_rows.append(
            encoder.label_input(tokens.hypotheses[pair.key], [segs[i] for i in chosen]).ids
        )
    _, lab_ref = torch_fwd(label_rows)
    report: dict[str, Any] = {
        "pairs": len(pairs),
        "evidence_rows": len(rows),
        "sample": "seeded DEV pairs",
    }
    for variant in ("fp32", "int8"):
        backend = OnnxBackend(
            out / f"model.{variant}.onnx",
            encoder,
            threads=4,
            max_tokens_per_forward=cfg.inference.max_tokens_per_forward,
        )
        ev, _ = backend(rows)
        _, lab = backend(label_rows)
        topk_same = 0
        for (a, b), ref in zip(spans, topk_ref, strict=True):
            topk_same += top_k_in_order([float(x) for x in sigmoid(ev[a:b])], min(k, b - a)) == ref
        report[variant] = {
            "max_abs_diff_evidence_logit": float(np.abs(ev - ev_ref).max()),
            "max_abs_diff_label_logit": float(np.abs(lab - lab_ref).max()),
            "label_argmax_agreement": float(np.mean(lab.argmax(1) == lab_ref.argmax(1))),
            "top_k_agreement": topk_same / len(pairs),
        }
    int8 = OnnxBackend(
        out / "model.int8.onnx",
        encoder,
        threads=4,
        max_tokens_per_forward=cfg.inference.max_tokens_per_forward,
    )
    sample = rows[:: max(1, len(rows) // 64)][:64]
    batched, _ = int8(sample)
    single = np.concatenate([int8([r])[0] for r in sample])
    report["int8"]["batch_sensitivity"] = {
        "rows": len(sample),
        "max_abs_diff_evidence_logit_batched_vs_single": float(np.abs(batched - single).max()),
        "note": "dynamic int8 activation scales are per input tensor (batch-dependent)",
    }
    fp32 = report["fp32"]
    report["fp32_tolerance"] = FP32_TOLERANCE
    report["fp32_ok"] = (
        max(fp32["max_abs_diff_evidence_logit"], fp32["max_abs_diff_label_logit"]) <= FP32_TOLERANCE
    )
    return report


def inference_and_metrics(
    params: dict[str, Any], model_eval: dict[str, Any] | None
) -> tuple[InferenceSpec, dict[str, Any]]:
    """The served parameters (all DEV-fitted) and an informational metrics summary."""
    metrics_summary: dict[str, Any] = {"scope": params.get("scope")}
    if model_eval is not None:
        test = model_eval["test"]
        metrics_summary.update(
            test_macro_f1=test["label"]["macro_f1"],
            test_accuracy=test["label"]["accuracy"],
            test_evidence_segment_f1=test["evidence"].get("segment", {}).get("f1"),
            test_ece_after_temperature=model_eval["calibration"]["test_after"]["ece"],
        )
    if params.get("tau_provisional"):
        metrics_summary["tau_provisional_not_served"] = params.get("tau")
    inference = InferenceSpec(
        top_k=int(params["top_k"]),
        temperature=float(params["temperature"]),
        evidence_threshold=float(params["evidence_threshold"]),
        tau=None if params.get("tau_provisional") else params.get("tau"),
    )
    return inference, metrics_summary


def model_version(run_name: str, run: dict[str, Any], inference: InferenceSpec) -> str:
    """Weights *and* served parameters identify a version: new calibration -> new version."""
    params_hash = provenance.config_hash(inference.model_dump(mode="json"))
    return f"rule-check-{run_name}-{run['config_hash'][:8]}-{run['best_weights_sha256'][:8]}-p{params_hash[:8]}"


def _check_params(
    run_dir: Path, run: dict[str, Any], params: dict[str, Any], params_path: Path
) -> None:
    if params["run_dir"] and Path(params["run_dir"]).resolve() != run_dir.resolve():
        raise SystemExit(f"{params_path} belongs to {params['run_dir']}, not {run_dir}")
    if params["data_version"] != run["data_version"]:
        raise SystemExit("inference parameters and run use different data versions")


def refresh_manifest(args: argparse.Namespace) -> int:
    """Rewrite only the manifest's served parameters/metrics after a later DEV selection
    (e.g. the final cascade tau once real LLM results exist). ONNX files are unchanged
    and re-verified; the refresh is recorded."""
    run = read_json(args.run_dir / "run.json")
    params = read_json(args.params)
    _check_params(args.run_dir, run, params, args.params)
    old = load_artifact(args.out, variant="int8", allow_smoke=True).manifest
    if old.provenance.get("best_weights_sha256") != run["best_weights_sha256"]:
        raise SystemExit("the artifact was not exported from this run's selected weights")
    inference, metrics_summary = inference_and_metrics(
        params, read_json(args.model_eval) if args.model_eval else None
    )
    version = model_version(old.provenance.get("run_name", run["run_name"]), run, inference)
    data = old.model_dump(mode="json")
    data.update(
        model_version=version,
        inference=inference.model_dump(mode="json"),
        metrics=metrics_summary,
        provenance={
            **old.provenance,
            "refreshes": [
                *old.provenance.get("refreshes", []),
                {
                    "previous_model_version": old.model_version,
                    "previous_inference": old.inference.model_dump(mode="json"),
                    "params_sha256": provenance.file_sha256(args.params),
                    "provenance": provenance.collect(sys.argv, seed=None),
                },
            ],
        },
    )
    Manifest.model_validate(data)
    write_json(args.out / "manifest.json", data)
    load_artifact(args.out, variant="int8", allow_smoke=True)
    report_path = args.out / "export_report.json"
    if report_path.is_file():
        write_json(report_path, {**read_json(report_path), "manifest_model_version": version})
    log(f"manifest refreshed: {old.model_version} -> {version} (weights unchanged)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ONNX fp32/int8 export + parity + artifact")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--params", type=Path, required=True, help="inference_params.json from ml.eval.evaluate"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--model-eval",
        type=Path,
        default=None,
        help="model_eval.json (summary metrics for the manifest)",
    )
    parser.add_argument(
        "--drop-fp32", action="store_true", help="do not keep model.fp32.onnx in the artifact"
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="only update the existing artifact's served parameters/metrics (no re-export)",
    )
    args = parser.parse_args(argv)
    if args.refresh_manifest:
        return refresh_manifest(args)

    run = read_json(args.run_dir / "run.json")
    params = read_json(args.params)
    _check_params(args.run_dir, run, params, args.params)
    model, cfg, encoder = load_model(args.run_dir)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    log("exporting ONNX fp32 ...")
    export_fp32(model, encoder, out / "model.fp32.onnx")
    log("quantizing to int8 (dynamic) ...")
    quantize_int8(out / "model.fp32.onnx", out / "model.int8.onnx")
    shutil.copyfile(args.run_dir / "tokenizer.json", out / "tokenizer.json")
    log("parity checks ...")
    report = parity(args.run_dir, cfg, model, encoder, int(params["top_k"]), out)
    report["sizes_bytes"] = {
        "pytorch_fp32_safetensors": (args.run_dir / "checkpoints" / "best" / "model.safetensors")
        .stat()
        .st_size,
        "onnx_fp32": (out / "model.fp32.onnx").stat().st_size,
        "onnx_int8": (out / "model.int8.onnx").stat().st_size,
    }
    log(f"parity: {report}")
    if not report["fp32_ok"]:
        raise SystemExit(
            f"ONNX fp32 deviates from PyTorch beyond {FP32_TOLERANCE}; artifact not written"
        )

    inference, metrics_summary = inference_and_metrics(
        params, read_json(args.model_eval) if args.model_eval else None
    )
    files = ArtifactFiles(
        tokenizer=describe_file(out / "tokenizer.json"),
        onnx_int8=describe_file(out / "model.int8.onnx"),
        onnx_fp32=None if args.drop_fp32 else describe_file(out / "model.fp32.onnx"),
    )
    if args.drop_fp32:
        (out / "model.fp32.onnx").unlink()
    manifest = Manifest(
        format_version=1,
        model_version=model_version(cfg.run_name, run, inference),
        kind=cfg.kind,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        backbone={
            "repo_id": cfg.backbone.repo_id,
            "revision": cfg.backbone.revision,
            "kind": cfg.backbone.kind,
        },
        labels=LABELS,
        encoder=EncoderSpec(**cfg.encoder.model_dump()),
        inference=inference,
        files=files,
        metrics=metrics_summary,
        provenance={
            "run_name": cfg.run_name,
            "config_hash": run["config_hash"],
            "data_version": run["data_version"],
            "best_weights_sha256": run["best_weights_sha256"],
            "selection": run["selection"],
            "training_git_head": run["provenance"].get("git_head"),
            "export": provenance.collect(sys.argv, seed=SEED),
            "parity": report,
        },
    )
    write_json(out / "manifest.json", manifest.model_dump(mode="json"))
    load_artifact(out, variant="int8", allow_smoke=True)  # verify what was written
    write_json(
        out / "export_report.json", {"manifest_model_version": manifest.model_version, **report}
    )
    log(f"artifact written to {out} ({manifest.model_version}, kind={manifest.kind})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
