"""Derive a validated fp32-only artifact from an exported one (no retraining, no re-export).

    uv run --frozen python -m ml.export.recover_fp32 --source <artifact dir> --out <new dir> \
        --quantization-eval <quantization_eval.json> [--params <inference_params.json>]

For an export whose int8 file is unusable (dynamic quantization collapsed the first
production DeBERTa export to TEST macro-F1 .063) while its fp32 file kept quality. The
new artifact holds verified *copies* (never links) of the tokenizer and fp32 ONNX file
and a manifest listing no int8 file, so the default ``RULE_CHECK_VARIANT=int8`` fails
loudly and ``RULE_CHECK_VARIANT=fp32`` serves the recovered model. Checks, all required:

1. every present source file matches its manifest (size, SHA-256) and fp32 is present;
2. the source's recorded DEV parity says fp32 matches PyTorch within ``FP32_TOLERANCE``;
3. the full-split ONNX evaluation (``ml.export.eval_onnx``) is for this model version
   (or an earlier version of the same files in the manifest's refresh history) and
   shows fp32 macro-F1 within ``FP32_MAX_MACRO_F1_DROP`` of the training run's
   predictions. This verifies export fidelity; nothing is selected with it.

A cascade tau is carried over only with ``--params`` from an evaluation whose LLM
baseline was usable (``ml.export.serving.tau_not_served_reason``) and whose served
parameters equal the source's; otherwise the recovered manifest serves no tau. The
source artifact is never modified.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compliance_model_service.rule_check.artifact import (
    ArtifactError,
    ArtifactFiles,
    Manifest,
    describe_file,
    int8_quality_problem,
    load_artifact,
    verify_artifact,
)
from ml.common import provenance
from ml.common.io import read_json, write_json
from ml.export.serving import FP32_TOLERANCE, model_version, tau_not_served_reason

FP32_MAX_MACRO_F1_DROP = 0.01
REPORT = "recovery_report.json"


class RecoveryError(SystemExit):
    pass


def log(message: str) -> None:
    print(message, flush=True)  # noqa: T201


def _finite(value: object, what: str) -> float:
    """``value`` as a finite float; RecoveryError for bool, non-numbers, NaN and infinities."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RecoveryError(f"{what} is {value!r}, not a number")
    try:
        number = float(value)
    except OverflowError:  # an int too large for a float
        number = math.inf
    if not math.isfinite(number):
        raise RecoveryError(f"{what} is {value!r}, not a finite number")
    return number


def check_fp32_parity(manifest: Manifest) -> dict[str, Any]:
    parity = manifest.provenance.get("parity")
    if (
        not isinstance(parity, dict)
        or parity.get("fp32_ok") is not True
        or not isinstance(parity.get("fp32"), dict)
    ):
        raise RecoveryError("the source manifest records no passing fp32 parity check")
    diffs = []
    for key in ("max_abs_diff_evidence_logit", "max_abs_diff_label_logit"):
        diff = _finite(parity["fp32"].get(key), f"the source manifest's fp32 parity {key}")
        if diff < 0:
            raise RecoveryError(
                f"the source manifest's fp32 parity {key} is {diff}; a maximum absolute "
                "difference cannot be negative (malformed parity record)"
            )
        diffs.append(diff)
    worst = max(diffs)
    if worst > FP32_TOLERANCE:
        raise RecoveryError(f"recorded fp32 parity {worst} exceeds {FP32_TOLERANCE}")
    return {"max_abs_diff_logit": worst, "tolerance": FP32_TOLERANCE}


def check_quantization_eval(manifest: Manifest, path: Path) -> dict[str, Any]:
    try:
        data = read_json(path)
    except (OSError, ValueError) as exc:
        raise RecoveryError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RecoveryError(f"{path} is not an ml.export.eval_onnx result (not a JSON object)")
    # A manifest refresh changes only served parameters (files unchanged and re-verified),
    # so an evaluation of an earlier version in this manifest's own history is valid.
    refreshes = manifest.provenance.get("refreshes")
    same_files = {manifest.model_version} | {
        r.get("previous_model_version")
        for r in (refreshes if isinstance(refreshes, list) else [])
        if isinstance(r, dict)
    }
    evaluated = data.get("model_version")
    if not isinstance(evaluated, str) or evaluated not in same_files:
        raise RecoveryError(
            f"{path} evaluated {evaluated!r}, not {manifest.model_version!r} "
            "or an earlier version of the same files"
        )
    summary = data.get("summary")
    fp32 = summary.get("onnx_fp32") if isinstance(summary, dict) else None
    reference = summary.get("pytorch_training_run") if isinstance(summary, dict) else None
    if not isinstance(fp32, dict) or not isinstance(reference, dict):
        raise RecoveryError(f"{path} lacks onnx_fp32 or pytorch_training_run scores")
    scores: dict[str, float] = {}
    for name, system in (("pytorch_training_run", reference), ("onnx_fp32", fp32)):
        score = _finite(system.get("macro_f1"), f"{path} {name} macro_f1")
        if not 0 <= score <= 1:
            raise RecoveryError(f"{path} {name} macro_f1 is {score}, outside [0, 1]")
        scores[name] = score
    drop = scores["pytorch_training_run"] - scores["onnx_fp32"]
    if drop > FP32_MAX_MACRO_F1_DROP:
        raise RecoveryError(
            f"ONNX fp32 macro-F1 is {drop:.4f} below the training run's "
            f"(limit {FP32_MAX_MACRO_F1_DROP}); not a faithful export"
        )
    return {
        "file": str(path),
        "sha256": provenance.file_sha256(path),
        "evaluated_model_version": data.get("model_version"),
        "scope": data.get("scope"),
        "split": data.get("split"),
        "summary": summary,
        "label_agreement_with_training_run": data.get("label_agreement_with_training_run"),
        "fp32_macro_f1_drop": drop,
        "max_allowed_drop": FP32_MAX_MACRO_F1_DROP,
    }


def recovered_inference(
    manifest: Manifest, params_path: Path | None
) -> tuple[dict[str, Any], str | None]:
    """The source's served parameters; tau kept only with valid evidence (else reason)."""
    inference = manifest.inference.model_dump(mode="json")
    if inference.get("tau") is None:
        return inference, None
    reason: str | None
    if params_path is None:
        reason = "no --params: the source's tau has no recorded evidence of a usable LLM baseline"
    else:
        params = read_json(params_path)
        if params.get("data_version") != manifest.provenance.get("data_version"):
            raise RecoveryError(f"{params_path} is for another data version")
        for key in ("top_k", "temperature", "evidence_threshold", "tau"):
            if params.get(key) != inference[key]:
                raise RecoveryError(
                    f"{params_path} {key}={params.get(key)} differs from the source"
                )
        reason = tau_not_served_reason(params)
    if reason is not None:
        inference = {**inference, "tau": None}
    return inference, reason


def recover(
    source: Path, out: Path, quantization_eval: Path, params: Path | None
) -> dict[str, Any]:
    manifest, present = verify_artifact(source)
    if "fp32" not in present or manifest.files.onnx_fp32 is None:
        raise RecoveryError(f"{source} has no fp32 ONNX file to recover")
    src, dst = source.resolve(), out.resolve()
    if src == dst or dst.is_relative_to(src) or src.is_relative_to(dst):
        raise RecoveryError("--out must be a separate directory from --source")
    if out.exists() and any(out.iterdir()):
        raise RecoveryError(f"{out} exists and is not empty")
    parity = check_fp32_parity(manifest)
    evaluation = check_quantization_eval(manifest, quantization_eval)
    inference, tau_reason = recovered_inference(manifest, params)
    run = {
        "config_hash": manifest.provenance["config_hash"],
        "best_weights_sha256": manifest.provenance["best_weights_sha256"],
    }
    run_name = str(manifest.provenance.get("run_name", "unknown"))
    version = model_version(run_name, run, manifest.inference.model_copy(update=inference))

    out.parent.mkdir(parents=True, exist_ok=True)
    # A fresh, uniquely named staging directory owned by this call: an existing directory
    # (an interrupted earlier run's, or anyone else's) is never reused or deleted.
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=out.parent))
    try:
        with contextlib.suppress(OSError):  # mkdtemp is 0o700; exFAT has fixed modes
            staging.chmod(0o755)
        for entry in (manifest.files.tokenizer, manifest.files.onnx_fp32):
            shutil.copyfile(source / entry.path, staging / entry.path)  # a copy, never a link
        metrics = {
            **manifest.metrics,
            "onnx_fp32": evaluation["summary"]["onnx_fp32"],
            "onnx_int8_not_served": evaluation["summary"].get("onnx_int8"),
        }
        if tau_reason is not None:
            metrics["tau_not_served"] = {"tau": manifest.inference.tau, "reason": tau_reason}
        recovery = {
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "source_model_version": manifest.model_version,
            "source_manifest_sha256": provenance.file_sha256(source / "manifest.json"),
            "reason": "fp32-only recovery; int8 not carried over",
            "int8_parity_problem": int8_quality_problem(manifest.provenance.get("parity")),
            "fp32_parity": parity,
            "onnx_evaluation": evaluation,
            "tau_not_served_reason": tau_reason,
            "provenance": provenance.collect(sys.argv, seed=None),
        }
        data = manifest.model_dump(mode="json")
        data.update(
            model_version=version,
            created_utc=recovery["created_utc"],
            inference=inference,
            files=ArtifactFiles(
                tokenizer=describe_file(staging / manifest.files.tokenizer.path),
                onnx_fp32=describe_file(staging / manifest.files.onnx_fp32.path),
            ).model_dump(mode="json"),
            metrics=metrics,
            provenance={**manifest.provenance, "recovery": recovery},
        )
        recovered = Manifest.model_validate(data)
        if recovered.files.onnx_fp32 != manifest.files.onnx_fp32:
            raise RecoveryError("the copied fp32 file differs from the source's verified file")
        write_json(staging / "manifest.json", recovered.model_dump(mode="json"))
        write_json(staging / REPORT, {"model_version": version, **recovery})
        if out.exists():
            out.rmdir()  # empty (checked above)
        staging.rename(out)
    finally:
        if staging.exists():  # only after a failure: success renamed it to ``out``
            shutil.rmtree(staging)  # this call's own staging directory, nothing else

    allow_smoke = recovered.kind == "smoke"
    load_artifact(out, variant="fp32", allow_smoke=allow_smoke)  # servable as fp32
    try:
        load_artifact(out, variant="int8", allow_smoke=allow_smoke)
    except ArtifactError:
        pass
    else:  # pragma: no cover - the manifest above lists no int8 file
        raise RecoveryError("the recovered artifact unexpectedly serves int8")
    log(
        f"recovered {manifest.model_version} -> {version} in {out} (fp32 only; tau "
        f"{inference['tau']}{'' if tau_reason is None else ' - ' + tau_reason}). "
        "Serve with RULE_CHECK_VARIANT=fp32."
    )
    return {"model_version": version, **recovery}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fp32-only artifact from a verified export")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--quantization-eval",
        type=Path,
        required=True,
        help="quantization_eval.json from ml.export.eval_onnx for the source artifact",
    )
    parser.add_argument(
        "--params", type=Path, default=None, help="inference_params.json (only to keep a tau)"
    )
    args = parser.parse_args(argv)
    recover(args.source, args.out, args.quantization_eval, args.params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
