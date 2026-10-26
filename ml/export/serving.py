"""Served parameters and version identity of a rule-check artifact (no torch needed).

Shared by ``ml.export.export`` (export, manifest refresh) and ``ml.export.recover_fp32``.
"""

from __future__ import annotations

from typing import Any

from compliance_model_service.rule_check.artifact import InferenceSpec
from ml.common import provenance

# Max |delta logit| of ONNX fp32 against PyTorch fp32 on the export's DEV parity pairs.
FP32_TOLERANCE = 1e-3


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
    reason = tau_not_served_reason(params)
    if reason is not None and params.get("tau") is not None:
        metrics_summary["tau_not_served"] = {"tau": params.get("tau"), "reason": reason}
    inference = InferenceSpec(
        top_k=int(params["top_k"]),
        temperature=float(params["temperature"]),
        evidence_threshold=float(params["evidence_threshold"]),
        tau=None if reason is not None else params.get("tau"),
    )
    return inference, metrics_summary


def tau_not_served_reason(params: dict[str, Any]) -> str | None:
    """Why a DEV-selected tau must not be served (None: it may be).

    Only a final selection made against a usable LLM baseline is served. Parameter files
    written before ``llm_baseline_usable`` existed lack that evidence (e.g. a selection
    against an LLM baseline whose every request failed) and are refused too.
    """
    if params.get("tau") is None:
        return "no tau selected"
    if params.get("tau_provisional"):
        return "provisional model-only tau (no usable LLM DEV results)"
    if params.get("llm_baseline_usable") is not True:
        return "no evidence of a usable LLM baseline behind the selection"
    return None


def model_version(run_name: str, run: dict[str, Any], inference: InferenceSpec) -> str:
    """Weights *and* served parameters identify a version: new calibration -> new version."""
    params_hash = provenance.config_hash(inference.model_dump(mode="json"))
    return f"rule-check-{run_name}-{run['config_hash'][:8]}-{run['best_weights_sha256'][:8]}-p{params_hash[:8]}"
