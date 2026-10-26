"""Guards added after the first Kaggle production run (2026-10-24 recovery).

That run trained well but: every Anthropic baseline request returned HTTP 400 with the
response body discarded; tau was "selected" against the resulting LLM-only accuracy of
0; and the int8 export collapsed (DEV parity label agreement 0.083, TEST macro-F1 .063)
while fp32 kept quality. These tests pin the fixes: sanitized error diagnostics and
fail-fast on systemic rejections, no final cascade selection against an unusable LLM
baseline, int8 refused for serving on failing parity, and a validated fp32-only path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from compliance_model_service.rule_check.artifact import (
    ArtifactError,
    int8_quality_problem,
    load_artifact,
    verify_artifact,
)
from compliance_model_service.rule_check.encoding import EncoderConfig
from ml.baselines import llm
from ml.colab import bundle
from ml.data.contractnli import Document, Pair
from ml.data.store import Dataset, Ranking
from ml.eval import cascade, evaluate
from ml.eval.systems import SystemPrediction, save_system
from ml.export import recover_fp32
from ml.export.serving import inference_and_metrics, tau_not_served_reason

# ------------------------------------------------------------------ LLM diagnostics


class FakeStatusError(Exception):
    """Duck-typed like anthropic.APIStatusError (status_code, body, request_id)."""

    def __init__(self, status: int, body: object, request_id: str | None = "req_1") -> None:
        super().__init__(f"Error code: {status}")
        self.status_code = status
        self.body = body
        self.request_id = request_id
        self.response = types.SimpleNamespace(headers={})


class FakeConnectionError(Exception):
    pass


def _body(message: str, kind: str = "invalid_request_error") -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": message}, "request_id": "req_b"}


def test_api_error_detail_is_actionable_and_sanitized() -> None:
    secret = "sk-ant-api03-SECRETSECRET_abc-123"
    exc = FakeStatusError(
        400,
        _body(f"bad x-api-key: {secret}\nAuthorization: Bearer tok.en/value= " + "x" * 600),
    )
    detail = llm.api_error_detail(exc)
    assert detail["status"] == 400
    assert detail["error_type"] == "invalid_request_error"
    assert detail["request_id"] == "req_1"
    assert detail["retryable"] is False
    text = json.dumps(detail)
    assert secret not in text
    assert "SECRETSECRET" not in text
    assert "tok.en/value" not in text
    assert "\n" not in detail["message"]
    assert detail["message"].endswith("...[truncated]")
    assert len(detail["message"]) <= llm.MAX_DIAGNOSTIC_CHARS + len("...[truncated]")
    # No body at all (what the recovered run could report): say so explicitly.
    bare = llm.api_error_detail(FakeStatusError(400, None, request_id=None))
    assert bare["message"] == "(no error message in the response body)"
    assert bare["error_type"] is None
    assert "HTTP 400" in llm.describe_detail(bare)


def test_api_error_detail_with_the_real_sdk_exception() -> None:
    anthropic = pytest.importorskip("anthropic")
    httpx = sys.modules.get("httpx2") or pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(400, request=request, headers={"request-id": "req_real"})
    exc = anthropic.BadRequestError(
        "Error code: 400", response=response, body=_body("output_config.effort: not supported")
    )
    detail = llm.api_error_detail(exc)
    assert detail == {
        "status": 400,
        "error_class": "BadRequestError",
        "error_type": "invalid_request_error",
        "message": "output_config.effort: not supported",
        "request_id": "req_real",
        "retryable": False,
    }


def _dataset(n: int) -> tuple[Dataset, list[Pair]]:
    text = "One. Two. Three."
    segs = [(0, 4), (5, 9), (10, 16)]
    doc = Document("dev", "d", "f", text, segs, segs, [(0,), (1,), (2,)])
    pairs = [Pair("dev", "d", f"nda-{i}", "entailment", (1,), (1,)) for i in range(n)]
    rank = Ranking((1, 0, 2), (0.9, 0.5, 0.1))
    ds = Dataset(
        Path("/x"),
        {"data_version": "v", "name": "unit"},
        {p.key: f"hypothesis {p.key}" for p in pairs},
        {"dev": {"d": doc}},
        {"dev": pairs},
        {"dev": {p.pair_id: rank for p in pairs}},
    )
    return ds, pairs


class ScriptedClient:
    """messages.create returns a response or raises, per call, from ``script(n, prompt)``."""

    def __init__(self, script: Any) -> None:
        self.calls: list[str] = []
        self.script = script
        self.messages = types.SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> Any:
        prompt = kwargs["messages"][0]["content"]
        self.calls.append(prompt)
        outcome = self.script(len(self.calls), prompt)
        if isinstance(outcome, Exception):
            raise outcome
        usage = types.SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        block = types.SimpleNamespace(
            type="text", text='{"label": "entailment", "evidence_ids": []}'
        )
        return types.SimpleNamespace(
            id="msg", model="m", stop_reason="end_turn", content=[block], usage=usage
        )

    async def close(self) -> None:
        pass


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The run's exception handling without the real SDK or network (no paid calls)."""
    module = types.ModuleType("anthropic")
    module.APIStatusError = FakeStatusError  # type: ignore[attr-defined]
    module.APIConnectionError = FakeConnectionError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)  # retry back-off only


def _run(ds: Dataset, pairs: list[Pair], client: ScriptedClient, cache: Path, **kw: Any) -> Any:
    return asyncio.run(
        llm.run_pairs(
            ds,
            pairs,
            model="m",
            effort="low",
            concurrency=kw.get("concurrency", 4),
            cache_path=cache,
            client=client,
        )
    )


@pytest.mark.usefixtures("fake_sdk")
def test_systemic_400_stops_after_the_preflight_request(tmp_path: Path) -> None:
    ds, pairs = _dataset(30)
    client = ScriptedClient(lambda n, p: FakeStatusError(400, _body("credit balance too low")))
    cache = tmp_path / "cache.jsonl"
    with pytest.raises(llm.SystemicRequestError, match="preflight") as info:
        _run(ds, pairs, client, cache)
    assert len(client.calls) == 1  # the other 29 pairs were never sent
    assert "credit balance too low" in info.value.reason
    rows = llm.read_attempts(cache)
    assert len(rows) == 1  # the attempt made is recorded for diagnosis and resume
    assert rows[0]["error"] == "api_status_400: FakeStatusError"
    assert rows[0]["error_detail"]["message"] == "credit balance too low"


@pytest.mark.usefixtures("fake_sdk")
def test_consecutive_rejections_after_a_good_preflight_stop_the_run(tmp_path: Path) -> None:
    ds, pairs = _dataset(40)

    def script(n: int, _: str) -> Any:
        return None if n == 1 else FakeStatusError(403, _body("forbidden", "permission_error"))

    client = ScriptedClient(script)
    with pytest.raises(llm.SystemicRequestError, match="consecutive"):
        _run(ds, pairs, client, tmp_path / "c.jsonl", concurrency=1)
    assert len(client.calls) == 1 + llm.SYSTEMIC_FAILURE_LIMIT


@pytest.mark.usefixtures("fake_sdk")
def test_isolated_rejections_and_retryable_errors_do_not_stop_the_run(tmp_path: Path) -> None:
    ds, pairs = _dataset(12)
    bad_prompt = llm.user_prompt(ds, pairs[5])
    overloaded = {"done": False}

    def script(n: int, prompt: str) -> Any:
        if n == 1 and not overloaded["done"]:  # preflight overloaded once, then answered
            overloaded["done"] = True
            return FakeStatusError(529, _body("overloaded", "overloaded_error"))
        if prompt == bad_prompt:  # one pair-specific rejection
            return FakeStatusError(400, _body("prompt too long"))
        return None

    client = ScriptedClient(script)
    attempts = _run(ds, pairs, client, tmp_path / "c.jsonl", concurrency=1)
    assert len(client.calls) == 13  # every pair requested; the 529 retried once
    req = llm.request_hash("m", "low")
    price = {"input": 1.0, "output": 1.0, "cache_write_5m": 1.0, "cache_read": 1.0}
    _, summary = llm.to_predictions(ds, pairs, attempts, price, req)
    assert summary["answered"] == 11
    assert summary["errors"] == {"api_status_400": 1}
    assert {(d["status"], d["message"]) for d in summary["error_diagnostics"]} == {
        (400, "prompt too long"),
        (529, "overloaded"),
    }


def test_main_writes_no_baseline_when_requests_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, _ = _dataset(3)
    monkeypatch.setattr(llm, "load_dataset", lambda _dir: ds)
    monkeypatch.setattr(llm, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(
        llm,
        "load_pricing",
        lambda _m: {"input": 1.0, "output": 1.0, "cache_write_5m": 1.0, "cache_read": 1.0},
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")

    async def rejected(*_: Any, **__: Any) -> Any:
        raise llm.SystemicRequestError("preflight rejected", {"status": 400})

    monkeypatch.setattr(llm, "run_pairs", rejected)
    out = tmp_path / "baselines"
    assert llm.main(["--split", "dev", "--out", str(out)]) == 3
    assert not out.exists()  # no llm_zero_shot predictions for evaluation to consume


# ------------------------------------------------------------------ cascade selection


def _rows(n: int, answered: int) -> list[SystemPrediction]:
    return [
        SystemPrediction(f"p{i}", "entailment")
        if i < answered
        else SystemPrediction(f"p{i}", None, error="api_status_400: BadRequestError")
        for i in range(n)
    ]


def test_final_tau_is_refused_against_zero_llm_accuracy() -> None:
    gold, model, conf = ["entailment"] * 4, ["entailment"] * 4, [0.9, 0.8, 0.7, 0.6]
    rows = cascade.curve(gold, model, conf, llm_label=[None] * 4, documents=1)
    with pytest.raises(cascade.UnusableLLMBaselineError, match="no usable answers"):
        cascade.select_tau(rows, llm_only_dev_accuracy=0.0)  # the recovered run's input


def test_llm_usability_requires_answers_on_every_split() -> None:
    report = cascade.llm_usability({"dev": _rows(100, 0), "test": _rows(100, 100)})
    assert report["usable"] is False
    assert report["splits"]["dev"]["errors"] == {"api_status_400": 100}
    almost = cascade.llm_usability({"dev": _rows(100, 94), "test": _rows(100, 100)})
    assert almost["usable"] is False  # below MIN_LLM_ANSWERED_FRACTION
    assert cascade.llm_usability({"dev": _rows(100, 95), "test": _rows(100, 99)})["usable"]


def test_unusable_llm_baseline_is_excluded_from_evaluation(tmp_path: Path) -> None:
    ds = types.SimpleNamespace(data_version="v")
    meta = {"data_version": "v", "partial": False}
    for split in ("dev", "test"):
        save_system(tmp_path, "llm_zero_shot", split, _rows(20, 0), meta)
        save_system(tmp_path, "tfidf_logreg", split, _rows(20, 20), meta)
    systems, notes = evaluate.load_baselines(tmp_path, ds)  # type: ignore[arg-type]
    assert "tfidf_logreg" in systems
    assert "llm_zero_shot" not in systems  # so no cascade input and no comparison
    assert notes["llm_zero_shot"]["excluded"].startswith("unusable")
    assert notes["llm_zero_shot"]["usability"]["splits"]["dev"]["answered"] == 0
    for split in ("dev", "test"):
        save_system(tmp_path, "llm_zero_shot", split, _rows(20, 20), meta)
    systems, notes = evaluate.load_baselines(tmp_path, ds)  # type: ignore[arg-type]
    assert "llm_zero_shot" in systems
    assert notes["llm_zero_shot"]["usability"]["usable"]


def test_exporter_serves_tau_only_with_usable_llm_evidence() -> None:
    base = {"top_k": 4, "temperature": 2.7, "evidence_threshold": 0.87, "scope": "s"}
    # The recovered run's parameters: non-provisional tau, written before the evidence key.
    recovered = {**base, "tau": 0.3, "tau_provisional": False}
    inference, metrics = inference_and_metrics(recovered, None)
    assert inference.tau is None
    assert metrics["tau_not_served"]["tau"] == 0.3
    assert "usable LLM" in metrics["tau_not_served"]["reason"]
    assert tau_not_served_reason({**recovered, "llm_baseline_usable": False}) is not None
    assert tau_not_served_reason({**base, "tau": 0.6, "tau_provisional": True}) is not None
    final = {**recovered, "tau": 0.8, "llm_baseline_usable": True}
    assert tau_not_served_reason(final) is None
    assert inference_and_metrics(final, None)[0].tau == 0.8


# ------------------------------------------------------------------ artifacts


def _entry(root: Path, name: str) -> dict[str, Any]:
    data = (root / name).read_bytes()
    return {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _artifact(
    root: Path,
    *,
    kind: str = "production",
    int8_agreement: float | None = 0.083,
    fp32: bool = True,
    tau: float | None = 0.3,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "tokenizer.json").write_bytes(b'{"tokenizer": true}')
    (root / "model.int8.onnx").write_bytes(b"int8 bytes")
    files = {
        "tokenizer": _entry(root, "tokenizer.json"),
        "onnx_int8": _entry(root, "model.int8.onnx"),
    }
    if fp32:
        (root / "model.fp32.onnx").write_bytes(b"fp32 bytes, larger")
        files["onnx_fp32"] = _entry(root, "model.fp32.onnx")
    parity: dict[str, Any] = {
        "fp32": {"max_abs_diff_evidence_logit": 1.2e-5, "max_abs_diff_label_logit": 4.8e-6},
        "fp32_ok": True,
    }
    if int8_agreement is not None:
        parity["int8"] = {"label_argmax_agreement": int8_agreement, "top_k_agreement": 0.0}
    manifest = {
        "format_version": 1,
        "model_version": "rule-check-unit-0e28e064-e9245464-paa1268cf",
        "kind": kind,
        "created_utc": "2026-10-24T10:00:00+00:00",
        "backbone": {"repo_id": "test"},
        "labels": ["entailment", "contradiction", "not_mentioned"],
        "encoder": EncoderConfig().to_dict(),
        "inference": {"top_k": 4, "temperature": 2.7, "evidence_threshold": 0.87, "tau": tau},
        "files": files,
        "metrics": {"test_macro_f1": 0.8138},
        "provenance": {
            "run_name": "unit",
            "config_hash": "0e28e064" + "0" * 56,
            "best_weights_sha256": "e9245464" + "0" * 56,
            "data_version": "v",
            "parity": parity,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_production_int8_with_failing_parity_is_refused_for_serving(tmp_path: Path) -> None:
    root = tmp_path / "a"
    _artifact(root)
    with pytest.raises(ArtifactError, match=r"int8 parity on DEV failed.*RULE_CHECK_VARIANT=fp32"):
        load_artifact(root, variant="int8", allow_smoke=False)
    assert load_artifact(root, variant="fp32", allow_smoke=False).variant == "fp32"
    # Integrity-only verification (tools that measure or move artifacts) still passes.
    assert verify_artifact(root)[1] == ("int8", "fp32")


def test_int8_gate_needs_passing_parity_evidence(tmp_path: Path) -> None:
    good = tmp_path / "good"
    _artifact(good, int8_agreement=1.0)
    manifest = json.loads((good / "manifest.json").read_text())
    manifest["provenance"]["parity"]["int8"]["top_k_agreement"] = 1.0
    (good / "manifest.json").write_text(json.dumps(manifest))
    assert load_artifact(good, variant="int8", allow_smoke=False).variant == "int8"
    missing = tmp_path / "missing"
    _artifact(missing, int8_agreement=None)
    with pytest.raises(ArtifactError, match="no int8 parity"):
        load_artifact(missing, variant="int8", allow_smoke=False)
    smoke = tmp_path / "smoke"  # CI smoke artifacts are not quality claims
    _artifact(smoke, kind="smoke")
    assert load_artifact(smoke, variant="int8", allow_smoke=True).variant == "int8"


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), float("-inf"), True, 1.5, -0.1, "0.95", None, 10**400],
)
def test_int8_gate_fails_closed_on_malformed_agreement(bad: object) -> None:
    for key in ("label_argmax_agreement", "top_k_agreement"):
        int8 = {"label_argmax_agreement": 1.0, "top_k_agreement": 1.0, key: bad}
        problem = int8_quality_problem({"int8": int8})
        assert problem is not None
        assert "malformed" in problem
    # Valid evidence at the thresholds (an int 1 included) still passes.
    for label in (1, 0.9):
        int8 = {"label_argmax_agreement": label, "top_k_agreement": 0.75}
        assert int8_quality_problem({"int8": int8}) is None


def test_int8_with_nan_parity_in_the_manifest_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "a"
    _artifact(root, int8_agreement=float("nan"))  # json writes NaN; the manifest accepts it
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["provenance"]["parity"]["int8"]["top_k_agreement"] = float("nan")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match=r"malformed.*RULE_CHECK_VARIANT=fp32"):
        load_artifact(root, variant="int8", allow_smoke=False)


def test_verify_artifact_accepts_a_bundle_without_the_fp32_file(tmp_path: Path) -> None:
    root = tmp_path / "a"
    _artifact(root)
    (root / "model.fp32.onnx").unlink()  # results bundles leave fp32 behind
    assert verify_artifact(root)[1] == ("int8",)
    (root / "model.int8.onnx").write_bytes(b"tampered!!")
    with pytest.raises(ArtifactError, match=r"SHA-256|bytes"):
        verify_artifact(root)


def _quantization_eval(path: Path, version: str, fp32_f1: float = 0.81276) -> Path:
    path.write_text(
        json.dumps(
            {
                "scope": "official test split, all pairs",
                "model_version": version,
                "split": "test",
                "summary": {
                    "pytorch_training_run": {"macro_f1": 0.81378, "accuracy": 0.8527},
                    "onnx_int8": {"macro_f1": 0.06346, "accuracy": 0.1052},
                    "onnx_fp32": {"macro_f1": fp32_f1, "accuracy": 0.8517},
                },
                "label_agreement_with_training_run": {"onnx_int8": 0.14, "onnx_fp32": 0.999},
            }
        )
    )
    return path


def _digest(root: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir())}


def test_fp32_recovery_produces_a_validated_fp32_only_artifact(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest = _artifact(source)
    before = _digest(source)
    qe = _quantization_eval(tmp_path / "q.json", manifest["model_version"])
    out = tmp_path / "recovered"
    result = recover_fp32.recover(source, out, qe, None)
    assert _digest(source) == before  # the source artifact is never modified
    recovered = load_artifact(out, variant="fp32", allow_smoke=False)
    m = recovered.manifest
    assert m.files.onnx_int8 is None
    assert m.files.onnx_fp32 is not None
    assert m.files.onnx_fp32.sha256 == manifest["files"]["onnx_fp32"]["sha256"]
    assert m.inference.tau is None  # the tau from the invalid selection is not served
    assert m.metrics["tau_not_served"]["tau"] == 0.3
    assert m.model_version != manifest["model_version"]
    assert m.provenance["recovery"]["source_model_version"] == manifest["model_version"]
    assert "int8 parity on DEV failed" in m.provenance["recovery"]["int8_parity_problem"]
    assert result["onnx_evaluation"]["fp32_macro_f1_drop"] == pytest.approx(0.00102)
    for name in ("tokenizer.json", "model.fp32.onnx"):
        assert not (out / name).is_symlink()  # copies (exFAT has no symlinks)
        assert (out / name).is_file()
    with pytest.raises(ArtifactError, match="no int8 ONNX file"):
        load_artifact(out, variant="int8", allow_smoke=False)
    assert not (tmp_path / ".recovered.partial").exists()


def test_fp32_recovery_keeps_tau_only_with_usable_llm_params(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest = _artifact(source, tau=0.8)
    qe = _quantization_eval(tmp_path / "q.json", manifest["model_version"])
    params = tmp_path / "params.json"
    good = {
        "top_k": 4,
        "temperature": 2.7,
        "evidence_threshold": 0.87,
        "tau": 0.8,
        "tau_provisional": False,
        "llm_baseline_usable": True,
        "data_version": "v",
    }
    params.write_text(json.dumps(good))
    recover_fp32.recover(source, tmp_path / "kept", qe, params)
    assert (
        load_artifact(tmp_path / "kept", variant="fp32", allow_smoke=False).manifest.inference.tau
        == 0.8
    )
    params.write_text(json.dumps({**good, "temperature": 1.0}))
    with pytest.raises(SystemExit, match="temperature"):
        recover_fp32.recover(source, tmp_path / "other", qe, params)
    # As in the real evaluation: a local re-fit of the DEV temperature differs in the 13th digit.
    params.write_text(json.dumps({**good, "temperature": 2.7 + 9e-14}))
    recover_fp32.recover(source, tmp_path / "refit", qe, params)
    assert (
        load_artifact(tmp_path / "refit", variant="fp32", allow_smoke=False).manifest.inference.tau
        == 0.8
    )


def test_fp32_recovery_refuses_params_for_a_source_without_tau(tmp_path: Path) -> None:
    # Re-recovering from an earlier fp32 recovery (tau None) must not silently drop --params.
    source = tmp_path / "source"
    manifest = _artifact(source, tau=None)
    qe = _quantization_eval(tmp_path / "q.json", manifest["model_version"])
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"tau": 0.8, "llm_baseline_usable": True, "data_version": "v"}))
    with pytest.raises(SystemExit, match="serves no tau"):
        recover_fp32.recover(source, tmp_path / "out", qe, params)
    assert not (tmp_path / "out").exists()


def test_fp32_recovery_refuses_unvalidated_inputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest = _artifact(source)
    version = manifest["model_version"]
    cases = {
        "wrong version": (_quantization_eval(tmp_path / "q1.json", "other"), "evaluated"),
        "fp32 lost quality": (
            _quantization_eval(tmp_path / "q2.json", version, fp32_f1=0.70),
            "not a faithful export",
        ),
    }
    for name, (qe, message) in cases.items():
        with pytest.raises(SystemExit, match=message):
            recover_fp32.recover(source, tmp_path / name.replace(" ", "_"), qe, None)
    # As in the real run: evaluated before a refresh that changed only the served tau.
    refreshed = json.loads((source / "manifest.json").read_text())
    refreshed["provenance"]["refreshes"] = [{"previous_model_version": "rule-check-pre-refresh"}]
    (source / "manifest.json").write_text(json.dumps(refreshed))
    earlier = _quantization_eval(tmp_path / "q3.json", "rule-check-pre-refresh")
    result = recover_fp32.recover(source, tmp_path / "earlier", earlier, None)
    assert result["onnx_evaluation"]["evaluated_model_version"] == "rule-check-pre-refresh"
    qe = _quantization_eval(tmp_path / "q.json", version)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "keep.txt").write_text("unrelated")
    with pytest.raises(SystemExit, match="not empty"):
        recover_fp32.recover(source, busy, qe, None)
    assert (busy / "keep.txt").read_text() == "unrelated"
    with pytest.raises(SystemExit, match="separate"):
        recover_fp32.recover(source, source / "inside", qe, None)
    no_fp32 = tmp_path / "no_fp32"
    _artifact(no_fp32, fp32=False)
    with pytest.raises(SystemExit, match="no fp32"):
        recover_fp32.recover(no_fp32, tmp_path / "x", qe, None)
    bad_parity = json.loads((source / "manifest.json").read_text())
    bad_parity["provenance"]["parity"]["fp32_ok"] = False
    (source / "manifest.json").write_text(json.dumps(bad_parity))
    with pytest.raises(SystemExit, match="fp32 parity"):
        recover_fp32.recover(source, tmp_path / "y", qe, None)


def _set_parity(source: Path, key: str, value: object) -> None:
    manifest = json.loads((source / "manifest.json").read_text())
    if value is _MISSING:
        del manifest["provenance"]["parity"]["fp32"][key]
    else:
        manifest["provenance"]["parity"]["fp32"][key] = value
    (source / "manifest.json").write_text(json.dumps(manifest))


_MISSING = object()
_MALFORMED = [float("nan"), float("inf"), True, "1e-6", None, 10**400, _MISSING]


@pytest.mark.parametrize("bad", [*_MALFORMED, -1e-6])
def test_fp32_recovery_refuses_malformed_parity(tmp_path: Path, bad: object) -> None:
    source = tmp_path / "source"
    manifest = _artifact(source)
    qe = _quantization_eval(tmp_path / "q.json", manifest["model_version"])
    _set_parity(source, "max_abs_diff_label_logit", bad)
    with pytest.raises(recover_fp32.RecoveryError, match="fp32 parity max_abs_diff_label_logit"):
        recover_fp32.recover(source, tmp_path / "out", qe, None)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("bad", [*_MALFORMED, 1.5, -0.2])
def test_fp32_recovery_refuses_malformed_evaluation_scores(tmp_path: Path, bad: object) -> None:
    source = tmp_path / "source"
    manifest = _artifact(source)
    for system in ("onnx_fp32", "pytorch_training_run"):
        qe = _quantization_eval(tmp_path / "q.json", manifest["model_version"])
        data = json.loads(qe.read_text())
        if bad is _MISSING:
            del data["summary"][system]["macro_f1"]
        else:
            data["summary"][system]["macro_f1"] = bad
        qe.write_text(json.dumps(data))
        with pytest.raises(recover_fp32.RecoveryError, match=f"{system} macro_f1"):
            recover_fp32.recover(source, tmp_path / "out", qe, None)
    assert not (tmp_path / "out").exists()


def test_fp32_recovery_refuses_unreadable_evaluation_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _artifact(source)
    for name, text, message in (
        ("list.json", "[]", "not a JSON object"),
        ("broken.json", "{", "cannot read"),
        ("version.json", '{"model_version": ["x"]}', "evaluated"),
        (
            "summary.json",
            '{"model_version": "rule-check-unit-0e28e064-e9245464-paa1268cf", "summary": []}',
            "lacks onnx_fp32",
        ),
    ):
        (tmp_path / name).write_text(text)
        with pytest.raises(recover_fp32.RecoveryError, match=message):
            recover_fp32.recover(source, tmp_path / "out", tmp_path / name, None)
    with pytest.raises(recover_fp32.RecoveryError, match="cannot read"):
        recover_fp32.recover(source, tmp_path / "out", tmp_path / "absent.json", None)


def test_fp32_recovery_never_deletes_an_existing_staging_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    manifest = _artifact(source)
    before = _digest(source)
    qe = _quantization_eval(tmp_path / "q.json", manifest["model_version"])
    # The name the previous version of the tool staged into (and deleted on every run).
    foreign = tmp_path / ".recovered.partial"
    foreign.mkdir()
    (foreign / "sentinel.txt").write_text("not the tool's")
    out = tmp_path / "recovered"

    def fail(*_: Any, **__: Any) -> None:
        raise OSError("disk full")  # after the copies, before the rename

    with monkeypatch.context() as patch:
        patch.setattr(recover_fp32, "write_json", fail)
        with pytest.raises(OSError, match="disk full"):
            recover_fp32.recover(source, out, qe, None)
    assert not out.exists()
    assert list(tmp_path.glob(".recovered.partial-*")) == []  # its own staging cleaned up
    assert (foreign / "sentinel.txt").read_text() == "not the tool's"
    recover_fp32.recover(source, out, qe, None)
    assert load_artifact(out, variant="fp32", allow_smoke=False).variant == "fp32"
    assert list(tmp_path.glob(".recovered.partial-*")) == []
    assert (foreign / "sentinel.txt").read_text() == "not the tool's"
    assert _digest(source) == before


def test_install_without_symlink_support_uses_a_verified_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    manifest = _artifact(source)
    qe = _quantization_eval(tmp_path / "q.json", manifest["model_version"])
    recovered = tmp_path / "recovered"
    recover_fp32.recover(source, recovered, qe, None)
    models = tmp_path / "models"
    monkeypatch.setattr(bundle, "MODELS", models)

    def no_symlinks(self: Path, target: Any, target_is_directory: bool = False) -> None:
        raise PermissionError(1, "Operation not permitted")  # what exFAT reports

    monkeypatch.setattr(Path, "symlink_to", no_symlinks)
    # The name the previous version staged into (and deleted): not this call's to remove.
    foreign = models / f".current-new-{os.getpid()}"
    foreign.mkdir(parents=True)
    (foreign / "sentinel.txt").write_text("not the installer's")
    target = bundle.install_artifact(recovered, move=False)
    assert (foreign / "sentinel.txt").read_text() == "not the installer's"
    assert [p.name for p in models.glob(".current-new-*")] == [foreign.name]  # own cleaned
    current = models / "current"
    assert current.is_dir()
    assert not current.is_symlink()
    assert bundle._tree_digest(current) == bundle._tree_digest(target)
    assert (
        load_artifact(current, variant="fp32", allow_smoke=False).manifest.files.onnx_int8 is None
    )
    bundle.install_artifact(recovered, move=False)  # reinstall: the old copy is kept aside
    assert len(list(models.glob("current.backup-*"))) == 1
    assert bundle._tree_digest(current) == bundle._tree_digest(target)
