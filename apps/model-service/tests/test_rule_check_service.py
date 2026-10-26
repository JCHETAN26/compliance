"""Artifact verification and the /v1/check contract with a deterministic backend.

The backend here is a fake *model* (logits computed from token ids) behind the real
encoder, engine, micro-batcher, FastAPI routes and response schemas, so selection,
mapping, batching and errors are checked exactly. Real ONNX inference with an exported
artifact is covered by ``ml/tests_tools`` (the CI ML smoke).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from compliance_core.config import Settings
from compliance_model_service.main import RuleCheckRuntime, Runtime, create_app
from compliance_model_service.rule_check.artifact import (
    ArtifactError,
    InferenceSpec,
    load_artifact,
)
from compliance_model_service.rule_check.batching import MicroBatcher
from compliance_model_service.rule_check.encoding import EncoderConfig, InputEncoder
from compliance_model_service.rule_check.engine import RuleCheckEngine

KEYWORD = "share"


def _write_artifact(root: Path, tokenizer: Path, **override: Any) -> dict[str, Any]:
    import hashlib

    root.mkdir(parents=True, exist_ok=True)
    (root / "tokenizer.json").write_bytes(tokenizer.read_bytes())
    (root / "model.int8.onnx").write_bytes(b"not really onnx")

    def entry(name: str) -> dict[str, Any]:
        data = (root / name).read_bytes()
        return {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    manifest: dict[str, Any] = {
        "format_version": 1,
        "model_version": "rule-check-test",
        "kind": "production",
        "created_utc": "2026-10-21T00:00:00+00:00",
        "backbone": {"repo_id": "test"},
        "labels": ["entailment", "contradiction", "not_mentioned"],
        "encoder": EncoderConfig().to_dict(),
        "inference": {"top_k": 2, "temperature": 1.5, "evidence_threshold": 0.5, "tau": 0.6},
        "files": {"tokenizer": entry("tokenizer.json"), "onnx_int8": entry("model.int8.onnx")},
        # A passing export parity record: production int8 is served only with one.
        "provenance": {"parity": {"int8": {"label_argmax_agreement": 1.0, "top_k_agreement": 1.0}}},
    }
    for key, value in override.items():
        manifest[key] = value
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_artifact_verification(tmp_path: Path, model_path: Path) -> None:
    tok = model_path / "tokenizer.json"
    good = tmp_path / "good"
    _write_artifact(good, tok)
    artifact = load_artifact(good, variant="int8", allow_smoke=False)
    assert artifact.manifest.inference.top_k == 2
    with pytest.raises(ArtifactError, match="no fp32"):
        load_artifact(good, variant="fp32", allow_smoke=False)

    (good / "model.int8.onnx").write_bytes(b"tampered weights!")
    with pytest.raises(ArtifactError, match=r"SHA-256|bytes"):
        load_artifact(good, variant="int8", allow_smoke=False)

    smoke = tmp_path / "smoke"
    _write_artifact(smoke, tok, kind="smoke")
    with pytest.raises(ArtifactError, match="smoke"):
        load_artifact(smoke, variant="int8", allow_smoke=False)
    assert load_artifact(smoke, variant="int8", allow_smoke=True).manifest.kind == "smoke"

    cases: dict[str, dict[str, Any]] = {
        "labels": {"labels": ["contradiction", "entailment", "not_mentioned"]},
        "traversal": {"files": {"tokenizer": {"path": "../x", "sha256": "0" * 64, "bytes": 1}}},
        "format": {"format_version": 2},
    }
    for name, override in cases.items():
        bad = tmp_path / name
        _write_artifact(bad, tok, **override)
        with pytest.raises(ArtifactError):
            load_artifact(bad, variant="int8", allow_smoke=False)
    for value in ("Infinity", "NaN"):
        bad = tmp_path / f"temp-{value}"
        _write_artifact(bad, tok)
        raw = (
            (bad / "manifest.json")
            .read_text()
            .replace('"temperature": 1.5', f'"temperature": {value}')
        )
        (bad / "manifest.json").write_text(raw)
        with pytest.raises(ArtifactError):
            load_artifact(bad, variant="int8", allow_smoke=False)
    with pytest.raises(ArtifactError, match="no rule-check artifact"):
        load_artifact(tmp_path / "missing", variant="int8", allow_smoke=False)


class KeywordModel:
    """Evidence logit +4 if the row contains the keyword's token id, else -4.
    Label logits favour entailment when any keyword token is present."""

    def __init__(self, keyword_id: int) -> None:
        self.keyword_id = keyword_id
        self.calls = 0

    def __call__(self, rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        hit = np.array(
            [self.keyword_id in r[r.index(102) + 1 :] if 102 in r else False for r in rows]
        )
        ev = np.where(hit, 4.0, -4.0).astype(np.float32)
        lab = np.array([[3.0, 0.0, 0.0] if h else [0.0, 0.0, 3.0] for h in hit], dtype=np.float32)
        return ev, lab


@pytest.fixture
async def check_client(
    test_settings: Settings, model_path: Path
) -> AsyncIterator[httpx.AsyncClient]:
    encoder = InputEncoder.from_file(model_path / "tokenizer.json", EncoderConfig())
    keyword_id = encoder.tokenize([KEYWORD])[0][0]
    model = KeywordModel(keyword_id)

    def factory(settings: Settings) -> Runtime:
        from compliance_model_service.main import load_runtime

        runtime = load_runtime(settings.model_copy(update={"rerank_enabled": False}))
        state = RuleCheckRuntime(artifact_dir=Path("/test"))
        batcher = MicroBatcher(
            model, max_batch_rows=16, max_wait_ms=5, max_queue_rows=256, executor=state.infer_pool
        )
        inference = InferenceSpec(top_k=2, temperature=1.0, evidence_threshold=0.5, tau=0.9)
        state.batcher = batcher
        state.engine = RuleCheckEngine(encoder, batcher, inference, state.tokenize_pool)
        artifact_dir = test_settings.data_dir / "fake-artifact"
        _write_artifact(artifact_dir, model_path / "tokenizer.json")
        state.artifact = load_artifact(artifact_dir, variant="int8", allow_smoke=False)
        runtime.rule_check = state
        return runtime

    app = create_app(test_settings, runtime_factory=factory)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://model") as client:
            yield client


async def test_check_selects_evidence_and_label(check_client: httpx.AsyncClient) -> None:
    spans = [
        {"span_id": "intro", "text": "This agreement is made between the parties."},
        {"span_id": "share", "text": "The recipient may share information with employees."},
        {"span_id": "law", "text": "Governed by the laws of Delaware."},
    ]
    body = (
        await check_client.post(
            "/v1/check", json={"hypothesis": "Sharing allowed?", "spans": spans}
        )
    ).json()
    assert body["label"] == "entailment"
    assert body["status"] == "satisfied"
    assert [e["span_id"] for e in body["evidence"]] == ["share"]  # score sigmoid(4) >= 0.5
    assert body["evidence"][0]["score"] == pytest.approx(1 / (1 + np.exp(-4)))
    # top-2 by evidence score (ties -> lower index), reported in document order
    assert [c["span_id"] for c in body["considered"]] == ["intro", "share"]
    probs = np.exp([3.0, 0.0, 0.0]) / np.exp([3.0, 0.0, 0.0]).sum()
    assert body["confidence"] == pytest.approx(probs[0])
    assert body["escalate"] is bool(probs[0] < 0.9)
    assert body["model_kind"] == "production"
    assert body["variant"] == "int8"

    none = [{"span_id": s["span_id"], "text": s["text"].replace("share", "give")} for s in spans]
    body = (
        await check_client.post("/v1/check", json={"hypothesis": "Sharing allowed?", "spans": none})
    ).json()
    assert body["label"] == "not_mentioned"
    assert body["evidence"] == []


async def test_concurrent_checks_are_batched_together(check_client: httpx.AsyncClient) -> None:
    payload = {
        "hypothesis": "h",
        "spans": [{"span_id": f"s{i}", "text": f"clause {i} may share"} for i in range(3)],
    }
    responses = await asyncio.gather(
        *(check_client.post("/v1/check", json=payload) for _ in range(8))
    )
    assert all(r.status_code == 200 for r in responses)
    batching = (await check_client.get("/metadata")).json()["rule_check"]["batching"]
    assert batching["stats"]["multi_request_batches"] >= 1
    assert batching["stats"]["max_rows_in_batch"] <= 16
    health = (await check_client.get("/health")).json()
    assert health["rule_check"] == "ready"
    assert health["rule_check_mode"] == "model"


class BlockingEncoder(InputEncoder):
    """Real encoder whose tokenize blocks until released (simulates slow tokenization)."""

    def __init__(self, path: Path) -> None:
        super().__init__(InputEncoder.from_file(path, EncoderConfig()).tokenizer, EncoderConfig())
        self.release = __import__("threading").Event()
        self.started = __import__("threading").Event()

    def tokenize(self, texts: Any) -> list[list[int]]:
        self.started.set()
        self.release.wait(timeout=10)
        return super().tokenize(texts)


async def test_tokenization_admission_is_bounded_by_real_work(model_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from compliance_model_service.rule_check.batching import OverloadedError
    from compliance_model_service.rule_check.engine import SpanText

    encoder = BlockingEncoder(model_path / "tokenizer.json")
    tok_pool, infer_pool = ThreadPoolExecutor(1), ThreadPoolExecutor(1)
    model = KeywordModel(
        InputEncoder.from_file(model_path / "tokenizer.json", EncoderConfig()).tokenize(["share"])[
            0
        ][0]
    )
    batcher = MicroBatcher(
        model, max_batch_rows=16, max_wait_ms=0, max_queue_rows=64, executor=infer_pool
    )
    batcher.start()
    engine = RuleCheckEngine(
        encoder,
        batcher,
        InferenceSpec(top_k=1, temperature=1.0, evidence_threshold=0.5),
        tok_pool,
        max_pending_tokenize=2,
    )
    spans = [SpanText("a", "may share")]
    running = asyncio.create_task(engine.check("h", spans))
    await asyncio.to_thread(encoder.started.wait, 5)
    queued = asyncio.create_task(engine.check("h", spans))
    await asyncio.sleep(0.01)
    assert engine.pending_tokenize == 2
    with pytest.raises(OverloadedError):
        await engine.check("h", spans)
    assert engine.rejected_tokenize == 1
    # Cancelling the queued request frees its slot (its work never started) ...
    queued.cancel()
    await asyncio.gather(queued, return_exceptions=True)
    await asyncio.sleep(0.01)
    assert engine.pending_tokenize == 1
    # ... but cancelling the running one keeps the slot until the work really finishes.
    running.cancel()
    await asyncio.gather(running, return_exceptions=True)
    await asyncio.sleep(0.01)
    assert engine.pending_tokenize == 1
    for _ in range(5):  # repeated attempts while the worker is still busy stay bounded
        tasks = [asyncio.create_task(engine.check("h", spans)) for _ in range(3)]
        await asyncio.sleep(0)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert engine.pending_tokenize <= 2
    encoder.release.set()
    for _ in range(100):
        if engine.pending_tokenize == 0:
            break
        await asyncio.sleep(0.01)
    assert engine.pending_tokenize == 0
    outcome = await engine.check("h", spans)
    assert outcome.label in ("entailment", "not_mentioned")
    await batcher.close()
    tok_pool.shutdown(wait=True)
    infer_pool.shutdown(wait=True)


async def test_check_rejects_blank_and_oversized_input(
    check_client: httpx.AsyncClient, test_settings: Settings
) -> None:
    for payload in (
        {"hypothesis": "   \n", "spans": [{"span_id": "a", "text": "x"}]},
        {"hypothesis": "h", "spans": [{"span_id": "a", "text": " \t "}]},
    ):
        response = await check_client.post("/v1/check", json=payload)
        assert response.status_code == 422
    limit = test_settings.rule_check_max_request_chars
    spans = [{"span_id": f"s{i}", "text": "x" * 20_000} for i in range(limit // 20_000 + 1)]
    response = await check_client.post("/v1/check", json={"hypothesis": "h", "spans": spans})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
