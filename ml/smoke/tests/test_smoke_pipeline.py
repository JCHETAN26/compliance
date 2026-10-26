"""End-to-end CPU smoke of the ML pipeline (needs the tool overlay; run by ``make ml-smoke``).

data subset -> tiny random-backbone training -> baselines -> evaluation -> ONNX fp32/int8
export with parity -> the real model service serving the int8 artifact. Every output is
SMOKE (meaningless numbers) and lives in a temporary directory; nothing is written to
``ml/eval/results``. The service check compares /v1/check responses with offline ONNX
inference on the same pairs, which ties the served path to the evaluated path.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from compliance_core.config import Settings
from compliance_model_service.main import create_app
from compliance_model_service.rule_check.artifact import load_artifact
from compliance_model_service.rule_check.encoding import (
    InputEncoder,
    sigmoid,
    softmax,
    top_k_in_order,
)
from compliance_model_service.rule_check.labels import LABELS
from compliance_model_service.rule_check.onnx_backend import OnnxBackend
from ml.baselines import classic
from ml.common.io import read_json
from ml.common.provenance import REPO
from ml.data import prepare
from ml.data.store import load_dataset
from ml.eval import evaluate
from ml.eval.infer import TokenCache, predict_pairs
from ml.export import export
from ml.train import train
from ml.train.config import load_config

SMOKE_CONFIG = REPO / "ml" / "train" / "configs" / "smoke.yaml"


@pytest.fixture(autouse=True)
def _remove_test_outputs(tmp_path: Path) -> Iterator[None]:
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)  # checkpoints are tens of MB; disk is tight


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    root = tmp_path_factory.mktemp("ml-smoke")
    data = root / "data"
    assert prepare.main(["--subset", "smoke", "--out", str(data), "--threads", "4"]) == 0
    cfg = load_config(SMOKE_CONFIG).model_copy(deep=True)
    cfg.data.data_dir = str(data)
    config_path = root / "smoke.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    run_dir = root / "run"
    record = train.run(cfg, run_dir, config_path)
    assert record["status"] == "complete"
    baselines = root / "baselines"
    assert classic.main(["--data-dir", str(data), "--out", str(baselines)]) == 0
    results = root / "results"
    assert (
        evaluate.main(
            [
                "--run-dir",
                str(run_dir),
                "--baselines-dir",
                str(baselines),
                "--data-dir",
                str(data),
                "--out",
                str(results),
            ]
        )
        == 0
    )
    artifact = root / "artifact"
    assert (
        export.main(
            [
                "--run-dir",
                str(run_dir),
                "--params",
                str(results / "inference_params.json"),
                "--model-eval",
                str(results / "model_eval.json"),
                "--out",
                str(artifact),
            ]
        )
        == 0
    )
    yield {
        "root": root,
        "data": data,
        "run": run_dir,
        "results": results,
        "artifact": artifact,
        "baselines": baselines,
    }
    shutil.rmtree(root, ignore_errors=True)  # ~100 MB; the host disk is tight


def test_outputs_are_labelled_smoke_and_complete(pipeline: dict[str, Path]) -> None:
    run = read_json(pipeline["run"] / "run.json")
    assert run["kind"] == "smoke"
    assert run["selection"]["k"] in (3, 4)
    model_eval = read_json(pipeline["results"] / "model_eval.json")
    assert model_eval["scope"].startswith("SMOKE")
    comparison = read_json(pipeline["results"] / "baselines_comparison.json")
    assert {
        "majority_global",
        "majority_per_hypothesis",
        "tfidf_logreg",
        "rule_check_model",
    } <= set(comparison["systems"])
    assert comparison["baseline_notes"]["llm_zero_shot"].startswith("not available")
    cascade = read_json(pipeline["results"] / "cascade.json")
    assert cascade["tau_selection"]["provisional"] is True  # no real LLM results exist
    assert cascade["llm_only_test"].startswith("pending")
    for png in (
        "confusion_test.png",
        "reliability_test.png",
        "coverage_accuracy.png",
        "systems_macro_f1_test.png",
    ):
        assert (pipeline["results"] / png).stat().st_size > 1000


def test_artifact_parity_and_verification(pipeline: dict[str, Path]) -> None:
    report = read_json(pipeline["artifact"] / "export_report.json")
    assert report["fp32_ok"] is True
    assert report["fp32"]["max_abs_diff_label_logit"] <= 1e-3
    assert report["sizes_bytes"]["onnx_int8"] < report["sizes_bytes"]["onnx_fp32"]
    artifact = load_artifact(pipeline["artifact"], variant="int8", allow_smoke=True)
    assert artifact.manifest.kind == "smoke"
    assert artifact.manifest.inference.tau is None  # provisional tau is never served
    assert (
        artifact.manifest.provenance["data_version"]
        == read_json(pipeline["data"] / "manifest.json")["data_version"]
    )


def test_smoke_results_are_refused_for_the_results_directory(pipeline: dict[str, Path]) -> None:
    with pytest.raises(evaluate.EvaluationInputError, match="smoke"):
        evaluate.main(["--run-dir", str(pipeline["run"]), "--data-dir", str(pipeline["data"])])


def test_production_config_refuses_cpu_training(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():  # pragma: no cover - only meaningful on CPU hosts
        pytest.skip("CUDA present")
    cfg = load_config(REPO / "ml" / "train" / "configs" / "deberta_v3_base.yaml")
    with pytest.raises(SystemExit, match="requires a CUDA GPU"):
        train.run(cfg, tmp_path / "run", REPO / "ml" / "train" / "configs" / "deberta_v3_base.yaml")


async def _serve(
    settings: Settings, payloads: list[dict[str, Any]]
) -> tuple[list[httpx.Response], dict[str, Any], dict[str, Any]]:
    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client,
    ):
        responses = await asyncio.gather(*(client.post("/v1/check", json=p) for p in payloads))
        return (
            list(responses),
            (await client.get("/health")).json(),
            (await client.get("/metadata")).json(),
        )


@pytest.mark.parametrize(("variant", "tol"), [("fp32", 1e-4), ("int8", 2e-2)])
def test_service_serves_the_artifact_as_evaluated(
    pipeline: dict[str, Path], test_settings: Settings, variant: str, tol: float
) -> None:
    """fp32: served == offline up to float noise. int8: a looser, stated tolerance (the
    quantized kernels may sum in a different order; export ``batch_sensitivity``)."""
    dataset = load_dataset(pipeline["data"], splits=("dev",))
    pairs = dataset.pairs["dev"][:10]
    artifact = load_artifact(pipeline["artifact"], variant=variant, allow_smoke=True)  # type: ignore[arg-type]
    inf = artifact.manifest.inference
    encoder = InputEncoder.from_file(artifact.tokenizer_path, artifact.manifest.encoder.to_config())
    offline = predict_pairs(
        pairs,
        TokenCache(dataset, encoder),
        OnnxBackend(artifact.onnx_path, encoder),
        top_k_values=[inf.top_k],
        chunk_rows=4096,
    )
    payloads = []
    for pair in pairs:
        doc = dataset.document(pair)
        payloads.append(
            {
                "hypothesis": dataset.hypotheses[pair.key],
                "spans": [
                    {"span_id": f"seg-{i}", "text": doc.segment_text(i)}
                    for i in range(len(doc.segments))
                ],
            }
        )
    settings = test_settings.model_copy(
        update={
            "rule_check_artifact_dir": pipeline["artifact"],
            "rule_check_allow_smoke": True,
            "rule_check_max_wait_ms": 20.0,
            "rerank_enabled": False,
            "rule_check_variant": variant,
            "rule_check_max_pending_requests": 16,  # 10 concurrent requests (default bound is 8)
        }
    )
    responses, health, meta = asyncio.run(_serve(settings, payloads))
    assert health["rule_check_mode"] == "model"
    assert meta["rule_check"]["kind"] == "smoke"
    assert meta["rule_check"]["batching"]["stats"]["batches"] >= 1
    tokens = TokenCache(dataset, encoder)
    backend = OnnxBackend(artifact.onnx_path, encoder)
    rebuilt: list[Any] = []
    exact = 0
    for pred, response in zip(offline, responses, strict=True):
        assert response.status_code == 200
        body = response.json()
        assert body["model_kind"] == "smoke"
        assert body["variant"] == variant
        scores = [float(x) for x in sigmoid(np.array(pred.evidence_logits))]
        served = [int(c["span_id"].removeprefix("seg-")) for c in body["considered"]]
        # served scores equal offline scores (different batch shapes -> float noise only)
        for c, i in zip(body["considered"], served, strict=True):
            assert c["score"] == pytest.approx(scores[i], abs=tol)
        # ... and the served choice is a valid top-k under that tolerance
        floor = min(scores[i] for i in served)
        assert all(scores[j] <= floor + tol for j in range(len(scores)) if j not in served)
        # Internal consistency of every response.
        returned = body["probabilities"]
        assert body["label"] == max(returned, key=returned.get)
        assert body["confidence"] == pytest.approx(max(returned.values()), abs=1e-9)
        # Two-pass contract for every pair: rebuild the label input from the spans the
        # service actually chose and compare its label distribution offline.
        pair = pairs[len(rebuilt)]
        segs = tokens.segments("dev", pair.doc_id)
        row = encoder.label_input(tokens.hypotheses[pair.key], [segs[i] for i in served]).ids
        offline_probs = softmax(backend([row])[1][0], inf.temperature)
        rebuilt.append(offline_probs)
        for label, value in zip(LABELS, offline_probs, strict=True):
            assert returned[label] == pytest.approx(float(value), abs=tol)
        if served == top_k_in_order(scores, min(inf.top_k, len(scores))):
            exact += 1
    if variant == "fp32":
        # fp32 differs from offline only by float noise; a random tiny model still has
        # near-ties, so most (not necessarily all) chosen top-k sets coincide exactly.
        assert exact >= 7

    refused = test_settings.model_copy(
        update={"rule_check_artifact_dir": pipeline["artifact"], "rerank_enabled": False}
    )
    responses, health, _ = asyncio.run(_serve(refused, payloads[:1]))
    assert health["rule_check_mode"] == "untrained_fallback"
    assert responses[0].status_code == 503


def _short_config(data: Path, root: Path, name: str) -> tuple[Any, Path]:
    import yaml

    cfg = load_config(SMOKE_CONFIG).model_copy(deep=True)
    cfg.data.data_dir = str(data)
    cfg.training.max_steps = 12
    cfg.training.checkpoint_every_steps = 6
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return cfg, path


def test_interrupted_run_resumes_to_identical_result(
    pipeline: dict[str, Path], tmp_path: Path
) -> None:
    from ml.common.provenance import file_sha256

    cfg, path = _short_config(pipeline["data"], tmp_path, "short")
    full = train.run(cfg, tmp_path / "uninterrupted", path)
    with pytest.raises(train._StopForTest):
        train.run(cfg, tmp_path / "resumed", path, _stop_after_step=6)
    state = read_json(tmp_path / "resumed" / "checkpoints" / "last" / "state.json")
    assert state["step"] == 6
    assert state["epoch"] == 0
    resumed = train.run(cfg, tmp_path / "resumed", path)
    assert resumed["optimizer_steps"] == full["optimizer_steps"] == 12
    assert resumed["best_weights_sha256"] == full["best_weights_sha256"]
    assert file_sha256(tmp_path / "resumed" / "predictions_dev.jsonl.gz") == file_sha256(
        tmp_path / "uninterrupted" / "predictions_dev.jsonl.gz"
    )
    assert len(resumed["sessions"]) == 2
    assert resumed["sessions"][1]["resumed_from"]["step"] == 6
    assert resumed["provenance"]["timestamp_utc"] == resumed["sessions"][0]["timestamp_utc"]


def test_resume_refuses_changed_identity(pipeline: dict[str, Path], tmp_path: Path) -> None:
    cfg, path = _short_config(pipeline["data"], tmp_path, "identity")
    with pytest.raises(train._StopForTest):
        train.run(cfg, tmp_path / "run", path, _stop_after_step=6)
    record = read_json(tmp_path / "run" / "run.json")
    record["data_version"] = "0" * 64
    (tmp_path / "run" / "run.json").write_text(__import__("json").dumps(record))
    with pytest.raises(train.ResumeIdentityError, match="data_version"):
        train.run(cfg, tmp_path / "run", path)


def test_production_config_refuses_subset_data(pipeline: dict[str, Path], tmp_path: Path) -> None:
    cfg = load_config(REPO / "ml" / "train" / "configs" / "deberta_v3_base.yaml").model_copy(
        deep=True
    )
    cfg.requires_cuda = False  # reach the data guard on this CPU host
    cfg.data.data_dir = str(pipeline["data"])
    with pytest.raises(SystemExit, match="full official data"):
        train.run(cfg, tmp_path / "run", SMOKE_CONFIG)


def test_cached_tokenizer_is_verified_and_regenerated(tmp_path: Path) -> None:
    from ml.common.provenance import file_sha256
    from ml.train.backbone import TOKENIZER_META, fetch_backbone

    spec = load_config(SMOKE_CONFIG).backbone
    files = fetch_backbone(spec, cache=tmp_path)
    good = file_sha256(files.tokenizer_json)
    files.tokenizer_json.write_text('{"corrupt": true}')  # stale/corrupt derivative
    assert file_sha256(fetch_backbone(spec, cache=tmp_path).tokenizer_json) == good
    meta = files.root / TOKENIZER_META
    data = __import__("json").loads(meta.read_text())
    data["converter"]["transformers"] = "0.0.0"  # produced by another converter version
    meta.write_text(__import__("json").dumps(data))
    regenerated = fetch_backbone(spec, cache=tmp_path)
    assert file_sha256(regenerated.tokenizer_json) == good
    assert __import__("json").loads(meta.read_text())["converter"]["transformers"] != "0.0.0"


def test_bundle_round_trip_and_refusals(
    pipeline: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tarfile

    from ml.colab import bundle

    out = tmp_path / "bundle.tar"
    assert (
        bundle.main(
            [
                "pack",
                "--out",
                str(out),
                "--data",
                str(pipeline["data"]),
                "--run",
                f"smoke={pipeline['run']}",
                "--baselines",
                str(pipeline["baselines"]),
                "--results",
                str(pipeline["results"]),
                "--artifact",
                str(pipeline["artifact"]),
            ]
        )
        == 0
    )
    with tarfile.open(out) as tar:
        names = tar.getnames()
    assert not any("checkpoints" in n or n.endswith(".fp32.onnx") for n in names)  # no weights
    cache, results, models = (
        tmp_path / "cache",
        tmp_path / "results",
        tmp_path / "models" / "rule-check",
    )
    monkeypatch.setattr(bundle, "CACHE", cache)
    monkeypatch.setattr(bundle, "RESULTS", results)
    monkeypatch.setattr(bundle, "MODELS", models)
    assert bundle.main(["import", "--bundle", str(out)]) == 0
    record = read_json(results / "colab_import.json")
    assert set(record["installed"]["runs"]) == {"smoke"}
    current = models / "current"
    assert current.is_symlink()
    assert not Path(str(current.readlink())).is_absolute()
    assert load_artifact(current, variant="int8", allow_smoke=True).manifest.kind == "smoke"
    assert (cache / "runs" / "smoke" / "predictions_test.jsonl.gz").is_file()

    tampered = tmp_path / "tampered.tar"
    with tarfile.open(out) as src, tarfile.open(tampered, "w") as dst:
        for member in src.getmembers():
            data = src.extractfile(member).read()  # type: ignore[union-attr]
            if member.name.endswith("model_eval.json"):
                data = data.replace(b"SMOKE", b"FAKED")
                member.size = len(data)
            import io

            dst.addfile(member, io.BytesIO(data))
    with pytest.raises(SystemExit, match="SHA-256"):
        bundle.main(["import", "--bundle", str(tampered)])

    unsafe = tmp_path / "unsafe.tar"
    with tarfile.open(unsafe, "w") as dst:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        import io

        dst.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(SystemExit, match="unsafe"):
        bundle.main(["import", "--bundle", str(unsafe)])
    assert not (tmp_path / "escape.txt").exists()


def _rebuild(src: Path, dst: Path, edits: dict[str, bytes]) -> None:
    """Copy a bundle, replacing some members, with a consistent BUNDLE_MANIFEST (an
    attacker can always produce matching hashes; only content validation can refuse)."""
    import hashlib
    import io
    import json
    import tarfile

    with tarfile.open(src) as tar:
        files = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}  # type: ignore[union-attr]
    files.update(edits)
    manifest = json.loads(files.pop("BUNDLE_MANIFEST.json"))
    manifest["files"] = {
        n: {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)} for n, b in files.items()
    }
    files["BUNDLE_MANIFEST.json"] = json.dumps(manifest).encode()
    with tarfile.open(dst, "w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_malicious_or_invalid_bundles_change_nothing(
    pipeline: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import tarfile

    from compliance_model_service.rule_check.artifact import ArtifactError
    from ml.colab import bundle

    good = tmp_path / "good.tar"
    assert (
        bundle.main(
            [
                "pack",
                "--out",
                str(good),
                "--data",
                str(pipeline["data"]),
                "--run",
                f"smoke={pipeline['run']}",
                "--results",
                str(pipeline["results"]),
                "--artifact",
                str(pipeline["artifact"]),
            ]
        )
        == 0
    )
    root = tmp_path / "local"
    cache, results, models = root / "cache", root / "results", root / "models" / "rule-check"
    monkeypatch.setattr(bundle, "CACHE", cache)
    monkeypatch.setattr(bundle, "RESULTS", results)
    monkeypatch.setattr(bundle, "MODELS", models)
    sentinel_data = cache / "data" / "smoke" / "manifest.json"
    sentinel_data.parent.mkdir(parents=True)
    sentinel_data.write_text('{"data_version": "local", "name": "smoke"}')
    results.mkdir(parents=True)
    (results / "model_eval.json").write_text('{"local": "sentinel"}')

    def snapshot() -> dict[str, bytes]:
        return {
            str(p.relative_to(tmp_path)): p.read_bytes()
            for p in tmp_path.rglob("*")
            if p.is_file() and "local" in p.parts and "imports" not in p.parts
        }

    before = snapshot()
    with tarfile.open(good) as tar:
        art_manifest = json.loads(tar.extractfile("artifact/manifest.json").read())  # type: ignore[union-attr]
        data_manifest = json.loads(tar.extractfile("data/manifest.json").read())  # type: ignore[union-attr]
    cases = {
        "escaping_model_version": {
            "artifact/manifest.json": json.dumps(
                {**art_manifest, "model_version": "../../escape"}
            ).encode()
        },
        "escaping_data_name": {
            "data/manifest.json": json.dumps({**data_manifest, "name": "../../outside"}).encode()
        },
        "late_invalid_artifact": {"artifact/tokenizer.json": b'{"tampered": true}'},
    }
    for name, edits in cases.items():
        crafted = tmp_path / f"{name}.tar"
        _rebuild(good, crafted, edits)
        with pytest.raises((SystemExit, ArtifactError)):
            bundle.main(["import", "--bundle", str(crafted)])
        assert snapshot() == before, name  # nothing local changed or was deleted
        assert not models.exists(), name
        assert not (tmp_path / "escape").exists()
        assert not (tmp_path / "outside").exists()
    monkeypatch.setattr(bundle, "MAX_BUNDLE_BYTES", 1000)
    with pytest.raises(SystemExit, match="limit"):
        bundle.main(["import", "--bundle", str(good)])
    assert snapshot() == before


def test_manifest_refresh_changes_version_not_weights(
    pipeline: dict[str, Path], tmp_path: Path
) -> None:
    import json
    import shutil as sh

    artifact = tmp_path / "artifact"
    sh.copytree(pipeline["artifact"], artifact)
    before = load_artifact(artifact, variant="int8", allow_smoke=True).manifest
    params = read_json(pipeline["results"] / "inference_params.json")
    # e.g. a final DEV cascade selection against a usable LLM baseline
    params.update(tau=0.8, tau_provisional=False, llm_baseline_usable=True)
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(params))
    assert (
        export.main(
            [
                "--run-dir",
                str(pipeline["run"]),
                "--params",
                str(params_path),
                "--out",
                str(artifact),
                "--refresh-manifest",
            ]
        )
        == 0
    )
    after = load_artifact(artifact, variant="int8", allow_smoke=True).manifest
    assert after.inference.tau == 0.8
    assert after.model_version != before.model_version
    assert after.files == before.files  # same verified ONNX and tokenizer files
    assert after.provenance["refreshes"][0]["previous_model_version"] == before.model_version


def test_install_artifact_reserved_names_and_ancestry(
    pipeline: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import shutil as sh

    from ml.colab import bundle

    models = tmp_path / "models" / "rule-check"
    monkeypatch.setattr(bundle, "MODELS", models)
    installed = bundle.install_artifact(pipeline["artifact"], move=False)
    pointer = (models / "current").readlink()
    # an artifact claiming the reserved name "current" is refused before any change
    reserved = tmp_path / "reserved"
    sh.copytree(pipeline["artifact"], reserved)
    manifest = json.loads((reserved / "manifest.json").read_text())
    manifest["model_version"] = "current"
    (reserved / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="reserved"):
        bundle.install_artifact(reserved, move=False)
    assert (models / "current").is_symlink()
    assert (models / "current").readlink() == pointer
    # Reinstalling the installed directory itself is idempotent (same source and target).
    assert bundle.install_artifact(installed, move=False) == installed
    assert (models / "current").readlink() == pointer

    def tree(root: Path) -> dict[str, bytes]:
        return {str(q.relative_to(root)): q.read_bytes() for q in root.rglob("*") if q.is_file()}

    # Descendant source: a valid artifact *inside* the target directory it would replace.
    nested = installed / "nested"
    sh.copytree(pipeline["artifact"], nested)
    before = tree(models)
    with pytest.raises(SystemExit, match="ancestor"):
        bundle.install_artifact(nested, move=False)
    assert tree(models) == before
    assert (models / "current").readlink() == pointer
    sh.rmtree(nested)

    # Ancestor source: a valid artifact directory that contains the models directory.
    outer = tmp_path / "outer"
    sh.copytree(pipeline["artifact"], outer)
    inner_models = outer / "models" / "rule-check"
    monkeypatch.setattr(bundle, "MODELS", inner_models)
    before_outer = tree(outer)
    with pytest.raises(SystemExit, match="ancestor"):
        bundle.install_artifact(outer, move=False)
    assert tree(outer) == before_outer
    assert not inner_models.exists()
