"""CPU latency per (rule, document) pair for PyTorch fp32, ONNX fp32 and ONNX int8.

    uv run --frozen --with-requirements ml/requirements-tools.txt python -m ml.export.bench \
        --artifact-dir <artifact> --run-dir <run> --label <host> --out <json>

Workload (identical for every variant): ``--pairs`` DEV pairs drawn with a fixed seed;
for each pair the complete inference path the service runs (tokenize hypothesis and
every segment, one evidence forward over all segments, top-k, one label forward,
temperature softmax), one pair at a time, after ``WARMUP`` untimed pairs. Threads are
fixed (``torch.set_num_threads`` / ONNX Runtime intra-op threads). The JSON records the
CPU model, thread count, the per-pair segment counts and the model file sizes. Numbers
from different machines are not comparable with each other.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from compliance_model_service.rule_check.artifact import verify_artifact
from compliance_model_service.rule_check.encoding import (
    InputEncoder,
    sigmoid,
    softmax,
    top_k_in_order,
)
from compliance_model_service.rule_check.onnx_backend import OnnxBackend
from ml.common import provenance
from ml.common.io import write_json
from ml.data.store import load_dataset
from ml.eval.infer import TorchForward
from ml.export.export import load_model

SEED = 20261021
WARMUP = 5
Forward = Callable[[list[list[int]]], tuple[np.ndarray, np.ndarray]]


def log(message: str) -> None:
    print(message, flush=True)  # noqa: T201


def check_pair(
    encoder: InputEncoder,
    forward: Forward,
    hypothesis: str,
    segments: list[str],
    k: int,
    temperature: float,
) -> int:
    tokens = encoder.tokenize([hypothesis, *segments])
    hyp, segs = tokens[0], tokens[1:]
    ev, _ = forward([encoder.evidence_input(hyp, s).ids for s in segs])
    chosen = top_k_in_order([float(x) for x in sigmoid(ev)], min(k, len(segs)))
    _, lab = forward([encoder.label_input(hyp, [segs[i] for i in chosen]).ids])
    return int(softmax(lab[0], temperature).argmax())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU latency benchmark")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--run-dir", type=Path, default=None, help="needed for the PyTorch fp32 variant"
    )
    parser.add_argument("--variants", default="torch_fp32,onnx_fp32,onnx_int8")
    parser.add_argument("--pairs", type=int, default=60)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--label", required=True, help="host label, e.g. colab_cpu or workstation")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(args.threads)
    manifest, _ = verify_artifact(args.artifact_dir)  # measured, not served: no quality gate
    encoder = InputEncoder.from_file(
        args.artifact_dir / manifest.files.tokenizer.path, manifest.encoder.to_config()
    )
    data_dir = args.data_dir or provenance.REPO / ".cache" / "ml" / "data" / (
        "full" if manifest.kind == "production" else "smoke"
    )
    dataset = load_dataset(data_dir, splits=("dev",))
    if dataset.data_version != manifest.provenance.get("data_version"):
        raise SystemExit("benchmark data differs from the artifact's training data version")
    pairs = random.Random(SEED).sample(  # noqa: S311 - reproducible sampling
        dataset.pairs["dev"], min(args.pairs + WARMUP, len(dataset.pairs["dev"]))
    )
    k, temperature = manifest.inference.top_k, manifest.inference.temperature

    forwards: dict[str, Forward] = {}
    sizes: dict[str, int] = {}
    skipped: dict[str, str] = {}
    for variant in args.variants.split(","):
        if variant == "torch_fp32":
            if args.run_dir is None:
                skipped[variant] = "no --run-dir"
                continue
            model, cfg, _ = load_model(args.run_dir)
            forwards[variant] = TorchForward(
                model,
                encoder,
                device="cpu",
                fp16=False,
                max_tokens=cfg.inference.max_tokens_per_forward,
            )
            sizes[variant] = (
                (args.run_dir / "checkpoints" / "best" / "model.safetensors").stat().st_size
            )
        elif variant in ("onnx_fp32", "onnx_int8"):
            name = "model.fp32.onnx" if variant == "onnx_fp32" else "model.int8.onnx"
            path = args.artifact_dir / name
            if not path.is_file():
                skipped[variant] = f"{name} not in the artifact"
                continue
            forwards[variant] = OnnxBackend(path, encoder, threads=args.threads)
            sizes[variant] = path.stat().st_size
        else:
            raise SystemExit(f"unknown variant {variant}")

    latency: dict[str, Any] = {}
    seg_counts = [len(dataset.document(p).segments) for p in pairs[WARMUP:]]
    for variant, forward in forwards.items():
        times = []
        for i, pair in enumerate(pairs):
            doc = dataset.document(pair)
            texts = [doc.segment_text(j) for j in range(len(doc.segments))]
            started = time.perf_counter()
            check_pair(encoder, forward, dataset.hypotheses[pair.key], texts, k, temperature)
            if i >= WARMUP:
                times.append(time.perf_counter() - started)
        latency[variant] = {
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
            "mean": float(np.mean(times)),
            "n": len(times),
        }
        log(f"  {variant}: {latency[variant]}")
    result = {
        "scope": "SMOKE artifact (tiny random backbone)"
        if manifest.kind == "smoke"
        else "trained artifact",
        "host_label": args.label,
        "model_version": manifest.model_version,
        "workload": {
            "pairs": len(pairs) - WARMUP,
            "warmup_pairs": WARMUP,
            "source": "seeded DEV sample; full pipeline per pair (all segments + top-k label pass)",
            "segments_per_pair": {"mean": float(np.mean(seg_counts)), "max": int(max(seg_counts))},
            "top_k": k,
        },
        "threads": args.threads,
        "per_pair_latency_s": latency,
        "model_size_bytes": sizes,
        "skipped": skipped,
        "provenance": provenance.collect(sys.argv, seed=SEED),
    }
    write_json(args.out, result)
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
