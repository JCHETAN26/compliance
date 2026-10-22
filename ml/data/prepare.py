"""Prepare ContractNLI for the rule-check model: segments, gold mapping, dense ranking.

    uv run python -m ml.data.prepare                 # full official splits -> .cache/ml/data/full
    uv run python -m ml.data.prepare --subset smoke  # first 12/4/4 documents (CI smoke only)

Writes the dataset directory described in ``ml.data.store`` and, for the full data
only, ``ml/eval/results/data_report.json`` (counts, label distributions, TRAIN-derived
class weights, segmentation and overlap statistics, cross-split duplicate texts and the
candidate-retrieval evidence coverage ceiling).

**Dense candidate ranking (shared by baselines and label-head training).** Every
segment is embedded with the application's pinned encoder through the production
``Embedder`` (same long-span windowing); each hypothesis is embedded as a query with the
production query instruction. Segments of the pair's document are ranked by cosine
similarity (ties -> lower index). Gold is never used. The TF-IDF/logistic-regression and
LLM zero-shot baselines read the top-8 of this ranking; the trained model scores every
segment with its own evidence head instead.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from compliance_core.config import Settings
from compliance_ingestion.embeddings import Embedder, ensure_model, spec_from_settings
from compliance_seed.job import ensure_dataset
from ml.common import provenance
from ml.common.io import CACHE, RESULTS, write_json, write_jsonl_gz
from ml.common.provenance import file_sha256
from ml.data.contractnli import SPLITS, SplitData, build_split, hypothesis_keys, label_counts
from ml.data.store import RANK_DEPTH, document_to_row, pair_to_row

SUBSETS: dict[str, dict[str, int]] = {"smoke": {"train": 12, "dev": 4, "test": 4}}
CANDIDATE_DEPTH = 8  # top-k candidates read by the baselines (fixed before any result)
COVERAGE_DEPTHS = (1, 3, 5, 8, 16)


def log(message: str) -> None:
    print(message, flush=True)  # noqa: T201


def subset_ids(archive: Path, split: str, count: int) -> list[str]:
    from ml.data.contractnli import load_raw

    return [str(d["id"]) for d in load_raw(archive, split)["documents"][:count]]


def inverse_frequency_weights(counts: dict[str, int]) -> dict[str, float]:
    """w_c = N / (C * n_c): the weights used by the label loss (from TRAIN counts)."""
    total = sum(counts.values())
    return {label: total / (len(counts) * n) for label, n in counts.items() if n > 0}


class DeviceEmbedder(Embedder):
    """The production ``Embedder`` (same weights, windowing, normalisation) on a device.

    Used only to make Colab's dense ranking fast on GPU; the model and its pre/post
    processing are unchanged. Scores can differ from CPU in the last float digits, so
    the manifest records the device and every result records the ``data_version`` it
    used; evaluation refuses to compare systems built on different data versions.
    """

    def __init__(self, path: Path, *, device: str, threads: int, batch_size: int) -> None:
        super().__init__(path, threads=threads, batch_size=batch_size)
        if device != "cpu":
            self._model.to(device)


def rank_segments(
    data: dict[str, SplitData], embedder: Embedder, instruction: str
) -> dict[str, dict[str, tuple[list[int], list[float]]]]:
    keys = hypothesis_keys(next(iter(data.values())).hypotheses)
    hypotheses = next(iter(data.values())).hypotheses
    queries = np.asarray(embedder.embed_queries([hypotheses[k] for k in keys], instruction))
    query_of = dict(zip(keys, queries, strict=True))
    result: dict[str, dict[str, tuple[list[int], list[float]]]] = {}
    for split, sd in data.items():
        started = time.perf_counter()
        doc_vectors: dict[str, np.ndarray] = {}
        for doc_id, doc in sd.documents.items():
            texts = [doc.segment_text(i) for i in range(len(doc.segments))]
            doc_vectors[doc_id] = np.asarray([e.vector for e in embedder.embed(texts)])
        ranks: dict[str, tuple[list[int], list[float]]] = {}
        for pair in sd.pairs:
            scores = doc_vectors[pair.doc_id] @ query_of[pair.key]
            rounded = [round(float(x), 6) for x in scores]
            order = sorted(range(len(rounded)), key=lambda i: (-rounded[i], i))[:RANK_DEPTH]
            ranks[pair.pair_id] = (order, [rounded[i] for i in order])
        result[split] = ranks
        log(f"  ranked {split}: {len(sd.pairs)} pairs in {time.perf_counter() - started:.0f}s")
    return result


def _stats(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": round(statistics.fmean(ordered), 2),
        "median": float(statistics.median(ordered)),
        "p95": float(ordered[int(0.95 * (len(ordered) - 1))]),
        "max": float(ordered[-1]),
    }


def coverage(sd: SplitData, ranks: dict[str, tuple[list[int], list[float]]]) -> dict[str, Any]:
    evidence_pairs = [p for p in sd.pairs if p.gold_segments]
    out: dict[str, Any] = {"evidence_pairs": len(evidence_pairs)}
    for depth in COVERAGE_DEPTHS:
        any_hit = recall = 0.0
        for pair in evidence_pairs:
            top = set(ranks[pair.pair_id][0][:depth])
            hit = len(top.intersection(pair.gold_segments))
            any_hit += hit > 0
            recall += hit / len(pair.gold_segments)
        out[f"top{depth}"] = {
            "pairs_with_any_gold": round(any_hit / len(evidence_pairs), 4),
            "mean_gold_segment_recall": round(recall / len(evidence_pairs), 4),
        }
    return out


def report(
    data: dict[str, SplitData], ranks: dict[str, dict[str, tuple[list[int], list[float]]]]
) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    text_hash: dict[str, list[str]] = defaultdict(list)
    for split, sd in data.items():
        per_hyp: dict[str, Counter[str]] = defaultdict(Counter)
        for pair in sd.pairs:
            per_hyp[pair.key][pair.label] += 1
        docs = list(sd.documents.values())
        for doc in docs:
            text_hash[hashlib.sha256(doc.text.encode()).hexdigest()].append(f"{split}:{doc.doc_id}")
        seg_counts = [len(d.segments) for d in docs]
        seg_chars = [e - s for d in docs for s, e in d.segments]
        official = [m for d in docs for m in d.official_to_segments]
        seg_official: Counter[int] = Counter()
        for d in docs:
            per_seg: Counter[int] = Counter(j for m in d.official_to_segments for j in m)
            seg_official.update(per_seg.values())
        ec = [p for p in sd.pairs if p.label != "not_mentioned"]
        splits[split] = {
            "documents": len(docs),
            "pairs": len(sd.pairs),
            "label_counts": label_counts(sd.pairs),
            "label_counts_by_hypothesis": {
                k: {lab: per_hyp[k].get(lab, 0) for lab in label_counts(sd.pairs)}
                for k in hypothesis_keys(sd.hypotheses)
            },
            "segments_per_document": _stats(seg_counts),
            "segment_chars": _stats(seg_chars),
            "official_spans": len(official),
            "official_spans_whitespace_only": sum(not m for m in official),
            "official_spans_overlapping_multiple_segments": sum(len(m) > 1 for m in official),
            "segments_overlapping_multiple_official_spans": sum(
                n for c, n in seg_official.items() if c > 1
            ),
            "gold_segments_per_evidence_pair": _stats([len(p.gold_segments) for p in ec]),
            "evidence_pairs_without_gold_segment": sum(not p.gold_segments for p in ec),
            "not_mentioned_pairs_with_evidence_spans": sum(
                bool(p.gold_official) for p in sd.pairs if p.label == "not_mentioned"
            ),
            "candidate_coverage_ceiling": coverage(sd, ranks[split]),
        }
    duplicates = {h: ids for h, ids in text_hash.items() if len(ids) > 1}
    train_counts = splits["train"]["label_counts"]
    return {
        "splits": splits,
        "train_class_weights_inverse_frequency": {
            k: round(v, 6) for k, v in inverse_frequency_weights(train_counts).items()
        },
        "handoff_correction": (
            "docs/BUILD_HANDOFF.md section 7.3 says not_mentioned dominates. The official "
            f"TRAIN counts are {train_counts}: entailment is the largest class and "
            "contradiction the rare one. Labels are unchanged; loss weights are derived "
            "from these actual TRAIN counts."
        ),
        "exact_duplicate_document_texts": [
            {"text_sha256": h, "documents": ids} for h, ids in sorted(duplicates.items())
        ],
        "candidate_depth_used_by_baselines": CANDIDATE_DEPTH,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subset", choices=sorted(SUBSETS), default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu or cuda (dense ranking only)")
    args = parser.parse_args(argv)
    name = args.subset or "full"
    out: Path = args.out or CACHE / "data" / name

    settings = Settings(artifact_cache_dir=provenance.REPO / ".cache", artifact_auto_download=False)
    archive = ensure_dataset(settings)
    log(f"building splits ({name}) from {archive.name} ...")
    data = {
        split: build_split(
            archive,
            split,
            subset_ids(archive, split, SUBSETS[args.subset][split]) if args.subset else None,
        )
        for split in SPLITS
    }
    for split, sd in data.items():
        log(
            f"  {split}: {len(sd.documents)} documents, {len(sd.pairs)} pairs {label_counts(sd.pairs)}"
        )

    encoder_path = ensure_model(
        settings.model_cache_dir, spec_from_settings(settings), allow_download=False
    )
    embedder = DeviceEmbedder(encoder_path, device=args.device, threads=args.threads, batch_size=64)
    log("dense ranking of segments with the pinned encoder ...")
    ranks = rank_segments(data, embedder, settings.embedding_query_instruction)

    files: dict[str, str] = {}
    hypotheses = data["train"].hypotheses
    write_json(out / "hypotheses.json", {k: hypotheses[k] for k in hypothesis_keys(hypotheses)})
    files["hypotheses.json"] = file_sha256(out / "hypotheses.json")
    for split, sd in data.items():
        for stem, rows in (
            ("documents", [document_to_row(d) for d in sd.documents.values()]),
            ("pairs", [pair_to_row(p) for p in sd.pairs]),
            (
                "retrieval",
                [
                    {"pair_id": pid, "segments": seg, "scores": sc}
                    for pid, (seg, sc) in ranks[split].items()
                ],
            ),
        ):
            fname = f"{stem}_{split}.jsonl.gz"
            write_jsonl_gz(out / fname, rows)
            files[fname] = file_sha256(out / fname)
    data_version = provenance.config_hash(files)
    manifest = {
        "name": name,
        "subset": SUBSETS.get(args.subset or ""),
        "data_version": data_version,
        "dataset_sha256": file_sha256(archive),
        "encoder": {
            "model": settings.embedding_model_id,
            "revision": settings.embedding_model_revision,
            "sha256": settings.embedding_model_sha256,
            "query_instruction": settings.embedding_query_instruction,
            "device": args.device,
        },
        "rank_depth": RANK_DEPTH,
        "counts": {
            split: {"documents": len(sd.documents), "pairs": len(sd.pairs)}
            for split, sd in data.items()
        },
        "files": files,
        "provenance": provenance.collect(sys.argv, seed=None),
    }
    write_json(out / "manifest.json", manifest)
    log(f"wrote {out} (data_version {data_version[:16]})")
    if args.subset is None:
        result = {
            "scope": "full official ContractNLI splits (train/dev/test), all pairs",
            "data_dir_version": data_version,
            "dataset_sha256": manifest["dataset_sha256"],
            **report(data, ranks),
            "provenance": manifest["provenance"],
        }
        write_json(RESULTS / "data_report.json", result)
        log(f"wrote {RESULTS / 'data_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
