"""Read/write the processed ML dataset directory produced by ``ml.data.prepare``.

Layout (``.cache/ml/data/<name>/``, git-ignored)::

    manifest.json               dataset SHA-256, subset, counts, file hashes, data_version
    hypotheses.json             ContractNLI key -> hypothesis text
    documents_<split>.jsonl.gz  canonical text, segments, official spans, overlap map
    pairs_<split>.jsonl.gz      label, gold official spans, gold segments
    retrieval_<split>.jsonl.gz  non-gold dense ranking of segments per pair (top RANK_DEPTH)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.common.io import read_json, read_jsonl_gz
from ml.common.provenance import file_sha256
from ml.data.contractnli import SPLITS, Document, Pair

RANK_DEPTH = 32


@dataclass(frozen=True)
class Ranking:
    segments: tuple[int, ...]  # segment indices, best first (ties -> lower index)
    scores: tuple[float, ...]


@dataclass
class Dataset:
    root: Path
    manifest: dict[str, Any]
    hypotheses: dict[str, str]
    documents: dict[str, dict[str, Document]]  # split -> doc_id -> document
    pairs: dict[str, list[Pair]]
    rankings: dict[str, dict[str, Ranking]]  # split -> pair_id -> ranking

    @property
    def data_version(self) -> str:
        return str(self.manifest["data_version"])

    def document(self, pair: Pair) -> Document:
        return self.documents[pair.split][pair.doc_id]

    def retrieved(self, pair: Pair, depth: int) -> list[int]:
        """The top-``depth`` dense-ranked segments for a pair (never uses gold)."""
        return list(self.rankings[pair.split][pair.pair_id].segments[:depth])


def document_to_row(doc: Document) -> dict[str, Any]:
    return {
        "doc_id": doc.doc_id,
        "file_name": doc.file_name,
        "text": doc.text,
        "segments": doc.segments,
        "official_spans": doc.official_spans,
        "official_to_segments": doc.official_to_segments,
    }


def pair_to_row(pair: Pair) -> dict[str, Any]:
    return {
        "doc_id": pair.doc_id,
        "key": pair.key,
        "label": pair.label,
        "gold_official": pair.gold_official,
        "gold_segments": pair.gold_segments,
    }


def verify_files(root: Path, manifest: dict[str, Any]) -> None:
    for name, expected in manifest["files"].items():
        actual = file_sha256(root / name)
        if actual != expected:
            raise ValueError(f"{root / name} SHA-256 {actual} != manifest {expected}")


def load_dataset(root: Path, *, splits: tuple[str, ...] = SPLITS, verify: bool = True) -> Dataset:
    manifest = read_json(root / "manifest.json")
    if verify:
        verify_files(root, manifest)
    hypotheses = {str(k): str(v) for k, v in read_json(root / "hypotheses.json").items()}
    documents: dict[str, dict[str, Document]] = {}
    pairs: dict[str, list[Pair]] = {}
    rankings: dict[str, dict[str, Ranking]] = {}
    for split in splits:
        docs: dict[str, Document] = {}
        for row in read_jsonl_gz(root / f"documents_{split}.jsonl.gz"):
            docs[row["doc_id"]] = Document(
                split=split,
                doc_id=row["doc_id"],
                file_name=row["file_name"],
                text=row["text"],
                segments=[(int(a), int(b)) for a, b in row["segments"]],
                official_spans=[(int(a), int(b)) for a, b in row["official_spans"]],
                official_to_segments=[
                    tuple(int(i) for i in m) for m in row["official_to_segments"]
                ],
            )
        documents[split] = docs
        pairs[split] = [
            Pair(
                split=split,
                doc_id=row["doc_id"],
                key=row["key"],
                label=row["label"],
                gold_official=tuple(row["gold_official"]),
                gold_segments=tuple(row["gold_segments"]),
            )
            for row in read_jsonl_gz(root / f"pairs_{split}.jsonl.gz")
        ]
        rankings[split] = {
            row["pair_id"]: Ranking(tuple(row["segments"]), tuple(row["scores"]))
            for row in read_jsonl_gz(root / f"retrieval_{split}.jsonl.gz")
        }
        missing = [p.pair_id for p in pairs[split] if p.pair_id not in rankings[split]]
        if missing:
            raise ValueError(f"{split}: {len(missing)} pairs have no retrieval ranking")
    return Dataset(root, manifest, hypotheses, documents, pairs, rankings)
