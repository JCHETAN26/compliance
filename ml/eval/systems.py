"""One prediction format for every system (model, baselines), so all are scored alike.

A system's predictions for a split are a gzipped JSON-lines file with one row per
official pair (all pairs; a system that could not answer a pair writes ``label: null``)
plus a ``.meta.json`` describing the system, its inputs (``data_version``, candidate
depth) and its provenance. ``evaluate`` refuses files whose pair set or data version
differs from the dataset's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compliance_model_service.rule_check.labels import LABELS
from ml.common.io import read_json, read_jsonl_gz, write_json, write_jsonl_gz


@dataclass
class SystemPrediction:
    pair_id: str
    label: str | None  # None: no answer (counted wrong, never dropped)
    probs: list[float] | None = None  # LABELS order (calibrated where the system has a T)
    evidence_scores: list[float] | None = None  # one score per document segment
    evidence_set: list[int] | None = None  # predicted evidence segments
    cost_usd: float | None = None
    latency_s: float | None = None
    error: str | None = None
    # Evidence ids the system claimed that do not exist among its inputs (LLM). Each counts
    # as an evidence false positive in scoring; they are never silently dropped.
    invalid_evidence_claims: int | None = None

    def to_row(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None or k == "label"}

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> SystemPrediction:
        label = row.get("label")
        if label is not None and label not in LABELS:
            raise ValueError(f"invalid label {label!r} for {row.get('pair_id')}")
        return cls(
            pair_id=row["pair_id"],
            label=label,
            probs=row.get("probs"),
            evidence_scores=row.get("evidence_scores"),
            evidence_set=row.get("evidence_set"),
            cost_usd=row.get("cost_usd"),
            latency_s=row.get("latency_s"),
            error=row.get("error"),
            invalid_evidence_claims=row.get("invalid_evidence_claims"),
        )


def predictions_path(root: Path, system: str, split: str) -> Path:
    return root / f"{system}_{split}.jsonl.gz"


def save_system(
    root: Path, system: str, split: str, rows: list[SystemPrediction], meta: dict[str, Any]
) -> None:
    write_jsonl_gz(predictions_path(root, system, split), (r.to_row() for r in rows))
    write_json(root / f"{system}_{split}.meta.json", meta)


def load_system(
    root: Path, system: str, split: str
) -> tuple[list[SystemPrediction], dict[str, Any]]:
    rows = [
        SystemPrediction.from_row(r) for r in read_jsonl_gz(predictions_path(root, system, split))
    ]
    meta = read_json(root / f"{system}_{split}.meta.json")
    return rows, meta


def has_system(root: Path, system: str, split: str) -> bool:
    return (
        predictions_path(root, system, split).is_file()
        and (root / f"{system}_{split}.meta.json").is_file()
    )
