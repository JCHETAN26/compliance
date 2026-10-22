"""Full-pipeline inference over a split, for any forward function (PyTorch or ONNX).

For every (document, hypothesis) pair this does exactly what the model service does
(``compliance_model_service.rule_check.engine``): score every segment with the evidence
head, take the evidence head's own top-k (``top_k_in_order``), build one label input
from them and read the label head. No gold information is used. The label logits are
stored per requested k so that k can be *selected on DEV*; TEST is predicted with the
single DEV-selected k.

Predictions are stored raw (logits); temperature scaling and thresholds are applied by
``ml.eval.evaluate`` with values fitted on DEV.

Candidate-scope variant (``candidates``): the same evidence pass, but the top-k for the
label head is chosen only among a fixed candidate list per pair (the BGE top-8 that the
TF-IDF and LLM baselines read). It isolates head quality from the learned all-segment
evidence retrieval; it is reported separately and never replaces the main pipeline.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from compliance_model_service.rule_check.encoding import InputEncoder, sigmoid, top_k_in_order
from compliance_model_service.rule_check.onnx_backend import length_groups
from ml.common.io import read_jsonl_gz, write_jsonl_gz
from ml.data.contractnli import Pair
from ml.data.store import Dataset

Forward = Callable[[list[list[int]]], tuple[np.ndarray, np.ndarray]]


@dataclass
class PairPrediction:
    pair_id: str
    doc_id: str
    key: str
    gold: str
    evidence_logits: list[float]
    label_logits: dict[int, list[float]]  # k -> 3 logits
    truncated_evidence_inputs: int = 0
    truncated_label_inputs: dict[int, bool] = field(default_factory=dict)
    candidates: list[int] | None = None  # fixed candidate segments (variant), if any
    candidate_label_logits: dict[int, list[float]] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "doc_id": self.doc_id,
            "key": self.key,
            "gold": self.gold,
            "evidence_logits": [round(x, 6) for x in self.evidence_logits],
            "label_logits": {
                str(k): [round(x, 6) for x in v] for k, v in self.label_logits.items()
            },
            "truncated_evidence_inputs": self.truncated_evidence_inputs,
            "truncated_label_inputs": {str(k): v for k, v in self.truncated_label_inputs.items()},
            "candidates": self.candidates,
            "candidate_label_logits": {
                str(k): [round(x, 6) for x in v] for k, v in self.candidate_label_logits.items()
            },
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> PairPrediction:
        return cls(
            pair_id=row["pair_id"],
            doc_id=row["doc_id"],
            key=row["key"],
            gold=row["gold"],
            evidence_logits=[float(x) for x in row["evidence_logits"]],
            label_logits={int(k): [float(x) for x in v] for k, v in row["label_logits"].items()},
            truncated_evidence_inputs=int(row.get("truncated_evidence_inputs", 0)),
            truncated_label_inputs={
                int(k): bool(v) for k, v in row.get("truncated_label_inputs", {}).items()
            },
            candidates=row.get("candidates"),
            candidate_label_logits={
                int(k): [float(x) for x in v]
                for k, v in row.get("candidate_label_logits", {}).items()
            },
        )


def save_predictions(path: Path, predictions: Sequence[PairPrediction]) -> None:
    write_jsonl_gz(path, (p.to_row() for p in predictions))


def load_predictions(path: Path) -> list[PairPrediction]:
    return [PairPrediction.from_row(r) for r in read_jsonl_gz(path)]


class TokenCache:
    """Token ids (no special tokens) for hypotheses and every segment, computed once."""

    def __init__(self, dataset: Dataset, encoder: InputEncoder) -> None:
        self.dataset = dataset
        self.encoder = encoder
        keys = list(dataset.hypotheses)
        self.hypotheses = dict(
            zip(keys, encoder.tokenize([dataset.hypotheses[k] for k in keys]), strict=True)
        )
        self._segments: dict[tuple[str, str], list[list[int]]] = {}

    def segments(self, split: str, doc_id: str) -> list[list[int]]:
        key = (split, doc_id)
        if key not in self._segments:
            doc = self.dataset.documents[split][doc_id]
            self._segments[key] = self.encoder.tokenize(
                [doc.segment_text(i) for i in range(len(doc.segments))]
            )
        return self._segments[key]


def run_rows(
    forward: Forward, rows: list[list[int]], chunk_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    evidence = np.empty(len(rows), dtype=np.float32)
    labels = np.empty((len(rows), 3), dtype=np.float32)
    for start in range(0, len(rows), chunk_rows):
        ev, lab = forward(rows[start : start + chunk_rows])
        evidence[start : start + len(ev)] = ev
        labels[start : start + len(ev)] = lab
    return evidence, labels


def predict_pairs(
    pairs: Sequence[Pair],
    tokens: TokenCache,
    forward: Forward,
    *,
    top_k_values: Sequence[int],
    chunk_rows: int,
    progress: Callable[[str], None] | None = None,
    candidates: Callable[[Pair], Sequence[int]] | None = None,
) -> list[PairPrediction]:
    encoder = tokens.encoder
    started = time.perf_counter()
    # Pass 1: evidence rows for every pair (rows sorted by length inside ``forward``).
    rows: list[list[int]] = []
    spans: list[tuple[int, int]] = []
    truncated: list[int] = []
    for pair in pairs:
        seg_ids = tokens.segments(pair.split, pair.doc_id)
        hyp = tokens.hypotheses[pair.key]
        start = len(rows)
        cut = 0
        for s in seg_ids:
            encoded = encoder.evidence_input(hyp, s)
            rows.append(encoded.ids)
            cut += encoded.truncated
        spans.append((start, len(rows)))
        truncated.append(cut)
    ev_logits, _ = run_rows(forward, rows, chunk_rows)
    if progress:
        progress(f"    evidence pass: {len(rows)} rows in {time.perf_counter() - started:.0f}s")
    # Pass 2: one label row per (pair, k) from the evidence head's own top-k.
    label_rows: list[list[int]] = []
    label_index: list[tuple[int, int, bool, bool]] = []  # pair, k, truncated, candidate variant
    pair_candidates: list[list[int] | None] = []
    for p_idx, pair in enumerate(pairs):
        a, b = spans[p_idx]
        scores = [float(x) for x in sigmoid(ev_logits[a:b])]
        seg_ids = tokens.segments(pair.split, pair.doc_id)
        hyp = tokens.hypotheses[pair.key]
        allowed = sorted(set(candidates(pair))) if candidates is not None else None
        pair_candidates.append(allowed)
        for k in top_k_values:
            chosen = top_k_in_order(scores, min(k, len(scores)))
            encoded = encoder.label_input(hyp, [seg_ids[i] for i in chosen])
            label_rows.append(encoded.ids)
            label_index.append((p_idx, k, encoded.truncated, False))
            if allowed is not None:
                local = top_k_in_order([scores[i] for i in allowed], min(k, len(allowed)))
                encoded = encoder.label_input(hyp, [seg_ids[allowed[i]] for i in local])
                label_rows.append(encoded.ids)
                label_index.append((p_idx, k, encoded.truncated, True))
    _, lab_logits = run_rows(forward, label_rows, chunk_rows)
    results = [
        PairPrediction(
            pair_id=pair.pair_id,
            doc_id=pair.doc_id,
            key=pair.key,
            gold=pair.label,
            evidence_logits=[float(x) for x in ev_logits[spans[i][0] : spans[i][1]]],
            label_logits={},
            truncated_evidence_inputs=truncated[i],
            candidates=pair_candidates[i],
        )
        for i, pair in enumerate(pairs)
    ]
    for row, (p_idx, k, cut, variant) in enumerate(label_index):
        if variant:
            results[p_idx].candidate_label_logits[k] = [float(x) for x in lab_logits[row]]
            continue
        results[p_idx].label_logits[k] = [float(x) for x in lab_logits[row]]
        results[p_idx].truncated_label_inputs[k] = cut
    if progress:
        progress(f"    predicted {len(pairs)} pairs in {time.perf_counter() - started:.0f}s")
    return results


class TorchForward:
    """Batched PyTorch forward over token rows (length-grouped, fp16 autocast on CUDA)."""

    def __init__(
        self, model: Any, encoder: InputEncoder, *, device: str, fp16: bool, max_tokens: int
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.device = device
        self.fp16 = fp16 and device.startswith("cuda")
        self.max_tokens = max_tokens

    def __call__(self, rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        import torch

        evidence = np.empty(len(rows), dtype=np.float32)
        labels = np.empty((len(rows), 3), dtype=np.float32)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for group in length_groups([len(r) for r in rows], self.max_tokens):
                    ids, mask = self.encoder.pad([rows[i] for i in group])
                    ids_t = torch.from_numpy(ids).to(self.device)
                    mask_t = torch.from_numpy(mask).to(self.device)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                        ev, lab = self.model(ids_t, mask_t)
                    evidence[group] = ev.float().cpu().numpy()
                    labels[group] = lab.float().cpu().numpy()
        finally:
            self.model.train(was_training)
        return evidence, labels


def select_label(prediction: PairPrediction, k: int) -> np.ndarray:
    return np.asarray(prediction.label_logits[k], dtype=np.float64)
