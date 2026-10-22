"""Label, evidence and calibration metrics (pure functions, no I/O).

Label metrics: accuracy, macro-F1 over the three labels, per-class precision / recall /
F1 / support, and the confusion matrix (rows = gold, columns = predicted, ``LABELS``
order). A pair with no prediction (e.g. a failed LLM call) is passed as ``None``: it
counts as wrong for accuracy and as a miss for its gold class's recall; it is never
dropped from the denominator.

Evidence metrics follow the ContractNLI convention: they are computed over pairs whose
*gold* label is entailment or contradiction (pairs with annotated evidence).

- ``evidence_prf``: micro precision / recall / F1 of a predicted evidence *set* against
  the gold set, summed over pairs.
- ``mean_average_precision``: mean over pairs of average precision of a ranked list
  (every unit scored), ties broken by unit index.
- ``precision_at_k``: mean over pairs of |top-k ∩ gold| / k.

Calibration: expected calibration error with equal-width confidence bins (default 15)
on the predicted label's probability, and a reliability table for plotting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from compliance_model_service.rule_check.labels import LABELS


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def label_metrics(gold: Sequence[str], pred: Sequence[str | None]) -> dict[str, Any]:
    if len(gold) != len(pred):
        raise ValueError("gold and predictions differ in length")
    n = len(gold)
    index = {label: i for i, label in enumerate(LABELS)}
    confusion = [[0] * len(LABELS) for _ in LABELS]
    unanswered = 0
    for g, p in zip(gold, pred, strict=True):
        if p is None:
            unanswered += 1
            continue
        confusion[index[g]][index[p]] += 1
    per_class: dict[str, dict[str, float]] = {}
    for label, i in index.items():
        tp = confusion[i][i]
        predicted = sum(row[i] for row in confusion)
        support = sum(1 for g in gold if g == label)
        precision = _safe_div(tp, predicted)
        recall = _safe_div(tp, support)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
        }
    correct = sum(confusion[i][i] for i in range(len(LABELS)))
    return {
        "n": n,
        "unanswered": unanswered,
        "accuracy": _safe_div(correct, n),
        "macro_f1": sum(c["f1"] for c in per_class.values()) / len(LABELS),
        "per_class": per_class,
        "confusion": {"labels": list(LABELS), "rows_gold_cols_pred": confusion},
    }


def macro_f1(gold: Sequence[str], pred: Sequence[str | None]) -> float:
    return float(label_metrics(gold, pred)["macro_f1"])


def evidence_prf(gold_sets: Sequence[set[int]], pred_sets: Sequence[set[int]]) -> dict[str, float]:
    tp = sum(len(g & p) for g, p in zip(gold_sets, pred_sets, strict=True))
    n_pred = sum(len(p) for p in pred_sets)
    n_gold = sum(len(g) for g in gold_sets)
    precision = _safe_div(tp, n_pred)
    recall = _safe_div(tp, n_gold)
    return {
        "pairs": len(gold_sets),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "predicted_units": n_pred,
        "gold_units": n_gold,
    }


def ranking(scores: Sequence[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def average_precision(scores: Sequence[float], gold: set[int]) -> float:
    if not gold:
        raise ValueError("average precision needs at least one gold unit")
    hits = 0
    total = 0.0
    for rank, unit in enumerate(ranking(scores), start=1):
        if unit in gold:
            hits += 1
            total += hits / rank
    return total / len(gold)


def mean_average_precision(scores: Sequence[Sequence[float]], gold: Sequence[set[int]]) -> float:
    values = [average_precision(s, g) for s, g in zip(scores, gold, strict=True)]
    return float(np.mean(values)) if values else 0.0


def precision_at_k(scores: Sequence[Sequence[float]], gold: Sequence[set[int]], k: int) -> float:
    values = [len(set(ranking(s)[:k]) & g) / k for s, g in zip(scores, gold, strict=True)]
    return float(np.mean(values)) if values else 0.0


def expected_calibration_error(
    confidence: Sequence[float], correct: Sequence[bool], bins: int = 15
) -> dict[str, Any]:
    conf = np.asarray(confidence, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        raise ValueError("no predictions")
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Bin i holds (edges[i], edges[i+1]]; confidence 0 goes to the first bin.
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, bins - 1)
    table = []
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        count = int(mask.sum())
        if count:
            acc = float(hit[mask].mean())
            mean_conf = float(conf[mask].mean())
            ece += count / conf.size * abs(acc - mean_conf)
        else:
            acc = mean_conf = math.nan
        table.append(
            {
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": count,
                "accuracy": None if math.isnan(acc) else acc,
                "mean_confidence": None if math.isnan(mean_conf) else mean_conf,
            }
        )
    return {"ece": float(ece), "bins": bins, "n": int(conf.size), "reliability": table}


def nll(probabilities: np.ndarray, gold_index: Sequence[int]) -> float:
    p = np.clip(probabilities[np.arange(len(gold_index)), list(gold_index)], 1e-12, 1.0)
    return float(-np.log(p).mean())
