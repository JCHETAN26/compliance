"""Score any system's predictions on a split with one fixed set of metrics.

Every system is scored on the *same* list of official pairs; a missing pair is an error
and an unanswered pair (``label is None``) counts as wrong. Evidence metrics use pairs
whose gold label is entailment or contradiction (ContractNLI convention); the number of
excluded NotMentioned pairs and the evidence a system nevertheless predicted on them are
reported alongside. Denominators are end-to-end: gold is every annotated evidence unit of
the document, so evidence a system never saw (e.g. missed by a top-8 candidate stage)
counts as a false negative, and an evidence id a system claimed that was not among its
inputs counts as a false positive.

- segment level: P/R/F1 of the predicted set; mAP and P@{1,3,5} of the score ranking
  over all document segments;
- official-span level (projection): an official span is predicted iff it overlaps a
  predicted segment; its score is the max score of the segments it overlaps.

Confidence intervals: document-clustered bootstrap (documents resampled with
replacement, all their pairs kept together), 95 % percentile intervals.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from compliance_model_service.rule_check.labels import LABELS
from ml.data.contractnli import Pair
from ml.data.store import Dataset
from ml.eval import metrics
from ml.eval.systems import SystemPrediction

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261021


def _index(pairs: Sequence[Pair], preds: Sequence[SystemPrediction]) -> list[SystemPrediction]:
    by_id = {p.pair_id: p for p in preds}
    if len(by_id) != len(preds):
        raise ValueError("duplicate pair ids in predictions")
    missing = [p.pair_id for p in pairs if p.pair_id not in by_id]
    extra = set(by_id) - {p.pair_id for p in pairs}
    if missing or extra:
        raise ValueError(f"prediction pair set differs: {len(missing)} missing, {len(extra)} extra")
    return [by_id[p.pair_id] for p in pairs]


def _official(dataset: Dataset, pair: Pair) -> tuple[list[tuple[int, ...]], set[int]]:
    doc = dataset.document(pair)
    mapping = doc.official_to_segments
    gold = {i for i in pair.gold_official if mapping[i]}
    return mapping, gold


def evidence_scores(
    dataset: Dataset, pairs: Sequence[Pair], preds: Sequence[SystemPrediction]
) -> dict[str, Any]:
    ev_pairs = [(p, s) for p, s in zip(pairs, preds, strict=True) if p.label != "not_mentioned"]
    nm_pairs = [s for p, s in zip(pairs, preds, strict=True) if p.label == "not_mentioned"]
    invalid = sum(s.invalid_evidence_claims or 0 for _, s in ev_pairs)
    out: dict[str, Any] = {
        "pairs": len(ev_pairs),
        "convention": "gold entailment/contradiction pairs; end-to-end denominators (all gold units of the document)",
        "excluded_not_mentioned_pairs": len(nm_pairs),
        "units_predicted_on_not_mentioned_pairs": sum(
            len(s.evidence_set or []) + (s.invalid_evidence_claims or 0) for s in nm_pairs
        ),
        "invalid_claims_counted_as_false_positives": invalid,
    }
    if all(s.evidence_set is not None for _, s in ev_pairs):
        gold = [set(p.gold_segments) for p, _ in ev_pairs]
        pred = [set(s.evidence_set or []) for _, s in ev_pairs]
        out["segment"] = _with_invalid(metrics.evidence_prf(gold, pred), invalid)
        off_gold, off_pred = [], []
        for (p, _), predicted in zip(ev_pairs, pred, strict=True):
            mapping, g = _official(dataset, p)
            off_gold.append(g)
            off_pred.append(
                {i for i, segs in enumerate(mapping) if segs and predicted.intersection(segs)}
            )
        out["official_span"] = _with_invalid(metrics.evidence_prf(off_gold, off_pred), invalid)
    if all(s.evidence_scores is not None for _, s in ev_pairs):
        gold = [set(p.gold_segments) for p, _ in ev_pairs]
        scores = [list(s.evidence_scores or []) for _, s in ev_pairs]
        for (p, _), row in zip(ev_pairs, scores, strict=True):
            if len(row) != len(dataset.document(p).segments):
                raise ValueError(
                    f"{p.pair_id}: {len(row)} evidence scores for {len(dataset.document(p).segments)} segments"
                )
        out.setdefault("segment", {})
        out["segment"]["map"] = metrics.mean_average_precision(scores, gold)
        for k in (1, 3, 5):
            out["segment"][f"p_at_{k}"] = metrics.precision_at_k(scores, gold, k)
        off_scores, off_gold = [], []
        for (p, _), row in zip(ev_pairs, scores, strict=True):
            mapping, g = _official(dataset, p)
            units = [i for i, segs in enumerate(mapping) if segs]
            off_scores.append([max(row[j] for j in mapping[i]) for i in units])
            off_gold.append({units.index(i) for i in g})
        out.setdefault("official_span", {})
        out["official_span"]["map"] = metrics.mean_average_precision(off_scores, off_gold)
        out["official_span"]["p_at_8"] = metrics.precision_at_k(off_scores, off_gold, 8)
    return out


def _with_invalid(prf: dict[str, float], invalid: int) -> dict[str, float]:
    """Add claimed-but-nonexistent evidence units to the predictions (false positives)."""
    if not invalid:
        return prf
    predicted = prf["predicted_units"] + invalid
    tp = prf["recall"] * prf["gold_units"]
    precision = tp / predicted if predicted else 0.0
    recall = prf["recall"]
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**prf, "predicted_units": predicted, "precision": precision, "f1": f1}


def _bootstrap(
    pairs: Sequence[Pair], statistic: Callable[[list[int]], float], *, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    by_doc: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        by_doc[p.doc_id].append(i)
    docs = sorted(by_doc)
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling
    values = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [i for d in rng.choices(docs, k=len(docs)) for i in by_doc[d]]
        values.append(statistic(sample))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {"low": float(lo), "high": float(hi), "samples": BOOTSTRAP_SAMPLES, "unit": "document"}


def score(
    dataset: Dataset,
    pairs: Sequence[Pair],
    preds: Sequence[SystemPrediction],
    *,
    bootstrap: bool = True,
) -> dict[str, Any]:
    ordered = _index(pairs, preds)
    gold = [p.label for p in pairs]
    pred = [s.label for s in ordered]
    result: dict[str, Any] = {"label": metrics.label_metrics(gold, pred)}
    if bootstrap:
        result["label"]["macro_f1_ci95"] = _bootstrap(
            pairs, lambda idx: metrics.macro_f1([gold[i] for i in idx], [pred[i] for i in idx])
        )
        result["label"]["accuracy_ci95"] = _bootstrap(
            pairs, lambda idx: float(np.mean([gold[i] == pred[i] for i in idx]))
        )
    per_key: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        per_key[p.key].append(i)
    result["per_hypothesis"] = {
        key: {
            "n": len(idx),
            "accuracy": float(np.mean([gold[i] == pred[i] for i in idx])),
            "macro_f1": metrics.macro_f1([gold[i] for i in idx], [pred[i] for i in idx]),
        }
        for key, idx in sorted(per_key.items(), key=lambda kv: int(kv[0].split("-")[-1]))
    }
    result["evidence"] = evidence_scores(dataset, pairs, ordered)
    if all(s.probs is not None for s in ordered):
        conf = [max(s.probs or []) for s in ordered]
        correct = [g == p for g, p in zip(gold, pred, strict=True)]
        result["calibration"] = metrics.expected_calibration_error(conf, correct)
    costs = [s.cost_usd for s in ordered if s.cost_usd is not None]
    latencies = [s.latency_s for s in ordered if s.latency_s is not None]
    if costs:
        docs = len({p.doc_id for p in pairs})
        result["cost"] = {
            "total_usd": float(sum(costs)),
            "per_document_usd": float(sum(costs)) / docs,
            "pairs_with_cost": len(costs),
        }
    if latencies:
        result["latency_s"] = {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "mean": float(np.mean(latencies)),
            "n": len(latencies),
        }
    result["errors"] = sum(s.error is not None for s in ordered)
    return result


def paired_difference(
    pairs: Sequence[Pair], a: Sequence[SystemPrediction], b: Sequence[SystemPrediction]
) -> dict[str, Any]:
    """macro-F1(a) - macro-F1(b) on the same pairs, with a document-clustered bootstrap CI."""
    a_o, b_o = _index(pairs, a), _index(pairs, b)
    gold = [p.label for p in pairs]
    pa = [s.label for s in a_o]
    pb = [s.label for s in b_o]

    def diff(idx: list[int]) -> float:
        g = [gold[i] for i in idx]
        return metrics.macro_f1(g, [pa[i] for i in idx]) - metrics.macro_f1(g, [pb[i] for i in idx])

    return {"macro_f1_difference": diff(list(range(len(pairs)))), "ci95": _bootstrap(pairs, diff)}


def label_from_probs(probs: Sequence[float]) -> str:
    return LABELS[int(np.argmax(probs))]
