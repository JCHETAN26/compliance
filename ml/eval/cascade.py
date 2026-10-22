"""Model + LLM cascade (handoff section 7.6): coverage/accuracy curves and DEV-chosen tau.

A pair is *covered* by the model when its calibrated confidence is >= tau; otherwise it
is escalated to the LLM. For a grid of tau values this reports coverage, the model's
accuracy on covered pairs and, when real LLM predictions exist for the same pairs, the
cascade's overall accuracy and macro-F1, LLM cost per document and mean latency per
pair.

Tau is selected on DEV only, with a rule fixed before any result was seen:

- with LLM DEV predictions: the smallest tau (= the largest coverage) whose DEV cascade
  accuracy is at least the LLM-only DEV accuracy minus ``TOLERANCE`` (1 point);
- without them: a *provisional* model-only tau, the smallest tau whose covered DEV
  accuracy is >= ``PROVISIONAL_TARGET`` (0.90). It is labelled provisional and no
  cascade accuracy or cost is claimed.

The selected tau is then applied unchanged to TEST.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ml.eval import metrics

TAU_GRID = tuple(round(x, 3) for x in np.arange(0.30, 1.0001, 0.01))
TOLERANCE = 0.01
PROVISIONAL_TARGET = 0.90


def curve(
    gold: Sequence[str],
    model_label: Sequence[str],
    confidence: Sequence[float],
    *,
    llm_label: Sequence[str | None] | None = None,
    llm_cost: Sequence[float] | None = None,
    llm_latency: Sequence[float] | None = None,
    model_latency_s: float | None = None,
    documents: int,
) -> list[dict[str, Any]]:
    n = len(gold)
    rows = []
    for tau in TAU_GRID:
        covered = [c >= tau for c in confidence]
        n_cov = sum(covered)
        row: dict[str, Any] = {
            "tau": tau,
            "coverage": n_cov / n,
            "model_accuracy_on_covered": (
                sum(g == m for g, m, c in zip(gold, model_label, covered, strict=True) if c) / n_cov
                if n_cov
                else None
            ),
        }
        if llm_label is not None:
            final = [m if c else x for m, x, c in zip(model_label, llm_label, covered, strict=True)]
            row["cascade_accuracy"] = sum(g == f for g, f in zip(gold, final, strict=True)) / n
            row["cascade_macro_f1"] = metrics.macro_f1(gold, final)
            if llm_cost is not None:
                cost = sum(x for x, c in zip(llm_cost, covered, strict=True) if not c)
                row["llm_cost_per_document_usd"] = cost / documents
            if llm_latency is not None and model_latency_s is not None:
                lat = [
                    model_latency_s + (0.0 if c else x)
                    for x, c in zip(llm_latency, covered, strict=True)
                ]
                row["mean_latency_per_pair_s"] = float(np.mean(lat))
        rows.append(row)
    return rows


def select_tau(
    dev_curve: list[dict[str, Any]], llm_only_dev_accuracy: float | None
) -> dict[str, Any]:
    if llm_only_dev_accuracy is not None:
        floor = llm_only_dev_accuracy - TOLERANCE
        ok = [r for r in dev_curve if r["cascade_accuracy"] >= floor]
        chosen = min(ok, key=lambda r: r["tau"]) if ok else max(dev_curve, key=lambda r: r["tau"])
        return {
            "tau": chosen["tau"],
            "rule": f"smallest tau with DEV cascade accuracy >= LLM-only DEV accuracy - {TOLERANCE}",
            "llm_only_dev_accuracy": llm_only_dev_accuracy,
            "satisfiable": bool(ok),
            "provisional": False,
        }
    ok = [
        r
        for r in dev_curve
        if r["model_accuracy_on_covered"] is not None
        and r["model_accuracy_on_covered"] >= PROVISIONAL_TARGET
    ]
    chosen = min(ok, key=lambda r: r["tau"]) if ok else max(dev_curve, key=lambda r: r["tau"])
    return {
        "tau": chosen["tau"],
        "rule": f"PROVISIONAL (no real LLM DEV results): smallest tau with covered DEV accuracy >= {PROVISIONAL_TARGET}",
        "satisfiable": bool(ok),
        "provisional": True,
    }


def at_tau(rows: list[dict[str, Any]], tau: float) -> dict[str, Any]:
    return next(r for r in rows if abs(r["tau"] - tau) < 1e-9)
