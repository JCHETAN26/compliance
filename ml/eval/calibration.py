"""Temperature scaling and decision thresholds, fitted on DEV only.

- ``fit_temperature``: the single T > 0 minimising DEV negative log-likelihood of the
  label head's logits (bounded scalar search over log T); probabilities are
  ``softmax(logits / T)``. T does not change the argmax, only confidence.
- ``select_threshold``: the evidence-probability threshold maximising DEV segment-level
  evidence F1 on a fixed grid (ties: the lower threshold).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

from compliance_model_service.rule_check.encoding import softmax
from ml.eval.metrics import evidence_prf, nll

THRESHOLD_GRID = tuple(round(x, 2) for x in np.arange(0.05, 0.951, 0.01))


def fit_temperature(logits: np.ndarray, gold_index: Sequence[int]) -> dict[str, float]:
    if len(gold_index) == 0:
        raise ValueError("cannot fit a temperature on no examples")

    def objective(log_t: float) -> float:
        return nll(softmax(logits, float(np.exp(log_t))), gold_index)

    result = minimize_scalar(
        objective, bounds=(-3.0, 3.0), method="bounded", options={"xatol": 1e-4}
    )
    temperature = float(np.exp(result.x))
    return {
        "temperature": temperature,
        "nll_before": objective(0.0),
        "nll_after": float(result.fun),
        "n": len(gold_index),
    }


def select_threshold(
    scores: Sequence[Sequence[float]],
    gold: Sequence[set[int]],
    grid: Sequence[float] = THRESHOLD_GRID,
) -> dict[str, float]:
    best_threshold, best_f1 = grid[0], -1.0
    for threshold in grid:
        pred = [{i for i, s in enumerate(row) if s >= threshold} for row in scores]
        f1 = evidence_prf(gold, pred)["f1"]
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return {"threshold": float(best_threshold), "dev_f1": float(best_f1)}
