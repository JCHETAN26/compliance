"""Metric, calibration, cascade and scoring definitions on hand-checked examples."""

from __future__ import annotations

import math

import numpy as np
import pytest

from compliance_model_service.rule_check.encoding import softmax
from ml.eval import cascade, metrics
from ml.eval.calibration import fit_temperature, select_threshold

E, C, N = "entailment", "contradiction", "not_mentioned"


def test_label_metrics_hand_computed() -> None:
    gold = [E, E, E, C, N, N]
    pred = [E, E, N, C, N, None]  # one unanswered -> wrong, still counted
    m = metrics.label_metrics(gold, pred)
    assert m["n"] == 6
    assert m["unanswered"] == 1
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"][E] == pytest.approx(
        {"precision": 1.0, "recall": 2 / 3, "f1": 0.8, "support": 3, "predicted": 2}
    )
    assert m["per_class"][C]["f1"] == 1.0
    # not_mentioned: predicted twice (one right), support 2 -> P 0.5, R 0.5
    assert m["per_class"][N]["f1"] == pytest.approx(0.5)
    assert m["macro_f1"] == pytest.approx((0.8 + 1.0 + 0.5) / 3)
    assert m["confusion"]["rows_gold_cols_pred"] == [[2, 0, 1], [0, 1, 0], [0, 0, 1]]


def test_evidence_and_ranking_metrics() -> None:
    assert metrics.evidence_prf([{1, 2}, {0}], [{1}, {0, 3}]) == {
        "pairs": 2,
        "precision": 2 / 3,
        "recall": 2 / 3,
        "f1": pytest.approx(2 / 3),
        "predicted_units": 3,
        "gold_units": 3,
    }
    # ranking 2, 0, 1 (ties -> lower index): gold {0, 1} -> AP = (1/2 + 2/3) / 2
    assert metrics.average_precision([0.5, 0.2, 0.9], {0, 1}) == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert metrics.average_precision([0.5, 0.5, 0.5], {0}) == 1.0
    assert metrics.precision_at_k([[0.9, 0.1, 0.8]], [{2}], 2) == 0.5
    with pytest.raises(ValueError, match="at least one gold"):
        metrics.average_precision([0.1], set())


def test_ece_definition() -> None:
    result = metrics.expected_calibration_error(
        [0.9, 0.9, 0.6, 0.6], [True, False, True, True], bins=10
    )
    # bin (0.8,0.9]: acc .5 vs conf .9; bin (0.5,0.6]: acc 1 vs conf .6
    assert result["ece"] == pytest.approx(0.5 * 0.4 + 0.5 * 0.4)
    assert sum(b["count"] for b in result["reliability"]) == 4


def test_temperature_recovers_generating_temperature() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, size=(4000, 3))
    true_t = 2.5
    probs = softmax(logits, true_t)
    gold = [int(rng.choice(3, p=p)) for p in probs]
    fit = fit_temperature(logits, gold)
    assert fit["temperature"] == pytest.approx(true_t, rel=0.1)
    assert fit["nll_after"] <= fit["nll_before"]


def test_threshold_selection_maximises_f1() -> None:
    scores = [[0.9, 0.4, 0.1], [0.7, 0.65, 0.2]]
    gold = [{0}, {0, 1}]
    choice = select_threshold(scores, gold, grid=(0.3, 0.5, 0.6, 0.8))
    assert choice == {"threshold": 0.5, "dev_f1": 1.0}  # 0.5 and 0.6 tie -> lower wins
    assert select_threshold([[0.9, 0.4]], [{0}], grid=(0.3, 0.8))["threshold"] == 0.8


def test_cascade_rules() -> None:
    gold = [E, E, C, N]
    model = [E, C, C, N]
    conf = [0.95, 0.55, 0.9, 0.99]
    llm = [E, E, N, N]
    rows = cascade.curve(
        gold,
        model,
        conf,
        llm_label=llm,
        llm_cost=[1.0] * 4,
        llm_latency=[2.0] * 4,
        model_latency_s=0.1,
        documents=2,
    )
    at = cascade.at_tau(rows, 0.9)
    assert at["coverage"] == 0.75
    assert at["model_accuracy_on_covered"] == 1.0
    assert at["cascade_accuracy"] == 1.0  # the uncertain wrong one goes to the LLM
    assert at["llm_cost_per_document_usd"] == 0.5
    assert at["mean_latency_per_pair_s"] == pytest.approx(0.1 + 2.0 / 4)
    choice = cascade.select_tau(rows, llm_only_dev_accuracy=0.75)
    assert choice["provisional"] is False
    assert choice["tau"] == 0.3  # all-model already reaches 0.75 - 0.01
    provisional = cascade.select_tau(cascade.curve(gold, model, conf, documents=2), None)
    assert provisional["provisional"] is True
    assert "PROVISIONAL" in provisional["rule"]
    assert math.isclose(provisional["tau"], 0.56)
