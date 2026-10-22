"""TRAIN sampling, held-out isolation, scoring guards and LLM answer accounting."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from compliance_model_service.rule_check.labels import LABELS
from ml.baselines.llm import OUTPUT_SCHEMA, cost_usd, parse_answer, request_hash
from ml.data.contractnli import Document, Pair
from ml.data.store import Dataset, Ranking
from ml.eval import scoring
from ml.eval.systems import SystemPrediction
from ml.train.config import SamplingSettings, load_config
from ml.train.examples import build_epoch, evidence_examples, label_example

CONFIGS = Path(__file__).resolve().parents[1] / "train" / "configs"


def _dataset() -> Dataset:
    text = " ".join(f"Sentence {i}." for i in range(12))
    segs = []
    pos = 0
    for i in range(12):
        s = text.index(f"Sentence {i}.", pos)
        segs.append((s, s + len(f"Sentence {i}.")))
        pos = s + 1
    doc = Document("train", "d1", "f", text, segs, list(segs), [(i,) for i in range(12)])
    pairs = [
        Pair("train", "d1", "nda-1", "entailment", (3, 4), (3, 4)),
        Pair("train", "d1", "nda-2", "not_mentioned", (), ()),
    ]
    rank = Ranking(tuple(range(11, -1, -1)), tuple(float(x) for x in range(12)))
    return Dataset(
        Path("/x"),
        {"data_version": "v", "name": "unit"},
        {"nda-1": "h1", "nda-2": "h2"},
        {"train": {"d1": doc}},
        {"train": pairs},
        {"train": {p.pair_id: rank for p in pairs}},
    )


def test_evidence_negatives_never_include_gold() -> None:
    ds = _dataset()
    sampling = SamplingSettings(hard_negatives=3, random_negatives=4, hard_negative_depth=5)
    rng = random.Random(1)
    for _ in range(50):
        ex = evidence_examples(ds, ds.pairs["train"][0], sampling, rng)
        positives = {e.segment for e in ex if e.target == 1.0}
        negatives = [e.segment for e in ex if e.target == 0.0]
        assert positives == {3, 4}
        assert not positives & set(negatives)
        assert len(negatives) == len(set(negatives)) == 7
    nm = evidence_examples(ds, ds.pairs["train"][1], sampling, rng)
    assert all(e.target == 0.0 for e in nm)


def test_label_inputs_are_document_ordered_and_sized() -> None:
    ds = _dataset()
    index = {label: i for i, label in enumerate(LABELS)}
    sampling = SamplingSettings(label_k_choices=[4], gold_drop_prob=0.0)
    ex = label_example(ds, ds.pairs["train"][0], sampling, random.Random(0), index)
    assert ex.segments == tuple(sorted(ex.segments))
    assert len(ex.segments) == 4
    assert {3, 4} <= set(ex.segments)  # gold kept, remaining slots from the ranking
    nm = label_example(ds, ds.pairs["train"][1], sampling, random.Random(0), index)
    assert nm.segments == (8, 9, 10, 11)  # top of the dense ranking (no gold exists)
    ev1, lab1 = build_epoch(ds, ds.pairs["train"], sampling, seed=7, epoch=1, label_index=index)
    ev2, lab2 = build_epoch(ds, ds.pairs["train"], sampling, seed=7, epoch=1, label_index=index)
    assert (ev1, lab1) == (ev2, lab2)  # deterministic per (seed, epoch) -> resumable


def test_configs_validate_and_unseen_rules_are_fixed() -> None:
    primary = load_config(CONFIGS / "deberta_v3_base.yaml")
    unseen = load_config(CONFIGS / "deberta_v3_base_unseen4.yaml")
    ablation = load_config(CONFIGS / "modernbert_base.yaml")
    smoke = load_config(CONFIGS / "smoke.yaml")
    assert unseen.data.held_out_hypotheses == ["nda-1", "nda-8", "nda-16", "nda-18"]
    drawn = random.Random(20261021).sample(
        [
            "nda-1",
            "nda-2",
            "nda-3",
            "nda-4",
            "nda-5",
            "nda-7",
            "nda-8",
            "nda-10",
            "nda-11",
            "nda-12",
            "nda-13",
            "nda-15",
            "nda-16",
            "nda-17",
            "nda-18",
            "nda-19",
            "nda-20",
        ],
        4,
    )
    assert sorted(drawn, key=lambda k: int(k[4:])) == unseen.data.held_out_hypotheses
    # the unseen run differs from the primary only in its name and held-out rules
    a = primary.model_dump()
    b = unseen.model_dump()
    for d in (a, b):
        d.pop("run_name")
        d["data"].pop("held_out_hypotheses")
    assert a == b
    # the ablation is a pure backbone swap
    a, c = primary.model_dump(), ablation.model_dump()
    for d in (a, c):
        for key in ("run_name", "backbone"):
            d.pop(key)
        d["encoder"].pop("normalization")
    assert a == c
    assert primary.kind == "production"
    assert primary.requires_cuda
    assert smoke.kind == "smoke"
    assert smoke.backbone.kind == "random_tiny"


def test_scoring_rejects_mismatched_pairs_and_counts_unanswered() -> None:
    ds = _dataset()
    pairs = ds.pairs["train"]
    good = [
        SystemPrediction(pairs[0].pair_id, "entailment"),
        SystemPrediction(pairs[1].pair_id, None),
    ]
    result = scoring.score(ds, pairs, good, bootstrap=False)
    assert result["label"]["n"] == 2
    assert result["label"]["accuracy"] == 0.5
    with pytest.raises(ValueError, match="pair set differs"):
        scoring.score(ds, pairs, good[:1], bootstrap=False)


def test_llm_answer_parsing_and_cost() -> None:
    assert OUTPUT_SCHEMA["additionalProperties"] is False
    label, kept, ignored = parse_answer(
        '{"label": "Entailment", "evidence_ids": ["s2", "S9", "S2"]}', {"S1", "S2"}
    )
    assert (label, kept, ignored) == ("entailment", ["S2"], ["S9"])
    with pytest.raises(ValueError, match="not in"):
        parse_answer('{"label": "maybe", "evidence_ids": []}', {"S1"})
    price = {"input": 2.0, "output": 10.0, "cache_write_5m": 2.5, "cache_read": 0.1}
    usage = {
        "input_tokens": 1000,
        "output_tokens": 100,
        "cache_creation_input_tokens": 400,
        "cache_read_input_tokens": 400,
    }
    assert cost_usd(usage, price) == pytest.approx((2000 + 1000 + 1000 + 40) / 1e6)
    assert request_hash("m", "low") == request_hash("m", "low") != request_hash("m", "medium")
