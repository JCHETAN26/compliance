"""Optimiser-skip semantics, checkpoint recovery, LLM accounting and evidence scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from ml.baselines import llm
from ml.data.contractnli import Document, Pair
from ml.data.store import Dataset, Ranking
from ml.eval import scoring
from ml.eval.systems import SystemPrediction
from ml.train.train import COMPLETE, CheckpointError, optimizer_update, recover_last


class FakeScaler:
    """GradScaler stand-in: ``overflow`` makes step() skip and update() halve the scale."""

    def __init__(self, overflow: bool) -> None:
        self.overflow = overflow
        self.scale = 1024.0
        self.stepped = False

    def is_enabled(self) -> bool:
        return True

    def unscale_(self, optimizer: Any) -> None:
        pass

    def get_scale(self) -> float:
        return self.scale

    def step(self, optimizer: Any) -> None:
        if not self.overflow:
            optimizer.step()
            self.stepped = True

    def update(self) -> None:
        self.scale = self.scale / 2 if self.overflow else self.scale


@pytest.mark.parametrize("overflow", [False, True])
def test_skipped_fp16_update_does_not_advance_schedule(overflow: bool) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))
    model(torch.ones(1, 2)).sum().backward()
    before = [p.detach().clone() for p in model.parameters()]
    scaler = FakeScaler(overflow)
    applied = optimizer_update(model, optimizer, scheduler, scaler, 1.0)
    assert applied is (not overflow)
    assert scheduler.last_epoch == (0 if overflow else 1)  # schedule advances only on a real update
    changed = any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    assert changed is (not overflow)
    assert all(p.grad is None for p in model.parameters())  # gradients cleared either way


def _checkpoint(path: Path, step: int, complete: bool = True) -> None:
    path.mkdir(parents=True)
    (path / "state.json").write_text(json.dumps({"step": step}))
    if complete:
        (path / COMPLETE).write_text("ok\n")


def test_checkpoint_recovery_layouts(tmp_path: Path) -> None:
    base = tmp_path / "run" / "checkpoints"
    assert recover_last(tmp_path / "run") is None  # nothing yet: a fresh start is correct

    # interrupted after last -> .last-old, before staging -> last
    _checkpoint(base / ".last-old", 10)
    _checkpoint(base / ".last-staging", 20)
    last = recover_last(tmp_path / "run")
    assert last == base / "last"
    assert json.loads((last / "state.json").read_text())["step"] == 20
    assert not (base / ".last-old").exists()
    assert not (base / ".last-staging").exists()

    # staging incomplete (died while writing): keep the previous complete checkpoint
    _checkpoint(base / ".last-staging", 30, complete=False)
    assert json.loads((recover_last(tmp_path / "run") / "state.json").read_text())["step"] == 20  # type: ignore[operator]

    # only an old complete one survived
    (base / "last").rename(base / ".last-old")
    assert json.loads((recover_last(tmp_path / "run") / "state.json").read_text())["step"] == 20  # type: ignore[operator]

    # nothing complete: refuse instead of silently restarting
    other = tmp_path / "other" / "checkpoints"
    _checkpoint(other / "last", 5, complete=False)
    with pytest.raises(CheckpointError):
        recover_last(tmp_path / "other")


def _tiny_dataset() -> tuple[Dataset, list[Pair]]:
    text = "One. Two. Three."
    segs = [(0, 4), (5, 9), (10, 16)]
    doc = Document("dev", "d", "f", text, segs, segs, [(0,), (1,), (2,)])
    pairs = [
        Pair("dev", "d", "nda-1", "entailment", (1,), (1,)),
        Pair("dev", "d", "nda-2", "not_mentioned", (), ()),
    ]
    rank = Ranking((1, 0, 2), (0.9, 0.5, 0.1))
    ds = Dataset(
        Path("/x"),
        {"data_version": "v", "name": "unit"},
        {"nda-1": "h1", "nda-2": "h2"},
        {"dev": {"d": doc}},
        {"dev": pairs},
        {"dev": {p.pair_id: rank for p in pairs}},
    )
    return ds, pairs


def test_llm_accounting_counts_every_known_attempt(tmp_path: Path) -> None:
    ds, pairs = _tiny_dataset()
    price = {"input": 2.0, "output": 10.0, "cache_write_5m": 2.5, "cache_read": 0.1}
    req = llm.request_hash("m", "low")
    usage = {
        "input_tokens": 1000,
        "output_tokens": 100,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    one_cost = (1000 * 2 + 100 * 10) / 1e6

    def attempt(
        pair: Pair,
        *,
        error: str | None,
        text: str | None,
        with_usage: bool = True,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "pair_id": pair.pair_id,
            "request_hash": req,
            "error": error,
            "latency_s": 1.0,
            "prompt_sha256": llm.prompt_hash(prompt or llm.user_prompt(ds, pair)),
        }
        if with_usage:
            row["response"] = {
                "usage": usage,
                "text": text or "",
                "stop_reason": "end_turn" if error is None else "max_tokens",
            }
        return row

    attempts = [
        attempt(pairs[0], error="stop_reason=max_tokens", text=None),  # billed failure
        attempt(pairs[0], error="api_connection: APITimeoutError", text=None, with_usage=False),
        attempt(pairs[0], error=None, text='{"label": "entailment", "evidence_ids": ["S2", "S7"]}'),
        # an answer to a different (stale) prompt must not be reused for pair 2
        attempt(
            pairs[1],
            error=None,
            text='{"label": "entailment", "evidence_ids": []}',
            prompt="old template",
        ),
    ]
    preds, summary = llm.to_predictions(ds, pairs, attempts, price, req)
    assert preds[0].label == "entailment"
    assert preds[0].evidence_set == [1]  # S2 = second candidate in document order = segment 1
    assert preds[0].invalid_evidence_claims == 1  # S7 was never shown
    assert preds[0].cost_usd == pytest.approx(2 * one_cost)  # both billed attempts
    assert preds[1].label is None
    assert preds[1].error == "no attempt recorded"
    bill = summary["billing_this_request_identity"]
    assert bill["attempts"] == 4
    assert bill["attempts_without_reported_usage"] == 1
    assert bill["cost_is_lower_bound"] is True
    assert bill["known_cost_usd"] == pytest.approx(3 * one_cost)
    assert llm.usable_answer(attempts, req, llm.prompt_hash(llm.user_prompt(ds, pairs[1]))) is None

    for bad in (
        ["--split", "dev", "--concurrency", "0"],
        ["--split", "dev", "--limit", "0"],
        ["--split", "dev", "--limit", "-3"],
    ):
        with pytest.raises(SystemExit):
            llm.main([*bad, "--data-dir", str(tmp_path)])


def test_invalid_evidence_claims_are_false_positives() -> None:
    ds, pairs = _tiny_dataset()
    clean = [
        SystemPrediction(pairs[0].pair_id, "entailment", evidence_set=[1]),
        SystemPrediction(pairs[1].pair_id, "not_mentioned", evidence_set=[]),
    ]
    claimed = [
        SystemPrediction(
            pairs[0].pair_id, "entailment", evidence_set=[1], invalid_evidence_claims=1
        ),
        SystemPrediction(pairs[1].pair_id, "not_mentioned", evidence_set=[0]),
    ]
    a = scoring.score(ds, pairs, clean, bootstrap=False)["evidence"]
    b = scoring.score(ds, pairs, claimed, bootstrap=False)["evidence"]
    assert a["segment"]["precision"] == 1.0
    assert b["segment"]["precision"] == 0.5
    assert b["segment"]["recall"] == 1.0
    assert b["invalid_claims_counted_as_false_positives"] == 1
    assert b["excluded_not_mentioned_pairs"] == 1
    assert b["units_predicted_on_not_mentioned_pairs"] == 1
