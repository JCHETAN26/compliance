"""Shared rule-check input encoding (the training == serving contract)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compliance_model_service.rule_check.encoding import (
    EncoderConfig,
    EncoderError,
    InputEncoder,
    softmax,
    top_k_in_order,
    water_fill,
)


@pytest.fixture(scope="module")
def encoder(model_path: Path) -> InputEncoder:
    # The pinned BGE WordPiece tokenizer has [CLS]/[SEP]/[PAD]; any tokenizer.json works.
    return InputEncoder.from_file(
        model_path / "tokenizer.json",
        EncoderConfig(max_hypothesis_tokens=8, max_evidence_tokens=24, max_label_tokens=40),
    )


def test_water_fill_shares_budget_fairly() -> None:
    assert water_fill([2, 50, 50], 30) == [2, 14, 14]
    assert water_fill([3, 4], 100) == [3, 4]
    assert water_fill([10, 10, 10], 7) == [2, 2, 3]
    assert sum(water_fill([5, 9, 1, 30], 20)) == 20
    with pytest.raises(EncoderError):
        water_fill([1], -1)


def test_evidence_input_layout_and_truncation(encoder: InputEncoder) -> None:
    hyp = list(range(1000, 1012))  # 12 ids, cap is 8
    span = list(range(2000, 2030))
    encoded = encoder.evidence_input(hyp, span)
    assert len(encoded.ids) == 24
    assert encoded.ids[0] == encoder.cls_id
    assert encoded.ids[1:9] == hyp[:8]
    assert encoded.ids[9] == encoder.sep_id
    assert encoded.ids[-1] == encoder.sep_id
    assert encoded.ids[10:-1] == span[: 24 - 8 - 3]  # span cut at its end
    assert encoded.truncated
    short = encoder.evidence_input([5, 6], [7, 8])
    assert short.ids == [encoder.cls_id, 5, 6, encoder.sep_id, 7, 8, encoder.sep_id]
    assert not short.truncated


def test_label_input_keeps_order_and_budget(encoder: InputEncoder) -> None:
    spans = [[1] * 3, [2] * 30, [3] * 30]
    encoded = encoder.label_input([9, 9], spans)
    assert len(encoded.ids) == 40
    assert encoded.truncated
    body = encoded.ids[4:]
    # document order is preserved: all 1s, then 2s, then 3s, separated by [SEP]
    flat = [t for t in body if t != encoder.sep_id]
    assert flat == sorted(flat)
    assert body.count(encoder.sep_id) == 3
    assert flat.count(1) == 3  # the short span keeps every token
    with pytest.raises(EncoderError):
        encoder.label_input([9], [[1]] * 30)


def test_top_k_in_order_breaks_ties_by_index() -> None:
    assert top_k_in_order([0.1, 0.9, 0.5, 0.9], 2) == [1, 3]
    assert top_k_in_order([0.5, 0.5, 0.5], 2) == [0, 1]
    assert top_k_in_order([0.2], 5) == [0]
    with pytest.raises(EncoderError):
        top_k_in_order([0.2], 0)


def test_nfkc_normalisation_is_applied_before_tokenisation(model_path: Path) -> None:
    plain = InputEncoder.from_file(model_path / "tokenizer.json", EncoderConfig())
    nfkc = InputEncoder.from_file(
        model_path / "tokenizer.json", EncoderConfig(normalization="NFKC")
    )
    text = "Section Ⅷ ﬁnal …"  # roman numeral 8, fi ligature, ellipsis
    assert nfkc.tokenize([text]) == plain.tokenize(["Section VIII final ..."])
    with pytest.raises(EncoderError):
        EncoderConfig(normalization="NFD")  # type: ignore[arg-type]


def test_pad_and_softmax(encoder: InputEncoder) -> None:
    ids, mask = encoder.pad([[1, 2, 3], [4]])
    assert ids.tolist() == [[1, 2, 3], [4, encoder.pad_id, encoder.pad_id]]
    assert mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    probs = softmax(np.array([2.0, 0.0, 0.0]), 2.0)
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] == pytest.approx(np.e / (np.e + 2))
    with pytest.raises(EncoderError):
        softmax(np.array([1.0]), 0.0)
