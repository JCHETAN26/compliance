"""Rule pack, official splits, segmentation and the evidence-mapping rule."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from ml.common.provenance import REPO
from ml.data.contractnli import (
    build_split,
    label_counts,
    load_raw,
    map_official_to_segments,
    overlaps,
    segment,
)
from ml.data.prepare import inverse_frequency_weights

RULE_PACK = REPO / "rule_packs" / "nda_standard.yaml"


def test_rule_pack_maps_all_17_hypotheses_verbatim(dataset_zip: Path) -> None:
    pack = yaml.safe_load(RULE_PACK.read_text(encoding="utf-8"))
    labels = load_raw(dataset_zip, "train")["labels"]
    rules = pack["rules"]
    assert len(rules) == 17
    assert {r["contractnli_key"] for r in rules} == set(labels)
    assert len({r["key"] for r in rules}) == 17
    for rule in rules:
        assert rule["hypothesis_text"] == labels[rule["contractnli_key"]]["hypothesis"]
        assert rule["model_supported"] is True
        assert rule["scope"] == "per_document"
        assert rule["severity"] in {"low", "medium", "high"}
    for split in ("dev", "test"):  # identical hypothesis texts in every official split
        assert load_raw(dataset_zip, split)["labels"] == labels


def test_official_splits_counts_and_handoff_correction(dataset_zip: Path) -> None:
    expected = {"train": (423, 7191), "dev": (61, 1037), "test": (123, 2091)}
    doc_ids: dict[str, set[str]] = {}
    for split, (docs, pairs) in expected.items():
        data = build_split(dataset_zip, split)
        assert (len(data.documents), len(data.pairs)) == (docs, pairs)
        doc_ids[split] = set(data.documents)
        for pair in data.pairs:
            if pair.label == "not_mentioned":
                assert not pair.gold_official
                assert not pair.gold_segments
            else:
                assert pair.gold_segments  # every evidence pair maps to >= 1 segment
        if split == "train":
            counts = label_counts(data.pairs)
            # Observed directly; the handoff's "not_mentioned dominates" is wrong.
            assert counts == {"entailment": 3530, "contradiction": 841, "not_mentioned": 2820}
            weights = inverse_frequency_weights(counts)
            assert weights["contradiction"] > weights["not_mentioned"] > weights["entailment"]
    assert not (doc_ids["train"] & doc_ids["dev"])
    assert not (doc_ids["train"] & doc_ids["test"])
    assert not (doc_ids["dev"] & doc_ids["test"])


def test_segments_are_exact_trimmed_slices(dataset_zip: Path) -> None:
    data = build_split(dataset_zip, "dev")
    for doc in data.documents.values():
        previous_end = 0
        for start, end in doc.segments:
            assert previous_end <= start < end
            text = doc.text[start:end]
            assert text == text.strip()
            previous_end = end
        covered = {i for s, e in doc.segments for i in range(s, e)}
        # every non-whitespace character belongs to a segment
        assert all(i in covered for i, ch in enumerate(doc.text) if not ch.isspace())


def test_mapping_rule_boundaries() -> None:
    text = "Alpha one. Beta two.\n(a) item; (b) item.\n   \nGamma three."
    segments = segment(text)
    assert [text[s:e] for s, e in segments] == [
        "Alpha one.",
        "Beta two.",
        "(a) item; (b) item.",
        "Gamma three.",
    ]
    a_start = text.index("(a)")
    b_start = text.index("(b)")
    official = [
        (0, 10),  # exactly segment 0
        (0, 21),  # crosses segments 0 and 1 (and the newline)
        (a_start, a_start + 9),  # inside segment 2 (part of it)
        (b_start, b_start + 9),  # also inside segment 2
        (text.index("   "), text.index("   ") + 3),  # whitespace only
        (10, 11),  # a single space between segments 0 and 1
    ]
    mapping = map_official_to_segments(text, segments, official)
    assert mapping == [(0,), (0, 1), (2,), (2,), (), ()]
    assert not overlaps(text, (0, 10), (10, 11))
    assert overlaps(text, (0, 10), (9, 11))


def test_smoke_subset_is_the_first_documents(dataset_zip: Path) -> None:
    ids = [str(d["id"]) for d in load_raw(dataset_zip, "dev")["documents"][:4]]
    data = build_split(dataset_zip, "dev", ids)
    assert list(data.documents) == ids
    assert Counter(p.doc_id for p in data.pairs) == dict.fromkeys(ids, 17)
    with pytest.raises(ValueError, match="unknown document"):
        build_split(dataset_zip, "dev", ["no-such-id"])
