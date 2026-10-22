"""Evidence-retrieval metric definitions and the ContractNLI segmentation contract."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from compliance_ingestion.chunking import ChunkOptions
from compliance_ingestion.tokens import HFTokenCounter
from evals.retrieval import metrics
from evals.retrieval.data import build_split, chunk_document, line_blocks


def test_coverage_threshold_is_half_the_span() -> None:
    span = (100, 200)
    assert metrics.covers((0, 1000), span)
    assert metrics.covers((150, 400), span)  # exactly 50 %
    assert not metrics.covers((151, 400), span)  # 49 %
    assert not metrics.covers((0, 100), span)  # touching, no overlap
    assert not metrics.covers((0, 10), (5, 5))  # empty span never counts


def test_recall_hit_and_reciprocal_rank() -> None:
    gold = [(10, 20), (500, 520)]
    ranked = [(300, 400), (0, 50), (450, 480), (490, 600)]
    score = metrics.score_ranking(ranked, gold)
    assert score.recall == {1: 0.0, 3: 0.5, 5: 1.0, 10: 1.0}
    assert score.hit == {1: 0.0, 3: 1.0, 5: 1.0, 10: 1.0}
    assert score.rr == 0.5
    miss = metrics.score_ranking([(0, 5)] * 12, gold)
    assert miss.rr == 0.0
    late = metrics.score_ranking([(900, 901)] * 10 + [(0, 50)], gold)
    assert late.rr == 0.0  # first hit at rank 11 is beyond MRR@10
    with pytest.raises(ValueError, match="excluded"):
        metrics.score_ranking(ranked, [])


def test_random_baseline_matches_exhaustive_enumeration() -> None:
    chunks = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 600)]
    gold = [(10, 30), (150, 160), (420, 480)]
    expected = metrics.random_baseline(chunks, gold)
    perms = list(itertools.permutations(chunks))
    scores = [metrics.score_ranking(list(p), gold) for p in perms]
    for k in metrics.KS:
        assert expected.recall[k] == pytest.approx(sum(s.recall[k] for s in scores) / len(perms))
        assert expected.hit[k] == pytest.approx(sum(s.hit[k] for s in scores) / len(perms))
    assert expected.rr == pytest.approx(sum(s.rr for s in scores) / len(perms))


def test_aggregate_and_selection_key() -> None:
    a = metrics.score_ranking([(0, 50)], [(10, 20)])
    b = metrics.score_ranking([(60, 70), (0, 50)], [(10, 20)])
    agg = metrics.aggregate([a, b])
    assert agg["recall@1"] == 0.5
    assert agg["recall@3"] == 1.0
    assert agg["mrr@10"] == 0.75
    assert metrics.selection_key(agg) == (1.0, 0.75)


def test_line_blocks_group_wrapped_lines_and_keep_offsets() -> None:
    text = (
        "  NON-DISCLOSURE AGREEMENT\n"
        "1. The Recipient shall keep the information\n"
        "secret and shall not disclose it.  \n"
        "   \n"
        "2. Definitions:\n"
        "The Recipient shall comply.\n"
        "Signed by"
    )
    blocks = line_blocks(text)
    assert [(b.kind, text[b.char_start : b.char_end]) for b in blocks] == [
        ("heading", "NON-DISCLOSURE AGREEMENT"),
        (
            "paragraph",
            "1. The Recipient shall keep the information\nsecret and shall not disclose it.",
        ),
        ("paragraph", "2. Definitions:"),
        ("paragraph", "The Recipient shall comply."),
        ("paragraph", "Signed by"),
    ]


def test_wrapped_long_sentence_is_never_split(model_path: Path) -> None:
    """A single sentence wrapped over many lines (no blank lines, one final period) and
    longer than the chunk target stays inside exactly one chunk."""
    words = [f"obligation{i}" for i in range(260)]
    lines = [" ".join(words[i : i + 9]) for i in range(0, len(words), 9)]
    sentence = "\n".join(lines) + "."
    text = "CONFIDENTIALITY\nThe parties agree as follows.\n" + sentence + "\nEnd of terms."
    counter = HFTokenCounter(model_path / "tokenizer.json")
    assert counter.count_many([sentence])[0] > ChunkOptions().max_tokens
    chunks = chunk_document(text, counter, ChunkOptions())
    start = text.index(sentence)
    end = start + len(sentence)
    for chunk in chunks:
        inside = chunk.char_start < end and chunk.char_end > start
        if inside:
            assert chunk.char_start <= start, (chunk, start, end)
            assert chunk.char_end >= end, (chunk, start, end)
    assert sum(1 for c in chunks if c.char_start <= start and c.char_end >= end) >= 1


def test_contractnli_dev_segmentation_preserves_gold(dataset_zip: Path, model_path: Path) -> None:
    data = build_split(
        dataset_zip, "dev", HFTokenCounter(model_path / "tokenizer.json"), ChunkOptions()
    )
    assert len(data.documents) == 61
    assert len(data.pairs) == 614  # Entailment + Contradiction pairs with evidence
    assert data.excluded_no_evidence == 61 * 17 - 614 - data.excluded_empty_evidence
    for pair in data.pairs:
        doc = data.documents[pair.doc_id]
        for start, end in pair.gold:
            assert 0 <= start < end <= len(doc.text)  # offsets index the unchanged text
        for chunk in doc.chunks:
            assert doc.text[chunk.char_start : chunk.char_end] == chunk.text
