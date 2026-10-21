"""Sentence segmentation and chunking invariants (offsets, no mid-sentence splits)."""

from __future__ import annotations

import itertools
import re

import pytest

from compliance_ingestion.chunking import ChunkOptions, build_chunks
from compliance_ingestion.layout import RawBlock, assign_sections
from compliance_ingestion.sentences import sentence_spans


class WordCounter:
    """Deterministic token counter for unit tests: one token per whitespace word."""

    def count_many(self, texts: list[str]) -> list[int]:
        return [len(t.split()) for t in texts]


def texts(text: str) -> list[str]:
    return [text[s:e] for s, e in sentence_spans(text)]


def test_sentence_boundaries_and_offsets() -> None:
    text = "  First sentence. Second one!  Third?\\nFourth without end"
    text = text.replace("\\n", "\n")
    spans = sentence_spans(text)
    assert [text[s:e] for s, e in spans] == [
        "First sentence.",
        "Second one!",
        "Third?",
        "Fourth without end",
    ]
    assert spans[0][0] == 2


@pytest.mark.parametrize(
    "text",
    [
        "Acme Inc. shall comply. The other party agrees.",
        "See e.g. the schedule. Then sign.",
        "Signed by J. Smith on behalf of the company. Done.",
        "Approx. five days are allowed. Next.",
        "Payment of U.S. dollars is required. Next.",
    ],
)
def test_abbreviations_and_initials_do_not_split(text: str) -> None:
    assert len(texts(text)) == 2


@pytest.mark.parametrize(
    "text",
    [
        "1. Definitions. Terms have meanings.",
        "3.2. The Recipient shall not disclose. Next sentence.",
        "Section 4. Term of this agreement. It ends.",
        "(a) the first item applies. Then more.",
    ],
)
def test_clause_enumerators_stay_with_their_sentence(text: str) -> None:
    first = texts(text)[0]
    assert not re.fullmatch(r"[\w().\s]{0,12}", first), first


def test_quotes_after_period_are_part_of_sentence() -> None:
    assert texts('He said "stop." Then left.') == ['He said "stop."', "Then left."]


def _blocks(text: str, kinds: list[tuple[int, int, str]]):  # type: ignore[no-untyped-def]
    raw = [RawBlock(char_start=s, char_end=e, kind=k) for s, e, k in kinds]  # type: ignore[arg-type]
    return assign_sections(raw, text)


def _paragraph(n: int, words: int, tag: str) -> str:
    # Sentences start with a capital letter: lowercase starts are treated as continuations.
    return " ".join(f"{tag.capitalize()}{i} " + "word " * (words - 2) + "end." for i in range(n))


def _doc(
    paragraphs: list[str], headings: dict[int, str] | None = None
) -> tuple[str, list[tuple[int, int, str]]]:
    parts: list[tuple[str, str]] = []
    for i, para in enumerate(paragraphs):
        if headings and i in headings:
            parts.append((headings[i], "heading"))
        parts.append((para, "paragraph"))
    text = ""
    spans: list[tuple[int, int, str]] = []
    for content, kind in parts:
        if text:
            text += "\n\n"
        start = len(text)
        text += content
        spans.append((start, len(text), kind))
    return text, spans


def _assert_invariants(text: str, chunks: list, sentence_bounds: set[int]) -> None:  # type: ignore[type-arg]
    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.char_start in sentence_bounds, "chunk must start at a sentence start"
        assert chunk.char_end in sentence_bounds, "chunk must end at a sentence end"
    # Every sentence is covered by at least one chunk, and in order.
    starts = [c.char_start for c in chunks]
    assert starts == sorted(starts)


def _bounds(text: str, spans: list[tuple[int, int, str]]) -> set[int]:
    bounds: set[int] = set()
    for s, e, kind in spans:
        pieces = [(s, e)] if kind == "heading" else sentence_spans(text, s, e)
        for a, b in pieces:
            bounds.update((a, b))
    return bounds


def test_chunks_respect_target_size_offsets_and_sentences() -> None:
    paragraphs = [_paragraph(6, 20, f"p{i}s") for i in range(8)]  # 8 x 120 words
    text, spans = _doc(paragraphs)
    chunks = build_chunks(text, _blocks(text, spans), WordCounter(), ChunkOptions(200, 400, 40))
    assert len(chunks) >= 3
    for chunk in chunks[:-1]:
        assert 200 <= chunk.token_count <= 400, chunk.token_count
    assert all(c.token_count <= 400 for c in chunks)
    _assert_invariants(text, chunks, _bounds(text, spans))


def test_overlap_repeats_trailing_sentences_within_budget() -> None:
    paragraphs = [_paragraph(30, 20, "s")]  # one long paragraph, 600 words
    text, spans = _doc(paragraphs)
    chunks = build_chunks(text, _blocks(text, spans), WordCounter(), ChunkOptions(200, 400, 40))
    assert len(chunks) >= 2
    for prev, nxt in itertools.pairwise(chunks):
        overlap = prev.char_end - nxt.char_start
        assert overlap > 0, "consecutive chunks in one section overlap"
        assert len(text[nxt.char_start : prev.char_end].split()) <= 40


def test_oversized_sentence_is_kept_whole_and_never_split() -> None:
    giant = "This clause " + "continues and " * 300 + "finally ends."  # ~600 words, 1 sentence
    text, spans = _doc([_paragraph(3, 20, "a"), giant, _paragraph(3, 20, "b")])
    chunks = build_chunks(text, _blocks(text, spans), WordCounter(), ChunkOptions(200, 400, 40))
    containing = [c for c in chunks if "finally ends." in c.text]
    assert len(containing) == 1
    big = containing[0]
    assert giant in big.text, "the oversized sentence is preserved intact"
    assert big.token_count == len(big.text.split()) > 400
    assert sum("This clause continues" in c.text for c in chunks) == 1
    _assert_invariants(text, chunks, _bounds(text, spans))


def test_section_breaks_and_headings() -> None:
    paragraphs = [_paragraph(12, 20, f"x{i}") for i in range(3)]  # 240 words each
    text, spans = _doc(
        paragraphs, headings={0: "1. Definitions", 1: "2. Obligations", 2: "3. Term"}
    )
    blocks = _blocks(text, spans)
    chunks = build_chunks(text, blocks, WordCounter(), ChunkOptions(200, 400, 40))
    for chunk in chunks:
        stripped = chunk.text.rstrip()
        assert not stripped.endswith(("Definitions", "Obligations", "Term")), "no trailing heading"
    starts = [c for c in chunks if c.text.startswith("2. Obligations")]
    assert starts, "a new numbered section starts a fresh chunk"
    assert starts[0].section_path == "2. Obligations"
    previous = chunks[chunks.index(starts[0]) - 1]
    assert previous.char_end <= starts[0].char_start, "no overlap across a section break"


def test_small_trailing_chunk_merges_into_previous() -> None:
    text, spans = _doc([_paragraph(12, 20, "m"), "Tiny closing sentence."])
    chunks = build_chunks(text, _blocks(text, spans), WordCounter(), ChunkOptions(200, 400, 40))
    assert chunks[-1].text.endswith("Tiny closing sentence.")
    assert chunks[-1].token_count >= 200


def test_section_paths_follow_numbering_depth() -> None:
    text, spans = _doc(
        ["Intro words here.", "Body text.", "More body.", "Other."],
        headings={
            0: "ARTICLE I DEFINITIONS",
            1: "1.1 Confidential Information",
            2: "1.2 Recipient",
            3: "2 Term",
        },
    )
    blocks = _blocks(text, spans)
    paths = [b.section_path for b in blocks if b.kind == "paragraph"]
    assert paths == [
        "ARTICLE I DEFINITIONS",
        "ARTICLE I DEFINITIONS > 1.1 Confidential Information",
        "ARTICLE I DEFINITIONS > 1.2 Recipient",
        "2 Term",
    ]


def test_document_of_only_headings_still_produces_a_chunk() -> None:
    text = "MUTUAL NON-DISCLOSURE AGREEMENT"
    blocks = _blocks(text, [(0, len(text), "heading")])
    chunks = build_chunks(text, blocks, WordCounter(), ChunkOptions(200, 400, 40))
    assert [c.text for c in chunks] == [text]


def test_trailing_heading_is_attached_to_last_chunk() -> None:
    text, spans = _doc(["Body sentence one. Body sentence two."])
    text += "\n\nSCHEDULE A"
    spans.append((len(text) - len("SCHEDULE A"), len(text), "heading"))
    chunks = build_chunks(text, _blocks(text, spans), WordCounter(), ChunkOptions(200, 400, 40))
    assert len(chunks) == 1
    assert chunks[0].text.endswith("SCHEDULE A")
    assert text[chunks[0].char_start : chunks[0].char_end] == chunks[0].text
