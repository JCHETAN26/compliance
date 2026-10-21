"""Layout-aware, sentence-preserving chunking with overlap.

Units are sentences inside blocks (headings are single units). Units are appended to
the current chunk until one of these boundaries is reached:

1. adding the next unit would exceed ``max_tokens``;
2. the next unit starts a section (heading or numbered clause) and the chunk already
   has ``min_tokens``;
3. the next unit starts a new paragraph and the chunk has reached the midpoint
   ``(min_tokens + max_tokens) / 2``.

A chunk never ends on a heading: trailing headings move to the next chunk. After a
size or paragraph break, the next chunk repeats the previous chunk's last sentences (up
to ``overlap_tokens``, same section only). After a section break there is no overlap.
A final chunk smaller than ``min_tokens`` is merged into its predecessor when the
result fits.

Chunks are contiguous source ranges, so ``text[char_start:char_end]`` is the chunk text
and citations stay exact. **Sentences are never split.** The 200-400 token range is a
target, not a hard cap: a single sentence longer than ``max_tokens`` becomes one
oversized chunk containing exactly that whole sentence (``token_count`` reports its
true size). Fitting oversized chunks into the embedding model's context is the
embedder's job (see ``embeddings.Embedder``: bounded token windows aggregated into one
vector for the unchanged source chunk). Nothing is truncated or cut here.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from compliance_ingestion.layout import merge_char_boxes
from compliance_ingestion.sentences import sentence_spans
from compliance_ingestion.textbuilder import CharBox
from compliance_ingestion.tokens import TokenCounter
from compliance_ingestion.types import BlockInfo, ChunkInfo, PageInfo


@dataclass(frozen=True)
class ChunkOptions:
    min_tokens: int = 200
    max_tokens: int = 400
    overlap_tokens: int = 40


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    tokens: int
    block_index: int
    section_path: str | None
    starts_block: bool
    starts_section: bool
    is_heading: bool


def _units(text: str, blocks: list[BlockInfo], counter: TokenCounter) -> list[_Unit]:
    spans: list[tuple[int, int, BlockInfo, bool]] = []
    for block in blocks:
        if block.kind == "heading":
            pieces = [(block.char_start, block.char_end)]
        else:
            pieces = sentence_spans(text, block.char_start, block.char_end)
        for i, (start, end) in enumerate(pieces):
            spans.append((start, end, block, i == 0))
    counts = counter.count_many([text[s:e] for s, e, _, _ in spans])
    return [
        _Unit(
            start=s,
            end=e,
            tokens=n,
            block_index=b.block_index,
            section_path=b.section_path,
            starts_block=first,
            starts_section=first and b.starts_section,
            is_heading=b.kind == "heading",
        )
        for (s, e, b, first), n in zip(spans, counts, strict=True)
    ]


def _plan(units: list[_Unit], options: ChunkOptions) -> list[tuple[list[_Unit], int]]:
    """Group units into chunks; returns (units, index of first non-overlap unit)."""
    target = (options.min_tokens + options.max_tokens) // 2
    chunks: list[tuple[list[_Unit], int]] = []
    current: list[_Unit] = []
    fresh_from = 0

    def tokens(items: list[_Unit]) -> int:
        return sum(u.tokens for u in items)

    def flush() -> list[_Unit]:
        nonlocal current
        carried: list[_Unit] = []
        while current and current[-1].is_heading:
            carried.insert(0, current.pop())
        if current:
            chunks.append((current, min(fresh_from, len(current) - 1)))
        current = []
        return carried

    for unit in units:
        if current:
            size = tokens(current)
            reason: str | None = None
            if size + unit.tokens > options.max_tokens:
                reason = "size"
            elif unit.starts_section and size >= options.min_tokens:
                reason = "section"
            elif unit.starts_block and size >= target:
                reason = "paragraph"
            if reason:
                previous = current
                carried = flush()
                overlap: list[_Unit] = []
                if reason != "section" and not carried and not unit.starts_section:
                    budget = options.overlap_tokens
                    for prior in reversed(previous):
                        if prior.is_heading or prior.section_path != unit.section_path:
                            break
                        if prior.tokens > budget:
                            break
                        overlap.insert(0, prior)
                        budget -= prior.tokens
                    if tokens(overlap) + unit.tokens > options.max_tokens:
                        overlap = []
                current = carried + overlap
                fresh_from = len(overlap) if not carried else 0
        current.append(unit)
    if current:
        trailing = flush()
        if trailing:
            # Headings at the very end (or a document that is only headings) still become
            # searchable text: append them to the last chunk when it fits, else their own.
            if chunks and tokens(chunks[-1][0]) + tokens(trailing) <= options.max_tokens:
                last, fresh = chunks[-1]
                chunks[-1] = (last + trailing, fresh)
            else:
                chunks.append((trailing, 0))

    # Merge a too-small final chunk into the previous one when the union fits.
    if len(chunks) >= 2:
        last, last_fresh = chunks[-1]
        prev, prev_fresh = chunks[-2]
        last_new = last[last_fresh:]
        if (
            last_new
            and tokens(last_new) < options.min_tokens
            and tokens(prev) + tokens(last_new) <= options.max_tokens
            and last_new[0].section_path == prev[-1].section_path
        ):
            chunks[-2:] = [(prev + last_new, prev_fresh)]
    return chunks


def build_chunks(
    text: str,
    blocks: list[BlockInfo],
    counter: TokenCounter,
    options: ChunkOptions,
    *,
    char_boxes: list[CharBox] | None = None,
    pages: list[PageInfo] | None = None,
) -> list[ChunkInfo]:
    planned = _plan(_units(text, blocks, counter), options)
    page_starts = [p.char_start for p in pages] if pages else []
    texts = [text[group[0].start : group[-1].end] for group, _ in planned]
    exact_counts = counter.count_many(texts)
    chunks: list[ChunkInfo] = []
    for index, ((group, fresh), chunk_text, count) in enumerate(
        zip(planned, texts, exact_counts, strict=True)
    ):
        start, end = group[0].start, group[-1].end
        page_start = page_end = None
        if pages:
            page_start = pages[max(bisect.bisect_right(page_starts, start) - 1, 0)].page_number
            page_end = pages[
                max(bisect.bisect_right(page_starts, max(end - 1, start)) - 1, 0)
            ].page_number
        chunks.append(
            ChunkInfo(
                chunk_index=index,
                char_start=start,
                char_end=end,
                text=chunk_text,
                token_count=count,
                section_path=group[fresh].section_path,
                page_start=page_start,
                page_end=page_end,
                bboxes=merge_char_boxes(char_boxes, start, end) if char_boxes else [],
            )
        )
    return chunks
