"""Read-only views of an ingested document: canonical text, chunks, sentence spans and
highlight geometry. Every quote the agents store is a slice of the canonical text at
exact offsets, so the citation verifier (and every later read) can re-check it."""

from __future__ import annotations

import uuid
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_core.db.models import (
    Chunk,
    Document,
    DocumentBlock,
    DocumentContent,
    DocumentStatus,
)
from compliance_ingestion.sentences import sentence_spans

MIN_SPAN_CHARS = 3


@dataclass(frozen=True)
class ChunkView:
    id: uuid.UUID
    index: int
    start: int
    end: int
    page_start: int | None
    bboxes: list[dict[str, Any]]
    table_row_id: uuid.UUID | None


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    @property
    def model_id(self) -> str:
        return f"{self.start}-{self.end}"


@dataclass
class DocView:
    id: uuid.UUID
    workspace_id: uuid.UUID
    filename: str
    file_format: str
    doc_type: str | None
    text: str
    text_sha256: str
    chunks: list[ChunkView]
    blocks: list[tuple[int, int]]
    line_boxes: list[list[float]] | None

    def slice(self, start: int, end: int) -> str:
        return self.text[start:end]

    def sentences(self) -> list[Span]:
        """Sentence spans per layout block (the ingestion/model segmentation)."""
        ranges = self.blocks or [(c.start, c.end) for c in self.chunks]
        out: list[Span] = []
        seen: set[tuple[int, int]] = set()
        for start, end in ranges:
            for s, e in sentence_spans(self.text, start, end):
                if e - s >= MIN_SPAN_CHARS and (s, e) not in seen:
                    seen.add((s, e))
                    out.append(Span(s, e))
        out.sort(key=lambda sp: sp.start)
        return out

    def sentences_in(self, start: int, end: int) -> list[Span]:
        return [s for s in self.sentences() if s.start >= start and s.end <= end] or [
            Span(start, end)
        ]

    def chunk_for(self, start: int, end: int) -> ChunkView | None:
        best: tuple[int, ChunkView] | None = None
        for chunk in self.chunks:
            overlap = min(end, chunk.end) - max(start, chunk.start)
            if chunk.start <= start and end <= chunk.end:
                return chunk
            if overlap > 0 and (best is None or overlap > best[0]):
                best = (overlap, chunk)
        return best[1] if best else None

    def geometry(self, start: int, end: int) -> tuple[list[dict[str, Any]], str, int | None]:
        """Highlight boxes for ``[start, end)``: exact visual lines when the document has
        line geometry, else the containing chunk's boxes, else none."""
        if self.line_boxes:
            starts = [int(line[1]) for line in self.line_boxes]
            first = max(bisect_right(starts, start) - 1, 0)
            boxes = []
            for line in self.line_boxes[first:]:
                page, s, e, x0, y0, x1, y1 = line
                if s >= end:
                    break
                if e > start:
                    boxes.append({"page": int(page), "x0": x0, "y0": y0, "x1": x1, "y1": y1})
            if boxes:
                return boxes, "line", int(boxes[0]["page"])
        chunk = self.chunk_for(start, end)
        if chunk and chunk.bboxes:
            return list(chunk.bboxes), "chunk", int(chunk.bboxes[0]["page"])
        return [], "none", chunk.page_start if chunk else None


async def load_doc(
    session: AsyncSession, document_id: uuid.UUID, workspace_id: uuid.UUID
) -> DocView | None:
    """The document if it still exists, is ready, and belongs to ``workspace_id``."""
    row = (
        await session.execute(
            select(Document, DocumentContent)
            .join(DocumentContent, DocumentContent.document_id == Document.id)
            .where(Document.id == document_id, Document.workspace_id == workspace_id)
        )
    ).one_or_none()
    if row is None:
        return None
    document, content = row
    if document.status is not DocumentStatus.READY:
        return None
    chunks = [
        ChunkView(
            id=c.id,
            index=c.chunk_index,
            start=c.char_start,
            end=c.char_end,
            page_start=c.page_start,
            bboxes=list(c.bbox_json or []),
            table_row_id=c.table_row_id,
        )
        for c in await session.scalars(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
        )
    ]
    blocks = [
        (b.char_start, b.char_end)
        for b in await session.scalars(
            select(DocumentBlock)
            .where(DocumentBlock.document_id == document_id)
            .order_by(DocumentBlock.block_index)
        )
    ]
    return DocView(
        id=document.id,
        workspace_id=document.workspace_id,
        filename=document.filename,
        file_format=document.file_format,
        doc_type=document.doc_type.value if document.doc_type else None,
        text=content.text,
        text_sha256=content.text_sha256,
        chunks=chunks,
        blocks=blocks,
        line_boxes=content.line_boxes,
    )
