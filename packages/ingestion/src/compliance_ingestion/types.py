"""Typed results of parsing + chunking one document.

All character offsets are **document-level**: indexes into ``ExtractionResult.text``,
the canonical text the API stores and that citations will later be verified against
(``text[chunk.char_start:chunk.char_end] == chunk.text``). Coordinates are PDF points
in the page's displayed (rotation-applied) orientation, origin at the top-left.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

BlockKind = Literal["heading", "paragraph", "list_item", "table_row", "sheet_row"]
DocType = Literal["contract", "spreadsheet", "policy", "other"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BBox(_Frozen):
    page: int = Field(ge=1)
    x0: float
    y0: float
    x1: float
    y1: float


class PageInfo(_Frozen):
    page_number: int = Field(ge=1)
    width: float
    height: float
    is_ocr: bool
    ocr_confidence: float | None = None
    char_start: int
    char_end: int


class BlockInfo(_Frozen):
    block_index: int
    kind: BlockKind
    style: str | None = None
    level: int | None = None
    starts_section: bool = False
    char_start: int
    char_end: int
    section_path: str | None = None
    page_number: int | None = None
    bbox: BBox | None = None


class SheetInfo(_Frozen):
    sheet_index: int
    sheet_name: str
    columns: list[str]
    header_row_number: int | None
    row_count: int
    truncated: bool = False


class CellValue(_Frozen):
    column: str
    value: Any
    display: str


class RowInfo(_Frozen):
    sheet_index: int
    row_number: int = Field(ge=1, description="1-based row number in the source sheet")
    cells: list[CellValue]
    char_start: int
    char_end: int


class ChunkInfo(_Frozen):
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    token_count: int
    section_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    bboxes: list[BBox] = Field(default_factory=list)
    sheet_index: int | None = None
    row_number: int | None = None


class ExtractionStats(_Frozen):
    pages: int = 0
    ocr_pages: int = 0
    blocks: int = 0
    chunks: int = 0
    rows: int = 0
    chars: int = 0


class ExtractionResult(_Frozen):
    format: Literal["pdf", "docx", "xlsx", "csv"]
    doc_type: DocType
    text: str
    pages: list[PageInfo] = Field(default_factory=list)
    blocks: list[BlockInfo] = Field(default_factory=list)
    sheets: list[SheetInfo] = Field(default_factory=list)
    rows: list[RowInfo] = Field(default_factory=list)
    chunks: list[ChunkInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    stats: ExtractionStats


class ExtractionError(Exception):
    """A document that cannot be ingested; the message is safe to show to users."""
