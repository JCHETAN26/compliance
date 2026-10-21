"""API/service schemas for documents and their ingestion outputs."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from compliance_core.db.models import DocType, DocumentStatus


class _Read(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class DocumentRead(_Read):
    id: uuid.UUID
    workspace_id: uuid.UUID
    filename: str
    mime_type: str
    file_format: Literal["pdf", "docx", "xlsx", "csv"]
    size_bytes: int
    sha256: str
    status: DocumentStatus
    stage: str | None = Field(description="Current ingestion stage while processing")
    progress: int = Field(ge=0, le=100)
    doc_type: DocType | None
    page_count: int | None
    ocr_page_count: int
    chunk_count: int
    error: str | None
    warnings: list[str]
    attempts: int
    embedding_model: str | None
    created_at: datetime
    updated_at: datetime
    ingested_at: datetime | None


UploadOutcome = Literal["created", "duplicate", "rejected"]


class UploadErrorBody(BaseModel):
    code: str
    message: str


class UploadResult(BaseModel):
    filename: str
    outcome: UploadOutcome
    document: DocumentRead | None = None
    error: UploadErrorBody | None = None


class UploadResponse(BaseModel):
    results: list[UploadResult]


class BBoxRead(BaseModel):
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


class ChunkRead(_Read):
    id: uuid.UUID
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    bboxes: list[BBoxRead]
    section_path: str | None
    token_count: int
    embedding_windows: int
    table_row_id: uuid.UUID | None


class ChunkPage(BaseModel):
    total: int
    items: list[ChunkRead]


class PageRead(_Read):
    page_number: int
    width: float
    height: float
    is_ocr: bool
    ocr_confidence: float | None
    char_start: int
    char_end: int
    text: str | None = None


class BlockRead(_Read):
    block_index: int
    kind: str
    style: str | None
    level: int | None
    starts_section: bool
    page_number: int | None
    char_start: int
    char_end: int
    section_path: str | None
    text: str


class SheetRead(_Read):
    sheet_index: int
    sheet_name: str
    columns: list[str]
    header_row_number: int | None
    row_count: int
    truncated: bool


class CellRead(BaseModel):
    column: str
    value: Any
    display: str


class RowRead(BaseModel):
    id: uuid.UUID
    sheet_index: int
    sheet_name: str
    row_number: int
    cells: list[CellRead]
    char_start: int
    char_end: int


class RowPage(BaseModel):
    total: int
    items: list[RowRead]


class EntityRead(_Read):
    id: uuid.UUID
    type: str
    value_json: dict[str, Any]
    chunk_id: uuid.UUID | None
    confidence: float | None


class EntitiesResponse(BaseModel):
    status: Literal["not_available", "available"]
    detail: str
    entities: list[EntityRead]
