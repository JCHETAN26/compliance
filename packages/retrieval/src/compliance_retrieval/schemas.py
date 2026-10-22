"""Search request/response schemas (shared by the REST API and, in Phase 5, MCP tools)."""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from compliance_core.db.models import DocType
from compliance_core.document_schemas import BBoxRead

SearchMode = Literal["hybrid", "vector", "keyword"]
QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: QueryText
    mode: SearchMode = "hybrid"
    top_k: Annotated[int, Field(ge=1, le=50)] = 8
    document_ids: Annotated[list[uuid.UUID], Field(max_length=100)] = Field(default_factory=list)
    doc_types: Annotated[list[DocType], Field(max_length=4)] = Field(default_factory=list)
    rerank: bool = False


class MethodScore(BaseModel):
    rank: int = Field(description="1-based rank within that method's candidate list")
    score: float = Field(description="Method-native score: cosine similarity or ts_rank_cd")


class Provenance(BaseModel):
    vector: MethodScore | None = None
    keyword: MethodScore | None = None
    fused_score: float | None = Field(
        default=None, description="Weighted RRF score (hybrid mode only)"
    )
    rerank_score: float | None = Field(default=None, description="Cross-encoder logit")


class SearchHit(BaseModel):
    rank: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    file_format: str
    doc_type: str | None
    chunk_index: int
    text: str = Field(description="The stored chunk text (canonical text slice)")
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    bboxes: list[BBoxRead]
    section_path: str | None
    token_count: int
    table_row_id: uuid.UUID | None
    sheet_name: str | None
    row_number: int | None
    provenance: Provenance


class SearchSettings(BaseModel):
    top_k: int
    candidate_k: int
    ef_search: int
    rrf_k: int
    vector_weight: float
    keyword_weight: float
    keyword_mode: str
    rerank_candidates: int | None
    embedding_model: str | None
    embedding_revision: str | None
    query_instruction: str | None
    rerank_model: str | None


class SearchTimings(BaseModel):
    embed_ms: float | None = None
    vector_ms: float | None = None
    keyword_ms: float | None = None
    rerank_ms: float | None = None
    total_ms: float


class StaleDocument(BaseModel):
    document_id: uuid.UUID
    filename: str
    embedding_model: str | None = Field(description="Encoder identity stored at ingestion")


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    reranked: bool
    results: list[SearchHit]
    candidates: dict[str, int] = Field(description="Candidate counts per method")
    excluded_from_vector: list[StaleDocument] = Field(
        description=(
            "Ready documents embedded by a different encoder than the query encoder. They are "
            "excluded from vector ranking (still keyword-searchable); re-ingest to include them."
        )
    )
    settings: SearchSettings
    timings: SearchTimings


class SearchCapabilities(BaseModel):
    modes: list[SearchMode]
    rerank_available: bool
    embedding_model: str
    embedding_revision: str
    query_instruction: str
    rerank_model: str | None
    default_top_k: int
    max_top_k: int
