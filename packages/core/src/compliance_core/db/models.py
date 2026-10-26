"""SQLAlchemy ORM models.

Phase 0: identity and tenancy (users, workspaces, memberships).
Phase 1: documents and everything ingestion derives from them (canonical text, pages,
layout blocks, chunks with embeddings and full-text vectors, spreadsheet sheets/rows)
plus the ``extracted_entities`` table the viewer reads (populated from Phase 4).
Phase 4: rule packs and rules, per-workspace review settings, review runs with their
tasks, findings, citations, human overrides, agent events, the LLM budget ledger, step
memoization, report snapshots and the LangGraph checkpoint tables.
Phase 5: per-user API tokens for the MCP server (only SHA-256 hashes are stored).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

WORKSPACE_NAME_MAX = 120
EMAIL_MAX = 320
FILENAME_MAX = 255
EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class WorkspaceRole(StrEnum):
    OWNER = "owner"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


workspace_role_enum = Enum(
    WorkspaceRole,
    name="workspace_role",
    values_callable=lambda enum: [member.value for member in enum],
)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("email = lower(email)", name="email_lowercase"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()")
    )
    email: Mapped[str] = mapped_column(String(EMAIL_MAX), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


API_TOKEN_SCOPES = ("read", "review", "upload")
API_TOKEN_NAME_MAX = 100


class ApiToken(Base):
    """A per-user MCP API token. The plaintext is shown once at creation; only its
    SHA-256 hash is stored. ``workspace_ids`` NULL means every workspace the user is a
    member of (membership is re-checked on every use)."""

    __tablename__ = "api_tokens"
    __table_args__ = (
        CheckConstraint(
            f"char_length(btrim(name)) between 1 and {API_TOKEN_NAME_MAX}", name="name_length"
        ),
        CheckConstraint(
            "scopes <@ ARRAY['read', 'review', 'upload']::varchar[] AND 'read' = ANY(scopes)",
            name="scopes_valid",
        ),
        CheckConstraint(
            "workspace_ids IS NULL OR cardinality(workspace_ids) > 0", name="allowlist_non_empty"
        ),
        CheckConstraint("expires_at > created_at", name="expires_after_creation"),
        Index("ix_api_tokens_user_created", "user_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()")
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(API_TOKEN_NAME_MAX), nullable=False)
    token_prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    scopes: Mapped[list[str]] = mapped_column(ARRAY(String(16)), nullable=False)
    workspace_ids: Mapped[list[uuid.UUID] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Workspace(Base):
    __tablename__ = "workspaces"
    __table_args__ = (
        CheckConstraint(
            f"char_length(btrim(name)) between 1 and {WORKSPACE_NAME_MAX}", name="name_length"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()")
    )
    name: Mapped[str] = mapped_column(String(WORKSPACE_NAME_MAX), nullable=False)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    members: Mapped[list[WorkspaceMember]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan", passive_deletes=True
    )


class WorkspaceMember(Base):
    __tablename__ = "workspace_members"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    role: Mapped[WorkspaceRole] = mapped_column(workspace_role_enum, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    workspace: Mapped[Workspace] = relationship(back_populates="members")
    user: Mapped[User] = relationship()


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocType(StrEnum):
    CONTRACT = "contract"
    SPREADSHEET = "spreadsheet"
    POLICY = "policy"
    OTHER = "other"


def _enum(enum_cls: type[StrEnum], name: str) -> Enum:
    return Enum(enum_cls, name=name, values_callable=lambda e: [m.value for m in e])


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()")
    )


def _doc_fk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("workspace_id", "sha256", name="uq_documents_workspace_sha256"),
        CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name="sha256_hex"),
        CheckConstraint("progress between 0 and 100", name="progress_range"),
        Index("ix_documents_workspace_created", "workspace_id", "created_at"),
        Index("ix_documents_status_updated", "status", "updated_at"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    filename: Mapped[str] = mapped_column(String(FILENAME_MAX), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(127), nullable=False)
    file_format: Mapped[str] = mapped_column(String(8), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(
        _enum(DocumentStatus, "document_status"), nullable=False
    )
    stage: Mapped[str | None] = mapped_column(String(32))
    progress: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default="0")
    doc_type: Mapped[DocType | None] = mapped_column(_enum(DocType, "doc_type"))
    page_count: Mapped[int | None] = mapped_column(Integer)
    ocr_page_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    error: Mapped[str | None] = mapped_column(Text)
    warnings: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, server_default=sql_text("'[]'::jsonb")
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    ingest_generation: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    embedding_model: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DocumentContent(Base):
    """Canonical extracted text; every stored offset indexes into ``text``."""

    __tablename__ = "document_contents"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    # PDF only: one ``[page, char_start, char_end, x0, y0, x1, y1]`` entry per visual line
    # run, so a citation's exact character range can be highlighted line by line. NULL for
    # other formats and for documents ingested before this column existed.
    line_boxes: Mapped[list[list[float]] | None] = mapped_column(JSONB)


class Page(Base):
    __tablename__ = "pages"
    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_pages_document_page"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = _doc_fk()
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    is_ocr: Mapped[bool] = mapped_column(Boolean, nullable=False)
    ocr_confidence: Mapped[float | None] = mapped_column(Float)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)


class DocumentBlock(Base):
    __tablename__ = "document_blocks"
    __table_args__ = (
        UniqueConstraint("document_id", "block_index", name="uq_document_blocks_document_index"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = _doc_fk()
    block_index: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    style: Mapped[str | None] = mapped_column(String(120))
    level: Mapped[int | None] = mapped_column(SmallInteger)
    starts_section: Mapped[bool] = mapped_column(Boolean, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    section_path: Mapped[str | None] = mapped_column(Text)
    bbox_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class TableSheet(Base):
    __tablename__ = "table_sheets"
    __table_args__ = (
        UniqueConstraint("document_id", "sheet_index", name="uq_table_sheets_document_index"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = _doc_fk()
    sheet_index: Mapped[int] = mapped_column(Integer, nullable=False)
    sheet_name: Mapped[str] = mapped_column(String(255), nullable=False)
    columns: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    header_row_number: Mapped[int | None] = mapped_column(Integer)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    truncated: Mapped[bool] = mapped_column(Boolean, nullable=False)


class TableRow(Base):
    __tablename__ = "table_rows"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "sheet_name", "row_index", name="uq_table_rows_document_row"
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = _doc_fk()
    sheet_name: Mapped[str] = mapped_column(String(255), nullable=False)
    sheet_index: Mapped[int] = mapped_column(Integer, nullable=False)
    # 1-based row number in the source sheet (header rows are not stored as data).
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    data_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
        CheckConstraint("char_end > char_start", name="non_empty_range"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index("ix_chunks_page_id", "page_id"),
        Index("ix_chunks_table_row_id", "table_row_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = _doc_fk()
    page_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pages.id", ondelete="SET NULL")
    )
    table_row_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("table_rows.id", ondelete="SET NULL")
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    bbox_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=sql_text("'[]'::jsonb")
    )
    section_path: Mapped[str | None] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    embedding_windows: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english'::regconfig, text)", persisted=True)
    )


class ExtractedEntity(Base):
    """Structured facts written by the extraction agent during a review run.

    Every row cites the chunk it came from with an exact ``quote`` at
    ``[char_start, char_end)`` of the document's canonical text, verified on write.
    Register links (spreadsheet row -> contract document) are stored on the spreadsheet
    document with ``type = "register_link"`` and cite the row's chunk.
    """

    __tablename__ = "extracted_entities"
    __table_args__ = (
        Index("ix_extracted_entities_document_id", "document_id"),
        Index("ix_extracted_entities_chunk_id", "chunk_id"),
        Index("ix_extracted_entities_review_run_id", "review_run_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = _doc_fk()
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    value_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="SET NULL")
    )
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    review_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("review_runs.id", ondelete="CASCADE")
    )
    quote: Mapped[str | None] = mapped_column(Text)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    page_number: Mapped[int | None] = mapped_column(Integer)
    method: Mapped[str | None] = mapped_column(String(24))
    prompt_version: Mapped[str | None] = mapped_column(String(120))
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")


# --------------------------------------------------------------------------- Phase 4

FINDING_STATUSES = ("satisfied", "violated", "not_mentioned", "needs_review")
RUN_STATUSES = (
    "queued",
    "running",
    "paused",
    "awaiting_review",
    "completed",
    "failed",
    "cancelled",
)


def _in(column: str, values: tuple[str, ...]) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{column} in ({quoted})"


def _created() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


def _run_fk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), ForeignKey("review_runs.id", ondelete="CASCADE"), nullable=False
    )


class RulePack(Base):
    """A versioned rule pack. ``workspace_id`` NULL means a built-in (global) pack."""

    __tablename__ = "rule_packs"
    __table_args__ = (
        CheckConstraint("key ~ '^[a-z0-9][a-z0-9_]{0,63}$'", name="key_format"),
        CheckConstraint(_in("source", ("builtin", "import", "api", "copy")), name="source"),
        Index(
            "uq_rule_packs_builtin_key_version",
            "key",
            "version",
            unique=True,
            postgresql_where=sql_text("workspace_id IS NULL"),
        ),
        Index(
            "uq_rule_packs_workspace_key_version",
            "workspace_id",
            "key",
            "version",
            unique=True,
            postgresql_where=sql_text("workspace_id IS NOT NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE")
    )
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    # Register column aliases for cross-document rules, and free-form provenance.
    register_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    provenance_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _created()

    rules: Mapped[list[Rule]] = relationship(
        back_populates="rule_pack",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Rule.position",
    )


class Rule(Base):
    __tablename__ = "rules"
    __table_args__ = (
        UniqueConstraint("rule_pack_id", "key", name="uq_rules_pack_key"),
        CheckConstraint(_in("scope", ("per_document", "cross_document")), name="scope"),
        CheckConstraint(_in("severity", ("low", "medium", "high")), name="severity"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    rule_pack_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rule_packs.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    hypothesis_text: Mapped[str] = mapped_column(Text, nullable=False)
    scope: Mapped[str] = mapped_column(String(16), nullable=False)
    applies_to_doc_types: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    severity: Mapped[str] = mapped_column(String(8), nullable=False)
    model_supported: Mapped[bool] = mapped_column(Boolean, nullable=False)
    contractnli_key: Mapped[str | None] = mapped_column(String(16))
    compliant_statuses: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    cross_document: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    remediation: Mapped[str | None] = mapped_column(Text)

    rule_pack: Mapped[RulePack] = relationship(back_populates="rules")


class ReviewSettings(Base):
    """Per-workspace defaults for new review runs. Provider credentials are never stored;
    only the provider name (one configured through environment variables) and model."""

    __tablename__ = "review_settings"
    __table_args__ = (
        CheckConstraint(
            _in("llm_provider", ("anthropic", "openai", "ollama", "mock")), name="provider"
        ),
        CheckConstraint("tau IS NULL OR (tau >= 0 AND tau <= 1)", name="tau_range"),
        CheckConstraint("budget_tokens > 0", name="budget_tokens_positive"),
        CheckConstraint("budget_microusd >= 0", name="budget_microusd_non_negative"),
        CheckConstraint("max_concurrency between 1 and 16", name="concurrency_range"),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    llm_provider: Mapped[str] = mapped_column(String(16), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(120), nullable=False)
    tau: Mapped[float | None] = mapped_column(Float)
    budget_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # Money is stored as integer micro-dollars (1e-6 USD): exact, and every conversion
    # from a computed cost rounds *up* (compliance_core.money).
    budget_microusd: Mapped[int] = mapped_column(BigInteger, nullable=False)
    pause_on_needs_review: Mapped[bool] = mapped_column(Boolean, nullable=False)
    max_concurrency: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    updated_at: Mapped[datetime] = _created()


class ReviewRun(Base):
    __tablename__ = "review_runs"
    __table_args__ = (
        CheckConstraint(_in("status", RUN_STATUSES), name="status"),
        CheckConstraint(
            "control IS NULL OR " + _in("control", ("pause", "cancel")), name="control"
        ),
        CheckConstraint("budget_tokens > 0", name="budget_tokens_positive"),
        CheckConstraint("budget_microusd >= 0", name="budget_microusd_non_negative"),
        Index("ix_review_runs_workspace_created", "workspace_id", "created_at"),
        Index("ix_review_runs_status_lease", "status", "lease_expires_at"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    rule_pack_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rule_packs.id", ondelete="SET NULL")
    )
    rule_pack_key: Mapped[str] = mapped_column(String(64), nullable=False)
    rule_pack_version: Mapped[str] = mapped_column(String(32), nullable=False)
    rule_pack_name: Mapped[str] = mapped_column(String(200), nullable=False)
    # The exact rules this run checks (immune to later rule-pack edits).
    rule_pack_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    rule_pack_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    control: Mapped[str | None] = mapped_column(String(10))
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _created()
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    llm_provider: Mapped[str] = mapped_column(String(16), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(120), nullable=False)
    llm_is_mock: Mapped[bool] = mapped_column(Boolean, nullable=False)
    # Run options without secrets: document scope, tau override, concurrency, pause flag.
    config_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    budget_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False)
    budget_microusd: Mapped[int] = mapped_column(BigInteger, nullable=False)
    totals_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=sql_text("'{}'::jsonb")
    )
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    lease_token: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    lease_owner: Mapped[str | None] = mapped_column(String(200))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    job_generation: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    event_seq: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")


class ReviewTask(Base):
    """One planned check: (rule x document) or (rule x subject). Drives the progress grid."""

    __tablename__ = "review_tasks"
    __table_args__ = (
        UniqueConstraint("run_id", "task_key", name="uq_review_tasks_run_task"),
        CheckConstraint(_in("kind", ("per_document", "cross_document")), name="kind"),
        CheckConstraint(
            _in("status", ("pending", "running", "done", "failed", "skipped")), name="status"
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = _run_fk()
    task_key: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    rule_key: Mapped[str] = mapped_column(String(64), nullable=False)
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL")
    )
    subject_key: Mapped[str] = mapped_column(String(200), nullable=False)
    subject_label: Mapped[str] = mapped_column(String(300), nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False)
    finding_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = _created()


class Finding(Base):
    """A machine finding. Never rewritten by humans: overrides are separate audited rows.

    ``machine_status`` is what the deciding component concluded; ``status`` is the
    status after citation verification (``needs_review`` when evidence is missing,
    inexact or unsupported).
    """

    __tablename__ = "findings"
    __table_args__ = (
        UniqueConstraint("run_id", "task_key", name="uq_findings_run_task"),
        CheckConstraint(_in("status", FINDING_STATUSES), name="status"),
        CheckConstraint(_in("machine_status", FINDING_STATUSES), name="machine_status"),
        CheckConstraint(_in("decided_by", ("model", "llm", "system")), name="decided_by"),
        CheckConstraint(_in("severity", ("low", "medium", "high")), name="severity"),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="confidence"
        ),
        Index("ix_findings_run_status", "run_id", "status"),
        Index("ix_findings_document_id", "document_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = _run_fk()
    task_key: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    rule_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rules.id", ondelete="SET NULL")
    )
    rule_key: Mapped[str] = mapped_column(String(64), nullable=False)
    rule_title: Mapped[str] = mapped_column(String(200), nullable=False)
    severity: Mapped[str] = mapped_column(String(8), nullable=False)
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL")
    )
    subject_key: Mapped[str] = mapped_column(String(200), nullable=False)
    subject_label: Mapped[str] = mapped_column(String(300), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    machine_status: Mapped[str] = mapped_column(String(16), nullable=False)
    compliance: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    decided_by: Mapped[str] = mapped_column(String(8), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    verification_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=sql_text("'{}'::jsonb")
    )
    escalated: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    escalation_reason: Mapped[str | None] = mapped_column(Text)
    model_version: Mapped[str | None] = mapped_column(String(200))
    prompt_version: Mapped[str | None] = mapped_column(String(120))
    llm_provider: Mapped[str | None] = mapped_column(String(16))
    llm_model: Mapped[str | None] = mapped_column(String(120))
    reasoning_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=sql_text("'[]'::jsonb")
    )
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _created()

    citations: Mapped[list[Citation]] = relationship(
        back_populates="finding",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Citation.position",
    )


class Citation(Base):
    __tablename__ = "citations"
    __table_args__ = (
        CheckConstraint("char_end > char_start", name="non_empty_range"),
        CheckConstraint(
            _in("role", ("evidence", "register_row", "entity", "supporting")), name="role"
        ),
        CheckConstraint(_in("bbox_precision", ("line", "chunk", "none")), name="bbox_precision"),
        Index("ix_citations_finding_id", "finding_id"),
        Index("ix_citations_document_id", "document_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    finding_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("findings.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    # SET NULL, not CASCADE: if the source document is deleted the citation stays as a
    # historical record and readers present the finding as needing review.
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL")
    )
    document_filename: Mapped[str] = mapped_column(String(FILENAME_MAX), nullable=False)
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="SET NULL")
    )
    table_row_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("table_rows.id", ondelete="SET NULL")
    )
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    bbox_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=sql_text("'[]'::jsonb")
    )
    bbox_precision: Mapped[str] = mapped_column(String(8), nullable=False)
    exact_match: Mapped[bool | None] = mapped_column(Boolean)
    supported: Mapped[bool | None] = mapped_column(Boolean)
    support_score: Mapped[float | None] = mapped_column(Float)
    support_method: Mapped[str | None] = mapped_column(String(64))
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")

    finding: Mapped[Finding] = relationship(back_populates="citations")


class HumanOverride(Base):
    """An audited human decision about a finding (the finding itself is unchanged)."""

    __tablename__ = "human_overrides"
    __table_args__ = (
        CheckConstraint(_in("new_status", FINDING_STATUSES), name="new_status"),
        CheckConstraint(_in("previous_status", FINDING_STATUSES), name="previous_status"),
        CheckConstraint("char_length(btrim(comment)) between 1 and 2000", name="comment_length"),
        Index("ix_human_overrides_finding_created", "finding_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    finding_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("findings.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    previous_status: Mapped[str] = mapped_column(String(16), nullable=False)
    new_status: Mapped[str] = mapped_column(String(16), nullable=False)
    comment: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _created()


class AgentEvent(Base):
    """Persisted agent step events. ``seq`` is a gap-tolerant, strictly increasing
    per-run cursor assigned under the run row lock, so commit order equals seq order."""

    __tablename__ = "agent_events"
    __table_args__ = (
        UniqueConstraint("run_id", "seq", name="uq_agent_events_run_seq"),
        UniqueConstraint("run_id", "dedupe_key", name="uq_agent_events_run_dedupe"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = _run_fk()
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    agent: Mapped[str] = mapped_column(String(32), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    dedupe_key: Mapped[str | None] = mapped_column(String(200))
    ts: Mapped[datetime] = _created()


class LlmCall(Base):
    """Budget ledger: one row per provider attempt, reserved *before* the request."""

    __tablename__ = "llm_calls"
    __table_args__ = (
        UniqueConstraint("run_id", "step_key", "attempt", name="uq_llm_calls_run_step_attempt"),
        CheckConstraint(
            _in("status", ("reserved", "succeeded", "failed", "unknown", "released")),
            name="status",
        ),
        CheckConstraint(
            _in("usage_source", ("provider", "mock_estimate", "none")), name="usage_source"
        ),
        CheckConstraint(
            "reserved_tokens >= 0 AND reserved_microusd >= 0", name="reserved_non_negative"
        ),
        Index("ix_llm_calls_run_status", "run_id", "status"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = _run_fk()
    step_key: Mapped[str] = mapped_column(String(200), nullable=False)
    attempt: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    agent: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    model: Mapped[str] = mapped_column(String(120), nullable=False)
    is_mock: Mapped[bool] = mapped_column(Boolean, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False)
    reserved_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    reserved_microusd: Mapped[int] = mapped_column(BigInteger, nullable=False)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cache_write_tokens: Mapped[int | None] = mapped_column(Integer)
    cache_read_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_microusd: Mapped[int | None] = mapped_column(BigInteger)
    usage_source: Mapped[str] = mapped_column(String(16), nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = _created()
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class StepResult(Base):
    """Memoized output of an expensive step (provider or model call), keyed
    deterministically, so a step re-executed after a crash is never re-billed."""

    __tablename__ = "step_results"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("review_runs.id", ondelete="CASCADE"), primary_key=True
    )
    step_key: Mapped[str] = mapped_column(String(200), primary_key=True)
    agent: Mapped[str] = mapped_column(String(32), nullable=False)
    output_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _created()


class ReviewReport(Base):
    """The report writer's snapshot at the end of a run (the API also renders live)."""

    __tablename__ = "review_reports"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("review_runs.id", ondelete="CASCADE"), primary_key=True
    )
    writer_version: Mapped[str] = mapped_column(String(64), nullable=False)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    markdown: Mapped[str] = mapped_column(Text, nullable=False)
    generated_at: Mapped[datetime] = _created()


class GraphCheckpoint(Base):
    """LangGraph checkpoints (``thread_id`` = review run id), serialized by LangGraph's
    serializer. Holds ids and small summaries only; never credentials."""

    __tablename__ = "graph_checkpoints"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("review_runs.id", ondelete="CASCADE"), primary_key=True
    )
    checkpoint_ns: Mapped[str] = mapped_column(Text, primary_key=True)
    checkpoint_id: Mapped[str] = mapped_column(Text, primary_key=True)
    parent_checkpoint_id: Mapped[str | None] = mapped_column(Text)
    checkpoint_type: Mapped[str] = mapped_column(String(32), nullable=False)
    checkpoint: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    metadata_type: Mapped[str] = mapped_column(String(32), nullable=False)
    metadata_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = _created()


class GraphCheckpointBlob(Base):
    __tablename__ = "graph_checkpoint_blobs"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("review_runs.id", ondelete="CASCADE"), primary_key=True
    )
    checkpoint_ns: Mapped[str] = mapped_column(Text, primary_key=True)
    channel: Mapped[str] = mapped_column(Text, primary_key=True)
    version: Mapped[str] = mapped_column(Text, primary_key=True)
    value_type: Mapped[str] = mapped_column(String(32), nullable=False)
    value: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class GraphCheckpointWrite(Base):
    __tablename__ = "graph_checkpoint_writes"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("review_runs.id", ondelete="CASCADE"), primary_key=True
    )
    checkpoint_ns: Mapped[str] = mapped_column(Text, primary_key=True)
    checkpoint_id: Mapped[str] = mapped_column(Text, primary_key=True)
    task_id: Mapped[str] = mapped_column(Text, primary_key=True)
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    channel: Mapped[str] = mapped_column(Text, nullable=False)
    value_type: Mapped[str] = mapped_column(String(32), nullable=False)
    value: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    task_path: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
