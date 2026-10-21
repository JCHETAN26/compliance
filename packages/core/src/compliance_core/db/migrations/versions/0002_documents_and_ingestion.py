"""Documents and ingestion outputs: documents, contents, pages, blocks, sheets, rows,
chunks (pgvector HNSW + generated tsvector GIN) and extracted_entities.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-21
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import SchemaItem

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

document_status = postgresql.ENUM(
    "uploaded", "processing", "ready", "failed", name="document_status", create_type=False
)
doc_type = postgresql.ENUM(
    "contract", "spreadsheet", "policy", "other", name="doc_type", create_type=False
)
UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB()


def _id() -> sa.Column[Any]:
    return sa.Column("id", UUID, server_default=sa.text("gen_random_uuid()"), nullable=False)


def _doc_id(table: str) -> list[SchemaItem]:
    return [
        sa.Column("document_id", UUID, nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f(f"fk_{table}_document_id_documents"),
            ondelete="CASCADE",
        ),
    ]


def upgrade() -> None:
    bind = op.get_bind()
    postgresql.ENUM("uploaded", "processing", "ready", "failed", name="document_status").create(
        bind
    )
    postgresql.ENUM("contract", "spreadsheet", "policy", "other", name="doc_type").create(bind)

    op.create_table(
        "documents",
        _id(),
        sa.Column("workspace_id", UUID, nullable=False),
        sa.Column("uploaded_by", UUID, nullable=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("mime_type", sa.String(127), nullable=False),
        sa.Column("file_format", sa.String(8), nullable=False),
        sa.Column("storage_key", sa.String(255), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("status", document_status, nullable=False),
        sa.Column("stage", sa.String(32), nullable=True),
        sa.Column("progress", sa.SmallInteger(), server_default="0", nullable=False),
        sa.Column("doc_type", doc_type, nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("ocr_page_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("chunk_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("warnings", JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("ingest_generation", sa.Integer(), server_default="1", nullable=False),
        sa.Column("embedding_model", sa.String(200), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name=op.f("ck_documents_sha256_hex")),
        sa.CheckConstraint("progress between 0 and 100", name=op.f("ck_documents_progress_range")),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_documents_workspace_id_workspaces"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["uploaded_by"],
            ["users.id"],
            name=op.f("fk_documents_uploaded_by_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint("storage_key", name=op.f("uq_documents_storage_key")),
        sa.UniqueConstraint("workspace_id", "sha256", name="uq_documents_workspace_sha256"),
    )
    op.create_index("ix_documents_workspace_created", "documents", ["workspace_id", "created_at"])
    op.create_index("ix_documents_status_updated", "documents", ["status", "updated_at"])

    op.create_table(
        "document_contents",
        sa.Column("document_id", UUID, nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_sha256", sa.String(64), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_document_contents_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("document_id", name=op.f("pk_document_contents")),
    )

    op.create_table(
        "pages",
        _id(),
        *_doc_id("pages"),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("width", sa.Float(), nullable=False),
        sa.Column("height", sa.Float(), nullable=False),
        sa.Column("is_ocr", sa.Boolean(), nullable=False),
        sa.Column("ocr_confidence", sa.Float(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_pages")),
        sa.UniqueConstraint("document_id", "page_number", name="uq_pages_document_page"),
    )

    op.create_table(
        "document_blocks",
        _id(),
        *_doc_id("document_blocks"),
        sa.Column("block_index", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("style", sa.String(120), nullable=True),
        sa.Column("level", sa.SmallInteger(), nullable=True),
        sa.Column("starts_section", sa.Boolean(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("section_path", sa.Text(), nullable=True),
        sa.Column("bbox_json", JSONB, nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_blocks")),
        sa.UniqueConstraint("document_id", "block_index", name="uq_document_blocks_document_index"),
    )

    op.create_table(
        "table_sheets",
        _id(),
        *_doc_id("table_sheets"),
        sa.Column("sheet_index", sa.Integer(), nullable=False),
        sa.Column("sheet_name", sa.String(255), nullable=False),
        sa.Column("columns", JSONB, nullable=False),
        sa.Column("header_row_number", sa.Integer(), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("truncated", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_table_sheets")),
        sa.UniqueConstraint("document_id", "sheet_index", name="uq_table_sheets_document_index"),
    )

    op.create_table(
        "table_rows",
        _id(),
        *_doc_id("table_rows"),
        sa.Column("sheet_name", sa.String(255), nullable=False),
        sa.Column("sheet_index", sa.Integer(), nullable=False),
        sa.Column("row_index", sa.Integer(), nullable=False),
        sa.Column("data_json", JSONB, nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_table_rows")),
        sa.UniqueConstraint(
            "document_id", "sheet_name", "row_index", name="uq_table_rows_document_row"
        ),
    )

    op.create_table(
        "chunks",
        _id(),
        *_doc_id("chunks"),
        sa.Column("page_id", UUID, nullable=True),
        sa.Column("table_row_id", UUID, nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("bbox_json", JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("section_path", sa.Text(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("embedding_windows", sa.SmallInteger(), nullable=False),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english'::regconfig, text)", persisted=True),
            nullable=True,
        ),
        sa.CheckConstraint("char_end > char_start", name=op.f("ck_chunks_non_empty_range")),
        sa.ForeignKeyConstraint(
            ["page_id"], ["pages.id"], name=op.f("fk_chunks_page_id_pages"), ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["table_row_id"],
            ["table_rows.id"],
            name=op.f("fk_chunks_table_row_id_table_rows"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
    )
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")
    op.create_index("ix_chunks_page_id", "chunks", ["page_id"])
    op.create_index("ix_chunks_table_row_id", "chunks", ["table_row_id"])

    op.create_table(
        "extracted_entities",
        _id(),
        *_doc_id("extracted_entities"),
        sa.Column("type", sa.String(64), nullable=False),
        sa.Column("value_json", JSONB, nullable=False),
        sa.Column("chunk_id", UUID, nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["chunks.id"],
            name=op.f("fk_extracted_entities_chunk_id_chunks"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_extracted_entities")),
    )
    op.create_index("ix_extracted_entities_document_id", "extracted_entities", ["document_id"])
    op.create_index("ix_extracted_entities_chunk_id", "extracted_entities", ["chunk_id"])


def downgrade() -> None:
    op.drop_table("extracted_entities")
    op.drop_table("chunks")
    op.drop_table("table_rows")
    op.drop_table("table_sheets")
    op.drop_table("document_blocks")
    op.drop_table("pages")
    op.drop_table("document_contents")
    op.drop_table("documents")
    bind = op.get_bind()
    postgresql.ENUM(name="doc_type").drop(bind, checkfirst=True)
    postgresql.ENUM(name="document_status").drop(bind, checkfirst=True)
