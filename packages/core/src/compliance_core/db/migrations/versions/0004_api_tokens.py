"""Phase 5: per-user API tokens for the MCP server (SHA-256 hashes only).

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = postgresql.UUID(as_uuid=True)


def upgrade() -> None:
    op.create_table(
        "api_tokens",
        sa.Column("id", UUID, server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("user_id", UUID, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("token_prefix", sa.String(length=16), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("scopes", postgresql.ARRAY(sa.String(length=16)), nullable=False),
        sa.Column("workspace_ids", postgresql.ARRAY(UUID), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "char_length(btrim(name)) between 1 and 100", name=op.f("ck_api_tokens_name_length")
        ),
        sa.CheckConstraint(
            "scopes <@ ARRAY['read', 'review', 'upload']::varchar[] AND 'read' = ANY(scopes)",
            name=op.f("ck_api_tokens_scopes_valid"),
        ),
        sa.CheckConstraint(
            "workspace_ids IS NULL OR cardinality(workspace_ids) > 0",
            name=op.f("ck_api_tokens_allowlist_non_empty"),
        ),
        sa.CheckConstraint(
            "expires_at > created_at", name=op.f("ck_api_tokens_expires_after_creation")
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_api_tokens_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_api_tokens")),
        sa.UniqueConstraint("token_hash", name=op.f("uq_api_tokens_token_hash")),
    )
    op.create_index("ix_api_tokens_user_created", "api_tokens", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_api_tokens_user_created", table_name="api_tokens")
    op.drop_table("api_tokens")
