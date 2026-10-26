"""Phase 4: rule packs, review settings, review runs, tasks, findings, citations, human
overrides, agent events, the LLM budget ledger, step memoization, report snapshots, the
LangGraph checkpoint tables, extraction provenance on extracted_entities and PDF line
boxes on document_contents.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB()


def upgrade() -> None:
    op.create_table(
        "review_settings",
        sa.Column("workspace_id", UUID, nullable=False),
        sa.Column("llm_provider", sa.String(length=16), nullable=False),
        sa.Column("llm_model", sa.String(length=120), nullable=False),
        sa.Column("tau", sa.Float(), nullable=True),
        sa.Column("budget_tokens", sa.BigInteger(), nullable=False),
        sa.Column("budget_microusd", sa.BigInteger(), nullable=False),
        sa.Column("pause_on_needs_review", sa.Boolean(), nullable=False),
        sa.Column("max_concurrency", sa.SmallInteger(), nullable=False),
        sa.Column("updated_by", UUID, nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "llm_provider in ('anthropic', 'openai', 'ollama', 'mock')",
            name=op.f("ck_review_settings_provider"),
        ),
        sa.CheckConstraint(
            "budget_tokens > 0", name=op.f("ck_review_settings_budget_tokens_positive")
        ),
        sa.CheckConstraint(
            "budget_microusd >= 0", name=op.f("ck_review_settings_budget_microusd_non_negative")
        ),
        sa.CheckConstraint(
            "max_concurrency between 1 and 16", name=op.f("ck_review_settings_concurrency_range")
        ),
        sa.CheckConstraint(
            "tau IS NULL OR (tau >= 0 AND tau <= 1)", name=op.f("ck_review_settings_tau_range")
        ),
        sa.ForeignKeyConstraint(
            ["updated_by"],
            ["users.id"],
            name=op.f("fk_review_settings_updated_by_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_review_settings_workspace_id_workspaces"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("workspace_id", name=op.f("pk_review_settings")),
    )
    op.create_table(
        "rule_packs",
        sa.Column("id", UUID, server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("workspace_id", UUID, nullable=True),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("register_json", JSONB, nullable=True),
        sa.Column("provenance_json", JSONB, nullable=True),
        sa.Column("created_by", UUID, nullable=True),
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
        sa.CheckConstraint(
            "key ~ '^[a-z0-9][a-z0-9_]{0,63}$'", name=op.f("ck_rule_packs_key_format")
        ),
        sa.CheckConstraint(
            "source in ('builtin', 'import', 'api', 'copy')", name=op.f("ck_rule_packs_source")
        ),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["users.id"],
            name=op.f("fk_rule_packs_created_by_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_rule_packs_workspace_id_workspaces"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rule_packs")),
    )
    op.create_index(
        "uq_rule_packs_builtin_key_version",
        "rule_packs",
        ["key", "version"],
        unique=True,
        postgresql_where=sa.text("workspace_id IS NULL"),
    )
    op.create_index(
        "uq_rule_packs_workspace_key_version",
        "rule_packs",
        ["workspace_id", "key", "version"],
        unique=True,
        postgresql_where=sa.text("workspace_id IS NOT NULL"),
    )
    op.create_table(
        "review_runs",
        sa.Column("id", UUID, server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("workspace_id", UUID, nullable=False),
        sa.Column("rule_pack_id", UUID, nullable=True),
        sa.Column("rule_pack_key", sa.String(length=64), nullable=False),
        sa.Column("rule_pack_version", sa.String(length=32), nullable=False),
        sa.Column("rule_pack_name", sa.String(length=200), nullable=False),
        sa.Column("rule_pack_snapshot", JSONB, nullable=False),
        sa.Column("rule_pack_sha256", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("control", sa.String(length=10), nullable=True),
        sa.Column("created_by", UUID, nullable=True),
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
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("llm_provider", sa.String(length=16), nullable=False),
        sa.Column("llm_model", sa.String(length=120), nullable=False),
        sa.Column("llm_is_mock", sa.Boolean(), nullable=False),
        sa.Column("config_json", JSONB, nullable=False),
        sa.Column("budget_tokens", sa.BigInteger(), nullable=False),
        sa.Column("budget_microusd", sa.BigInteger(), nullable=False),
        sa.Column("totals_json", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("lease_token", UUID, nullable=True),
        sa.Column("lease_owner", sa.String(length=200), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("job_generation", sa.Integer(), server_default="1", nullable=False),
        sa.Column("event_seq", sa.BigInteger(), server_default="0", nullable=False),
        sa.CheckConstraint(
            "control IS NULL OR control in ('pause', 'cancel')", name=op.f("ck_review_runs_control")
        ),
        sa.CheckConstraint(
            "status in ('queued', 'running', 'paused', 'awaiting_review', 'completed', 'failed', 'cancelled')",
            name=op.f("ck_review_runs_status"),
        ),
        sa.CheckConstraint("budget_tokens > 0", name=op.f("ck_review_runs_budget_tokens_positive")),
        sa.CheckConstraint(
            "budget_microusd >= 0", name=op.f("ck_review_runs_budget_microusd_non_negative")
        ),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["users.id"],
            name=op.f("fk_review_runs_created_by_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["rule_pack_id"],
            ["rule_packs.id"],
            name=op.f("fk_review_runs_rule_pack_id_rule_packs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_review_runs_workspace_id_workspaces"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_review_runs")),
    )
    op.create_index(
        "ix_review_runs_status_lease", "review_runs", ["status", "lease_expires_at"], unique=False
    )
    op.create_index(
        "ix_review_runs_workspace_created",
        "review_runs",
        ["workspace_id", "created_at"],
        unique=False,
    )
    op.create_table(
        "rules",
        sa.Column("id", UUID, server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("rule_pack_id", UUID, nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        sa.Column("hypothesis_text", sa.Text(), nullable=False),
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("applies_to_doc_types", JSONB, nullable=False),
        sa.Column("severity", sa.String(length=8), nullable=False),
        sa.Column("model_supported", sa.Boolean(), nullable=False),
        sa.Column("contractnli_key", sa.String(length=16), nullable=True),
        sa.Column("compliant_statuses", JSONB, nullable=False),
        sa.Column("cross_document", JSONB, nullable=True),
        sa.Column("remediation", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "scope in ('per_document', 'cross_document')", name=op.f("ck_rules_scope")
        ),
        sa.CheckConstraint("severity in ('low', 'medium', 'high')", name=op.f("ck_rules_severity")),
        sa.ForeignKeyConstraint(
            ["rule_pack_id"],
            ["rule_packs.id"],
            name=op.f("fk_rules_rule_pack_id_rule_packs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rules")),
        sa.UniqueConstraint("rule_pack_id", "key", name="uq_rules_pack_key"),
    )
    op.create_table(
        "agent_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", UUID, nullable=False),
        sa.Column("seq", sa.BigInteger(), nullable=False),
        sa.Column("agent", sa.String(length=32), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("payload_json", JSONB, nullable=False),
        sa.Column("dedupe_key", sa.String(length=200), nullable=True),
        sa.Column(
            "ts", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["review_runs.id"],
            name=op.f("fk_agent_events_run_id_review_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_events")),
        sa.UniqueConstraint("run_id", "dedupe_key", name="uq_agent_events_run_dedupe"),
        sa.UniqueConstraint("run_id", "seq", name="uq_agent_events_run_seq"),
    )
    op.create_table(
        "findings",
        sa.Column("id", UUID, server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("run_id", UUID, nullable=False),
        sa.Column("task_key", sa.String(length=200), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("rule_id", UUID, nullable=True),
        sa.Column("rule_key", sa.String(length=64), nullable=False),
        sa.Column("rule_title", sa.String(length=200), nullable=False),
        sa.Column("severity", sa.String(length=8), nullable=False),
        sa.Column("document_id", UUID, nullable=True),
        sa.Column("subject_key", sa.String(length=200), nullable=False),
        sa.Column("subject_label", sa.String(length=300), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("machine_status", sa.String(length=16), nullable=False),
        sa.Column("compliance", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("decided_by", sa.String(length=8), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("verified", sa.Boolean(), server_default="false", nullable=False),
        sa.Column(
            "verification_json", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column("escalated", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("escalation_reason", sa.Text(), nullable=True),
        sa.Column("model_version", sa.String(length=200), nullable=True),
        sa.Column("prompt_version", sa.String(length=120), nullable=True),
        sa.Column("llm_provider", sa.String(length=16), nullable=True),
        sa.Column("llm_model", sa.String(length=120), nullable=True),
        sa.Column("reasoning_json", JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False),
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
        sa.CheckConstraint(
            "decided_by in ('model', 'llm', 'system')", name=op.f("ck_findings_decided_by")
        ),
        sa.CheckConstraint(
            "machine_status in ('satisfied', 'violated', 'not_mentioned', 'needs_review')",
            name=op.f("ck_findings_machine_status"),
        ),
        sa.CheckConstraint(
            "severity in ('low', 'medium', 'high')", name=op.f("ck_findings_severity")
        ),
        sa.CheckConstraint(
            "status in ('satisfied', 'violated', 'not_mentioned', 'needs_review')",
            name=op.f("ck_findings_status"),
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name=op.f("ck_findings_confidence"),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_findings_document_id_documents"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["rule_id"], ["rules.id"], name=op.f("fk_findings_rule_id_rules"), ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["review_runs.id"],
            name=op.f("fk_findings_run_id_review_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_findings")),
        sa.UniqueConstraint("run_id", "task_key", name="uq_findings_run_task"),
    )
    op.create_index("ix_findings_document_id", "findings", ["document_id"], unique=False)
    op.create_index("ix_findings_run_status", "findings", ["run_id", "status"], unique=False)
    op.create_table(
        "graph_checkpoint_blobs",
        sa.Column("run_id", UUID, nullable=False),
        sa.Column("checkpoint_ns", sa.Text(), nullable=False),
        sa.Column("channel", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("value_type", sa.String(length=32), nullable=False),
        sa.Column("value", sa.LargeBinary(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["review_runs.id"],
            name=op.f("fk_graph_checkpoint_blobs_run_id_review_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "run_id", "checkpoint_ns", "channel", "version", name=op.f("pk_graph_checkpoint_blobs")
        ),
    )
    op.create_table(
        "graph_checkpoint_writes",
        sa.Column("run_id", UUID, nullable=False),
        sa.Column("checkpoint_ns", sa.Text(), nullable=False),
        sa.Column("checkpoint_id", sa.Text(), nullable=False),
        sa.Column("task_id", sa.Text(), nullable=False),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("channel", sa.Text(), nullable=False),
        sa.Column("value_type", sa.String(length=32), nullable=False),
        sa.Column("value", sa.LargeBinary(), nullable=False),
        sa.Column("task_path", sa.Text(), server_default="", nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["review_runs.id"],
            name=op.f("fk_graph_checkpoint_writes_run_id_review_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "run_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
            name=op.f("pk_graph_checkpoint_writes"),
        ),
    )
    op.create_table(
        "graph_checkpoints",
        sa.Column("run_id", UUID, nullable=False),
        sa.Column("checkpoint_ns", sa.Text(), nullable=False),
        sa.Column("checkpoint_id", sa.Text(), nullable=False),
        sa.Column("parent_checkpoint_id", sa.Text(), nullable=True),
        sa.Column("checkpoint_type", sa.String(length=32), nullable=False),
        sa.Column("checkpoint", sa.LargeBinary(), nullable=False),
        sa.Column("metadata_type", sa.String(length=32), nullable=False),
        sa.Column("metadata_blob", sa.LargeBinary(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["review_runs.id"],
            name=op.f("fk_graph_checkpoints_run_id_review_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "run_id", "checkpoint_ns", "checkpoint_id", name=op.f("pk_graph_checkpoints")
        ),
    )
    op.create_table(
        "llm_calls",
        sa.Column("id", UUID, server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("run_id", UUID, nullable=False),
        sa.Column("step_key", sa.String(length=200), nullable=False),
        sa.Column("attempt", sa.SmallInteger(), nullable=False),
        sa.Column("agent", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=16), nullable=False),
        sa.Column("model", sa.String(length=120), nullable=False),
        sa.Column("is_mock", sa.Boolean(), nullable=False),
        sa.Column("prompt_version", sa.String(length=120), nullable=False),
        sa.Column("status", sa.String(length=10), nullable=False),
        sa.Column("reserved_tokens", sa.Integer(), nullable=False),
        sa.Column("reserved_microusd", sa.BigInteger(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("cache_write_tokens", sa.Integer(), nullable=True),
        sa.Column("cache_read_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_microusd", sa.BigInteger(), nullable=True),
        sa.Column("usage_source", sa.String(length=16), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status in ('reserved', 'succeeded', 'failed', 'unknown', 'released')",
            name=op.f("ck_llm_calls_status"),
        ),
        sa.CheckConstraint(
            "usage_source in ('provider', 'mock_estimate', 'none')",
            name=op.f("ck_llm_calls_usage_source"),
        ),
        sa.CheckConstraint(
            "reserved_tokens >= 0 AND reserved_microusd >= 0",
            name=op.f("ck_llm_calls_reserved_non_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["review_runs.id"],
            name=op.f("fk_llm_calls_run_id_review_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_llm_calls")),
        sa.UniqueConstraint("run_id", "step_key", "attempt", name="uq_llm_calls_run_step_attempt"),
    )
    op.create_index("ix_llm_calls_run_status", "llm_calls", ["run_id", "status"], unique=False)
    op.create_table(
        "review_reports",
        sa.Column("run_id", UUID, nullable=False),
        sa.Column("writer_version", sa.String(length=64), nullable=False),
        sa.Column("report_json", JSONB, nullable=False),
        sa.Column("markdown", sa.Text(), nullable=False),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["review_runs.id"],
            name=op.f("fk_review_reports_run_id_review_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_review_reports")),
    )
    op.create_table(
        "review_tasks",
        sa.Column("id", UUID, server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("run_id", UUID, nullable=False),
        sa.Column("task_key", sa.String(length=200), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("rule_key", sa.String(length=64), nullable=False),
        sa.Column("document_id", UUID, nullable=True),
        sa.Column("subject_key", sa.String(length=200), nullable=False),
        sa.Column("subject_label", sa.String(length=300), nullable=False),
        sa.Column("status", sa.String(length=10), nullable=False),
        sa.Column("finding_id", UUID, nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind in ('per_document', 'cross_document')", name=op.f("ck_review_tasks_kind")
        ),
        sa.CheckConstraint(
            "status in ('pending', 'running', 'done', 'failed', 'skipped')",
            name=op.f("ck_review_tasks_status"),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_review_tasks_document_id_documents"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["review_runs.id"],
            name=op.f("fk_review_tasks_run_id_review_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_review_tasks")),
        sa.UniqueConstraint("run_id", "task_key", name="uq_review_tasks_run_task"),
    )
    op.create_table(
        "step_results",
        sa.Column("run_id", UUID, nullable=False),
        sa.Column("step_key", sa.String(length=200), nullable=False),
        sa.Column("agent", sa.String(length=32), nullable=False),
        sa.Column("output_json", JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["review_runs.id"],
            name=op.f("fk_step_results_run_id_review_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "step_key", name=op.f("pk_step_results")),
    )
    op.create_table(
        "citations",
        sa.Column("id", UUID, server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("finding_id", UUID, nullable=False),
        sa.Column("position", sa.SmallInteger(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("document_id", UUID, nullable=True),
        sa.Column("document_filename", sa.String(length=255), nullable=False),
        sa.Column("chunk_id", UUID, nullable=True),
        sa.Column("table_row_id", UUID, nullable=True),
        sa.Column("quote", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("bbox_json", JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("bbox_precision", sa.String(length=8), nullable=False),
        sa.Column("exact_match", sa.Boolean(), nullable=True),
        sa.Column("supported", sa.Boolean(), nullable=True),
        sa.Column("support_score", sa.Float(), nullable=True),
        sa.Column("support_method", sa.String(length=64), nullable=True),
        sa.Column("verified", sa.Boolean(), server_default="false", nullable=False),
        sa.CheckConstraint(
            "bbox_precision in ('line', 'chunk', 'none')", name=op.f("ck_citations_bbox_precision")
        ),
        sa.CheckConstraint(
            "role in ('evidence', 'register_row', 'entity', 'supporting')",
            name=op.f("ck_citations_role"),
        ),
        sa.CheckConstraint("char_end > char_start", name=op.f("ck_citations_non_empty_range")),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["chunks.id"],
            name=op.f("fk_citations_chunk_id_chunks"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_citations_document_id_documents"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["finding_id"],
            ["findings.id"],
            name=op.f("fk_citations_finding_id_findings"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["table_row_id"],
            ["table_rows.id"],
            name=op.f("fk_citations_table_row_id_table_rows"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_citations")),
    )
    op.create_index("ix_citations_document_id", "citations", ["document_id"], unique=False)
    op.create_index("ix_citations_finding_id", "citations", ["finding_id"], unique=False)
    op.create_table(
        "human_overrides",
        sa.Column("id", UUID, server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("finding_id", UUID, nullable=False),
        sa.Column("user_id", UUID, nullable=True),
        sa.Column("previous_status", sa.String(length=16), nullable=False),
        sa.Column("new_status", sa.String(length=16), nullable=False),
        sa.Column("comment", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "new_status in ('satisfied', 'violated', 'not_mentioned', 'needs_review')",
            name=op.f("ck_human_overrides_new_status"),
        ),
        sa.CheckConstraint(
            "previous_status in ('satisfied', 'violated', 'not_mentioned', 'needs_review')",
            name=op.f("ck_human_overrides_previous_status"),
        ),
        sa.CheckConstraint(
            "char_length(btrim(comment)) between 1 and 2000",
            name=op.f("ck_human_overrides_comment_length"),
        ),
        sa.ForeignKeyConstraint(
            ["finding_id"],
            ["findings.id"],
            name=op.f("fk_human_overrides_finding_id_findings"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_human_overrides_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_human_overrides")),
    )
    op.create_index(
        "ix_human_overrides_finding_created",
        "human_overrides",
        ["finding_id", "created_at"],
        unique=False,
    )
    op.add_column("document_contents", sa.Column("line_boxes", JSONB, nullable=True))
    op.add_column("extracted_entities", sa.Column("review_run_id", UUID, nullable=True))
    op.add_column("extracted_entities", sa.Column("quote", sa.Text(), nullable=True))
    op.add_column("extracted_entities", sa.Column("char_start", sa.Integer(), nullable=True))
    op.add_column("extracted_entities", sa.Column("char_end", sa.Integer(), nullable=True))
    op.add_column("extracted_entities", sa.Column("page_number", sa.Integer(), nullable=True))
    op.add_column("extracted_entities", sa.Column("method", sa.String(length=24), nullable=True))
    op.add_column(
        "extracted_entities", sa.Column("prompt_version", sa.String(length=120), nullable=True)
    )
    op.add_column(
        "extracted_entities",
        sa.Column("verified", sa.Boolean(), server_default="false", nullable=False),
    )
    op.create_index(
        "ix_extracted_entities_review_run_id", "extracted_entities", ["review_run_id"], unique=False
    )
    op.create_foreign_key(
        op.f("fk_extracted_entities_review_run_id_review_runs"),
        "extracted_entities",
        "review_runs",
        ["review_run_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("fk_extracted_entities_review_run_id_review_runs"),
        "extracted_entities",
        type_="foreignkey",
    )
    op.drop_index("ix_extracted_entities_review_run_id", table_name="extracted_entities")
    for column in (
        "verified",
        "prompt_version",
        "method",
        "page_number",
        "char_end",
        "char_start",
        "quote",
        "review_run_id",
    ):
        op.drop_column("extracted_entities", column)
    op.drop_column("document_contents", "line_boxes")
    for table in (
        "human_overrides",
        "citations",
        "step_results",
        "review_tasks",
        "review_reports",
        "llm_calls",
        "graph_checkpoints",
        "graph_checkpoint_writes",
        "graph_checkpoint_blobs",
        "findings",
        "agent_events",
        "rules",
        "review_runs",
        "rule_packs",
        "review_settings",
    ):
        op.drop_table(table)
