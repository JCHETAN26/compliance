"""Helpers shared by the agent nodes."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from compliance_agents.store import CitationDraft
from compliance_agents.text import DocView
from compliance_core.services.reviews import compliance_of

PER_DOCUMENT = "per_document"
CROSS_DOCUMENT = "cross_document"
MAX_EVIDENCE_CITATIONS = 3


def digest(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def span_citation(
    doc: DocView,
    start: int,
    end: int,
    *,
    role: str = "evidence",
    support_method: str | None = None,
    supported: bool | None = None,
    table_row_id: uuid.UUID | None = None,
) -> CitationDraft:
    """A citation of ``doc.text[start:end]`` with its chunk, page and highlight geometry."""
    chunk = doc.chunk_for(start, end)
    boxes, precision, page = doc.geometry(start, end)
    return CitationDraft(
        role=role,
        document_id=doc.id,
        document_filename=doc.filename,
        chunk_id=chunk.id if chunk else None,
        table_row_id=table_row_id if table_row_id else (chunk.table_row_id if chunk else None),
        quote=doc.slice(start, end),
        char_start=start,
        char_end=end,
        page_number=page,
        bboxes=boxes,
        bbox_precision=precision,
        support_method=support_method,
        supported=supported,
    )


def rule_compliance(rule: dict[str, Any], status: str) -> str:
    return compliance_of(status, rule.get("compliant_statuses"))


def error_record(
    agent: str, code: str, message: str, task_key: str | None = None
) -> dict[str, Any]:
    return {"agent": agent, "code": code, "message": message[:300], "task_key": task_key}
