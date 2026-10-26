"""Typed shared state of the review graph.

The handoff's state fields are all present. Large data (document text, spans, quotes,
citations) lives in Postgres; state carries ids, statuses and small summaries so the
checkpoints written after every super-step stay small. Fields written by parallel tasks
use merge reducers keyed by task, finding or document id, so a resumed super-step that
re-applies stored writes produces the same state.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    return {**(left or {}), **(right or {})}


def last(left: Any, right: Any) -> Any:
    return right


class TaskRef(TypedDict):
    task_key: str
    kind: str  # per_document | cross_document
    rule_key: str
    document_id: str | None
    subject_key: str
    subject_label: str
    position: int


class ReviewState(TypedDict, total=False):
    run_id: str
    workspace_id: str
    # key, name, version, sha256 of the snapshot this run checks
    rule_pack: dict[str, Any]
    # [{id, filename, doc_type, file_format}]
    documents: list[dict[str, Any]]
    # vendor-register rows: [{subject_key, label, document_id, row_id, chunk_id, values, raw}]
    subjects: list[dict[str, Any]]
    tasks: list[TaskRef]
    model_info: dict[str, Any]
    prompt_versions: dict[str, str]
    # document id -> ids of verified extracted entities
    extracted_entities: Annotated[dict[str, list[str]], merge]
    # subject key -> register link outcome
    links: Annotated[dict[str, dict[str, Any]], merge]
    # task key -> retrieval summary
    candidates: Annotated[dict[str, dict[str, Any]], merge]
    # task key -> {finding_id, kind, machine_status}
    findings: Annotated[dict[str, dict[str, Any]], merge]
    # finding id -> citation ids
    citations: Annotated[dict[str, list[str]], merge]
    # finding id -> {status, verified}
    verified: Annotated[dict[str, dict[str, Any]], merge]
    budget: Annotated[dict[str, Any], merge]
    errors: Annotated[list[dict[str, Any]], operator.add]
    phase: Annotated[str, last]
    human_review: Annotated[dict[str, Any], merge]
    report: Annotated[dict[str, Any], merge]
