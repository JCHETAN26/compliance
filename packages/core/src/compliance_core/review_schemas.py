"""API/service schemas for rule packs, review settings, review runs and their results."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from compliance_core.document_schemas import BBoxRead
from compliance_core.rule_packs import RulePackSpec

FindingStatus = Literal["satisfied", "violated", "not_mentioned", "needs_review"]
RunStatus = Literal[
    "queued", "running", "paused", "awaiting_review", "completed", "failed", "cancelled"
]
Compliance = Literal["compliant", "non_compliant", "needs_review", "not_assessed"]
ProviderName = Literal["mock", "anthropic", "openai", "ollama"]
Comment = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- rule packs


class RuleRead(BaseModel):
    id: uuid.UUID
    key: str
    title: str
    description: str
    hypothesis_text: str
    scope: Literal["per_document", "cross_document"]
    applies_to_doc_types: list[str]
    severity: Literal["low", "medium", "high"]
    model_supported: bool
    contractnli_key: str | None
    compliant_statuses: list[str] | None
    cross_document: dict[str, Any] | None
    remediation: str | None


class RulePackSummary(BaseModel):
    id: uuid.UUID
    key: str
    name: str
    version: str
    description: str
    builtin: bool
    workspace_id: uuid.UUID | None
    source: str
    rule_count: int
    model_supported_count: int
    cross_document_count: int
    content_sha256: str
    created_at: datetime
    updated_at: datetime
    editable: bool = Field(description="Whether the caller may edit this pack")


class RulePackRead(RulePackSummary):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    rules: list[RuleRead]
    register_: dict[str, Any] | None = Field(alias="register")
    provenance: dict[str, str] | None


class RulePackCreate(_Body):
    workspace_id: uuid.UUID
    pack: RulePackSpec


class RulePackImport(_Body):
    workspace_id: uuid.UUID
    yaml: Annotated[str, StringConstraints(min_length=1, max_length=262_144)]


# --------------------------------------------------------------------------- settings


class ProviderOption(BaseModel):
    provider: ProviderName
    default_model: str
    is_mock: bool
    credential_configured: bool = Field(
        description="Whether the provider's key variable is set (its value is never read here)"
    )
    price_source: str | None = Field(description="Where token prices come from; None if unknown")
    usable: bool
    problem: str | None = None


class LlmSettingsRead(BaseModel):
    deployment_provider: ProviderName
    deployment_model: str
    providers: list[ProviderOption]
    rule_check_backend: Literal["service", "mock"]
    mock_only: bool = Field(description="No real LLM provider is configured (UI shows a banner)")


class ReviewSettingsRead(BaseModel):
    workspace_id: uuid.UUID
    llm_provider: ProviderName
    llm_model: str
    tau: float | None = Field(
        description="Cascade threshold override; None = the model's DEV-selected tau"
    )
    budget_tokens: int
    budget_usd: float
    pause_on_needs_review: bool
    max_concurrency: int
    is_default: bool
    updated_at: datetime | None


class ReviewSettingsUpdate(_Body):
    llm_provider: ProviderName
    llm_model: Annotated[str, StringConstraints(strip_whitespace=True, max_length=120)] = ""
    tau: Annotated[float, Field(ge=0, le=1)] | None = None
    budget_tokens: Annotated[int, Field(ge=1_000, le=50_000_000)]
    budget_usd: Annotated[float, Field(ge=0, le=1_000)]
    pause_on_needs_review: bool
    max_concurrency: Annotated[int, Field(ge=1, le=16)]


# --------------------------------------------------------------------------- runs


class StartReview(_Body):
    rule_pack_id: uuid.UUID
    document_ids: Annotated[list[uuid.UUID], Field(max_length=500)] | None = Field(
        default=None, description="Documents to review; omitted = every ready document"
    )
    llm_provider: ProviderName | None = None
    llm_model: Annotated[str, StringConstraints(strip_whitespace=True, max_length=120)] | None = (
        None
    )
    tau: Annotated[float, Field(ge=0, le=1)] | None = None
    budget_tokens: Annotated[int, Field(ge=1_000, le=50_000_000)] | None = None
    budget_usd: Annotated[float, Field(ge=0, le=1_000)] | None = None
    pause_on_needs_review: bool | None = None
    max_concurrency: Annotated[int, Field(ge=1, le=16)] | None = None


class RunControl(_Body):
    action: Literal["pause", "resume", "cancel"]


class RulePackRef(BaseModel):
    id: uuid.UUID | None
    key: str
    name: str
    version: str


class BudgetRead(BaseModel):
    tokens_cap: int
    usd_cap: float
    tokens_used: int = Field(description="Settled tokens plus held reservations")
    usd_used: float = Field(description="Settled cost plus held reservations (rounded up)")
    tokens_reserved_open: int
    usd_reserved_open: float
    calls: int
    calls_unknown: int = Field(description="Attempts with unknown outcome (reservation kept)")
    usage_source: Literal["provider", "mock_estimate", "none", "mixed"]


class ReviewRunRead(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    rule_pack: RulePackRef
    status: RunStatus
    control: Literal["pause", "cancel"] | None
    created_by: uuid.UUID | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    llm_provider: ProviderName
    llm_model: str
    llm_is_mock: bool
    rule_check_backend: str
    model_version: str | None
    tau: float | None
    pause_on_needs_review: bool
    max_concurrency: int
    document_count: int
    budget: BudgetRead
    totals: dict[str, Any]
    error_code: str | None
    error_message: str | None
    last_event_seq: int


class ReviewTaskRead(BaseModel):
    task_key: str
    kind: Literal["per_document", "cross_document"]
    rule_key: str
    document_id: uuid.UUID | None
    subject_key: str
    subject_label: str
    status: Literal["pending", "running", "done", "failed", "skipped"]
    finding_id: uuid.UUID | None
    finding_status: FindingStatus | None


class RulePackRuleRef(BaseModel):
    key: str
    title: str
    scope: Literal["per_document", "cross_document"]
    severity: Literal["low", "medium", "high"]
    model_supported: bool


class ReviewTasksRead(BaseModel):
    rules: list[RulePackRuleRef]
    tasks: list[ReviewTaskRead]


# --------------------------------------------------------------------------- findings


class CitationRead(BaseModel):
    id: uuid.UUID
    role: Literal["evidence", "register_row", "entity", "supporting"]
    document_id: uuid.UUID | None
    document_filename: str
    chunk_id: uuid.UUID | None
    table_row_id: uuid.UUID | None
    quote: str
    char_start: int
    char_end: int
    page_number: int | None
    bboxes: list[BBoxRead]
    bbox_precision: Literal["line", "chunk", "none"]
    exact_match: bool | None
    supported: bool | None
    support_score: float | None
    support_method: str | None
    verified: bool
    current: Literal["valid", "source_deleted", "source_changed"] = Field(
        description="Re-checked on read: the quote still sits at its offsets in the source"
    )


class CitationDetail(CitationRead):
    run_id: uuid.UUID
    finding_id: uuid.UUID
    workspace_id: uuid.UUID
    file_format: str | None
    context_before: str
    context_after: str


class HumanOverrideRead(BaseModel):
    id: uuid.UUID
    finding_id: uuid.UUID
    user_id: uuid.UUID | None
    user_email: str | None
    previous_status: FindingStatus
    new_status: FindingStatus
    comment: str
    created_at: datetime


class FindingRead(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    task_key: str
    kind: Literal["per_document", "cross_document"]
    rule_key: str
    rule_title: str
    severity: Literal["low", "medium", "high"]
    document_id: uuid.UUID | None
    document_filename: str | None
    subject_key: str
    subject_label: str
    status: FindingStatus = Field(
        description="Effective status: latest human override, else the verified machine status "
        "(needs_review if a cited source was deleted or changed after verification)"
    )
    machine_status: FindingStatus = Field(description="What the deciding component concluded")
    verified_status: FindingStatus = Field(description="Status after citation verification")
    compliance: Compliance
    confidence: float | None
    decided_by: Literal["model", "llm", "system", "human"]
    machine_decided_by: Literal["model", "llm", "system"]
    rationale: str
    verified: bool
    source_changed: bool
    verification: dict[str, Any]
    escalated: bool
    escalation_reason: str | None
    model_version: str | None
    prompt_version: str | None
    llm_provider: str | None
    llm_model: str | None
    reasoning: list[dict[str, Any]]
    citations: list[CitationRead]
    override: HumanOverrideRead | None
    override_count: int
    created_at: datetime


class FindingPage(BaseModel):
    total: int
    items: list[FindingRead]


class OverrideRequest(_Body):
    status: FindingStatus
    comment: Comment


# --------------------------------------------------------------------------- events


class AgentEventRead(BaseModel):
    seq: int
    agent: str
    event_type: str
    payload: dict[str, Any]
    ts: datetime


class EventPage(BaseModel):
    items: list[AgentEventRead]
    next_after: int = Field(description="Pass as ?after= (or Last-Event-ID) to continue")
    has_more: bool
    run_status: RunStatus


# --------------------------------------------------------------------------- report


class ReportRuleRow(BaseModel):
    rule_key: str
    rule_title: str
    severity: Literal["low", "medium", "high"]
    scope: Literal["per_document", "cross_document"]
    counts: dict[str, int]
    compliance_counts: dict[str, int]


class ReportSubjectRow(BaseModel):
    subject_key: str
    subject_label: str
    document_id: uuid.UUID | None
    findings: int
    violated: int
    needs_review: int
    compliant: int
    worst_severity: Literal["low", "medium", "high"] | None
    status: Literal["compliant", "at_risk", "needs_review", "not_assessed"]


class ReportStatement(BaseModel):
    text: str
    finding_ids: list[uuid.UUID] = Field(description="The findings this statement is based on")


class ReportRisk(BaseModel):
    finding_id: uuid.UUID
    rank: int
    severity: Literal["low", "medium", "high"]
    rule_title: str
    subject_label: str
    status: FindingStatus
    compliance: Compliance
    decided_by: str


class ReportAction(BaseModel):
    finding_ids: list[uuid.UUID]
    priority: Literal["high", "medium", "low"]
    action: str


class ReportRead(BaseModel):
    run_id: uuid.UUID
    workspace_id: uuid.UUID
    generated_at: datetime
    writer_version: str
    run_status: RunStatus
    rule_pack: RulePackRef
    llm_provider: str
    llm_model: str
    llm_is_mock: bool
    rule_check_backend: str
    model_version: str | None
    tau: float | None
    prompt_versions: dict[str, str]
    budget: BudgetRead
    complete: bool = Field(description="False while the run has not finished all tasks")
    summary: list[ReportStatement]
    rules: list[ReportRuleRow]
    subjects: list[ReportSubjectRow]
    risks: list[ReportRisk]
    actions: list[ReportAction]
    needs_review: list[ReportRisk]
    findings: list[FindingRead]
    limitations: list[str]
