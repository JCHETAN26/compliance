"""Review runs: start, read, control, tasks, findings, citations and human overrides.

Authorisation (every function takes the caller's user id):
- any workspace member reads runs, tasks, findings, events, citations and reports;
- owners and reviewers start runs, pause/resume/cancel them and record overrides;
- runs, findings and citations outside the caller's workspaces are "not found", exactly
  like ids that do not exist (no existence disclosure, no cross-workspace leaks). A
  citation is only returned if its document (when it still exists) belongs to the run's
  workspace as well.

The API only writes rows and enqueues work; all checking happens in the worker.

Findings are presented, never rewritten:
- a human override is a separate audited row; the latest one decides the effective status;
- every read re-checks each verified citation against the *current* source text. If the
  document was deleted or its text no longer has the quote at the stored offsets, the
  finding is presented as ``needs_review`` (``source_changed``), not as currently verified.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from compliance_core.db.models import (
    Citation,
    Document,
    DocumentContent,
    DocumentStatus,
    Finding,
    HumanOverride,
    ReviewRun,
    ReviewTask,
    User,
    WorkspaceMember,
    WorkspaceRole,
)
from compliance_core.document_schemas import BBoxRead
from compliance_core.money import micro_to_usd
from compliance_core.review_schemas import (
    BudgetRead,
    CitationDetail,
    CitationRead,
    FindingPage,
    FindingRead,
    HumanOverrideRead,
    ReviewRunRead,
    ReviewTaskRead,
    ReviewTasksRead,
    RulePackRef,
    RulePackRuleRef,
)
from compliance_core.rule_packs import spec_document
from compliance_core.services.budget import run_usage
from compliance_core.services.errors import (
    InvalidOperationError,
    NotFoundError,
    PermissionDeniedError,
)
from compliance_core.services.review_events import append_event
from compliance_core.services.rule_packs import load_pack, pack_spec
from compliance_core.services.workspaces import require_role

RUN_NOT_FOUND = "Review run not found."
FINDING_NOT_FOUND = "Finding not found."
CITATION_NOT_FOUND = "Citation not found."
WRITE_ROLES = frozenset({WorkspaceRole.OWNER, WorkspaceRole.REVIEWER})
TERMINAL = frozenset({"completed", "cancelled"})
ACTIVE = frozenset({"queued", "running"})
CONTEXT_CHARS = 300


@dataclass(frozen=True)
class ProviderSelection:
    """A provider choice already validated by the caller (no credentials)."""

    provider: str
    model: str
    is_mock: bool
    price_source: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float


@dataclass(frozen=True)
class RunOptions:
    budget_tokens: int
    budget_microusd: int
    tau: float | None
    pause_on_needs_review: bool
    max_concurrency: int
    rule_check_backend: str
    max_documents: int


# --------------------------------------------------------------------------- access


async def run_for_member(
    session: AsyncSession, run_id: uuid.UUID, user_id: uuid.UUID, *, lock: bool = False
) -> tuple[ReviewRun, WorkspaceRole]:
    query = (
        select(ReviewRun, WorkspaceMember.role)
        .join(WorkspaceMember, WorkspaceMember.workspace_id == ReviewRun.workspace_id)
        .where(ReviewRun.id == run_id, WorkspaceMember.user_id == user_id)
    )
    if lock:
        # Refresh an already-loaded run: its status must be the one read under the lock.
        query = query.with_for_update(of=ReviewRun).execution_options(populate_existing=True)
    row = (await session.execute(query)).one_or_none()
    if row is None:
        raise NotFoundError(RUN_NOT_FOUND)
    return row[0], row[1]


def _require_writer(role: WorkspaceRole, action: str) -> None:
    if role not in WRITE_ROLES:
        raise PermissionDeniedError(f"Your role ({role.value}) does not allow you to {action}.")


# --------------------------------------------------------------------------- start


async def start_review(
    session: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    rule_pack_id: uuid.UUID,
    document_ids: Sequence[uuid.UUID] | None,
    provider: ProviderSelection,
    options: RunOptions,
) -> ReviewRun:
    await require_role(session, workspace_id, user_id, WRITE_ROLES, "start reviews")
    pack, _ = await load_pack(session, rule_pack_id, user_id)
    if pack.workspace_id not in (None, workspace_id):
        raise NotFoundError("Rule pack not found.")
    spec = pack_spec(pack)

    if document_ids is not None:
        wanted = list(dict.fromkeys(document_ids))
        if not wanted:
            raise InvalidOperationError("Select at least one document.")
        docs = list(
            await session.scalars(
                select(Document).where(
                    Document.workspace_id == workspace_id, Document.id.in_(wanted)
                )
            )
        )
        if len(docs) != len(wanted):
            raise NotFoundError("Document not found.")
        not_ready = sorted(d.filename for d in docs if d.status is not DocumentStatus.READY)
        if not_ready:
            raise InvalidOperationError(
                "These documents are not ready yet: " + ", ".join(not_ready[:10])
            )
    else:
        docs = list(
            await session.scalars(
                select(Document)
                .where(
                    Document.workspace_id == workspace_id,
                    Document.status == DocumentStatus.READY,
                )
                .order_by(Document.created_at, Document.id)
            )
        )
        if not docs:
            raise InvalidOperationError("The workspace has no processed documents to review.")
    if len(docs) > options.max_documents:
        raise InvalidOperationError(
            f"A review can cover at most {options.max_documents} documents ({len(docs)} selected)."
        )

    run = ReviewRun(
        workspace_id=workspace_id,
        rule_pack_id=pack.id,
        rule_pack_key=pack.key,
        rule_pack_version=pack.version,
        rule_pack_name=pack.name,
        rule_pack_snapshot={
            "spec": spec_document(spec),
            "rule_ids": {r.key: str(r.id) for r in pack.rules},
        },
        rule_pack_sha256=pack.content_sha256,
        status="queued",
        created_by=user_id,
        llm_provider=provider.provider,
        llm_model=provider.model,
        llm_is_mock=provider.is_mock,
        config_json={
            "document_ids": [str(d.id) for d in docs],
            "tau_override": options.tau,
            "pause_on_needs_review": options.pause_on_needs_review,
            "max_concurrency": options.max_concurrency,
            "rule_check_backend": options.rule_check_backend,
            "price": {
                "source": provider.price_source,
                "input_usd_per_mtok": provider.input_usd_per_mtok,
                "output_usd_per_mtok": provider.output_usd_per_mtok,
            },
        },
        budget_tokens=options.budget_tokens,
        budget_microusd=options.budget_microusd,
        totals_json={},
    )
    session.add(run)
    await session.flush()
    await append_event(
        session,
        run.id,
        "system",
        "run_queued",
        {
            "rule_pack": f"{pack.key}@{pack.version}",
            "documents": len(docs),
            "llm_provider": provider.provider,
            "llm_model": provider.model,
            "mock": provider.is_mock,
            "budget_tokens": options.budget_tokens,
            "budget_usd": micro_to_usd(options.budget_microusd),
            "by_user": str(user_id),
        },
    )
    await session.commit()
    await session.refresh(run)
    return run


# --------------------------------------------------------------------------- reads


async def budget_read(session: AsyncSession, run: ReviewRun) -> BudgetRead:
    usage = await run_usage(session, run.id)
    return BudgetRead(
        tokens_cap=run.budget_tokens,
        usd_cap=micro_to_usd(run.budget_microusd),
        tokens_used=usage.tokens,
        usd_used=micro_to_usd(usage.microusd),
        tokens_reserved_open=usage.tokens_open,
        usd_reserved_open=micro_to_usd(usage.microusd_open),
        calls=usage.calls,
        calls_unknown=usage.unknown,
        usage_source=usage.usage_source,  # type: ignore[arg-type]
    )


async def run_read(session: AsyncSession, run: ReviewRun) -> ReviewRunRead:
    """Run summary. ``findings_by_status`` uses the same *effective* statuses as the
    findings and report views (human overrides, and citations re-checked against the
    current sources); the original machine and post-verification counts are reported
    separately and labelled as such."""
    config = run.config_json
    presented = await present_findings(session, run)
    effective: dict[str, int] = defaultdict(int)
    machine: dict[str, int] = defaultdict(int)
    verified_status: dict[str, int] = defaultdict(int)
    compliance: dict[str, int] = defaultdict(int)
    for f in presented:
        effective[f.status] += 1
        machine[f.machine_status] += 1
        verified_status[f.verified_status] += 1
        compliance[f.compliance] += 1
    task_counts = dict(
        (
            await session.execute(
                select(ReviewTask.status, func.count())
                .where(ReviewTask.run_id == run.id)
                .group_by(ReviewTask.status)
            )
        ).all()
    )
    totals = dict(run.totals_json)
    totals["findings_by_status"] = dict(effective)
    totals["findings_by_compliance"] = dict(compliance)
    totals["machine_findings_by_status"] = dict(machine)
    totals["verified_findings_by_status"] = dict(verified_status)
    totals["source_changed"] = sum(f.source_changed for f in presented)
    totals["tasks_by_status"] = {k: int(v) for k, v in task_counts.items()}
    totals["overrides"] = sum(f.override_count for f in presented)
    totals["human_decided"] = sum(f.decided_by == "human" for f in presented)
    return ReviewRunRead(
        id=run.id,
        workspace_id=run.workspace_id,
        rule_pack=RulePackRef(
            id=run.rule_pack_id,
            key=run.rule_pack_key,
            name=run.rule_pack_name,
            version=run.rule_pack_version,
        ),
        status=run.status,  # type: ignore[arg-type]
        control=run.control,  # type: ignore[arg-type]
        created_by=run.created_by,
        created_at=run.created_at,
        updated_at=run.updated_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        llm_provider=run.llm_provider,  # type: ignore[arg-type]
        llm_model=run.llm_model,
        llm_is_mock=run.llm_is_mock,
        rule_check_backend=str(config.get("rule_check_backend", "service")),
        model_version=totals.get("model_version"),
        tau=totals.get("tau", config.get("tau_override")),
        pause_on_needs_review=bool(config.get("pause_on_needs_review")),
        max_concurrency=int(config.get("max_concurrency", 1)),
        document_count=len(config.get("document_ids", [])),
        budget=await budget_read(session, run),
        totals=totals,
        error_code=run.error_code,
        error_message=run.error_message,
        last_event_seq=run.event_seq,
    )


async def get_run(session: AsyncSession, run_id: uuid.UUID, user_id: uuid.UUID) -> ReviewRunRead:
    run, _ = await run_for_member(session, run_id, user_id)
    return await run_read(session, run)


async def list_runs(
    session: AsyncSession, workspace_id: uuid.UUID, user_id: uuid.UUID, limit: int = 50
) -> list[ReviewRunRead]:
    await require_role(session, workspace_id, user_id, frozenset(WorkspaceRole), "view reviews")
    runs = await session.scalars(
        select(ReviewRun)
        .where(ReviewRun.workspace_id == workspace_id)
        .order_by(ReviewRun.created_at.desc(), ReviewRun.id)
        .limit(limit)
    )
    return [await run_read(session, r) for r in runs]


# --------------------------------------------------------------------------- control


async def control_run(
    session: AsyncSession, run_id: uuid.UUID, user_id: uuid.UUID, action: str
) -> tuple[ReviewRun, bool]:
    """Apply pause/resume/cancel. Returns ``(run, enqueue)``: when ``enqueue`` is true the
    caller must enqueue the job for ``run.job_generation``."""
    run, role = await run_for_member(session, run_id, user_id, lock=True)
    _require_writer(role, "control reviews")
    status = run.status
    enqueue = False
    values: dict[str, Any] = {"updated_at": func.now()}
    if action == "pause":
        if status == "running":
            values["control"] = "pause"
        elif status == "queued":
            values.update(status="paused", control=None)
        else:
            raise InvalidOperationError(f"A {status} review cannot be paused.")
    elif action == "cancel":
        if status == "running":
            values["control"] = "cancel"
        elif status in ("queued", "paused", "awaiting_review", "failed"):
            values.update(status="cancelled", control=None, finished_at=func.now())
        else:
            raise InvalidOperationError(f"A {status} review cannot be cancelled.")
    elif action == "resume":
        if status in ("paused", "failed", "awaiting_review"):
            values.update(
                status="queued",
                control=None,
                error_code=None,
                error_message=None,
                job_generation=ReviewRun.job_generation + 1,
            )
            enqueue = True
        elif status == "queued":
            enqueue = True  # re-enqueue the same generation (job ids de-duplicate)
        elif status == "running" and run.control == "pause":
            values["control"] = None  # withdraw a pause that has not taken effect yet
        else:
            raise InvalidOperationError(f"A {status} review cannot be resumed.")
    else:  # pragma: no cover - validated by the schema
        raise InvalidOperationError(f"Unknown action {action!r}.")
    await session.execute(update(ReviewRun).where(ReviewRun.id == run.id).values(**values))
    await append_event(
        session,
        run.id,
        "system",
        f"{action}_requested",
        {"by_user": str(user_id), "from_status": status},
    )
    await session.commit()
    await session.refresh(run)
    return run, enqueue


# --------------------------------------------------------------------------- tasks


def _snapshot_rules(run: ReviewRun) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = run.rule_pack_snapshot["spec"]["rules"]
    return rules


async def list_tasks(
    session: AsyncSession, run_id: uuid.UUID, user_id: uuid.UUID
) -> ReviewTasksRead:
    run, _ = await run_for_member(session, run_id, user_id)
    findings = await present_findings(session, run)
    status_by_id = {f.id: f.status for f in findings}
    tasks = await session.scalars(
        select(ReviewTask).where(ReviewTask.run_id == run.id).order_by(ReviewTask.position)
    )
    return ReviewTasksRead(
        rules=[
            RulePackRuleRef(
                key=r["key"],
                title=r["title"],
                scope=r["scope"],
                severity=r["severity"],
                model_supported=bool(r.get("model_supported", False)),
            )
            for r in _snapshot_rules(run)
        ],
        tasks=[
            ReviewTaskRead(
                task_key=t.task_key,
                kind=t.kind,  # type: ignore[arg-type]
                rule_key=t.rule_key,
                document_id=t.document_id,
                subject_key=t.subject_key,
                subject_label=t.subject_label,
                status=t.status,  # type: ignore[arg-type]
                finding_id=t.finding_id,
                finding_status=status_by_id.get(t.finding_id) if t.finding_id else None,
            )
            for t in tasks
        ],
    )


# --------------------------------------------------------------------------- findings


async def citation_states(
    session: AsyncSession, citation_ids: Collection[uuid.UUID]
) -> dict[uuid.UUID, str]:
    """``valid`` | ``source_deleted`` | ``source_changed`` for each citation, re-checked
    against the current canonical text (Postgres ``substr`` is code-point based, like
    Python slicing)."""
    if not citation_ids:
        return {}
    same = (
        func.substr(
            DocumentContent.text, Citation.char_start + 1, Citation.char_end - Citation.char_start
        )
        == Citation.quote
    )
    rows = await session.execute(
        select(Citation.id, Citation.document_id, DocumentContent.document_id, same)
        .outerjoin(DocumentContent, DocumentContent.document_id == Citation.document_id)
        .where(Citation.id.in_(list(citation_ids)))
    )
    states: dict[uuid.UUID, str] = {}
    for cid, doc_id, content_doc, matches in rows.all():
        if doc_id is None or content_doc is None:
            states[cid] = "source_deleted"
        elif not matches:
            states[cid] = "source_changed"
        else:
            states[cid] = "valid"
    return states


def _citation_read(c: Citation, state: str) -> CitationRead:
    return CitationRead(
        id=c.id,
        role=c.role,  # type: ignore[arg-type]
        document_id=c.document_id,
        document_filename=c.document_filename,
        chunk_id=c.chunk_id,
        table_row_id=c.table_row_id,
        quote=c.quote,
        char_start=c.char_start,
        char_end=c.char_end,
        page_number=c.page_number,
        bboxes=[BBoxRead.model_validate(b) for b in c.bbox_json],
        bbox_precision=c.bbox_precision,  # type: ignore[arg-type]
        exact_match=c.exact_match,
        supported=c.supported,
        support_score=c.support_score,
        support_method=c.support_method,
        verified=c.verified,
        current=state,  # type: ignore[arg-type]
    )


def compliance_of(status: str, compliant_statuses: list[str] | None) -> str:
    if compliant_statuses is None:
        return "not_assessed"
    if status == "needs_review":
        return "needs_review"
    return "compliant" if status in compliant_statuses else "non_compliant"


async def present_findings(
    session: AsyncSession,
    run: ReviewRun,
    *,
    finding_ids: Collection[uuid.UUID] | None = None,
) -> list[FindingRead]:
    query = (
        select(Finding)
        .options(selectinload(Finding.citations))
        .where(Finding.run_id == run.id)
        .order_by(Finding.created_at, Finding.task_key)
    )
    if finding_ids is not None:
        query = query.where(Finding.id.in_(list(finding_ids)))
    findings = list(await session.scalars(query))
    if not findings:
        return []
    ids = [f.id for f in findings]
    states = await citation_states(session, [c.id for f in findings for c in f.citations])
    override_rows = await session.execute(
        select(HumanOverride, User.email)
        .outerjoin(User, User.id == HumanOverride.user_id)
        .where(HumanOverride.finding_id.in_(ids))
        .order_by(HumanOverride.created_at, HumanOverride.id)
    )
    overrides: dict[uuid.UUID, list[HumanOverrideRead]] = defaultdict(list)
    for o, email in override_rows.all():
        overrides[o.finding_id].append(
            HumanOverrideRead(
                id=o.id,
                finding_id=o.finding_id,
                user_id=o.user_id,
                user_email=email,
                previous_status=o.previous_status,  # type: ignore[arg-type]
                new_status=o.new_status,  # type: ignore[arg-type]
                comment=o.comment,
                created_at=o.created_at,
            )
        )
    doc_ids = {f.document_id for f in findings if f.document_id}
    filenames = (
        dict(
            (
                await session.execute(
                    select(Document.id, Document.filename).where(Document.id.in_(list(doc_ids)))
                )
            ).all()
        )
        if doc_ids
        else {}
    )
    compliant_by_rule = {r["key"]: r.get("compliant_statuses") for r in _snapshot_rules(run)}

    out = []
    for f in findings:
        cits = [_citation_read(c, states.get(c.id, "source_deleted")) for c in f.citations]
        source_changed = f.verified and any(c.verified and c.current != "valid" for c in cits)
        history = overrides.get(f.id, [])
        latest = history[-1] if history else None
        machine_view = "needs_review" if source_changed else f.status
        status = latest.new_status if latest else machine_view
        out.append(
            FindingRead(
                id=f.id,
                run_id=f.run_id,
                task_key=f.task_key,
                kind=f.kind,  # type: ignore[arg-type]
                rule_key=f.rule_key,
                rule_title=f.rule_title,
                severity=f.severity,  # type: ignore[arg-type]
                document_id=f.document_id,
                document_filename=filenames.get(f.document_id) if f.document_id else None,
                subject_key=f.subject_key,
                subject_label=f.subject_label,
                status=status,  # type: ignore[arg-type]
                machine_status=f.machine_status,  # type: ignore[arg-type]
                verified_status=f.status,  # type: ignore[arg-type]
                compliance=compliance_of(status, compliant_by_rule.get(f.rule_key)),  # type: ignore[arg-type]
                confidence=f.confidence,
                decided_by="human" if latest else f.decided_by,  # type: ignore[arg-type]
                machine_decided_by=f.decided_by,  # type: ignore[arg-type]
                rationale=f.rationale,
                verified=f.verified and not source_changed,
                source_changed=source_changed,
                verification=f.verification_json,
                escalated=f.escalated,
                escalation_reason=f.escalation_reason,
                model_version=f.model_version,
                prompt_version=f.prompt_version,
                llm_provider=f.llm_provider,
                llm_model=f.llm_model,
                reasoning=f.reasoning_json,
                citations=cits,
                override=latest,
                override_count=len(history),
                created_at=f.created_at,
            )
        )
    return out


async def list_findings(
    session: AsyncSession,
    run_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    status: str | None = None,
    severity: str | None = None,
    decided_by: str | None = None,
    subject_key: str | None = None,
    offset: int = 0,
    limit: int = 500,
) -> FindingPage:
    run, _ = await run_for_member(session, run_id, user_id)
    items = await present_findings(session, run)
    if status:
        items = [f for f in items if f.status == status]
    if severity:
        items = [f for f in items if f.severity == severity]
    if decided_by:
        items = [f for f in items if f.decided_by == decided_by]
    if subject_key:
        items = [f for f in items if f.subject_key == subject_key]
    return FindingPage(total=len(items), items=items[offset : offset + limit])


async def _finding_for_member(
    session: AsyncSession, finding_id: uuid.UUID, user_id: uuid.UUID, *, lock: bool = False
) -> tuple[Finding, ReviewRun, WorkspaceRole]:
    query = (
        select(Finding, ReviewRun, WorkspaceMember.role)
        .join(ReviewRun, ReviewRun.id == Finding.run_id)
        .join(WorkspaceMember, WorkspaceMember.workspace_id == ReviewRun.workspace_id)
        .where(Finding.id == finding_id, WorkspaceMember.user_id == user_id)
    )
    if lock:
        query = query.with_for_update(of=Finding).execution_options(populate_existing=True)
    row = (await session.execute(query)).one_or_none()
    if row is None:
        raise NotFoundError(FINDING_NOT_FOUND)
    return row[0], row[1], row[2]


async def get_finding(
    session: AsyncSession, finding_id: uuid.UUID, user_id: uuid.UUID
) -> FindingRead:
    finding, run, _ = await _finding_for_member(session, finding_id, user_id)
    return (await present_findings(session, run, finding_ids=[finding.id]))[0]


async def override_finding(
    session: AsyncSession,
    finding_id: uuid.UUID,
    user_id: uuid.UUID,
    new_status: str,
    comment: str,
) -> FindingRead:
    # Lock order is run, then finding, as in the worker and in control_run(); the active
    # check is made under the run lock so a concurrent resume cannot slip in between.
    finding, run, role = await _finding_for_member(session, finding_id, user_id)
    _require_writer(role, "override findings")
    await run_for_member(session, run.id, user_id, lock=True)
    finding, run, role = await _finding_for_member(session, finding_id, user_id, lock=True)
    _require_writer(role, "override findings")
    if run.status in ACTIVE:
        raise InvalidOperationError(
            "Findings can be overridden once the run is paused, waiting for review or finished."
        )
    current = (await present_findings(session, run, finding_ids=[finding.id]))[0]
    override = HumanOverride(
        finding_id=finding.id,
        user_id=user_id,
        previous_status=current.status,
        new_status=new_status,
        comment=comment,
    )
    session.add(override)
    await session.flush()
    await append_event(
        session,
        run.id,
        "human",
        "override_recorded",
        {
            "finding_id": str(finding.id),
            "task_key": finding.task_key,
            "previous_status": current.status,
            "new_status": new_status,
            "by_user": str(user_id),
        },
    )
    await session.commit()
    return (await present_findings(session, run, finding_ids=[finding.id]))[0]


async def get_citation(
    session: AsyncSession, citation_id: uuid.UUID, user_id: uuid.UUID
) -> CitationDetail:
    row = (
        await session.execute(
            select(Citation, Finding.run_id, ReviewRun.workspace_id)
            .join(Finding, Finding.id == Citation.finding_id)
            .join(ReviewRun, ReviewRun.id == Finding.run_id)
            .join(
                WorkspaceMember,
                and_(
                    WorkspaceMember.workspace_id == ReviewRun.workspace_id,
                    WorkspaceMember.user_id == user_id,
                ),
            )
            .where(Citation.id == citation_id)
        )
    ).one_or_none()
    if row is None:
        raise NotFoundError(CITATION_NOT_FOUND)
    citation, run_id, workspace_id = row
    file_format: str | None = None
    before = after = ""
    if citation.document_id is not None:
        document = await session.get(Document, citation.document_id)
        if document is None or document.workspace_id != workspace_id:
            # Defence in depth: never serve a document from another workspace.
            raise NotFoundError(CITATION_NOT_FOUND)
        file_format = document.file_format
    state = (await citation_states(session, [citation.id])).get(citation.id, "source_deleted")
    if state == "valid" and citation.document_id is not None:
        start = max(citation.char_start - CONTEXT_CHARS, 0)
        context = await session.execute(
            select(
                func.substr(DocumentContent.text, start + 1, citation.char_start - start),
                func.substr(DocumentContent.text, citation.char_end + 1, CONTEXT_CHARS),
            ).where(DocumentContent.document_id == citation.document_id)
        )
        before, after = context.one()
    base = _citation_read(citation, state)
    return CitationDetail(
        **base.model_dump(),
        run_id=run_id,
        finding_id=citation.finding_id,
        workspace_id=workspace_id,
        file_format=file_format,
        context_before=before or "",
        context_after=after or "",
    )
