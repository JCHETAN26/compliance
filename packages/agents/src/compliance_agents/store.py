"""Worker-side persistence for one review run.

Every write transaction starts by locking the run row (``SELECT ... FOR UPDATE``) and
checking the lease token (fencing). Taking the same lock first in every transaction keeps
lock ordering simple (no share-to-exclusive upgrades) and serialises budget reservations,
so parallel graph tasks can never jointly exceed the run's caps.

All writes are idempotent per deterministic key: findings per ``(run, task_key)``,
memoized step results per ``(run, step_key)``, entities per ``(run, document)``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from compliance_core.db.models import (
    Citation,
    ExtractedEntity,
    Finding,
    LlmCall,
    ReviewRun,
    ReviewTask,
    StepResult,
)
from compliance_core.services.budget import run_usage
from compliance_core.services.review_events import LeaseLostError, append_event


class BudgetExceededError(Exception):
    def __init__(self, message: str, *, tokens_left: int, microusd_left: int) -> None:
        super().__init__(message)
        self.tokens_left = tokens_left
        self.microusd_left = microusd_left


class RunStoppedError(Exception):
    """The run is no longer in a state where work may continue (paused/cancelled)."""


@dataclass
class CitationDraft:
    role: str
    document_id: uuid.UUID
    document_filename: str
    chunk_id: uuid.UUID | None
    table_row_id: uuid.UUID | None
    quote: str
    char_start: int
    char_end: int
    page_number: int | None
    bboxes: list[dict[str, Any]]
    bbox_precision: str
    exact_match: bool | None = None
    supported: bool | None = None
    support_score: float | None = None
    support_method: str | None = None
    verified: bool = False


@dataclass
class FindingDraft:
    task_key: str
    kind: str
    rule_key: str
    rule_id: uuid.UUID | None
    rule_title: str
    severity: str
    document_id: uuid.UUID | None
    subject_key: str
    subject_label: str
    machine_status: str
    decided_by: str
    rationale: str
    confidence: float | None = None
    escalated: bool = False
    escalation_reason: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    reasoning: list[dict[str, Any]] = field(default_factory=list)
    compliance: str = "needs_review"


@dataclass
class EntityDraft:
    document_id: uuid.UUID
    type: str
    value: dict[str, Any]
    chunk_id: uuid.UUID | None
    quote: str | None
    char_start: int | None
    char_end: int | None
    page_number: int | None
    method: str
    prompt_version: str | None
    confidence: float | None = None


class RunStore:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        run_id: uuid.UUID,
        lease_token: uuid.UUID,
    ) -> None:
        self.sessions = sessions
        self.run_id = run_id
        self.lease_token = lease_token

    async def _fence(self, session: AsyncSession) -> ReviewRun:
        run = await session.scalar(
            select(ReviewRun).where(ReviewRun.id == self.run_id).with_for_update()
        )
        if run is None or run.lease_token != self.lease_token:
            raise LeaseLostError(f"lease on run {self.run_id} is no longer held")
        return run

    # ------------------------------------------------------------------ events

    async def emit(
        self,
        agent: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        dedupe_key: str | None = None,
    ) -> int | None:
        async with self.sessions() as session, session.begin():
            await self._fence(session)
            return await append_event(
                session,
                self.run_id,
                agent,
                event_type,
                payload,
                dedupe_key=dedupe_key,
                lease_token=self.lease_token,
            )

    # ------------------------------------------------------------------ tasks

    async def upsert_tasks(self, tasks: list[dict[str, Any]]) -> None:
        if not tasks:
            return
        async with self.sessions() as session, session.begin():
            await self._fence(session)
            await session.execute(
                insert(ReviewTask)
                .values(
                    [
                        {
                            "run_id": self.run_id,
                            "task_key": t["task_key"],
                            "kind": t["kind"],
                            "rule_key": t["rule_key"],
                            "document_id": uuid.UUID(t["document_id"])
                            if t.get("document_id")
                            else None,
                            "subject_key": t["subject_key"],
                            "subject_label": t["subject_label"][:300],
                            "status": "pending",
                            "position": t["position"],
                        }
                        for t in tasks
                    ]
                )
                .on_conflict_do_nothing(constraint="uq_review_tasks_run_task")
            )

    async def set_task(
        self, task_key: str, status: str, finding_id: uuid.UUID | None = None
    ) -> None:
        values: dict[str, Any] = {"status": status, "updated_at": func.now()}
        if finding_id is not None:
            values["finding_id"] = finding_id
        async with self.sessions() as session, session.begin():
            await self._fence(session)
            await session.execute(
                update(ReviewTask)
                .where(ReviewTask.run_id == self.run_id, ReviewTask.task_key == task_key)
                .values(**values)
            )

    # ------------------------------------------------------------------ findings

    async def save_finding(self, draft: FindingDraft, citations: list[CitationDraft]) -> uuid.UUID:
        """Create or replace the finding for ``draft.task_key`` (idempotent on re-runs)."""
        values: dict[str, Any] = {
            "kind": draft.kind,
            "rule_id": draft.rule_id,
            "rule_key": draft.rule_key,
            "rule_title": draft.rule_title,
            "severity": draft.severity,
            "document_id": draft.document_id,
            "subject_key": draft.subject_key,
            "subject_label": draft.subject_label[:300],
            "status": draft.machine_status,
            "machine_status": draft.machine_status,
            "compliance": draft.compliance,
            "confidence": draft.confidence,
            "decided_by": draft.decided_by,
            "rationale": draft.rationale,
            "verified": False,
            "verification_json": {},
            "escalated": draft.escalated,
            "escalation_reason": draft.escalation_reason,
            "model_version": draft.model_version,
            "prompt_version": draft.prompt_version,
            "llm_provider": draft.llm_provider,
            "llm_model": draft.llm_model,
            "reasoning_json": draft.reasoning,
            "updated_at": func.now(),
        }
        async with self.sessions() as session, session.begin():
            await self._fence(session)
            finding_id = await session.scalar(
                insert(Finding)
                .values(run_id=self.run_id, task_key=draft.task_key, **values)
                .on_conflict_do_update(constraint="uq_findings_run_task", set_=values)
                .returning(Finding.id)
            )
            assert finding_id is not None  # noqa: S101
            await session.execute(delete(Citation).where(Citation.finding_id == finding_id))
            if citations:
                await session.execute(
                    insert(Citation).values(
                        [
                            {
                                "finding_id": finding_id,
                                "position": i,
                                "role": c.role,
                                "document_id": c.document_id,
                                "document_filename": c.document_filename[:255],
                                "chunk_id": c.chunk_id,
                                "table_row_id": c.table_row_id,
                                "quote": c.quote,
                                "char_start": c.char_start,
                                "char_end": c.char_end,
                                "page_number": c.page_number,
                                "bbox_json": c.bboxes,
                                "bbox_precision": c.bbox_precision,
                                "exact_match": c.exact_match,
                                "supported": c.supported,
                                "support_score": c.support_score,
                                "support_method": c.support_method,
                                "verified": c.verified,
                            }
                            for i, c in enumerate(citations)
                        ]
                    )
                )
            await session.execute(
                update(ReviewTask)
                .where(ReviewTask.run_id == self.run_id, ReviewTask.task_key == draft.task_key)
                .values(status="done", finding_id=finding_id, updated_at=func.now())
            )
            return uuid.UUID(str(finding_id))

    async def save_verification(
        self,
        finding_id: uuid.UUID,
        *,
        status: str,
        verified: bool,
        compliance: str,
        verification: dict[str, Any],
        citation_results: dict[uuid.UUID, dict[str, Any]],
    ) -> None:
        async with self.sessions() as session, session.begin():
            await self._fence(session)
            await session.execute(
                update(Finding)
                .where(Finding.id == finding_id, Finding.run_id == self.run_id)
                .values(
                    status=status,
                    verified=verified,
                    compliance=compliance,
                    verification_json=verification,
                    updated_at=func.now(),
                )
            )
            for citation_id, values in citation_results.items():
                await session.execute(
                    update(Citation)
                    .where(Citation.id == citation_id, Citation.finding_id == finding_id)
                    .values(**values)
                )

    async def load_findings(self, task_keys: list[str] | None = None) -> list[Finding]:
        async with self.sessions() as session:
            query = (
                select(Finding)
                .options(selectinload(Finding.citations))
                .where(Finding.run_id == self.run_id)
            )
            if task_keys is not None:
                query = query.where(Finding.task_key.in_(task_keys))
            return list(await session.scalars(query))

    # ------------------------------------------------------------------ entities

    async def replace_entities(
        self, document_id: uuid.UUID, kinds: list[str], entities: list[EntityDraft]
    ) -> list[uuid.UUID]:
        async with self.sessions() as session, session.begin():
            await self._fence(session)
            await session.execute(
                delete(ExtractedEntity).where(
                    ExtractedEntity.review_run_id == self.run_id,
                    ExtractedEntity.document_id == document_id,
                    ExtractedEntity.type.in_(kinds),
                )
            )
            if not entities:
                return []
            ids = await session.scalars(
                insert(ExtractedEntity)
                .values(
                    [
                        {
                            "document_id": e.document_id,
                            "type": e.type,
                            "value_json": e.value,
                            "chunk_id": e.chunk_id,
                            "confidence": e.confidence,
                            "review_run_id": self.run_id,
                            "quote": e.quote,
                            "char_start": e.char_start,
                            "char_end": e.char_end,
                            "page_number": e.page_number,
                            "method": e.method,
                            "prompt_version": e.prompt_version,
                            "verified": True,
                        }
                        for e in entities
                    ]
                )
                .returning(ExtractedEntity.id)
            )
            return list(ids)

    async def load_entities(self, types: list[str]) -> list[ExtractedEntity]:
        async with self.sessions() as session:
            return list(
                await session.scalars(
                    select(ExtractedEntity).where(
                        ExtractedEntity.review_run_id == self.run_id,
                        ExtractedEntity.type.in_(types),
                    )
                )
            )

    # ------------------------------------------------------------------ memo

    async def memo_get(self, step_key: str) -> dict[str, Any] | None:
        async with self.sessions() as session:
            return await session.scalar(
                select(StepResult.output_json).where(
                    StepResult.run_id == self.run_id, StepResult.step_key == step_key
                )
            )

    async def memo_put(self, step_key: str, agent: str, output: dict[str, Any]) -> None:
        async with self.sessions() as session, session.begin():
            await self._fence(session)
            await session.execute(
                insert(StepResult)
                .values(run_id=self.run_id, step_key=step_key, agent=agent, output_json=output)
                .on_conflict_do_nothing()
            )

    # ------------------------------------------------------------------ budget ledger

    async def reserve(
        self,
        *,
        step_key: str,
        agent: str,
        provider: str,
        model: str,
        is_mock: bool,
        prompt_version: str,
        tokens: int,
        microusd: int,
    ) -> tuple[uuid.UUID, int]:
        """Durably reserve one attempt before the provider request. Serialised per run by
        the run row lock, so concurrent reservations see each other."""
        async with self.sessions() as session, session.begin():
            run = await self._fence(session)
            if run.status != "running" or run.control == "cancel":
                raise RunStoppedError(f"run is {run.status} (control={run.control})")
            usage = await run_usage(session, self.run_id)
            tokens_left = run.budget_tokens - usage.tokens
            micro_left = run.budget_microusd - usage.microusd
            if tokens > tokens_left or microusd > micro_left:
                raise BudgetExceededError(
                    f"reserving {tokens} tokens / {microusd} micro-USD would exceed the run "
                    f"budget ({max(tokens_left, 0)} tokens / {max(micro_left, 0)} micro-USD left)",
                    tokens_left=tokens_left,
                    microusd_left=micro_left,
                )
            attempt = (
                await session.scalar(
                    select(func.coalesce(func.max(LlmCall.attempt), 0)).where(
                        LlmCall.run_id == self.run_id, LlmCall.step_key == step_key
                    )
                )
                or 0
            ) + 1
            call_id = await session.scalar(
                insert(LlmCall)
                .values(
                    run_id=self.run_id,
                    step_key=step_key,
                    attempt=attempt,
                    agent=agent,
                    provider=provider,
                    model=model,
                    is_mock=is_mock,
                    prompt_version=prompt_version,
                    status="reserved",
                    reserved_tokens=tokens,
                    reserved_microusd=microusd,
                    usage_source="none",
                )
                .returning(LlmCall.id)
            )
            assert call_id is not None  # noqa: S101
            return call_id, attempt

    async def settle(
        self,
        call_id: uuid.UUID,
        *,
        status: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_write_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cost_microusd: int | None = None,
        usage_source: str = "none",
        latency_ms: int | None = None,
        error_code: str | None = None,
        memo: tuple[str, str, dict[str, Any]] | None = None,
        fenced: bool = True,
    ) -> None:
        """Record an attempt's outcome (and, on success, its memoized result in the same
        transaction). ``fenced=False`` is used only to mark an attempt ``unknown`` while
        shutting down; it can only make the ledger more conservative."""
        async with self.sessions() as session, session.begin():
            if fenced:
                await self._fence(session)
            await session.execute(
                update(LlmCall)
                .where(LlmCall.id == call_id, LlmCall.status == "reserved")
                .values(
                    status=status,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_write_tokens=cache_write_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cost_microusd=cost_microusd,
                    usage_source=usage_source,
                    latency_ms=latency_ms,
                    error_code=error_code,
                    settled_at=func.now(),
                )
            )
            if memo is not None:
                step_key, agent, output = memo
                await session.execute(
                    insert(StepResult)
                    .values(run_id=self.run_id, step_key=step_key, agent=agent, output_json=output)
                    .on_conflict_do_nothing()
                )

    async def merge_totals(self, values: dict[str, Any]) -> None:
        async with self.sessions() as session, session.begin():
            run = await self._fence(session)
            run.totals_json = {**run.totals_json, **values}

    async def run_row(self) -> ReviewRun:
        async with self.sessions() as session:
            run = await session.get(ReviewRun, self.run_id)
            if run is None:
                raise LeaseLostError("run deleted")
            return run
