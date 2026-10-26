"""Execute (or resume) one review run inside a worker job.

1. **Lease.** The job atomically claims a ``queued`` run (or a ``running`` run whose lease
   expired after a crash) with a fresh token; a duplicate job finds nothing to claim and
   exits. A heartbeat renews the lease; if renewal fails the job stops. Every write is
   fenced on the token.
2. **Ledger hygiene.** Reservations left ``reserved`` by a crashed attempt become
   ``unknown``: they may have been billed, so they stay charged.
3. **Graph.** The LangGraph graph runs with the Postgres checkpointer: a fresh run starts
   from its input, a paused or failed run resumes from its last checkpoint (completed
   tasks are restored, memoized steps are not re-billed), and a run waiting for human
   review resumes the interrupt.
4. **Control.** ``pause`` stops at the next super-step boundary (in-flight tasks finish);
   ``cancel`` stops at once (new provider reservations are refused).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

from langgraph.types import Command
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from compliance_agents.checkpoint import PostgresCheckpointSaver
from compliance_agents.context import RunContext
from compliance_agents.gateway import BudgetInvariantError, LLMGateway, LLMStepError
from compliance_agents.graph import build_graph
from compliance_agents.model import (
    MockRuleCheckModel,
    ModelUnavailableError,
    RuleCheckModel,
    ServiceRuleCheckModel,
)
from compliance_agents.store import RunStoppedError, RunStore
from compliance_core.config import Settings
from compliance_core.db.models import LlmCall, ReviewRun
from compliance_core.logs import run_id_var
from compliance_core.services.review_events import LeaseLostError, append_event
from compliance_llm import LLMConfigurationError, LLMProvider, build_provider, resolve_choice
from compliance_retrieval.service import SearchDependencies

logger = logging.getLogger("compliance_agents.runner")
RECURSION_LIMIT = 80
CONTROL_POLL_SECONDS = 1.0


@dataclass
class RunnerDeps:
    settings: Settings
    sessions: async_sessionmaker[AsyncSession]
    search: SearchDependencies | None
    worker_id: str
    # Test seams: build the provider / model for a run (defaults: from settings).
    provider_factory: Callable[[Settings, ReviewRun], LLMProvider] | None = None
    model_factory: Callable[[Settings, ReviewRun], RuleCheckModel] | None = None


def default_provider(settings: Settings, run: ReviewRun) -> LLMProvider:
    return build_provider(settings, resolve_choice(settings, run.llm_provider, run.llm_model))


def default_model(settings: Settings, run: ReviewRun) -> RuleCheckModel:
    if run.config_json.get("rule_check_backend") == "mock":
        return MockRuleCheckModel()
    return ServiceRuleCheckModel(
        settings.model_service_url, timeout=settings.review_model_timeout_seconds
    )


async def acquire_lease(
    sessions: async_sessionmaker[AsyncSession], run_id: uuid.UUID, worker_id: str, seconds: int
) -> uuid.UUID | None:
    token = uuid.uuid4()
    async with sessions() as session, session.begin():
        claimed = await session.scalar(
            update(ReviewRun)
            .where(
                ReviewRun.id == run_id,
                (ReviewRun.status == "queued")
                | (
                    (ReviewRun.status == "running")
                    & (
                        ReviewRun.lease_expires_at.is_(None)
                        | (ReviewRun.lease_expires_at < func.now())
                    )
                ),
            )
            .values(
                status="running",
                lease_token=token,
                lease_owner=worker_id[:200],
                lease_expires_at=func.now() + timedelta(seconds=seconds),
                started_at=func.coalesce(ReviewRun.started_at, func.now()),
                updated_at=func.now(),
            )
            .returning(ReviewRun.id)
        )
        if claimed is None:
            return None
        # Attempts reserved by a crashed worker may have reached the provider.
        await session.execute(
            update(LlmCall)
            .where(LlmCall.run_id == run_id, LlmCall.status == "reserved")
            .values(status="unknown", error_code="worker_lost", settled_at=func.now())
        )
    return token


async def renew_lease(
    sessions: async_sessionmaker[AsyncSession], run_id: uuid.UUID, token: uuid.UUID, seconds: int
) -> bool:
    async with sessions() as session, session.begin():
        renewed = await session.scalar(
            update(ReviewRun)
            .where(ReviewRun.id == run_id, ReviewRun.lease_token == token)
            .values(lease_expires_at=func.now() + timedelta(seconds=seconds))
            .returning(ReviewRun.id)
        )
    return renewed is not None


async def _finish(
    sessions: async_sessionmaker[AsyncSession],
    run_id: uuid.UUID,
    token: uuid.UUID,
    status: str,
    event: str,
    payload: dict[str, Any],
    *,
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    """Set the final status for this job and release the lease (fenced)."""
    values: dict[str, Any] = {
        "status": status,
        "lease_token": None,
        "lease_owner": None,
        "lease_expires_at": None,
        "updated_at": func.now(),
        "error_code": error_code,
        "error_message": error_message,
    }
    if status in ("completed", "cancelled"):
        values["finished_at"] = func.now()
    if status in ("paused", "cancelled", "completed"):
        values["control"] = None
    async with sessions() as session, session.begin():
        locked = await session.scalar(
            select(ReviewRun.lease_token).where(ReviewRun.id == run_id).with_for_update()
        )
        if locked != token:
            raise LeaseLostError("lease lost before finishing")
        await append_event(session, run_id, "system", event, payload, lease_token=token)
        await session.execute(
            update(ReviewRun)
            .where(ReviewRun.id == run_id, ReviewRun.lease_token == token)
            .values(**values)
        )


async def _control(sessions: async_sessionmaker[AsyncSession], run_id: uuid.UUID) -> str | None:
    async with sessions() as session:
        return await session.scalar(select(ReviewRun.control).where(ReviewRun.id == run_id))


# Exceptions whose messages we write ourselves (ids, counts, codes; never document text
# or credentials). Everything else is reported by type and code location only: driver and
# library messages can carry SQL parameters (document text) or request details.
_OWN_MESSAGES = (BudgetInvariantError, LLMConfigurationError, ModelUnavailableError, LLMStepError)


def safe_error(exc: BaseException) -> tuple[str, str]:
    """(code, message) safe to persist, stream and log."""
    if isinstance(exc, BudgetInvariantError):
        return "budget_invariant", str(exc)[:300]
    if isinstance(exc, _OWN_MESSAGES):
        return "agent_error", f"{type(exc).__name__}: {str(exc)[:300]}"
    return "agent_error", f"{type(exc).__name__} (details withheld; see code location in logs)"


def _where(exc: BaseException) -> list[str]:
    """Code locations of the last frames, without any values."""
    frames = traceback.extract_tb(exc.__traceback__)[-6:]
    return [f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}:{f.name}" for f in frames]


async def execute_run(deps: RunnerDeps, run_id: uuid.UUID) -> str:
    """Run or resume ``run_id``. Returns the outcome (``skipped`` if not claimable)."""
    settings = deps.settings
    token = await acquire_lease(
        deps.sessions, run_id, deps.worker_id, settings.review_lease_seconds
    )
    if token is None:
        return "skipped"
    run_token = run_id_var.set(str(run_id))
    store = RunStore(deps.sessions, run_id, token)
    main = asyncio.current_task()
    lost = asyncio.Event()

    async def heartbeat() -> None:
        interval = max(settings.review_lease_seconds / 3, 1)
        while True:
            await asyncio.sleep(interval)
            try:
                ok = await renew_lease(deps.sessions, run_id, token, settings.review_lease_seconds)
            except Exception:  # transient DB error: try again next beat
                logger.warning("lease renewal failed", exc_info=True)
                continue
            if not ok:
                lost.set()
                if main is not None:
                    main.cancel()
                return

    beat = asyncio.create_task(heartbeat())
    provider: LLMProvider | None = None
    model: RuleCheckModel | None = None
    try:
        run = await store.run_row()
        try:
            provider = (deps.provider_factory or default_provider)(settings, run)
        except LLMConfigurationError as exc:  # our own message: names the missing setting
            await _finish(
                deps.sessions,
                run_id,
                token,
                "failed",
                "run_failed",
                {"code": "provider_unavailable", "message": str(exc)},
                error_code="provider_unavailable",
                error_message=str(exc),
            )
            return "failed"
        model = (deps.model_factory or default_model)(settings, run)
        llm = LLMGateway(store, provider, max_attempts=settings.llm_max_attempts)
        ctx = RunContext(
            settings=settings,
            sessions=deps.sessions,
            run_id=run_id,
            workspace_id=run.workspace_id,
            store=store,
            provider=provider,
            llm=llm,
            model=model,
            search=deps.search,
            spec=run.rule_pack_snapshot["spec"],
            rule_ids=run.rule_pack_snapshot.get("rule_ids", {}),
            config=run.config_json,
        )
        graph = build_graph(ctx, PostgresCheckpointSaver(deps.sessions, lease_token=token))
        config: dict[str, Any] = {
            "configurable": {"thread_id": str(run_id)},
            "max_concurrency": int(run.config_json.get("max_concurrency", 4)),
            "recursion_limit": RECURSION_LIMIT,
        }
        snapshot = await graph.aget_state(config)  # type: ignore[arg-type]
        interrupted = any(task.interrupts for task in snapshot.tasks)
        graph_input: Any
        if interrupted:
            graph_input = Command(resume={"resumed": True})
            phase = "resumed_after_review"
        elif snapshot.created_at is None:
            graph_input = {"run_id": str(run_id), "workspace_id": str(run.workspace_id)}
            phase = "started"
        else:
            graph_input = None
            phase = "resumed"
        await store.emit(
            "system",
            f"run_{phase}",
            {
                "worker": deps.worker_id,
                "llm_provider": provider.name,
                "llm_model": provider.model,
                "mock": provider.is_mock,
                "rule_check_backend": model.backend,
            },
        )
        stop: str | None = None

        async def consume() -> str | None:
            stream: Any = graph.astream(
                graph_input,
                cast(Any, config),
                stream_mode=["updates", "checkpoints"],
                durability="sync",
            )
            try:
                async for mode, _chunk in stream:
                    control = await _control(deps.sessions, run_id)
                    if control == "cancel":
                        return "cancel"
                    if control == "pause" and mode == "checkpoints":
                        return "pause"  # a super-step boundary: nothing is in flight
                return None
            finally:
                # Close the stream explicitly: LangGraph cancels any branch still running,
                # so no provider work continues after this job lets go of the lease.
                await stream.aclose()

        async def watch(task: asyncio.Task[str | None]) -> None:
            # Cancel must also interrupt branches that are waiting on a slow provider.
            while not task.done():
                await asyncio.sleep(CONTROL_POLL_SECONDS)
                if await _control(deps.sessions, run_id) == "cancel":
                    task.cancel()
                    return

        consumer = asyncio.create_task(consume())
        watcher = asyncio.create_task(watch(consumer))
        try:
            stop = await consumer
        except asyncio.CancelledError:
            if (
                consumer.cancelled()
                and not lost.is_set()
                and (main is None or not main.cancelling())
            ):
                stop = "cancel"
            else:
                raise
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
        after = await graph.aget_state(config)  # type: ignore[arg-type]
        if stop == "cancel":
            await _finish(deps.sessions, run_id, token, "cancelled", "run_cancelled", {})
            return "cancelled"
        if stop == "pause":
            await _finish(
                deps.sessions, run_id, token, "paused", "run_paused", {"next": list(after.next)}
            )
            return "paused"
        if any(task.interrupts for task in after.tasks):
            pending = [i.value for t in after.tasks for i in t.interrupts]
            await _finish(
                deps.sessions,
                run_id,
                token,
                "awaiting_review",
                "awaiting_review",
                {"interrupts": pending[:3]},
            )
            return "awaiting_review"
        if after.next:
            await _finish(
                deps.sessions,
                run_id,
                token,
                "failed",
                "run_failed",
                {"code": "incomplete", "next": list(after.next)},
                error_code="incomplete",
                error_message="The graph stopped before finishing.",
            )
            return "failed"
        await _finish(deps.sessions, run_id, token, "completed", "run_completed", {})
        return "completed"
    except asyncio.CancelledError:
        if lost.is_set():
            logger.warning("review run lease lost; stopping", extra={"run_id": str(run_id)})
            return "lease_lost"
        raise
    except LeaseLostError:
        logger.warning("review run lease lost; stopping", extra={"run_id": str(run_id)})
        return "lease_lost"
    except RunStoppedError:
        control = await _control(deps.sessions, run_id)
        status = "cancelled" if control == "cancel" else "paused"
        with contextlib.suppress(LeaseLostError):
            await _finish(deps.sessions, run_id, token, status, f"run_{status}", {})
        return status
    except Exception as exc:
        code, message = safe_error(exc)
        logger.error(  # no exc_info: tracebacks can contain bound document text
            "review run failed",
            extra={
                "run_id": str(run_id),
                "error_type": type(exc).__name__,
                "error_code": code,
                "where": _where(exc),
            },
        )
        with contextlib.suppress(LeaseLostError):
            await _finish(
                deps.sessions,
                run_id,
                token,
                "failed",
                "run_failed",
                {"code": code, "message": message},
                error_code=code,
                error_message=message,
            )
        return "failed"
    finally:
        beat.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await beat
        if provider is not None:
            with contextlib.suppress(Exception):
                await provider.aclose()
        if model is not None:
            with contextlib.suppress(Exception):
                await model.aclose()
        run_id_var.reset(run_token)
