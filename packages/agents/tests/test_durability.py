"""Durability, control and budget: failing branch + resume without re-billing, duplicate
jobs, lease fencing, pause/resume, cancel of in-flight provider work, budget caps under
parallelism (exact micro-dollar accounting), unknown and usage-less attempts, the
reservation invariant, and sanitized errors."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import Counter
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import func, select, update

from compliance_agents.store import BudgetExceededError, RunStore
from compliance_core.db.models import AgentEvent, Finding, LlmCall, ReviewReport, ReviewRun
from compliance_core.services import reviews
from compliance_core.services.budget import run_usage
from compliance_llm import (
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    LLMUsage,
    MockProvider,
    Price,
)
from compliance_seed.synthetic import build

pytestmark = pytest.mark.integration
SECRET = "sk-ant-SENTINEL-do-not-leak-123"
SMALL = ("vendor_register.xlsx", "acme_nda.pdf", "globex_nda.docx")


def _small() -> dict[str, bytes]:
    files = build()
    return {k: v for k, v in files.items() if k in SMALL}


class Scripted(MockProvider):
    """The deterministic mock with hooks: ``on_call(n, request)`` may raise, block or
    return a replacement response."""

    def __init__(self, on_call: Any = None, *, price: Price | None = None, name: str = "mock"):
        super().__init__()
        self.calls = 0
        self.on_call = on_call
        if price is not None:
            self.price = price
            self.is_mock = False
            self.name = name

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self.on_call is not None:
            replacement = await self.on_call(self.calls, request)
            if replacement is not None:
                return replacement  # type: ignore[no-any-return]
        return await super().complete(request)


async def _ledger(env: Any, run_id: uuid.UUID) -> list[LlmCall]:
    async with env.sessions() as session:
        return list(await session.scalars(select(LlmCall).where(LlmCall.run_id == run_id)))


async def _set_control(env: Any, run_id: uuid.UUID, control: str) -> None:
    async with env.sessions() as session, session.begin():
        await session.execute(
            update(ReviewRun).where(ReviewRun.id == run_id).values(control=control)
        )


async def test_failing_branch_then_resume_without_rebilling(
    review_env: Any, caplog: pytest.LogCaptureFixture
) -> None:
    env = await review_env(_small())
    run_id = await env.start(max_concurrency=8)

    async def boom(n: int, request: LLMRequest) -> None:
        if request.task == "rule_verdict" and "may disclose" in request.user:
            raise RuntimeError(f"connection reset; Authorization: Bearer {SECRET}")

    caplog.set_level(logging.INFO)
    assert await env.execute(run_id, provider=Scripted(boom)) == "failed"
    run = await env.run(run_id)
    assert run.status == "failed"
    assert run.error_code == "agent_error"
    assert SECRET not in (run.error_message or "")
    assert "RuntimeError" in (run.error_message or "")
    assert SECRET not in caplog.text
    async with env.sessions() as session:
        payloads = [e.payload_json for e in await session.scalars(select(AgentEvent))]
    assert SECRET not in str(payloads)
    first = await _ledger(env, run_id)
    succeeded_before = {c.step_key for c in first if c.status == "succeeded"}
    assert succeeded_before
    assert any(c.status == "unknown" and c.error_code == "provider_exception" for c in first)

    async with env.sessions() as session:
        resumed, enqueue = await reviews.control_run(session, run_id, env.user_id, "resume")
    assert enqueue
    assert resumed.status == "queued"
    resume_provider = Scripted()
    assert await env.execute(run_id, provider=resume_provider) == "completed"

    calls = await _ledger(env, run_id)
    per_step = Counter(c.step_key for c in calls if c.status == "succeeded")
    assert all(n == 1 for n in per_step.values()), per_step  # nothing billed twice
    assert succeeded_before <= set(per_step)
    new_steps = set(per_step) - succeeded_before
    assert resume_provider.calls == len(new_steps)  # only unfinished steps were requested
    tasks = (await env.client.get(f"/api/v1/reviews/{run_id}/tasks")).json()["tasks"]
    async with env.sessions() as session:
        findings = await session.scalar(
            select(func.count()).select_from(Finding).where(Finding.run_id == run_id)
        )
        report = await session.get(ReviewReport, run_id)
    assert findings == len(tasks)  # every branch finished, each exactly once
    assert report is not None
    assert len(report.report_json["findings"]) == len(tasks)


async def test_duplicate_jobs_run_once(review_env: Any) -> None:
    env = await review_env(_small())
    run_id = await env.start()
    outcomes = await asyncio.gather(
        env.execute(run_id, worker_id="worker-a"), env.execute(run_id, worker_id="worker-b")
    )
    assert sorted(outcomes) == ["completed", "skipped"]
    assert await env.execute(run_id) == "skipped"  # a late duplicate after completion
    calls = await _ledger(env, run_id)
    assert all(n == 1 for n in Counter(c.step_key for c in calls).values())


async def test_lost_lease_stops_all_writes(review_env: Any) -> None:
    env = await review_env(_small())
    run_id = await env.start(max_concurrency=1)
    thief = uuid.uuid4()

    async def steal(n: int, request: LLMRequest) -> None:
        if n == 1:
            async with env.sessions() as session, session.begin():
                await session.execute(
                    update(ReviewRun).where(ReviewRun.id == run_id).values(lease_token=thief)
                )

    assert await env.execute(run_id, provider=Scripted(steal)) == "lease_lost"
    run = await env.run(run_id)
    assert run.lease_token == thief  # the new owner's lease is untouched
    assert run.status == "running"
    async with env.sessions() as session:
        finished = await session.scalar(
            select(func.count())
            .select_from(AgentEvent)
            .where(AgentEvent.run_id == run_id, AgentEvent.event_type == "run_completed")
        )
    assert finished == 0


async def test_killed_worker_is_recovered_promptly_under_fencing(
    review_env: Any, redis_client: Any
) -> None:
    """A worker killed mid-step keeps its lease and arq's in-progress lock. Once the lease
    expires, the sweeper's recovery job must finish the run without waiting for arq's job
    timeout, and the dead holder's token must no longer be able to write."""
    from arq.connections import ArqRedis
    from arq.constants import in_progress_key_prefix
    from arq.jobs import Job, JobStatus

    from compliance_agents.runner import renew_lease
    from compliance_core.jobs import QUEUE_NAME, RUN_REVIEW, review_job_id
    from compliance_worker.reviews import requeue_stale_reviews, run_review

    env = await review_env(_small())
    run_id = await env.start(max_concurrency=1)
    generation = (await env.run(run_id)).job_generation
    redis = ArqRedis(connection_pool=redis_client.connection_pool)
    original = review_job_id(run_id, generation)
    await redis.enqueue_job(
        RUN_REVIEW, str(run_id), generation, _job_id=original, _queue_name=QUEUE_NAME
    )
    await redis.set(in_progress_key_prefix + original, b"1", px=3_600_000)

    stuck = asyncio.Event()

    async def freeze(n: int, request: LLMRequest) -> None:
        if n == 2:  # one step succeeded; the worker dies inside the next request
            stuck.set()
            await asyncio.sleep(3600)

    killed = asyncio.create_task(env.execute(run_id, provider=Scripted(freeze), worker_id="dead"))
    await asyncio.wait_for(stuck.wait(), timeout=60)
    dead_token = (await env.run(run_id)).lease_token
    assert dead_token is not None
    killed.cancel()  # abrupt stop: no finish, the lease and the arq lock stay behind
    with pytest.raises(asyncio.CancelledError):
        await killed
    run = await env.run(run_id)
    assert (run.status, run.lease_token) == ("running", dead_token)
    # Time passes beyond the lease (the dead worker no longer heartbeats).
    async with env.sessions() as session, session.begin():
        await session.execute(
            update(ReviewRun)
            .where(ReviewRun.id == run_id)
            .values(lease_expires_at=func.now() - timedelta(seconds=1))
        )

    deps = env.deps(provider=Scripted(), worker_id="recovery")
    ctx = {"review": deps, "redis": redis}
    assert await requeue_stale_reviews(ctx) == {"requeued": 1}
    assert await Job(original, redis, _queue_name=QUEUE_NAME).status() == JobStatus.in_progress
    sweep = [
        k.decode().removeprefix("arq:job:")
        async for k in redis_client.scan_iter(f"arq:job:{original}:sweep*")
    ]
    assert len(sweep) == 1
    info = await Job(sweep[0], redis, _queue_name=QUEUE_NAME).info()
    assert info is not None
    assert await run_review(ctx, *info.args) == {"status": "completed"}

    run = await env.run(run_id)
    assert run.status == "completed"
    assert not await renew_lease(env.sessions, run_id, dead_token, 30)  # fenced out
    calls = await _ledger(env, run_id)
    assert all(c.status != "reserved" for c in calls)
    assert any(c.status == "unknown" for c in calls)  # the request the dead worker made
    assert all(
        n == 1 for n in Counter(c.step_key for c in calls if c.status == "succeeded").values()
    )
    async with env.sessions() as session:
        recovered = await session.scalar(
            select(func.count())
            .select_from(AgentEvent)
            .where(AgentEvent.run_id == run_id, AgentEvent.event_type == "run_resumed")
        )
    assert recovered == 1


async def test_pause_at_step_boundary_then_resume(review_env: Any) -> None:
    env = await review_env(_small())
    run_id = await env.start()

    async def pause_once(n: int, request: LLMRequest) -> None:
        if n == 1:
            await _set_control(env, run_id, "pause")

    assert await env.execute(run_id, provider=Scripted(pause_once)) == "paused"
    run = await env.run(run_id)
    assert run.status == "paused"
    assert run.control is None
    paused_calls = await _ledger(env, run_id)
    assert {c.status for c in paused_calls} <= {"succeeded"}  # in-flight work finished
    async with env.sessions() as session:
        _, enqueue = await reviews.control_run(session, run_id, env.user_id, "resume")
    assert enqueue
    assert await env.execute(run_id) == "completed"
    calls = await _ledger(env, run_id)
    assert all(n == 1 for n in Counter(c.step_key for c in calls).values())


async def test_cancel_interrupts_inflight_provider_calls(review_env: Any) -> None:
    env = await review_env(_small())
    run_id = await env.start(max_concurrency=4)
    started = asyncio.Event()
    cancelled: list[int] = []

    async def hang(n: int, request: LLMRequest) -> None:
        if n == 1:
            await _set_control(env, run_id, "cancel")
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.append(n)
            raise

    provider = Scripted(hang)
    outcome = await asyncio.wait_for(env.execute(run_id, provider=provider), timeout=60)
    assert outcome == "cancelled"
    assert started.is_set()
    assert cancelled
    calls_at_return = provider.calls
    await asyncio.sleep(1.5)
    assert provider.calls == calls_at_return  # no branch keeps working after the job ends
    run = await env.run(run_id)
    assert run.status == "cancelled"
    assert run.lease_token is None
    ledger = await _ledger(env, run_id)
    assert all(c.status != "reserved" for c in ledger)
    assert any(c.status == "unknown" and c.error_code == "cancelled" for c in ledger)


async def test_parallel_reservations_never_exceed_the_cap(review_env: Any) -> None:
    env = await review_env({"vendor_register.xlsx": build()["vendor_register.xlsx"]})
    run_id = await env.start(budget_tokens=10_000, budget_usd=0.000_010)  # 10 micro-USD
    async with env.sessions() as session, session.begin():
        await session.execute(
            update(ReviewRun)
            .where(ReviewRun.id == run_id)
            .values(status="running", lease_token=(token := uuid.uuid4()))
        )
    store = RunStore(env.sessions, run_id, token)

    async def reserve(i: int) -> bool:
        try:
            await store.reserve(
                step_key=f"s{i}",
                agent="t",
                provider="p",
                model="m",
                is_mock=False,
                prompt_version="v",
                tokens=100,
                microusd=3,
            )
        except BudgetExceededError:
            return False
        return True

    granted = await asyncio.gather(*(reserve(i) for i in range(20)))
    assert sum(granted) == 3  # 3 x 3 micro-USD fit under 10; the 4th would be 12
    async with env.sessions() as session:
        usage = await run_usage(session, run_id)
    assert usage.microusd == 9
    assert usage.tokens == 300


async def test_budget_cap_with_fractional_prices_under_parallelism(review_env: Any) -> None:
    env = await review_env(_small())
    price = Price(0.37, 1.13, "test fractional", 0.41, 0.05)
    run_id = await env.start(
        provider="mock", budget_usd=0.006, max_concurrency=8, price=(0.37, 1.13)
    )

    async def priced(n: int, request: LLMRequest) -> LLMResponse:
        base = await MockProvider().complete(request)
        return LLMResponse(
            data=base.data,
            usage=LLMUsage(
                input_tokens=333,
                output_tokens=77,
                source="provider",
                cache_write_tokens=5,
                cache_read_tokens=7,
            ),
            model="priced-test",
            latency_ms=1,
        )

    assert (
        await env.execute(run_id, provider=Scripted(priced, price=price, name="anthropic"))
        == "completed"
    )
    run = await env.run(run_id)
    calls = await _ledger(env, run_id)
    async with env.sessions() as session:
        usage = await run_usage(session, run_id)
    assert usage.microusd <= run.budget_microusd  # the cap holds exactly
    measured = [c for c in calls if c.status == "succeeded"]
    assert measured
    for c in measured:
        # 333*0.37 + 77*1.13 + 5*0.41 + 7*0.05 = 212.62 micro-USD, charged as 213 (rounded up)
        assert c.cost_microusd == 213
    findings = await env.findings(run_id)
    skipped = [f for f in findings.values() if "budget is exhausted" in f["rationale"]]
    assert skipped  # some LLM checks did not fit and need review
    assert all(f["status"] == "needs_review" for f in skipped)
    assert run.totals_json.get("budget_exhausted") is True


async def test_unknown_and_usageless_attempts_keep_their_reservation(review_env: Any) -> None:
    env = await review_env(_small())
    run_id = await env.start(documents=["acme_nda.pdf"], pack_key="vendor_confidentiality_review")

    async def flaky(n: int, request: LLMRequest) -> LLMResponse | None:
        if n == 1:
            raise LLMTransientError("read timeout", request_sent=True)
        base = await MockProvider().complete(request)
        return LLMResponse(data=base.data, usage=None, model="m", latency_ms=1)

    assert await env.execute(run_id, provider=Scripted(flaky)) == "completed"
    calls = await _ledger(env, run_id)
    unknown = [c for c in calls if c.status == "unknown"]
    usageless = [c for c in calls if c.status == "succeeded" and c.usage_source == "none"]
    assert len(unknown) == 1
    assert usageless
    async with env.sessions() as session:
        usage = await run_usage(session, run_id)
    held = sum(c.reserved_tokens for c in unknown + usageless)
    assert usage.tokens == held  # nothing counted as free


async def test_usage_above_reservation_stops_the_run(review_env: Any) -> None:
    env = await review_env(_small())
    run_id = await env.start(documents=["acme_nda.pdf"])

    async def overrun(n: int, request: LLMRequest) -> LLMResponse:
        base = await MockProvider().complete(request)
        return LLMResponse(
            data=base.data,
            usage=LLMUsage(input_tokens=10_000_000, output_tokens=1, source="provider"),
            model="m",
            latency_ms=1,
        )

    assert await env.execute(run_id, provider=Scripted(overrun)) == "failed"
    run = await env.run(run_id)
    assert run.error_code == "budget_invariant"
