"""The review job wrapper: generation checks and the stale-run sweeper."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import update

from compliance_agents.runner import RunnerDeps
from compliance_core.db.models import ReviewRun, User, Workspace, WorkspaceMember, WorkspaceRole
from compliance_worker import reviews as job

pytestmark = pytest.mark.integration


async def _run(api_app: Any, **values: Any) -> uuid.UUID:
    sessions = api_app.state.ctx.session_factory
    async with sessions() as session, session.begin():
        user = User(email=f"{uuid.uuid4().hex}@example.com", password_hash="x")
        session.add(user)
        await session.flush()
        ws = Workspace(name="w", owner_id=user.id)
        session.add(ws)
        await session.flush()
        session.add(WorkspaceMember(workspace_id=ws.id, user_id=user.id, role=WorkspaceRole.OWNER))
        run = ReviewRun(
            workspace_id=ws.id,
            rule_pack_key="k",
            rule_pack_version="1.0.0",
            rule_pack_name="n",
            rule_pack_snapshot={"spec": {"rules": []}, "rule_ids": {}},
            rule_pack_sha256="0" * 64,
            status="queued",
            llm_provider="mock",
            llm_model="mock-deterministic-v1",
            llm_is_mock=True,
            config_json={"document_ids": []},
            budget_tokens=1000,
            budget_microusd=0,
            **values,
        )
        session.add(run)
        await session.flush()
        return run.id


def _deps(api_app: Any, test_settings: Any) -> RunnerDeps:
    return RunnerDeps(
        settings=test_settings,
        sessions=api_app.state.ctx.session_factory,
        search=None,
        worker_id="t",
    )


async def test_missing_and_stale_generations_are_ignored(
    api_app: Any, clean_state: None, test_settings: Any
) -> None:
    ctx = {"review": _deps(api_app, test_settings)}
    assert await job.run_review(ctx, str(uuid.uuid4()), 1) == {"status": "missing"}
    run_id = await _run(api_app, job_generation=3)
    assert await job.run_review(ctx, str(run_id), 2) == {"status": "stale_generation"}


async def test_sweeper_requeues_lost_and_expired_runs(
    api_app: Any, clean_state: None, test_settings: Any, redis_client: Any
) -> None:
    from arq.connections import ArqRedis

    stale_queued = await _run(api_app)
    expired = await _run(api_app)
    healthy = await _run(api_app)
    sessions = api_app.state.ctx.session_factory
    old = datetime.now(UTC) - timedelta(minutes=10)
    async with sessions() as session, session.begin():
        await session.execute(
            update(ReviewRun).where(ReviewRun.id == stale_queued).values(updated_at=old)
        )
        await session.execute(
            update(ReviewRun)
            .where(ReviewRun.id == expired)
            .values(status="running", lease_expires_at=old)
        )
        await session.execute(
            update(ReviewRun)
            .where(ReviewRun.id == healthy)
            .values(status="running", lease_expires_at=datetime.now(UTC) + timedelta(minutes=5))
        )
    redis = ArqRedis(connection_pool=redis_client.connection_pool)
    result = await job.requeue_stale_reviews(
        {"review": _deps(api_app, test_settings), "redis": redis}
    )
    assert result == {"requeued": 2}
    keys = {k.decode() async for k in redis_client.scan_iter("arq:job:review:*")}
    assert any(stale_queued.hex in k for k in keys)
    assert any(expired.hex in k for k in keys)
    assert not any(healthy.hex in k for k in keys)


async def test_sweeper_recovers_an_expired_lease_despite_a_stale_in_progress_job(
    api_app: Any, clean_state: None, test_settings: Any, redis_client: Any
) -> None:
    """A killed worker leaves arq's in-progress lock for the job timeout; the expired
    run lease alone must be enough to schedule a (fenced) recovery job promptly."""
    from arq.connections import ArqRedis
    from arq.constants import in_progress_key_prefix
    from arq.jobs import Job, JobStatus

    from compliance_core.jobs import QUEUE_NAME, RUN_REVIEW, review_job_id

    killed = await _run(api_app)
    starting = await _run(api_app)
    sessions = api_app.state.ctx.session_factory
    expired = datetime.now(UTC) - timedelta(seconds=5)
    async with sessions() as session, session.begin():
        await session.execute(
            update(ReviewRun)
            .where(ReviewRun.id.in_([killed, starting]))
            .values(status="running", lease_token=uuid.uuid4(), lease_expires_at=expired)
        )
    redis = ArqRedis(connection_pool=redis_client.connection_pool)
    for run_id in (killed, starting):
        await redis.enqueue_job(
            RUN_REVIEW, str(run_id), 1, _job_id=review_job_id(run_id, 1), _queue_name=QUEUE_NAME
        )
    # What arq holds while the killed worker's job "runs": the lock outlives the process.
    original = review_job_id(killed, 1)
    await redis.set(in_progress_key_prefix + original, b"1", px=3_600_000)
    assert await Job(original, redis, _queue_name=QUEUE_NAME).status() == JobStatus.in_progress

    ctx = {"review": _deps(api_app, test_settings), "redis": redis}
    minute = int(datetime.now(UTC).timestamp()) // 60
    assert await job.requeue_stale_reviews(ctx) == {"requeued": 1}
    keys = {k.decode() async for k in redis_client.scan_iter("arq:job:review:*")}
    recovery = [k.removeprefix("arq:job:") for k in keys if killed.hex in k and ":sweep" in k]
    assert len(recovery) == 1
    info = await Job(recovery[0], redis, _queue_name=QUEUE_NAME).info()
    assert info is not None
    assert (info.function, info.args) == (RUN_REVIEW, (str(killed), 1))
    # The job that is merely waiting to start will take the expired lease itself.
    assert not any(starting.hex in k and ":sweep" in k for k in keys)
    again = await job.requeue_stale_reviews(ctx)
    if int(datetime.now(UTC).timestamp()) // 60 == minute:
        assert again == {"requeued": 0}  # one recovery job per run and minute
