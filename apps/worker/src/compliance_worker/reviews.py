"""Review-run jobs: ``run_review`` executes or resumes one run; a cron sweep re-enqueues
runs whose job was lost (Redis down at start) or whose worker died (expired lease).

Duplicate deliveries are safe: a run can only be executed by the holder of its lease
(see ``compliance_agents.runner``), and completed steps are memoized, so re-execution
never re-bills a provider call that succeeded.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from arq.connections import ArqRedis
from arq.jobs import Job, JobStatus
from redis.exceptions import RedisError
from sqlalchemy import or_, select

from compliance_agents.runner import RunnerDeps, execute_run
from compliance_core.db.models import ReviewRun
from compliance_core.jobs import QUEUE_NAME, RUN_REVIEW, review_job_id

logger = logging.getLogger("compliance_worker.reviews")
QUEUED_REQUEUE_AFTER = timedelta(seconds=60)


async def run_review(ctx: dict[str, Any], run_id: str, generation: int) -> dict[str, Any]:
    deps: RunnerDeps = ctx["review"]
    rid = uuid.UUID(run_id)
    async with deps.sessions() as session:
        current = await session.scalar(select(ReviewRun.job_generation).where(ReviewRun.id == rid))
    if current is None:
        return {"status": "missing"}
    if generation < current:
        return {"status": "stale_generation"}
    started = time.perf_counter()
    outcome = await execute_run(deps, rid)
    logger.info(
        "review job finished",
        extra={
            "run_id": run_id,
            "outcome": outcome,
            "seconds": round(time.perf_counter() - started, 2),
        },
    )
    return {"status": outcome}


async def requeue_stale_reviews(ctx: dict[str, Any]) -> dict[str, int]:
    deps: RunnerDeps = ctx["review"]
    redis: ArqRedis = ctx["redis"]
    now = datetime.now(UTC)
    async with deps.sessions() as session:
        rows = (
            await session.execute(
                select(ReviewRun.id, ReviewRun.job_generation, ReviewRun.status).where(
                    or_(
                        (ReviewRun.status == "queued")
                        & (ReviewRun.updated_at < now - QUEUED_REQUEUE_AFTER),
                        (ReviewRun.status == "running") & (ReviewRun.lease_expires_at < now),
                    )
                )
            )
        ).all()
    requeued = 0
    bucket = str(int(now.timestamp()) // 60)
    for run_id, generation, status in rows:
        try:
            job = Job(review_job_id(run_id, generation), redis, _queue_name=QUEUE_NAME)
            waiting = (JobStatus.queued, JobStatus.deferred)
            # An expired lease means the job holding it stopped heartbeating. A killed
            # worker leaves its ARQ in-progress lock until the job timeout, so only a job
            # still waiting to start counts as pending; the recovery job is fenced by the
            # lease, so a slow original that is somehow alive cannot write after it.
            if await job.status() in (
                waiting if status == "running" else (*waiting, JobStatus.in_progress)
            ):
                continue
            enqueued = await redis.enqueue_job(
                RUN_REVIEW,
                str(run_id),
                generation,
                _job_id=review_job_id(run_id, generation, f"sweep{bucket}"),
                _queue_name=QUEUE_NAME,
            )
            if enqueued is not None:  # None: this minute's recovery job already exists
                requeued += 1
        except RedisError:
            logger.warning("could not re-enqueue review run", extra={"run_id": str(run_id)})
    return {"requeued": requeued}
