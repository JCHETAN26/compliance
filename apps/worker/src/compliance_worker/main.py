"""arq worker entry point.

Run: ``arq compliance_worker.main.WorkerSettings``
Healthcheck: ``arq compliance_worker.main.WorkerSettings --check`` (exits non-zero unless
the worker's periodic heartbeat key is present in Redis).

Phase 0 registers only the ``ping`` diagnostic job, which proves the queue -> worker ->
database path end to end. Ingestion (Phase 1) and review-run (Phase 4) jobs are not
implemented yet.
"""

from __future__ import annotations

import logging
import socket
from typing import Any, ClassVar

from arq.connections import RedisSettings
from sqlalchemy import text

from compliance_core.config import get_settings
from compliance_core.db.session import create_engine
from compliance_core.logs import configure_logging

QUEUE_NAME = "compliance:jobs"
logger = logging.getLogger("compliance_worker")


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging("worker", settings.log_level)
    engine = create_engine(settings)
    # Fail fast if the database is unreachable instead of accepting jobs we cannot run.
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    ctx["engine"] = engine
    logger.info("worker started", extra={"queue": QUEUE_NAME})


async def shutdown(ctx: dict[str, Any]) -> None:
    engine = ctx.get("engine")
    if engine is not None:
        await engine.dispose()


async def ping(ctx: dict[str, Any]) -> dict[str, Any]:
    """Diagnostic job: round-trips through the database and reports the worker host."""
    async with ctx["engine"].connect() as conn:
        db_time = await conn.scalar(text("SELECT now()"))
    return {
        "pong": True,
        "worker_host": socket.gethostname(),
        "job_try": ctx.get("job_try"),
        "database_time": db_time.isoformat() if db_time else None,
    }


class WorkerSettings:
    functions: ClassVar[list[Any]] = [ping]
    on_startup = startup
    on_shutdown = shutdown
    queue_name = QUEUE_NAME
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    health_check_interval = 10
    max_jobs = 10
    job_timeout = 300
