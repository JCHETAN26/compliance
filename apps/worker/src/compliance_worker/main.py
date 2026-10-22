"""arq worker entry point.

Run: ``arq compliance_worker.main.WorkerSettings``
Healthcheck: ``arq compliance_worker.main.WorkerSettings --check`` (exits non-zero unless
the worker's periodic heartbeat key is present in Redis).

Jobs:
- ``ping``: diagnostic queue -> worker -> database round trip.
- ``ingest_document``: Phase 1 document ingestion (see ``ingest.py``).
- ``seed_demo``: idempotent demo-workspace seeding (see ``compliance_seed``).
- cron ``requeue_stale_documents`` every minute.

Startup fails, so the worker never reports healthy, if the database is unreachable or
the pinned embedding model cannot be verified and loaded. There is no fallback model.
Review-run jobs (Phase 4) are not implemented yet.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import Any, ClassVar

from arq import cron
from arq.connections import RedisSettings
from sqlalchemy import text

from compliance_core.config import AppEnv, get_settings
from compliance_core.db.session import create_engine, create_session_factory
from compliance_core.jobs import QUEUE_NAME, SEED_DEMO
from compliance_core.logs import configure_logging
from compliance_core.model_client import embedding_label
from compliance_core.storage import LocalStorage
from compliance_ingestion.embeddings import Embedder, ensure_model, spec_from_settings
from compliance_seed.job import demo_workspace_exists, seed_demo
from compliance_worker.ingest import IngestDeps, ingest_document, requeue_stale_documents

logger = logging.getLogger("compliance_worker")


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging("worker", settings.log_level)
    engine = create_engine(settings)
    # Fail fast if the database is unreachable instead of accepting jobs we cannot run.
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    ctx["engine"] = engine
    spec = spec_from_settings()
    model_path = await asyncio.to_thread(
        ensure_model, settings.model_cache_dir, spec, allow_download=settings.artifact_auto_download
    )
    embedder = await asyncio.to_thread(
        Embedder,
        model_path,
        threads=settings.embedding_threads,
        batch_size=settings.embedding_batch_size,
    )
    session_factory = create_session_factory(engine)
    ctx["ingest"] = IngestDeps(
        settings=settings,
        session_factory=session_factory,
        storage=LocalStorage(settings.storage_dir),
        embedder=embedder,
        tokenizer_path=model_path / "tokenizer.json",
        model_label=embedding_label(spec.repo_id, spec.revision),
        embed_lock=asyncio.Lock(),
    )
    logger.info(
        "worker started",
        extra={"queue": QUEUE_NAME, "embedding_model": spec.repo_id, "revision": spec.revision},
    )
    # Automatic demo seeding happens only on a clean start (no demo workspace yet), so
    # deleting demo documents or changing the demo password is never undone by restarts.
    if settings.seed_demo and settings.app_env is not AppEnv.PRODUCTION:
        async with session_factory() as session:
            exists = await demo_workspace_exists(session)
        if not exists:
            await ctx["redis"].enqueue_job(
                SEED_DEMO, _job_id="seed-demo-initial", _queue_name=QUEUE_NAME
            )
            logger.info("queued initial demo seeding")


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


_settings = get_settings()


class WorkerSettings:
    functions: ClassVar[list[Any]] = [ping, ingest_document, seed_demo]
    cron_jobs: ClassVar[list[Any]] = [
        cron(requeue_stale_documents, second=15, run_at_startup=False, unique=True)
    ]
    on_startup = startup
    on_shutdown = shutdown
    queue_name = QUEUE_NAME
    redis_settings = RedisSettings.from_dsn(_settings.redis_url)
    health_check_interval = 10
    max_jobs = _settings.worker_max_jobs
    job_timeout = _settings.ingest_parse_timeout_seconds + 900
    max_tries = _settings.ingest_max_attempts + 2
