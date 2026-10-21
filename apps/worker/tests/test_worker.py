"""Runs the real arq worker against test Redis/Postgres and round-trips the ping job."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from arq import create_pool
from arq.connections import RedisSettings
from arq.worker import Worker, async_check_health

from compliance_core.config import Settings, get_settings
from compliance_core.jobs import QUEUE_NAME
from compliance_worker import main as worker_main

pytestmark = pytest.mark.integration


@pytest.fixture
def worker_env(
    monkeypatch: pytest.MonkeyPatch, test_settings: Settings, migrated_database: str
) -> Iterator[RedisSettings]:
    monkeypatch.setenv("DATABASE_URL", test_settings.database_url)
    monkeypatch.setenv("REDIS_URL", test_settings.redis_url)
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("STORAGE_DIR", str(test_settings.storage_dir))
    monkeypatch.setenv("ARTIFACT_CACHE_DIR", str(test_settings.artifact_cache_dir))
    monkeypatch.setenv("ARTIFACT_AUTO_DOWNLOAD", "false")
    monkeypatch.setenv("SEED_DEMO", "false")
    get_settings.cache_clear()
    yield RedisSettings.from_dsn(test_settings.redis_url)
    get_settings.cache_clear()


def _worker(redis_settings: RedisSettings, **kwargs: object) -> Worker:
    return Worker(
        functions=worker_main.WorkerSettings.functions,
        on_startup=worker_main.startup,
        on_shutdown=worker_main.shutdown,
        queue_name=QUEUE_NAME,
        redis_settings=redis_settings,
        health_check_interval=1,
        handle_signals=False,
        **kwargs,  # type: ignore[arg-type]
    )


async def test_ping_job_round_trip_and_heartbeat(worker_env: RedisSettings) -> None:
    pool = await create_pool(worker_env)
    try:
        await pool.flushdb()
        assert await async_check_health(worker_env, queue_name=QUEUE_NAME) == 1, (
            "health check must fail when no worker is running"
        )

        job = await pool.enqueue_job("ping", _queue_name=QUEUE_NAME)
        assert job is not None
        worker = _worker(worker_env)
        task = asyncio.create_task(worker.async_run())
        try:
            result = await job.result(timeout=15, poll_delay=0.1)
            assert result["pong"] is True
            assert result["database_time"]
            assert result["job_try"] == 1
            async with asyncio.timeout(5):
                # Polls state owned by another component (Redis key); no Event to await.
                while await async_check_health(worker_env, queue_name=QUEUE_NAME):  # noqa: ASYNC110
                    await asyncio.sleep(0.2)
        finally:
            await worker.close()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert await async_check_health(worker_env, queue_name=QUEUE_NAME) == 1, (
            "heartbeat key must be removed when the worker stops"
        )
    finally:
        await pool.aclose()


async def test_worker_startup_fails_without_database(
    monkeypatch: pytest.MonkeyPatch, worker_env: RedisSettings
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://nobody:x@127.0.0.1:9/none")
    get_settings.cache_clear()
    with pytest.raises(OSError, match=r"Connect call failed|refused"):
        await worker_main.startup({})


def test_only_implemented_jobs_are_registered() -> None:
    names = [getattr(f, "__name__", None) for f in worker_main.WorkerSettings.functions]
    assert names == ["ping", "ingest_document", "seed_demo"]
