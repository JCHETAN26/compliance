"""Process-wide resources shared by every transport that calls the service layer.

The REST API (``compliance_api.context.AppContext``) and the MCP server both build on
:class:`ServiceContext`, so uploads, searches and review starts use the same limiters,
storage, model client and job queue whichever transport received the request.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from compliance_core.config import Settings
from compliance_core.db.session import create_engine, create_session_factory
from compliance_core.jobs import (
    INGEST_DOCUMENT,
    QUEUE_NAME,
    RUN_REVIEW,
    ingest_job_id,
    review_job_id,
)
from compliance_core.model_client import ModelServiceClient
from compliance_core.services.session_store import FixedWindowLimiter, UploadRateLimiter
from compliance_core.storage import LocalStorage

logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class ServiceContext:
    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    redis: Redis
    storage: LocalStorage
    models: ModelServiceClient
    upload_limiter: UploadRateLimiter
    search_limiter: FixedWindowLimiter
    review_start_limiter: FixedWindowLimiter
    queue: ArqRedis | None = None

    @staticmethod
    def shared_resources(settings: Settings) -> dict[str, Any]:
        """Keyword arguments for the shared fields (subclasses add their own)."""
        engine = create_engine(settings)
        redis = Redis.from_url(settings.redis_url, socket_timeout=5, socket_connect_timeout=5)
        return {
            "settings": settings,
            "engine": engine,
            "session_factory": create_session_factory(engine),
            "redis": redis,
            "storage": LocalStorage(settings.storage_dir),
            "models": ModelServiceClient.from_settings(settings),
            "upload_limiter": UploadRateLimiter(
                redis, settings.upload_rate_limit_per_minute, settings.upload_requests_per_minute
            ),
            "search_limiter": FixedWindowLimiter(
                redis, "search", settings.search_rate_limit_per_minute, "searches"
            ),
            # One per-user budget for review starts from any transport (REST or MCP).
            "review_start_limiter": FixedWindowLimiter(
                redis, "review-start", settings.review_start_rate_limit_per_minute, "review starts"
            ),
        }

    @classmethod
    def create(cls, settings: Settings) -> ServiceContext:
        return cls(**cls.shared_resources(settings))

    async def _pool(self) -> ArqRedis:
        if self.queue is None:
            redis_settings = RedisSettings.from_dsn(self.settings.redis_url)
            redis_settings.conn_retries = 1  # fail fast; the worker sweeps re-queue
            self.queue = await create_pool(redis_settings, default_queue_name=QUEUE_NAME)
        return self.queue

    async def enqueue_ingestion(self, document_id: uuid.UUID, generation: int) -> bool:
        """Queue ingestion; returns False if Redis is unavailable (the worker's periodic
        sweep re-queues documents left in ``uploaded``, so nothing is lost)."""
        try:
            queue = await self._pool()
            await queue.enqueue_job(
                INGEST_DOCUMENT,
                str(document_id),
                generation,
                _job_id=ingest_job_id(document_id, generation),
                _queue_name=QUEUE_NAME,
            )
        except (OSError, ConnectionError, RedisError):
            logger.warning("could not enqueue ingestion", extra={"document_id": str(document_id)})
            return False
        return True

    async def enqueue_review(self, run_id: uuid.UUID, generation: int) -> bool:
        """Queue a review run; returns False if Redis is unavailable (the worker's sweep
        re-queues runs left ``queued``)."""
        try:
            queue = await self._pool()
            await queue.enqueue_job(
                RUN_REVIEW,
                str(run_id),
                generation,
                _job_id=review_job_id(run_id, generation),
                _queue_name=QUEUE_NAME,
            )
        except (OSError, ConnectionError, RedisError):
            logger.warning("could not enqueue review run", extra={"run_id": str(run_id)})
            return False
        return True

    async def close(self) -> None:
        if self.queue is not None:
            await self.queue.aclose()
        await self.models.aclose()
        await self.redis.aclose()
        await self.engine.dispose()
