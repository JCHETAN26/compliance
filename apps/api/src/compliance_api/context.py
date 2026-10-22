"""Process-wide resources created in the application lifespan."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import timedelta

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from compliance_core.config import Settings
from compliance_core.db.session import create_engine, create_session_factory
from compliance_core.jobs import INGEST_DOCUMENT, QUEUE_NAME, ingest_job_id
from compliance_core.model_client import ModelServiceClient
from compliance_core.security.tokens import TokenSigner
from compliance_core.services.session_store import (
    FixedWindowLimiter,
    LoginThrottle,
    TokenRevocationList,
    UploadRateLimiter,
)
from compliance_core.storage import LocalStorage
from compliance_retrieval.config import RetrievalConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AppContext:
    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    redis: Redis
    signer: TokenSigner
    throttle: LoginThrottle
    revocations: TokenRevocationList
    storage: LocalStorage
    upload_limiter: UploadRateLimiter
    models: ModelServiceClient
    search_limiter: FixedWindowLimiter
    retrieval: RetrievalConfig
    queue: ArqRedis | None = None

    @classmethod
    def create(cls, settings: Settings) -> AppContext:
        engine = create_engine(settings)
        redis = Redis.from_url(settings.redis_url, socket_timeout=5, socket_connect_timeout=5)
        return cls(
            settings=settings,
            engine=engine,
            session_factory=create_session_factory(engine),
            redis=redis,
            signer=TokenSigner(
                secret=settings.resolved_jwt_secret(),
                issuer=settings.jwt_issuer,
                audience=settings.jwt_audience,
                ttl=timedelta(minutes=settings.jwt_access_ttl_minutes),
            ),
            throttle=LoginThrottle(
                redis, settings.login_max_failures, settings.login_failure_window_seconds
            ),
            revocations=TokenRevocationList(redis),
            storage=LocalStorage(settings.storage_dir),
            models=ModelServiceClient.from_settings(settings),
            search_limiter=FixedWindowLimiter(
                redis, "search", settings.search_rate_limit_per_minute, "searches"
            ),
            retrieval=RetrievalConfig.from_settings(settings),
            upload_limiter=UploadRateLimiter(
                redis, settings.upload_rate_limit_per_minute, settings.upload_requests_per_minute
            ),
        )

    async def enqueue_ingestion(self, document_id: uuid.UUID, generation: int) -> bool:
        """Queue ingestion; returns False if Redis is unavailable (the worker's periodic
        sweep re-queues documents left in ``uploaded``, so nothing is lost)."""
        try:
            if self.queue is None:
                redis_settings = RedisSettings.from_dsn(self.settings.redis_url)
                redis_settings.conn_retries = 1  # fail fast; the worker sweep re-queues
                self.queue = await create_pool(redis_settings, default_queue_name=QUEUE_NAME)
            await self.queue.enqueue_job(
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

    async def close(self) -> None:
        if self.queue is not None:
            await self.queue.aclose()
        await self.models.aclose()
        await self.redis.aclose()
        await self.engine.dispose()
