"""Process-wide resources created in the application lifespan."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from compliance_core.config import Settings
from compliance_core.db.session import create_engine, create_session_factory
from compliance_core.security.tokens import TokenSigner
from compliance_core.services.session_store import LoginThrottle, TokenRevocationList


@dataclass(slots=True)
class AppContext:
    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    redis: Redis
    signer: TokenSigner
    throttle: LoginThrottle
    revocations: TokenRevocationList

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
        )

    async def close(self) -> None:
        await self.redis.aclose()
        await self.engine.dispose()
