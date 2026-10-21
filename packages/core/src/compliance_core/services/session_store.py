"""Redis-backed auth state: failed-login throttling and access-token revocation."""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime

from redis.asyncio import Redis

from compliance_core.services.errors import RateLimitedError

_FAILURES_PREFIX = "auth:login-failures:"
_REVOKED_PREFIX = "auth:revoked:"


def _account_key(email: str) -> str:
    # Hash so raw email addresses never appear in Redis keys.
    return _FAILURES_PREFIX + hashlib.sha256(email.encode()).hexdigest()


class LoginThrottle:
    """Fixed-window limit on failed logins per account."""

    def __init__(self, redis: Redis, max_failures: int, window_seconds: int) -> None:
        self._redis = redis
        self._max_failures = max_failures
        self._window_seconds = window_seconds

    async def ensure_allowed(self, email: str) -> None:
        key = _account_key(email)
        failures = await self._redis.get(key)
        if failures is not None and int(failures) >= self._max_failures:
            ttl = await self._redis.ttl(key)
            retry_after = ttl if ttl > 0 else self._window_seconds
            raise RateLimitedError(
                "Too many failed sign-in attempts. Try again later.", retry_after
            )

    async def record_failure(self, email: str) -> None:
        key = _account_key(email)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, self._window_seconds, nx=True)
            await pipe.execute()

    async def reset(self, email: str) -> None:
        await self._redis.delete(_account_key(email))


class TokenRevocationList:
    """Remembers revoked token ids until the token would have expired anyway."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def revoke(self, jti: str, expires_at: datetime) -> None:
        remaining = math.ceil((expires_at - datetime.now(UTC)).total_seconds())
        if remaining > 0:
            await self._redis.set(_REVOKED_PREFIX + jti, "1", ex=remaining)

    async def is_revoked(self, jti: str) -> bool:
        return bool(await self._redis.exists(_REVOKED_PREFIX + jti))
