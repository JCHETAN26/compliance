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


class UploadRateLimiter:
    """Per-user upload quotas in fixed one-minute windows, reserved atomically.

    Two units are counted, both *attempts* (rejected and duplicate files count too):

    - requests: reserved with ``INCR`` before the body is streamed, so concurrent
      requests cannot all slip past a read-then-write check;
    - files: every file part is reserved with ``INCRBY`` after streaming and before any
      validation or storage, whatever its outcome.

    A reservation that exceeds the cap is refused (HTTP 429) and is not rolled back:
    hammering the endpoint keeps the user over the limit until the window ends.
    """

    def __init__(self, redis: Redis, files_per_minute: int, requests_per_minute: int) -> None:
        self._redis = redis
        self._files_per_minute = files_per_minute
        self._requests_per_minute = requests_per_minute

    @staticmethod
    def _window() -> tuple[int, int]:
        now = int(datetime.now(UTC).timestamp())
        return now // 60, 60 - now % 60

    async def _reserve(self, unit: str, user_id: str, amount: int, cap: int, noun: str) -> None:
        window, retry_after = self._window()
        key = f"ratelimit:uploads:{unit}:{user_id}:{window}"
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incrby(key, amount)
            pipe.expire(key, 120)
            total, _ = await pipe.execute()
        if int(total) > cap:
            raise RateLimitedError(
                f"Upload limit of {cap} {noun} per minute reached. Try again shortly.", retry_after
            )

    async def reserve_request(self, user_id: str) -> None:
        await self._reserve("requests", user_id, 1, self._requests_per_minute, "upload requests")

    async def reserve_files(self, user_id: str, files: int) -> None:
        if files > 0:
            await self._reserve("files", user_id, files, self._files_per_minute, "files")


class FixedWindowLimiter:
    """Atomic per-key, per-minute request counter (INCR before work is done)."""

    def __init__(self, redis: Redis, name: str, per_minute: int, noun: str) -> None:
        self._redis = redis
        self._name = name
        self._per_minute = per_minute
        self._noun = noun

    async def reserve(self, key: str) -> None:
        now = int(datetime.now(UTC).timestamp())
        redis_key = f"ratelimit:{self._name}:{key}:{now // 60}"
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incr(redis_key)
            pipe.expire(redis_key, 120)
            total, _ = await pipe.execute()
        if int(total) > self._per_minute:
            raise RateLimitedError(
                f"Limit of {self._per_minute} {self._noun} per minute reached. Try again shortly.",
                60 - now % 60,
            )
