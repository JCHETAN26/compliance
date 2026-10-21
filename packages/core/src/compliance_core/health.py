"""Dependency probes used by service readiness endpoints."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from compliance_core.db.migrate import current_revision, head_revision

CHECK_TIMEOUT_SECONDS = 3.0


class CheckResult(BaseModel):
    status: Literal["ok", "error"]
    latency_ms: float
    detail: str | None = None


class ReadinessReport(BaseModel):
    status: Literal["ok", "error"]
    service: str
    checks: dict[str, CheckResult]


async def _timed(probe: Callable[[], Awaitable[str | None]]) -> CheckResult:
    started = time.perf_counter()
    try:
        detail = await asyncio.wait_for(probe(), timeout=CHECK_TIMEOUT_SECONDS)
    except Exception as exc:  # any failure makes the dependency unhealthy
        return CheckResult(
            status="error",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            detail=f"{type(exc).__name__}: {exc}"[:300],
        )
    return CheckResult(
        status="ok", latency_ms=round((time.perf_counter() - started) * 1000, 2), detail=detail
    )


def database_probe(engine: AsyncEngine) -> Callable[[], Awaitable[str | None]]:
    async def probe() -> str | None:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return None

    return probe


def migrations_probe(engine: AsyncEngine) -> Callable[[], Awaitable[str | None]]:
    async def probe() -> str | None:
        expected = head_revision()
        async with engine.connect() as conn:
            current = await current_revision(conn)
        if current != expected:
            raise RuntimeError(f"database at revision {current!r}, expected {expected!r}")
        return f"revision {current}"

    return probe


def pgvector_probe(engine: AsyncEngine) -> Callable[[], Awaitable[str | None]]:
    async def probe() -> str | None:
        async with engine.connect() as conn:
            version = await conn.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
        if version is None:
            raise RuntimeError("pgvector extension is not installed")
        return f"vector {version}"

    return probe


def redis_probe(redis: Redis) -> Callable[[], Awaitable[str | None]]:
    async def probe() -> str | None:
        if not await redis.ping():
            raise RuntimeError("PING did not return PONG")
        return None

    return probe


async def readiness(
    service: str, probes: dict[str, Callable[[], Awaitable[str | None]]]
) -> ReadinessReport:
    names = list(probes)
    results = await asyncio.gather(*(_timed(probes[name]) for name in names))
    checks = dict(zip(names, results, strict=True))
    status: Literal["ok", "error"] = (
        "ok" if all(c.status == "ok" for c in checks.values()) else "error"
    )
    return ReadinessReport(status=status, service=service, checks=checks)
