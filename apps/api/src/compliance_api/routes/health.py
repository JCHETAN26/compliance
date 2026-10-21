"""Liveness and readiness probes (unauthenticated, no sensitive data)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from compliance_api.deps import Ctx
from compliance_core.health import (
    ReadinessReport,
    database_probe,
    migrations_probe,
    pgvector_probe,
    readiness,
    redis_probe,
)

router = APIRouter(prefix="/health", tags=["health"])


class Liveness(BaseModel):
    status: Literal["ok"]
    service: str


@router.get("")
async def live() -> Liveness:
    return Liveness(status="ok", service="api")


@router.get("/ready", response_model=ReadinessReport, responses={503: {"model": ReadinessReport}})
async def ready(ctx: Ctx) -> JSONResponse:
    report = await readiness(
        "api",
        {
            "database": database_probe(ctx.engine),
            "migrations": migrations_probe(ctx.engine),
            "pgvector": pgvector_probe(ctx.engine),
            "redis": redis_probe(ctx.redis),
        },
    )
    return JSONResponse(
        status_code=200 if report.status == "ok" else 503,
        content=report.model_dump(mode="json"),
    )
