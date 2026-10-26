"""FastAPI application factory.

Run with ``uvicorn --factory compliance_api.main:create_app``.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from sqlalchemy.exc import SQLAlchemyError
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from compliance_api.context import AppContext
from compliance_api.errors import install_error_handlers
from compliance_api.routes import (
    auth,
    documents,
    health,
    models,
    reviews,
    rule_packs,
    search,
    workspaces,
)
from compliance_api.routes import settings as settings_routes
from compliance_core.config import Settings, get_settings
from compliance_core.logs import configure_logging, request_id_var
from compliance_core.services.rule_packs import sync_builtin_packs

API_PREFIX = "/api/v1"
logger = logging.getLogger("compliance_api.access")

_SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
]


class RequestContextMiddleware:
    """Assigns a request id, adds security headers and emits one JSON access log line.

    Implemented as pure ASGI so streaming (SSE) responses are not buffered.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if 0 < len(incoming) <= 128 and incoming.isprintable() else ""
        request_id = request_id or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_holder = {"status": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                headers.extend(_SECURITY_HEADERS)
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            logger.info(
                "request",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status_holder["status"],
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            request_id_var.reset(token)


async def sync_rule_packs(ctx: AppContext) -> None:
    """Insert or refresh the built-in rule packs (idempotent, serialised across processes).
    A database that is not migrated yet is reported, not fatal: the packs appear on the
    next start."""
    try:
        async with ctx.session_factory() as session:
            await sync_builtin_packs(session)
    except (OSError, SQLAlchemyError):
        logger.warning("could not synchronise built-in rule packs", exc_info=True)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging("api", settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        ctx = AppContext.create(settings)
        app.state.ctx = ctx
        await sync_rule_packs(ctx)
        logger.info("api started", extra={"app_env": settings.app_env.value})
        try:
            yield
        finally:
            await ctx.close()

    app = FastAPI(
        title="Compliance Workspace API",
        version="0.1.0",
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    install_error_handlers(app)
    app.add_middleware(RequestContextMiddleware)

    api = APIRouter(prefix=API_PREFIX)
    api.include_router(health.router)
    api.include_router(auth.router)
    api.include_router(workspaces.router)
    api.include_router(documents.router)
    api.include_router(search.router)
    api.include_router(models.router)
    api.include_router(rule_packs.router)
    api.include_router(reviews.router)
    api.include_router(settings_routes.router)
    app.include_router(api)
    return app
