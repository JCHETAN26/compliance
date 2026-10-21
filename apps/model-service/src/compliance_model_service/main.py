"""Rule-check model service.

PHASE 0 SKELETON. No model is trained, loaded or served yet; that is Phase 3 of
docs/BUILD_HANDOFF.md. This service exists so the compose topology, healthchecks and
API contract are in place:

- ``GET /health`` reports that the HTTP server is up and that no model is loaded.
- ``GET /metadata`` states explicitly that inference is not implemented.
- ``POST /v1/check`` validates the planned request contract and then answers
  ``501 Not Implemented``. It never returns a fabricated verdict.

Run with ``uvicorn --factory compliance_model_service.main:create_app``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from compliance_core.config import get_settings
from compliance_core.logs import configure_logging

SERVICE = "model-service"
NOT_IMPLEMENTED_MESSAGE = (
    "The rule-check model is not implemented yet (planned for Phase 3). "
    "No inference is performed and no verdict is returned."
)


class Health(BaseModel):
    status: Literal["ok"]
    service: str
    mode: Literal["not_implemented"]
    model_loaded: Literal[False]


class Metadata(BaseModel):
    service: str
    implemented: Literal[False]
    planned_phase: int
    model_version: None
    metrics: None
    message: str


class Span(BaseModel):
    model_config = ConfigDict(extra="forbid")
    span_id: Annotated[str, Field(min_length=1, max_length=128)]
    text: Annotated[str, Field(min_length=1, max_length=20_000)]


class CheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypothesis: Annotated[str, Field(min_length=1, max_length=2_000)]
    spans: Annotated[list[Span], Field(min_length=1, max_length=512)]


class NotImplementedBody(BaseModel):
    code: Literal["not_implemented"]
    message: str


class NotImplementedResponse(BaseModel):
    error: NotImplementedBody


def create_app() -> FastAPI:
    configure_logging(SERVICE, get_settings().log_level)
    app = FastAPI(title="Rule-check model service (not implemented)", version="0.0.0")

    @app.get("/health")
    async def health() -> Health:
        return Health(status="ok", service=SERVICE, mode="not_implemented", model_loaded=False)

    @app.get("/metadata")
    async def metadata() -> Metadata:
        return Metadata(
            service=SERVICE,
            implemented=False,
            planned_phase=3,
            model_version=None,
            metrics=None,
            message=NOT_IMPLEMENTED_MESSAGE,
        )

    @app.post("/v1/check", status_code=501, responses={501: {"model": NotImplementedResponse}})
    async def check(_: CheckRequest) -> JSONResponse:
        body = NotImplementedResponse(
            error=NotImplementedBody(code="not_implemented", message=NOT_IMPLEMENTED_MESSAGE)
        )
        return JSONResponse(status_code=501, content=body.model_dump())

    return app
