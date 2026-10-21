"""Consistent error envelope: ``{"error": {"code", "message", "details"}}``."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from compliance_core.services.errors import (
    AuthenticationError,
    ConflictError,
    InvalidOperationError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitedError,
    ServiceError,
)
from compliance_core.uploads import UploadError

logger = logging.getLogger(__name__)

_STATUS_BY_ERROR: dict[type[ServiceError], int] = {
    NotFoundError: 404,
    ConflictError: 409,
    PermissionDeniedError: 403,
    AuthenticationError: 401,
    InvalidOperationError: 409,
    RateLimitedError: 429,
}


class ErrorDetail(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: list[ErrorDetail] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json", exclude_none=True),
        headers=headers,
    )


def _code_for_status(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase.lower().replace(" ", "_").replace("-", "_")
    except ValueError:
        return "error"


async def _service_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ServiceError)  # noqa: S101 - registered for ServiceError only
    status_code = next(
        (status for cls, status in _STATUS_BY_ERROR.items() if isinstance(exc, cls)), 400
    )
    headers = None
    if isinstance(exc, RateLimitedError):
        headers = {"Retry-After": str(exc.retry_after_seconds)}
    return error_response(status_code, exc.code, exc.message, headers=headers)


async def _upload_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, UploadError)  # noqa: S101
    return error_response(exc.status_code, exc.code, str(exc))


async def _http_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101
    message = exc.detail if isinstance(exc.detail, str) else HTTPStatus(exc.status_code).phrase
    code = _code_for_status(exc.status_code)
    if exc.status_code == 401:
        code = "unauthenticated"
    headers = dict(exc.headers) if exc.headers else None
    return error_response(exc.status_code, code, message, headers=headers)


async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)  # noqa: S101
    # Deliberately omit the offending input values (they may contain passwords).
    details = [
        ErrorDetail(
            loc=list(err.get("loc", ())), msg=str(err.get("msg")), type=str(err.get("type"))
        )
        for err in exc.errors()
    ]
    return error_response(422, "validation_error", "Request validation failed.", details)


async def _unhandled_error(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error", exc_info=exc)
    return error_response(500, "internal_error", "An unexpected error occurred.")


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ServiceError, _service_error)
    app.add_exception_handler(UploadError, _upload_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled_error)


def error_responses(*codes: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries documenting the error envelope."""
    return {code: {"model": ErrorResponse} for code in codes}
