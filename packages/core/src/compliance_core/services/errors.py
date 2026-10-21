"""Transport-agnostic service errors. The API and MCP layers map these to responses."""

from __future__ import annotations


class ServiceError(Exception):
    code = "service_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(ServiceError):
    code = "not_found"


class ConflictError(ServiceError):
    code = "conflict"


class PermissionDeniedError(ServiceError):
    code = "forbidden"


class AuthenticationError(ServiceError):
    code = "unauthenticated"


class InvalidOperationError(ServiceError):
    code = "invalid_operation"


class RateLimitedError(ServiceError):
    code = "rate_limited"

    def __init__(self, message: str, retry_after_seconds: int) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
