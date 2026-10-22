"""Typed async client for the model service (embeddings + optional rerank).

Every call has a bounded timeout. Connection errors, timeouts and 502/503/504 are
retried with exponential backoff up to ``retries`` times; anything else fails at once.

Failure classes:

- ``ModelServiceUnavailableError``: transport failure, timeout or overload after
  retries, or any other 5xx answer.
- ``ModelServiceRequestError``: the service rejected *the caller's input* (4xx with an
  error code, e.g. ``query_too_long`` or ``rerank_disabled``).
- ``ModelServiceProtocolError``: the service answered 2xx with a body that violates
  the contract (malformed JSON, missing fields, wrong vector count/dimension, non-finite
  numbers, unexpected or missing rerank ids).

Responses are validated against typed contracts; nothing falls back to a different
model, a fake vector or a partial ranking.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Annotated, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from compliance_core.config import Settings

RETRYABLE_STATUS = {502, 503, 504}
EMBEDDING_DIM = 384
_M = TypeVar("_M", bound=BaseModel)


def embedding_label(model: str, revision: str) -> str:
    """The encoder identity stored on each ingested document (``documents.embedding_model``)."""
    return f"{model}@{revision[:12]}"


class ModelServiceError(Exception):
    def __init__(self, message: str, code: str = "model_service_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class ModelServiceUnavailableError(ModelServiceError):
    """Transport failure, timeout, overload or server error."""


class ModelServiceRequestError(ModelServiceError):
    """The service rejected the request input (4xx), e.g. query too long or rerank disabled."""


class ModelServiceProtocolError(ModelServiceError):
    """The service returned a success response that does not satisfy the contract."""


# --------------------------------------------------------------------------- contracts


class _Body(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _ErrorDetail(_Body):
    code: str
    message: str


class _ErrorBody(_Body):
    error: _ErrorDetail


class _EmbedBody(_Body):
    model: Annotated[str, Field(min_length=1)]
    revision: Annotated[str, Field(min_length=1)]
    dim: int
    kind: str
    query_instruction: str | None
    vectors: list[list[float]]


class _RerankScore(_Body):
    id: str
    score: float


class _RerankBody(_Body):
    model: str
    revision: str
    scores: list[_RerankScore]


class _ModelInfo(_Body):
    model: Annotated[str, Field(min_length=1)]
    revision: Annotated[str, Field(min_length=1)]


class _EmbeddingInfo(_ModelInfo):
    dim: int
    query_instruction: str


class _MetadataBody(_Body):
    embedding: _EmbeddingInfo
    rerank: _ModelInfo | None


@dataclass(frozen=True)
class QueryEmbedding:
    vector: list[float]
    model: str
    revision: str
    query_instruction: str

    @property
    def label(self) -> str:
        return embedding_label(self.model, self.revision)


@dataclass(frozen=True)
class ServiceInfo:
    embedding_model: str
    embedding_revision: str
    query_instruction: str
    rerank_available: bool
    rerank_model: str | None
    rerank_revision: str | None


def _finite(values: list[float], what: str) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ModelServiceProtocolError(f"model service returned non-finite {what}", "bad_response")


class ModelServiceClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float,
        retries: int,
        transport: httpx.AsyncBaseTransport | None = None,
        backoff: float = 0.2,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=min(timeout, 3.0)),
            transport=transport,
        )
        self._retries = retries
        self._backoff = backoff

    @classmethod
    def from_settings(cls, settings: Settings) -> ModelServiceClient:
        return cls(
            settings.model_service_url,
            timeout=settings.model_service_timeout_seconds,
            retries=settings.model_service_retries,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(
        self, contract: type[_M], method: str, path: str, json: object | None = None
    ) -> _M:
        last = "no attempt"
        for attempt in range(self._retries + 1):
            if attempt:
                await asyncio.sleep(self._backoff * (2 ** (attempt - 1)))
            try:
                response = await self._client.request(method, path, json=json)
            except httpx.TimeoutException:
                last = "timeout"
                continue
            except httpx.TransportError as exc:
                last = type(exc).__name__
                continue
            status = response.status_code
            if status in RETRYABLE_STATUS:
                last = f"HTTP {status}"
                continue
            if status >= 500:
                raise ModelServiceUnavailableError(
                    f"model service failed with HTTP {status}", "model_service_failed"
                )
            if status >= 400:
                try:
                    detail = _ErrorBody.model_validate_json(response.content).error
                except ValidationError:
                    raise ModelServiceProtocolError(
                        f"model service returned HTTP {status} without an error body",
                        "bad_response",
                    ) from None
                raise ModelServiceRequestError(detail.message, detail.code)
            if status != 200:
                raise ModelServiceProtocolError(
                    f"model service returned unexpected HTTP {status}", "bad_response"
                )
            try:
                return contract.model_validate_json(response.content)
            except ValidationError as exc:
                raise ModelServiceProtocolError(
                    f"model service response for {path} violates its contract "
                    f"({exc.error_count()} error(s))",
                    "bad_response",
                ) from None
        raise ModelServiceUnavailableError(
            f"model service unavailable after {self._retries + 1} attempt(s): {last}",
            "model_service_unavailable",
        )

    async def embed_query(self, query: str) -> QueryEmbedding:
        body = await self._request(
            _EmbedBody, "POST", "/v1/embed", {"inputs": [query], "kind": "query"}
        )
        if body.kind != "query" or body.query_instruction is None:
            raise ModelServiceProtocolError(
                "embed response is not a query embedding", "bad_response"
            )
        if len(body.vectors) != 1:
            raise ModelServiceProtocolError(
                f"expected 1 vector, got {len(body.vectors)}", "bad_response"
            )
        vector = body.vectors[0]
        if body.dim != EMBEDDING_DIM or len(vector) != EMBEDDING_DIM:
            raise ModelServiceProtocolError(
                f"expected a {EMBEDDING_DIM}-d vector, got {len(vector)} (dim={body.dim})",
                "bad_response",
            )
        _finite(vector, "vector values")
        norm = math.sqrt(sum(v * v for v in vector))
        if not 0.99 <= norm <= 1.01:
            raise ModelServiceProtocolError(
                f"query vector is not unit-normalised (norm={norm:.4f})", "bad_response"
            )
        return QueryEmbedding(
            vector=vector,
            model=body.model,
            revision=body.revision,
            query_instruction=body.query_instruction,
        )

    async def rerank(self, query: str, passages: list[tuple[str, str]]) -> dict[str, float]:
        body = await self._request(
            _RerankBody,
            "POST",
            "/v1/rerank",
            {"query": query, "passages": [{"id": pid, "text": text} for pid, text in passages]},
        )
        scores = {item.id: item.score for item in body.scores}
        if len(body.scores) != len(passages) or set(scores) != {pid for pid, _ in passages}:
            raise ModelServiceProtocolError(
                "rerank response ids do not match the submitted passages", "bad_response"
            )
        _finite(list(scores.values()), "rerank scores")
        return scores

    async def info(self) -> ServiceInfo:
        body = await self._request(_MetadataBody, "GET", "/metadata")
        if body.embedding.dim != EMBEDDING_DIM:
            raise ModelServiceProtocolError(
                f"model service embedding dim {body.embedding.dim} != {EMBEDDING_DIM}",
                "bad_response",
            )
        rerank = body.rerank
        return ServiceInfo(
            embedding_model=body.embedding.model,
            embedding_revision=body.embedding.revision,
            query_instruction=body.embedding.query_instruction,
            rerank_available=rerank is not None,
            rerank_model=rerank.model if rerank else None,
            rerank_revision=rerank.revision if rerank else None,
        )
