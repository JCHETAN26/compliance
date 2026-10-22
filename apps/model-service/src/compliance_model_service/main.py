"""Local model inference service.

Implemented (Phase 2):

- ``POST /v1/embed``: embeddings from the **same pinned encoder and code path the
  ingestion worker uses** (``compliance_ingestion.embeddings.Embedder``:
  BAAI/bge-small-en-v1.5 at a pinned revision, SHA-256-verified weights, CPU).
  ``kind="query"`` prepends the configured query instruction and rejects inputs longer
  than the encoder context. ``kind="passage"`` embeds with no prefix and uses bounded
  token windows for long inputs, exactly as ingestion does.
- ``POST /v1/rerank``: the optional cross-encoder, only when ``RERANK_ENABLED=true``.

Not implemented yet (Phase 3): the rule-check model. ``POST /v1/check`` validates its
request contract and answers ``501 Not Implemented``; it never returns a verdict.

Inference runs in a single-thread executor per model (torch parallelises internally), so
the event loop stays responsive. A bounded semaphore sheds load with HTTP 503 instead of
queueing without limit.

Run with ``uvicorn --factory compliance_model_service.main:create_app``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeVar

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from compliance_core.config import Settings, get_settings
from compliance_core.logs import configure_logging
from compliance_ingestion.embeddings import (
    EMBEDDING_DIM,
    Embedder,
    QueryTooLongError,
    ensure_model,
    spec_from_settings,
)
from compliance_model_service.rerank import CrossEncoderReranker, rerank_spec

SERVICE = "model-service"
RULE_CHECK_NOT_IMPLEMENTED = (
    "The rule-check model is not implemented yet (planned for Phase 3). "
    "No inference is performed and no verdict is returned."
)
MAX_PENDING = 32
BUSY = "The model service is busy; retry shortly."
T = TypeVar("T")
logger = logging.getLogger("compliance_model_service")


# --------------------------------------------------------------------------- schemas


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmbedRequest(_Strict):
    inputs: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=20_000)]],
        Field(min_length=1, max_length=64),
    ]
    kind: Literal["query", "passage"]


class EmbedResponse(BaseModel):
    model: str
    revision: str
    dim: int
    kind: Literal["query", "passage"]
    query_instruction: str | None
    vectors: list[list[float]]
    windows: list[int]


class RerankPassage(_Strict):
    id: Annotated[str, Field(min_length=1, max_length=128)]
    text: Annotated[str, Field(min_length=1, max_length=20_000)]


class RerankRequest(_Strict):
    query: Annotated[str, Field(min_length=1, max_length=2_000)]
    passages: Annotated[list[RerankPassage], Field(min_length=1, max_length=100)]


class RerankScore(BaseModel):
    id: str
    score: float


class RerankResponse(BaseModel):
    model: str
    revision: str
    scores: list[RerankScore]


class Span(_Strict):
    span_id: Annotated[str, Field(min_length=1, max_length=128)]
    text: Annotated[str, Field(min_length=1, max_length=20_000)]


class CheckRequest(_Strict):
    hypothesis: Annotated[str, Field(min_length=1, max_length=2_000)]
    spans: Annotated[list[Span], Field(min_length=1, max_length=512)]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class ModelInfo(BaseModel):
    model: str
    revision: str
    sha256: str
    loaded: bool


class EmbeddingInfo(ModelInfo):
    dim: int
    max_seq_length: int
    query_instruction: str


class Health(BaseModel):
    status: Literal["ok"]
    service: str
    embedding: Literal["ready"]
    rerank: Literal["ready", "disabled"]
    rule_check: Literal["not_implemented"]


class Metadata(BaseModel):
    service: str
    embedding: EmbeddingInfo
    rerank: ModelInfo | None
    rule_check: dict[str, Any]


# --------------------------------------------------------------------------- runtime


@dataclass
class Runtime:
    settings: Settings
    embedder: Embedder
    reranker: CrossEncoderReranker | None
    embed_pool: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(1, "embed"))
    rerank_pool: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(1, "rerank"))
    slots: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(MAX_PENDING))

    async def run(self, pool: ThreadPoolExecutor, fn: Callable[[], T]) -> T:
        if self.slots.locked():
            raise OverloadedError
        async with self.slots:
            return await asyncio.get_running_loop().run_in_executor(pool, fn)

    def close(self) -> None:
        self.embed_pool.shutdown(wait=False, cancel_futures=True)
        self.rerank_pool.shutdown(wait=False, cancel_futures=True)


class OverloadedError(Exception):
    pass


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def load_runtime(settings: Settings) -> Runtime:
    spec = spec_from_settings(settings)
    path = ensure_model(
        settings.model_cache_dir, spec, allow_download=settings.artifact_auto_download
    )
    embedder = Embedder(
        path, threads=settings.embedding_threads, batch_size=settings.embedding_batch_size
    )
    reranker = None
    if settings.rerank_enabled:
        # Explicitly enabled: an unavailable reranker is a startup failure, not a fallback.
        rpath = ensure_model(
            settings.model_cache_dir,
            rerank_spec(settings),
            allow_download=settings.artifact_auto_download,
        )
        reranker = CrossEncoderReranker(rpath, threads=settings.embedding_threads)
    return Runtime(settings=settings, embedder=embedder, reranker=reranker)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error=ErrorBody(code=code, message=message)).model_dump(),
    )


def create_app(
    settings: Settings | None = None,
    runtime_factory: Callable[[Settings], Runtime] = load_runtime,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(SERVICE, settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = await asyncio.to_thread(runtime_factory, settings)
        app.state.runtime = runtime
        logger.info(
            "model service ready",
            extra={
                "embedding_model": settings.embedding_model_id,
                "rerank": runtime.reranker is not None,
            },
        )
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(
        title="Model service (embeddings; rule-check model not implemented)",
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.exception_handler(ServiceError)
    async def service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message)

    def runtime_of() -> Runtime:
        runtime: Runtime = app.state.runtime
        return runtime

    @app.get("/health")
    async def health() -> Health:
        runtime = runtime_of()
        return Health(
            status="ok",
            service=SERVICE,
            embedding="ready",
            rerank="ready" if runtime.reranker is not None else "disabled",
            rule_check="not_implemented",
        )

    @app.get("/metadata")
    async def metadata() -> Metadata:
        runtime = runtime_of()
        s = runtime.settings
        return Metadata(
            service=SERVICE,
            embedding=EmbeddingInfo(
                model=s.embedding_model_id,
                revision=s.embedding_model_revision,
                sha256=s.embedding_model_sha256,
                loaded=True,
                dim=EMBEDDING_DIM,
                max_seq_length=runtime.embedder.max_seq_length,
                query_instruction=s.embedding_query_instruction,
            ),
            rerank=(
                ModelInfo(
                    model=s.rerank_model_id,
                    revision=s.rerank_model_revision,
                    sha256=s.rerank_model_sha256,
                    loaded=True,
                )
                if runtime.reranker is not None
                else None
            ),
            rule_check={
                "implemented": False,
                "planned_phase": 3,
                "message": RULE_CHECK_NOT_IMPLEMENTED,
            },
        )

    @app.post(
        "/v1/embed",
        responses={
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def embed(request: EmbedRequest) -> EmbedResponse:
        runtime = runtime_of()
        instruction = runtime.settings.embedding_query_instruction
        try:
            if request.kind == "query":
                vectors = await runtime.run(
                    runtime.embed_pool,
                    lambda: runtime.embedder.embed_queries(request.inputs, instruction),
                )
                windows = [1] * len(vectors)
            else:
                embedded = await runtime.run(
                    runtime.embed_pool, lambda: runtime.embedder.embed(request.inputs)
                )
                vectors = [e.vector for e in embedded]
                windows = [e.windows for e in embedded]
        except QueryTooLongError as exc:
            raise ServiceError(422, "query_too_long", str(exc)) from exc
        except OverloadedError as exc:
            raise ServiceError(503, "overloaded", BUSY) from exc
        return EmbedResponse(
            model=runtime.settings.embedding_model_id,
            revision=runtime.settings.embedding_model_revision,
            dim=EMBEDDING_DIM,
            kind=request.kind,
            query_instruction=instruction if request.kind == "query" else None,
            vectors=vectors,
            windows=windows,
        )

    @app.post(
        "/v1/rerank",
        responses={409: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def rerank(request: RerankRequest) -> RerankResponse:
        runtime = runtime_of()
        reranker = runtime.reranker
        if reranker is None:
            raise ServiceError(
                409, "rerank_disabled", "Reranking is not enabled on this model service."
            )
        texts = [p.text for p in request.passages]
        try:
            scores = await runtime.run(
                runtime.rerank_pool, lambda: reranker.score(request.query, texts)
            )
        except OverloadedError as exc:
            raise ServiceError(503, "overloaded", BUSY) from exc
        return RerankResponse(
            model=runtime.settings.rerank_model_id,
            revision=runtime.settings.rerank_model_revision,
            scores=[
                RerankScore(id=p.id, score=s) for p, s in zip(request.passages, scores, strict=True)
            ],
        )

    @app.post("/v1/check", status_code=501, responses={501: {"model": ErrorResponse}})
    async def check(_: CheckRequest) -> JSONResponse:
        return _error(501, "not_implemented", RULE_CHECK_NOT_IMPLEMENTED)

    return app
