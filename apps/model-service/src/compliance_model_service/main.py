"""Local model inference service.

Implemented (Phase 2):

- ``POST /v1/embed``: embeddings from the **same pinned encoder and code path the
  ingestion worker uses** (``compliance_ingestion.embeddings.Embedder``:
  BAAI/bge-small-en-v1.5 at a pinned revision, SHA-256-verified weights, CPU).
  ``kind="query"`` prepends the configured query instruction and rejects inputs longer
  than the encoder context. ``kind="passage"`` embeds with no prefix and uses bounded
  token windows for long inputs, exactly as ingestion does.
- ``POST /v1/rerank``: the optional cross-encoder, only when ``RERANK_ENABLED=true``.

Phase 3: ``POST /v1/check`` serves the trained rule-check model from a verified ONNX
artifact (int8 by default, fp32 with ``RULE_CHECK_VARIANT=fp32``) with dynamic
micro-batching across concurrent requests (``compliance_model_service.rule_check``).
If no verified artifact is present the service starts in **untrained_fallback** mode:
embeddings keep working, ``/health`` and ``/metadata`` say so, and ``/v1/check`` answers
503 ``rule_check_unavailable`` with the reason. It never returns a made-up verdict.

Inference runs in a single-thread executor per model (torch and ONNX Runtime
parallelise internally), so the event loop stays responsive. Bounded admission sheds
load with HTTP 503 instead of queueing without limit.

Run with ``uvicorn --factory compliance_model_service.main:create_app``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
from compliance_model_service.rule_check.artifact import (
    ArtifactError,
    InferenceSpec,
    VerifiedArtifact,
    load_artifact,
)
from compliance_model_service.rule_check.batching import (
    BatcherClosedError,
    InvalidModelOutputError,
    MicroBatcher,
)
from compliance_model_service.rule_check.batching import OverloadedError as BatchOverloadedError
from compliance_model_service.rule_check.encoding import InputEncoder
from compliance_model_service.rule_check.engine import RuleCheckEngine, SpanText
from compliance_model_service.rule_check.onnx_backend import OnnxBackend

SERVICE = "model-service"
UNTRAINED_FALLBACK = (
    "No verified trained rule-check model is loaded (untrained fallback mode). "
    "No inference is performed and no verdict is returned; callers must escalate."
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


def _not_blank(value: str) -> str:
    # Validated, not normalised: the text is kept exactly as sent (source offsets).
    if not value.strip():
        raise ValueError("must contain non-whitespace text")
    return value


class Span(_Strict):
    span_id: Annotated[str, Field(min_length=1, max_length=128)]
    text: Annotated[str, Field(min_length=1, max_length=20_000)]

    _text_not_blank = field_validator("text")(_not_blank)


class CheckRequest(_Strict):
    hypothesis: Annotated[str, Field(min_length=1, max_length=2_000)]
    spans: Annotated[list[Span], Field(min_length=1, max_length=512)]

    _hypothesis_not_blank = field_validator("hypothesis")(_not_blank)

    @field_validator("spans")
    @classmethod
    def _unique_ids(cls, spans: list[Span]) -> list[Span]:
        if len({s.span_id for s in spans}) != len(spans):
            raise ValueError("span_id values must be unique")
        return spans

    def total_chars(self) -> int:
        return len(self.hypothesis) + sum(len(s.text) for s in self.spans)


class CheckEvidence(BaseModel):
    span_id: str
    score: float  # evidence-head probability


class CheckResponse(BaseModel):
    label: Literal["entailment", "contradiction", "not_mentioned"]
    status: Literal["satisfied", "violated", "not_mentioned"]
    confidence: float  # temperature-calibrated probability of ``label``
    probabilities: dict[str, float]
    evidence: list[CheckEvidence]  # spans scoring >= the DEV-selected evidence threshold
    considered: list[CheckEvidence]  # the top-k spans the label head read (document order)
    escalate: bool | None  # confidence < DEV-selected tau; None if no tau in the artifact
    truncated_inputs: int
    model_version: str
    model_kind: Literal["production", "smoke"]
    variant: Literal["int8", "fp32"]


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
    rule_check: Literal["ready", "unavailable"]
    rule_check_mode: Literal["model", "untrained_fallback"]


class RuleCheckInfo(BaseModel):
    available: bool
    mode: Literal["model", "untrained_fallback"]
    reason: str | None
    artifact_dir: str
    model_version: str | None = None
    kind: Literal["production", "smoke"] | None = None
    variant: Literal["int8", "fp32"] | None = None
    backbone: dict[str, Any] | None = None
    inference: InferenceSpec | None = None
    metrics: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    batching: dict[str, Any]


class Metadata(BaseModel):
    service: str
    embedding: EmbeddingInfo
    rerank: ModelInfo | None
    rule_check: RuleCheckInfo


# --------------------------------------------------------------------------- runtime


@dataclass
class RuleCheckRuntime:
    """Either a ready engine or the reason the service is in untrained fallback mode."""

    artifact_dir: Path
    artifact: VerifiedArtifact | None = None
    engine: RuleCheckEngine | None = None
    batcher: MicroBatcher | None = None
    reason: str | None = None
    infer_pool: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(1, "rule-check")
    )
    tokenize_pool: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(1, "rule-check-tok")
    )

    @property
    def ready(self) -> bool:
        return self.engine is not None


@dataclass
class Runtime:
    settings: Settings
    embedder: Embedder
    reranker: CrossEncoderReranker | None
    rule_check: RuleCheckRuntime | None = None
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
        if self.rule_check is not None:
            self.rule_check.infer_pool.shutdown(wait=False, cancel_futures=True)
            self.rule_check.tokenize_pool.shutdown(wait=False, cancel_futures=True)


class OverloadedError(Exception):
    pass


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def load_rule_check(settings: Settings) -> RuleCheckRuntime:
    """Load the verified rule-check artifact, or explain why the fallback mode is used.

    A missing, unverifiable or smoke-only artifact never fails startup (the rest of the
    service keeps working); it leaves the rule check unavailable with a stated reason.
    """
    state = RuleCheckRuntime(artifact_dir=settings.resolved_rule_check_dir)
    try:
        artifact = load_artifact(
            state.artifact_dir,
            variant=settings.rule_check_variant,
            allow_smoke=settings.rule_check_allow_smoke,
        )
        encoder = InputEncoder.from_file(
            artifact.tokenizer_path, artifact.manifest.encoder.to_config()
        )
        backend = OnnxBackend(
            artifact.onnx_path,
            encoder,
            threads=settings.rule_check_threads,
            max_tokens_per_forward=settings.rule_check_max_tokens_per_forward,
        )
    except ArtifactError as exc:
        state.reason = str(exc)
        logger.warning("rule-check model unavailable", extra={"reason": state.reason})
        return state
    except Exception as exc:  # a verified file that ONNX Runtime cannot load
        state.reason = f"verified artifact failed to load: {type(exc).__name__}: {exc}"
        logger.error("rule-check model failed to load", extra={"reason": state.reason})
        return state
    batcher = MicroBatcher(
        backend,
        max_batch_rows=settings.rule_check_max_batch_rows,
        max_wait_ms=settings.rule_check_max_wait_ms,
        max_queue_rows=settings.rule_check_max_queue_rows,
        executor=state.infer_pool,
    )
    state.artifact = artifact
    state.batcher = batcher
    state.engine = RuleCheckEngine(
        encoder,
        batcher,
        artifact.manifest.inference,
        state.tokenize_pool,
        max_pending_tokenize=settings.rule_check_max_pending_requests,
    )
    return state


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
    return Runtime(
        settings=settings,
        embedder=embedder,
        reranker=reranker,
        rule_check=load_rule_check(settings),
    )


def rule_check_info(state: RuleCheckRuntime, settings: Settings) -> RuleCheckInfo:
    batching: dict[str, Any] = {
        "max_batch_rows": settings.rule_check_max_batch_rows,
        "max_wait_ms": settings.rule_check_max_wait_ms,
        "max_queue_rows": settings.rule_check_max_queue_rows,
        "stats": state.batcher.stats.snapshot() if state.batcher is not None else None,
        "max_pending_tokenize_requests": settings.rule_check_max_pending_requests,
        "max_request_chars": settings.rule_check_max_request_chars,
        "tokenize_rejections": state.engine.rejected_tokenize if state.engine is not None else 0,
    }
    if state.artifact is None:
        return RuleCheckInfo(
            available=False,
            mode="untrained_fallback",
            reason=state.reason,
            artifact_dir=str(state.artifact_dir),
            batching=batching,
        )
    manifest = state.artifact.manifest
    return RuleCheckInfo(
        available=True,
        mode="model",
        reason=None,
        artifact_dir=str(state.artifact_dir),
        model_version=manifest.model_version,
        kind=manifest.kind,
        variant=state.artifact.variant,
        backbone=manifest.backbone,
        inference=manifest.inference,
        metrics=manifest.metrics,
        provenance=manifest.provenance,
        batching=batching,
    )


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
        if runtime.rule_check is None:
            runtime.rule_check = RuleCheckRuntime(
                artifact_dir=settings.resolved_rule_check_dir,
                reason="rule-check loading was not configured for this runtime",
            )
        if runtime.rule_check.batcher is not None:
            runtime.rule_check.batcher.start()
        app.state.runtime = runtime
        logger.info(
            "model service ready",
            extra={
                "embedding_model": settings.embedding_model_id,
                "rerank": runtime.reranker is not None,
                "rule_check": "model" if runtime.rule_check.ready else "untrained_fallback",
            },
        )
        try:
            yield
        finally:
            if runtime.rule_check.batcher is not None:
                await runtime.rule_check.batcher.close()
            runtime.close()

    app = FastAPI(
        title="Model service (embeddings, rerank, rule-check model)",
        version="0.3.0",
        lifespan=lifespan,
    )

    @app.exception_handler(ServiceError)
    async def service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message)

    def runtime_of() -> Runtime:
        runtime: Runtime = app.state.runtime
        return runtime

    def rule_check_of() -> RuleCheckRuntime:
        state = runtime_of().rule_check
        assert state is not None  # set in lifespan  # noqa: S101
        return state

    @app.get("/health")
    async def health() -> Health:
        runtime = runtime_of()
        return Health(
            status="ok",
            service=SERVICE,
            embedding="ready",
            rerank="ready" if runtime.reranker is not None else "disabled",
            rule_check="ready" if rule_check_of().ready else "unavailable",
            rule_check_mode="model" if rule_check_of().ready else "untrained_fallback",
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
            rule_check=rule_check_info(rule_check_of(), s),
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

    @app.post(
        "/v1/check",
        responses={
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def check(request: CheckRequest) -> CheckResponse:
        state = rule_check_of()
        if state.engine is None or state.artifact is None:
            raise ServiceError(
                503, "rule_check_unavailable", f"{UNTRAINED_FALLBACK} Reason: {state.reason}"
            )
        limit = runtime_of().settings.rule_check_max_request_chars
        if request.total_chars() > limit:
            raise ServiceError(
                413,
                "request_too_large",
                f"hypothesis + spans total {request.total_chars()} characters; the limit is {limit}",
            )
        try:
            outcome = await state.engine.check(
                request.hypothesis, [SpanText(s.span_id, s.text) for s in request.spans]
            )
        except BatchOverloadedError as exc:
            raise ServiceError(503, "overloaded", BUSY) from exc
        except BatcherClosedError as exc:
            raise ServiceError(503, "shutting_down", str(exc)) from exc
        except InvalidModelOutputError as exc:
            logger.error("rule-check model produced invalid output", extra={"error": str(exc)})
            raise ServiceError(502, "invalid_model_output", str(exc)) from exc
        manifest = state.artifact.manifest
        return CheckResponse(
            label=outcome.label,  # type: ignore[arg-type]  # one of LABELS
            status=outcome.status,  # type: ignore[arg-type]
            confidence=outcome.confidence,
            probabilities=outcome.probabilities,
            evidence=[CheckEvidence(span_id=e.span_id, score=e.score) for e in outcome.evidence],
            considered=[
                CheckEvidence(span_id=e.span_id, score=e.score) for e in outcome.considered
            ],
            escalate=outcome.escalate,
            truncated_inputs=outcome.truncated_inputs,
            model_version=manifest.model_version,
            model_kind=manifest.kind,
            variant=state.artifact.variant,
        )

    return app
