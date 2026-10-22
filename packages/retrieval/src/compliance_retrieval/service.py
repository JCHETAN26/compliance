"""Hybrid search service shared by the REST API (and later the MCP server).

Flow: authorise (any workspace member, viewers included) -> validate filters -> embed the
query on the model service (vector/hybrid) -> run vector and keyword candidate queries
**in parallel, each in its own session/connection** -> fuse (weighted RRF) or rank a
single method -> optional cross-encoder rerank -> hydrate stored chunk spans.

Untrusted input: the query is only ever a bound SQL parameter or an HTTP JSON field; it
is never interpolated into SQL or HTML. Document ids outside the caller's workspace are
reported as "not found", exactly like non-existent ones.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from compliance_core.db.models import Document, WorkspaceRole
from compliance_core.document_schemas import BBoxRead
from compliance_core.model_client import (
    ModelServiceError,
    ModelServiceProtocolError,
    ModelServiceRequestError,
    QueryEmbedding,
    ServiceInfo,
)
from compliance_core.services.errors import InvalidOperationError, NotFoundError, ServiceError
from compliance_core.services.workspaces import require_role
from compliance_retrieval.config import RetrievalConfig
from compliance_retrieval.fusion import Candidate, FusedHit, reciprocal_rank_fusion, single_method
from compliance_retrieval.schemas import (
    MethodScore,
    Provenance,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SearchSettings,
    SearchTimings,
    StaleDocument,
)
from compliance_retrieval.sql import (
    HydratedChunk,
    Scope,
    hydrate,
    keyword_candidates,
    mismatched_encoder_documents,
    vector_candidates,
)

T = TypeVar("T")


class SearchUnavailableError(ServiceError):
    code = "search_unavailable"


class ModelGateway(Protocol):
    async def embed_query(self, query: str) -> QueryEmbedding: ...

    async def rerank(self, query: str, passages: list[tuple[str, str]]) -> dict[str, float]: ...

    async def info(self) -> ServiceInfo: ...


@dataclass
class SearchDependencies:
    session_factory: async_sessionmaker[AsyncSession]
    models: ModelGateway
    config: RetrievalConfig


async def _timed(fn: Callable[[], Awaitable[T]]) -> tuple[T, float]:
    started = time.perf_counter()
    result = await fn()
    return result, round((time.perf_counter() - started) * 1000, 2)


async def resolve_scope(
    session: AsyncSession, workspace_id: uuid.UUID, user_id: uuid.UUID, request: SearchRequest
) -> Scope:
    await require_role(session, workspace_id, user_id, frozenset(WorkspaceRole), "search documents")
    document_ids = tuple(dict.fromkeys(request.document_ids))
    if document_ids:
        found = set(
            await session.scalars(
                select(Document.id).where(
                    Document.workspace_id == workspace_id, Document.id.in_(document_ids)
                )
            )
        )
        if found != set(document_ids):
            raise NotFoundError("Document not found.")
    return Scope(
        workspace_id=workspace_id,
        document_ids=document_ids,
        doc_types=tuple(dict.fromkeys(t.value for t in request.doc_types)),
    )


async def run_search(
    deps: SearchDependencies,
    scope: Scope,
    request: SearchRequest,
) -> SearchResponse:
    """Execute a search for an already-authorised scope."""
    started = time.perf_counter()
    config = deps.config.with_top_k(request.top_k)
    use_vector = request.mode in ("hybrid", "vector") and (
        request.mode == "vector" or config.vector_weight > 0
    )
    use_keyword = request.mode in ("hybrid", "keyword") and (
        request.mode == "keyword" or config.keyword_weight > 0
    )
    timings: dict[str, float] = {}
    embedding: QueryEmbedding | None = None
    stale: list[tuple[uuid.UUID, str, str | None]] = []

    rerank_model: str | None = None
    if request.rerank:
        info = await model_call(deps.models.info)
        if not info.rerank_available:
            raise InvalidOperationError("Reranking is not enabled on this server.")
        rerank_model = f"{info.rerank_model}@{(info.rerank_revision or '')[:12]}"

    async def vector() -> list[Candidate]:
        nonlocal embedding, stale
        emb, timings["embed_ms"] = await _timed(
            lambda: model_call(lambda: deps.models.embed_query(request.query))
        )
        embedding = emb
        async with deps.session_factory() as session:
            result, timings["vector_ms"] = await _timed(
                lambda: vector_candidates(
                    session,
                    scope,
                    emb.vector,
                    encoder=emb.label,
                    k=config.candidate_k,
                    ef_search=config.ef_search,
                    statement_timeout_ms=config.statement_timeout_ms,
                )
            )
            stale = await mismatched_encoder_documents(session, scope, emb.label)
        return result

    async def keyword() -> list[Candidate]:
        async with deps.session_factory() as session:
            result, timings["keyword_ms"] = await _timed(
                lambda: keyword_candidates(
                    session,
                    scope,
                    request.query,
                    k=config.candidate_k,
                    mode=config.keyword_mode,
                    statement_timeout_ms=config.statement_timeout_ms,
                )
            )
        return result

    tasks: dict[str, asyncio.Task[list[Candidate]]] = {}
    try:
        async with asyncio.TaskGroup() as group:
            if use_vector:
                tasks["vector"] = group.create_task(vector())
            if use_keyword:
                tasks["keyword"] = group.create_task(keyword())
    except BaseExceptionGroup as group_error:
        # Surface the first real failure (the sibling task is cancelled by the group).
        raise group_error.exceptions[0] from None
    ranked = {method: task.result() for method, task in tasks.items()}

    if request.mode == "hybrid":
        hits = reciprocal_rank_fusion(
            ranked,
            {"vector": config.vector_weight, "keyword": config.keyword_weight},
            config.rrf_k,
        )
    else:
        hits = single_method(request.mode, ranked.get(request.mode, []))

    rerank_scores: dict[str, float] = {}
    pool = hits[: config.rerank_pool] if request.rerank else hits[: config.top_k]
    async with deps.session_factory() as session:
        hydrated = await hydrate(session, scope, [h.chunk_id for h in pool])
    pool = [h for h in pool if h.chunk_id in hydrated]  # documents deleted mid-search
    if request.rerank and pool:
        passages = [(h.chunk_id, hydrated[h.chunk_id].text) for h in pool]
        rerank_scores, timings["rerank_ms"] = await _timed(
            lambda: model_call(lambda: deps.models.rerank(request.query, passages))
        )
        position = {h.chunk_id: i for i, h in enumerate(pool)}
        pool.sort(key=lambda h: (-rerank_scores[h.chunk_id], position[h.chunk_id]))
    final = pool[: config.top_k]

    results = [
        _hit(rank, hit, hydrated[hit.chunk_id], request.mode, rerank_scores.get(hit.chunk_id))
        for rank, hit in enumerate(final, start=1)
    ]
    return SearchResponse(
        query=request.query,
        mode=request.mode,
        reranked=request.rerank,
        results=results,
        candidates={method: len(c) for method, c in ranked.items()},
        excluded_from_vector=[
            StaleDocument(document_id=doc_id, filename=name, embedding_model=model)
            for doc_id, name, model in stale
        ],
        settings=SearchSettings(
            top_k=config.top_k,
            candidate_k=config.candidate_k,
            ef_search=config.ef_search,
            rrf_k=config.rrf_k,
            vector_weight=config.vector_weight,
            keyword_weight=config.keyword_weight,
            keyword_mode=config.keyword_mode,
            rerank_candidates=config.rerank_pool if request.rerank else None,
            embedding_model=embedding.model if embedding else None,
            embedding_revision=embedding.revision if embedding else None,
            query_instruction=embedding.query_instruction if embedding else None,
            rerank_model=rerank_model,
        ),
        timings=SearchTimings(
            embed_ms=timings.get("embed_ms"),
            vector_ms=timings.get("vector_ms"),
            keyword_ms=timings.get("keyword_ms"),
            rerank_ms=timings.get("rerank_ms"),
            total_ms=round((time.perf_counter() - started) * 1000, 2),
        ),
    )


async def model_call(fn: Callable[[], Awaitable[T]]) -> T:
    try:
        return await fn()
    except ModelServiceRequestError as exc:
        if exc.code == "query_too_long":
            raise InvalidOperationError("The query is too long for the embedding model.") from exc
        if exc.code == "rerank_disabled":
            raise InvalidOperationError("Reranking is not enabled on this server.") from exc
        raise SearchUnavailableError(f"Search model request failed: {exc.message}") from exc
    except ModelServiceProtocolError as exc:
        raise SearchUnavailableError(
            "Search is temporarily unavailable: the model service returned an invalid response."
        ) from exc
    except ModelServiceError as exc:
        raise SearchUnavailableError(
            "Search is temporarily unavailable: the model service could not be reached."
        ) from exc


def _hit(
    rank: int, hit: FusedHit, chunk: HydratedChunk, mode: str, rerank_score: float | None
) -> SearchHit:
    return SearchHit(
        rank=rank,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        filename=chunk.filename,
        file_format=chunk.file_format,
        doc_type=chunk.doc_type,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        bboxes=[BBoxRead.model_validate(b) for b in chunk.bboxes],
        section_path=chunk.section_path,
        token_count=chunk.token_count,
        table_row_id=chunk.table_row_id,
        sheet_name=chunk.sheet_name,
        row_number=chunk.row_number,
        provenance=Provenance(
            vector=_method(hit, "vector"),
            keyword=_method(hit, "keyword"),
            fused_score=round(hit.fused_score, 6) if mode == "hybrid" else None,
            rerank_score=rerank_score,
        ),
    )


def _method(hit: FusedHit, method: str) -> MethodScore | None:
    if method not in hit.ranks:
        return None
    return MethodScore(rank=hit.ranks[method], score=round(hit.scores[method], 6))
