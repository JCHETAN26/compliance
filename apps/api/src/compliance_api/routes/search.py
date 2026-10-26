"""Hybrid search over a workspace's indexed chunks (any member, viewers included)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Response

from compliance_api.deps import Ctx, CurrentPrincipal, Session
from compliance_api.errors import error_responses
from compliance_retrieval.schemas import SearchCapabilities, SearchRequest, SearchResponse
from compliance_retrieval.service import SearchDependencies, model_call, search_for_user

router = APIRouter(tags=["search"])


@router.post(
    "/workspaces/{workspace_id}/search",
    responses=error_responses(401, 403, 404, 409, 422, 429, 503),
)
async def search(
    workspace_id: uuid.UUID,
    payload: SearchRequest,
    principal: CurrentPrincipal,
    session: Session,
    ctx: Ctx,
    response: Response,
) -> SearchResponse:
    """Hybrid (pgvector HNSW + full-text ts_rank_cd, RRF-fused), vector-only or
    keyword-only search. Results are stored chunk spans with exact document, page,
    character, bounding-box and spreadsheet-row references."""
    response.headers["Cache-Control"] = "no-store"
    deps = SearchDependencies(
        session_factory=ctx.session_factory, models=ctx.models, config=ctx.retrieval
    )
    return await search_for_user(
        session,
        deps,
        ctx.search_limiter,
        workspace_id=workspace_id,
        user_id=principal.user.id,
        request=payload,
    )


@router.get("/search/capabilities", responses=error_responses(401, 503))
async def capabilities(principal: CurrentPrincipal, ctx: Ctx) -> SearchCapabilities:
    """What the search backend supports right now (e.g. whether reranking is enabled)."""
    del principal
    info = await model_call(ctx.models.info)
    return SearchCapabilities(
        modes=["hybrid", "vector", "keyword"],
        rerank_available=info.rerank_available,
        embedding_model=info.embedding_model,
        embedding_revision=info.embedding_revision,
        query_instruction=info.query_instruction,
        rerank_model=info.rerank_model,
        default_top_k=ctx.retrieval.top_k,
        max_top_k=50,
    )
