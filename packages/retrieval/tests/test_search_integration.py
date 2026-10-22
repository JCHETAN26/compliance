# mypy: disallow-untyped-calls=False
# (fixtures are built with PyMuPDF/openpyxl, which ship partial annotations)
"""Hybrid search against real Postgres/pgvector, the real ingestion pipeline and the
real pinned encoder/reranker served by the model service (in-process over HTTP)."""

from __future__ import annotations

import asyncio
import io
import uuid
from dataclasses import dataclass, replace
from typing import Any

import pymupdf
import pytest
from openpyxl import Workbook
from sqlalchemy import func, select, text, update

from compliance_core.db.models import (
    Chunk,
    Document,
    DocumentContent,
    DocumentStatus,
    User,
    Workspace,
    WorkspaceMember,
    WorkspaceRole,
)
from compliance_core.model_client import (
    ModelServiceClient,
    ModelServiceUnavailableError,
    QueryEmbedding,
    ServiceInfo,
)
from compliance_core.services.errors import InvalidOperationError, NotFoundError
from compliance_retrieval.config import RetrievalConfig
from compliance_retrieval.schemas import SearchRequest, SearchResponse
from compliance_retrieval.service import (
    SearchDependencies,
    SearchUnavailableError,
    resolve_scope,
    run_search,
)
from compliance_retrieval.sql import Scope, explain, keyword_candidates, vector_statement

pytestmark = pytest.mark.integration

PAGE_ONE = [
    "MUTUAL NON-DISCLOSURE AGREEMENT",
    "1. Confidential Information. The Recipient shall keep all disclosed information secret",
    "and use it only to evaluate the proposed business relationship between the parties.",
    "2. Governing Law. This Agreement is governed by the laws of the State of Delaware",
    "and disputes are resolved exclusively in the courts located in Wilmington.",
]
PAGE_TWO = [
    "3. Term. The obligations of confidentiality survive termination of this Agreement",
    "for a period of five years after the date of the last disclosure.",
    "4. Return of Materials. On written request the Recipient shall promptly return",
    "or destroy all copies of Confidential Information in its possession.",
]


def _pdf() -> bytes:
    doc = pymupdf.open()
    for lines in (PAGE_ONE, PAGE_TWO):
        page = doc.new_page()
        y = 72.0
        for line in lines:
            page.insert_text((72, y), line, fontsize=10)
            y += 16
    return bytes(doc.tobytes())


def _xlsx() -> bytes:
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Vendors"
    ws.append(["Vendor", "Service", "Data access", "Region"])
    ws.append(["Acme Logistics", "Freight forwarding", "Customer shipping addresses", "EU"])
    ws.append(["Borealis Cloud", "Database hosting", "Production databases", "US"])
    ws.append(["Cobalt Payroll", "Payroll processing", "Employee bank account details", "UK"])
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


OTHER_CSV = b"Vendor,Notes\nZephyr Quantum,Holds the xylophone escrow key for Borealis Cloud\n"


@dataclass
class Corpus:
    workspace: uuid.UUID
    owner: uuid.UUID
    viewer: uuid.UUID
    outsider: uuid.UUID
    other_workspace: uuid.UUID
    pdf: uuid.UUID
    xlsx: uuid.UUID
    other_doc: uuid.UUID
    session_factory: Any


async def _user(session: Any, name: str) -> uuid.UUID:
    user = User(email=f"{name}-{uuid.uuid4().hex[:6]}@example.com", password_hash="x")
    session.add(user)
    await session.flush()
    return user.id


@pytest.fixture(scope="module")
async def corpus(api_app: Any, ingest_ready: Any) -> Corpus:
    ctx = api_app.state.ctx
    async with ctx.engine.begin() as conn:
        await conn.exec_driver_sql(
            "TRUNCATE documents, workspace_members, workspaces, users RESTART IDENTITY CASCADE"
        )
    async with ctx.session_factory() as session:
        owner = await _user(session, "owner")
        viewer = await _user(session, "viewer")
        outsider = await _user(session, "outsider")
        ws = Workspace(name="A", owner_id=owner)
        other = Workspace(name="B", owner_id=outsider)
        session.add_all([ws, other])
        await session.flush()
        session.add_all(
            [
                WorkspaceMember(workspace_id=ws.id, user_id=owner, role=WorkspaceRole.OWNER),
                WorkspaceMember(workspace_id=ws.id, user_id=viewer, role=WorkspaceRole.VIEWER),
                WorkspaceMember(workspace_id=other.id, user_id=outsider, role=WorkspaceRole.OWNER),
            ]
        )
        await session.commit()
        ws_id, other_id = ws.id, other.id
    pdf = await ingest_ready(ws_id, owner, "nda.pdf", _pdf(), "pdf")
    xlsx = await ingest_ready(ws_id, owner, "vendors.xlsx", _xlsx(), "xlsx")
    other_doc = await ingest_ready(other_id, outsider, "secret.csv", OTHER_CSV, "csv")
    return Corpus(
        ws_id, owner, viewer, outsider, other_id, pdf, xlsx, other_doc, ctx.session_factory
    )


@pytest.fixture(scope="module")
def deps(corpus: Corpus, model_client: ModelServiceClient) -> SearchDependencies:
    return SearchDependencies(
        session_factory=corpus.session_factory,
        models=model_client,
        config=RetrievalConfig(candidate_k=50, ef_search=100),
    )


async def _search(
    deps: SearchDependencies, corpus: Corpus, user: uuid.UUID | None = None, **kwargs: Any
) -> SearchResponse:
    request = SearchRequest(**kwargs)
    async with corpus.session_factory() as session:
        scope = await resolve_scope(session, corpus.workspace, user or corpus.owner, request)
    return await run_search(deps, scope, request)


async def test_hybrid_results_are_exact_stored_spans(
    deps: SearchDependencies, corpus: Corpus
) -> None:
    response = await _search(
        deps, corpus, query="How long do confidentiality obligations survive termination?"
    )
    assert response.results, "expected results"
    assert response.candidates.keys() == {"vector", "keyword"}
    top = response.results[0]
    assert top.document_id == corpus.pdf
    assert "five years" in top.text
    assert top.page_start == 2  # a chunk on the later page, with real coordinates
    assert top.bboxes
    assert all(b.page == 2 for b in top.bboxes)
    assert top.provenance.vector is not None
    assert top.provenance.keyword is not None
    assert top.provenance.fused_score is not None
    async with corpus.session_factory() as session:
        for hit in response.results:
            content = await session.get(DocumentContent, hit.document_id)
            chunk = await session.get(Chunk, hit.chunk_id)
            assert content is not None
            assert chunk is not None
            assert hit.text == chunk.text == content.text[hit.char_start : hit.char_end]
    ranks = [hit.rank for hit in response.results]
    assert ranks == list(range(1, len(ranks) + 1))
    fused = [hit.provenance.fused_score or 0 for hit in response.results]
    assert fused == sorted(fused, reverse=True)
    assert response.settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert response.settings.query_instruction
    assert response.excluded_from_vector == []


async def test_spreadsheet_row_result_has_sheet_reference(
    deps: SearchDependencies, corpus: Corpus
) -> None:
    response = await _search(
        deps, corpus, query="which vendor has access to employee bank account details"
    )
    row = next(hit for hit in response.results if hit.document_id == corpus.xlsx)
    assert response.results.index(row) == 0
    assert "Cobalt Payroll" in row.text
    assert row.sheet_name == "Vendors"
    assert row.row_number == 4
    assert row.table_row_id is not None
    assert row.file_format == "xlsx"


@pytest.mark.parametrize("mode", ["hybrid", "vector", "keyword"])
async def test_no_cross_workspace_leakage(
    deps: SearchDependencies, corpus: Corpus, mode: str
) -> None:
    response = await _search(
        deps, corpus, query="Zephyr Quantum xylophone escrow key", mode=mode, top_k=50
    )
    assert all(hit.document_id != corpus.other_doc for hit in response.results)
    assert all(hit.document_id in {corpus.pdf, corpus.xlsx} for hit in response.results)


async def test_scope_rejects_outsiders_and_foreign_documents(corpus: Corpus) -> None:
    async with corpus.session_factory() as session:
        with pytest.raises(NotFoundError):
            await resolve_scope(
                session, corpus.workspace, corpus.outsider, SearchRequest(query="secret")
            )
        with pytest.raises(NotFoundError, match="Document not found"):
            await resolve_scope(
                session,
                corpus.workspace,
                corpus.owner,
                SearchRequest(query="secret", document_ids=[corpus.pdf, corpus.other_doc]),
            )
        with pytest.raises(NotFoundError, match="Document not found"):
            await resolve_scope(
                session,
                corpus.workspace,
                corpus.owner,
                SearchRequest(query="secret", document_ids=[uuid.uuid4()]),
            )


async def test_viewers_can_search(deps: SearchDependencies, corpus: Corpus) -> None:
    response = await _search(deps, corpus, user=corpus.viewer, query="governing law Delaware")
    assert response.results[0].document_id == corpus.pdf


async def test_document_and_type_filters(deps: SearchDependencies, corpus: Corpus) -> None:
    only_xlsx = await _search(
        deps, corpus, query="confidential information", document_ids=[corpus.xlsx], top_k=50
    )
    assert only_xlsx.results
    assert {hit.document_id for hit in only_xlsx.results} == {corpus.xlsx}
    sheets = await _search(
        deps, corpus, query="confidential information", doc_types=["spreadsheet"], top_k=50
    )
    assert {hit.document_id for hit in sheets.results} == {corpus.xlsx}
    contracts = await _search(deps, corpus, query="vendor", doc_types=["policy"], top_k=50)
    assert contracts.results == []


async def test_keyword_mode_is_lexical_and_vector_mode_has_no_keyword(
    deps: SearchDependencies, corpus: Corpus
) -> None:
    keyword = await _search(deps, corpus, query="Wilmington", mode="keyword")
    assert keyword.candidates == {"keyword": 1}
    assert "Wilmington" in keyword.results[0].text
    assert keyword.results[0].provenance.vector is None
    assert keyword.results[0].provenance.fused_score is None
    assert keyword.settings.embedding_model is None  # no model call in keyword mode
    vector = await _search(deps, corpus, query="Wilmington", mode="vector", top_k=3)
    assert vector.candidates.keys() == {"vector"}
    assert all(hit.provenance.keyword is None for hit in vector.results)
    scores = [hit.provenance.vector.score for hit in vector.results if hit.provenance.vector]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize(
    "query",
    [
        "'; DROP TABLE chunks; --",
        "a & b | !c <-> d:*",
        "((( ')) \\ \x01",
        '"governing law" -Delaware OR return',
        "the and of",  # only stop words: no lexemes
    ],
)
async def test_hostile_queries_are_plain_data(
    deps: SearchDependencies, corpus: Corpus, query: str
) -> None:
    for mode in ("or", "websearch"):
        config = replace(deps.config, keyword_mode=mode)
        await run_search(
            replace(deps, config=config),
            Scope(corpus.workspace),
            SearchRequest(query=query, mode="hybrid"),
        )
    async with corpus.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Chunk)) > 0


async def test_stop_word_query_has_no_keyword_candidates(corpus: Corpus) -> None:
    async with corpus.session_factory() as session:
        for mode in ("or", "websearch"):
            hits = await keyword_candidates(
                session,
                Scope(corpus.workspace),
                "the and of",
                k=10,
                mode=mode,
                statement_timeout_ms=5000,
            )
            assert hits == []


async def test_websearch_mode_requires_all_terms(corpus: Corpus) -> None:
    async with corpus.session_factory() as session:
        loose = await keyword_candidates(
            session,
            Scope(corpus.workspace),
            "Delaware payroll",
            k=10,
            mode="or",
            statement_timeout_ms=5000,
        )
        strict = await keyword_candidates(
            session,
            Scope(corpus.workspace),
            "Delaware payroll",
            k=10,
            mode="websearch",
            statement_timeout_ms=5000,
        )
    assert len(loose) >= 2
    assert strict == []


async def test_vector_and_keyword_run_concurrently(
    deps: SearchDependencies, corpus: Corpus, model_client: ModelServiceClient
) -> None:
    """The query embedding waits until the keyword query has opened its own session; a
    sequential implementation would time out here."""
    opened = asyncio.Event()

    def factory() -> Any:
        opened.set()
        return corpus.session_factory()

    class Gateway:
        async def embed_query(self, query: str) -> QueryEmbedding:
            await asyncio.wait_for(opened.wait(), timeout=5)
            return await model_client.embed_query(query)

        async def rerank(self, query: str, passages: list[tuple[str, str]]) -> dict[str, float]:
            raise AssertionError("not called")

        async def info(self) -> ServiceInfo:
            return await model_client.info()

    response = await run_search(
        SearchDependencies(session_factory=factory, models=Gateway(), config=deps.config),  # type: ignore[arg-type]
        Scope(corpus.workspace),
        SearchRequest(query="return or destroy copies"),
    )
    assert response.results


async def test_rerank_uses_cross_encoder_scores(deps: SearchDependencies, corpus: Corpus) -> None:
    response = await _search(
        deps, corpus, query="how long do confidentiality obligations last", rerank=True, top_k=3
    )
    assert response.reranked
    assert response.settings.rerank_model is not None
    assert response.settings.rerank_model.startswith("cross-encoder/ms-marco-MiniLM-L6-v2@")
    scores = [hit.provenance.rerank_score for hit in response.results]
    assert all(score is not None for score in scores)
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]
    assert response.timings.rerank_ms is not None


class _Unavailable:
    def __init__(self, rerank_available: bool = True) -> None:
        self.rerank_available = rerank_available

    async def embed_query(self, query: str) -> QueryEmbedding:
        raise ModelServiceUnavailableError("down", "model_service_unavailable")

    async def rerank(self, query: str, passages: list[tuple[str, str]]) -> dict[str, float]:
        raise ModelServiceUnavailableError("down", "model_service_unavailable")

    async def info(self) -> ServiceInfo:
        return ServiceInfo("m", "r", "q", self.rerank_available, None, None)


async def test_model_outage_fails_clearly_and_keyword_mode_still_works(
    deps: SearchDependencies, corpus: Corpus
) -> None:
    down = replace(deps, models=_Unavailable())
    with pytest.raises(SearchUnavailableError, match="temporarily unavailable"):
        await run_search(down, Scope(corpus.workspace), SearchRequest(query="governing law"))
    keyword = await run_search(
        down, Scope(corpus.workspace), SearchRequest(query="governing law", mode="keyword")
    )
    assert keyword.results
    with pytest.raises(SearchUnavailableError):  # rerank requested and available, but fails
        await run_search(
            down,
            Scope(corpus.workspace),
            SearchRequest(query="governing law", mode="keyword", rerank=True),
        )


async def test_rerank_requested_but_disabled_is_an_error(
    deps: SearchDependencies, corpus: Corpus
) -> None:
    disabled = replace(deps, models=_Unavailable(rerank_available=False))
    with pytest.raises(InvalidOperationError, match="Reranking is not enabled"):
        await run_search(
            disabled,
            Scope(corpus.workspace),
            SearchRequest(query="governing law", mode="keyword", rerank=True),
        )


async def test_mismatched_encoder_documents_are_excluded_and_reported(
    deps: SearchDependencies, corpus: Corpus
) -> None:
    async with corpus.session_factory() as session:
        await session.execute(
            update(Document)
            .where(Document.id == corpus.xlsx)
            .values(embedding_model="intfloat/e5-small-v2@ffffffffffff")
        )
        await session.commit()
    try:
        response = await _search(
            deps, corpus, query="employee bank account details", mode="vector", top_k=50
        )
        assert all(hit.document_id != corpus.xlsx for hit in response.results)
        assert [(d.document_id, d.embedding_model) for d in response.excluded_from_vector] == [
            (corpus.xlsx, "intfloat/e5-small-v2@ffffffffffff")
        ]
        keyword = await _search(deps, corpus, query="Cobalt Payroll", mode="keyword")
        assert keyword.results[0].document_id == corpus.xlsx  # still keyword-searchable
        assert keyword.excluded_from_vector == []  # no encoder involved in keyword mode
    finally:
        async with corpus.session_factory() as session:
            await session.execute(
                update(Document)
                .where(Document.id == corpus.xlsx)
                .values(embedding_model=(await _label(session, corpus.pdf)))
            )
            await session.commit()


async def _label(session: Any, doc_id: uuid.UUID) -> str | None:
    label: str | None = await session.scalar(
        select(Document.embedding_model).where(Document.id == doc_id)
    )
    return label


async def test_only_ready_documents_are_searchable(
    deps: SearchDependencies, corpus: Corpus
) -> None:
    async with corpus.session_factory() as session:
        await session.execute(
            update(Document)
            .where(Document.id == corpus.pdf)
            .values(status=DocumentStatus.PROCESSING)
        )
        await session.commit()
    try:
        response = await _search(deps, corpus, query="governing law Delaware", top_k=50)
        assert all(hit.document_id != corpus.pdf for hit in response.results)
    finally:
        async with corpus.session_factory() as session:
            await session.execute(
                update(Document)
                .where(Document.id == corpus.pdf)
                .values(status=DocumentStatus.READY)
            )
            await session.commit()


async def test_vector_query_plan_uses_hnsw_index(
    corpus: Corpus, model_client: ModelServiceClient
) -> None:
    embedding = await model_client.embed_query("confidentiality term")
    statement = vector_statement(
        Scope(corpus.workspace), embedding.vector, encoder=embedding.label, k=10
    )
    async with corpus.session_factory() as session, session.begin():
        # Tiny tables favour an exact scan; disable it to prove the index serves the query.
        await session.execute(text("SET LOCAL enable_seqscan = off"))
        await session.execute(text("SET LOCAL enable_sort = off"))
        rendered = await explain(session, statement)
    assert "ix_chunks_embedding_hnsw" in rendered, rendered
