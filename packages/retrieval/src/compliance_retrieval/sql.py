"""SQL for vector (pgvector HNSW) and keyword (full-text, ts_rank_cd) candidate search.

Every query is scoped to one workspace through ``documents.workspace_id`` and to READY
documents, optionally narrowed to document ids / document types; all values are bound
parameters (``ef_search`` included, via ``set_config``). Each function runs in its own
transaction with ``SET LOCAL``-equivalent settings so concurrent searches cannot affect
each other.

Encoder-space consistency: vector search only compares the query against documents
whose ``documents.embedding_model`` equals the query encoder's identity
(``model@revision``); documents embedded by another encoder are excluded from vector
ranking and reported to the caller (they need re-ingestion), never silently compared.

Vector search: ``ORDER BY embedding <=> :q LIMIT :k`` uses the HNSW cosine index
(``ix_chunks_embedding_hnsw``). Because HNSW scans the whole graph and filters afterwards,
pgvector's iterative scan (``hnsw.iterative_scan = strict_order``) keeps scanning until
``k`` rows pass the workspace/document filters, preserving exact distance order. On small
tables the planner may prefer an exact sequential scan, which returns the true nearest
neighbours (``evals/`` forces and verifies the index separately).

Keyword search ("or" mode, default for natural-language queries): the query is analysed
with the same ``english`` configuration as ``chunks.tsv`` and its lexemes are OR-ed, so
any matching term retrieves a chunk and ``ts_rank_cd`` (cover density) orders them.
"websearch" mode uses ``websearch_to_tsquery`` (quoted phrases, ``-`` exclusion, AND).
A query with no indexable terms (only stop words/punctuation) returns no keyword hits.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import BindParameter, Select, and_, bindparam, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.expression import ClauseElement, Executable

from compliance_core.db.models import Chunk, Document, DocumentStatus, TableRow
from compliance_retrieval.config import KeywordMode
from compliance_retrieval.fusion import Candidate


@dataclass(frozen=True)
class Scope:
    workspace_id: uuid.UUID
    document_ids: tuple[uuid.UUID, ...] = ()
    doc_types: tuple[str, ...] = ()


def _scope_filters(scope: Scope) -> list[Any]:
    filters: list[Any] = [
        Document.workspace_id == scope.workspace_id,
        Document.status == DocumentStatus.READY,
    ]
    if scope.document_ids:
        filters.append(Document.id.in_(scope.document_ids))
    if scope.doc_types:
        filters.append(Document.doc_type.in_(scope.doc_types))
    return filters


async def _local_settings(session: AsyncSession, settings: dict[str, str]) -> None:
    for name, value in settings.items():
        # set_config(..., is_local => true) == SET LOCAL, with a bound value.
        await session.execute(select(func.set_config(name, value, True)))


def vector_statement(
    scope: Scope, query_vector: list[float], *, encoder: str, k: int
) -> Select[Any]:
    """Nearest chunks by cosine distance (``<=>``, served by the HNSW index)."""
    distance = Chunk.embedding.cosine_distance(query_vector)
    return (
        select(Chunk.id, distance.label("distance"))
        .join(Document, Document.id == Chunk.document_id)
        .where(*_scope_filters(scope), Document.embedding_model == encoder)
        .order_by(distance)
        .limit(k)
    )


async def vector_candidates(
    session: AsyncSession,
    scope: Scope,
    query_vector: list[float],
    *,
    encoder: str,
    k: int,
    ef_search: int,
    statement_timeout_ms: int,
) -> list[Candidate]:
    async with session.begin():
        await _local_settings(
            session,
            {
                "hnsw.ef_search": str(ef_search),
                "hnsw.iterative_scan": "strict_order",
                "statement_timeout": str(statement_timeout_ms),
            },
        )
        rows = await session.execute(vector_statement(scope, query_vector, encoder=encoder, k=k))
        # Cosine similarity (higher is better) as the method-native score.
        return [Candidate(str(chunk_id), 1.0 - float(dist)) for chunk_id, dist in rows.all()]


async def mismatched_encoder_documents(
    session: AsyncSession, scope: Scope, encoder: str
) -> list[tuple[uuid.UUID, str, str | None]]:
    """READY documents in scope whose stored vectors come from a different encoder."""
    async with session.begin():
        rows = await session.execute(
            select(Document.id, Document.filename, Document.embedding_model)
            .where(
                *_scope_filters(scope),
                Document.embedding_model.is_distinct_from(encoder),
            )
            .order_by(Document.filename, Document.id)
        )
        return [(r.id, r.filename, r.embedding_model) for r in rows.all()]


_OR_TSQUERY = """
    to_tsquery('simple', coalesce(nullif(array_to_string(array(
        SELECT quote_literal(lexeme)
        FROM unnest(tsvector_to_array(to_tsvector('english', :query))) AS lexeme
    ), ' | '), ''), ''))
"""
_WEBSEARCH_TSQUERY = "websearch_to_tsquery('english', :query)"


async def keyword_candidates(
    session: AsyncSession,
    scope: Scope,
    query: str,
    *,
    k: int,
    mode: KeywordMode,
    statement_timeout_ms: int,
) -> list[Candidate]:
    tsquery = _OR_TSQUERY if mode == "or" else _WEBSEARCH_TSQUERY
    filters = ["d.workspace_id = :workspace_id", "d.status = 'ready'", "c.tsv @@ q.tsq"]
    params: dict[str, Any] = {"query": query, "workspace_id": scope.workspace_id, "k": k}
    binds: list[BindParameter[Any]] = []
    if scope.document_ids:
        filters.append("d.id IN :document_ids")
        params["document_ids"] = list(scope.document_ids)
        binds.append(bindparam("document_ids", expanding=True))
    if scope.doc_types:
        filters.append("d.doc_type::text IN :doc_types")
        params["doc_types"] = list(scope.doc_types)
        binds.append(bindparam("doc_types", expanding=True))
    statement = text(
        f"""
        WITH q AS (SELECT {tsquery} AS tsq)
        SELECT c.id, ts_rank_cd(c.tsv, q.tsq) AS rank
        FROM q, chunks c JOIN documents d ON d.id = c.document_id
        WHERE {" AND ".join(filters)}
        ORDER BY rank DESC, c.id
        LIMIT :k
        """  # noqa: S608 - only fixed SQL fragments are interpolated; all values are bound
    ).bindparams(*binds)
    async with session.begin():
        await _local_settings(session, {"statement_timeout": str(statement_timeout_ms)})
        rows = await session.execute(statement, params)
        return [Candidate(str(chunk_id), float(rank)) for chunk_id, rank in rows.all()]


class Explain(Executable, ClauseElement):
    """``EXPLAIN [ (ANALYZE) ]`` around a statement, with normally processed binds."""

    inherit_cache = False

    def __init__(self, statement: Executable, *, analyze: bool = False) -> None:
        self.statement = statement
        self.analyze = analyze


@compiles(Explain, "postgresql")
def _compile_explain(element: Explain, compiler: SQLCompiler, **kw: Any) -> str:
    prefix = "EXPLAIN (ANALYZE, BUFFERS) " if element.analyze else "EXPLAIN "
    return prefix + compiler.process(element.statement, **kw)  # type: ignore[arg-type]


async def explain(session: AsyncSession, statement: Executable, *, analyze: bool = False) -> str:
    rows = await session.execute(Explain(statement, analyze=analyze))
    return "\n".join(str(row[0]) for row in rows.all())


@dataclass(frozen=True)
class HydratedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    file_format: str
    doc_type: str | None
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    bboxes: list[dict[str, Any]]
    section_path: str | None
    token_count: int
    table_row_id: uuid.UUID | None
    sheet_name: str | None
    row_number: int | None


async def hydrate(
    session: AsyncSession, scope: Scope, chunk_ids: list[str]
) -> dict[str, HydratedChunk]:
    """Load stored chunk spans (never generated snippets), re-checking the workspace scope."""
    if not chunk_ids:
        return {}
    ids = [uuid.UUID(c) for c in chunk_ids]
    async with session.begin():
        rows = await session.execute(
            select(
                Chunk.id,
                Chunk.document_id,
                Document.filename,
                Document.file_format,
                Document.doc_type,
                Chunk.chunk_index,
                Chunk.text,
                Chunk.char_start,
                Chunk.char_end,
                Chunk.page_start,
                Chunk.page_end,
                Chunk.bbox_json,
                Chunk.section_path,
                Chunk.token_count,
                Chunk.table_row_id,
                TableRow.sheet_name,
                TableRow.row_index,
            )
            .join(Document, Document.id == Chunk.document_id)
            .outerjoin(TableRow, and_(TableRow.id == Chunk.table_row_id))
            .where(Chunk.id.in_(ids), *_scope_filters(scope))
        )
        out: dict[str, HydratedChunk] = {}
        for r in rows.all():
            out[str(r.id)] = HydratedChunk(
                chunk_id=r.id,
                document_id=r.document_id,
                filename=r.filename,
                file_format=r.file_format,
                doc_type=r.doc_type.value if r.doc_type is not None else None,
                chunk_index=r.chunk_index,
                text=r.text,
                char_start=r.char_start,
                char_end=r.char_end,
                page_start=r.page_start,
                page_end=r.page_end,
                bboxes=list(r.bbox_json),
                section_path=r.section_path,
                token_count=r.token_count,
                table_row_id=r.table_row_id,
                sheet_name=r.sheet_name,
                row_number=r.row_index,
            )
        return out
