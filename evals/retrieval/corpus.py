"""Isolated evaluation database: create, migrate and load the ContractNLI corpus.

Isolation: the eval only ever uses a database named ``compliance_eval`` or
``compliance_eval_<[a-z0-9_]>`` (default ``compliance_eval`` on the throwaway test
Postgres at 127.0.0.1:55432; override with ``EVAL_DATABASE_URL``). The name is
validated inside every function that creates or drops it. A database is only dropped
if it carries the evaluator's ownership marker (table ``eval_corpus_meta`` with
``managed_by``), written immediately after the evaluator creates it, so an unmanaged
database is never destroyed even if its name matches. Runs hold a per-database
advisory lock (``EvalLock``), so concurrent runs cannot reset each other. The database
is migrated with the production Alembic migrations (same HNSW/GIN indexes); the app
and integration-test databases are never touched.

The corpus is cached across runs: a fingerprint of the dataset archive, the encoder
pin, the chunking options and this loader's code is stored in the eval database, and
a matching database is reused instead of re-embedding everything.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg
from sqlalchemy import insert, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from compliance_core.db import migrate
from compliance_core.db.models import (
    Chunk,
    DocType,
    Document,
    DocumentContent,
    DocumentStatus,
    User,
    Workspace,
    WorkspaceMember,
    WorkspaceRole,
)
from compliance_ingestion.embeddings import Embedder
from evals.retrieval.data import EvalDocument, SplitData

DEFAULT_URL = "postgresql+asyncpg://compliance:compliance@127.0.0.1:55432/compliance_eval"
# Only these names are ever used, created or dropped: lowercase letters, digits and
# underscores (no quoting/escaping needed), at most 46 characters.
SAFE_NAME = re.compile(r"compliance_eval(?:_[a-z0-9_]{1,30})?")
META_TABLE = "eval_corpus_meta"
MANAGED_BY = "compliance-evals/retrieval"


class IsolationError(RuntimeError):
    pass


def safe_database_name(url: str) -> str:
    """The URL's database name, if it is a database this evaluator may manage."""
    name = make_url(url).database or ""
    if not SAFE_NAME.fullmatch(name):
        raise IsolationError(
            f"refusing to use database {name!r}: the eval only uses databases named "
            "compliance_eval or compliance_eval_<lowercase letters, digits, _>"
        )
    return name


def eval_database_url() -> str:
    url = os.environ.get("EVAL_DATABASE_URL", DEFAULT_URL)
    safe_database_name(url)
    return url


async def _connect(url: str, database: str) -> asyncpg.Connection:
    parsed = make_url(url)
    return await asyncpg.connect(
        user=parsed.username,
        password=parsed.password,
        host=parsed.host,
        port=parsed.port,
        database=database,
    )


async def _exists(url: str, name: str) -> bool:
    admin = await _connect(url, "postgres")
    try:
        return bool(await admin.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name))
    finally:
        await admin.close()


async def _meta(url: str, name: str) -> dict[str, str] | None:
    """The evaluator's marker table, or None if the database is not managed by it."""
    conn = await _connect(url, name)
    try:
        if await conn.fetchval("SELECT to_regclass($1)", META_TABLE) is None:
            return None
        rows = await conn.fetch("SELECT key, value FROM eval_corpus_meta")
        meta = {r["key"]: r["value"] for r in rows}
        return meta if meta.get("managed_by") == MANAGED_BY else None
    finally:
        await conn.close()


class EvalLock:
    """Exclusive per-database advisory lock held for a whole eval/sweep run, so two runs
    can never reset or re-index the same evaluation database concurrently."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.name = safe_database_name(url)
        self._conn: asyncpg.Connection | None = None

    async def __aenter__(self) -> EvalLock:
        conn = await _connect(self.url, "postgres")
        key = f"compliance-evals:{self.name}"
        if not await conn.fetchval("SELECT pg_try_advisory_lock(hashtext($1))", key):
            await conn.close()
            raise IsolationError(f"another evaluation run is using database {self.name!r}")
        self._conn = conn
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._conn is not None:
            await self._conn.close()  # session end releases the advisory lock


async def stored_fingerprint(url: str) -> str | None:
    name = safe_database_name(url)
    if not await _exists(url, name):
        return None
    meta = await _meta(url, name)
    return meta.get("fingerprint") if meta else None


async def recreate_database(url: str) -> None:
    """Drop (only if evaluator-managed) and recreate the eval database, then migrate it.

    The name is validated here, not only by callers. An existing database without the
    evaluator's marker is never dropped, whatever its name.
    """
    name = safe_database_name(url)
    if await _exists(url, name) and await _meta(url, name) is None:
        raise IsolationError(
            f"database {name!r} exists but was not created by this evaluator "
            f"(no {META_TABLE} marker); refusing to drop it"
        )
    admin = await _connect(url, "postgres")
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()
    conn = await _connect(url, name)
    try:  # mark ownership before anything else can fail
        await conn.execute(
            "CREATE TABLE eval_corpus_meta (key text PRIMARY KEY, value text NOT NULL)"
        )
        await conn.execute("INSERT INTO eval_corpus_meta VALUES ('managed_by', $1)", MANAGED_BY)
    finally:
        await conn.close()
    await asyncio.to_thread(migrate.upgrade, url, "head")


def corpus_fingerprint(parts: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class StoredDocument:
    workspace_id: uuid.UUID
    document_id: uuid.UUID


@dataclass(frozen=True)
class LoadedCorpus:
    # (split, ContractNLI doc id) -> stored ids
    documents: dict[tuple[str, str], StoredDocument]
    chunk_counts: dict[str, int]  # split -> stored chunks


async def load_corpus(
    url: str,
    splits: Sequence[SplitData],
    embedder: Embedder,
    encoder_label: str,
    fingerprint: str,
) -> None:
    engine = create_async_engine(url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as session:
            user = User(email="eval@contractnli.invalid", password_hash="!")  # noqa: S106 - not a hash: cannot log in
            session.add(user)
            await session.flush()
            for data in splits:
                # One workspace per split. The schema keeps document hashes unique per
                # workspace, so an exact duplicate text (one pair of documents in the
                # official test split) goes to an overflow workspace of the same split.
                seen: list[tuple[uuid.UUID, set[str]]] = []
                for document in data.documents.values():
                    digest = hashlib.sha256(document.text.encode()).hexdigest()
                    slot = next((s for s in seen if digest not in s[1]), None)
                    if slot is None:
                        suffix = "" if not seen else f"#{len(seen) + 1}"
                        workspace = Workspace(
                            name=f"ContractNLI {data.split}{suffix}", owner_id=user.id
                        )
                        session.add(workspace)
                        await session.flush()
                        session.add(
                            WorkspaceMember(
                                workspace_id=workspace.id,
                                user_id=user.id,
                                role=WorkspaceRole.OWNER,
                            )
                        )
                        slot = (workspace.id, set())
                        seen.append(slot)
                    slot[1].add(digest)
                    await _load_document(session, slot[0], document, embedder, encoder_label)
                await session.commit()
                print(f"  loaded {data.split}: {len(data.documents)} documents")  # noqa: T201
            # Written last: a partially loaded corpus has no fingerprint and is rebuilt.
            await session.execute(
                text("INSERT INTO eval_corpus_meta VALUES ('fingerprint', :value)"),
                {"value": fingerprint},
            )
            await session.commit()
        async with engine.begin() as conn:
            await conn.execute(text("ANALYZE"))
    finally:
        await engine.dispose()


async def _load_document(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    document: EvalDocument,
    embedder: Embedder,
    encoder_label: str,
) -> None:
    data = document.text.encode()
    digest = hashlib.sha256(data).hexdigest()
    doc_id = uuid.uuid4()
    session.add(
        Document(
            id=doc_id,
            workspace_id=workspace_id,
            filename=f"contractnli-{document.split}-{document.doc_id}.txt",
            mime_type="text/plain",
            file_format="txt",
            storage_key=f"eval/{document.split}/{document.doc_id}",  # no stored original
            sha256=digest,
            size_bytes=len(data),
            status=DocumentStatus.READY,
            stage=None,
            progress=100,
            doc_type=DocType.CONTRACT,
            chunk_count=len(document.chunks),
            embedding_model=encoder_label,
        )
    )
    session.add(DocumentContent(document_id=doc_id, text=document.text, text_sha256=digest))
    await session.flush()
    embedded = await asyncio.to_thread(embedder.embed, [c.text for c in document.chunks])
    rows = [
        {
            "document_id": doc_id,
            "chunk_index": chunk.index,
            "text": chunk.text,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "section_path": chunk.section_path,
            "token_count": chunk.token_count,
            "embedding": vector.vector,
            "embedding_windows": vector.windows,
        }
        for chunk, vector in zip(document.chunks, embedded, strict=True)
    ]
    if rows:
        await session.execute(insert(Chunk), rows)


async def corpus_index(url: str) -> LoadedCorpus:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            documents: dict[tuple[str, str], StoredDocument] = {}
            counts: dict[str, int] = {}
            rows = await conn.execute(
                text("SELECT d.id, d.workspace_id, d.filename, d.chunk_count FROM documents d")
            )
            for doc_id, workspace_id, filename, chunk_count in rows.all():
                _, split, cnli_id = str(filename).removesuffix(".txt").split("-", 2)
                documents[(split, cnli_id)] = StoredDocument(workspace_id, doc_id)
                counts[split] = counts.get(split, 0) + int(chunk_count)
        return LoadedCorpus(documents, counts)
    finally:
        await engine.dispose()


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_size=4)


def code_hash(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
