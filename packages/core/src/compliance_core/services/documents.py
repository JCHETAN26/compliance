"""Document service: upload registration, listing, retrieval, deletion and retry.

Authorisation: any workspace member may read documents. Only ``owner`` and
``reviewer`` may upload, delete or retry them (viewers are read-only). Documents in a
workspace the caller does not belong to are reported as not found.

Idempotence: a document is identified by ``(workspace_id, sha256)``. Re-uploading
identical bytes into the same workspace returns the existing document instead of
creating (and re-ingesting) a copy; the same file in another workspace is independent.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_core.db.models import (
    Chunk,
    DocType,
    Document,
    DocumentBlock,
    DocumentContent,
    DocumentStatus,
    ExtractedEntity,
    Page,
    TableRow,
    TableSheet,
    WorkspaceMember,
    WorkspaceRole,
)
from compliance_core.document_schemas import (
    BBoxRead,
    BlockRead,
    CellRead,
    ChunkPage,
    ChunkRead,
    DocumentRead,
    EntitiesResponse,
    EntityRead,
    PageRead,
    RowPage,
    RowRead,
    SheetRead,
)
from compliance_core.files import MIME_TYPES, FileFormat
from compliance_core.services.errors import (
    InvalidOperationError,
    NotFoundError,
    PermissionDeniedError,
)
from compliance_core.services.workspaces import require_role
from compliance_core.storage import Storage, document_key

READ_ROLES = frozenset(WorkspaceRole)
WRITE_ROLES = frozenset({WorkspaceRole.OWNER, WorkspaceRole.REVIEWER})
DOCUMENT_NOT_FOUND = "Document not found."
ENTITIES_NOT_AVAILABLE = (
    "Entities are extracted by the extraction agent during review runs. No review has "
    "extracted entities from this document yet."
)


@dataclass(frozen=True)
class RegisteredDocument:
    document: DocumentRead
    created: bool


async def require_writer(
    session: AsyncSession, workspace_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    await require_role(session, workspace_id, user_id, WRITE_ROLES, "upload or modify documents")


async def _document_for_member(
    session: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID, *, lock: bool = False
) -> tuple[Document, WorkspaceRole]:
    query = (
        select(Document, WorkspaceMember.role)
        .join(WorkspaceMember, WorkspaceMember.workspace_id == Document.workspace_id)
        .where(Document.id == document_id, WorkspaceMember.user_id == user_id)
    )
    if lock:
        query = query.with_for_update(of=Document)
    row = (await session.execute(query)).one_or_none()
    if row is None:
        raise NotFoundError(DOCUMENT_NOT_FOUND)
    return row[0], row[1]


def _require_write(role: WorkspaceRole, action: str) -> None:
    if role not in WRITE_ROLES:
        raise PermissionDeniedError(f"Your role ({role.value}) does not allow you to {action}.")


async def register_upload(
    session: AsyncSession,
    storage: Storage,
    *,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID | None,
    filename: str,
    fmt: FileFormat,
    staged_path: Path,
    sha256: str,
    size: int,
) -> RegisteredDocument:
    """Store a verified upload and create its document row (or return the duplicate).

    The caller must already have checked the user's write permission. The staged file
    is moved into storage (or deleted for duplicates); on failure nothing is left behind.
    """
    existing = await session.scalar(
        select(Document).where(Document.workspace_id == workspace_id, Document.sha256 == sha256)
    )
    if existing is not None:
        staged_path.unlink(missing_ok=True)
        return RegisteredDocument(DocumentRead.model_validate(existing), created=False)

    document_id = uuid.uuid4()
    key = document_key(workspace_id, document_id)
    storage.put(key, staged_path)
    document = Document(
        id=document_id,
        workspace_id=workspace_id,
        uploaded_by=user_id,
        filename=filename,
        mime_type=MIME_TYPES[fmt],
        file_format=fmt,
        storage_key=key,
        sha256=sha256,
        size_bytes=size,
        status=DocumentStatus.UPLOADED,
        stage="queued",
        doc_type=DocType.SPREADSHEET if fmt in ("xlsx", "csv") else None,
    )
    session.add(document)
    try:
        await session.flush()
        await session.refresh(document)
        await session.commit()
    except IntegrityError:
        # Concurrent upload of the same bytes into the same workspace won the race.
        await session.rollback()
        storage.delete(key)
        winner = await session.scalar(
            select(Document).where(Document.workspace_id == workspace_id, Document.sha256 == sha256)
        )
        if winner is None:
            raise
        return RegisteredDocument(DocumentRead.model_validate(winner), created=False)
    except BaseException:
        await session.rollback()
        storage.delete(key)
        raise
    return RegisteredDocument(DocumentRead.model_validate(document), created=True)


async def list_documents(
    session: AsyncSession, workspace_id: uuid.UUID, user_id: uuid.UUID
) -> list[DocumentRead]:
    await require_role(session, workspace_id, user_id, READ_ROLES, "view documents")
    rows = await session.scalars(
        select(Document)
        .where(Document.workspace_id == workspace_id)
        .order_by(Document.created_at.desc(), Document.id)
    )
    return [DocumentRead.model_validate(d) for d in rows]


async def document_counts(
    session: AsyncSession, workspace_ids: list[uuid.UUID]
) -> dict[uuid.UUID, int]:
    if not workspace_ids:
        return {}
    rows = await session.execute(
        select(Document.workspace_id, func.count())
        .where(Document.workspace_id.in_(workspace_ids))
        .group_by(Document.workspace_id)
    )
    return {ws: int(n) for ws, n in rows.all()}


async def get_document(
    session: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID
) -> DocumentRead:
    document, _ = await _document_for_member(session, document_id, user_id)
    return DocumentRead.model_validate(document)


async def document_file(
    session: AsyncSession, storage: Storage, document_id: uuid.UUID, user_id: uuid.UUID
) -> tuple[Path, DocumentRead]:
    document, _ = await _document_for_member(session, document_id, user_id)
    return storage.local_path(document.storage_key), DocumentRead.model_validate(document)


async def delete_document(
    session: AsyncSession, storage: Storage, document_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    """Delete the row (cascading to all derived data), then the stored file.

    The row lock serialises against a worker that is storing results: the worker
    re-reads the row under the same lock and gives up if it is gone, so a queued or
    running ingestion job can never recreate a deleted document.
    """
    document, role = await _document_for_member(session, document_id, user_id, lock=True)
    _require_write(role, "delete documents")
    key = document.storage_key
    await session.delete(document)
    await session.commit()
    storage.delete(key)


async def retry_document(
    session: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID
) -> DocumentRead:
    """Re-queue a failed document under a new ingestion generation."""
    document, role = await _document_for_member(session, document_id, user_id, lock=True)
    _require_write(role, "retry document processing")
    if document.status is not DocumentStatus.FAILED:
        raise InvalidOperationError("Only failed documents can be retried.")
    await session.execute(
        update(Document)
        .where(Document.id == document.id)
        .values(
            status=DocumentStatus.UPLOADED,
            stage="queued",
            progress=0,
            error=None,
            attempts=0,
            ingest_generation=Document.ingest_generation + 1,
            updated_at=func.now(),
        )
    )
    await session.commit()
    await session.refresh(document)
    return DocumentRead.model_validate(document)


async def list_chunks(
    session: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID, *, offset: int, limit: int
) -> ChunkPage:
    await _document_for_member(session, document_id, user_id)
    total = await session.scalar(select(func.count()).where(Chunk.document_id == document_id))
    rows = await session.execute(
        select(
            Chunk.id,
            Chunk.chunk_index,
            Chunk.text,
            Chunk.char_start,
            Chunk.char_end,
            Chunk.page_start,
            Chunk.page_end,
            Chunk.bbox_json,
            Chunk.section_path,
            Chunk.token_count,
            Chunk.embedding_windows,
            Chunk.table_row_id,
        )
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
        .offset(offset)
        .limit(limit)
    )
    items = [
        ChunkRead(
            id=r.id,
            chunk_index=r.chunk_index,
            text=r.text,
            char_start=r.char_start,
            char_end=r.char_end,
            page_start=r.page_start,
            page_end=r.page_end,
            bboxes=[BBoxRead.model_validate(b) for b in r.bbox_json],
            section_path=r.section_path,
            token_count=r.token_count,
            embedding_windows=r.embedding_windows,
            table_row_id=r.table_row_id,
        )
        for r in rows.all()
    ]
    return ChunkPage(total=int(total or 0), items=items)


async def _content(session: AsyncSession, document_id: uuid.UUID) -> str:
    text = await session.scalar(
        select(DocumentContent.text).where(DocumentContent.document_id == document_id)
    )
    return text or ""


async def list_pages(
    session: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID, *, include_text: bool
) -> list[PageRead]:
    await _document_for_member(session, document_id, user_id)
    pages = (
        await session.scalars(
            select(Page).where(Page.document_id == document_id).order_by(Page.page_number)
        )
    ).all()
    text = await _content(session, document_id) if include_text else None
    return [
        PageRead(
            page_number=p.page_number,
            width=p.width,
            height=p.height,
            is_ocr=p.is_ocr,
            ocr_confidence=p.ocr_confidence,
            char_start=p.char_start,
            char_end=p.char_end,
            text=text[p.char_start : p.char_end] if text is not None else None,
        )
        for p in pages
    ]


async def list_blocks(
    session: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID
) -> list[BlockRead]:
    await _document_for_member(session, document_id, user_id)
    blocks = (
        await session.scalars(
            select(DocumentBlock)
            .where(DocumentBlock.document_id == document_id)
            .order_by(DocumentBlock.block_index)
        )
    ).all()
    text = await _content(session, document_id)
    return [
        BlockRead(
            block_index=b.block_index,
            kind=b.kind,
            style=b.style,
            level=b.level,
            starts_section=b.starts_section,
            page_number=b.page_number,
            char_start=b.char_start,
            char_end=b.char_end,
            section_path=b.section_path,
            text=text[b.char_start : b.char_end],
        )
        for b in blocks
    ]


async def list_sheets(
    session: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID
) -> list[SheetRead]:
    await _document_for_member(session, document_id, user_id)
    sheets = await session.scalars(
        select(TableSheet)
        .where(TableSheet.document_id == document_id)
        .order_by(TableSheet.sheet_index)
    )
    return [SheetRead.model_validate(s) for s in sheets]


async def list_rows(
    session: AsyncSession,
    document_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    sheet_index: int | None,
    offset: int,
    limit: int,
) -> RowPage:
    await _document_for_member(session, document_id, user_id)
    where = [TableRow.document_id == document_id]
    if sheet_index is not None:
        where.append(TableRow.sheet_index == sheet_index)
    total = await session.scalar(select(func.count()).select_from(TableRow).where(*where))
    rows = await session.scalars(
        select(TableRow)
        .where(*where)
        .order_by(TableRow.sheet_index, TableRow.row_index)
        .offset(offset)
        .limit(limit)
    )
    items = [
        RowRead(
            id=r.id,
            sheet_index=r.sheet_index,
            sheet_name=r.sheet_name,
            row_number=r.row_index,
            cells=[CellRead.model_validate(c) for c in r.data_json["cells"]],
            char_start=r.char_start,
            char_end=r.char_end,
        )
        for r in rows
    ]
    return RowPage(total=int(total or 0), items=items)


async def list_entities(
    session: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID
) -> EntitiesResponse:
    await _document_for_member(session, document_id, user_id)
    # The most recent review that extracted entities from this document (runs are
    # workspace-scoped, and the document already passed the membership check).
    latest_run = await session.scalar(
        select(ExtractedEntity.review_run_id)
        .where(ExtractedEntity.document_id == document_id)
        .order_by(ExtractedEntity.created_at.desc())
        .limit(1)
    )
    query = select(ExtractedEntity).where(ExtractedEntity.document_id == document_id)
    query = (
        query.where(ExtractedEntity.review_run_id == latest_run)
        if latest_run is not None
        else query.where(ExtractedEntity.review_run_id.is_(None))
    )
    entities = (
        await session.scalars(
            query.order_by(ExtractedEntity.char_start.nulls_last(), ExtractedEntity.type)
        )
    ).all()
    if not entities:
        return EntitiesResponse(status="not_available", detail=ENTITIES_NOT_AVAILABLE, entities=[])
    return EntitiesResponse(
        status="available",
        detail=(
            "Verified entities from the latest review run: each quote was found exactly in "
            "the cited chunk and its value re-read from the quote."
        ),
        review_run_id=latest_run,
        entities=[EntityRead.model_validate(e) for e in entities],
    )


async def current_generation(session: AsyncSession, document_id: uuid.UUID) -> int:
    generation = await session.scalar(
        select(Document.ingest_generation).where(Document.id == document_id)
    )
    if generation is None:
        raise NotFoundError(DOCUMENT_NOT_FOUND)
    return int(generation)
