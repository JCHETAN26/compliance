"""Document upload, listing, retrieval, original-file streaming, derived data and deletion.

Every endpoint is scoped to workspace membership (non-members get 404). Upload, delete
and retry additionally require the ``owner`` or ``reviewer`` role.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Query, Request, Response, status
from fastapi.responses import FileResponse

from compliance_api.deps import Ctx, CurrentPrincipal, Session
from compliance_api.errors import error_responses
from compliance_core.document_schemas import (
    BlockRead,
    ChunkPage,
    DocumentRead,
    EntitiesResponse,
    PageRead,
    RowPage,
    SheetRead,
    UploadErrorBody,
    UploadResponse,
    UploadResult,
)
from compliance_core.files import UnsupportedFileError, ZipLimits, detect_format
from compliance_core.services import documents as service
from compliance_core.uploads import (
    MalformedUploadError,
    UploadLimits,
    UploadTooLargeError,
    stage_multipart,
)

router = APIRouter(tags=["documents"])
logger = logging.getLogger(__name__)

_MULTIPART_SCHEMA = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["files"],
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string", "format": "binary"},
                            "description": "One or more PDF, DOCX, XLSX or CSV files",
                        }
                    },
                }
            }
        },
    }
}


@router.post(
    "/workspaces/{workspace_id}/documents",
    responses=error_responses(400, 401, 403, 404, 413, 422, 429),
    openapi_extra=_MULTIPART_SCHEMA,
)
async def upload_documents(
    workspace_id: uuid.UUID,
    request: Request,
    principal: CurrentPrincipal,
    session: Session,
    ctx: Ctx,
) -> UploadResponse:
    """Upload files (multipart field ``files``). Returns one result per file:
    ``created`` (queued for ingestion), ``duplicate`` (same bytes already in this
    workspace) or ``rejected`` (with a reason)."""
    settings = ctx.settings
    user_id = principal.user.id
    await service.require_writer(session, workspace_id, user_id)
    await session.rollback()  # do not hold a transaction open while streaming
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > settings.max_upload_request_bytes:
        raise UploadTooLargeError(
            f"The upload exceeds the {settings.max_upload_request_bytes // (1024 * 1024)} MB request limit."
        )
    # Reserve the request atomically before reading the body (concurrency-safe).
    await ctx.upload_limiter.reserve_request(str(user_id))
    staged = await stage_multipart(
        request.stream(),
        request.headers.get("content-type"),
        UploadLimits(
            max_file_bytes=settings.max_upload_file_bytes,
            max_request_bytes=settings.max_upload_request_bytes,
            max_files=settings.max_upload_files,
        ),
        ctx.storage.new_staging_file,
    )
    if not staged:
        raise MalformedUploadError("No files were included in the upload (field 'files').")
    try:
        # Every file part counts, whatever its outcome, before validation or storage.
        await ctx.upload_limiter.reserve_files(str(user_id), len(staged))
    except BaseException:
        for item in staged:
            if item.path is not None:
                item.path.unlink(missing_ok=True)
        raise
    zip_limits = ZipLimits(
        max_entries=settings.max_zip_entries,
        max_uncompressed_bytes=settings.max_zip_uncompressed_bytes,
        max_ratio=settings.max_zip_compression_ratio,
    )
    results: list[UploadResult] = []
    try:
        for item in staged:
            if item.error or item.path is None:
                results.append(
                    UploadResult(
                        filename=item.filename,
                        outcome="rejected",
                        error=UploadErrorBody(
                            code="file_too_large", message=item.error or "Rejected."
                        ),
                    )
                )
                continue
            try:
                fmt = await asyncio.to_thread(detect_format, item.path, item.filename, zip_limits)
            except UnsupportedFileError as exc:
                item.path.unlink(missing_ok=True)
                results.append(
                    UploadResult(
                        filename=item.filename,
                        outcome="rejected",
                        error=UploadErrorBody(code="unsupported_file", message=str(exc)),
                    )
                )
                continue
            registered = await service.register_upload(
                session,
                ctx.storage,
                workspace_id=workspace_id,
                user_id=user_id,
                filename=item.filename,
                fmt=fmt,
                staged_path=item.path,
                sha256=item.sha256,
                size=item.size,
            )
            if registered.created:
                await ctx.enqueue_ingestion(registered.document.id, 1)
                logger.info(
                    "document uploaded",
                    extra={
                        "document_id": str(registered.document.id),
                        "format": fmt,
                        "bytes": item.size,
                    },
                )
            results.append(
                UploadResult(
                    filename=item.filename,
                    outcome="created" if registered.created else "duplicate",
                    document=registered.document,
                )
            )
    finally:
        for item in staged:
            if item.path is not None:
                item.path.unlink(missing_ok=True)  # no-op for files moved into storage
    return UploadResponse(results=results)


@router.get("/workspaces/{workspace_id}/documents", responses=error_responses(401, 404, 422))
async def list_documents(
    workspace_id: uuid.UUID, principal: CurrentPrincipal, session: Session
) -> list[DocumentRead]:
    return await service.list_documents(session, workspace_id, principal.user.id)


@router.get("/documents/{document_id}", responses=error_responses(401, 404, 422))
async def get_document(
    document_id: uuid.UUID, principal: CurrentPrincipal, session: Session
) -> DocumentRead:
    return await service.get_document(session, document_id, principal.user.id)


@router.get(
    "/documents/{document_id}/file",
    response_class=FileResponse,
    responses={
        200: {"content": {"application/octet-stream": {}}},
        **error_responses(401, 404, 422),
    },
)
async def download_file(
    document_id: uuid.UUID,
    principal: CurrentPrincipal,
    session: Session,
    ctx: Ctx,
    download: bool = False,
) -> FileResponse:
    """Stream the original upload. PDFs are served inline unless ``download=true``."""
    path, document = await service.document_file(
        session, ctx.storage, document_id, principal.user.id
    )
    disposition = "inline" if document.file_format == "pdf" and not download else "attachment"
    ascii_name = document.filename.encode("ascii", "ignore").decode() or "document"
    ascii_name = ascii_name.replace('"', "").replace("\\", "")
    headers = {
        "Content-Disposition": (
            f"{disposition}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(document.filename)}"
        ),
        "Cache-Control": "private, max-age=0, must-revalidate",
    }
    if disposition == "attachment":
        headers["Content-Security-Policy"] = "default-src 'none'; sandbox"
    return FileResponse(path, media_type=document.mime_type, headers=headers)


@router.get("/documents/{document_id}/chunks", responses=error_responses(401, 404, 422))
async def list_chunks(
    document_id: uuid.UUID,
    principal: CurrentPrincipal,
    session: Session,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=2000)] = 500,
) -> ChunkPage:
    return await service.list_chunks(
        session, document_id, principal.user.id, offset=offset, limit=limit
    )


@router.get("/documents/{document_id}/pages", responses=error_responses(401, 404, 422))
async def list_pages(
    document_id: uuid.UUID,
    principal: CurrentPrincipal,
    session: Session,
    include_text: bool = False,
) -> list[PageRead]:
    return await service.list_pages(
        session, document_id, principal.user.id, include_text=include_text
    )


@router.get("/documents/{document_id}/blocks", responses=error_responses(401, 404, 422))
async def list_blocks(
    document_id: uuid.UUID, principal: CurrentPrincipal, session: Session
) -> list[BlockRead]:
    return await service.list_blocks(session, document_id, principal.user.id)


@router.get("/documents/{document_id}/sheets", responses=error_responses(401, 404, 422))
async def list_sheets(
    document_id: uuid.UUID, principal: CurrentPrincipal, session: Session
) -> list[SheetRead]:
    return await service.list_sheets(session, document_id, principal.user.id)


@router.get("/documents/{document_id}/rows", responses=error_responses(401, 404, 422))
async def list_rows(
    document_id: uuid.UUID,
    principal: CurrentPrincipal,
    session: Session,
    sheet_index: Annotated[int | None, Query(ge=0)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> RowPage:
    return await service.list_rows(
        session, document_id, principal.user.id, sheet_index=sheet_index, offset=offset, limit=limit
    )


@router.get("/documents/{document_id}/entities", responses=error_responses(401, 404, 422))
async def list_entities(
    document_id: uuid.UUID, principal: CurrentPrincipal, session: Session
) -> EntitiesResponse:
    return await service.list_entities(session, document_id, principal.user.id)


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=error_responses(401, 403, 404, 422),
)
async def delete_document(
    document_id: uuid.UUID, principal: CurrentPrincipal, session: Session, ctx: Ctx
) -> Response:
    await service.delete_document(session, ctx.storage, document_id, principal.user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/documents/{document_id}/retry", responses=error_responses(401, 403, 404, 409, 422))
async def retry_document(
    document_id: uuid.UUID, principal: CurrentPrincipal, session: Session, ctx: Ctx
) -> DocumentRead:
    document = await service.retry_document(session, document_id, principal.user.id)
    generation = await service.current_generation(session, document.id)
    await ctx.enqueue_ingestion(document.id, generation)
    return document
