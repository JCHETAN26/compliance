"""Registering staged uploads, shared by the multipart REST endpoint and MCP stdio uploads.

The transport authorises the caller (owner/reviewer), reserves the per-request upload
quota and stages each file (streamed multipart parts or decoded base64). From there the
rules are the same everywhere: every staged file counts against the per-user file quota
before any validation, the content is sniffed under the zip-bomb limits, the document is
stored and registered (or reported as a duplicate), ingestion is queued, and no staged
file survives the call.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from compliance_core.document_schemas import UploadErrorBody, UploadResponse, UploadResult
from compliance_core.files import UnsupportedFileError, ZipLimits, detect_format
from compliance_core.runtime import ServiceContext
from compliance_core.services import documents as service
from compliance_core.uploads import StagedFile

logger = logging.getLogger(__name__)


def _discard(staged: Sequence[StagedFile]) -> None:
    for item in staged:
        if item.path is not None:
            item.path.unlink(missing_ok=True)  # no-op for files moved into storage


async def register_staged_files(
    ctx: ServiceContext,
    session: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    staged: Sequence[StagedFile],
) -> UploadResponse:
    """One result per staged file: ``created`` (queued for ingestion), ``duplicate``
    (same bytes already in this workspace) or ``rejected`` (with a reason)."""
    settings = ctx.settings
    try:
        # Every file part counts, whatever its outcome, before validation or storage.
        await ctx.upload_limiter.reserve_files(str(user_id), len(staged))
    except BaseException:
        _discard(staged)
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
        _discard(staged)
    return UploadResponse(results=results)
