"""Document ingestion job: parse (isolated child process) -> embed -> store.

Correctness rules:

- **Generation-keyed idempotence.** A job carries ``(document_id, generation)``. It
  only runs if the document still exists, has that generation and is ``uploaded`` or
  ``processing``. Uploads of identical bytes are de-duplicated per workspace (sha256)
  before a job is ever queued.
- **No resurrection.** Results are written in one transaction that first re-locks the
  document row (``SELECT ... FOR UPDATE``). If the row is gone (deleted), its generation
  changed, or it is no longer ``processing``, the results are discarded. Deletion takes
  the same row lock, so the two cannot interleave.
- **Safe retries.** Before inserting, all previously derived rows for the document are
  deleted inside the same transaction, so a retry never duplicates chunks or rows.
  Transient failures (database/Redis/OS errors, a crashed parser process) are retried
  up to ``INGEST_MAX_ATTEMPTS``; document problems (``ExtractionError``) fail at once
  with a user-visible message.
- **No document content in logs.** Only ids, counts and timings are logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from arq import Retry
from arq.connections import ArqRedis
from arq.jobs import Job, JobStatus
from redis.exceptions import RedisError
from sqlalchemy import delete, func, insert, or_, select, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from compliance_core.config import Settings
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
)
from compliance_core.jobs import INGEST_DOCUMENT, QUEUE_NAME, ingest_job_id
from compliance_core.storage import LocalStorage, StorageError
from compliance_ingestion.embeddings import EmbeddedText, Embedder
from compliance_ingestion.runner import ExtractionCrashError, RunnerLimits, run_extraction
from compliance_ingestion.types import ExtractionError, ExtractionResult

logger = logging.getLogger("compliance_worker.ingest")
EMBED_BATCH = 64
UPLOAD_REQUEUE_AFTER = timedelta(seconds=30)


@dataclass
class IngestDeps:
    settings: Settings
    session_factory: async_sessionmaker[AsyncSession]
    storage: LocalStorage
    embedder: Embedder
    tokenizer_path: Path
    model_label: str
    embed_lock: asyncio.Lock


@dataclass(frozen=True)
class _Claim:
    storage_key: str
    file_format: str
    attempts: int


def pipeline_options(settings: Settings) -> dict[str, int | str]:
    return {
        "max_pdf_pages": settings.max_pdf_pages,
        "ocr_language": settings.ocr_language,
        "ocr_dpi": settings.ocr_dpi,
        "ocr_min_text_chars": settings.ocr_min_text_chars,
        "ocr_timeout_seconds": settings.ocr_timeout_seconds,
        "chunk_min_tokens": settings.chunk_min_tokens,
        "chunk_max_tokens": settings.chunk_max_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "max_sheet_rows": settings.max_sheet_rows,
    }


async def _claim(deps: IngestDeps, document_id: uuid.UUID, generation: int) -> _Claim | None:
    async with deps.session_factory() as session:
        document = await session.scalar(
            select(Document).where(Document.id == document_id).with_for_update()
        )
        if document is None or document.ingest_generation != generation:
            return None
        if document.status not in (DocumentStatus.UPLOADED, DocumentStatus.PROCESSING):
            return None
        document.status = DocumentStatus.PROCESSING
        document.stage = "parsing"
        document.progress = 5
        document.attempts += 1
        document.updated_at = datetime.now(UTC)
        claim = _Claim(document.storage_key, document.file_format, document.attempts)
        await session.commit()
        return claim


async def _set_stage(
    deps: IngestDeps, document_id: uuid.UUID, generation: int, stage: str, progress: int
) -> None:
    async with deps.session_factory() as session:
        await session.execute(
            update(Document)
            .where(
                Document.id == document_id,
                Document.ingest_generation == generation,
                Document.status == DocumentStatus.PROCESSING,
            )
            .values(stage=stage, progress=progress, updated_at=func.now())
        )
        await session.commit()


async def _fail(deps: IngestDeps, document_id: uuid.UUID, generation: int, message: str) -> bool:
    """Mark the claimed run failed. Only a PROCESSING document of the same generation
    transitions, so a late failure can never overwrite a READY result (or a newer run)."""
    async with deps.session_factory() as session:
        changed = await session.execute(
            update(Document)
            .where(
                Document.id == document_id,
                Document.ingest_generation == generation,
                Document.status == DocumentStatus.PROCESSING,
            )
            .values(
                status=DocumentStatus.FAILED,
                stage=None,
                progress=0,
                error=message[:1000],
                updated_at=func.now(),
            )
        )
        await session.commit()
        return bool(changed.rowcount)  # type: ignore[attr-defined]


async def _embed(
    deps: IngestDeps, document_id: uuid.UUID, generation: int, texts: list[str]
) -> list[EmbeddedText]:
    vectors: list[EmbeddedText] = []
    for start in range(0, len(texts), EMBED_BATCH):
        async with deps.embed_lock:
            batch = await asyncio.to_thread(deps.embedder.embed, texts[start : start + EMBED_BATCH])
        vectors.extend(batch)
        progress = 60 + int(30 * len(vectors) / max(len(texts), 1))
        await _set_stage(deps, document_id, generation, "embedding", min(progress, 90))
    return vectors


async def _store(
    deps: IngestDeps,
    document_id: uuid.UUID,
    generation: int,
    result: ExtractionResult,
    vectors: list[EmbeddedText],
) -> bool:
    async with deps.session_factory() as session:
        document = await session.scalar(
            select(Document).where(Document.id == document_id).with_for_update()
        )
        if (
            document is None
            or document.ingest_generation != generation
            or document.status is not DocumentStatus.PROCESSING
        ):
            await session.rollback()
            return False
        for model in (
            ExtractedEntity,
            Chunk,
            TableRow,
            TableSheet,
            DocumentBlock,
            Page,
            DocumentContent,
        ):
            await session.execute(delete(model).where(model.document_id == document_id))

        session.add(
            DocumentContent(
                document_id=document_id,
                text=result.text,
                text_sha256=hashlib.sha256(result.text.encode()).hexdigest(),
                line_boxes=result.line_boxes or None,
            )
        )
        page_ids: dict[int, uuid.UUID] = {}
        if result.pages:
            rows = await session.execute(
                insert(Page).returning(Page.id, Page.page_number),
                [{"document_id": document_id, **p.model_dump()} for p in result.pages],
            )
            page_ids = {number: pid for pid, number in rows.all()}
        if result.blocks:
            await session.execute(
                insert(DocumentBlock),
                [
                    {
                        "document_id": document_id,
                        "block_index": b.block_index,
                        "kind": b.kind,
                        "style": (b.style or "")[:120] or None,
                        "level": b.level,
                        "starts_section": b.starts_section,
                        "page_number": b.page_number,
                        "char_start": b.char_start,
                        "char_end": b.char_end,
                        "section_path": b.section_path,
                        "bbox_json": b.bbox.model_dump() if b.bbox else None,
                    }
                    for b in result.blocks
                ],
            )
        sheet_names = {s.sheet_index: s.sheet_name for s in result.sheets}
        if result.sheets:
            await session.execute(
                insert(TableSheet),
                [{"document_id": document_id, **s.model_dump()} for s in result.sheets],
            )
        row_ids: dict[tuple[int, int], uuid.UUID] = {}
        if result.rows:
            row_result = await session.execute(
                insert(TableRow).returning(TableRow.id, TableRow.sheet_index, TableRow.row_index),
                [
                    {
                        "document_id": document_id,
                        "sheet_name": sheet_names[r.sheet_index],
                        "sheet_index": r.sheet_index,
                        "row_index": r.row_number,
                        "data_json": {"cells": [c.model_dump(mode="json") for c in r.cells]},
                        "char_start": r.char_start,
                        "char_end": r.char_end,
                    }
                    for r in result.rows
                ],
            )
            row_ids = {(sheet, number): rid for rid, sheet, number in row_result.all()}
        if result.chunks:
            await session.execute(
                insert(Chunk),
                [
                    {
                        "document_id": document_id,
                        "page_id": page_ids.get(c.page_start) if c.page_start else None,
                        "table_row_id": (
                            row_ids.get((c.sheet_index, c.row_number))
                            if c.sheet_index is not None and c.row_number is not None
                            else None
                        ),
                        "chunk_index": c.chunk_index,
                        "text": c.text,
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "bbox_json": [b.model_dump() for b in c.bboxes],
                        "section_path": c.section_path,
                        "token_count": c.token_count,
                        "embedding": v.vector,
                        "embedding_windows": v.windows,
                    }
                    for c, v in zip(result.chunks, vectors, strict=True)
                ],
            )
        document.status = DocumentStatus.READY
        document.stage = None
        document.progress = 100
        document.error = None
        document.doc_type = DocType(result.doc_type)
        document.page_count = len(result.pages) or None
        document.ocr_page_count = result.stats.ocr_pages
        document.chunk_count = len(result.chunks)
        document.warnings = result.warnings[:50]
        document.embedding_model = deps.model_label
        document.ingested_at = datetime.now(UTC)
        document.updated_at = datetime.now(UTC)
        await session.commit()
        return True


async def ingest_document(ctx: dict[str, Any], document_id: str, generation: int) -> dict[str, Any]:
    deps: IngestDeps = ctx["ingest"]
    doc_id = uuid.UUID(document_id)
    log = {"document_id": document_id, "generation": generation}
    claim = await _claim(deps, doc_id, generation)
    if claim is None:
        logger.info("ingestion skipped (deleted, superseded or already done)", extra=log)
        return {"status": "skipped"}
    started = time.perf_counter()
    settings = deps.settings
    try:
        path = deps.storage.local_path(claim.storage_key)
        result = await run_extraction(
            path,
            claim.file_format,
            deps.tokenizer_path,
            pipeline_options(settings),
            RunnerLimits(settings.ingest_parse_timeout_seconds, settings.ingest_parse_memory_mb),
        )
        await _set_stage(deps, doc_id, generation, "embedding", 60)
        vectors = await _embed(deps, doc_id, generation, [c.text for c in result.chunks])
        await _set_stage(deps, doc_id, generation, "storing", 95)
        stored = await _store(deps, doc_id, generation, result, vectors)
    except ExtractionError as exc:
        await _fail(deps, doc_id, generation, str(exc))
        logger.info("ingestion rejected document", extra=log | {"reason": str(exc)[:200]})
        return {"status": "failed"}
    except StorageError:
        await _fail(deps, doc_id, generation, "The original file is missing from storage.")
        logger.error("stored file missing", extra=log)
        return {"status": "failed"}
    except Exception as exc:
        # Crashed parser child, database/Redis/OS errors, embedding runtime errors or any
        # unexpected bug: retry with backoff while attempts remain, then fail visibly.
        # Never leave the document silently PROCESSING.
        transient = isinstance(exc, ExtractionCrashError | DBAPIError | RedisError | OSError)
        if transient and claim.attempts < settings.ingest_max_attempts:
            logger.warning(
                "ingestion attempt failed; retrying",
                extra=log | {"attempt": claim.attempts, "error": type(exc).__name__},
            )
            raise Retry(defer=min(60, 5 * claim.attempts)) from exc
        reason = (
            f"Processing failed after {claim.attempts} attempts ({type(exc).__name__})."
            if transient
            else f"Internal processing error ({type(exc).__name__}); the document was not ingested."
        )
        await _fail(deps, doc_id, generation, reason)
        logger.exception("ingestion failed", extra=log)
        return {"status": "failed"}
    elapsed = round(time.perf_counter() - started, 2)
    if not stored:
        logger.info("ingestion results discarded (document deleted or superseded)", extra=log)
        return {"status": "discarded"}
    logger.info(
        "document ready",
        extra=log
        | {
            "chunks": result.stats.chunks,
            "pages": result.stats.pages,
            "ocr_pages": result.stats.ocr_pages,
            "rows": result.stats.rows,
            "seconds": elapsed,
            "windowed_chunks": sum(1 for v in vectors if v.windows > 1),
        },
    )
    return {"status": "ready", "chunks": result.stats.chunks, "seconds": elapsed}


async def requeue_stale_documents(ctx: dict[str, Any]) -> dict[str, int]:
    """Periodic sweep: re-queue uploads whose job was lost (e.g. Redis was down at
    upload time) and recover documents stuck in ``processing`` after a worker crash."""
    deps: IngestDeps = ctx["ingest"]
    redis: ArqRedis = ctx["redis"]
    settings = deps.settings
    now = datetime.now(UTC)
    stale_processing = now - timedelta(seconds=settings.ingest_parse_timeout_seconds + 300)
    requeued = failed = 0
    async with deps.session_factory() as session:
        candidates = (
            await session.execute(
                select(
                    Document.id, Document.ingest_generation, Document.status, Document.attempts
                ).where(
                    or_(
                        (Document.status == DocumentStatus.UPLOADED)
                        & (Document.updated_at < now - UPLOAD_REQUEUE_AFTER),
                        (Document.status == DocumentStatus.PROCESSING)
                        & (Document.updated_at < stale_processing),
                    )
                )
            )
        ).all()
    for doc_id, generation, status, attempts in candidates:
        job = Job(ingest_job_id(doc_id, generation), redis, _queue_name=QUEUE_NAME)
        if await job.status() in (JobStatus.queued, JobStatus.deferred, JobStatus.in_progress):
            continue
        # Every transition re-checks status and staleness in the same UPDATE, so a run
        # that finished (or a document that changed) after the candidate query is never
        # reset, failed or re-queued.
        cutoff = (
            now - UPLOAD_REQUEUE_AFTER if status is DocumentStatus.UPLOADED else stale_processing
        )
        still_stale = (
            (Document.id == doc_id)
            & (Document.ingest_generation == generation)
            & (Document.status == status)
            & (Document.updated_at < cutoff)
        )
        async with deps.session_factory() as session:
            if status is DocumentStatus.PROCESSING and attempts >= settings.ingest_max_attempts:
                result = await session.execute(
                    update(Document)
                    .where(still_stale)
                    .values(
                        status=DocumentStatus.FAILED,
                        stage=None,
                        error="Processing was interrupted repeatedly and has been stopped.",
                        updated_at=func.now(),
                    )
                )
                await session.commit()
                failed += int(bool(result.rowcount))  # type: ignore[attr-defined]
                continue
            # A finished/unknown job id cannot be re-enqueued, so move to a new generation.
            new_generation = generation + 1
            result = await session.execute(
                update(Document)
                .where(still_stale)
                .values(
                    status=DocumentStatus.UPLOADED,
                    stage="queued",
                    progress=0,
                    ingest_generation=new_generation,
                    updated_at=func.now(),
                )
            )
            await session.commit()
        if result.rowcount:  # type: ignore[attr-defined]
            await redis.enqueue_job(
                INGEST_DOCUMENT,
                str(doc_id),
                new_generation,
                _job_id=ingest_job_id(doc_id, new_generation),
                _queue_name=QUEUE_NAME,
            )
            requeued += 1
    if requeued or failed:
        logger.info("stale document sweep", extra={"requeued": requeued, "failed": failed})
    return {"requeued": requeued, "failed": failed}
