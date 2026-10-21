# mypy: disallow-untyped-calls=False
# (fixtures are built with PyMuPDF/python-docx, which ship partial annotations)
"""Ingestion job against real Postgres/pgvector, the real embedding model, Tesseract and
the sandboxed parser process: outputs, idempotence, races and failure handling."""

from __future__ import annotations

import asyncio
import hashlib
import io
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from arq import Retry
from openpyxl import Workbook
from PIL import Image
from sqlalchemy import func, select, text, update

from compliance_core.config import Settings
from compliance_core.db.models import (
    Chunk,
    Document,
    DocumentContent,
    DocumentStatus,
    Page,
    TableRow,
    User,
    Workspace,
    WorkspaceMember,
    WorkspaceRole,
)
from compliance_core.services import documents as document_service
from compliance_ingestion.runner import ExtractionCrashError
from compliance_worker import ingest
from compliance_worker.ingest import IngestDeps, ingest_document, requeue_stale_documents

pytestmark = pytest.mark.integration


@pytest.fixture
def deps(
    api_app: Any, embedder: Any, model_path: Path, test_settings: Settings, clean_state: None
) -> IngestDeps:
    ctx = api_app.state.ctx
    return IngestDeps(
        settings=test_settings,
        session_factory=ctx.session_factory,
        storage=ctx.storage,
        embedder=embedder,
        tokenizer_path=model_path / "tokenizer.json",
        model_label="BAAI/bge-small-en-v1.5@test",
        embed_lock=asyncio.Lock(),
    )


async def _workspace(deps: IngestDeps) -> tuple[uuid.UUID, uuid.UUID]:
    async with deps.session_factory() as session:
        user = User(email=f"u{uuid.uuid4().hex[:8]}@example.com", password_hash="x")
        session.add(user)
        await session.flush()
        workspace = Workspace(name="W", owner_id=user.id)
        session.add(workspace)
        await session.flush()
        session.add(
            WorkspaceMember(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.OWNER)
        )
        await session.commit()
        return workspace.id, user.id


async def _register(deps: IngestDeps, name: str, data: bytes, fmt: str) -> uuid.UUID:
    workspace_id, user_id = await _workspace(deps)
    staged = deps.storage.new_staging_file()
    staged.write_bytes(data)
    async with deps.session_factory() as session:
        registered = await document_service.register_upload(
            session,
            deps.storage,
            workspace_id=workspace_id,
            user_id=user_id,
            filename=name,
            fmt=fmt,  # type: ignore[arg-type]
            staged_path=staged,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
        )
    return registered.document.id


async def _document(deps: IngestDeps, doc_id: uuid.UUID) -> Document | None:
    async with deps.session_factory() as session:
        return await session.get(Document, doc_id)


async def _ingest_ready(deps: IngestDeps, doc_id: uuid.UUID) -> None:
    result = await ingest_document({"ingest": deps}, str(doc_id), 1)
    document = await _document(deps, doc_id)
    assert result["status"] == "ready", document.error if document else "document missing"


def _pdf(lines: list[str]) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72.0
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 16
    return bytes(doc.tobytes())


CLAUSES = [
    "MUTUAL NON-DISCLOSURE AGREEMENT",
    "1. Confidential Information. The Recipient shall keep all information secret.",
    "2. Return. On request the Recipient shall return or destroy all copies.",
    "3. Third parties. The Recipient shall not disclose information to any third party.",
]


async def test_pdf_ingests_end_to_end(deps: IngestDeps) -> None:
    doc_id = await _register(deps, "nda.pdf", _pdf(CLAUSES), "pdf")
    result = await ingest_document({"ingest": deps}, str(doc_id), 1)
    assert result["status"] == "ready"
    document = await _document(deps, doc_id)
    assert document is not None
    assert document.status is DocumentStatus.READY
    assert (document.page_count, document.ocr_page_count, document.progress) == (1, 0, 100)
    assert document.doc_type is not None
    assert document.doc_type.value == "contract"
    assert document.embedding_model == "BAAI/bge-small-en-v1.5@test"
    async with deps.session_factory() as session:
        content = await session.scalar(
            select(DocumentContent.text).where(DocumentContent.document_id == doc_id)
        )
        chunks = (await session.scalars(select(Chunk).where(Chunk.document_id == doc_id))).all()
        page = await session.scalar(select(Page).where(Page.document_id == doc_id))
        assert content is not None
        assert page is not None
        assert chunks
        assert document.chunk_count == len(chunks)
        for chunk in chunks:
            assert content[chunk.char_start : chunk.char_end] == chunk.text
            assert len(chunk.embedding) == 384
            assert abs(sum(float(v) * float(v) for v in chunk.embedding) - 1.0) < 1e-3
            assert chunk.bbox_json
            assert all(b["page"] == 1 for b in chunk.bbox_json)
            assert chunk.page_id == page.id
            assert chunk.embedding_windows == 1
        # Full-text vector is generated by Postgres; the GIN/HNSW indexes answer queries.
        hits = await session.scalar(
            select(func.count()).where(
                Chunk.document_id == doc_id,
                Chunk.tsv.op("@@")(func.plainto_tsquery("english", "third party")),
            )
        )
        assert hits == 1
        nearest = await session.scalar(
            select(Chunk.id).order_by(Chunk.embedding.cosine_distance(chunks[0].embedding)).limit(1)
        )
        assert nearest == chunks[0].id


async def test_xlsx_rows_and_chunks_are_linked(deps: IngestDeps) -> None:
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.append(["Vendor name", "Data access level"])
    sheet.append(["Acme Ltd", "High"])
    sheet.append(["Beta GmbH", "Low"])
    buffer = io.BytesIO()
    workbook.save(buffer)
    doc_id = await _register(deps, "reg.xlsx", buffer.getvalue(), "xlsx")
    await _ingest_ready(deps, doc_id)
    async with deps.session_factory() as session:
        rows = (
            await session.scalars(
                select(TableRow).where(TableRow.document_id == doc_id).order_by(TableRow.row_index)
            )
        ).all()
        chunks = (
            await session.scalars(
                select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.chunk_index)
            )
        ).all()
    assert [r.row_index for r in rows] == [2, 3]
    assert rows[0].data_json["cells"][0] == {
        "column": "Vendor name",
        "value": "Acme Ltd",
        "display": "Acme Ltd",
    }
    assert [c.table_row_id for c in chunks] == [r.id for r in rows]
    assert chunks[0].text == "Vendor name: Acme Ltd; Data access level: High"


async def test_scanned_pdf_is_ocrd(deps: IngestDeps) -> None:
    source = pymupdf.open()
    page = source.new_page()
    page.insert_text((72, 150), "THE RECIPIENT SHALL RETURN ALL DOCUMENTS", fontsize=16)
    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    png = io.BytesIO()
    Image.frombytes("L", (pix.width, pix.height), pix.samples).save(png, format="PNG")
    scan = pymupdf.open()
    scan.new_page().insert_image(pymupdf.Rect(0, 0, 595, 842), stream=png.getvalue())
    doc_id = await _register(deps, "scan.pdf", bytes(scan.tobytes()), "pdf")
    assert (await ingest_document({"ingest": deps}, str(doc_id), 1))["status"] == "ready"
    document = await _document(deps, doc_id)
    assert document is not None
    assert document.ocr_page_count == 1
    async with deps.session_factory() as session:
        stored = await session.scalar(select(Page).where(Page.document_id == doc_id))
        chunk = await session.scalar(select(Chunk).where(Chunk.document_id == doc_id))
    assert stored is not None
    assert stored.is_ocr
    assert (stored.ocr_confidence or 0) > 60
    assert chunk is not None
    assert "RETURN ALL DOCUMENTS" in chunk.text


async def test_rerun_after_interrupted_attempt_does_not_duplicate(deps: IngestDeps) -> None:
    doc_id = await _register(deps, "nda.pdf", _pdf(CLAUSES), "pdf")
    assert (await ingest_document({"ingest": deps}, str(doc_id), 1))["status"] == "ready"
    async with deps.session_factory() as session:
        before = await session.scalar(select(func.count()).where(Chunk.document_id == doc_id))
        # Simulate an at-least-once redelivery of the same generation after a crash.
        await session.execute(
            update(Document).where(Document.id == doc_id).values(status=DocumentStatus.PROCESSING)
        )
        await session.commit()
    assert (await ingest_document({"ingest": deps}, str(doc_id), 1))["status"] == "ready"
    async with deps.session_factory() as session:
        after = await session.scalar(select(func.count()).where(Chunk.document_id == doc_id))
        pages = await session.scalar(select(func.count()).where(Page.document_id == doc_id))
    assert after == before
    assert pages == 1
    # A delivery for a ready document is a no-op.
    assert (await ingest_document({"ingest": deps}, str(doc_id), 1))["status"] == "skipped"


async def test_stale_generation_and_deleted_document_are_skipped(deps: IngestDeps) -> None:
    doc_id = await _register(deps, "nda.pdf", _pdf(CLAUSES), "pdf")
    assert (await ingest_document({"ingest": deps}, str(doc_id), 99))["status"] == "skipped"
    async with deps.session_factory() as session:
        await session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": doc_id})
        await session.commit()
    assert (await ingest_document({"ingest": deps}, str(doc_id), 1))["status"] == "skipped"


async def test_delete_during_processing_is_not_resurrected(
    deps: IngestDeps, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_id = await _register(deps, "nda.pdf", _pdf(CLAUSES), "pdf")
    original = ingest._embed

    async def embed_then_delete(
        d: IngestDeps, document_id: uuid.UUID, generation: int, texts: list[str]
    ) -> Any:
        vectors = await original(d, document_id, generation, texts)
        async with d.session_factory() as session:  # the user deletes while we embed
            await session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})
            await session.commit()
        return vectors

    monkeypatch.setattr(ingest, "_embed", embed_then_delete)
    assert (await ingest_document({"ingest": deps}, str(doc_id), 1))["status"] == "discarded"
    async with deps.session_factory() as session:
        assert await session.get(Document, doc_id) is None
        assert await session.scalar(select(func.count()).where(Chunk.document_id == doc_id)) == 0
        assert await session.scalar(select(func.count()).where(Page.document_id == doc_id)) == 0


async def test_corrupt_document_fails_visibly(deps: IngestDeps) -> None:
    doc_id = await _register(deps, "bad.pdf", b"%PDF-1.7 truncated garbage", "pdf")
    assert (await ingest_document({"ingest": deps}, str(doc_id), 1))["status"] == "failed"
    document = await _document(deps, doc_id)
    assert document is not None
    assert document.status is DocumentStatus.FAILED
    assert document.error
    assert "PDF" in document.error


async def test_late_failure_never_overwrites_ready(deps: IngestDeps) -> None:
    doc_id = await _register(deps, "nda.pdf", _pdf(CLAUSES), "pdf")
    assert (await ingest_document({"ingest": deps}, str(doc_id), 1))["status"] == "ready"
    assert await ingest._fail(deps, doc_id, 1, "late failure from a duplicate delivery") is False
    document = await _document(deps, doc_id)
    assert document is not None
    assert document.status is DocumentStatus.READY
    assert document.error is None


async def test_transient_errors_retry_then_fail(
    deps: IngestDeps, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_id = await _register(deps, "nda.pdf", _pdf(CLAUSES), "pdf")

    async def crash(*_: Any, **__: Any) -> Any:
        raise ExtractionCrashError("child died")

    monkeypatch.setattr(ingest, "run_extraction", crash)
    with pytest.raises(Retry):
        await ingest_document({"ingest": deps}, str(doc_id), 1)
    document = await _document(deps, doc_id)
    assert document is not None
    assert document.status is DocumentStatus.PROCESSING
    # test settings: ingest_max_attempts=2 -> the second attempt fails permanently.
    assert (await ingest_document({"ingest": deps}, str(doc_id), 1))["status"] == "failed"
    document = await _document(deps, doc_id)
    assert document is not None
    assert document.status is DocumentStatus.FAILED
    assert "after 2 attempts" in (document.error or "")


async def test_unexpected_embedding_error_fails_without_getting_stuck(
    deps: IngestDeps, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_id = await _register(deps, "nda.pdf", _pdf(CLAUSES), "pdf")

    def broken(_: list[str]) -> Any:
        raise ValueError("tensor shape mismatch")

    monkeypatch.setattr(deps.embedder, "embed", broken)
    assert (await ingest_document({"ingest": deps}, str(doc_id), 1))["status"] == "failed"
    document = await _document(deps, doc_id)
    assert document is not None
    assert document.status is DocumentStatus.FAILED
    assert "Internal processing error (ValueError)" in (document.error or "")


class _FakeRedis:
    def __init__(self) -> None:
        self.enqueued: list[tuple[Any, ...]] = []

    async def enqueue_job(self, *args: Any, **kwargs: Any) -> None:
        self.enqueued.append((args, kwargs))


async def _age(deps: IngestDeps, doc_id: uuid.UUID, status: DocumentStatus, minutes: int) -> None:
    async with deps.session_factory() as session:
        await session.execute(
            update(Document)
            .where(Document.id == doc_id)
            .values(status=status, updated_at=datetime.now(UTC) - timedelta(minutes=minutes))
        )
        await session.commit()


async def test_sweep_requeues_lost_upload_with_new_generation(
    deps: IngestDeps, redis_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_id = await _register(deps, "nda.pdf", _pdf(CLAUSES), "pdf")
    await _age(deps, doc_id, DocumentStatus.UPLOADED, 5)
    fake = _FakeRedis()

    async def not_found(self: Any) -> Any:
        from arq.jobs import JobStatus

        return JobStatus.not_found

    monkeypatch.setattr("compliance_worker.ingest.Job.status", not_found)
    result = await requeue_stale_documents({"ingest": deps, "redis": fake})
    assert result == {"requeued": 1, "failed": 0}
    document = await _document(deps, doc_id)
    assert document is not None
    assert document.ingest_generation == 2
    assert fake.enqueued[0][0][1:] == (str(doc_id), 2)


async def test_sweep_does_not_touch_a_run_that_completes_concurrently(
    deps: IngestDeps, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_id = await _register(deps, "nda.pdf", _pdf(CLAUSES), "pdf")
    await _age(deps, doc_id, DocumentStatus.PROCESSING, 60)
    fake = _FakeRedis()

    async def completes_meanwhile(self: Any) -> Any:
        from arq.jobs import JobStatus

        # The job finishes between the sweep's candidate query and its update.
        async with deps.session_factory() as session:
            await session.execute(
                update(Document)
                .where(Document.id == doc_id)
                .values(status=DocumentStatus.READY, updated_at=func.now())
            )
            await session.commit()
        return JobStatus.complete

    monkeypatch.setattr("compliance_worker.ingest.Job.status", completes_meanwhile)
    result = await requeue_stale_documents({"ingest": deps, "redis": fake})
    assert result == {"requeued": 0, "failed": 0}
    assert fake.enqueued == []
    document = await _document(deps, doc_id)
    assert document is not None
    assert document.status is DocumentStatus.READY
    assert document.ingest_generation == 1
