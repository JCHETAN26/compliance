# mypy: disallow-untyped-calls=False
# (fixtures are built with PyMuPDF/python-docx, which ship partial annotations)
"""Document API: authz, uploads (limits, spoofing, dedupe), file access, deletion."""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Any

import httpx
import pymupdf
import pytest
from arq.jobs import Job, JobStatus

from compliance_core.jobs import QUEUE_NAME, ingest_job_id
from conftest import TEST_MAX_FILE_BYTES

pytestmark = pytest.mark.integration


def pdf_bytes(text: str = "Mutual non-disclosure agreement between the parties.") -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return bytes(data)


async def _workspace(client: httpx.AsyncClient, name: str = "Docs") -> str:
    response = await client.post("/api/v1/workspaces", json={"name": name})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def _upload(
    client: httpx.AsyncClient, workspace_id: str, files: list[tuple[str, bytes]]
) -> httpx.Response:
    return await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents",
        files=[
            ("files", (name, io.BytesIO(data), "application/octet-stream")) for name, data in files
        ],
    )


async def _add_member(owner: httpx.AsyncClient, ws: str, email: str, role: str) -> None:
    response = await owner.post(
        f"/api/v1/workspaces/{ws}/members", json={"email": email, "role": role}
    )
    assert response.status_code == 201, response.text


async def test_upload_lists_and_queues_ingestion(clients: Any, api_app: Any) -> None:
    owner = await clients.signed_up("owner@example.com")
    ws = await _workspace(owner)
    response = await _upload(
        owner, ws, [("nda.pdf", pdf_bytes()), ("register.csv", b"Vendor,Access\nAcme,High\n")]
    )
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert [r["outcome"] for r in results] == ["created", "created"]
    pdf = results[0]["document"]
    assert pdf["status"] == "uploaded"
    assert pdf["file_format"] == "pdf"
    assert pdf["mime_type"] == "application/pdf"
    assert pdf["stage"] == "queued"
    assert results[1]["document"]["doc_type"] == "spreadsheet"
    listed = (await owner.get(f"/api/v1/workspaces/{ws}/documents")).json()
    assert {d["filename"] for d in listed} == {"nda.pdf", "register.csv"}
    workspace = (await owner.get(f"/api/v1/workspaces/{ws}")).json()
    assert workspace["document_count"] == 2
    job = Job(
        ingest_job_id(uuid.UUID(pdf["id"]), 1), api_app.state.ctx.redis, _queue_name=QUEUE_NAME
    )
    assert await job.status() is JobStatus.queued


async def test_original_file_streams_back_exactly(clients: Any) -> None:
    owner = await clients.signed_up("owner@example.com")
    ws = await _workspace(owner)
    data = pdf_bytes()
    doc = (await _upload(owner, ws, [("Ünïcode nda.pdf", data)])).json()["results"][0]["document"]
    response = await owner.get(f"/api/v1/documents/{doc['id']}/file")
    assert response.status_code == 200
    assert response.content == data
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline;")
    assert "filename*=UTF-8''%C3%9Cn%C3%AFcode%20nda.pdf" in response.headers["content-disposition"]
    download = await owner.get(f"/api/v1/documents/{doc['id']}/file?download=true")
    assert download.headers["content-disposition"].startswith("attachment;")
    assert download.headers["content-security-policy"] == "default-src 'none'; sandbox"


async def test_duplicates_are_scoped_to_the_workspace(clients: Any) -> None:
    owner = await clients.signed_up("owner@example.com")
    ws1, ws2 = await _workspace(owner, "One"), await _workspace(owner, "Two")
    data = pdf_bytes("Same bytes twice.")
    first = (await _upload(owner, ws1, [("a.pdf", data)])).json()["results"][0]
    again = (await _upload(owner, ws1, [("renamed.pdf", data)])).json()["results"][0]
    other = (await _upload(owner, ws2, [("a.pdf", data)])).json()["results"][0]
    assert first["outcome"] == "created"
    assert again["outcome"] == "duplicate"
    assert again["document"]["id"] == first["document"]["id"]
    assert other["outcome"] == "created"
    assert other["document"]["id"] != first["document"]["id"]
    assert len((await owner.get(f"/api/v1/workspaces/{ws1}/documents")).json()) == 1


async def test_spoofed_and_unsupported_files_rejected_per_file(clients: Any, api_app: Any) -> None:
    owner = await clients.signed_up("owner@example.com")
    ws = await _workspace(owner)
    response = await _upload(
        owner,
        ws,
        [
            ("fake.pdf", b"PK\x03\x04 this is a zip"),
            ("evil.exe", b"MZ..."),
            ("../../etc/passwd.csv", b"name,value\na,1\n"),
            ("real.pdf", pdf_bytes()),
        ],
    )
    results = response.json()["results"]
    assert [r["outcome"] for r in results] == ["rejected", "rejected", "created", "created"]
    assert results[0]["error"]["code"] == "unsupported_file"
    assert results[2]["filename"] == "passwd.csv"
    storage_root: Path = api_app.state.ctx.storage.root
    assert list((storage_root / ".staging").iterdir()) == [], "no staging leftovers"


async def test_size_limits(clients: Any) -> None:
    owner = await clients.signed_up("owner@example.com")
    ws = await _workspace(owner)
    big = b"a,b\n" + b"x" * (TEST_MAX_FILE_BYTES + 10)
    response = await _upload(owner, ws, [("big.csv", big), ("ok.csv", b"a,b\n1,2\n")])
    results = response.json()["results"]
    assert results[0]["outcome"] == "rejected"
    assert results[0]["error"]["code"] == "file_too_large"
    assert results[1]["outcome"] == "created"
    too_big = await owner.post(
        f"/api/v1/workspaces/{ws}/documents",
        content=b"x" * 10,
        headers={"content-type": "multipart/form-data; boundary=x", "content-length": str(10**10)},
    )
    assert too_big.status_code == 413
    assert too_big.json()["error"]["code"] == "payload_too_large"


async def test_malformed_requests(clients: Any) -> None:
    owner = await clients.signed_up("owner@example.com")
    ws = await _workspace(owner)
    boundary = "b0undary"
    only_field = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="note"\r\n\r\nhello\r\n--{boundary}--\r\n'
    ).encode()
    no_files = await owner.post(
        f"/api/v1/workspaces/{ws}/documents",
        content=only_field,
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )
    assert no_files.status_code == 400
    assert "No files" in no_files.json()["error"]["message"]
    # httpx omits an empty filename, so the part is a plain field, not a file.
    nameless = await _upload(owner, ws, [("", b"data")])
    assert nameless.status_code == 400
    json_body = await owner.post(f"/api/v1/workspaces/{ws}/documents", json={"files": []})
    assert json_body.status_code == 400
    assert json_body.json()["error"]["code"] == "malformed_upload"


async def test_roles_and_membership_scoping(clients: Any) -> None:
    owner = await clients.signed_up("owner@example.com")
    reviewer = await clients.signed_up("reviewer@example.com")
    viewer = await clients.signed_up("viewer@example.com")
    outsider = await clients.signed_up("outsider@example.com")
    ws = await _workspace(owner)
    await _add_member(owner, ws, "reviewer@example.com", "reviewer")
    await _add_member(owner, ws, "viewer@example.com", "viewer")

    created = await _upload(reviewer, ws, [("r.pdf", pdf_bytes("reviewer upload"))])
    assert created.status_code == 200
    assert created.json()["results"][0]["outcome"] == "created"
    doc_id = created.json()["results"][0]["document"]["id"]

    denied = await _upload(viewer, ws, [("v.pdf", pdf_bytes("viewer upload"))])
    assert denied.status_code == 403
    assert (await viewer.delete(f"/api/v1/documents/{doc_id}")).status_code == 403
    for path in ("", "/file", "/chunks", "/pages", "/blocks", "/sheets", "/rows", "/entities"):
        assert (await viewer.get(f"/api/v1/documents/{doc_id}{path}")).status_code == 200, path

    assert (await _upload(outsider, ws, [("o.pdf", pdf_bytes("x"))])).status_code == 404
    for path in ("", "/file", "/chunks", "/pages", "/blocks", "/sheets", "/rows", "/entities"):
        response = await outsider.get(f"/api/v1/documents/{doc_id}{path}")
        assert response.status_code == 404, path
        assert response.json()["error"]["message"] == "Document not found."
    assert (await outsider.delete(f"/api/v1/documents/{doc_id}")).status_code == 404
    assert (await outsider.get(f"/api/v1/workspaces/{ws}/documents")).status_code == 404
    assert (await owner.get(f"/api/v1/documents/{uuid.uuid4()}")).status_code == 404


async def test_upload_requires_auth_and_csrf(clients: Any) -> None:
    owner = await clients.signed_up("owner@example.com")
    ws = await _workspace(owner)
    anonymous = clients.new()
    assert (await _upload(anonymous, ws, [("a.pdf", pdf_bytes())])).status_code == 401
    del owner.headers["X-CSRF-Token"]
    assert (await _upload(owner, ws, [("a.pdf", pdf_bytes())])).status_code == 403


async def test_unprocessed_document_has_no_derived_data_and_honest_entities(clients: Any) -> None:
    owner = await clients.signed_up("owner@example.com")
    ws = await _workspace(owner)
    doc_id = (await _upload(owner, ws, [("a.pdf", pdf_bytes())])).json()["results"][0]["document"][
        "id"
    ]
    assert (await owner.get(f"/api/v1/documents/{doc_id}/chunks")).json() == {
        "total": 0,
        "items": [],
    }
    assert (await owner.get(f"/api/v1/documents/{doc_id}/pages")).json() == []
    entities = (await owner.get(f"/api/v1/documents/{doc_id}/entities")).json()
    assert entities["status"] == "not_available"
    assert entities["entities"] == []
    assert "Phase 4" in entities["detail"]


async def test_delete_removes_row_and_file_and_retry_rules(clients: Any, api_app: Any) -> None:
    owner = await clients.signed_up("owner@example.com")
    ws = await _workspace(owner)
    doc = (await _upload(owner, ws, [("a.pdf", pdf_bytes())])).json()["results"][0]["document"]
    storage = api_app.state.ctx.storage
    async with api_app.state.ctx.engine.connect() as conn:
        key = (
            await conn.exec_driver_sql(
                "SELECT storage_key FROM documents WHERE id = $1", (uuid.UUID(doc["id"]),)
            )
        ).scalar_one()
    assert storage.exists(key)
    retry = await owner.post(f"/api/v1/documents/{doc['id']}/retry")
    assert retry.status_code == 409, "only failed documents can be retried"
    async with api_app.state.ctx.engine.begin() as conn:
        await conn.exec_driver_sql(
            "UPDATE documents SET status = 'failed', error = 'boom' WHERE id = $1",
            (uuid.UUID(doc["id"]),),
        )
    retried = await owner.post(f"/api/v1/documents/{doc['id']}/retry")
    assert retried.status_code == 200
    assert retried.json()["status"] == "uploaded"
    assert retried.json()["error"] is None
    job = Job(
        ingest_job_id(uuid.UUID(doc["id"]), 2), api_app.state.ctx.redis, _queue_name=QUEUE_NAME
    )
    assert await job.status() is JobStatus.queued, "retry queues a new generation"
    assert (await owner.delete(f"/api/v1/documents/{doc['id']}")).status_code == 204
    assert not storage.exists(key)
    assert (await owner.get(f"/api/v1/documents/{doc['id']}")).status_code == 404


async def test_workspace_deletion_removes_stored_files(clients: Any, api_app: Any) -> None:
    owner = await clients.signed_up("owner@example.com")
    ws = await _workspace(owner)
    await _upload(owner, ws, [("a.pdf", pdf_bytes("one")), ("b.pdf", pdf_bytes("two"))])
    root: Path = api_app.state.ctx.storage.root
    workspace_dir = root / "documents" / uuid.UUID(ws).hex
    assert len(list(workspace_dir.iterdir())) == 2
    assert (await owner.delete(f"/api/v1/workspaces/{ws}")).status_code == 204
    assert not workspace_dir.exists()


async def test_upload_rate_limit_is_atomic_and_counts_rejected_files(
    clients: Any, api_app: Any
) -> None:
    from compliance_core.services.session_store import UploadRateLimiter

    ctx = api_app.state.ctx
    original = ctx.upload_limiter
    ctx.upload_limiter = UploadRateLimiter(ctx.redis, files_per_minute=3, requests_per_minute=100)
    try:
        owner = await clients.signed_up("owner@example.com")
        ws = await _workspace(owner)
        # Rejected files consume the file quota too.
        first = await _upload(owner, ws, [("a.exe", b"MZ"), ("b.exe", b"MZ")])
        assert [r["outcome"] for r in first.json()["results"]] == ["rejected", "rejected"]
        over = await _upload(owner, ws, [("c.exe", b"MZ"), ("d.pdf", pdf_bytes())])
        assert over.status_code == 429
        assert over.json()["error"]["code"] == "rate_limited"
        assert 0 < int(over.headers["retry-after"]) <= 60
        assert list((ctx.storage.root / ".staging").iterdir()) == []
        assert (await owner.get(f"/api/v1/workspaces/{ws}/documents")).json() == []
    finally:
        ctx.upload_limiter = original


async def test_concurrent_upload_requests_respect_the_request_cap(
    clients: Any, api_app: Any
) -> None:
    import asyncio

    from compliance_core.services.session_store import UploadRateLimiter

    ctx = api_app.state.ctx
    original = ctx.upload_limiter
    ctx.upload_limiter = UploadRateLimiter(ctx.redis, files_per_minute=1000, requests_per_minute=4)
    try:
        owner = await clients.signed_up("owner@example.com")
        ws = await _workspace(owner)
        responses = await asyncio.gather(
            *(_upload(owner, ws, [(f"f{i}.csv", f"a,b\n{i},2\n".encode())]) for i in range(10))
        )
        codes = sorted(r.status_code for r in responses)
        assert codes.count(200) == 4
        assert codes.count(429) == 6
    finally:
        ctx.upload_limiter = original
