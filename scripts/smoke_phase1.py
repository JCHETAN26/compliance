# mypy: disallow-untyped-calls=False
# (builds a fixture PDF with PyMuPDF, which ships partial annotations)
"""Phase 1 end-to-end smoke test against a running stack (through the web origin).

Checks: seeded demo workspace (demo login, 11 READY documents incl. an OCR'd scan and
a structured register), then a fresh upload -> worker ingestion -> chunks/pages/file
API -> viewer page, role enforcement, spoofed-file rejection, dedupe and deletion.

    uv run python scripts/smoke_phase1.py --web http://127.0.0.1:3000
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from collections.abc import Callable

import httpx
import pymupdf

PASSWORD = "smoke-phase1-password"  # noqa: S105 - throwaway test account
DEMO = ("demo@example.com", "demo1234")


class SmokeError(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def step(name: str, fn: Callable[[], None]) -> None:
    fn()
    print(f"PASS  {name}")  # noqa: T201


def login(base: str, email: str, password: str) -> tuple[httpx.Client, str]:
    client = httpx.Client(base_url=base, timeout=30)
    r = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    if r.status_code == 401:
        r = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    expect(r.status_code in (200, 201), f"auth {email}: {r.status_code} {r.text}")
    return client, r.json()["csrf_token"]


def nda_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    lines = [
        "MUTUAL NON-DISCLOSURE AGREEMENT",
        f"Smoke reference {secrets.token_hex(4)}.",
        "1. The Recipient shall keep the Confidential Information secret.",
        "2. The Recipient shall return or destroy all copies on request.",
    ]
    for i, line in enumerate(lines):
        page.insert_text((72, 72 + 18 * i), line, fontsize=11)
    data = bytes(doc.tobytes())
    doc.close()
    return data


def wait_ready(client: httpx.Client, document_id: str, timeout: float = 180) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        doc = client.get(f"/api/v1/documents/{document_id}").json()
        if doc["status"] in ("ready", "failed"):
            return dict(doc)
        time.sleep(1.5)
    raise SmokeError(f"document {document_id} not processed within {timeout}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", default="http://127.0.0.1:3000")
    args = parser.parse_args()
    state: dict[str, str] = {}

    def demo_workspace() -> None:
        demo, _ = login(args.web, *DEMO)
        workspaces = demo.get("/api/v1/workspaces").json()
        seeded = [w for w in workspaces if w["name"].startswith("Demo")]
        expect(len(seeded) == 1, f"demo workspace missing: {workspaces}")
        docs = demo.get(f"/api/v1/workspaces/{seeded[0]['id']}/documents").json()
        expect(len(docs) == 11, f"expected 11 seeded documents, got {len(docs)}")
        not_ready = [d["filename"] for d in docs if d["status"] != "ready"]
        expect(not not_ready, f"seeded documents not ready: {not_ready}")
        scanned = next(d for d in docs if d["filename"].endswith("_scanned.pdf"))
        expect(scanned["ocr_page_count"] == scanned["page_count"] > 0, "scan was not OCR'd")
        pages = demo.get(f"/api/v1/documents/{scanned['id']}/pages?include_text=true").json()
        expect(all(p["is_ocr"] for p in pages), "scan pages not marked OCR")
        expect("Non-Disclosure Agreement" in "".join(p["text"] for p in pages), "OCR text missing")
        register = next(d for d in docs if d["filename"] == "vendor_register.xlsx")
        sheets = demo.get(f"/api/v1/documents/{register['id']}/sheets").json()
        expect(sheets[0]["columns"][:2] == ["Vendor name", "Data access level"], str(sheets[0]))
        rows = demo.get(f"/api/v1/documents/{register['id']}/rows?sheet_index=0").json()
        expect(rows["total"] == 11, f"register rows: {rows['total']}")
        entities = demo.get(f"/api/v1/documents/{scanned['id']}/entities").json()
        expect(entities["status"] == "not_available" and entities["entities"] == [], str(entities))

    def upload_and_ingest() -> None:
        owner, csrf = login(args.web, f"smoke1-{secrets.token_hex(4)}@example.com", PASSWORD)
        owner.headers["X-CSRF-Token"] = csrf
        ws = owner.post("/api/v1/workspaces", json={"name": "Phase 1 smoke"}).json()["id"]
        state["ws"] = ws
        data = nda_pdf()
        r = owner.post(
            f"/api/v1/workspaces/{ws}/documents",
            files=[
                ("files", ("smoke_nda.pdf", data, "application/pdf")),
                ("files", ("fake.pdf", b"PK\x03\x04 not a pdf", "application/pdf")),
                ("files", ("vendors.csv", b"Vendor,Access\nAcme Ltd,High\n", "text/csv")),
            ],
        )
        expect(r.status_code == 200, f"upload: {r.status_code} {r.text}")
        outcomes = [x["outcome"] for x in r.json()["results"]]
        expect(outcomes == ["created", "rejected", "created"], str(outcomes))
        pdf_id = r.json()["results"][0]["document"]["id"]
        dup = owner.post(
            f"/api/v1/workspaces/{ws}/documents",
            files=[("files", ("again.pdf", data, "application/pdf"))],
        )
        expect(dup.json()["results"][0]["outcome"] == "duplicate", "dedupe failed")
        doc = wait_ready(owner, pdf_id)
        expect(doc["status"] == "ready", f"ingestion failed: {doc.get('error')}")
        chunks = owner.get(f"/api/v1/documents/{pdf_id}/chunks").json()["items"]
        expect(chunks and all(c["bboxes"] for c in chunks), "chunks without boxes")
        expect(any("return or destroy" in c["text"] for c in chunks), "chunk text missing")
        original = owner.get(f"/api/v1/documents/{pdf_id}/file")
        expect(original.content == data, "original file differs")
        viewer = owner.get(f"/workspaces/{ws}/documents/{pdf_id}")
        expect(viewer.status_code == 200, f"viewer page: {viewer.status_code}")
        state.update(owner_csrf=csrf, pdf=pdf_id)
        state["owner_cookie"] = owner.cookies.get("cw_access") or ""

    def roles_and_delete() -> None:
        owner = httpx.Client(
            base_url=args.web, timeout=30, cookies={"cw_access": state["owner_cookie"]}
        )
        owner.headers["X-CSRF-Token"] = state["owner_csrf"]
        viewer, viewer_csrf = login(
            args.web, f"smoke1v-{secrets.token_hex(4)}@example.com", PASSWORD
        )
        email = viewer.get("/api/v1/auth/me").json()["email"]
        owner.post(
            f"/api/v1/workspaces/{state['ws']}/members", json={"email": email, "role": "viewer"}
        )
        viewer.headers["X-CSRF-Token"] = viewer_csrf
        expect(
            viewer.get(f"/api/v1/documents/{state['pdf']}/chunks").status_code == 200, "viewer read"
        )
        expect(
            viewer.delete(f"/api/v1/documents/{state['pdf']}").status_code == 403, "viewer delete"
        )
        up = viewer.post(
            f"/api/v1/workspaces/{state['ws']}/documents",
            files=[("files", ("x.csv", b"a,b\n", "text/csv"))],
        )
        expect(up.status_code == 403, f"viewer upload: {up.status_code}")
        expect(owner.delete(f"/api/v1/documents/{state['pdf']}").status_code == 204, "owner delete")
        expect(
            owner.get(f"/api/v1/documents/{state['pdf']}/file").status_code == 404,
            "deleted file served",
        )

    try:
        step(
            "seeded demo workspace: 11 ready docs, OCR scan, structured register, honest entities",
            demo_workspace,
        )
        step(
            "upload (multi-file, spoof rejected, dedupe) -> worker ingest -> chunks/boxes/file/viewer",
            upload_and_ingest,
        )
        step("viewer role read-only; owner delete removes document and file", roles_and_delete)
    except (SmokeError, httpx.HTTPError, KeyError, StopIteration) as exc:
        print(f"FAIL  {type(exc).__name__}: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    print("Phase 1 smoke test passed")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
