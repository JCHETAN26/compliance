"""Phase 2 end-to-end smoke test against a running stack (through the web origin).

Checks hybrid search on the seeded demo workspace: real query embeddings from the model
service, results that are exact stored spans (verified against the chunk API), a
spreadsheet-row hit with its sheet/row reference, keyword/vector modes, filters,
viewer access, cross-workspace isolation, validation, an honest rule-check state from
the model service, and the search page itself.

    uv run python scripts/smoke_phase2.py --web http://127.0.0.1:3000 \
        --model-service http://127.0.0.1:8010
"""

from __future__ import annotations

import argparse
import secrets
import sys
from collections.abc import Callable
from typing import Any

import httpx

PASSWORD = "smoke-phase2-password"  # noqa: S105 - throwaway test account
DEMO = ("demo@example.com", "demo1234")


class SmokeError(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def step(name: str, fn: Callable[[], None]) -> None:
    fn()
    print(f"PASS  {name}")  # noqa: T201


def login(base: str, email: str, password: str) -> httpx.Client:
    client = httpx.Client(base_url=base, timeout=60)
    r = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    if r.status_code == 401:
        r = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    expect(r.status_code in (200, 201), f"auth {email}: {r.status_code} {r.text}")
    client.headers["X-CSRF-Token"] = r.json()["csrf_token"]
    return client


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", default="http://127.0.0.1:3000")
    parser.add_argument("--model-service", default="http://127.0.0.1:8010")
    args = parser.parse_args()
    state: dict[str, Any] = {}

    def search(client: httpx.Client, ws: str, **body: Any) -> httpx.Response:
        return client.post(f"/api/v1/workspaces/{ws}/search", json=body)

    def model_service() -> None:
        ms = httpx.Client(base_url=args.model_service, timeout=30)
        health = ms.get("/health").json()
        expect(health["embedding"] == "ready", str(health))
        meta = ms.get("/metadata").json()
        expect(meta["embedding"]["model"] == "BAAI/bge-small-en-v1.5", str(meta))
        expect(meta["rule_check"]["available"] == (health["rule_check"] == "ready"), str(meta))
        check = ms.post(
            "/v1/check", json={"hypothesis": "h", "spans": [{"span_id": "s", "text": "t"}]}
        )
        expected = 200 if health["rule_check"] == "ready" else 503
        expect(
            check.status_code == expected,
            f"/v1/check gave {check.status_code}, expected {expected}",
        )
        state["rerank"] = health["rerank"] == "ready"

    def demo_hybrid() -> None:
        demo = login(args.web, *DEMO)
        ws = next(w for w in demo.get("/api/v1/workspaces").json() if w["name"].startswith("Demo"))
        state.update(demo=demo, ws=ws["id"])
        caps = demo.get("/api/v1/search/capabilities").json()
        expect(caps["embedding_model"] == "BAAI/bge-small-en-v1.5", str(caps))
        expect(caps["rerank_available"] is state["rerank"], str(caps))
        r = search(demo, ws["id"], query="obligation to return or destroy confidential information")
        expect(r.status_code == 200, f"search: {r.status_code} {r.text}")
        body = r.json()
        expect(len(body["results"]) == 8, f"expected 8 results, got {len(body['results'])}")
        expect(body["candidates"]["vector"] > 0 and body["candidates"]["keyword"] > 0, str(body))
        expect(body["settings"]["query_instruction"], "query instruction not reported")
        for hit in body["results"]:
            chunks = demo.get(f"/api/v1/documents/{hit['document_id']}/chunks").json()["items"]
            stored = next(c for c in chunks if c["id"] == hit["chunk_id"])
            for key in ("text", "char_start", "char_end", "page_start", "bboxes"):
                expect(stored[key] == hit[key], f"{key} differs from stored chunk")
        expect(
            any("destroy" in h["text"].lower() for h in body["results"][:3]),
            "no top-3 result mentions destroy",
        )
        state["first"] = body["results"][0]

    def spreadsheet_row() -> None:
        demo, ws = state["demo"], state["ws"]
        r = search(demo, ws, query="Anton Paar data access level renewal date")
        hit = r.json()["results"][0]
        expect(hit["file_format"] == "xlsx", f"top hit is not the register: {hit['filename']}")
        expect("Anton Paar" in hit["text"], hit["text"])
        expect(hit["sheet_name"] and hit["row_number"] and hit["table_row_id"], str(hit))
        rows = demo.get(f"/api/v1/documents/{hit['document_id']}/rows?sheet_index=0").json()
        row = next(x for x in rows["items"] if x["id"] == hit["table_row_id"])
        expect(row["row_number"] == hit["row_number"], "row number mismatch")
        expect((row["char_start"], row["char_end"]) == (hit["char_start"], hit["char_end"]), "")
        page = demo.get(f"/workspaces/{ws}/documents/{hit['document_id']}?chunk={hit['chunk_id']}")
        expect(page.status_code == 200, f"viewer deep link: {page.status_code}")

    def modes_and_filters() -> None:
        demo, ws = state["demo"], state["ws"]
        kw = search(demo, ws, query="Delaware", mode="keyword", top_k=50).json()
        expect(kw["candidates"].keys() == {"keyword"}, str(kw["candidates"]))
        expect(all("delaware" in h["text"].lower() for h in kw["results"]), "keyword non-match")
        vec = search(demo, ws, query="governing law", mode="vector", top_k=5).json()
        expect(vec["candidates"].keys() == {"vector"}, str(vec["candidates"]))
        sheets = search(demo, ws, query="vendor", doc_types=["spreadsheet"], top_k=50).json()
        expect(sheets["results"], "no spreadsheet results")
        expect({h["file_format"] for h in sheets["results"]} <= {"xlsx", "csv"}, "type filter")
        doc = state["first"]["document_id"]
        one = search(demo, ws, query="confidential", document_ids=[doc], top_k=50).json()
        expect({h["document_id"] for h in one["results"]} == {doc}, "document filter leaked")
        bad = search(demo, ws, query="", mode="hybrid")
        expect(bad.status_code == 422, f"empty query: {bad.status_code}")
        hostile = search(demo, ws, query="'); DROP TABLE chunks; -- & | !:*", mode="keyword")
        expect(hostile.status_code == 200, f"hostile query: {hostile.status_code}")
        page = demo.get(f"/workspaces/{ws}/search?q=confidential")
        expect(page.status_code == 200, f"search page: {page.status_code}")

    def isolation_and_roles() -> None:
        demo, ws = state["demo"], state["ws"]
        stranger = login(args.web, f"smoke2-{secrets.token_hex(4)}@example.com", PASSWORD)
        r = search(stranger, ws, query="confidential")
        expect(r.status_code == 404, f"non-member search: {r.status_code}")
        own = stranger.post("/api/v1/workspaces", json={"name": "Phase 2 smoke"}).json()["id"]
        empty = search(stranger, own, query="confidential information").json()
        expect(empty["results"] == [], "a new workspace sees other workspaces' chunks")
        foreign = search(stranger, own, query="x", document_ids=[state["first"]["document_id"]])
        expect(foreign.status_code == 404, f"foreign document filter: {foreign.status_code}")
        email = stranger.get("/api/v1/auth/me").json()["email"]
        demo_ws = demo.get(f"/api/v1/workspaces/{ws}").json()
        if demo_ws["role"] in ("owner", "admin"):
            added = demo.post(
                f"/api/v1/workspaces/{ws}/members", json={"email": email, "role": "viewer"}
            )
            expect(added.status_code == 201, f"add viewer: {added.status_code} {added.text}")
            ok = search(stranger, ws, query="confidential")
            expect(ok.status_code == 200 and ok.json()["results"], "viewer cannot search")
            members = demo.get(f"/api/v1/workspaces/{ws}/members").json()
            member = next(m for m in members if m["email"] == email)
            demo.delete(f"/api/v1/workspaces/{ws}/members/{member['user_id']}")

    try:
        step("model service: pinned encoder ready, rule check state honest", model_service)
        step("demo hybrid search: real embeddings + FTS, results == stored chunks", demo_hybrid)
        step("spreadsheet row hit with sheet/row reference and viewer deep link", spreadsheet_row)
        step(
            "keyword/vector modes, type/document filters, validation, search page",
            modes_and_filters,
        )
        step("no cross-workspace leakage; viewers can search", isolation_and_roles)
    except (SmokeError, httpx.HTTPError, KeyError, StopIteration, IndexError) as exc:
        print(f"FAIL  {type(exc).__name__}: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    print("Phase 2 smoke test passed")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
