# mypy: disallow-untyped-calls=False
# (fixtures are built with PyMuPDF, which ships partial annotations)
"""Search API: auth, membership (viewers included), validation, rate limits, failure
states, and exact source references - against real Postgres and the real model service."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import httpx
import pymupdf
import pytest

from compliance_core.model_client import ModelServiceClient
from compliance_core.services.session_store import FixedWindowLimiter

pytestmark = pytest.mark.integration

REGISTER = b"Vendor,Service,Data access\nAcme Logistics,Freight,Customer addresses\nCobalt Payroll,Payroll processing,Employee bank account details\n"


def policy_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Access Control Policy")
    page.insert_text((72, 100), "All production database access requires multi-factor")
    page.insert_text((72, 116), "authentication and quarterly access reviews by security.")
    data = doc.tobytes()
    doc.close()
    return bytes(data)


@pytest.fixture
def real_models(api_app: Any, model_client: ModelServiceClient) -> Iterator[None]:
    ctx = api_app.state.ctx
    original = ctx.models
    ctx.models = model_client
    yield
    ctx.models = original


def _client_with(handler: Any) -> ModelServiceClient:
    return ModelServiceClient(
        "http://model", timeout=1, retries=1, backoff=0, transport=httpx.MockTransport(handler)
    )


@pytest.fixture
def swap_models(api_app: Any) -> Iterator[Any]:
    ctx = api_app.state.ctx
    original = ctx.models

    def swap(client: ModelServiceClient) -> None:
        ctx.models = client

    yield swap
    ctx.models = original


async def _user_id(client: httpx.AsyncClient) -> uuid.UUID:
    return uuid.UUID((await client.get("/api/v1/auth/me")).json()["id"])


async def _setup(clients: Any, ingest_ready: Any) -> tuple[httpx.AsyncClient, str, dict[str, str]]:
    owner = await clients.signed_up("owner@example.com")
    ws = (await owner.post("/api/v1/workspaces", json={"name": "Search"})).json()["id"]
    uid = await _user_id(owner)
    docs = {
        "register": str(await ingest_ready(uuid.UUID(ws), uid, "vendors.csv", REGISTER, "csv")),
        "policy": str(await ingest_ready(uuid.UUID(ws), uid, "access.pdf", policy_pdf(), "pdf")),
    }
    return owner, ws, docs


@pytest.fixture
async def setup(
    clients: Any, ingest_ready: Any, real_models: None
) -> tuple[Any, httpx.AsyncClient, str, dict[str, str]]:
    owner, ws, docs = await _setup(clients, ingest_ready)
    return clients, owner, ws, docs


async def test_search_returns_exact_references(setup: Any) -> None:
    _, owner, ws, docs = setup
    response = await owner.post(
        f"/api/v1/workspaces/{ws}/search",
        json={"query": "who processes employee bank account details?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    top = body["results"][0]
    assert top["document_id"] == docs["register"]
    assert top["sheet_name"] is not None
    assert top["row_number"] == 3
    assert "Cobalt Payroll" in top["text"]
    assert top["char_end"] > top["char_start"]
    chunks = (await owner.get(f"/api/v1/documents/{docs['register']}/chunks")).json()["items"]
    stored = next(c for c in chunks if c["id"] == top["chunk_id"])
    for key in ("text", "char_start", "char_end", "table_row_id", "section_path"):
        assert stored[key] == top[key]
    rows = (await owner.get(f"/api/v1/documents/{docs['register']}/rows")).json()
    row = next(r for r in rows["items"] if r["id"] == top["table_row_id"])
    assert (row["row_number"], row["sheet_name"]) == (top["row_number"], top["sheet_name"])
    assert (row["char_start"], row["char_end"]) == (top["char_start"], top["char_end"])
    assert body["settings"]["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert body["timings"]["total_ms"] > 0
    assert response.headers["cache-control"] == "no-store"


async def test_viewers_search_but_outsiders_and_anonymous_cannot(setup: Any) -> None:
    clients, owner, ws, _ = setup
    viewer = await clients.signed_up("viewer@example.com")
    added = await owner.post(
        f"/api/v1/workspaces/{ws}/members", json={"email": "viewer@example.com", "role": "viewer"}
    )
    assert added.status_code == 201
    outsider = await clients.signed_up("outsider@example.com")
    payload = {"query": "multi-factor authentication"}
    ok = await viewer.post(f"/api/v1/workspaces/{ws}/search", json=payload)
    assert ok.status_code == 200
    assert ok.json()["results"]
    denied = await outsider.post(f"/api/v1/workspaces/{ws}/search", json=payload)
    assert denied.status_code == 404
    assert "results" not in denied.json()
    anonymous = clients.new()
    assert (
        await anonymous.post(f"/api/v1/workspaces/{ws}/search", json=payload)
    ).status_code == 401
    missing = await owner.post(f"/api/v1/workspaces/{uuid.uuid4()}/search", json=payload)
    assert missing.status_code == 404


async def test_foreign_document_filter_is_not_found(setup: Any, ingest_ready: Any) -> None:
    clients, owner, ws, docs = setup
    other = await clients.signed_up("other@example.com")
    other_ws = (await other.post("/api/v1/workspaces", json={"name": "Other"})).json()["id"]
    foreign = await ingest_ready(
        uuid.UUID(other_ws), await _user_id(other), "x.csv", b"Name,Note\nA,Other secret\n", "csv"
    )
    response = await owner.post(
        f"/api/v1/workspaces/{ws}/search",
        json={"query": "secret", "document_ids": [docs["policy"], str(foreign)]},
    )
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Document not found."
    scoped = await owner.post(
        f"/api/v1/workspaces/{ws}/search",
        json={"query": "authentication", "document_ids": [docs["policy"]]},
    )
    assert {r["document_id"] for r in scoped.json()["results"]} == {docs["policy"]}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 1001},
        {"query": "x", "top_k": 0},
        {"query": "x", "top_k": 51},
        {"query": "x", "mode": "semantic"},
        {"query": "x", "doc_types": ["poem"]},
        {"query": "x", "document_ids": ["not-a-uuid"]},
        {"query": "x", "ef_search": 5},
    ],
)
async def test_invalid_payloads_are_rejected(setup: Any, payload: dict[str, Any]) -> None:
    _, owner, ws, _ = setup
    response = await owner.post(f"/api/v1/workspaces/{ws}/search", json=payload)
    assert response.status_code == 422


async def test_search_is_rate_limited(setup: Any, api_app: Any) -> None:
    _, owner, ws, _ = setup
    ctx = api_app.state.ctx
    original = ctx.search_limiter
    ctx.search_limiter = FixedWindowLimiter(ctx.redis, "search-test", 2, "searches")
    try:
        statuses = [
            (
                await owner.post(
                    f"/api/v1/workspaces/{ws}/search", json={"query": "vendor", "mode": "keyword"}
                )
            ).status_code
            for _ in range(3)
        ]
    finally:
        ctx.search_limiter = original
    assert statuses == [200, 200, 429]


async def test_model_service_outage_is_503_but_keyword_mode_works(
    clients: Any, ingest_ready: Any, swap_models: Any
) -> None:
    owner, ws, _ = await _setup(clients, ingest_ready)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    swap_models(_client_with(refuse))
    response = await owner.post(f"/api/v1/workspaces/{ws}/search", json={"query": "vendor"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "search_unavailable"
    keyword = await owner.post(
        f"/api/v1/workspaces/{ws}/search", json={"query": "vendor", "mode": "keyword"}
    )
    assert keyword.status_code == 200
    assert keyword.json()["results"]
    capabilities = await owner.get("/api/v1/search/capabilities")
    assert capabilities.status_code == 503


async def test_rerank_when_disabled_is_409(
    clients: Any, ingest_ready: Any, swap_models: Any
) -> None:
    owner, ws, _ = await _setup(clients, ingest_ready)
    metadata = {
        "embedding": {"model": "m", "revision": "r", "dim": 384, "query_instruction": "q"},
        "rerank": None,
    }
    swap_models(_client_with(lambda request: httpx.Response(200, json=metadata)))
    response = await owner.post(
        f"/api/v1/workspaces/{ws}/search", json={"query": "vendor", "rerank": True}
    )
    assert response.status_code == 409
    assert "Reranking is not enabled" in response.json()["error"]["message"]
    capabilities = (await owner.get("/api/v1/search/capabilities")).json()
    assert capabilities["rerank_available"] is False


async def test_capabilities_and_rerank(setup: Any) -> None:
    _, owner, ws, _ = setup
    capabilities = (await owner.get("/api/v1/search/capabilities")).json()
    assert capabilities["rerank_available"] is True
    assert capabilities["modes"] == ["hybrid", "vector", "keyword"]
    assert capabilities["embedding_model"] == "BAAI/bge-small-en-v1.5"
    response = await owner.post(
        f"/api/v1/workspaces/{ws}/search",
        json={"query": "multi-factor authentication", "rerank": True, "top_k": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reranked"] is True
    assert all(r["provenance"]["rerank_score"] is not None for r in body["results"])
