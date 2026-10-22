"""Model service with the real pinned encoder and cross-encoder (no stubs, CPU).

The rule-check endpoint must stay honest: it validates input and answers 501 with no
fabricated verdict until a trained model exists (Phase 3).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from compliance_core.config import Settings
from compliance_ingestion.embeddings import Embedder, ModelUnavailableError
from compliance_model_service.main import MAX_PENDING, Runtime, create_app, load_runtime


@pytest.fixture(scope="module")
async def client(model_service_app: Any) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=model_service_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://model") as client:
        yield client


@pytest.fixture(scope="module")
async def disabled_client(
    test_settings: Settings, model_path: Path
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(test_settings.model_copy(update={"rerank_enabled": False}))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://model") as client:
            yield client


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_health_reports_loaded_models_and_unimplemented_rule_check(
    client: httpx.AsyncClient, disabled_client: httpx.AsyncClient
) -> None:
    assert (await client.get("/health")).json() == {
        "status": "ok",
        "service": "model-service",
        "embedding": "ready",
        "rerank": "ready",
        "rule_check": "not_implemented",
    }
    assert (await disabled_client.get("/health")).json()["rerank"] == "disabled"


async def test_metadata_pins_models(client: httpx.AsyncClient, test_settings: Settings) -> None:
    body = (await client.get("/metadata")).json()
    assert body["embedding"] == {
        "model": "BAAI/bge-small-en-v1.5",
        "revision": test_settings.embedding_model_revision,
        "sha256": test_settings.embedding_model_sha256,
        "loaded": True,
        "dim": 384,
        "max_seq_length": 512,
        "query_instruction": "Represent this sentence for searching relevant passages: ",
    }
    assert body["rerank"]["model"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert body["rerank"]["revision"] == test_settings.rerank_model_revision
    assert body["rule_check"]["implemented"] is False
    assert body["rule_check"]["planned_phase"] == 3


async def test_passage_embeddings_match_ingestion_embedder(
    client: httpx.AsyncClient, embedder: Embedder
) -> None:
    texts = ["The Recipient shall not disclose Confidential Information.", "word " * 900]
    response = await client.post("/v1/embed", json={"inputs": texts, "kind": "passage"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "passage"
    assert body["query_instruction"] is None
    expected = embedder.embed(texts)
    assert body["windows"] == [e.windows for e in expected]
    assert body["windows"][1] > 1  # long input uses bounded windows, as in ingestion
    for got, want in zip(body["vectors"], expected, strict=True):
        assert _cos(got, want.vector) > 0.9999


async def test_query_embedding_uses_instruction(
    client: httpx.AsyncClient, embedder: Embedder
) -> None:
    query = "Does the agreement allow disclosure to employees?"
    body = (await client.post("/v1/embed", json={"inputs": [query], "kind": "query"})).json()
    assert body["query_instruction"] == "Represent this sentence for searching relevant passages: "
    vector = body["vectors"][0]
    assert len(vector) == 384
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, abs_tol=1e-4)
    with_instruction = embedder.embed_queries([query], body["query_instruction"])[0]
    bare = embedder.embed([query])[0].vector
    assert _cos(vector, with_instruction) > 0.9999
    assert _cos(vector, bare) < 0.999  # the instruction actually changes the vector


async def test_overlong_query_is_rejected_not_truncated(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/embed", json={"inputs": ["clause " * 600], "kind": "query"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "query_too_long"


async def test_rerank_scores_relevant_passage_higher(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/rerank",
        json={
            "query": "How long do confidentiality obligations survive termination?",
            "passages": [
                {"id": "a", "text": "This Agreement shall be governed by the laws of Delaware."},
                {
                    "id": "b",
                    "text": "The obligations of confidentiality shall survive termination "
                    "of this Agreement for a period of five years.",
                },
            ],
        },
    )
    assert response.status_code == 200, response.text
    scores = {s["id"]: s["score"] for s in response.json()["scores"]}
    assert list(scores) == ["a", "b"]  # order preserved
    assert scores["b"] > scores["a"]


async def test_rerank_disabled_is_explicit(disabled_client: httpx.AsyncClient) -> None:
    response = await disabled_client.post(
        "/v1/rerank", json={"query": "q", "passages": [{"id": "a", "text": "t"}]}
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "rerank_disabled"


def test_enabled_reranker_that_is_unavailable_fails_startup(
    test_settings: Settings, tmp_path: Path
) -> None:
    settings = test_settings.model_copy(
        update={
            "rerank_enabled": True,
            "rerank_model_revision": "0" * 40,  # not in the cache, downloads disabled
        }
    )
    with pytest.raises(ModelUnavailableError):
        load_runtime(settings)


async def test_overload_sheds_with_503(test_settings: Settings, embedder: Embedder) -> None:
    def factory(settings: Settings) -> Runtime:
        return Runtime(settings=settings, embedder=embedder, reranker=None)

    app = create_app(test_settings, runtime_factory=factory)
    async with app.router.lifespan_context(app):
        runtime: Runtime = app.state.runtime
        for _ in range(MAX_PENDING):
            await runtime.slots.acquire()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://model") as client:
            response = await client.post("/v1/embed", json={"inputs": ["x"], "kind": "query"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "overloaded"
        for _ in range(MAX_PENDING):
            runtime.slots.release()
        await asyncio.sleep(0)


async def test_check_returns_501_without_verdict(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/check",
        json={
            "hypothesis": "The receiving party may share information with employees.",
            "spans": [{"span_id": "s1", "text": "Recipient may disclose to its employees."}],
        },
    )
    assert response.status_code == 501
    body = response.json()
    assert body["error"]["code"] == "not_implemented"
    for fabricated in ("label", "confidence", "evidence"):
        assert fabricated not in body


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/check", {}),
        ("/v1/check", {"hypothesis": "", "spans": [{"span_id": "s", "text": "t"}]}),
        ("/v1/check", {"hypothesis": "h", "spans": []}),
        ("/v1/embed", {"inputs": [], "kind": "query"}),
        ("/v1/embed", {"inputs": ["x"], "kind": "document"}),
        ("/v1/embed", {"inputs": ["x"] * 65, "kind": "passage"}),
        ("/v1/embed", {"inputs": [""], "kind": "query"}),
        ("/v1/rerank", {"query": "q", "passages": []}),
        ("/v1/rerank", {"query": "q", "passages": [{"id": "a", "text": "t", "x": 1}]}),
    ],
)
async def test_invalid_requests_are_rejected(
    client: httpx.AsyncClient, path: str, payload: dict[str, Any]
) -> None:
    assert (await client.post(path, json=payload)).status_code == 422
