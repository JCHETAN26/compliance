"""Model-service client: retries, timeouts, error classes and response contracts."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from compliance_core.model_client import (
    ModelServiceClient,
    ModelServiceProtocolError,
    ModelServiceRequestError,
    ModelServiceUnavailableError,
    embedding_label,
)

UNIT = [1.0] + [0.0] * 383


def embed_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "dim": 384,
        "kind": "query",
        "query_instruction": "Represent this sentence for searching relevant passages: ",
        "vectors": [UNIT],
        "windows": [1],
    }
    body.update(overrides)
    return body


def client_for(
    handler: Callable[[httpx.Request], httpx.Response], retries: int = 2
) -> tuple[ModelServiceClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = ModelServiceClient(
        "http://model",
        timeout=1.0,
        retries=retries,
        transport=httpx.MockTransport(record),
        backoff=0.0,
    )
    return client, seen


async def test_embed_query_success() -> None:
    client, seen = client_for(lambda r: httpx.Response(200, json=embed_body()))
    embedding = await client.embed_query("q")
    assert embedding.vector == UNIT
    assert embedding.label == "BAAI/bge-small-en-v1.5@5c38ec7c405e"
    assert embedding.label == embedding_label(embedding.model, embedding.revision)
    assert json.loads(seen[0].content) == {"inputs": ["q"], "kind": "query"}
    await client.aclose()


async def test_retryable_failures_are_retried_then_succeed() -> None:
    responses = iter(
        [httpx.Response(503), httpx.Response(502), httpx.Response(200, json=embed_body())]
    )
    client, seen = client_for(lambda r: next(responses))
    assert (await client.embed_query("q")).vector == UNIT
    assert len(seen) == 3
    await client.aclose()


async def test_exhausted_retries_and_timeouts_are_unavailable() -> None:
    client, seen = client_for(lambda r: httpx.Response(503), retries=1)
    with pytest.raises(ModelServiceUnavailableError, match="2 attempt"):
        await client.embed_query("q")
    assert len(seen) == 2

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client2, seen2 = client_for(timeout, retries=2)
    with pytest.raises(ModelServiceUnavailableError, match="timeout"):
        await client2.embed_query("q")
    assert len(seen2) == 3

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client3, _ = client_for(refused, retries=0)
    with pytest.raises(ModelServiceUnavailableError, match="ConnectError"):
        await client3.embed_query("q")
    for c in (client, client2, client3):
        await c.aclose()


async def test_server_error_is_not_a_user_error_and_not_retried() -> None:
    client, seen = client_for(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(ModelServiceUnavailableError) as info:
        await client.embed_query("q")
    assert not isinstance(info.value, ModelServiceRequestError)
    assert info.value.code == "model_service_failed"
    assert len(seen) == 1
    await client.aclose()


async def test_rejected_input_surfaces_error_code() -> None:
    client, seen = client_for(
        lambda r: httpx.Response(
            422, json={"error": {"code": "query_too_long", "message": "too long"}}
        )
    )
    with pytest.raises(ModelServiceRequestError) as info:
        await client.embed_query("q")
    assert (info.value.code, info.value.message) == ("query_too_long", "too long")
    assert len(seen) == 1
    await client.aclose()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"vectors": [UNIT]}),  # missing fields
        httpx.Response(200, json=embed_body(vectors=[])),
        httpx.Response(200, json=embed_body(vectors=[UNIT, UNIT])),
        httpx.Response(200, json=embed_body(vectors=[[1.0] * 10])),
        httpx.Response(200, json=embed_body(dim=768)),
        httpx.Response(200, json=embed_body(vectors=[["x"] * 384])),
        httpx.Response(200, json=embed_body(vectors=[[0.5] * 384])),  # not unit-norm
        httpx.Response(200, json=embed_body(kind="passage", query_instruction=None)),
        httpx.Response(
            200,
            content=json.dumps(embed_body(vectors=[[math.nan] + [0.0] * 383])).encode(),
            headers={"content-type": "application/json"},
        ),
        httpx.Response(404, text="<html>not found</html>"),  # 4xx without an error body
        httpx.Response(204),
    ],
)
async def test_contract_violations_fail_clearly(response: httpx.Response) -> None:
    client, _ = client_for(lambda r: response)
    with pytest.raises(ModelServiceProtocolError):
        await client.embed_query("q")
    await client.aclose()


def respond(response: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _: response


def rerank_response(scores: Any) -> httpx.Response:
    return httpx.Response(200, json={"model": "m", "revision": "r", "scores": scores})


async def test_rerank_contract() -> None:
    client, _ = client_for(
        lambda r: rerank_response([{"id": "a", "score": 1.5}, {"id": "b", "score": -2}])
    )
    assert await client.rerank("q", [("a", "x"), ("b", "y")]) == {"a": 1.5, "b": -2.0}
    await client.aclose()
    for bad in (
        [{"id": "a", "score": 1.0}],  # missing id
        [{"id": "a", "score": 1.0}, {"id": "zzz", "score": 1.0}],  # unexpected id
        [{"id": "a", "score": 1.0}, {"id": "a", "score": 2.0}],  # duplicate id
        [{"id": "a", "score": 1.0}, {"id": "b", "score": "high"}],
    ):
        client, _ = client_for(respond(rerank_response(bad)))
        with pytest.raises(ModelServiceProtocolError):
            await client.rerank("q", [("a", "x"), ("b", "y")])
        await client.aclose()
    nan_body = json.dumps(
        {"model": "m", "revision": "r", "scores": [{"id": "a", "score": math.inf}]}
    ).encode()
    client, _ = client_for(
        lambda r: httpx.Response(
            200, content=nan_body, headers={"content-type": "application/json"}
        )
    )
    with pytest.raises(ModelServiceProtocolError):
        await client.rerank("q", [("a", "x")])
    await client.aclose()


async def test_info_contract() -> None:
    metadata = {
        "embedding": {"model": "e", "revision": "r1", "dim": 384, "query_instruction": "Q: "},
        "rerank": None,
    }
    client, _ = client_for(lambda r: httpx.Response(200, json=metadata))
    info = await client.info()
    assert (info.embedding_model, info.rerank_available, info.rerank_model) == ("e", False, None)
    await client.aclose()
    metadata["embedding"]["dim"] = 768  # type: ignore[index]
    client, _ = client_for(lambda r: httpx.Response(200, json=metadata))
    with pytest.raises(ModelServiceProtocolError):
        await client.info()
    await client.aclose()
