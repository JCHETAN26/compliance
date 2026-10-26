"""GET /api/v1/models/status: honest rule-check model state for the UI banner."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from compliance_core.model_client import ModelServiceClient

EMBEDDING = {"model": "m", "revision": "r", "dim": 384, "query_instruction": "q"}


def _client_with(handler: Any) -> ModelServiceClient:
    return ModelServiceClient(
        "http://model", timeout=1, retries=0, backoff=0, transport=httpx.MockTransport(handler)
    )


@pytest.fixture
def swap_models(api_app: Any) -> Iterator[Any]:
    ctx = api_app.state.ctx
    original = ctx.models

    def swap(client: ModelServiceClient) -> None:
        ctx.models = client

    yield swap
    ctx.models = original


async def test_real_service_without_artifact_is_untrained_fallback(
    clients: Any, api_app: Any, model_client: ModelServiceClient
) -> None:
    ctx = api_app.state.ctx
    original, ctx.models = ctx.models, model_client
    try:
        assert (await clients.new().get("/api/v1/models/status")).status_code == 401
        user = await clients.signed_up("status@example.com")
        response = await user.get("/api/v1/models/status")
    finally:
        ctx.models = original
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()["rule_check"]
    assert body["state"] == "untrained_fallback"
    assert body["model_version"] is None
    assert "/" not in body["message"]  # no server paths or internal reasons leak


@pytest.mark.parametrize(
    ("rule_check", "state"),
    [
        (
            {
                "available": True,
                "mode": "model",
                "kind": "production",
                "model_version": "rule-check-x",
                "variant": "int8",
                "artifact_dir": "/cache/x",
            },
            "trained",
        ),
        (
            {
                "available": True,
                "mode": "model",
                "kind": "smoke",
                "model_version": "rule-check-smoke",
                "variant": "int8",
            },
            "smoke",
        ),
        (
            {"available": False, "mode": "untrained_fallback", "reason": "/secret/path missing"},
            "untrained_fallback",
        ),
        (
            {"available": True, "mode": "untrained_fallback"},
            "unreachable",
        ),  # inconsistent -> not trusted
        (None, "unreachable"),
    ],
)
async def test_states(
    clients: Any, swap_models: Any, rule_check: dict[str, Any] | None, state: str
) -> None:
    metadata: dict[str, Any] = {"embedding": EMBEDDING, "rerank": None}
    if rule_check is not None:
        metadata["rule_check"] = rule_check
    swap_models(_client_with(lambda request: httpx.Response(200, json=metadata)))
    user = await clients.signed_up("states@example.com")
    body = (await user.get("/api/v1/models/status")).json()["rule_check"]
    assert body["state"] == state
    assert "secret" not in str(body)
    if state == "trained":
        assert body["model_version"] == "rule-check-x"
        assert body["variant"] == "int8"


async def test_outage_is_reported_not_raised(clients: Any, swap_models: Any) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    swap_models(_client_with(refuse))
    user = await clients.signed_up("outage@example.com")
    response = await user.get("/api/v1/models/status")
    assert response.status_code == 200
    assert response.json()["rule_check"]["state"] == "unreachable"


async def test_messages_reflect_the_review_backend(
    clients: Any, api_app: Any, model_client: ModelServiceClient
) -> None:
    client = await clients.signed_up("backend@example.com")
    ctx = api_app.state.ctx
    original, original_models = ctx.settings, ctx.models
    ctx.models = model_client  # the real in-process service, untrained fallback
    try:
        ctx.settings = original.model_copy(update={"review_rule_check_backend": "mock"})
        body = (await client.get("/api/v1/models/status")).json()
        assert body["review_rule_check_backend"] == "mock"
        assert "mock rule-check backend" in body["rule_check"]["message"]
        assert "unavailable" not in body["rule_check"]["message"]
        ctx.settings = original
        body = (await client.get("/api/v1/models/status")).json()
        assert body["review_rule_check_backend"] == "service"
        assert "send model-supported rules to the LLM" in body["rule_check"]["message"]
    finally:
        ctx.settings = original
        ctx.models = original_models
