"""The skeleton must be honest: it is healthy, but never returns a fabricated verdict."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from compliance_model_service.main import create_app


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://model") as client:
        yield client


async def test_health_reports_no_model_loaded(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "model-service",
        "mode": "not_implemented",
        "model_loaded": False,
    }


async def test_metadata_declares_not_implemented(client: httpx.AsyncClient) -> None:
    body = (await client.get("/metadata")).json()
    assert body["implemented"] is False
    assert body["planned_phase"] == 3
    assert body["model_version"] is None
    assert body["metrics"] is None


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
    "payload",
    [
        {},
        {"hypothesis": "", "spans": [{"span_id": "s", "text": "t"}]},
        {"hypothesis": "h", "spans": []},
        {"hypothesis": "h", "spans": [{"span_id": "s"}]},
    ],
)
async def test_check_validates_contract(client: httpx.AsyncClient, payload: object) -> None:
    response = await client.post("/v1/check", json=payload)
    assert response.status_code == 422
