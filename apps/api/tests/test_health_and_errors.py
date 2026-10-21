from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from compliance_api.main import create_app
from compliance_api.openapi import render
from compliance_core.config import AppEnv, Settings

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED_SCHEMA = REPO_ROOT / "apps" / "web" / "src" / "lib" / "api" / "openapi.json"


@pytest.mark.integration
async def test_readiness_reports_all_dependencies(clients: Any) -> None:
    response = await clients.new().get("/api/v1/health/ready")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["checks"]) == {"database", "migrations", "pgvector", "redis"}
    assert all(check["status"] == "ok" for check in body["checks"].values())
    assert body["checks"]["migrations"]["detail"].startswith("revision ")


@pytest.mark.integration
async def test_liveness_and_request_id(clients: Any) -> None:
    client = clients.new()
    response = await client.get("/api/v1/health", headers={"X-Request-ID": "abc123"})
    assert response.json() == {"status": "ok", "service": "api"}
    assert response.headers["x-request-id"] == "abc123"
    assert response.headers["x-content-type-options"] == "nosniff"
    generated = await client.get("/api/v1/health")
    assert len(generated.headers["x-request-id"]) == 32


async def test_readiness_fails_when_dependencies_are_down(tmp_path: Path) -> None:
    settings = Settings(
        app_env=AppEnv.TEST,
        database_url="postgresql+asyncpg://nobody:nothing@127.0.0.1:9/none",
        redis_url="redis://127.0.0.1:9/0",
        jwt_secret=SecretStr("k" * 40),
        data_dir=tmp_path,
        log_level="CRITICAL",
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            response = await client.get("/api/v1/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert {name for name, c in body["checks"].items() if c["status"] == "error"} == {
        "database",
        "migrations",
        "pgvector",
        "redis",
    }


@pytest.mark.integration
async def test_unknown_route_uses_error_envelope(clients: Any) -> None:
    response = await clients.new().get("/api/v1/nope")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found", "message": "Not Found"}}


def test_committed_openapi_schema_is_current() -> None:
    """The web app's generated TS types come from this file; it must not drift."""
    assert COMMITTED_SCHEMA.exists(), "run `make openapi`"
    assert json.loads(COMMITTED_SCHEMA.read_text()) == json.loads(render()), (
        "apps/web/src/lib/api/openapi.json is stale; run `make openapi`"
    )


def test_openapi_documents_error_envelope() -> None:
    schema = json.loads(render())
    login = schema["paths"]["/api/v1/auth/login"]["post"]["responses"]
    assert login["401"]["content"]["application/json"]["schema"]["$ref"].endswith("ErrorResponse")
    assert login["422"]["content"]["application/json"]["schema"]["$ref"].endswith("ErrorResponse")
    assert "HTTPValidationError" not in schema["components"]["schemas"]
