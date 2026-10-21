"""Shared pytest fixtures.

Integration tests use a real Postgres (pgvector) and Redis. Start them with
``make test-infra-up`` (docker-compose.test.yml) or point TEST_DATABASE_URL /
TEST_REDIS_URL at your own instances. If they are unreachable the integration tests
fail loudly; they are never silently skipped.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg
import httpx
import pytest
from pydantic import SecretStr
from redis.asyncio import Redis
from sqlalchemy.engine import make_url

from compliance_core.config import AppEnv, Settings
from compliance_core.db import migrate

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://compliance:compliance@127.0.0.1:55432/compliance_test",
)
TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://127.0.0.1:56379/15")
TEST_PASSWORD = "correct horse battery"
REPO_ROOT = Path(__file__).resolve().parent
# Downloaded once by `make fetch-artifacts` (CI caches it); tests never download.
ARTIFACT_CACHE_DIR = Path(os.environ.get("TEST_ARTIFACT_CACHE_DIR", REPO_ROOT / ".cache"))
TEST_MAX_FILE_BYTES = 2 * 1024 * 1024


async def _ensure_database(url: str) -> None:
    parsed = make_url(url)
    admin = await asyncpg.connect(
        user=parsed.username,
        password=parsed.password,
        host=parsed.host,
        port=parsed.port,
        database="postgres",
    )
    try:
        exists = await admin.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", parsed.database
        )
        if not exists:
            await admin.execute(f'CREATE DATABASE "{parsed.database}"')
    finally:
        await admin.close()


@pytest.fixture(scope="session")
def ensure_database() -> Callable[[str], Awaitable[None]]:
    return _ensure_database


@pytest.fixture(scope="session")
def test_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(
        app_env=AppEnv.TEST,
        database_url=TEST_DATABASE_URL,
        redis_url=TEST_REDIS_URL,
        jwt_secret=SecretStr("test-secret-" + "x" * 40),
        data_dir=Path(tmp_path_factory.mktemp("data")),
        login_max_failures=3,
        login_failure_window_seconds=60,
        log_level="WARNING",
        storage_dir=Path(tmp_path_factory.mktemp("storage")),
        artifact_cache_dir=ARTIFACT_CACHE_DIR,
        artifact_auto_download=False,
        max_upload_file_bytes=TEST_MAX_FILE_BYTES,
        max_upload_request_bytes=3 * TEST_MAX_FILE_BYTES,
        max_upload_files=5,
        upload_rate_limit_per_minute=1000,
        ingest_max_attempts=2,
    )


@pytest.fixture(scope="session")
def model_path(test_settings: Settings) -> Path:
    """The verified, pinned embedding model (fails loudly if not fetched)."""
    from compliance_ingestion.embeddings import ensure_model, spec_from_settings

    try:
        return ensure_model(
            test_settings.model_cache_dir, spec_from_settings(), allow_download=False
        )
    except Exception as exc:
        pytest.fail(f"embedding model unavailable ({exc}); run `make fetch-artifacts`")


@pytest.fixture(scope="session")
def dataset_zip(test_settings: Settings) -> Path:
    path = test_settings.dataset_cache_dir / "contractnli" / "contract-nli.zip"
    if not path.is_file():
        pytest.fail(f"ContractNLI archive missing at {path}; run `make fetch-artifacts`")
    return path


@pytest.fixture(scope="session")
def embedder(model_path: Path) -> Any:
    from compliance_ingestion.embeddings import Embedder

    return Embedder(model_path, threads=2, batch_size=16)


@pytest.fixture(scope="session")
def migrated_database(test_settings: Settings) -> str:
    """Create the test database if needed and migrate it from scratch to head."""
    asyncio.run(_ensure_database(test_settings.database_url))
    migrate.downgrade(test_settings.database_url, "base")
    migrate.upgrade(test_settings.database_url, "head")
    return test_settings.database_url


@pytest.fixture(scope="session")
async def api_app(test_settings: Settings, migrated_database: str) -> AsyncIterator[Any]:
    from compliance_api.main import create_app

    app = create_app(test_settings)
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
async def clean_state(api_app: Any) -> None:
    ctx = api_app.state.ctx
    async with ctx.engine.begin() as conn:
        await conn.exec_driver_sql(
            "TRUNCATE documents, workspace_members, workspaces, users RESTART IDENTITY CASCADE"
        )
    await ctx.redis.flushdb()


@dataclass
class ApiClientFactory:
    app: Any
    clients: list[httpx.AsyncClient]

    def new(self) -> httpx.AsyncClient:
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://testserver"
        )
        self.clients.append(client)
        return client

    async def signed_up(self, email: str, password: str = TEST_PASSWORD) -> httpx.AsyncClient:
        client = self.new()
        response = await client.post(
            "/api/v1/auth/register", json={"email": email, "password": password}
        )
        assert response.status_code == 201, response.text
        client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
        return client


@pytest.fixture
async def clients(api_app: Any, clean_state: None) -> AsyncIterator[ApiClientFactory]:
    factory = ApiClientFactory(app=api_app, clients=[])
    yield factory
    for client in factory.clients:
        await client.aclose()


@pytest.fixture
async def redis_client(test_settings: Settings) -> AsyncIterator[Redis]:
    client = Redis.from_url(test_settings.redis_url)
    yield client
    await client.aclose()


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any app settings from the process environment for config unit tests."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
