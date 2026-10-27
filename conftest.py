"""Shared pytest fixtures.

Integration tests use a real Postgres (pgvector) and Redis. Start them with
``make test-infra-up`` (docker-compose.test.yml) or point TEST_DATABASE_URL /
TEST_REDIS_URL at your own instances. If they are unreachable the integration tests
fail loudly; they are never silently skipped.
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass, field
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
        # Deterministic: never pick up a rule-check artifact installed in ./.cache.
        rule_check_artifact_dir=Path(tmp_path_factory.mktemp("no-rule-check-artifact")),
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
def reranker_path(test_settings: Settings) -> Path:
    """The verified, pinned cross-encoder (fails loudly if not fetched)."""
    from compliance_ingestion.embeddings import ensure_model
    from compliance_model_service.rerank import rerank_spec

    try:
        return ensure_model(
            test_settings.model_cache_dir, rerank_spec(test_settings), allow_download=False
        )
    except Exception as exc:
        pytest.fail(f"reranker model unavailable ({exc}); run `make fetch-artifacts`")


@pytest.fixture(scope="session")
async def model_service_app(
    test_settings: Settings, model_path: Path, reranker_path: Path
) -> AsyncIterator[Any]:
    """The real model service (pinned encoder + pinned reranker), served in-process."""
    from compliance_model_service.main import create_app

    app = create_app(test_settings.model_copy(update={"rerank_enabled": True}))
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture(scope="session")
async def model_client(model_service_app: Any) -> AsyncIterator[Any]:
    """The production HTTP client, talking to the real in-process model service."""
    from compliance_core.model_client import ModelServiceClient

    client = ModelServiceClient(
        "http://model-service",
        timeout=60,
        retries=0,
        transport=httpx.ASGITransport(app=model_service_app),
    )
    yield client
    await client.aclose()


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


Ingest = Callable[[Any, Any, str, bytes, str], Awaitable[Any]]


@pytest.fixture(scope="session")
def ingest_ready(api_app: Any, embedder: Any, model_path: Path, test_settings: Settings) -> Ingest:
    """Register an upload and run the real ingestion job to READY.

    Small chunk sizes give several chunks per short test document. Documents are
    labelled with the real encoder identity, exactly as the worker does.
    """
    import hashlib

    from compliance_core.model_client import embedding_label
    from compliance_core.services import documents as document_service
    from compliance_worker.ingest import IngestDeps, ingest_document

    ctx = api_app.state.ctx
    deps = IngestDeps(
        settings=test_settings.model_copy(
            update={"chunk_min_tokens": 24, "chunk_max_tokens": 48, "chunk_overlap_tokens": 0}
        ),
        session_factory=ctx.session_factory,
        storage=ctx.storage,
        embedder=embedder,
        tokenizer_path=model_path / "tokenizer.json",
        model_label=embedding_label(
            test_settings.embedding_model_id, test_settings.embedding_model_revision
        ),
        embed_lock=asyncio.Lock(),
    )

    async def run(workspace_id: Any, user_id: Any, filename: str, data: bytes, fmt: str) -> Any:
        staged = deps.storage.new_staging_file()
        staged.write_bytes(data)
        async with deps.session_factory() as session:
            registered = await document_service.register_upload(
                session,
                deps.storage,
                workspace_id=workspace_id,
                user_id=user_id,
                filename=filename,
                fmt=fmt,  # type: ignore[arg-type]
                staged_path=staged,
                sha256=hashlib.sha256(data).hexdigest(),
                size=len(data),
            )
        doc_id = registered.document.id
        result = await ingest_document({"ingest": deps}, str(doc_id), 1)
        assert result["status"] == "ready", result
        return doc_id

    return run


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


# --------------------------------------------------------------------------- MCP server
# A real model service and MCP HTTP app on loopback ports (threads; reachable from MCP
# subprocesses), a workspace with an ingested contract, API tokens, and an in-process
# stand-in for the worker that executes queued runs with the mock LLM and mock model.

MCP_OWNER = "mcp-owner@example.com"
MCP_CONTRACT = "globex_nda.docx"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class ThreadedServer:
    url: str
    server: Any
    thread: threading.Thread

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=20)


def serve_in_thread(app: Any, startup_seconds: float = 120) -> ThreadedServer:
    import uvicorn

    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, lifespan="on")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + startup_seconds
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("server did not start")
        time.sleep(0.05)
    return ThreadedServer(f"http://127.0.0.1:{port}", server, thread)


@pytest.fixture(scope="session")
def model_service_url(test_settings: Settings, model_path: Path) -> Iterator[str]:
    """The real model service (pinned encoder, no rule-check artifact) on a loopback
    port. Never the host's own model service."""
    from compliance_model_service.main import create_app

    served = serve_in_thread(create_app(test_settings))
    yield served.url
    served.stop()


@pytest.fixture(scope="session")
def mcp_settings_for_tests(test_settings: Settings, model_service_url: str) -> Settings:
    return test_settings.model_copy(
        update={
            "model_service_url": model_service_url,
            "review_rule_check_backend": "mock",
            "llm_provider": "mock",
        }
    )


@dataclass
class McpEnv:
    client: httpx.AsyncClient
    user_id: uuid.UUID
    workspace_id: uuid.UUID
    sessions: Any
    settings: Settings
    model_service_url: str
    docs: dict[str, uuid.UUID] = field(default_factory=dict)

    async def token(
        self,
        scopes: tuple[str, ...] = ("read",),
        workspace_ids: list[uuid.UUID] | None = None,
        user_id: uuid.UUID | None = None,
    ) -> str:
        from compliance_core.services import api_tokens
        from compliance_core.token_schemas import ApiTokenCreate

        async with self.sessions() as session:
            created = await api_tokens.create_token(
                session,
                user_id or self.user_id,
                ApiTokenCreate.model_validate(
                    {"name": "test", "scopes": list(scopes), "workspace_ids": workspace_ids}
                ),
            )
        return created.token

    def runner(self) -> Any:
        from compliance_agents.model import MockRuleCheckModel
        from compliance_agents.runner import RunnerDeps
        from compliance_core.model_client import ModelServiceClient
        from compliance_llm import MockProvider
        from compliance_retrieval.config import RetrievalConfig
        from compliance_retrieval.service import SearchDependencies

        return RunnerDeps(
            settings=self.settings,
            sessions=self.sessions,
            search=SearchDependencies(
                session_factory=self.sessions,
                models=ModelServiceClient(self.model_service_url, timeout=60, retries=0),
                config=RetrievalConfig.from_settings(self.settings),
            ),
            worker_id="mcp-test-worker",
            provider_factory=lambda settings, run: MockProvider(),
            model_factory=lambda settings, run: MockRuleCheckModel(),
        )

    async def drain(self) -> dict[uuid.UUID, str]:
        """Execute every queued run, as the worker would."""
        from sqlalchemy import select

        from compliance_agents.runner import execute_run
        from compliance_core.db.models import ReviewRun

        async with self.sessions() as session:
            queued = list(
                await session.scalars(select(ReviewRun.id).where(ReviewRun.status == "queued"))
            )
        return {run_id: await execute_run(self.runner(), run_id) for run_id in queued}

    async def background_worker(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.drain()
            await asyncio.sleep(0.2)


@pytest.fixture
async def mcp_env(
    clients: ApiClientFactory,
    api_app: Any,
    ingest_ready: Ingest,
    mcp_settings_for_tests: Settings,
    model_service_url: str,
) -> McpEnv:
    from compliance_core.services.rule_packs import sync_builtin_packs
    from compliance_seed.synthetic import build

    ctx = api_app.state.ctx
    client = await clients.signed_up(MCP_OWNER)
    me = (await client.get("/api/v1/auth/me")).json()
    ws = await client.post("/api/v1/workspaces", json={"name": "MCP workspace"})
    assert ws.status_code == 201, ws.text
    async with ctx.session_factory() as session:
        await sync_builtin_packs(session)
    env = McpEnv(
        client=client,
        user_id=uuid.UUID(me["id"]),
        workspace_id=uuid.UUID(ws.json()["id"]),
        sessions=ctx.session_factory,
        settings=mcp_settings_for_tests,
        model_service_url=model_service_url,
    )
    env.docs[MCP_CONTRACT] = await ingest_ready(
        env.workspace_id, env.user_id, MCP_CONTRACT, build()[MCP_CONTRACT], "docx"
    )
    return env


@pytest.fixture
async def mcp_http(mcp_settings_for_tests: Settings, migrated_database: str) -> AsyncIterator[str]:
    """The MCP streamable-HTTP app (bearer auth, stateless) on a loopback port."""
    from compliance_mcp_server.config import McpSettings
    from compliance_mcp_server.runtime import McpRuntime
    from compliance_mcp_server.server import build_http_app

    runtime = McpRuntime(transport="http", settings=mcp_settings_for_tests, mcp=McpSettings())
    served = await asyncio.to_thread(serve_in_thread, build_http_app(runtime), 30)
    yield served.url
    await asyncio.to_thread(served.stop)
