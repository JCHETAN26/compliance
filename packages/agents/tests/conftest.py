"""Fixtures for review-graph integration tests (real Postgres/Redis, real ingestion with
the pinned embedder and Tesseract OCR, the real in-process model service for query
embeddings, the mock LLM and the mock rule-check backend)."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from compliance_agents.model import MockRuleCheckModel, RuleCheckModel
from compliance_agents.runner import RunnerDeps, execute_run
from compliance_core.config import Settings
from compliance_core.db.models import ReviewRun, RulePack
from compliance_core.money import usd_to_micro_ceil
from compliance_core.services import reviews
from compliance_core.services.rule_packs import sync_builtin_packs
from compliance_llm import LLMProvider, MockProvider
from compliance_retrieval.config import RetrievalConfig
from compliance_retrieval.service import SearchDependencies

OWNER = "owner@example.com"


@dataclass
class ReviewEnv:
    client: httpx.AsyncClient
    user_id: uuid.UUID
    workspace_id: uuid.UUID
    sessions: async_sessionmaker[AsyncSession]
    settings: Settings
    search: SearchDependencies
    docs: dict[str, uuid.UUID] = field(default_factory=dict)

    async def pack_id(self, key: str = "vendor_confidentiality_review") -> uuid.UUID:
        async with self.sessions() as session:
            pack = await session.scalar(
                select(RulePack).where(RulePack.key == key, RulePack.workspace_id.is_(None))
            )
            assert pack is not None
            return pack.id

    async def start(
        self,
        *,
        pack_key: str = "vendor_confidentiality_review",
        documents: list[str] | None = None,
        budget_tokens: int = 2_000_000,
        budget_usd: float = 5.0,
        tau: float | None = None,
        pause_on_needs_review: bool = False,
        max_concurrency: int = 4,
        provider: str = "mock",
        model: str = "mock-deterministic-v1",
        price: tuple[float, float] = (0.0, 0.0),
    ) -> uuid.UUID:
        async with self.sessions() as session:
            run = await reviews.start_review(
                session,
                workspace_id=self.workspace_id,
                user_id=self.user_id,
                rule_pack_id=await self.pack_id(pack_key),
                document_ids=[self.docs[d] for d in documents] if documents else None,
                provider=reviews.ProviderSelection(
                    provider=provider,
                    model=model,
                    is_mock=provider == "mock",
                    price_source="test",
                    input_usd_per_mtok=price[0],
                    output_usd_per_mtok=price[1],
                ),
                options=reviews.RunOptions(
                    budget_tokens=budget_tokens,
                    budget_microusd=usd_to_micro_ceil(budget_usd),
                    tau=tau,
                    pause_on_needs_review=pause_on_needs_review,
                    max_concurrency=max_concurrency,
                    rule_check_backend="mock",
                    max_documents=100,
                ),
            )
            return run.id

    def deps(
        self,
        provider: LLMProvider | Callable[[], LLMProvider] | None = None,
        model: RuleCheckModel | None = None,
        worker_id: str = "test-worker",
    ) -> RunnerDeps:
        def make_provider(settings: Settings, run: ReviewRun) -> LLMProvider:
            if provider is None:
                return MockProvider()
            return provider() if callable(provider) else provider

        return RunnerDeps(
            settings=self.settings,
            sessions=self.sessions,
            search=self.search,
            worker_id=worker_id,
            provider_factory=make_provider,
            model_factory=lambda settings, run: model or MockRuleCheckModel(),
        )

    async def execute(self, run_id: uuid.UUID, **kwargs: Any) -> str:
        return await execute_run(self.deps(**kwargs), run_id)

    async def findings(self, run_id: uuid.UUID) -> dict[str, dict[str, Any]]:
        response = await self.client.get(f"/api/v1/reviews/{run_id}/findings")
        assert response.status_code == 200, response.text
        return {f["task_key"]: f for f in response.json()["items"]}

    async def run(self, run_id: uuid.UUID) -> ReviewRun:
        async with self.sessions() as session:
            run = await session.get(ReviewRun, run_id)
            assert run is not None
            return run


EnvFactory = Callable[..., Awaitable[ReviewEnv]]


@pytest.fixture
async def review_env(
    clients: Any, api_app: Any, ingest_ready: Any, test_settings: Settings, model_client: Any
) -> EnvFactory:
    ctx = api_app.state.ctx
    settings = test_settings.model_copy(update={"review_lease_seconds": 30, "llm_max_attempts": 3})

    async def make(files: dict[str, bytes], email: str = OWNER) -> ReviewEnv:
        client = await clients.signed_up(email)
        me = (await client.get("/api/v1/auth/me")).json()
        ws = await client.post("/api/v1/workspaces", json={"name": "Vendor review"})
        assert ws.status_code == 201, ws.text
        async with ctx.session_factory() as session:
            await sync_builtin_packs(session)
        env = ReviewEnv(
            client=client,
            user_id=uuid.UUID(me["id"]),
            workspace_id=uuid.UUID(ws.json()["id"]),
            sessions=ctx.session_factory,
            settings=settings,
            search=SearchDependencies(
                session_factory=ctx.session_factory,
                models=model_client,
                config=RetrievalConfig.from_settings(settings),
            ),
        )
        for name, data in files.items():
            env.docs[name] = await ingest_ready(
                env.workspace_id, env.user_id, name, data, name.rsplit(".", 1)[1]
            )
        return env

    return make
