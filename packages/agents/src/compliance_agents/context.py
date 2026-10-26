"""Everything one run's agents need, bound once per job (never stored in graph state)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from compliance_agents.gateway import LLMGateway
from compliance_agents.model import RuleCheckModel
from compliance_agents.store import RunStore
from compliance_core.config import Settings
from compliance_llm import LLMProvider
from compliance_retrieval.service import SearchDependencies


@dataclass
class RunContext:
    settings: Settings
    sessions: async_sessionmaker[AsyncSession]
    run_id: uuid.UUID
    workspace_id: uuid.UUID
    store: RunStore
    provider: LLMProvider
    llm: LLMGateway
    model: RuleCheckModel
    search: SearchDependencies | None
    spec: dict[str, Any]
    rule_ids: dict[str, str]
    config: dict[str, Any]
    rules: dict[str, dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self.rules = {r["key"]: r for r in self.spec["rules"]}

    def rule_id(self, key: str) -> uuid.UUID | None:
        value = self.rule_ids.get(key)
        return uuid.UUID(value) if value else None

    @property
    def register_columns(self) -> dict[str, list[str]]:
        register = self.spec.get("register") or {}
        columns: dict[str, list[str]] = register.get("columns", {})
        return columns
