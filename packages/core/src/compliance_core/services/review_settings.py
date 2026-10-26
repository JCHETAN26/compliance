"""Per-workspace review defaults. Any member may read them; only the owner may change
them. Provider *names* and models are stored, never credentials."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_core.config import Settings
from compliance_core.db.models import ReviewSettings, WorkspaceRole
from compliance_core.money import micro_to_usd, usd_to_micro_ceil
from compliance_core.review_schemas import ReviewSettingsRead, ReviewSettingsUpdate
from compliance_core.services.workspaces import require_role


@dataclass(frozen=True)
class EffectiveReviewSettings:
    llm_provider: str
    llm_model: str
    tau: float | None
    budget_tokens: int
    budget_microusd: int
    pause_on_needs_review: bool
    max_concurrency: int
    is_default: bool


def _defaults(settings: Settings, default_model: str) -> EffectiveReviewSettings:
    return EffectiveReviewSettings(
        llm_provider=settings.llm_provider,
        llm_model=default_model,
        tau=None,
        budget_tokens=settings.review_budget_tokens,
        budget_microusd=usd_to_micro_ceil(settings.review_budget_usd),
        pause_on_needs_review=settings.review_pause_on_needs_review,
        max_concurrency=settings.review_max_concurrency,
        is_default=True,
    )


async def effective_settings(
    session: AsyncSession, workspace_id: uuid.UUID, settings: Settings, default_model: str
) -> EffectiveReviewSettings:
    row = await session.get(ReviewSettings, workspace_id)
    if row is None:
        return _defaults(settings, default_model)
    return EffectiveReviewSettings(
        llm_provider=row.llm_provider,
        llm_model=row.llm_model,
        tau=row.tau,
        budget_tokens=row.budget_tokens,
        budget_microusd=row.budget_microusd,
        pause_on_needs_review=row.pause_on_needs_review,
        max_concurrency=row.max_concurrency,
        is_default=False,
    )


async def read_settings(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    settings: Settings,
    default_model: str,
) -> ReviewSettingsRead:
    await require_role(
        session, workspace_id, user_id, frozenset(WorkspaceRole), "view review settings"
    )
    eff = await effective_settings(session, workspace_id, settings, default_model)
    row = await session.get(ReviewSettings, workspace_id)
    return ReviewSettingsRead(
        workspace_id=workspace_id,
        llm_provider=eff.llm_provider,  # type: ignore[arg-type]
        llm_model=eff.llm_model,
        tau=eff.tau,
        budget_tokens=eff.budget_tokens,
        budget_usd=micro_to_usd(eff.budget_microusd),
        pause_on_needs_review=eff.pause_on_needs_review,
        max_concurrency=eff.max_concurrency,
        is_default=eff.is_default,
        updated_at=row.updated_at if row else None,
    )


async def write_settings(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    payload: ReviewSettingsUpdate,
    resolved_model: str,
) -> None:
    """``resolved_model`` has been validated by the caller against the provider catalog."""
    await require_role(
        session, workspace_id, user_id, {WorkspaceRole.OWNER}, "change review settings"
    )
    values = {
        "llm_provider": payload.llm_provider,
        "llm_model": resolved_model,
        "tau": payload.tau,
        "budget_tokens": payload.budget_tokens,
        "budget_microusd": usd_to_micro_ceil(payload.budget_usd),
        "pause_on_needs_review": payload.pause_on_needs_review,
        "max_concurrency": payload.max_concurrency,
        "updated_by": user_id,
        "updated_at": func.now(),
    }
    await session.execute(
        insert(ReviewSettings)
        .values(workspace_id=workspace_id, **values)
        .on_conflict_do_update(index_elements=[ReviewSettings.workspace_id], set_=values)
    )
    await session.commit()
