"""Starting review runs, shared by every transport (REST API and MCP).

Order of checks (unchanged from the original REST route): the caller's role, the
workspace's review defaults, the provider/model choice (validated by a
:class:`ProviderResolver`, which never sees a credential), the per-user review-start rate
limit, then run creation and enqueueing. The worker executes the run.

The resolver is injected because provider resolution lives in ``compliance_llm`` (which
depends on this package) and because transports differ in how they learn whether the
worker holds a provider credential: the API checks its own environment, the MCP server a
non-secret capability declaration.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from compliance_core.db.models import ReviewRun
from compliance_core.money import usd_to_micro_ceil
from compliance_core.review_schemas import StartReview
from compliance_core.runtime import ServiceContext
from compliance_core.services import reviews
from compliance_core.services.review_settings import EffectiveReviewSettings, effective_settings
from compliance_core.services.workspaces import require_role


class ProviderResolver(Protocol):
    def default_model(self, provider: str) -> str:
        """The default model of ``provider`` on this server."""
        ...

    def resolve(self, provider: str, model: str) -> reviews.ProviderSelection:
        """Validate a provider/model choice; raises ``InvalidOperationError``."""
        ...


@dataclass(frozen=True)
class BudgetCaps:
    """Upper bounds a transport applies on top of the workspace defaults (never raises
    them)."""

    tokens: int
    microusd: int


async def _defaults(
    ctx: ServiceContext, session: AsyncSession, workspace_id: uuid.UUID, resolver: ProviderResolver
) -> EffectiveReviewSettings:
    settings = ctx.settings
    return await effective_settings(
        session, workspace_id, settings, resolver.default_model(settings.llm_provider)
    )


async def start_review_for_user(
    ctx: ServiceContext,
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    workspace_id: uuid.UUID,
    payload: StartReview,
    resolver: ProviderResolver,
    caps: BudgetCaps | None = None,
) -> ReviewRun:
    """Validate, record and queue a review run for ``user_id``."""
    settings = ctx.settings
    await require_role(session, workspace_id, user_id, reviews.WRITE_ROLES, "start reviews")
    defaults = await _defaults(ctx, session, workspace_id, resolver)
    provider = payload.llm_provider or defaults.llm_provider
    model = payload.llm_model or (defaults.llm_model if provider == defaults.llm_provider else "")
    selection = resolver.resolve(provider, model)
    await ctx.review_start_limiter.reserve(str(user_id))
    budget_tokens = payload.budget_tokens or defaults.budget_tokens
    budget_microusd = (
        usd_to_micro_ceil(payload.budget_usd)
        if payload.budget_usd is not None
        else defaults.budget_microusd
    )
    if caps is not None:
        budget_tokens = min(budget_tokens, caps.tokens)
        budget_microusd = min(budget_microusd, caps.microusd)
    run = await reviews.start_review(
        session,
        workspace_id=workspace_id,
        user_id=user_id,
        rule_pack_id=payload.rule_pack_id,
        document_ids=payload.document_ids,
        provider=selection,
        options=reviews.RunOptions(
            budget_tokens=budget_tokens,
            budget_microusd=budget_microusd,
            tau=payload.tau if payload.tau is not None else defaults.tau,
            pause_on_needs_review=(
                payload.pause_on_needs_review
                if payload.pause_on_needs_review is not None
                else defaults.pause_on_needs_review
            ),
            max_concurrency=payload.max_concurrency or defaults.max_concurrency,
            rule_check_backend=settings.review_rule_check_backend,
            max_documents=settings.review_max_documents,
        ),
    )
    await ctx.enqueue_review(run.id, run.job_generation)
    return run


async def start_rule_check_for_user(
    ctx: ServiceContext,
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    hypothesis: str,
    resolver: ProviderResolver,
    caps: BudgetCaps,
    origin: str,
) -> ReviewRun:
    """Queue an ad-hoc run checking one hypothesis against one document.

    It is an ordinary review run (ledgered budget, citation verifier, audit events), so
    an escalated LLM call is never made outside the budget ledger. The workspace's
    provider and tau apply; the budget is capped; it never pauses for human review.
    """
    settings = ctx.settings
    await require_role(session, workspace_id, user_id, reviews.WRITE_ROLES, "run rule checks")
    defaults = await _defaults(ctx, session, workspace_id, resolver)
    selection = resolver.resolve(defaults.llm_provider, defaults.llm_model)
    await ctx.review_start_limiter.reserve(str(user_id))
    run = await reviews.start_rule_check(
        session,
        workspace_id=workspace_id,
        user_id=user_id,
        document_id=document_id,
        hypothesis=hypothesis,
        provider=selection,
        options=reviews.RunOptions(
            budget_tokens=min(defaults.budget_tokens, caps.tokens),
            budget_microusd=min(defaults.budget_microusd, caps.microusd),
            tau=defaults.tau,
            pause_on_needs_review=False,
            max_concurrency=1,
            rule_check_backend=settings.review_rule_check_backend,
            max_documents=1,
        ),
        origin=origin,
    )
    await ctx.enqueue_review(run.id, run.job_generation)
    return run
