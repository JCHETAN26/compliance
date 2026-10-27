"""LLM provider catalog (global, read-only, from env), per-workspace review defaults and
the caller's MCP API tokens.

Credentials are never accepted, stored or returned; the catalog only says whether the
key *variable* of the deployment's provider is set.

API tokens are managed with the browser session (cookie + CSRF) only: the REST API never
accepts an API token itself. A new token's secret appears once, in the creation response.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Response, status

from compliance_api.deps import Ctx, CurrentPrincipal, Session
from compliance_api.errors import error_responses
from compliance_core.config import Settings
from compliance_core.db.models import WorkspaceRole
from compliance_core.review_schemas import (
    LlmSettingsRead,
    ProviderOption,
    ReviewSettingsRead,
    ReviewSettingsUpdate,
)
from compliance_core.services import api_tokens
from compliance_core.services import review_settings as service
from compliance_core.services.errors import InvalidOperationError
from compliance_core.services.session_store import FixedWindowLimiter
from compliance_core.services.workspaces import require_role
from compliance_core.token_schemas import ApiTokenCreate, ApiTokenCreated, ApiTokenRead
from compliance_llm import (
    LLMConfigurationError,
    available_providers,
    credential_present,
    default_model,
    resolve_choice,
)

router = APIRouter(tags=["settings"])
TOKEN_CREATIONS_PER_MINUTE = 10


def provider_catalog(settings: Settings) -> list[ProviderOption]:
    options = []
    for name in available_providers(settings):
        model = default_model(settings, name)
        try:
            choice = resolve_choice(settings, name, model)
            price_source: str | None = choice.price.source
            problem = None
        except LLMConfigurationError as exc:
            price_source, problem = None, str(exc)
        has_key = credential_present(name)
        if not has_key and problem is None:
            problem = "The provider's API key variable is not set on the server."
        options.append(
            ProviderOption(
                provider=name,  # type: ignore[arg-type]
                default_model=model,
                is_mock=name == "mock",
                credential_configured=has_key,
                price_source=price_source,
                usable=problem is None,
                problem=problem,
            )
        )
    return options


@router.get("/settings/llm", responses=error_responses(401))
async def llm_settings(principal: CurrentPrincipal, ctx: Ctx) -> LlmSettingsRead:
    del principal
    settings = ctx.settings
    options = provider_catalog(settings)
    return LlmSettingsRead(
        deployment_provider=settings.llm_provider,
        deployment_model=default_model(settings, settings.llm_provider),
        providers=options,
        rule_check_backend=settings.review_rule_check_backend,
        mock_only=not any(o.usable and not o.is_mock for o in options),
    )


@router.get("/workspaces/{workspace_id}/review-settings", responses=error_responses(401, 404, 422))
async def get_review_settings(
    workspace_id: uuid.UUID, principal: CurrentPrincipal, session: Session, ctx: Ctx
) -> ReviewSettingsRead:
    settings = ctx.settings
    return await service.read_settings(
        session,
        workspace_id,
        principal.user.id,
        settings,
        default_model(settings, settings.llm_provider),
    )


@router.put(
    "/workspaces/{workspace_id}/review-settings",
    responses=error_responses(401, 403, 404, 409, 422),
)
async def put_review_settings(
    workspace_id: uuid.UUID,
    payload: ReviewSettingsUpdate,
    principal: CurrentPrincipal,
    session: Session,
    ctx: Ctx,
) -> ReviewSettingsRead:
    """Owner only. The provider must be configured on this server (mock always is)."""
    settings = ctx.settings
    # Authorise before validating, so non-owners learn nothing about server configuration.
    await require_role(
        session, workspace_id, principal.user.id, {WorkspaceRole.OWNER}, "change review settings"
    )
    try:
        choice = resolve_choice(settings, payload.llm_provider, payload.llm_model)
    except LLMConfigurationError as exc:
        raise InvalidOperationError(str(exc)) from exc
    await service.write_settings(session, workspace_id, principal.user.id, payload, choice.model)
    return await service.read_settings(
        session,
        workspace_id,
        principal.user.id,
        settings,
        default_model(settings, settings.llm_provider),
    )


@router.get("/settings/api-tokens", responses=error_responses(401))
async def list_api_tokens(
    principal: CurrentPrincipal, session: Session, response: Response
) -> list[ApiTokenRead]:
    """The caller's MCP API tokens (newest first; secrets are never returned)."""
    response.headers["Cache-Control"] = "no-store"
    return await api_tokens.list_tokens(session, principal.user.id)


@router.post(
    "/settings/api-tokens",
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(401, 403, 404, 409, 422, 429),
)
async def create_api_token(
    payload: ApiTokenCreate,
    principal: CurrentPrincipal,
    session: Session,
    ctx: Ctx,
    response: Response,
) -> ApiTokenCreated:
    """Create a token for the MCP server. ``token`` is shown only in this response.
    Defaults: ``read`` scope, 90-day expiry, every workspace the caller belongs to."""
    response.headers["Cache-Control"] = "no-store"
    limiter = FixedWindowLimiter(
        ctx.redis, "api-token-create", TOKEN_CREATIONS_PER_MINUTE, "token creations"
    )
    await limiter.reserve(str(principal.user.id))
    return await api_tokens.create_token(session, principal.user.id, payload)


@router.delete("/settings/api-tokens/{token_id}", responses=error_responses(401, 403, 404, 422))
async def revoke_api_token(
    token_id: uuid.UUID, principal: CurrentPrincipal, session: Session, response: Response
) -> ApiTokenRead:
    """Revoke one of the caller's tokens; MCP calls with it fail from the next call on."""
    response.headers["Cache-Control"] = "no-store"
    return await api_tokens.revoke_token(session, principal.user.id, token_id)
