"""Status of the trained rule-check model, for the UI's model banner (any signed-in user)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Response
from pydantic import BaseModel

from compliance_api.deps import Ctx, CurrentPrincipal
from compliance_api.errors import error_responses
from compliance_core.model_client import ModelServiceError

router = APIRouter(tags=["models"])

MESSAGES = {
    "trained": "A trained rule-check model is available.",
    "smoke": "A test artifact is loaded, not a trained model: rule-check results are not meaningful.",
    "untrained_fallback": "No trained rule-check model is installed, so automated rule checks are unavailable.",
    "unreachable": "The model service is unreachable, so rule-check availability is unknown.",
}


class RuleCheckModelStatus(BaseModel):
    state: Literal["trained", "smoke", "untrained_fallback", "unreachable"]
    message: str
    model_version: str | None = None
    variant: Literal["int8", "fp32"] | None = None


class ModelStatus(BaseModel):
    rule_check: RuleCheckModelStatus


@router.get("/models/status", responses=error_responses(401))
async def model_status(principal: CurrentPrincipal, ctx: Ctx, response: Response) -> ModelStatus:
    """Whether a trained rule-check model is available. Never exposes server paths or
    internal error details; an unreachable or invalid model service is reported as such."""
    del principal
    response.headers["Cache-Control"] = "no-store"
    try:
        status = await ctx.models.rule_check_status()
    except ModelServiceError:
        return ModelStatus(
            rule_check=RuleCheckModelStatus(state="unreachable", message=MESSAGES["unreachable"])
        )
    if not status.available:
        state: Literal["trained", "smoke", "untrained_fallback"] = "untrained_fallback"
    else:
        state = "smoke" if status.kind == "smoke" else "trained"
    return ModelStatus(
        rule_check=RuleCheckModelStatus(
            state=state,
            message=MESSAGES[state],
            model_version=status.model_version if state != "untrained_fallback" else None,
            variant=status.variant if status.variant in ("int8", "fp32") else None,  # type: ignore[arg-type]
        )
    )
