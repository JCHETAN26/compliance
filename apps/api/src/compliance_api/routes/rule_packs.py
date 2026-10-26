"""Rule packs: built-in (read-only) and workspace packs (owner/reviewer editable).

Workspace packs are visible to members only; other callers get 404. Trained-model support
is validated server-side (verbatim ContractNLI hypotheses only), so a custom rule cannot
claim ``model_supported`` and bypass the LLM path.
"""

from __future__ import annotations

import uuid
from urllib.parse import quote

from fastapi import APIRouter, Response, status

from compliance_api.deps import CurrentPrincipal, Session
from compliance_api.errors import error_responses
from compliance_core.review_schemas import (
    RulePackCreate,
    RulePackImport,
    RulePackRead,
    RulePackSummary,
)
from compliance_core.rule_packs import RulePackSpec, parse_yaml
from compliance_core.services import rule_packs as service
from compliance_core.services.workspaces import require_role

router = APIRouter(prefix="/rule-packs", tags=["rule-packs"])


@router.get("", responses=error_responses(401, 404, 422))
async def list_rule_packs(
    workspace_id: uuid.UUID, principal: CurrentPrincipal, session: Session
) -> list[RulePackSummary]:
    """Built-in packs plus the given workspace's packs (members only)."""
    return await service.list_packs(session, principal.user.id, workspace_id)


@router.post(
    "", status_code=status.HTTP_201_CREATED, responses=error_responses(401, 403, 404, 409, 422)
)
async def create_rule_pack(
    payload: RulePackCreate, principal: CurrentPrincipal, session: Session
) -> RulePackRead:
    return await service.create_pack(
        session, payload.workspace_id, principal.user.id, payload.pack, "api"
    )


@router.post(
    "/import",
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(401, 403, 404, 409, 422),
)
async def import_rule_pack(
    payload: RulePackImport, principal: CurrentPrincipal, session: Session
) -> RulePackRead:
    """Import a YAML rule pack into a workspace (validated like the built-in packs)."""
    # Authorise before parsing: callers without write access learn nothing from errors.
    await require_role(
        session, payload.workspace_id, principal.user.id, service.WRITE_ROLES, "import rule packs"
    )
    spec = parse_yaml(payload.yaml)
    return await service.create_pack(
        session, payload.workspace_id, principal.user.id, spec, "import"
    )


@router.get("/{rule_pack_id}", responses=error_responses(401, 404, 422))
async def get_rule_pack(
    rule_pack_id: uuid.UUID, principal: CurrentPrincipal, session: Session
) -> RulePackRead:
    return await service.get_pack(session, rule_pack_id, principal.user.id)


@router.put("/{rule_pack_id}", responses=error_responses(401, 403, 404, 409, 422))
async def update_rule_pack(
    rule_pack_id: uuid.UUID, payload: RulePackSpec, principal: CurrentPrincipal, session: Session
) -> RulePackRead:
    """Replace a workspace pack's content (same key). Existing runs keep their snapshot."""
    return await service.update_pack(session, rule_pack_id, principal.user.id, payload)


@router.get(
    "/{rule_pack_id}/export",
    response_class=Response,
    responses={200: {"content": {"application/yaml": {}}}, **error_responses(401, 404, 422)},
)
async def export_rule_pack(
    rule_pack_id: uuid.UUID, principal: CurrentPrincipal, session: Session
) -> Response:
    filename, text = await service.export_pack(session, rule_pack_id, principal.user.id)
    return Response(
        content=text,
        media_type="application/yaml",
        headers={
            "Content-Disposition": f"attachment; filename=\"{filename}\"; filename*=UTF-8''{quote(filename)}",
            "Cache-Control": "no-store",
        },
    )
