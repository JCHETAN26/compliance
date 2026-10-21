"""Workspace CRUD and membership endpoints, scoped to the caller's membership."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Response, status

from compliance_api.deps import CurrentPrincipal, Session
from compliance_api.errors import error_responses
from compliance_core.schemas import (
    MemberAdd,
    MemberRead,
    MemberUpdate,
    WorkspaceCreate,
    WorkspaceRead,
    WorkspaceUpdate,
)
from compliance_core.services import workspaces as service

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.get("", responses=error_responses(401))
async def list_workspaces(principal: CurrentPrincipal, session: Session) -> list[WorkspaceRead]:
    return await service.list_workspaces(session, principal.user.id)


@router.post("", status_code=status.HTTP_201_CREATED, responses=error_responses(401, 403, 422))
async def create_workspace(
    payload: WorkspaceCreate, principal: CurrentPrincipal, session: Session
) -> WorkspaceRead:
    return await service.create_workspace(session, principal.user.id, payload)


@router.get("/{workspace_id}", responses=error_responses(401, 404, 422))
async def get_workspace(
    workspace_id: uuid.UUID, principal: CurrentPrincipal, session: Session
) -> WorkspaceRead:
    return await service.get_workspace(session, workspace_id, principal.user.id)


@router.patch("/{workspace_id}", responses=error_responses(401, 403, 404, 422))
async def update_workspace(
    workspace_id: uuid.UUID,
    payload: WorkspaceUpdate,
    principal: CurrentPrincipal,
    session: Session,
) -> WorkspaceRead:
    return await service.update_workspace(session, workspace_id, principal.user.id, payload)


@router.delete(
    "/{workspace_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=error_responses(401, 403, 404, 422),
)
async def delete_workspace(
    workspace_id: uuid.UUID, principal: CurrentPrincipal, session: Session
) -> Response:
    await service.delete_workspace(session, workspace_id, principal.user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{workspace_id}/members", responses=error_responses(401, 404, 422))
async def list_members(
    workspace_id: uuid.UUID, principal: CurrentPrincipal, session: Session
) -> list[MemberRead]:
    return await service.list_members(session, workspace_id, principal.user.id)


@router.post(
    "/{workspace_id}/members",
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(401, 403, 404, 409, 422),
)
async def add_member(
    workspace_id: uuid.UUID, payload: MemberAdd, principal: CurrentPrincipal, session: Session
) -> MemberRead:
    return await service.add_member(session, workspace_id, principal.user.id, payload)


@router.patch(
    "/{workspace_id}/members/{user_id}",
    responses=error_responses(401, 403, 404, 409, 422),
)
async def update_member(
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    payload: MemberUpdate,
    principal: CurrentPrincipal,
    session: Session,
) -> MemberRead:
    return await service.update_member(session, workspace_id, principal.user.id, user_id, payload)


@router.delete(
    "/{workspace_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=error_responses(401, 403, 404, 409, 422),
)
async def remove_member(
    workspace_id: uuid.UUID, user_id: uuid.UUID, principal: CurrentPrincipal, session: Session
) -> Response:
    await service.remove_member(session, workspace_id, principal.user.id, user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
