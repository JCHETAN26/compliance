"""Workspace CRUD and membership management with role-based authorisation.

Authorisation rules:
- Any member can view a workspace and its member list.
- Only the owner can rename or delete the workspace and add, change or remove members.
- A non-owner member can remove themselves (leave). The owner cannot leave.
- Non-members get ``NotFoundError`` so workspace existence is not disclosed.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_core.db.models import User, Workspace, WorkspaceMember, WorkspaceRole
from compliance_core.schemas import (
    MemberAdd,
    MemberRead,
    MemberUpdate,
    WorkspaceCreate,
    WorkspaceRead,
    WorkspaceUpdate,
)
from compliance_core.services.errors import (
    ConflictError,
    InvalidOperationError,
    NotFoundError,
    PermissionDeniedError,
)

WORKSPACE_NOT_FOUND = "Workspace not found."

_member_count = (
    select(func.count())
    .select_from(WorkspaceMember)
    .where(WorkspaceMember.workspace_id == Workspace.id)
    .correlate(Workspace)
    .scalar_subquery()
)


def _to_read(workspace: Workspace, role: WorkspaceRole, member_count: int) -> WorkspaceRead:
    return WorkspaceRead(
        id=workspace.id,
        name=workspace.name,
        owner_id=workspace.owner_id,
        created_at=workspace.created_at,
        role=role,
        member_count=member_count,
    )


async def _membership(
    session: AsyncSession, workspace_id: uuid.UUID, user_id: uuid.UUID
) -> tuple[Workspace, WorkspaceRole, int]:
    row = (
        await session.execute(
            select(Workspace, WorkspaceMember.role, _member_count)
            .join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id)
            .where(Workspace.id == workspace_id, WorkspaceMember.user_id == user_id)
        )
    ).one_or_none()
    if row is None:
        raise NotFoundError(WORKSPACE_NOT_FOUND)
    return row[0], row[1], int(row[2])


def _require(role: WorkspaceRole, allowed: Collection[WorkspaceRole], action: str) -> None:
    if role not in allowed:
        raise PermissionDeniedError(f"Your role ({role.value}) does not allow you to {action}.")


async def require_role(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    allowed: Collection[WorkspaceRole],
    action: str,
) -> WorkspaceRole:
    """Public guard for other services (documents, reviews) in later phases."""
    _, role, _ = await _membership(session, workspace_id, user_id)
    _require(role, allowed, action)
    return role


async def list_workspaces(session: AsyncSession, user_id: uuid.UUID) -> list[WorkspaceRead]:
    rows = await session.execute(
        select(Workspace, WorkspaceMember.role, _member_count)
        .join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id)
        .where(WorkspaceMember.user_id == user_id)
        .order_by(Workspace.created_at.desc(), Workspace.id)
    )
    return [_to_read(ws, role, int(count)) for ws, role, count in rows.all()]


async def create_workspace(
    session: AsyncSession, user_id: uuid.UUID, payload: WorkspaceCreate
) -> WorkspaceRead:
    workspace = Workspace(name=payload.name, owner_id=user_id)
    session.add(workspace)
    await session.flush()
    session.add(
        WorkspaceMember(workspace_id=workspace.id, user_id=user_id, role=WorkspaceRole.OWNER)
    )
    await session.flush()
    await session.refresh(workspace)
    await session.commit()
    return _to_read(workspace, WorkspaceRole.OWNER, 1)


async def get_workspace(
    session: AsyncSession, workspace_id: uuid.UUID, user_id: uuid.UUID
) -> WorkspaceRead:
    workspace, role, count = await _membership(session, workspace_id, user_id)
    return _to_read(workspace, role, count)


async def update_workspace(
    session: AsyncSession, workspace_id: uuid.UUID, user_id: uuid.UUID, payload: WorkspaceUpdate
) -> WorkspaceRead:
    workspace, role, count = await _membership(session, workspace_id, user_id)
    _require(role, {WorkspaceRole.OWNER}, "rename this workspace")
    workspace.name = payload.name
    await session.commit()
    return _to_read(workspace, role, count)


async def delete_workspace(
    session: AsyncSession, workspace_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    workspace, role, _ = await _membership(session, workspace_id, user_id)
    _require(role, {WorkspaceRole.OWNER}, "delete this workspace")
    await session.delete(workspace)
    await session.commit()


async def list_members(
    session: AsyncSession, workspace_id: uuid.UUID, user_id: uuid.UUID
) -> list[MemberRead]:
    await _membership(session, workspace_id, user_id)
    return await _members(session, workspace_id)


async def _members(session: AsyncSession, workspace_id: uuid.UUID) -> list[MemberRead]:
    rows = await session.execute(
        select(WorkspaceMember, User.email)
        .join(User, User.id == WorkspaceMember.user_id)
        .where(WorkspaceMember.workspace_id == workspace_id)
        .order_by(WorkspaceMember.created_at, User.email)
    )
    return [
        MemberRead(user_id=m.user_id, email=email, role=m.role, joined_at=m.created_at)
        for m, email in rows.all()
    ]


async def _member_read(
    session: AsyncSession, workspace_id: uuid.UUID, member_user_id: uuid.UUID
) -> MemberRead:
    for member in await _members(session, workspace_id):
        if member.user_id == member_user_id:
            return member
    raise NotFoundError("Member not found.")  # pragma: no cover - caller guarantees existence


async def add_member(
    session: AsyncSession, workspace_id: uuid.UUID, user_id: uuid.UUID, payload: MemberAdd
) -> MemberRead:
    _, role, _ = await _membership(session, workspace_id, user_id)
    _require(role, {WorkspaceRole.OWNER}, "add members")
    target = await session.scalar(select(User).where(User.email == payload.email))
    if target is None:
        raise NotFoundError("No registered user has that email address.")
    already = await session.get(WorkspaceMember, (workspace_id, target.id))
    if already is not None:
        raise ConflictError("That user is already a member of this workspace.")
    session.add(WorkspaceMember(workspace_id=workspace_id, user_id=target.id, role=payload.role))
    try:
        await session.flush()
    except IntegrityError as exc:  # concurrent add of the same member
        await session.rollback()
        raise ConflictError("That user is already a member of this workspace.") from exc
    await session.commit()
    return await _member_read(session, workspace_id, target.id)


async def update_member(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    member_user_id: uuid.UUID,
    payload: MemberUpdate,
) -> MemberRead:
    workspace, role, _ = await _membership(session, workspace_id, user_id)
    _require(role, {WorkspaceRole.OWNER}, "change member roles")
    if member_user_id == workspace.owner_id:
        raise InvalidOperationError("The workspace owner's role cannot be changed.")
    member = await session.get(WorkspaceMember, (workspace_id, member_user_id))
    if member is None:
        raise NotFoundError("Member not found.")
    member.role = payload.role
    await session.commit()
    return await _member_read(session, workspace_id, member_user_id)


async def remove_member(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    member_user_id: uuid.UUID,
) -> None:
    workspace, role, _ = await _membership(session, workspace_id, user_id)
    if member_user_id == workspace.owner_id:
        raise InvalidOperationError(
            "The workspace owner cannot be removed. Delete the workspace instead."
        )
    if member_user_id != user_id:
        _require(role, {WorkspaceRole.OWNER}, "remove other members")
    member = await session.get(WorkspaceMember, (workspace_id, member_user_id))
    if member is None:
        raise NotFoundError("Member not found.")
    await session.delete(member)
    await session.commit()
