"""Pydantic schemas shared by the service layer, REST API and MCP server."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    StringConstraints,
)

from compliance_core.db.models import EMAIL_MAX, WORKSPACE_NAME_MAX, WorkspaceRole
from compliance_core.security.passwords import PASSWORD_MAX_LENGTH, PASSWORD_MIN_LENGTH


def _normalise_email(value: str) -> str:
    return value.strip().lower()


NormalisedEmail = Annotated[EmailStr, Field(max_length=EMAIL_MAX), AfterValidator(_normalise_email)]
WorkspaceName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=WORKSPACE_NAME_MAX),
]
AssignableRole = Literal[WorkspaceRole.REVIEWER, WorkspaceRole.VIEWER]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ReadModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class RegisterRequest(_Model):
    email: NormalisedEmail
    password: Annotated[str, Field(min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH)]


class LoginRequest(_Model):
    email: NormalisedEmail
    password: Annotated[str, Field(min_length=1, max_length=PASSWORD_MAX_LENGTH)]


class UserRead(_ReadModel):
    id: uuid.UUID
    email: str
    created_at: datetime


class WorkspaceCreate(_Model):
    name: WorkspaceName


class WorkspaceUpdate(_Model):
    name: WorkspaceName


class WorkspaceRead(_ReadModel):
    id: uuid.UUID
    name: str
    owner_id: uuid.UUID
    created_at: datetime
    role: WorkspaceRole = Field(description="The requesting user's role in this workspace")
    member_count: int
    document_count: int = 0


class MemberAdd(_Model):
    email: NormalisedEmail
    role: AssignableRole = WorkspaceRole.VIEWER


class MemberUpdate(_Model):
    role: AssignableRole


class MemberRead(_ReadModel):
    user_id: uuid.UUID
    email: str
    role: WorkspaceRole
    joined_at: datetime
