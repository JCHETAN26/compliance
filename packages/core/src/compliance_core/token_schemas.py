"""API token schemas (Settings page and MCP authentication)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from compliance_core.db.models import API_TOKEN_NAME_MAX

ApiTokenScope = Literal["read", "review", "upload"]
ApiTokenStatus = Literal["active", "expired", "revoked"]
DEFAULT_TOKEN_DAYS = 90
MAX_TOKEN_DAYS = 365
MAX_TOKEN_WORKSPACES = 50


def _read_only() -> list[ApiTokenScope]:
    return ["read"]


class ApiTokenCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=API_TOKEN_NAME_MAX)
    ]
    scopes: Annotated[list[ApiTokenScope], Field(max_length=3)] = Field(
        default_factory=_read_only,
        description=(
            "`read` is always granted. `review` lets the token start review runs and rule "
            "checks (spends the workspace budget); `upload` lets it upload documents (stdio "
            "MCP only). The token never exceeds the user's role in a workspace."
        ),
    )
    workspace_ids: (
        Annotated[list[uuid.UUID], Field(min_length=1, max_length=MAX_TOKEN_WORKSPACES)] | None
    ) = Field(
        default=None,
        description="Restrict the token to these workspaces; omitted = all of the user's",
    )
    expires_in_days: Annotated[int, Field(ge=1, le=MAX_TOKEN_DAYS)] = DEFAULT_TOKEN_DAYS


class ApiTokenRead(BaseModel):
    id: uuid.UUID
    name: str
    prefix: str = Field(description="The token's first characters, for recognising it")
    scopes: list[ApiTokenScope]
    workspace_ids: list[uuid.UUID] | None
    created_at: datetime
    expires_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None
    status: ApiTokenStatus


class ApiTokenCreated(ApiTokenRead):
    token: str = Field(description="The secret token. It is shown only in this response.")
