"""MCP-only settings (``MCP_*`` environment variables).

They are read only by this server, so nothing here can loosen the REST API: budgets are
upper bounds applied *on top of* the workspace's review defaults, and the provider
capability list replaces only this process's credential check.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_MIB = 1024 * 1024


class McpSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MCP_", env_file=None, extra="ignore", case_sensitive=False
    )

    # Non-secret: comma-separated providers whose API key the *worker* holds (Compose
    # derives it from the host environment with ${ANTHROPIC_API_KEY:+anthropic}). Used only
    # to refuse review starts the worker could not run; this process never holds a key.
    credentialed_providers: str = ""

    # stdio ``upload_document`` (decoded bytes; never above MAX_UPLOAD_FILE_BYTES).
    upload_max_bytes: Annotated[int, Field(ge=1024, le=50 * _MIB)] = 10 * _MIB

    # ``check_rule``: default and maximum time to wait for the ad-hoc run's verdict before
    # returning its run id, and its budget cap (applied below the workspace default).
    check_rule_wait_seconds: Annotated[float, Field(ge=0, le=300)] = 30.0
    check_rule_max_wait_seconds: Annotated[float, Field(ge=0, le=300)] = 120.0
    check_rule_budget_tokens: Annotated[int, Field(ge=1_000, le=200_000)] = 20_000
    check_rule_budget_usd: Annotated[float, Field(ge=0, le=1.0)] = 0.05

    # ``run_review``: budget caps applied below the workspace default.
    review_budget_tokens: Annotated[int, Field(ge=1_000, le=50_000_000)] = 400_000
    review_budget_usd: Annotated[float, Field(ge=0, le=1_000)] = 1.0

    # Streamable HTTP DNS-rebinding protection.
    allowed_hosts: str = "localhost:*,127.0.0.1:*,mcp-server:*"
    allowed_origins: str = "http://localhost:*,http://127.0.0.1:*"


def split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_mcp_settings() -> McpSettings:
    return McpSettings()
