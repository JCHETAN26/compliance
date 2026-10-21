"""Environment-driven configuration shared by every Python service.

All configuration comes from environment variables (see ``.env.example``). Nothing is
read from code-level constants at runtime except safe development defaults.
"""

from __future__ import annotations

import os
import secrets
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_SECRET_LENGTH = 32


class AppEnv(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore", case_sensitive=False)

    app_env: AppEnv = AppEnv.DEVELOPMENT
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://compliance:compliance@localhost:5432/compliance"
    database_pool_size: Annotated[int, Field(ge=1, le=100)] = 10
    redis_url: str = "redis://localhost:6379/0"

    # Directory for persistent service data (generated secrets, later: uploaded files).
    data_dir: Path = Path("./.data")

    # Auth. If JWT_SECRET is empty outside production, a random secret is generated once
    # and persisted to ``<data_dir>/secrets/jwt_secret`` (mode 0600).
    jwt_secret: SecretStr = SecretStr("")
    jwt_access_ttl_minutes: Annotated[int, Field(ge=5, le=7 * 24 * 60)] = 480
    jwt_issuer: str = "compliance-api"
    jwt_audience: str = "compliance-web"
    cookie_secure: bool = False
    cookie_domain: str | None = None

    # Brute-force protection: failed logins per account within a sliding window.
    login_max_failures: Annotated[int, Field(ge=1)] = 10
    login_failure_window_seconds: Annotated[int, Field(ge=10)] = 900

    @field_validator("cookie_domain", mode="before")
    @classmethod
    def _empty_domain_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        upper = value.upper()
        if upper not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid LOG_LEVEL {value!r}")
        return upper

    @model_validator(mode="after")
    def _production_guards(self) -> Settings:
        if self.app_env is AppEnv.PRODUCTION:
            if len(self.jwt_secret.get_secret_value()) < MIN_SECRET_LENGTH:
                raise ValueError(
                    f"JWT_SECRET must be set to at least {MIN_SECRET_LENGTH} characters in production"
                )
            if not self.cookie_secure:
                raise ValueError("COOKIE_SECURE must be true in production")
        secret = self.jwt_secret.get_secret_value()
        if secret and len(secret) < MIN_SECRET_LENGTH:
            raise ValueError(f"JWT_SECRET must be at least {MIN_SECRET_LENGTH} characters")
        return self

    def resolved_jwt_secret(self) -> str:
        """Return the signing secret, generating and persisting one for local development."""
        secret = self.jwt_secret.get_secret_value()
        if secret:
            return secret
        if self.app_env is AppEnv.PRODUCTION:  # pragma: no cover - guarded by validator
            raise RuntimeError("JWT_SECRET is required in production")
        return _load_or_create_secret(self.data_dir / "secrets" / "jwt_secret")


def _load_or_create_secret(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        # O_EXCL makes creation atomic when several processes start at once.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        value = path.read_text(encoding="utf-8").strip()
        if len(value) < MIN_SECRET_LENGTH:
            raise RuntimeError(f"persisted secret at {path} is too short or empty") from None
        return value
    value = secrets.token_urlsafe(48)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(value)
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
