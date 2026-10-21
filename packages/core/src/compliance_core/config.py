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

    # Uploaded originals (local storage backend). Shared by the API (writes) and worker.
    storage_dir: Path = Path("./.data/storage")

    # Upload limits, enforced while the multipart body is streamed.
    max_upload_file_bytes: Annotated[int, Field(ge=1024)] = 50 * 1024 * 1024
    max_upload_files: Annotated[int, Field(ge=1, le=100)] = 20
    max_upload_request_bytes: Annotated[int, Field(ge=1024)] = 200 * 1024 * 1024
    # Per user per minute; attempts count (rejected files and requests included).
    upload_rate_limit_per_minute: Annotated[int, Field(ge=1)] = 60
    upload_requests_per_minute: Annotated[int, Field(ge=1)] = 30
    # Zip containers (DOCX/XLSX): decompression-bomb guards.
    max_zip_entries: Annotated[int, Field(ge=10)] = 5000
    max_zip_uncompressed_bytes: Annotated[int, Field(ge=1024)] = 256 * 1024 * 1024
    max_zip_compression_ratio: Annotated[int, Field(ge=10)] = 200

    # Parsing limits and OCR.
    max_pdf_pages: Annotated[int, Field(ge=1)] = 1000
    max_sheet_rows: Annotated[int, Field(ge=1)] = 50_000
    ocr_language: str = "eng"
    ocr_dpi: Annotated[int, Field(ge=72, le=600)] = 300
    ocr_min_text_chars: Annotated[int, Field(ge=0)] = 25
    ocr_timeout_seconds: Annotated[int, Field(ge=5)] = 180
    ingest_parse_timeout_seconds: Annotated[int, Field(ge=10)] = 900
    ingest_parse_memory_mb: Annotated[int, Field(ge=256)] = 3072
    ingest_max_attempts: Annotated[int, Field(ge=1, le=10)] = 3
    worker_max_jobs: Annotated[int, Field(ge=1, le=32)] = 2

    # Chunking (token counts use the embedding model's tokenizer).
    chunk_min_tokens: Annotated[int, Field(ge=16)] = 200
    chunk_max_tokens: Annotated[int, Field(ge=32, le=500)] = 400
    chunk_overlap_tokens: Annotated[int, Field(ge=0)] = 40

    # Local embedding model and dataset cache (git-ignored; see `make fetch-artifacts`).
    artifact_cache_dir: Path = Path("./.cache")
    artifact_auto_download: bool = True
    embedding_model_id: str = "BAAI/bge-small-en-v1.5"
    embedding_model_revision: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    embedding_model_sha256: str = "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad"
    embedding_batch_size: Annotated[int, Field(ge=1, le=256)] = 32
    embedding_threads: Annotated[int, Field(ge=1, le=64)] = 4

    # Demo data. Seeding is refused in production regardless of this flag.
    seed_demo: bool = False

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
        if self.chunk_min_tokens >= self.chunk_max_tokens:
            raise ValueError("CHUNK_MIN_TOKENS must be smaller than CHUNK_MAX_TOKENS")
        if self.chunk_overlap_tokens >= self.chunk_min_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_MIN_TOKENS")
        if self.max_upload_request_bytes < self.max_upload_file_bytes:
            raise ValueError("MAX_UPLOAD_REQUEST_BYTES must be >= MAX_UPLOAD_FILE_BYTES")
        return self

    @property
    def model_cache_dir(self) -> Path:
        return self.artifact_cache_dir / "models"

    @property
    def dataset_cache_dir(self) -> Path:
        return self.artifact_cache_dir / "datasets"

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
