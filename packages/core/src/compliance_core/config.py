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
from typing import Annotated, Literal

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
    # Query formatting for retrieval (passages are always embedded without a prefix).
    # Applied by the model service to every query embedding; see compliance_retrieval.
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "

    # Model service (embedding + optional rerank inference over HTTP).
    model_service_url: str = "http://localhost:8010"
    model_service_timeout_seconds: Annotated[float, Field(gt=0, le=120)] = 10.0
    model_service_retries: Annotated[int, Field(ge=0, le=5)] = 2

    # Optional cross-encoder reranker (off unless explicitly enabled).
    rerank_enabled: bool = False
    rerank_model_id: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_model_revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    rerank_model_sha256: str = "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"

    # Trained rule-check model (Phase 3), served from a verified ONNX artifact directory
    # (manifest + SHA-256-checked files; see compliance_model_service.rule_check). Empty
    # means <ARTIFACT_CACHE_DIR>/models/rule-check/current. Without a verified artifact
    # the service runs in "untrained_fallback" mode: /v1/check answers 503, never a verdict.
    rule_check_artifact_dir: Path | None = None
    rule_check_variant: Literal["int8", "fp32"] = "int8"
    # CI smoke artifacts (tiny random backbone) are refused unless this is set (tests only).
    rule_check_allow_smoke: bool = False
    rule_check_threads: Annotated[int, Field(ge=1, le=64)] = 4
    # Dynamic micro-batching: a batch runs when it has max_batch_rows rows or when
    # max_wait_ms has passed since its first row arrived.
    rule_check_max_batch_rows: Annotated[int, Field(ge=1, le=1024)] = 64
    rule_check_max_wait_ms: Annotated[float, Field(ge=0, le=1000)] = 5.0
    rule_check_max_queue_rows: Annotated[int, Field(ge=1, le=100_000)] = 4096
    rule_check_max_tokens_per_forward: Annotated[int, Field(ge=512, le=1_000_000)] = 16_384
    # Admission before tokenization: requests with tokenization queued/running, and the
    # per-request budget of hypothesis + span characters (413 above it).
    rule_check_max_pending_requests: Annotated[int, Field(ge=1, le=256)] = 8
    rule_check_max_request_chars: Annotated[int, Field(ge=1_000, le=10_000_000)] = 400_000

    # Hybrid retrieval defaults. rrf_k, the weights, the keyword mode and the query
    # instruction were selected on the ContractNLI DEV split and frozen in
    # evals/retrieval/frozen_config.json (config hash 6cb1378f...); see evals/README.md.
    search_top_k: Annotated[int, Field(ge=1, le=50)] = 8
    search_candidate_k: Annotated[int, Field(ge=1, le=500)] = 50
    search_ef_search: Annotated[int, Field(ge=1, le=1000)] = 100
    search_rrf_k: Annotated[int, Field(ge=1, le=1000)] = 5
    search_vector_weight: Annotated[float, Field(ge=0, le=10)] = 1.0
    search_keyword_weight: Annotated[float, Field(ge=0, le=10)] = 1.5
    search_keyword_mode: Literal["or", "websearch"] = "or"
    search_rerank_candidates: Annotated[int, Field(ge=1, le=100)] = 30
    search_statement_timeout_ms: Annotated[int, Field(ge=100, le=60_000)] = 5000
    search_rate_limit_per_minute: Annotated[int, Field(ge=1)] = 120

    # Demo data. Seeding is refused in production regardless of this flag.
    seed_demo: bool = False

    # LLM provider for review runs (Phase 4). ``mock`` is deterministic, offline and reads
    # no credentials. Real providers read their key from the environment only when they
    # are actually used (ANTHROPIC_API_KEY / OPENAI_API_KEY); keys are never settings
    # fields, so they cannot leak through settings dumps, the database or checkpoints.
    llm_provider: Literal["mock", "anthropic", "openai", "ollama"] = "mock"
    # Empty: the provider default (mock-deterministic-v1 / claude-haiku-5-5); OpenAI and
    # Ollama require an explicit model.
    llm_model: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_effort: Literal["low", "medium", "high"] | None = None
    openai_base_url: str = "https://api.openai.com"
    ollama_base_url: str = "http://localhost:11434"
    llm_timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 60.0
    # Attempts per LLM step (each attempt is reserved against the run budget first).
    llm_max_attempts: Annotated[int, Field(ge=1, le=5)] = 3
    llm_max_output_tokens: Annotated[int, Field(ge=64, le=16_384)] = 1024
    # USD per million tokens. Overrides the built-in price table; required for OpenAI and
    # for Anthropic models without a built-in price. Ollama defaults to 0 (local).
    llm_input_usd_per_mtok: Annotated[float | None, Field(ge=0)] = None
    llm_output_usd_per_mtok: Annotated[float | None, Field(ge=0)] = None

    # Review runs (Phase 4): defaults for new workspaces and worker behaviour.
    review_budget_tokens: Annotated[int, Field(ge=1_000, le=50_000_000)] = 400_000
    review_budget_usd: Annotated[float, Field(ge=0, le=1_000)] = 1.0
    review_max_concurrency: Annotated[int, Field(ge=1, le=16)] = 4
    review_pause_on_needs_review: bool = False
    # ``service`` = the trained model behind the model service; ``mock`` = deterministic
    # lexical stand-in for tests and offline demos (refused in production).
    review_rule_check_backend: Literal["service", "mock"] = "service"
    # The model service's CPU rule check takes ~10-13 s per document on this hardware.
    review_model_timeout_seconds: Annotated[float, Field(gt=0, le=900)] = 300.0
    review_lease_seconds: Annotated[int, Field(ge=10, le=600)] = 60
    review_job_timeout_seconds: Annotated[int, Field(ge=60, le=86_400)] = 4 * 3600
    review_start_rate_limit_per_minute: Annotated[int, Field(ge=1)] = 10
    review_max_documents: Annotated[int, Field(ge=1, le=500)] = 100

    @field_validator(
        "cookie_domain",
        "rule_check_artifact_dir",
        "anthropic_effort",
        "llm_input_usd_per_mtok",
        "llm_output_usd_per_mtok",
        mode="before",
    )
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
            if self.review_rule_check_backend == "mock":
                raise ValueError("REVIEW_RULE_CHECK_BACKEND=mock is not allowed in production")
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

    @property
    def resolved_rule_check_dir(self) -> Path:
        return self.rule_check_artifact_dir or self.model_cache_dir / "rule-check" / "current"

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
