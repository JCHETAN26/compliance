from __future__ import annotations

import stat
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from compliance_core.config import AppEnv, Settings


@pytest.mark.usefixtures("clean_env")
def test_development_generates_and_persists_secret(tmp_path: Path) -> None:
    settings = Settings(app_env=AppEnv.DEVELOPMENT, data_dir=tmp_path)
    first = settings.resolved_jwt_secret()
    second = Settings(app_env=AppEnv.DEVELOPMENT, data_dir=tmp_path).resolved_jwt_secret()
    assert first == second
    assert len(first) >= 32
    secret_file = tmp_path / "secrets" / "jwt_secret"
    assert stat.S_IMODE(secret_file.stat().st_mode) == 0o600


@pytest.mark.usefixtures("clean_env")
def test_explicit_secret_is_used(tmp_path: Path) -> None:
    settings = Settings(jwt_secret=SecretStr("k" * 40), data_dir=tmp_path)
    assert settings.resolved_jwt_secret() == "k" * 40
    assert not (tmp_path / "secrets").exists()


@pytest.mark.usefixtures("clean_env")
def test_short_secret_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 32"):
        Settings(jwt_secret=SecretStr("short"))


@pytest.mark.usefixtures("clean_env")
def test_production_requires_secret_and_secure_cookies() -> None:
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        Settings(app_env=AppEnv.PRODUCTION, cookie_secure=True)
    with pytest.raises(ValidationError, match="COOKIE_SECURE"):
        Settings(app_env=AppEnv.PRODUCTION, jwt_secret=SecretStr("k" * 40), cookie_secure=False)
    ok = Settings(app_env=AppEnv.PRODUCTION, jwt_secret=SecretStr("k" * 40), cookie_secure=True)
    assert ok.resolved_jwt_secret() == "k" * 40


@pytest.mark.usefixtures("clean_env")
def test_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/x")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("COOKIE_DOMAIN", "")
    settings = Settings()
    assert settings.database_url == "postgresql+asyncpg://u:p@db:5432/x"
    assert settings.log_level == "DEBUG"
    assert settings.cookie_domain is None


@pytest.mark.usefixtures("clean_env")
def test_invalid_log_level_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(log_level="loud")
