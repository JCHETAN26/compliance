from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from compliance_core.security.passwords import hash_password, needs_rehash, verify_password
from compliance_core.security.tokens import InvalidTokenError, TokenSigner

SECRET = "s" * 48


def make_signer(**overrides: object) -> TokenSigner:
    params: dict[str, object] = {
        "secret": SECRET,
        "issuer": "compliance-api",
        "audience": "compliance-web",
        "ttl": timedelta(minutes=30),
    }
    params.update(overrides)
    return TokenSigner(**params)  # type: ignore[arg-type]


def test_password_hash_is_argon2id_and_verifies() -> None:
    hashed = hash_password("hunter2hunter2")
    assert hashed.startswith("$argon2id$")
    assert "hunter2hunter2" not in hashed
    assert verify_password(hashed, "hunter2hunter2")
    assert not verify_password(hashed, "hunter2hunter3")
    assert not needs_rehash(hashed)


def test_password_hashes_are_salted() -> None:
    assert hash_password("same-password") != hash_password("same-password")


def test_verify_password_rejects_missing_or_corrupt_hash() -> None:
    assert not verify_password(None, "anything")
    assert not verify_password("not-a-hash", "anything")


def test_token_roundtrip() -> None:
    signer = make_signer()
    user_id = uuid.uuid4()
    issued = signer.issue(user_id)
    decoded = signer.decode(issued.token)
    assert decoded.user_id == user_id
    assert decoded.jti == issued.jti
    assert decoded.expires_at == issued.expires_at.replace(microsecond=0)


def test_each_token_has_unique_jti() -> None:
    signer = make_signer()
    user_id = uuid.uuid4()
    assert signer.issue(user_id).jti != signer.issue(user_id).jti


def test_expired_token_rejected() -> None:
    signer = make_signer()
    issued = signer.issue(uuid.uuid4(), now=datetime.now(UTC) - timedelta(hours=1))
    with pytest.raises(InvalidTokenError, match="expired"):
        signer.decode(issued.token)


def test_token_signed_with_other_secret_rejected() -> None:
    forged = make_signer(secret="o" * 48).issue(uuid.uuid4())
    with pytest.raises(InvalidTokenError):
        make_signer().decode(forged.token)


def test_wrong_audience_or_issuer_rejected() -> None:
    token = make_signer(audience="someone-else").issue(uuid.uuid4()).token
    with pytest.raises(InvalidTokenError):
        make_signer().decode(token)
    token = make_signer(issuer="someone-else").issue(uuid.uuid4()).token
    with pytest.raises(InvalidTokenError):
        make_signer().decode(token)


def test_unsigned_alg_none_token_rejected() -> None:
    now = int(datetime.now(UTC).timestamp())
    claims = {
        "sub": str(uuid.uuid4()),
        "jti": "x",
        "typ": "access",
        "iss": "compliance-api",
        "aud": "compliance-web",
        "iat": now,
        "exp": now + 600,
    }
    token = jwt.encode(claims, key="", algorithm="none")
    with pytest.raises(InvalidTokenError):
        make_signer().decode(token)


def test_tampered_payload_rejected() -> None:
    token = make_signer().issue(uuid.uuid4()).token
    header, payload, signature = token.split(".")
    other = make_signer().issue(uuid.uuid4()).token.split(".")[1]
    with pytest.raises(InvalidTokenError):
        make_signer().decode(f"{header}.{other}.{signature}")
    assert payload != other


def test_wrong_token_type_rejected() -> None:
    now = int(datetime.now(UTC).timestamp())
    claims = {
        "sub": str(uuid.uuid4()),
        "jti": "x",
        "typ": "refresh",
        "iss": "compliance-api",
        "aud": "compliance-web",
        "iat": now,
        "exp": now + 600,
    }
    with pytest.raises(InvalidTokenError, match="type"):
        make_signer().decode(jwt.encode(claims, SECRET, algorithm="HS256"))


def test_csrf_token_bound_to_session_and_secret() -> None:
    signer = make_signer()
    token = signer.csrf_token("jti-1")
    assert signer.verify_csrf("jti-1", token)
    assert not signer.verify_csrf("jti-2", token)
    assert not signer.verify_csrf("jti-1", None)
    assert not signer.verify_csrf("jti-1", "")
    assert not make_signer(secret="o" * 48).verify_csrf("jti-1", token)
