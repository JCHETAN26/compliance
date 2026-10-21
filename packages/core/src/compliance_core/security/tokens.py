"""Signed JWT access tokens and session-bound CSRF tokens."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt

ALGORITHM = "HS256"
TOKEN_TYPE = "access"  # noqa: S105 - JWT "typ" claim, not a credential


class InvalidTokenError(Exception):
    """Raised for any token that is malformed, expired, tampered with or of the wrong type."""


@dataclass(frozen=True, slots=True)
class AccessToken:
    token: str
    user_id: uuid.UUID
    jti: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class TokenSigner:
    secret: str
    issuer: str
    audience: str
    ttl: timedelta

    def issue(self, user_id: uuid.UUID, *, now: datetime | None = None) -> AccessToken:
        issued_at = now or datetime.now(UTC)
        expires_at = issued_at + self.ttl
        jti = secrets.token_urlsafe(24)
        claims = {
            "sub": str(user_id),
            "jti": jti,
            "typ": TOKEN_TYPE,
            "iss": self.issuer,
            "aud": self.audience,
            "iat": int(issued_at.timestamp()),
            "exp": int(expires_at.timestamp()),
        }
        token = jwt.encode(claims, self.secret, algorithm=ALGORITHM)
        return AccessToken(token=token, user_id=user_id, jti=jti, expires_at=expires_at)

    def decode(self, token: str) -> AccessToken:
        try:
            claims = jwt.decode(
                token,
                self.secret,
                algorithms=[ALGORITHM],
                issuer=self.issuer,
                audience=self.audience,
                options={"require": ["sub", "jti", "exp", "iat", "iss", "aud", "typ"]},
            )
        except jwt.PyJWTError as exc:
            raise InvalidTokenError(str(exc)) from exc
        if claims.get("typ") != TOKEN_TYPE:
            raise InvalidTokenError("unexpected token type")
        try:
            user_id = uuid.UUID(str(claims["sub"]))
        except ValueError as exc:
            raise InvalidTokenError("invalid subject") from exc
        return AccessToken(
            token=token,
            user_id=user_id,
            jti=str(claims["jti"]),
            expires_at=datetime.fromtimestamp(int(claims["exp"]), tz=UTC),
        )

    def csrf_token(self, jti: str) -> str:
        """Derive the CSRF token for a session; it cannot be forged without the secret."""
        return hmac.new(self.secret.encode(), f"csrf:{jti}".encode(), hashlib.sha256).hexdigest()

    def verify_csrf(self, jti: str, presented: str | None) -> bool:
        if not presented:
            return False
        return hmac.compare_digest(self.csrf_token(jti), presented)
