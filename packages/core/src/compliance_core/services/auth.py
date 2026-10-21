"""User registration and credential verification."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_core.db.models import User
from compliance_core.schemas import LoginRequest, RegisterRequest, UserRead
from compliance_core.security.passwords import hash_password, needs_rehash, verify_password
from compliance_core.services.errors import AuthenticationError, ConflictError
from compliance_core.services.session_store import LoginThrottle

INVALID_CREDENTIALS = "Invalid email or password."


async def register_user(session: AsyncSession, payload: RegisterRequest) -> UserRead:
    existing = await session.scalar(select(User.id).where(User.email == payload.email))
    if existing is not None:
        raise ConflictError("An account with this email already exists.")
    user = User(email=payload.email, password_hash=hash_password(payload.password))
    session.add(user)
    try:
        await session.flush()
    except IntegrityError as exc:  # concurrent registration with the same email
        await session.rollback()
        raise ConflictError("An account with this email already exists.") from exc
    await session.refresh(user)
    await session.commit()
    return UserRead.model_validate(user)


async def authenticate(
    session: AsyncSession, throttle: LoginThrottle, payload: LoginRequest
) -> UserRead:
    await throttle.ensure_allowed(payload.email)
    user = await session.scalar(select(User).where(User.email == payload.email))
    if user is None or not verify_password(user.password_hash, payload.password):
        if user is None:
            # Burn comparable CPU time so unknown emails are not distinguishable by latency.
            verify_password(None, payload.password)
        await throttle.record_failure(payload.email)
        raise AuthenticationError(INVALID_CREDENTIALS)
    await throttle.reset(payload.email)
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(payload.password)
        await session.commit()
    return UserRead.model_validate(user)


async def get_user(session: AsyncSession, user_id: uuid.UUID) -> UserRead | None:
    user = await session.get(User, user_id)
    return UserRead.model_validate(user) if user else None
