"""FastAPI dependencies: database session, authentication and CSRF enforcement."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_api.context import AppContext
from compliance_core.schemas import UserRead
from compliance_core.security.tokens import AccessToken, InvalidTokenError
from compliance_core.services import auth as auth_service

ACCESS_COOKIE = "cw_access"
CSRF_COOKIE = "cw_csrf"
CSRF_HEADER = "X-CSRF-Token"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def get_context(request: Request) -> AppContext:
    ctx: AppContext = request.app.state.ctx
    return ctx


Ctx = Annotated[AppContext, Depends(get_context)]


async def get_session(ctx: Ctx) -> AsyncIterator[AsyncSession]:
    async with ctx.session_factory() as session:
        try:
            yield session
        except BaseException:
            await session.rollback()
            raise


Session = Annotated[AsyncSession, Depends(get_session)]


@dataclass(frozen=True, slots=True)
class Principal:
    user: UserRead
    token: AccessToken


def _unauthenticated(message: str = "Authentication required.") -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=message)


async def decode_session_cookie(request: Request, ctx: AppContext) -> AccessToken | None:
    """Return the valid, non-revoked access token from the cookie, if any."""
    raw = request.cookies.get(ACCESS_COOKIE)
    if not raw:
        return None
    try:
        token = ctx.signer.decode(raw)
    except InvalidTokenError:
        return None
    if await ctx.revocations.is_revoked(token.jti):
        return None
    return token


async def current_principal(request: Request, ctx: Ctx, session: Session) -> Principal:
    token = await decode_session_cookie(request, ctx)
    if token is None:
        raise _unauthenticated()
    user = await auth_service.get_user(session, token.user_id)
    if user is None:
        raise _unauthenticated()
    # Cookie-authenticated state changes must carry the session-bound CSRF token.
    if request.method not in SAFE_METHODS and not ctx.signer.verify_csrf(
        token.jti, request.headers.get(CSRF_HEADER)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Missing or invalid CSRF token."
        )
    return Principal(user=user, token=token)


CurrentPrincipal = Annotated[Principal, Depends(current_principal)]
