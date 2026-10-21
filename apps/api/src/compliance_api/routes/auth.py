"""Authentication endpoints. Sessions are JWTs in an httpOnly cookie."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

from compliance_api.context import AppContext
from compliance_api.deps import (
    ACCESS_COOKIE,
    CSRF_COOKIE,
    Ctx,
    CurrentPrincipal,
    Session,
    decode_session_cookie,
)
from compliance_api.errors import error_responses
from compliance_core.schemas import LoginRequest, RegisterRequest, UserRead
from compliance_core.security.tokens import AccessToken
from compliance_core.services import auth as auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


class SessionRead(BaseModel):
    user: UserRead
    csrf_token: str
    expires_at: datetime


def _set_session_cookies(response: Response, ctx: AppContext, token: AccessToken) -> str:
    settings = ctx.settings
    csrf = ctx.signer.csrf_token(token.jti)
    # The CSRF cookie is readable by the web app so it can echo it in X-CSRF-Token.
    for name, value, httponly in ((ACCESS_COOKIE, token.token, True), (CSRF_COOKIE, csrf, False)):
        response.set_cookie(
            name,
            value,
            max_age=settings.jwt_access_ttl_minutes * 60,
            path="/",
            domain=settings.cookie_domain,
            secure=settings.cookie_secure,
            httponly=httponly,
            samesite="lax",
        )
    response.headers["Cache-Control"] = "no-store"
    return csrf


def _clear_session_cookies(response: Response, ctx: AppContext) -> None:
    for name in (ACCESS_COOKIE, CSRF_COOKIE):
        response.delete_cookie(
            name,
            path="/",
            domain=ctx.settings.cookie_domain,
            secure=ctx.settings.cookie_secure,
            httponly=name == ACCESS_COOKIE,
            samesite="lax",
        )


async def _start_session(
    request: Request, response: Response, ctx: AppContext, user: UserRead
) -> SessionRead:
    # Never let a previous session survive re-authentication in the same browser.
    previous = await decode_session_cookie(request, ctx)
    if previous is not None:
        await ctx.revocations.revoke(previous.jti, previous.expires_at)
    token = ctx.signer.issue(user.id)
    csrf = _set_session_cookies(response, ctx, token)
    return SessionRead(user=user, csrf_token=csrf, expires_at=token.expires_at)


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(409, 422),
)
async def register(
    payload: RegisterRequest, request: Request, response: Response, ctx: Ctx, session: Session
) -> SessionRead:
    user = await auth_service.register_user(session, payload)
    return await _start_session(request, response, ctx, user)


@router.post("/login", responses=error_responses(401, 422, 429))
async def login(
    payload: LoginRequest, request: Request, response: Response, ctx: Ctx, session: Session
) -> SessionRead:
    user = await auth_service.authenticate(session, ctx.throttle, payload)
    return await _start_session(request, response, ctx, user)


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=error_responses(401, 403),
)
async def logout(principal: CurrentPrincipal, ctx: Ctx) -> Response:
    await ctx.revocations.revoke(principal.token.jti, principal.token.expires_at)
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    _clear_session_cookies(response, ctx)
    return response


@router.get("/me", responses=error_responses(401))
async def me(principal: CurrentPrincipal, response: Response) -> UserRead:
    response.headers["Cache-Control"] = "no-store"
    return principal.user
