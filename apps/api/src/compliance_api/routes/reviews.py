"""Review runs: start, read, control, tasks, findings, overrides, citations, events, report.

All routes require a signed-in workspace member (cookie session; CSRF on writes). Starting,
controlling and overriding additionally need the owner or reviewer role. Ids outside the
caller's workspaces are 404. The API never runs checks itself: it records the request and
enqueues the worker job.

Events: ``GET /reviews/{id}/events`` is Server-Sent Events. Each event carries
``id: <seq>``; reconnecting with ``Last-Event-ID`` (browsers do this automatically) or
``?after=<seq>`` replays everything after that cursor. ``/events/history`` is the same
data as paginated JSON. The stream re-checks membership and session revocation every
15 s, closes after the run stops (``event: end``), and otherwise after 10 minutes so
clients reconnect with their cursor.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Header, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from compliance_api.deps import Ctx, CurrentPrincipal, Session
from compliance_api.errors import error_responses
from compliance_core.db.models import AgentEvent, ReviewRun, WorkspaceMember
from compliance_core.reporting import render_markdown
from compliance_core.review_schemas import (
    AgentEventRead,
    CitationDetail,
    EventPage,
    FindingPage,
    FindingRead,
    OverrideRequest,
    ReportRead,
    ReviewRunRead,
    ReviewTasksRead,
    RunControl,
    StartReview,
)
from compliance_core.services import reviews as service
from compliance_core.services.review_events import events_after
from compliance_core.services.review_start import start_review_for_user
from compliance_llm import ProviderResolver, credential_present

router = APIRouter(tags=["reviews"])
ACTIVE = ("queued", "running")
SSE_POLL_SECONDS = 1.0
SSE_HEARTBEAT_SECONDS = 15.0
SSE_REAUTH_SECONDS = 15.0
SSE_MAX_SECONDS = 600.0


def _event(e: AgentEvent) -> AgentEventRead:
    return AgentEventRead(
        seq=e.seq, agent=e.agent, event_type=e.event_type, payload=e.payload_json, ts=e.ts
    )


@router.post(
    "/workspaces/{workspace_id}/reviews",
    status_code=status.HTTP_202_ACCEPTED,
    responses=error_responses(401, 403, 404, 409, 422, 429),
)
async def start_review(
    workspace_id: uuid.UUID,
    payload: StartReview,
    principal: CurrentPrincipal,
    session: Session,
    ctx: Ctx,
) -> ReviewRunRead:
    """Validate and queue a review run (the worker executes it)."""
    run = await start_review_for_user(
        ctx,
        session,
        user_id=principal.user.id,
        workspace_id=workspace_id,
        payload=payload,
        resolver=ProviderResolver(ctx.settings, credential_present),
    )
    return await service.run_read(session, run)


@router.get("/workspaces/{workspace_id}/reviews", responses=error_responses(401, 404, 422))
async def list_reviews(
    workspace_id: uuid.UUID, principal: CurrentPrincipal, session: Session
) -> list[ReviewRunRead]:
    return await service.list_runs(session, workspace_id, principal.user.id)


@router.get("/reviews/{run_id}", responses=error_responses(401, 404, 422))
async def get_review(
    run_id: uuid.UUID, principal: CurrentPrincipal, session: Session, response: Response
) -> ReviewRunRead:
    response.headers["Cache-Control"] = "no-store"
    return await service.get_run(session, run_id, principal.user.id)


async def _control(
    run_id: uuid.UUID, action: str, principal: CurrentPrincipal, session: Session, ctx: Ctx
) -> ReviewRunRead:
    run, enqueue = await service.control_run(session, run_id, principal.user.id, action)
    if enqueue:
        await ctx.enqueue_review(run.id, run.job_generation)
    return await service.run_read(session, run)


@router.post("/reviews/{run_id}/control", responses=error_responses(401, 403, 404, 409, 422))
async def control_review(
    run_id: uuid.UUID,
    payload: RunControl,
    principal: CurrentPrincipal,
    session: Session,
    ctx: Ctx,
) -> ReviewRunRead:
    """Pause (at the next step boundary), resume (from the last checkpoint) or cancel."""
    return await _control(run_id, payload.action, principal, session, ctx)


@router.post("/reviews/{run_id}/resume", responses=error_responses(401, 403, 404, 409, 422))
async def resume_review(
    run_id: uuid.UUID, principal: CurrentPrincipal, session: Session, ctx: Ctx
) -> ReviewRunRead:
    """Resume a paused, failed or review-waiting run from its last checkpoint."""
    return await _control(run_id, "resume", principal, session, ctx)


@router.get("/reviews/{run_id}/tasks", responses=error_responses(401, 404, 422))
async def review_tasks(
    run_id: uuid.UUID, principal: CurrentPrincipal, session: Session, response: Response
) -> ReviewTasksRead:
    response.headers["Cache-Control"] = "no-store"
    return await service.list_tasks(session, run_id, principal.user.id)


@router.get("/reviews/{run_id}/findings", responses=error_responses(401, 404, 422))
async def review_findings(
    run_id: uuid.UUID,
    principal: CurrentPrincipal,
    session: Session,
    response: Response,
    status_filter: Annotated[
        Literal["satisfied", "violated", "not_mentioned", "needs_review"] | None,
        Query(alias="status"),
    ] = None,
    severity: Literal["low", "medium", "high"] | None = None,
    decided_by: Literal["model", "llm", "system", "human"] | None = None,
    subject_key: Annotated[str | None, Query(max_length=200)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 500,
) -> FindingPage:
    response.headers["Cache-Control"] = "no-store"
    return await service.list_findings(
        session,
        run_id,
        principal.user.id,
        status=status_filter,
        severity=severity,
        decided_by=decided_by,
        subject_key=subject_key,
        offset=offset,
        limit=limit,
    )


@router.get("/findings/{finding_id}", responses=error_responses(401, 404, 422))
async def get_finding(
    finding_id: uuid.UUID, principal: CurrentPrincipal, session: Session
) -> FindingRead:
    return await service.get_finding(session, finding_id, principal.user.id)


@router.post("/findings/{finding_id}/override", responses=error_responses(401, 403, 404, 409, 422))
async def override_finding(
    finding_id: uuid.UUID, payload: OverrideRequest, principal: CurrentPrincipal, session: Session
) -> FindingRead:
    """Record an audited human decision (with a comment). The machine finding is kept."""
    return await service.override_finding(
        session, finding_id, principal.user.id, payload.status, payload.comment
    )


@router.get("/citations/{citation_id}", responses=error_responses(401, 404, 422))
async def get_citation(
    citation_id: uuid.UUID, principal: CurrentPrincipal, session: Session
) -> CitationDetail:
    """The quote, its document/page/offsets/boxes, surrounding context, and whether the
    quote is still exactly at its offsets."""
    return await service.get_citation(session, citation_id, principal.user.id)


@router.get("/reviews/{run_id}/events/history", responses=error_responses(401, 404, 422))
async def review_event_history(
    run_id: uuid.UUID,
    principal: CurrentPrincipal,
    session: Session,
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> EventPage:
    run, _ = await service.run_for_member(session, run_id, principal.user.id)
    events, more = await events_after(session, run.id, after, limit)
    return EventPage(
        items=[_event(e) for e in events],
        next_after=events[-1].seq if events else after,
        has_more=more,
        run_status=run.status,  # type: ignore[arg-type]
    )


def _sse(event: AgentEventRead) -> str:
    data = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
    return f"id: {event.seq}\nevent: agent_event\ndata: {data}\n\n"


@router.get(
    "/reviews/{run_id}/events",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}, **error_responses(401, 404, 422)},
)
async def review_events(
    run_id: uuid.UUID,
    request: Request,
    principal: CurrentPrincipal,
    session: Session,
    ctx: Ctx,
    after: Annotated[int | None, Query(ge=0)] = None,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    run, _ = await service.run_for_member(session, run_id, principal.user.id)
    # Capture plain values first: rollback expires ORM instances (no lazy loads later).
    workspace_id = run.workspace_id
    user_id, jti = principal.user.id, principal.token.jti
    await session.rollback()  # release the connection; polls use short sessions
    cursor = after or 0
    if last_event_id and last_event_id.strip().isdigit():
        cursor = max(cursor, int(last_event_id.strip()))
    expires_at = principal.token.expires_at.timestamp()

    async def still_authorised() -> bool:
        if time.time() >= expires_at or await ctx.revocations.is_revoked(jti):
            return False
        async with ctx.session_factory() as s:
            member = await s.scalar(
                select(WorkspaceMember.role).where(
                    WorkspaceMember.workspace_id == workspace_id,
                    WorkspaceMember.user_id == user_id,
                )
            )
        return member is not None

    async def stream() -> AsyncIterator[str]:
        nonlocal cursor
        started = last_beat = last_auth = time.monotonic()
        yield "retry: 3000\n\n"
        while True:
            if await request.is_disconnected():
                return
            now = time.monotonic()
            if now - last_auth >= SSE_REAUTH_SECONDS:
                last_auth = now
                if not await still_authorised():
                    yield 'event: end\ndata: {"reason":"unauthorised"}\n\n'
                    return
            async with ctx.session_factory() as s:
                events, more = await events_after(s, run_id, cursor, 200)
                run_status = await s.scalar(select(ReviewRun.status).where(ReviewRun.id == run_id))
            if run_status is None:
                yield 'event: end\ndata: {"reason":"deleted"}\n\n'
                return
            for e in events:
                cursor = e.seq
                yield _sse(_event(e))
            if more:
                continue
            if run_status not in ACTIVE and not events:
                yield f'event: end\ndata: {{"reason":"stopped","status":"{run_status}"}}\n\n'
                return
            if now - started >= SSE_MAX_SECONDS:
                yield 'event: end\ndata: {"reason":"reconnect"}\n\n'
                return
            if now - last_beat >= SSE_HEARTBEAT_SECONDS:
                last_beat = now
                yield ": keep-alive\n\n"
            await asyncio.sleep(SSE_POLL_SECONDS)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            # no-transform: proxies (and Next's compression) must not buffer the stream.
            "Cache-Control": "no-cache, no-store, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/reviews/{run_id}/report",
    response_model=ReportRead,
    responses={
        200: {"content": {"text/markdown": {}}},
        **error_responses(401, 404, 422),
    },
)
async def review_report(
    run_id: uuid.UUID,
    principal: CurrentPrincipal,
    session: Session,
    report_format: Annotated[Literal["json", "md"], Query(alias="format")] = "json",
) -> Any:
    """The report, rendered live from verified findings and human overrides (citations are
    re-checked against the current documents). ``format=md`` downloads Markdown."""
    run, _ = await service.run_for_member(session, run_id, principal.user.id)
    report = await service.report_for_run(session, run)
    if report_format == "md":
        return Response(
            content=render_markdown(report),
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="report-{run.id}.md"',
                "Cache-Control": "no-store",
            },
        )
    return report
