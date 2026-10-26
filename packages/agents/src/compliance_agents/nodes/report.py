"""Human-review gate and report-writer agent."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from compliance_agents.context import RunContext
from compliance_core.db.models import Finding, ReviewReport, ReviewRun
from compliance_core.reporting import WRITER_VERSION, build_report, render_markdown
from compliance_core.services.review_events import LeaseLostError
from compliance_core.services.reviews import budget_read, present_findings

GATE = "human_review"
WRITER = "report_writer"


async def human_gate(ctx: RunContext, state: dict[str, Any]) -> dict[str, Any]:
    """Pause the run (LangGraph ``interrupt``) while findings need human review, if the
    run asked for it. Resuming continues to the report; humans record their decisions as
    audited overrides in the meantime. No side effects happen before ``interrupt`` because
    the node re-executes from the top on resume."""
    async with ctx.sessions() as session:
        pending = int(
            await session.scalar(
                select(func.count())
                .select_from(Finding)
                .where(Finding.run_id == ctx.run_id, Finding.status == "needs_review")
            )
            or 0
        )
    if ctx.config.get("pause_on_needs_review") and pending:
        decision = interrupt({"needs_review": pending})
        await ctx.store.emit(GATE, "result", {"resumed": True, "needs_review": pending})
        return {"human_review": {"paused": True, "needs_review": pending, "resume": decision}}
    return {"human_review": {"paused": False, "needs_review": pending}}


async def write_report(ctx: RunContext, state: dict[str, Any]) -> dict[str, Any]:
    await ctx.store.emit(WRITER, "agent_start", {})
    async with ctx.sessions() as session:
        run = await session.get(ReviewRun, ctx.run_id)
        assert run is not None  # noqa: S101
        findings = await present_findings(session, run)
        decided = {"model": 0, "llm": 0, "system": 0, "human": 0}
        for f in findings:
            decided[f.decided_by] = decided.get(f.decided_by, 0) + 1
        totals = {
            "decided_by": decided,
            "escalated": sum(f.escalated for f in findings),
            "model_decided_unescalated": sum(
                f.machine_decided_by == "model" and not f.escalated for f in findings
            ),
            "verified": sum(f.verified for f in findings),
            "needs_review": sum(f.status == "needs_review" for f in findings),
            "budget_exhausted": bool(state.get("budget", {}).get("exhausted")),
            "errors": len(state.get("errors", [])),
        }
        # The report reads the totals from the (uncommitted) run object; they are
        # persisted below under the lease.
        run.totals_json = {**run.totals_json, **totals}
        # This is the graph's last step: the snapshot describes the finished run (the
        # runner marks it completed right after). The API also renders reports live.
        run.status = "completed"
        report = build_report(run, findings, await budget_read(session, run))
        markdown = render_markdown(report)
    await ctx.store.merge_totals(totals)
    async with ctx.sessions() as session, session.begin():
        locked = await session.scalar(
            select(ReviewRun).where(ReviewRun.id == ctx.run_id).with_for_update()
        )
        if locked is None or locked.lease_token != ctx.store.lease_token:
            raise LeaseLostError("lease lost before writing the report")
        values = {
            "writer_version": WRITER_VERSION,
            "report_json": report.model_dump(mode="json"),
            "markdown": markdown,
            "generated_at": func.now(),
        }
        await session.execute(
            insert(ReviewReport)
            .values(run_id=ctx.run_id, **values)
            .on_conflict_do_update(index_elements=[ReviewReport.run_id], set_=values)
        )
    await ctx.store.emit(
        WRITER,
        "result",
        {
            "findings": len(findings),
            "statements": len(report.summary),
            "risks": len(report.risks),
            "needs_review": len(report.needs_review),
        },
    )
    return {"report": {"writer_version": WRITER_VERSION, "findings": len(findings)}}
