"""Persisted agent events with a stable per-run cursor.

``seq`` is taken from ``review_runs.event_seq`` with ``UPDATE ... RETURNING`` in the same
transaction as the insert. The run row stays locked until commit, so events of one run
commit in ``seq`` order: a reader that has seen ``seq = n`` can never later observe a
new event with ``seq < n``. Gaps are possible (a deduplicated or rolled-back insert) and
harmless for cursors.

Worker writes pass the run's ``lease_token``: if another worker has taken the lease the
update matches no row and :class:`LeaseLostError` is raised (fencing).

Payloads hold identifiers, statuses, counts, usage and short messages only. They never
hold document text, quotes or credentials; :func:`safe_payload` enforces size limits and
drops credential-like keys defensively.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_core.db.models import AgentEvent, ReviewRun
from compliance_core.services.errors import NotFoundError

MAX_STRING = 500
MAX_PAYLOAD_BYTES = 8_000
_SECRET_KEY = re.compile(
    r"api[_-]?key|authorization|password|secret|cookie|bearer|access[_-]?token"
    r"|refresh[_-]?token|session[_-]?token|csrf",
    re.IGNORECASE,
)


class LeaseLostError(Exception):
    """The run's lease is held by someone else: this worker must stop writing."""


def _clean(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "…"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in list(value.items())[:50]:
            name = str(k)
            if _SECRET_KEY.search(name):
                continue
            out[name] = _clean(v, depth + 1)
        return out
    if isinstance(value, list | tuple):
        return [_clean(v, depth + 1) for v in list(value)[:50]]
    if isinstance(value, str):
        return value if len(value) <= MAX_STRING else value[: MAX_STRING - 1] + "…"
    if value is None or isinstance(value, bool | int | float):
        return value
    return _clean(str(value), depth + 1)


def safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = _clean(payload)
    assert isinstance(cleaned, dict)  # noqa: S101
    if len(json.dumps(cleaned, default=str)) > MAX_PAYLOAD_BYTES:
        return {"truncated": True, "keys": sorted(cleaned)[:20]}
    return cleaned


async def append_event(
    session: AsyncSession,
    run_id: uuid.UUID,
    agent: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    dedupe_key: str | None = None,
    lease_token: uuid.UUID | None = None,
) -> int | None:
    """Insert one event; returns its seq, or None if ``dedupe_key`` was already used.
    The caller commits."""
    stmt = update(ReviewRun).where(ReviewRun.id == run_id)
    if lease_token is not None:
        stmt = stmt.where(ReviewRun.lease_token == lease_token)
    seq = await session.scalar(
        stmt.values(event_seq=ReviewRun.event_seq + 1).returning(ReviewRun.event_seq)
    )
    if seq is None:
        if lease_token is not None:
            raise LeaseLostError(f"lease on run {run_id} is no longer held")
        raise NotFoundError("Review run not found.")
    inserted = await session.scalar(
        insert(AgentEvent)
        .values(
            run_id=run_id,
            seq=seq,
            agent=agent[:32],
            event_type=event_type[:32],
            payload_json=safe_payload(payload or {}),
            dedupe_key=dedupe_key[:200] if dedupe_key else None,
        )
        .on_conflict_do_nothing(constraint="uq_agent_events_run_dedupe")
        .returning(AgentEvent.seq)
    )
    return int(inserted) if inserted is not None else None


async def events_after(
    session: AsyncSession, run_id: uuid.UUID, after: int, limit: int
) -> tuple[list[AgentEvent], bool]:
    rows = list(
        await session.scalars(
            select(AgentEvent)
            .where(AgentEvent.run_id == run_id, AgentEvent.seq > after)
            .order_by(AgentEvent.seq)
            .limit(limit + 1)
        )
    )
    return rows[:limit], len(rows) > limit
