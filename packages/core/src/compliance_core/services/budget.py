"""Run budget accounting over the ``llm_calls`` ledger (one row per provider attempt).

What an attempt is charged against the run's caps:

| status      | meaning                                         | charged                 |
|-------------|-------------------------------------------------|-------------------------|
| reserved    | written before the request; outcome pending     | the full reservation    |
| unknown     | may have been processed; no usage was reported  | the full reservation    |
| succeeded   | answered; usage measured                         | measured usage and cost |
| failed      | answered but unusable; usage measured            | measured usage and cost |
| succeeded/failed with ``usage_source = none`` | answered without trustworthy usage | the full reservation |
| released    | definitely not processed (connect error, 429, 4xx) | nothing               |

The reservation for an attempt covers its maximum output and a conservative input
estimate, so the sum over attempts never under-counts what a run may have spent.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from compliance_core.db.models import LlmCall

HELD = ("reserved", "unknown")
MEASURED = ("succeeded", "failed")


def _measured() -> ColumnElement[bool]:
    return LlmCall.status.in_(MEASURED) & (LlmCall.usage_source != "none")


def _reservation_held() -> ColumnElement[bool]:
    return LlmCall.status.in_(HELD) | (
        LlmCall.status.in_(MEASURED) & (LlmCall.usage_source == "none")
    )


def charged_tokens() -> ColumnElement[int]:
    return case(
        (_reservation_held(), LlmCall.reserved_tokens),
        (
            _measured(),
            func.coalesce(LlmCall.input_tokens, 0)
            + func.coalesce(LlmCall.output_tokens, 0)
            + func.coalesce(LlmCall.cache_write_tokens, 0)
            + func.coalesce(LlmCall.cache_read_tokens, 0),
        ),
        else_=0,
    )


def charged_microusd() -> ColumnElement[int]:
    return case(
        (_reservation_held(), LlmCall.reserved_microusd),
        (_measured(), func.coalesce(LlmCall.cost_microusd, 0)),
        else_=0,
    )


@dataclass(frozen=True)
class RunUsage:
    tokens: int
    microusd: int
    tokens_open: int
    microusd_open: int
    calls: int
    unknown: int
    sources: frozenset[str]

    @property
    def usage_source(self) -> str:
        if not self.sources:
            return "none"
        if len(self.sources) == 1:
            return next(iter(self.sources))
        return "mixed"


async def run_usage(session: AsyncSession, run_id: uuid.UUID) -> RunUsage:
    row = (
        await session.execute(
            select(
                func.coalesce(func.sum(charged_tokens()), 0),
                func.coalesce(func.sum(charged_microusd()), 0),
                func.coalesce(
                    func.sum(
                        case((LlmCall.status == "reserved", LlmCall.reserved_tokens), else_=0)
                    ),
                    0,
                ),
                func.coalesce(
                    func.sum(
                        case((LlmCall.status == "reserved", LlmCall.reserved_microusd), else_=0)
                    ),
                    0,
                ),
                func.count(LlmCall.id),
                func.count(LlmCall.id).filter(LlmCall.status == "unknown"),
            ).where(LlmCall.run_id == run_id)
        )
    ).one()
    sources = set(
        await session.scalars(
            select(LlmCall.usage_source)
            .where(LlmCall.run_id == run_id, LlmCall.usage_source != "none")
            .distinct()
        )
    )
    return RunUsage(
        tokens=int(row[0]),
        microusd=int(row[1]),
        tokens_open=int(row[2]),
        microusd_open=int(row[3]),
        calls=int(row[4]),
        unknown=int(row[5]),
        sources=frozenset(sources),
    )
