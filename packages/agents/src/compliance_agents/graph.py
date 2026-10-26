"""The LangGraph review graph: seven agents with typed shared state.

```
START -> planner -> extract (Send per contract) ----> link_register
                 \\-> link_register (no contracts)        |
link_register -> retrieve (Send per per-document task) -> rule_check (Command/Send)
              \\-> verify_dispatch (no per-document tasks)
rule_check -> verify_dispatch -> verify (Send per unverified finding) -> verify_dispatch
verify_dispatch -> cross_document (once, after per-document findings are verified)
cross_document -> verify_dispatch -> verify ... -> human_review -> report_writer -> END
```

Parallel fan-out is bounded by the run's ``max_concurrency`` (graph config). Nodes that
call services retry transient failures with exponential backoff; budget exhaustion,
stop requests and lost leases are never retried.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy, Send
from sqlalchemy.exc import DBAPIError, OperationalError

from compliance_agents.context import RunContext
from compliance_agents.nodes import checks, cross, extraction, planner, report, verify
from compliance_agents.state import ReviewState

# Document text must never leave the deployment through tracing.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

NODES = (
    "planner",
    "extract",
    "link_register",
    "retrieve",
    "rule_check",
    "verify_dispatch",
    "verify",
    "cross_document",
    "human_review",
    "report_writer",
)
AGENT_OF_NODE = {
    "planner": "planner",
    "extract": "extraction",
    "link_register": "extraction",
    "retrieve": "retrieval",
    "rule_check": "rule_check",
    "verify": "citation_verifier",
    "cross_document": "cross_document",
    "report_writer": "report_writer",
}


def _transient(exc: Exception) -> bool:
    if isinstance(exc, DBAPIError) and exc.connection_invalidated:
        return True
    return isinstance(exc, OperationalError | ConnectionError | TimeoutError)


async def _noop() -> dict[str, Any]:
    return {}


RETRY = RetryPolicy(initial_interval=1.0, backoff_factor=2.0, max_attempts=3, retry_on=_transient)


def build_graph(
    ctx: RunContext, checkpointer: BaseCheckpointSaver[Any] | None
) -> CompiledStateGraph[ReviewState, None, ReviewState, ReviewState]:
    def bind(fn: Callable[[RunContext, Any], Awaitable[Any]]) -> Any:
        """Bind a node to this run's context. Nodes reached through ``Send`` receive the
        payload rather than the whole state, so the input is typed loosely (``Any``)."""

        async def node(state: Any) -> Any:
            return await fn(ctx, state)

        node.__name__ = getattr(fn, "__name__", "node")
        return node

    def route_extraction(state: ReviewState) -> list[Send] | str:
        contracts = [
            d for d in state.get("documents", []) if d["file_format"] not in ("xlsx", "csv")
        ]
        if not contracts:
            return "link_register"
        return [Send("extract", {"document_id": d["id"]}) for d in contracts]

    def route_checks(state: ReviewState) -> list[Send] | str:
        tasks = [t for t in state.get("tasks", []) if t["kind"] == "per_document"]
        if not tasks:
            return "verify_dispatch"
        model_info = state.get("model_info", {})
        return [Send("retrieve", {"task": dict(t), "model_info": model_info}) for t in tasks]

    def route_verify(state: ReviewState) -> list[Send] | str:
        verified = state.get("verified", {})
        pending = [
            f["finding_id"]
            for f in state.get("findings", {}).values()
            if f["finding_id"] not in verified
        ]
        if pending:
            tau = state.get("model_info", {}).get("tau")
            return [Send("verify", {"finding_id": fid, "tau": tau}) for fid in pending]
        if state.get("phase") != "cross_done":
            return "cross_document"
        return "human_review"

    graph = StateGraph(ReviewState)
    graph.add_node("planner", bind(planner.plan), retry_policy=RETRY)
    graph.add_node("extract", bind(extraction.extract), retry_policy=RETRY)
    graph.add_node("link_register", bind(extraction.link_register), retry_policy=RETRY)
    graph.add_node(
        "retrieve", bind(checks.retrieve), retry_policy=RETRY, destinations=("rule_check",)
    )
    graph.add_node("rule_check", bind(checks.rule_check), retry_policy=RETRY)
    graph.add_node("verify_dispatch", bind(lambda _ctx, _state: _noop()))
    graph.add_node("verify", bind(verify.verify), retry_policy=RETRY)
    graph.add_node("cross_document", bind(cross.reason), retry_policy=RETRY)
    graph.add_node("human_review", bind(report.human_gate))
    graph.add_node("report_writer", bind(report.write_report), retry_policy=RETRY)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_extraction, ["extract", "link_register"])
    graph.add_edge("extract", "link_register")
    graph.add_conditional_edges("link_register", route_checks, ["retrieve", "verify_dispatch"])
    graph.add_edge("rule_check", "verify_dispatch")
    graph.add_conditional_edges(
        "verify_dispatch", route_verify, ["verify", "cross_document", "human_review"]
    )
    graph.add_edge("verify", "verify_dispatch")
    graph.add_edge("cross_document", "verify_dispatch")
    graph.add_edge("human_review", "report_writer")
    graph.add_edge("report_writer", END)
    return graph.compile(checkpointer=checkpointer)
