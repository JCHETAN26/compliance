"""Retrieval agent and rule-check agent (per-document tasks).

Retrieval: a model-supported rule is checked by the trained model over *every* sentence
span of the document (the evaluated full-document pipeline); other rules get hybrid
retrieval (pgvector + full text, RRF) scoped to the task's document, falling back to
keyword-only search if the embedding service is unavailable (recorded as an event).

Rule check (cascade): the trained model decides when its calibrated confidence is at
least tau; below tau, or when the model is unavailable or the input does not fit, the
task escalates to the LLM with the model's top spans. Rules that are not model-supported
always use the LLM. LLM verdicts are grounded: evidence must be cited by span ids that
were presented, otherwise the output is rejected (``needs_review``). Verdicts of
satisfied/violated without evidence also need review. Nothing here marks a finding as
verified; the citation verifier does that.
"""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.types import Command, Send

from compliance_agents.context import RunContext
from compliance_agents.gateway import LLMStepError
from compliance_agents.model import ModelUnavailableError, ModelVerdict
from compliance_agents.nodes.common import (
    MAX_EVIDENCE_CITATIONS,
    PER_DOCUMENT,
    digest,
    error_record,
    rule_compliance,
    span_citation,
)
from compliance_agents.outputs import RULE_VERDICT_SCHEMA, RuleVerdictOut
from compliance_agents.prompting import load_prompt, nonce, numbered
from compliance_agents.store import BudgetExceededError, CitationDraft, FindingDraft
from compliance_agents.text import DocView, Span, load_doc
from compliance_core.model_client import ModelServiceError
from compliance_core.services.errors import ServiceError
from compliance_llm import LLMRequest
from compliance_retrieval.schemas import SearchMode, SearchRequest
from compliance_retrieval.service import run_search
from compliance_retrieval.sql import Scope

RETRIEVAL = "retrieval"
RULE_CHECK = "rule_check"
TOP_K = 6
MAX_LLM_SPANS = 40
MAX_LLM_CHARS = 12_000


def _model_path(ctx: RunContext, state_model: dict[str, Any], rule: dict[str, Any]) -> bool:
    return bool(rule.get("model_supported")) and bool(state_model.get("available"))


async def retrieve(ctx: RunContext, payload: dict[str, Any]) -> Command[str]:
    task = payload["task"]
    rule = ctx.rules[task["rule_key"]]
    model_info = payload["model_info"]
    document_id = task["document_id"]
    context = {"task_key": task["task_key"], "rule_key": rule["key"], "document_id": document_id}
    if _model_path(ctx, model_info, rule):
        candidates: dict[str, Any] = {"mode": "document_sentences"}
        await ctx.store.emit(
            RETRIEVAL, "tool_call", {**context, "tool": "retrieval.document_scope"}
        )
    else:
        chunk_ids: list[str] = []
        mode = "hybrid"
        if ctx.search is not None:
            scope = Scope(
                workspace_id=ctx.workspace_id,
                document_ids=(uuid.UUID(document_id),),
                doc_types=(),
            )
            modes: tuple[SearchMode, ...] = ("hybrid", "keyword")
            for attempt_mode in modes:
                mode = attempt_mode
                await ctx.store.emit(
                    RETRIEVAL,
                    "tool_call",
                    {**context, "tool": f"retrieval.{attempt_mode}", "top_k": TOP_K},
                )
                try:
                    response = await run_search(
                        ctx.search,
                        scope,
                        SearchRequest(
                            query=rule["hypothesis_text"][:1000],
                            mode=attempt_mode,
                            top_k=TOP_K,
                            document_ids=[uuid.UUID(document_id)],
                        ),
                    )
                except (ServiceError, ModelServiceError) as exc:
                    await ctx.store.emit(
                        RETRIEVAL,
                        "error",
                        {
                            **context,
                            "code": "retrieval_degraded",
                            "message": str(exc)[:200],
                            "fallback": "keyword" if attempt_mode == "hybrid" else "document_start",
                        },
                    )
                    continue
                chunk_ids = [str(hit.chunk_id) for hit in response.results]
                break
        candidates = {"mode": mode if chunk_ids else "document_start", "chunk_ids": chunk_ids}
    await ctx.store.emit(
        RETRIEVAL,
        "result",
        {**context, "mode": candidates["mode"], "chunks": len(candidates.get("chunk_ids", []))},
    )
    return Command(
        update={"candidates": {task["task_key"]: candidates}},
        goto=Send("rule_check", {"task": task, "candidates": candidates, "model_info": model_info}),
    )


def _llm_spans(doc: DocView, candidates: dict[str, Any], seed: list[Span]) -> list[Span]:
    """Spans shown to the LLM: the model's spans on escalation, else sentences of the
    retrieved chunks (document start if retrieval found nothing); capped."""
    spans: list[Span] = list(seed)
    if not spans:
        wanted = {uuid.UUID(c) for c in candidates.get("chunk_ids", [])}
        chunks = [c for c in doc.chunks if c.id in wanted] or doc.chunks[:TOP_K]
        for chunk in sorted(chunks, key=lambda c: c.index):
            spans.extend(doc.sentences_in(chunk.start, chunk.end))
    out: list[Span] = []
    total = 0
    seen: set[tuple[int, int]] = set()
    for span in sorted(spans, key=lambda s: s.start):
        if (span.start, span.end) in seen:
            continue
        size = span.end - span.start
        if len(out) >= MAX_LLM_SPANS or (total + size > MAX_LLM_CHARS and out):
            break
        seen.add((span.start, span.end))
        out.append(span)
        total += size
    return out


async def llm_verdict(
    ctx: RunContext,
    *,
    task_key: str,
    rule: dict[str, Any],
    doc: DocView,
    spans: list[Span],
    agent: str = RULE_CHECK,
) -> tuple[str, list[Span], str, dict[str, Any]]:
    """Ask the provider for a structured verdict over ``spans``.

    Returns ``(machine_status, evidence_spans, rationale, info)``. Ungrounded output (ids
    that were not presented) or a decisive verdict without evidence gives
    ``needs_review`` with the reason in ``info``.
    """
    prompt = load_prompt("rule_verdict")
    presented = {f"S{i}": s for i, s in enumerate(spans, start=1)}
    request = LLMRequest(
        task="rule_verdict",
        system=prompt.render_system(),
        user=prompt.render(
            hypothesis=rule["hypothesis_text"],
            spans=numbered([(sid, doc.slice(s.start, s.end)) for sid, s in presented.items()]),
            nonce=nonce(str(ctx.run_id), task_key),
        ),
        schema_name="rule_verdict",
        schema=RULE_VERDICT_SCHEMA,
        max_output_tokens=min(ctx.settings.llm_max_output_tokens, 512),
        context={
            "hypothesis": rule["hypothesis_text"],
            "spans": [
                {"id": sid, "text": doc.slice(s.start, s.end)} for sid, s in presented.items()
            ],
        },
    )
    step_key = (
        f"llm|verdict|{task_key}|{prompt.version_id}|{ctx.provider.model}|"
        f"{digest(doc.text_sha256, *(f'{s.start}:{s.end}' for s in spans))}"
    )
    out, meta = await ctx.llm.run(
        step_key=step_key,
        agent=agent,
        prompt_version=prompt.version_id,
        request=request,
        output_model=RuleVerdictOut,
        event_context={"task_key": task_key, "rule_key": rule["key"]},
    )
    info: dict[str, Any] = {
        "prompt_version": prompt.version_id,
        "llm_model": meta.model,
        "cached": meta.cached,
    }
    unknown = [i for i in out.evidence_span_ids if i not in presented]
    if unknown:
        info["grounding_violation"] = unknown[:10]
        return (
            "needs_review",
            [],
            "The LLM cited span ids that were not presented to it "
            f"({', '.join(unknown[:5])}); its answer was rejected.",
            info,
        )
    evidence = [presented[i] for i in dict.fromkeys(out.evidence_span_ids)]
    if out.verdict != "not_mentioned" and not evidence:
        return (
            "needs_review",
            [],
            f"The LLM answered {out.verdict} without citing evidence; not accepted.",
            info,
        )
    return out.verdict, evidence, out.rationale.strip()[:1000] or "(no rationale)", info


async def rule_check(ctx: RunContext, payload: dict[str, Any]) -> dict[str, Any]:
    task = payload["task"]
    model_info = payload["model_info"]
    rule = ctx.rules[task["rule_key"]]
    task_key = task["task_key"]
    document_id = uuid.UUID(task["document_id"])
    await ctx.store.set_task(task_key, "running")
    await ctx.store.emit(RULE_CHECK, "agent_start", {"task_key": task_key, "rule_key": rule["key"]})
    async with ctx.sessions() as session:
        doc = await load_doc(session, document_id, ctx.workspace_id)
    draft = FindingDraft(
        task_key=task_key,
        kind=PER_DOCUMENT,
        rule_key=rule["key"],
        rule_id=ctx.rule_id(rule["key"]),
        rule_title=rule["title"],
        severity=rule["severity"],
        document_id=document_id,
        subject_key=task["subject_key"],
        subject_label=task["subject_label"],
        machine_status="needs_review",
        decided_by="system",
        rationale="",
    )
    errors: list[dict[str, Any]] = []
    citations: list[CitationDraft] = []
    budget_update: dict[str, Any] = {}
    if doc is None:
        draft.rationale = (
            "The document was deleted or is no longer ready; the rule was not checked."
        )
        errors.append(error_record(RULE_CHECK, "document_unavailable", draft.rationale, task_key))
    else:
        tau = model_info.get("tau")
        escalate_reason: str | None = None
        seed: list[Span] = []
        verdict: ModelVerdict | None = None
        if _model_path(ctx, model_info, rule):
            sentences = doc.sentences()
            by_id = {s.model_id: s for s in sentences}
            step_key = f"model|{task_key}|{model_info.get('model_version')}|{doc.text_sha256[:16]}"
            memo = await ctx.store.memo_get(step_key)
            try:
                if memo is not None:
                    verdict = ModelVerdict(
                        status=memo["status"],
                        label=memo["label"],
                        confidence=memo["confidence"],
                        probabilities=memo["probabilities"],
                        evidence=[(e[0], e[1]) for e in memo["evidence"]],
                        considered=[(e[0], e[1]) for e in memo["considered"]],
                        model_version=memo["model_version"],
                    )
                else:
                    await ctx.store.emit(
                        RULE_CHECK,
                        "tool_call",
                        {
                            "task_key": task_key,
                            "tool": "model.check",
                            "backend": ctx.model.backend,
                            "spans": len(sentences),
                        },
                    )
                    verdict = await ctx.model.check(
                        rule["hypothesis_text"],
                        [(s.model_id, doc.slice(s.start, s.end)) for s in sentences],
                    )
                    await ctx.store.memo_put(
                        step_key,
                        RULE_CHECK,
                        {
                            "status": verdict.status,
                            "label": verdict.label,
                            "confidence": verdict.confidence,
                            "probabilities": verdict.probabilities,
                            "evidence": verdict.evidence,
                            "considered": verdict.considered,
                            "model_version": verdict.model_version,
                        },
                    )
            except ModelUnavailableError as exc:
                escalate_reason = f"model unavailable ({exc.code})"
                verdict = None
            if verdict is not None:
                await ctx.store.emit(
                    RULE_CHECK,
                    "result",
                    {
                        "task_key": task_key,
                        "tool": "model.check",
                        "status": verdict.status,
                        "confidence": round(verdict.confidence, 4),
                        "tau": tau,
                    },
                )
                if tau is not None and verdict.confidence < tau:
                    escalate_reason = f"model confidence {verdict.confidence:.3f} < tau {tau}"
                    seed = [
                        by_id[sid]
                        for sid, _ in sorted(
                            {**dict(verdict.considered), **dict(verdict.evidence)}.items(),
                            key=lambda kv: -kv[1],
                        )
                        if sid in by_id
                    ][:12]
                else:
                    draft.machine_status = verdict.status
                    draft.decided_by = "model"
                    draft.confidence = round(verdict.confidence, 6)
                    draft.model_version = verdict.model_version
                    draft.rationale = (
                        f"Trained rule-check model: {verdict.label} with calibrated confidence "
                        f"{verdict.confidence:.3f} (tau {tau})."
                    )
                    evidence = sorted(verdict.evidence, key=lambda e: -e[1])[
                        :MAX_EVIDENCE_CITATIONS
                    ]
                    citations = [
                        span_citation(doc, by_id[sid].start, by_id[sid].end, support_method=None)
                        for sid, _ in evidence
                        if sid in by_id
                    ]
            else:
                draft.model_version = model_info.get("model_version")
        else:
            escalate_reason = (
                None
                if not rule.get("model_supported")
                else (f"model unavailable ({model_info.get('reason') or 'not loaded'})")
            )
        if draft.decided_by != "model":
            draft.escalated = escalate_reason is not None
            draft.escalation_reason = escalate_reason
            spans = _llm_spans(doc, payload.get("candidates", {}), seed)
            try:
                status, evidence_spans, rationale, info = await llm_verdict(
                    ctx, task_key=task_key, rule=rule, doc=doc, spans=spans
                )
                draft.machine_status = status
                draft.decided_by = "llm"
                draft.rationale = rationale
                draft.prompt_version = info["prompt_version"]
                draft.llm_provider = ctx.provider.name
                draft.llm_model = info["llm_model"]
                if verdict is not None:
                    draft.model_version = verdict.model_version
                    draft.confidence = None
                citations = [span_citation(doc, s.start, s.end) for s in evidence_spans]
                if "grounding_violation" in info:
                    errors.append(
                        error_record(RULE_CHECK, "grounding_violation", rationale, task_key)
                    )
            except BudgetExceededError as exc:
                draft.rationale = f"LLM check not performed: the run budget is exhausted ({exc})."
                budget_update = {"exhausted": True}
                errors.append(error_record(RULE_CHECK, "budget_exhausted", str(exc), task_key))
            except LLMStepError as exc:
                draft.rationale = f"LLM check failed ({exc.code}): {exc.message}"
                errors.append(error_record(RULE_CHECK, exc.code, exc.message, task_key))
    draft.compliance = rule_compliance(rule, "needs_review")
    finding_id = await ctx.store.save_finding(draft, citations)
    for err in errors:
        await ctx.store.emit(RULE_CHECK, "error", err)
    await ctx.store.emit(
        RULE_CHECK,
        "result",
        {
            "task_key": task_key,
            "finding_id": str(finding_id),
            "machine_status": draft.machine_status,
            "decided_by": draft.decided_by,
            "escalated": draft.escalated,
            "citations": len(citations),
        },
    )
    update: dict[str, Any] = {
        "findings": {
            task_key: {
                "finding_id": str(finding_id),
                "kind": PER_DOCUMENT,
                "machine_status": draft.machine_status,
            }
        },
        "errors": errors,
    }
    if budget_update:
        update["budget"] = budget_update
    return update
