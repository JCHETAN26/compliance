"""Citation verifier agent.

A finding keeps its machine status only if **both** checks pass:

1. Exactness: every citation's quote equals the document's current canonical text at the
   stored offsets, the document still exists in the run's workspace, and the cited chunk
   (if any) covers the range.
2. Semantic support (a valid quote is not proof):
   - model-decided findings are re-checked by the trained model on the cited spans alone;
     supported only if it reaches the same status with confidence >= tau;
   - LLM-decided findings get a separate structured support check over the quotes,
     grounded on the quote ids presented;
   - system (cross-document) findings: the register row must name the vendor, entity
     quotes must be exact, and every supporting citation must come from a finding that is
     itself still verified.

Otherwise the finding becomes ``needs_review`` with the reasons recorded. A finding with
no citation (e.g. ``not_mentioned``: absence cannot be quoted) always needs review.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from compliance_agents.context import RunContext
from compliance_agents.facts import text_contains
from compliance_agents.gateway import LLMStepError
from compliance_agents.model import ModelUnavailableError
from compliance_agents.nodes.common import digest, rule_compliance
from compliance_agents.outputs import SUPPORT_SCHEMA, SupportOut
from compliance_agents.prompting import load_prompt, nonce, numbered
from compliance_agents.store import BudgetExceededError
from compliance_agents.text import DocView, load_doc
from compliance_core.db.models import Citation, Finding
from compliance_llm import LLMRequest

AGENT = "citation_verifier"


async def _exact(
    ctx: RunContext, c: Citation, docs: dict[uuid.UUID, DocView | None]
) -> tuple[bool, str]:
    if c.document_id is None:
        return False, "source document deleted"
    if c.document_id not in docs:
        async with ctx.sessions() as session:
            docs[c.document_id] = await load_doc(session, c.document_id, ctx.workspace_id)
    doc = docs[c.document_id]
    if doc is None:
        return False, "source document unavailable or outside the workspace"
    if not (0 <= c.char_start < c.char_end <= len(doc.text)):
        return False, "offsets outside the document"
    if doc.slice(c.char_start, c.char_end) != c.quote:
        return False, "quote differs from the document text at the stored offsets"
    if c.chunk_id is not None:
        chunk = next((ch for ch in doc.chunks if ch.id == c.chunk_id), None)
        if chunk is None or not (chunk.start <= c.char_start and c.char_end <= chunk.end):
            return False, "the quote is not contained in the cited chunk"
    return True, ""


async def _model_support(
    ctx: RunContext, finding: Finding, evidence: list[Citation], hypothesis: str, tau: float | None
) -> tuple[bool | None, float | None, str, str]:
    spans = [(f"{c.char_start}-{c.char_end}", c.quote) for c in evidence]
    step_key = f"verify-model|{finding.id}|{finding.model_version}|{digest(*(s for s, _ in spans))}"
    memo = await ctx.store.memo_get(step_key)
    try:
        if memo is None:
            await ctx.store.emit(
                AGENT,
                "tool_call",
                {"finding_id": str(finding.id), "tool": "model.check", "spans": len(spans)},
            )
            verdict = await ctx.model.check(hypothesis, spans)
            memo = {
                "status": verdict.status,
                "confidence": verdict.confidence,
                "probabilities": verdict.probabilities,
            }
            await ctx.store.memo_put(step_key, AGENT, memo)
    except ModelUnavailableError as exc:
        return None, None, "model", f"model unavailable for support check ({exc.code})"
    label = {
        "satisfied": "entailment",
        "violated": "contradiction",
        "not_mentioned": "not_mentioned",
    }
    score = float(memo["probabilities"].get(label[finding.machine_status], 0.0))
    supported = memo["status"] == finding.machine_status and (
        tau is None or memo["confidence"] >= tau
    )
    detail = (
        f"model on cited spans only: {memo['status']} ({memo['confidence']:.3f}); "
        f"claim {finding.machine_status}"
    )
    return (
        supported,
        round(score, 6),
        "mock_model" if ctx.model.backend == "mock" else "model",
        detail,
    )


async def _llm_support(
    ctx: RunContext, finding: Finding, evidence: list[Citation], hypothesis: str
) -> tuple[bool | None, set[str], str, str]:
    prompt = load_prompt("support_check")
    quotes = {f"Q{i}": c for i, c in enumerate(evidence, start=1)}
    request = LLMRequest(
        task="support_check",
        system=prompt.render_system(),
        user=prompt.render(
            hypothesis=hypothesis,
            claimed_status=finding.machine_status,
            quotes=numbered([(qid, c.quote) for qid, c in quotes.items()]),
            nonce=nonce(str(ctx.run_id), "support", str(finding.id)),
        ),
        schema_name="support_check",
        schema=SUPPORT_SCHEMA,
        max_output_tokens=min(ctx.settings.llm_max_output_tokens, 384),
        context={
            "hypothesis": hypothesis,
            "claimed_status": finding.machine_status,
            "quotes": [{"id": qid, "text": c.quote} for qid, c in quotes.items()],
        },
    )
    step_key = (
        f"llm|support|{finding.id}|{prompt.version_id}|{ctx.provider.model}|"
        f"{digest(*(f'{c.char_start}:{c.char_end}' for c in evidence))}"
    )
    out, _ = await ctx.llm.run(
        step_key=step_key,
        agent=AGENT,
        prompt_version=prompt.version_id,
        request=request,
        output_model=SupportOut,
        event_context={"finding_id": str(finding.id)},
    )
    method = "mock_llm" if ctx.provider.is_mock else "llm"
    unknown = [q for q in out.supporting_quote_ids if q not in quotes]
    if unknown:
        return False, set(), method, f"support check cited unknown quote ids {unknown[:5]}"
    ids = {str(quotes[q].id) for q in out.supporting_quote_ids}
    supported = out.supported and bool(ids)
    return supported, ids, method, out.rationale.strip()[:300]


async def verify(ctx: RunContext, payload: dict[str, Any]) -> dict[str, Any]:
    finding_id = uuid.UUID(payload["finding_id"])
    tau = payload.get("tau")
    async with ctx.sessions() as session:
        finding = await session.scalar(
            select(Finding)
            .options(selectinload(Finding.citations))
            .where(Finding.id == finding_id, Finding.run_id == ctx.run_id)
        )
    if finding is None:
        return {"verified": {str(finding_id): {"status": "needs_review", "verified": False}}}
    rule = ctx.rules[finding.rule_key]
    reasons: list[str] = []
    results: dict[uuid.UUID, dict[str, Any]] = {}
    docs: dict[uuid.UUID, DocView | None] = {}
    citations = list(finding.citations)
    exact_all = True
    for c in citations:
        ok, why = await _exact(ctx, c, docs)
        results[c.id] = {"exact_match": ok, "verified": False}
        if not ok:
            exact_all = False
            reasons.append(f"{c.document_filename}: {why}")
    status = finding.machine_status
    supported: bool | None = None
    method = None
    details = ""
    budget: dict[str, Any] = {}
    if finding.machine_status == "needs_review":
        reasons.append("the deciding agent could not reach a supported decision")
    elif not citations or (
        finding.decided_by != "system" and not any(c.role == "evidence" for c in citations)
    ):
        reasons.append(
            "no citable evidence (a 'not mentioned' result cannot be quoted)"
            if finding.machine_status == "not_mentioned"
            else "no evidence was cited"
        )
    elif not exact_all:
        reasons.append("at least one quote is not exact")
    elif finding.decided_by == "model":
        evidence = [c for c in citations if c.role == "evidence"]
        supported, score, method, details = await _model_support(
            ctx, finding, evidence, rule["hypothesis_text"], tau
        )
        if supported is None:
            # The model service is unavailable: fall back to the LLM support check.
            try:
                supported, ids, method, details = await _llm_support(
                    ctx, finding, evidence, rule["hypothesis_text"]
                )
                for c in evidence:
                    results[c.id].update(supported=str(c.id) in ids, support_method=method)
            except BudgetExceededError as exc:
                supported, budget = None, {"exhausted": True}
                reasons.append(f"support check not performed: budget exhausted ({exc})")
            except LLMStepError as exc:
                supported = None
                reasons.append(f"support check failed ({exc.code})")
        else:
            for c in evidence:
                results[c.id].update(
                    supported=supported, support_score=score, support_method=method
                )
    elif finding.decided_by == "llm":
        evidence = [c for c in citations if c.role == "evidence"]
        try:
            supported, ids, method, details = await _llm_support(
                ctx, finding, evidence, rule["hypothesis_text"]
            )
            for c in evidence:
                results[c.id].update(supported=str(c.id) in ids, support_method=method)
        except BudgetExceededError as exc:
            supported, budget = None, {"exhausted": True}
            reasons.append(f"support check not performed: budget exhausted ({exc})")
        except LLMStepError as exc:
            supported = None
            reasons.append(f"support check failed ({exc.code})")
    else:  # system (cross-document): deterministic support
        method = "deterministic"
        ok = True
        for c in citations:
            good = True
            if c.role == "register_row":
                good = text_contains(c.quote, finding.subject_label)
                if not good:
                    reasons.append("register row does not name the vendor")
            elif c.role == "supporting":
                source = (c.support_method or "").removeprefix("finding:")
                async with ctx.sessions() as session:
                    src = (
                        await session.scalar(
                            select(Finding.verified).where(
                                Finding.id == uuid.UUID(source), Finding.run_id == ctx.run_id
                            )
                        )
                        if source
                        else None
                    )
                good = bool(src)
                if not good:
                    reasons.append("a supporting finding is not verified")
            results[c.id].update(supported=good)
            ok = ok and good
        supported = ok
        details = "deterministic checks of register row, entities and supporting findings"
    if supported is False and not reasons:
        reasons.append(f"citations do not support the claim ({details})")
    verified = bool(exact_all and supported and not reasons)
    if not verified:
        status = "needs_review"
    for values in results.values():
        # A citation is verified only if it is exact *and* was itself judged supporting
        # (an LLM support check may endorse only some of the quotes).
        values["verified"] = (
            verified and bool(values.get("exact_match")) and values.get("supported") is not False
        )
    compliance = rule_compliance(rule, status)
    await ctx.store.save_verification(
        finding.id,
        status=status,
        verified=verified,
        compliance=compliance,
        verification={
            "exact": exact_all,
            "supported": supported,
            "method": method,
            "details": details,
            "reasons": reasons,
            "tau": tau,
        },
        citation_results=results,
    )
    await ctx.store.emit(
        AGENT,
        "result",
        {
            "finding_id": str(finding.id),
            "task_key": finding.task_key,
            "machine_status": finding.machine_status,
            "status": status,
            "verified": verified,
            "method": method,
            "reasons": reasons[:3],
        },
    )
    update: dict[str, Any] = {
        "verified": {str(finding.id): {"status": status, "verified": verified}}
    }
    if budget:
        update["budget"] = budget
    return update
