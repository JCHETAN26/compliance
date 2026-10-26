"""Citation verifier negatives: an LLM support check that endorses only some of the exact
quotes, and an exact quote that extends beyond the chunk its citation claims."""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from compliance_agents.text import load_doc
from compliance_core.db.models import Citation, Finding, ReviewRun
from compliance_core.services import reviews
from compliance_llm import LLMRequest, LLMResponse, MockProvider
from compliance_seed.synthetic import build

pytestmark = pytest.mark.integration
SMALL = ("vendor_register.xlsx", "acme_nda.pdf", "globex_nda.docx")


def _small() -> dict[str, bytes]:
    return {k: v for k, v in build().items() if k in SMALL}


async def _findings(env: Any, run_id: uuid.UUID) -> list[Finding]:
    async with env.sessions() as session:
        return list(
            await session.scalars(
                select(Finding)
                .options(selectinload(Finding.citations))
                .where(Finding.run_id == run_id)
            )
        )


class PartialSupport(MockProvider):
    """LLM verdicts cite two spans; the support check then endorses only the first quote."""

    def __init__(self) -> None:
        super().__init__()
        self.endorsed: set[str] = set()
        self.passed_over: set[str] = set()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        response = await super().complete(request)
        if request.task == "rule_verdict" and len(request.context["spans"]) >= 2:
            data: dict[str, Any] = {
                "verdict": "satisfied",
                "evidence_span_ids": ["S1", "S2"],
                "rationale": "Both spans bear on the rule.",
            }
        elif request.task == "support_check" and len(request.context["quotes"]) >= 2:
            first, *rest = request.context["quotes"]
            self.endorsed.add(first["text"])
            self.passed_over.update(q["text"] for q in rest)
            data = {
                "supported": True,
                "supporting_quote_ids": [first["id"]],
                "rationale": "Only the first quote supports the claim.",
            }
        else:
            return response
        return dataclasses.replace(response, data=data)


async def test_support_for_a_subset_of_quotes_verifies_only_those_citations(
    review_env: Any,
) -> None:
    env = await review_env(_small())
    run_id = await env.start()
    provider = PartialSupport()
    assert await env.execute(run_id, provider=provider) == "completed"
    assert provider.endorsed
    assert provider.passed_over - provider.endorsed

    partial = [
        f
        for f in await _findings(env, run_id)
        if f.decided_by == "llm" and sum(c.role == "evidence" for c in f.citations) >= 2
    ]
    assert partial
    presented = await env.findings(run_id)
    for finding in partial:
        assert (finding.status, finding.verified) == ("satisfied", True)
        shown = {c["id"]: c for c in presented[finding.task_key]["citations"]}
        evidence = [c for c in finding.citations if c.role == "evidence"]
        assert all(c.exact_match for c in evidence)  # every quote is exact ...
        for c in evidence:
            endorsed = c.quote in provider.endorsed
            # ... but only the endorsed ones are supported, hence verified.
            assert (c.supported, c.verified) == (endorsed, endorsed), c.quote
            assert shown[str(c.id)]["verified"] is endorsed
        assert any(not c.verified for c in evidence)


async def test_quote_extending_beyond_its_cited_chunk_needs_review(review_env: Any) -> None:
    env = await review_env(_small())
    run_id = await env.start(max_concurrency=1)
    paused = False

    class PauseAtRuleCheck(MockProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            nonlocal paused
            if request.task == "rule_verdict" and not paused:
                paused = True  # takes effect at the end of the rule-check step
                async with env.sessions() as session, session.begin():
                    await session.execute(
                        update(ReviewRun).where(ReviewRun.id == run_id).values(control="pause")
                    )
            return await super().complete(request)

    assert await env.execute(run_id, provider=PauseAtRuleCheck()) == "paused"
    candidates = [
        f
        for f in await _findings(env, run_id)
        if f.decided_by == "model"
        and f.machine_status in ("satisfied", "violated")
        and not f.verification_json
        and any(c.role == "evidence" and c.chunk_id for c in f.citations)
    ]
    assert len(candidates) >= 2  # citations exist and the verifier has not run yet
    target, control = candidates[0], candidates[1]

    # Stretch one evidence quote past the end of its chunk. The longer quote is still the
    # document's exact text at the stored offsets, so only the chunk check can catch it.
    cited = next(c for c in target.citations if c.role == "evidence" and c.chunk_id)
    async with env.sessions() as session, session.begin():
        doc = await load_doc(session, cited.document_id, env.workspace_id)  # type: ignore[arg-type]
        assert doc is not None
        chunk = next(ch for ch in doc.chunks if ch.id == cited.chunk_id)
        assert chunk.start <= cited.char_start < cited.char_end <= chunk.end
        end = min(chunk.end + 40, len(doc.text))
        assert end > chunk.end
        await session.execute(
            update(Citation)
            .where(Citation.id == cited.id)
            .values(char_end=end, quote=doc.slice(cited.char_start, end))
        )

    async with env.sessions() as session:
        _, enqueue = await reviews.control_run(session, run_id, env.user_id, "resume")
    assert enqueue
    assert await env.execute(run_id) == "completed"

    after = {f.id: f for f in await _findings(env, run_id)}
    stretched = after[target.id]
    assert (stretched.status, stretched.verified) == ("needs_review", False)
    assert stretched.machine_status == target.machine_status  # the machine label is kept
    reasons = " ".join(stretched.verification_json["reasons"])
    assert "not contained in the cited chunk" in reasons
    row = next(c for c in stretched.citations if c.id == cited.id)
    assert (row.exact_match, row.verified) == (False, False)
    assert after[control.id].verified  # the untouched finding still verifies
    shown = (await env.findings(run_id))[target.task_key]
    assert shown["status"] == "needs_review"
    assert shown["verified"] is False
