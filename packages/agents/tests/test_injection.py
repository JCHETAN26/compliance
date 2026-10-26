"""Prompt-injection regression (handoff section 8): document text is untrusted data.

1. The same NDA with and without an "ignore previous instructions" paragraph yields the
   same findings.
2. A *compromised* model that obeys the injection (fabricated span ids, quotes and dates)
   cannot turn the injected document compliant: ungrounded output is rejected and
   fabricated extractions are dropped.
3. Prompts carry the document only inside the delimited data block, with the delimiter
   spoof defanged, and requests never ask for tools.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select

from compliance_agents.prompting import DEFANGED_LT
from compliance_core.db.models import ExtractedEntity
from compliance_llm import MockProvider
from compliance_seed.synthetic import INJECTION, build

pytestmark = pytest.mark.integration

FILES = ("vendor_register.xlsx", "hooli_nda.pdf", "acme_nda.pdf")


def _subset(include_injection: bool) -> dict[str, bytes]:
    files = build(include_injection=include_injection)
    return {k: v for k, v in files.items() if k in FILES}


def _outcome(findings: dict[str, dict[str, Any]]) -> dict[tuple[str, str], tuple[str, str]]:
    return {
        (f["rule_key"], f["subject_label"]): (f["machine_status"], f["status"])
        for f in findings.values()
    }


async def test_injection_does_not_change_findings(review_env: Any) -> None:
    control = await review_env(_subset(False), email="control@example.com")
    control_run = await control.start()
    assert await control.execute(control_run) == "completed"
    injected = await review_env(_subset(True), email="injected@example.com")
    provider = MockProvider()
    injected_run = await injected.start()
    assert await injected.execute(injected_run, provider=provider) == "completed"
    assert _outcome(await injected.findings(injected_run)) == _outcome(
        await control.findings(control_run)
    )
    hooli = [
        r for r in provider.requests if any("ignore previous" in str(v) for v in r.context.values())
    ]
    assert hooli, "the injected text must have reached the provider as data"
    for request in provider.requests:
        if INJECTION[:40] not in request.user:
            continue
        data_open = request.user.index("<document_")
        assert request.user.index("ignore previous instructions") > data_open
        assert "</document_excerpts>" not in request.user.replace(
            request.user[request.user.rindex("</document_") :], ""
        )
        assert DEFANGED_LT + "/document_excerpts>" in request.user  # spoof defanged


async def test_model_obeying_the_injection_is_contained(review_env: Any) -> None:
    """The injected text reaches extraction; an adversary that obeys it fabricates an
    expiry date. The fabricated fact is rejected and the dependent check abstains."""
    env = await review_env(_subset(True))
    run_id = await env.start()
    adversary = MockProvider(behavior="follow_injections")
    assert await env.execute(run_id, provider=adversary) == "completed"
    findings = await env.findings(run_id)
    async with env.sessions() as session:
        fabricated = await session.scalar(
            select(ExtractedEntity).where(
                ExtractedEntity.review_run_id == run_id,
                ExtractedEntity.type == "expiration_date",
            )
        )
    assert fabricated is None  # "expires on January 1, 2099" is not in the document
    events = (await env.client.get(f"/api/v1/reviews/{run_id}/events/history?limit=1000")).json()
    rejected = [
        e["payload"].get("rejected", {})
        for e in events["items"]
        if e["agent"] == "extraction" and e["event_type"] == "result"
    ]
    assert any(r.get("quote_not_found") for r in rejected)
    term = next(
        f
        for f in findings.values()
        if f["rule_key"] == "contract_term_covers_renewal" and f["subject_label"] == "Hooli Inc"
    )
    assert term["status"] == "needs_review"
    for f in findings.values():
        assert f["status"] == "needs_review" or f["verified"]


async def test_always_lying_model_cannot_make_anything_compliant(review_env: Any) -> None:
    """A model that answers "satisfied" with fabricated span ids on every request: every
    LLM-decided finding is rejected (needs_review) and nothing depends on its word."""
    env = await review_env(_subset(True))
    run_id = await env.start()
    adversary = MockProvider(behavior="always_comply")
    assert await env.execute(run_id, provider=adversary) == "completed"
    findings = list((await env.findings(run_id)).values())
    llm = [f for f in findings if f["machine_decided_by"] == "llm"]
    assert llm
    for f in llm:
        assert f["status"] == "needs_review", f
        assert "not presented" in f["rationale"]
        assert not f["citations"]
    for f in findings:
        if f["rule_key"] == "high_access_required_protections":
            # Needs the LLM-checked rule: never satisfied on the liar's word (vendors whose
            # NDA is absent from this subset are violated regardless of any model).
            assert f["status"] in ("needs_review", "violated")
            if f["subject_label"] in ("Acme Corporation", "Hooli Inc"):
                assert f["status"] == "needs_review"
        if f["compliance"] == "compliant":
            assert f["verified"]
            assert f["machine_decided_by"] in ("model", "system")
    events = (await env.client.get(f"/api/v1/reviews/{run_id}/events/history?limit=1000")).json()
    codes = {e["payload"].get("code") for e in events["items"] if e["event_type"] == "error"}
    assert "grounding_violation" in codes
    assert all("tools" not in r.context for r in adversary.requests)
