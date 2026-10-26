"""Human-in-the-loop interrupt with audited overrides, cited sources that change or
disappear (after completion and mid-run), escalation below tau, and strict validation of
the model service's answers."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import httpx
import pytest
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import DBAPIError

from compliance_agents.model import ModelUnavailableError, ServiceRuleCheckModel
from compliance_core.db.models import (
    AgentEvent,
    DocumentContent,
    ExtractedEntity,
    Finding,
    HumanOverride,
    ReviewRun,
)
from compliance_core.services import reviews
from compliance_core.services.errors import InvalidOperationError
from compliance_llm import LLMRequest, MockProvider
from compliance_seed.synthetic import build

pytestmark = pytest.mark.integration
SMALL = ("vendor_register.xlsx", "acme_nda.pdf", "globex_nda.docx")


def _small() -> dict[str, bytes]:
    return {k: v for k, v in build().items() if k in SMALL}


async def test_interrupt_override_and_resume(review_env: Any) -> None:
    env = await review_env(_small())
    run_id = await env.start(pause_on_needs_review=True)
    assert await env.execute(run_id) == "awaiting_review"
    run = (await env.client.get(f"/api/v1/reviews/{run_id}")).json()
    assert run["status"] == "awaiting_review"
    findings = await env.findings(run_id)
    pending = [f for f in findings.values() if f["status"] == "needs_review"]
    assert pending
    target = next(f for f in pending if f["rule_key"] == "notice_on_compelled_disclosure")
    response = await env.client.post(
        f"/api/v1/findings/{target['id']}/override",
        json={"status": "violated", "comment": "Checked the Globex NDA: no notice clause."},
    )
    assert response.status_code == 200, response.text
    overridden = response.json()
    assert overridden["status"] == "violated"
    assert overridden["decided_by"] == "human"
    assert overridden["machine_status"] == "not_mentioned"  # machine finding preserved
    assert overridden["verified_status"] == "needs_review"
    assert overridden["override"]["comment"].startswith("Checked")
    resumed = await env.client.post(f"/api/v1/reviews/{run_id}/resume")
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["status"] == "queued"
    assert await env.execute(run_id) == "completed"
    async with env.sessions() as session:
        row = await session.get(Finding, uuid.UUID(target["id"]))
        assert row is not None
        assert (row.status, row.machine_status, row.decided_by) == (
            "needs_review",
            "not_mentioned",
            "model",
        )
        audits = await session.scalar(
            select(func.count())
            .select_from(HumanOverride)
            .where(HumanOverride.finding_id == row.id)
        )
    assert audits == 1
    run = (await env.client.get(f"/api/v1/reviews/{run_id}")).json()
    assert run["totals"]["human_decided"] == 1
    assert run["totals"]["findings_by_status"]["violated"] >= 1
    report = (await env.client.get(f"/api/v1/reviews/{run_id}/report")).json()
    risk_ids = {r["finding_id"] for r in report["risks"]}
    assert target["id"] in risk_ids  # a human decision may be stated, marked as such
    assert (
        next(r for r in report["risks"] if r["finding_id"] == target["id"])["decided_by"] == "human"
    )


async def test_overrides_are_refused_while_running(review_env: Any) -> None:
    env = await review_env(_small())
    run_id = await env.start()
    assert await env.execute(run_id) == "completed"
    finding = next(iter((await env.findings(run_id)).values()))
    async with env.sessions() as session, session.begin():
        await session.execute(
            update(ReviewRun).where(ReviewRun.id == run_id).values(status="running")
        )
    response = await env.client.post(
        f"/api/v1/findings/{finding['id']}/override",
        json={"status": "satisfied", "comment": "x"},
    )
    assert response.status_code == 409


async def _lock_waiters(env: Any, expected: int) -> None:
    """Wait until ``expected`` backends of this database are blocked on a lock."""
    query = text(
        "SELECT count(*) FROM pg_stat_activity "
        "WHERE datname = current_database() AND wait_event_type = 'Lock'"
    )
    for _ in range(200):
        async with env.sessions() as session:
            if await session.scalar(query) >= expected:
                return
        await asyncio.sleep(0.05)
    raise AssertionError(f"expected {expected} lock waiter(s)")


async def test_override_locks_the_run_first_and_loses_to_a_concurrent_resume(
    review_env: Any,
) -> None:
    env = await review_env(_small())
    run_id = await env.start(pause_on_needs_review=True)
    assert await env.execute(run_id) == "awaiting_review"
    pending = [f for f in (await env.findings(run_id)).values() if f["status"] == "needs_review"]
    assert len(pending) >= 2
    first, second = (uuid.UUID(f["id"]) for f in pending[:2])

    async def override(finding_id: uuid.UUID) -> Any:
        async with env.sessions() as session:
            return await reviews.override_finding(
                session, finding_id, env.user_id, "violated", "Checked the source."
            )

    async def resume() -> Any:
        async with env.sessions() as session:
            return await reviews.control_run(session, run_id, env.user_id, "resume")

    # 1. Lock order. A worker-style transaction holds the run and then needs the finding:
    # the waiting override must not already hold the finding (opposite order deadlocks).
    async with env.sessions() as holder, holder.begin():
        await holder.execute(select(ReviewRun.id).where(ReviewRun.id == run_id).with_for_update())
        waiting = asyncio.create_task(override(first))
        await _lock_waiters(env, 1)
        try:
            await holder.execute(
                select(Finding.id).where(Finding.id == first).with_for_update(nowait=True)
            )
        except DBAPIError as exc:  # pragma: no cover - the regression this test guards
            waiting.cancel()
            raise AssertionError("override locked the finding before the run") from exc
    assert (await asyncio.wait_for(waiting, 30)).status == "violated"

    # 2. A resume that takes the run lock first wins; the override queued behind it must
    # re-check the status under the lock and be refused, not recorded after the resume.
    async with env.sessions() as holder, holder.begin():
        await holder.execute(select(ReviewRun.id).where(ReviewRun.id == run_id).with_for_update())
        resuming = asyncio.create_task(resume())
        await _lock_waiters(env, 1)
        overriding = asyncio.create_task(override(second))
        await _lock_waiters(env, 2)
    resumed, enqueue = await asyncio.wait_for(resuming, 30)
    assert (resumed.status, enqueue) == ("queued", True)
    with pytest.raises(InvalidOperationError):
        await asyncio.wait_for(overriding, 30)

    async with env.sessions() as session:
        audited = set(await session.scalars(select(HumanOverride.finding_id)))
        events = list(
            await session.scalars(
                select(AgentEvent.event_type)
                .where(
                    AgentEvent.run_id == run_id,
                    AgentEvent.event_type.in_(["override_recorded", "resume_requested"]),
                )
                .order_by(AgentEvent.seq)
            )
        )
    assert audited == {first}
    assert events == ["override_recorded", "resume_requested"]
    assert await env.execute(run_id) == "completed"


async def test_changed_or_deleted_sources_are_no_longer_presented_as_verified(
    review_env: Any,
) -> None:
    env = await review_env(_small())
    run_id = await env.start()
    assert await env.execute(run_id) == "completed"
    before = await env.findings(run_id)
    acme_doc = env.docs["acme_nda.pdf"]
    acme = [
        f
        for f in before.values()
        if f["verified"]
        and any(
            c["document_id"] == str(acme_doc) and "Receiving Party" in c["quote"]
            for c in f["citations"]
        )
    ]
    assert acme
    # 1. The source text changes (e.g. re-processing): quotes are no longer at their offsets.
    async with env.sessions() as session, session.begin():
        content = await session.get(DocumentContent, acme_doc)
        assert content is not None
        content.text = content.text.replace("Receiving Party", "Recipient Party")
    after = await env.findings(run_id)
    for f in acme:
        now = after[f["task_key"]]
        assert now["status"] == "needs_review"
        assert now["source_changed"]
        assert not now["verified"]
        assert now["verified_status"] == f["status"]  # the historical record is kept
    run = (await env.client.get(f"/api/v1/reviews/{run_id}")).json()
    assert run["totals"]["source_changed"] >= len(acme)
    report = (await env.client.get(f"/api/v1/reviews/{run_id}/report")).json()
    stated = {
        i
        for s in report["summary"]
        if "need human review" not in s["text"]
        for i in s["finding_ids"]
    }
    assert not stated & {f["id"] for f in acme}
    # 2. The document is deleted: citations stay as history, findings need review.
    globex_doc = env.docs["globex_nda.docx"]
    globex = [
        f
        for f in after.values()
        if f["verified"] and any(c["document_id"] == str(globex_doc) for c in f["citations"])
    ]
    assert globex
    assert (await env.client.delete(f"/api/v1/documents/{globex_doc}")).status_code == 204
    final = await env.findings(run_id)
    for f in globex:
        now = final[f["task_key"]]
        assert now["status"] == "needs_review"
        cited = [c for c in now["citations"] if c["document_filename"] == "globex_nda.docx"]
        assert cited
        assert all(c["current"] == "source_deleted" and c["document_id"] is None for c in cited)
    detail = (await env.client.get(f"/api/v1/citations/{cited[0]['id']}")).json()
    assert detail["current"] == "source_deleted"
    assert detail["context_before"] == ""


async def test_source_changed_mid_run_does_not_lend_new_text_to_old_facts(
    review_env: Any,
) -> None:
    """Pause after extraction, change the date the extracted entity cited (same length),
    resume: the cross-document term check must not use the stale entity."""
    env = await review_env(_small())
    run_id = await env.start()

    async def pause_after_extract(n: int, request: LLMRequest) -> None:
        if request.task == "extract_entities":
            async with env.sessions() as session, session.begin():
                await session.execute(
                    update(ReviewRun).where(ReviewRun.id == run_id).values(control="pause")
                )

    class Hooked(MockProvider):
        async def complete(self, request: LLMRequest) -> Any:
            await pause_after_extract(0, request)
            return await super().complete(request)

    assert await env.execute(run_id, provider=Hooked()) == "paused"
    async with env.sessions() as session:
        entity = await session.scalar(
            select(ExtractedEntity).where(
                ExtractedEntity.review_run_id == run_id,
                ExtractedEntity.document_id == env.docs["globex_nda.docx"],
                ExtractedEntity.type == "effective_date",
            )
        )
    assert entity is not None
    assert entity.quote is not None
    assert "2024" in entity.quote
    async with env.sessions() as session, session.begin():
        content = await session.get(DocumentContent, env.docs["globex_nda.docx"])
        assert content is not None
        content.text = content.text.replace("January 15, 2024", "January 15, 2034")
    async with env.sessions() as session, session.begin():
        await session.execute(
            update(ReviewRun).where(ReviewRun.id == run_id).values(status="queued", control=None)
        )
    assert await env.execute(run_id) == "completed"
    term = next(
        f
        for f in (await env.findings(run_id)).values()
        if f["rule_key"] == "contract_term_covers_renewal" and f["subject_label"] == "Globex Ltd"
    )
    assert term["status"] == "needs_review"
    assert "changed after its dates were extracted" in term["rationale"]


async def test_low_confidence_escalates_to_the_llm(review_env: Any) -> None:
    env = await review_env(_small())
    run_id = await env.start(tau=1.0)  # a workspace override above every mock confidence
    assert await env.execute(run_id) == "completed"
    findings = (await env.findings(run_id)).values()
    model_rules = [f for f in findings if f["rule_key"] == "return_or_destruction"]
    assert model_rules
    for f in model_rules:
        assert f["escalated"]
        assert "< tau 1.0" in (f["escalation_reason"] or "")
        assert f["machine_decided_by"] == "llm"
    run = (await env.client.get(f"/api/v1/reviews/{run_id}")).json()
    assert run["tau"] == 1.0


def _check_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "label": "entailment",
        "status": "satisfied",
        "confidence": 0.8,
        "probabilities": {"entailment": 0.8, "contradiction": 0.1, "not_mentioned": 0.1},
        "evidence": [{"span_id": "a", "score": 0.9}],
        "considered": [{"span_id": "a", "score": 0.9}],
        "escalate": False,
        "truncated_inputs": 0,
        "model_version": "v1",
        "model_kind": "production",
        "variant": "fp32",
    }
    body.update(overrides)
    return body


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "violated"},
        {"probabilities": {"entailment": 0.8, "contradiction": 0.1}},
        {
            "probabilities": {
                "entailment": 0.8,
                "contradiction": 0.1,
                "not_mentioned": 0.1,
                "x": 0.0,
            }
        },
        {"probabilities": {"entailment": 0.9, "contradiction": 0.1, "not_mentioned": 0.1}},
        {"probabilities": {"entailment": 1.2, "contradiction": -0.1, "not_mentioned": -0.1}},
        {"confidence": 0.5},
        {"label": "contradiction", "status": "violated", "confidence": 0.1},
        {"evidence": [{"span_id": "a", "score": 0.9}, {"span_id": "a", "score": 0.8}]},
        {"evidence": [{"span_id": "zzz", "score": 0.9}]},
        {"evidence": [{"span_id": "a", "score": 1.5}]},
        {"model_kind": "smoke"},
    ],
)
async def test_malformed_model_answers_are_rejected(overrides: dict[str, Any]) -> None:
    payload = json.dumps(_check_body(**overrides)).replace("NaN", "NaN")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload, headers={"content-type": "application/json"})

    model = ServiceRuleCheckModel(
        "http://model.test", timeout=5, retries=0, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ModelUnavailableError):
        await model.check("hypothesis", [("a", "text")])
    await model.aclose()


async def test_nan_probabilities_are_rejected() -> None:
    raw = json.dumps(_check_body()).replace('"contradiction": 0.1', '"contradiction": NaN')

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=raw, headers={"content-type": "application/json"})

    model = ServiceRuleCheckModel(
        "http://model.test", timeout=5, retries=0, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ModelUnavailableError):
        await model.check("hypothesis", [("a", "text")])
    await model.aclose()


@pytest.mark.parametrize("tau", [None, "0.3", True, -0.1, 1.5])
async def test_model_without_valid_tau_is_not_used(tau: Any) -> None:
    inference = {} if tau is None else {"tau": tau}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "rule_check": {
                    "available": True,
                    "mode": "model",
                    "kind": "production",
                    "model_version": "v1",
                    "variant": "fp32",
                    "inference": inference,
                }
            },
        )

    model = ServiceRuleCheckModel(
        "http://model.test", timeout=5, retries=0, transport=httpx.MockTransport(handler)
    )
    info = await model.info()
    await model.aclose()
    assert not info.available
    assert info.reason is not None


async def test_valid_model_answer_and_tau_are_accepted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metadata":
            return httpx.Response(
                200,
                json={
                    "rule_check": {
                        "available": True,
                        "kind": "production",
                        "model_version": "v1",
                        "variant": "fp32",
                        "inference": {"tau": 0.3},
                    }
                },
            )
        return httpx.Response(200, json=_check_body())

    model = ServiceRuleCheckModel(
        "http://model.test", timeout=5, retries=0, transport=httpx.MockTransport(handler)
    )
    info = await model.info()
    verdict = await model.check("hypothesis", [("a", "text")])
    await model.aclose()
    assert info.available
    assert info.tau == 0.3
    assert verdict.status == "satisfied"
    assert verdict.evidence == [("a", 0.9)]


async def test_mis_roled_dates_are_not_used_for_the_term_decision(review_env: Any) -> None:
    """A provider proposes an offer's expiry as the agreement's expiry and a
    confidentiality period as the agreement's term (both quotes really exist): the
    extraction rejects them and the term check abstains instead of deciding."""
    from compliance_seed.synthetic import register_xlsx, text_pdf

    nda = text_pdf(
        [
            "MUTUAL NON-DISCLOSURE AGREEMENT",
            "This Mutual Non-Disclosure Agreement is entered into between Acme Corporation and "
            "Contoso Ltd (the parties) to protect confidentiality of information.",
            "The offer expires on January 1, 2099.",
            "The confidentiality obligation continues for a period of three years.",
            "The Receiving Party shall return or destroy all Confidential Information.",
        ],
        "acme nda",
    )
    env = await review_env({"vendor_register.xlsx": register_xlsx(), "acme_nda.pdf": nda})
    run_id = await env.start()

    class MisRoled(MockProvider):
        def _answer(self, request: LLMRequest) -> dict[str, Any]:
            if request.task != "extract_entities":
                return super()._answer(request)
            chunks = {str(c["ref"]): str(c["text"]) for c in request.context["chunks"]}
            ref = next(r for r, t in chunks.items() if "offer expires" in t)
            ref2 = next(r for r, t in chunks.items() if "continues for a period" in t)
            return {
                "entities": [
                    {
                        "type": "expiration_date",
                        "chunk_ref": ref,
                        "quote": "The offer expires on January 1, 2099.",
                        "value": "2099-01-01",
                    },
                    {
                        "type": "term",
                        "chunk_ref": ref2,
                        "quote": "The confidentiality obligation continues for a period of three years.",
                        "value": "3 years",
                    },
                ]
            }

    assert await env.execute(run_id, provider=MisRoled()) == "completed"
    async with env.sessions() as session:
        kept = await session.scalar(
            select(func.count())
            .select_from(ExtractedEntity)
            .where(
                ExtractedEntity.review_run_id == run_id,
                ExtractedEntity.type.in_(["expiration_date", "term"]),
            )
        )
    assert kept == 0
    events = (await env.client.get(f"/api/v1/reviews/{run_id}/events/history?limit=1000")).json()
    rejected = [
        e["payload"].get("rejected", {})
        for e in events["items"]
        if e["agent"] == "extraction" and e["event_type"] == "result"
    ]
    assert any(r.get("role_not_supported_by_quote") == 2 for r in rejected), rejected
    term = next(
        f
        for f in (await env.findings(run_id)).values()
        if f["rule_key"] == "contract_term_covers_renewal"
        and f["subject_label"] == "Acme Corporation"
    )
    assert term["status"] == "needs_review"
    assert "could not be determined" in term["rationale"]
