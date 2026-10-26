"""Flagship vendor review end to end with the mock LLM (and the mock rule-check backend):
vendor register + matching contracts + a scanned (OCR) PDF + a prompt-injection NDA +
a missing contract. Checks the acceptance criteria: every finding is verified with exact,
supported citations or is ``needs_review``; the report states only verified facts."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy import select

from compliance_core.db.models import (
    AgentEvent,
    DocumentContent,
    ExtractedEntity,
    LlmCall,
    Page,
    ReviewReport,
    ReviewTask,
)
from compliance_seed.synthetic import build

pytestmark = pytest.mark.integration

AGENTS = {
    "planner",
    "extraction",
    "retrieval",
    "rule_check",
    "cross_document",
    "citation_verifier",
    "report_writer",
}


async def test_flagship_vendor_review_with_mock_llm(review_env: Any) -> None:
    env = await review_env(build(include_injection=True))
    run_id = await env.start()
    assert await env.execute(run_id) == "completed"
    run = await env.run(run_id)
    assert run.status == "completed"
    assert run.lease_token is None
    assert run.finished_at is not None

    async with env.sessions() as session:
        # The scanned contract really went through OCR.
        scanned = env.docs["initech_nda_scanned.pdf"]
        pages = list(await session.scalars(select(Page).where(Page.document_id == scanned)))
        assert pages
        assert all(p.is_ocr for p in pages)
        content = await session.get(DocumentContent, scanned)
        assert content is not None
        assert content.line_boxes

    by_task = await env.findings(run_id)
    findings = list(by_task.values())
    tasks = (await env.client.get(f"/api/v1/reviews/{run_id}/tasks")).json()["tasks"]
    assert len(findings) == len(tasks)
    assert all(t["status"] == "done" for t in tasks)
    # Ten register vendors (the handoff flagship), nine contract files:
    # 4 per-document rules x 9 contracts + contract-on-file (6 high-access rows) +
    # protections (6 high-access rows) + term (10 rows).
    assert len(findings) == 36 + 6 + 6 + 10

    # Acceptance: verified with exact, supported, currently valid citations, or needs_review.
    for f in findings:
        if f["status"] != "needs_review":
            assert f["verified"], f
            assert f["citations"], f
            assert any(c["verified"] for c in f["citations"]), f
            for c in f["citations"]:
                assert c["exact_match"], c
                assert c["current"] == "valid", c
        if f["compliance"] == "compliant":
            assert f["verified"]

    # Every stored quote is the document text at its offsets.
    async with env.sessions() as session:
        for f in findings:
            for c in f["citations"]:
                text = await session.scalar(
                    select(DocumentContent.text).where(
                        DocumentContent.document_id == uuid.UUID(c["document_id"])
                    )
                )
                assert text is not None
                assert text[c["char_start"] : c["char_end"]] == c["quote"]

    def find(rule: str, subject: str) -> dict[str, Any]:
        matches = [
            f
            for f in findings
            if f["rule_key"] == rule and subject.lower() in f["subject_label"].lower()
        ]
        assert len(matches) == 1, (rule, subject, [m["subject_label"] for m in matches])
        found: dict[str, Any] = matches[0]
        return found

    # Per-document checks.
    assert find("third_party_sharing_restricted", "acme_nda")["status"] == "satisfied"
    assert find("third_party_sharing_restricted", "acme_nda")["decided_by"] == "llm"
    assert find("return_or_destruction", "acme_nda")["decided_by"] == "model"
    globex_tp = find("third_party_sharing_restricted", "globex")
    assert (globex_tp["status"], globex_tp["compliance"]) == ("violated", "non_compliant")
    notice = find("notice_on_compelled_disclosure", "globex")
    assert notice["status"] == "needs_review"
    assert notice["machine_status"] == "not_mentioned"
    assert notice["compliance"] == "needs_review"
    assert not notice["citations"]
    hooli_return = find("return_or_destruction", "hooli")
    assert hooli_return["status"] == "needs_review"  # injected NDA lacks the clause
    assert find("return_or_destruction", "initech")["status"] == "satisfied"  # via OCR text
    pdf_cits = [
        c
        for f in findings
        for c in f["citations"]
        if c["document_filename"] in ("acme_nda.pdf", "initech_nda_scanned.pdf")
        and c["role"] == "evidence"
    ]
    assert pdf_cits
    assert all(c["bbox_precision"] == "line" and c["bboxes"] for c in pdf_cits)

    # Cross-document reasoning.
    assert find("high_access_contract_on_file", "Acme")["status"] == "satisfied"
    missing = find("high_access_contract_on_file", "Umbrella")
    assert (missing["status"], missing["compliance"]) == ("violated", "non_compliant")
    assert missing["citations"][0]["role"] == "register_row"
    assert "umbrella_nda.pdf" in missing["citations"][0]["quote"]
    assert find("high_access_required_protections", "Acme")["status"] == "satisfied"
    assert find("high_access_required_protections", "Umbrella")["status"] == "violated"
    hooli = find("high_access_required_protections", "Hooli")
    assert hooli["status"] == "needs_review"  # never assumed compliant
    assert find("contract_term_covers_renewal", "Acme")["status"] == "satisfied"
    globex_term = find("contract_term_covers_renewal", "Globex")
    assert globex_term["status"] == "violated"
    assert {c["role"] for c in globex_term["citations"]} == {"register_row", "entity"}
    assert "2025-01-15" in globex_term["rationale"]
    assert find("contract_term_covers_renewal", "Vandelay")["status"] == "needs_review"
    for vendor in ("Stark", "Soylent"):
        assert find("high_access_contract_on_file", vendor)["status"] == "satisfied"
        assert find("high_access_required_protections", vendor)["status"] == "satisfied"
    assert find("contract_term_covers_renewal", "Stark")["status"] == "satisfied"
    soylent = find("contract_term_covers_renewal", "Soylent")
    assert soylent["status"] == "violated"  # two-year term ended 2026-01-01
    assert "2026-01-01" in soylent["rationale"]
    wayne = find("contract_term_covers_renewal", "Wayne")
    assert wayne["status"] == "satisfied"  # explicit expiry date 2027-12-31
    assert "2027-12-31" in wayne["rationale"]
    assert find("survival_of_obligations", "wayne_nda")["status"] == "needs_review"
    tyrell = find("contract_term_covers_renewal", "Tyrell")
    assert tyrell["status"] == "needs_review"  # only tyrell_nda_v1/_v2 exist: ambiguous
    assert any("ambiguous" in step["detail"] for step in tyrell["reasoning"])
    assert find("contract_term_covers_renewal", "Umbrella")["status"] == "needs_review"
    assert not [
        f
        for f in findings
        if f["rule_key"].startswith("high_access") and "Globex" in f["subject_label"]
    ]
    assert all(
        step["detail"] for step in find("high_access_required_protections", "Acme")["reasoning"]
    )

    async with env.sessions() as session:
        # Extraction: dates and terms cite chunks with exact quotes; links are recorded.
        entities = list(
            await session.scalars(
                select(ExtractedEntity).where(ExtractedEntity.review_run_id == run_id)
            )
        )
        dated = [e for e in entities if e.type in ("effective_date", "term")]
        assert dated
        assert all(e.chunk_id and e.verified and e.quote for e in dated)
        links = [e for e in entities if e.type == "register_link"]
        assert len(links) == 10
        assert {e.value_json["outcome"] for e in links} == {
            "linked",
            "missing",
            "no_contract_listed",
            "ambiguous",
        }
        # Events: every agent reported, cursor strictly increasing.
        events = list(
            await session.scalars(
                select(AgentEvent).where(AgentEvent.run_id == run_id).order_by(AgentEvent.seq)
            )
        )
        assert {e.agent for e in events} >= AGENTS
        seqs = [e.seq for e in events]
        assert seqs == sorted(set(seqs))
        assert all("quote" not in e.payload_json for e in events)
        # Budget ledger: all attempts settled, mock usage labelled, cost 0.
        calls = list(await session.scalars(select(LlmCall).where(LlmCall.run_id == run_id)))
        assert calls
        assert {c.status for c in calls} == {"succeeded"}
        assert {c.usage_source for c in calls} == {"mock_estimate"}
        assert sum(c.cost_microusd or 0 for c in calls) == 0
        report = await session.get(ReviewReport, run_id)
        assert report is not None
        assert "MOCK" in report.markdown
        tasks_done = await session.scalar(
            select(ReviewTask.status).where(ReviewTask.run_id == run_id).limit(1)
        )
        assert tasks_done == "done"

    # The report states only verified/human facts, with finding ids.
    response = await env.client.get(f"/api/v1/reviews/{run_id}/report")
    assert response.status_code == 200
    body = response.json()
    by_id = {f["id"]: f for f in findings}
    for statement in body["summary"][1:]:
        ids = statement["finding_ids"]
        assert ids
        if "need human review" not in statement["text"]:
            assert all(by_id[i]["verified"] for i in ids)
    assert all(by_id[r["finding_id"]]["verified"] for r in body["risks"])
    assert body["llm_is_mock"]
    assert body["complete"]
    assert any("MOCK" in item for item in body["limitations"])
    md = await env.client.get(f"/api/v1/reviews/{run_id}/report", params={"format": "md"})
    assert md.status_code == 200
    assert md.headers["content-type"].startswith("text/markdown")
    assert "Umbrella GmbH" in md.text

    # Citation detail opens the right document region (and only for members).
    cid = missing["citations"][0]["id"]
    detail = (await env.client.get(f"/api/v1/citations/{cid}")).json()
    assert detail["document_id"] == str(env.docs["vendor_register.xlsx"])
    assert detail["current"] == "valid"
    assert detail["file_format"] == "xlsx"
