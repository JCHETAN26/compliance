"""Cross-document reasoning agent.

Deterministic multi-step reasoning per register row, over the register row itself, the
conservative register link, *verified* per-document findings and *verified* extracted
entities. Every step is recorded in ``reasoning`` and every fact is cited (the register
row, the supporting findings' evidence, the entity quotes). Missing, ambiguous or
unverified inputs give ``needs_review``; nothing is assumed compliant.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

from compliance_agents.context import RunContext
from compliance_agents.facts import add_months, parse_iso, parse_register_date, text_contains
from compliance_agents.nodes.common import (
    CROSS_DOCUMENT,
    error_record,
    rule_compliance,
    span_citation,
)
from compliance_agents.nodes.planner import NO_REGISTER
from compliance_agents.store import CitationDraft, FindingDraft
from compliance_agents.text import DocView, load_doc
from compliance_core.db.models import ExtractedEntity, Finding

AGENT = "cross_document"


class _Docs:
    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.cache: dict[uuid.UUID, DocView | None] = {}

    async def get(self, document_id: uuid.UUID) -> DocView | None:
        if document_id not in self.cache:
            async with self.ctx.sessions() as session:
                self.cache[document_id] = await load_doc(
                    session, document_id, self.ctx.workspace_id
                )
        return self.cache[document_id]


def _copy(c: Any, *, source_finding: uuid.UUID) -> CitationDraft:
    return CitationDraft(
        role="supporting",
        document_id=c.document_id,
        document_filename=c.document_filename,
        chunk_id=c.chunk_id,
        table_row_id=c.table_row_id,
        quote=c.quote,
        char_start=c.char_start,
        char_end=c.char_end,
        page_number=c.page_number,
        bboxes=list(c.bbox_json),
        bbox_precision=c.bbox_precision,
        support_method=f"finding:{source_finding}",
        supported=True,
    )


async def _row_citation(docs: _Docs, subject: dict[str, Any]) -> CitationDraft | None:
    register = await docs.get(uuid.UUID(subject["document_id"]))
    if register is None:
        return None
    return span_citation(
        register,
        subject["char_start"],
        subject["char_end"],
        role="register_row",
        support_method="deterministic",
        supported=True,
        table_row_id=uuid.UUID(subject["row_id"]),
    )


async def _entity_citation(docs: _Docs, entity: ExtractedEntity) -> CitationDraft | None:
    doc = await docs.get(entity.document_id)
    if doc is None or entity.char_start is None or entity.char_end is None:
        return None
    if doc.slice(entity.char_start, entity.char_end) != entity.quote:
        return None
    return span_citation(
        doc,
        entity.char_start,
        entity.char_end,
        role="entity",
        support_method="deterministic",
        supported=True,
    )


def _step(name: str, detail: str, **refs: Any) -> dict[str, Any]:
    return {"step": name, "detail": detail, **({"refs": refs} if refs else {})}


async def reason(ctx: RunContext, state: dict[str, Any]) -> dict[str, Any]:
    tasks = [t for t in state.get("tasks", []) if t["kind"] == CROSS_DOCUMENT]
    if not tasks:
        return {"phase": "cross_done"}
    await ctx.store.emit(AGENT, "agent_start", {"tasks": len(tasks)})
    subjects = {s["subject_key"]: s for s in state.get("subjects", [])}
    links: dict[str, dict[str, Any]] = state.get("links", {})
    per_doc: dict[tuple[str, str], Finding] = {}
    for finding in await ctx.store.load_findings():
        if finding.kind == "per_document" and finding.document_id is not None:
            per_doc[(finding.rule_key, str(finding.document_id))] = finding
    docs = _Docs(ctx)
    entities: dict[str, list[ExtractedEntity]] = {}
    stale: dict[str, int] = {}
    for entity in await ctx.store.load_entities(["effective_date", "expiration_date", "term"]):
        # Facts are only reused if their quote is still exactly at its offsets: a source
        # that changed after extraction must not lend its new text to an old value.
        doc = await docs.get(entity.document_id)
        current = (
            doc.slice(entity.char_start, entity.char_end)
            if doc is not None and entity.char_start is not None and entity.char_end is not None
            else None
        )
        if entity.quote is None or current != entity.quote:
            stale[str(entity.document_id)] = stale.get(str(entity.document_id), 0) + 1
            continue
        entities.setdefault(str(entity.document_id), []).append(entity)
    out: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    for task in tasks:
        rule = ctx.rules[task["rule_key"]]
        spec = rule.get("cross_document") or {}
        kind = spec.get("kind")
        draft = FindingDraft(
            task_key=task["task_key"],
            kind=CROSS_DOCUMENT,
            rule_key=rule["key"],
            rule_id=ctx.rule_id(rule["key"]),
            rule_title=rule["title"],
            severity=rule["severity"],
            document_id=None,
            subject_key=task["subject_key"],
            subject_label=task["subject_label"],
            machine_status="needs_review",
            decided_by="system",
            rationale="",
        )
        citations: list[CitationDraft] = []
        steps: list[dict[str, Any]] = []
        await ctx.store.set_task(task["task_key"], "running")

        if task["subject_key"] == NO_REGISTER:
            draft.rationale = (
                "No vendor register (a spreadsheet with vendor and contract-file columns) is in "
                "the review scope, so this rule could not be evaluated."
            )
            steps.append(_step("register", draft.rationale))
        else:
            subject = subjects[task["subject_key"]]
            link = links.get(task["subject_key"], {"outcome": "unlinked"})
            row = await _row_citation(docs, subject)
            if row is not None:
                citations.append(row)
            values = subject["values"]
            steps.append(
                _step(
                    "register row",
                    f"Row {subject['row_number']} of {subject['register_filename']}: vendor "
                    f"{values.get('vendor') or '(blank)'}, data access "
                    f"{values.get('data_access') or '(blank)'}, contract "
                    f"{values.get('contract_file') or '(none listed)'}, renewal "
                    f"{values.get('renewal_date') or '(blank)'}.",
                )
            )
            flt = spec.get("subject_filter")
            linked_id = link.get("document_id")
            if link.get("outcome") == "linked":
                draft.document_id = uuid.UUID(linked_id)
                steps.append(
                    _step(
                        "contract match",
                        f"Matched {link['filename']} by {link['method']}"
                        + (f" (confidence {link['confidence']})" if link.get("confidence") else "")
                        + ".",
                        document_id=linked_id,
                    )
                )
            else:
                steps.append(
                    _step(
                        "contract match", f"{link.get('outcome')}: {link.get('reason', '')}".strip()
                    )
                )
            if row is None:
                draft.rationale = "The register row is no longer available; not evaluated."
            elif flt and not values.get(flt["column"]):
                draft.rationale = (
                    f"The register row has no {flt['column'].replace('_', ' ')} value, so it is "
                    "unknown whether this rule applies."
                )
            elif kind == "contract_on_file":
                outcome = link.get("outcome")
                if outcome == "linked":
                    draft.machine_status = "satisfied"
                    draft.rationale = (
                        f"{task['subject_label']} has its listed contract on file "
                        f"({link['filename']})."
                    )
                elif outcome == "missing":
                    draft.machine_status = "violated"
                    draft.rationale = (
                        f"The register lists {values.get('contract_file')} for "
                        f"{task['subject_label']}, but no such document is in the review scope."
                    )
                elif outcome == "no_contract_listed":
                    draft.machine_status = "violated"
                    draft.rationale = (
                        f"No contract is listed for {task['subject_label']} and no document "
                        "names this vendor."
                    )
                else:
                    draft.rationale = f"The contract match is ambiguous ({link.get('reason', 'several candidates')})."
            elif kind == "required_protections":
                outcome = link.get("outcome")
                if outcome in ("missing", "no_contract_listed"):
                    draft.machine_status = "violated"
                    draft.rationale = (
                        f"No NDA is on file for {task['subject_label']}, so the required "
                        "protections are not in place."
                    )
                elif outcome != "linked":
                    draft.rationale = "The contract match is ambiguous; protections not evaluated."
                else:
                    required = list(spec.get("required_rules", []))
                    found = [per_doc.get((key, str(linked_id))) for key in required]
                    states: list[tuple[str, str, Finding | None]] = []
                    for key, item in zip(required, found, strict=True):
                        req_rule = ctx.rules[key]
                        if item is None:
                            states.append((key, "missing", None))
                            continue
                        compliance = (
                            rule_compliance(req_rule, item.status)
                            if item.verified
                            else "needs_review"
                        )
                        states.append((key, compliance, item))
                        steps.append(
                            _step(
                                "required protection",
                                f"{req_rule['title']}: {item.status}"
                                + (" (verified)" if item.verified else " (not verified)")
                                + f" -> {compliance}.",
                                finding_id=str(item.id),
                            )
                        )
                    if all(c == "compliant" for _, c, _ in states):
                        draft.machine_status = "satisfied"
                        draft.rationale = (
                            f"The NDA for {task['subject_label']} has every required protection "
                            "(each finding verified)."
                        )
                        for _, _, src in states:
                            assert src is not None  # noqa: S101
                            citations += [
                                _copy(c, source_finding=src.id)
                                for c in src.citations
                                if c.role == "evidence"
                            ]
                    elif any(c == "non_compliant" for _, c, _ in states):
                        bad = [ctx.rules[k]["title"] for k, c, _ in states if c == "non_compliant"]
                        draft.machine_status = "violated"
                        draft.rationale = (
                            f"The NDA for {task['subject_label']} lacks: {', '.join(bad)} "
                            "(verified findings)."
                        )
                        for _, c, src in states:
                            if c == "non_compliant" and src is not None:
                                citations += [
                                    _copy(cit, source_finding=src.id)
                                    for cit in src.citations
                                    if cit.role == "evidence"
                                ]
                    else:
                        pending = [ctx.rules[k]["title"] for k, c, _ in states if c != "compliant"]
                        draft.rationale = (
                            "Some required protections could not be confirmed and need review: "
                            + ", ".join(pending)
                            + "."
                        )
            elif kind == "term_covers_date":
                date_role = spec.get("date_column") or "renewal_date"
                renewal = parse_register_date(subject["raw"].get(date_role)) or parse_register_date(
                    values.get(date_role, "")
                )
                if link.get("outcome") != "linked":
                    draft.rationale = "No matched contract, so its term could not be checked."
                elif stale.get(str(linked_id)):
                    draft.rationale = (
                        "The contract's text changed after its dates were extracted; the stale "
                        "facts were not used."
                    )
                elif renewal is None:
                    draft.rationale = "The register's renewal date is missing or unreadable."
                else:
                    mine = entities.get(str(linked_id), [])
                    expiry = {e.value_json.get("date") for e in mine if e.type == "expiration_date"}
                    effective = {
                        e.value_json.get("date") for e in mine if e.type == "effective_date"
                    }
                    terms = {e.value_json.get("months") for e in mine if e.type == "term"}
                    end: date | None = None
                    used: list[ExtractedEntity] = []
                    if len(expiry) == 1:
                        end = parse_iso(str(next(iter(expiry))))
                        used = [e for e in mine if e.type == "expiration_date"][:1]
                        steps.append(_step("contract end", f"Stated expiry date {end}."))
                    elif len(expiry) > 1:
                        draft.rationale = "The contract states conflicting expiry dates."
                    elif len(effective) == 1 and len(terms) == 1:
                        start = parse_iso(str(next(iter(effective))))
                        months = int(next(iter(terms)) or 0)
                        if start is not None and months > 0:
                            end = add_months(start, months)
                            used = [e for e in mine if e.type == "effective_date"][:1] + [
                                e for e in mine if e.type == "term"
                            ][:1]
                            steps.append(
                                _step(
                                    "contract end",
                                    f"Effective {start} plus a term of {months} months gives {end}.",
                                )
                            )
                    else:
                        draft.rationale = (
                            "The contract's end date could not be determined from verified "
                            "extracted dates (an explicit expiry date, or one effective date and "
                            "one term, is required)."
                        )
                    entity_cits = [await _entity_citation(docs, e) for e in used]
                    if end is not None and any(c is None for c in entity_cits):
                        end = None  # a date we cannot cite is not used
                        draft.rationale = "The extracted dates could not be cited exactly."
                    if end is not None:
                        citations += [c for c in entity_cits if c is not None]
                        steps.append(
                            _step("compare", f"Contract end {end} vs renewal date {renewal}.")
                        )
                        if end >= renewal:
                            draft.machine_status = "satisfied"
                            draft.rationale = f"The contract runs until {end}, on or after the renewal date {renewal}."
                        else:
                            draft.machine_status = "violated"
                            draft.rationale = (
                                f"The contract ends {end}, before the renewal date {renewal}."
                            )
            else:
                draft.rationale = f"Unknown cross-document rule kind {kind!r}."
                errors.append(
                    error_record(AGENT, "unknown_rule_kind", draft.rationale, task["task_key"])
                )
            if row is not None and not text_contains(row.quote, task["subject_label"]):
                steps.append(
                    _step("check", "The register row text does not contain the vendor name.")
                )
        draft.reasoning = steps
        draft.compliance = rule_compliance(rule, "needs_review")
        finding_id = await ctx.store.save_finding(draft, citations)
        out[task["task_key"]] = {
            "finding_id": str(finding_id),
            "kind": CROSS_DOCUMENT,
            "machine_status": draft.machine_status,
        }
    statuses: dict[str, int] = {}
    for ref in out.values():
        statuses[ref["machine_status"]] = statuses.get(ref["machine_status"], 0) + 1
    for err in errors:
        await ctx.store.emit(AGENT, "error", err)
    await ctx.store.emit(AGENT, "result", {"findings": len(out), "machine_status": statuses})
    return {"findings": out, "phase": "cross_done", "errors": errors}
