"""Deterministic compliance report writer.

The report is built only from presented findings (see ``services.reviews``):
- a finding may be stated as a fact only if it is verified (exact quote at the stored
  offsets *and* semantic support, re-checked on read) or decided by a human override;
- everything else is listed under "needs human review", never as compliant or violated;
- every summary statement carries the ids of the findings it is based on;
- recommended actions come from the rule pack's ``remediation`` text, not from a model.

No dates, clauses or quotes are generated here: quotes are copied from stored citations.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from compliance_core.db.models import ReviewRun
from compliance_core.review_schemas import (
    BudgetRead,
    FindingRead,
    ReportAction,
    ReportRead,
    ReportRisk,
    ReportRuleRow,
    ReportStatement,
    ReportSubjectRow,
    RulePackRef,
)

WRITER_VERSION = "report-writer@1.0.0"
SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
STATUSES = ("satisfied", "violated", "not_mentioned", "needs_review")
COMPLIANCE = ("compliant", "non_compliant", "needs_review", "not_assessed")


def _stated(f: FindingRead) -> bool:
    """May this finding be stated as a fact?"""
    return f.status != "needs_review" and (f.verified or f.decided_by == "human")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def build_report(
    run: ReviewRun,
    findings: list[FindingRead],
    budget: BudgetRead,
    *,
    generated_at: datetime | None = None,
) -> ReportRead:
    spec_rules: list[dict[str, Any]] = run.rule_pack_snapshot["spec"]["rules"]
    rules_by_key = {r["key"]: r for r in spec_rules}
    totals = run.totals_json
    stated = [f for f in findings if _stated(f)]
    review = [f for f in findings if not _stated(f)]
    non_compliant = [f for f in stated if f.compliance == "non_compliant"]
    compliant = [f for f in stated if f.compliance == "compliant"]

    def risk(f: FindingRead, rank: int) -> ReportRisk:
        return ReportRisk(
            finding_id=f.id,
            rank=rank,
            severity=f.severity,
            rule_title=f.rule_title,
            subject_label=f.subject_label,
            status=f.status,
            compliance=f.compliance,
            decided_by=f.decided_by,
        )

    order = sorted(
        non_compliant,
        key=lambda f: (SEVERITY_RANK[f.severity], f.rule_title, f.subject_label, str(f.id)),
    )
    risks = [risk(f, i) for i, f in enumerate(order, start=1)]
    review_sorted = sorted(
        review, key=lambda f: (SEVERITY_RANK[f.severity], f.rule_title, f.subject_label)
    )
    needs_review = [risk(f, i) for i, f in enumerate(review_sorted, start=1)]

    summary: list[ReportStatement] = []
    documents = len(run.config_json.get("document_ids", []))
    summary.append(
        ReportStatement(
            text=(
                f"Reviewed {_plural(documents, 'document')} against "
                f"{run.rule_pack_name} {run.rule_pack_version}: "
                f"{_plural(len(findings), 'finding')}, {len(stated)} verified or human-decided."
            ),
            finding_ids=[],
        )
    )
    if non_compliant:
        high = [f for f in non_compliant if f.severity == "high"]
        summary.append(
            ReportStatement(
                text=(
                    f"{_plural(len(non_compliant), 'finding')} "
                    f"{'is' if len(non_compliant) == 1 else 'are'} non-compliant "
                    f"({len(high)} high severity)."
                ),
                finding_ids=[f.id for f in order],
            )
        )
    if compliant:
        summary.append(
            ReportStatement(
                text=f"{_plural(len(compliant), 'finding')} verified as compliant.",
                finding_ids=[f.id for f in compliant],
            )
        )
    if review:
        summary.append(
            ReportStatement(
                text=(
                    f"{_plural(len(review), 'finding')} need human review: the evidence is "
                    "missing, could not be verified, or the decision was uncertain. These are "
                    "not counted as compliant or non-compliant."
                ),
                finding_ids=[f.id for f in review_sorted],
            )
        )
    by_subject_nc: dict[str, list[FindingRead]] = defaultdict(list)
    for f in order:
        by_subject_nc[f.subject_label].append(f)
    for subject, items in sorted(
        by_subject_nc.items(), key=lambda kv: (SEVERITY_RANK[kv[1][0].severity], kv[0])
    ):
        titles = sorted({f.rule_title for f in items})
        summary.append(
            ReportStatement(
                text=f"{subject}: non-compliant on {', '.join(titles)}.",
                finding_ids=[f.id for f in items],
            )
        )

    rule_rows = []
    for rule in spec_rules:
        mine = [f for f in findings if f.rule_key == rule["key"]]
        rule_rows.append(
            ReportRuleRow(
                rule_key=rule["key"],
                rule_title=rule["title"],
                severity=rule["severity"],
                scope=rule["scope"],
                counts={s: sum(f.status == s for f in mine) for s in STATUSES},
                compliance_counts={c: sum(f.compliance == c for f in mine) for c in COMPLIANCE},
            )
        )

    subjects: dict[str, list[FindingRead]] = defaultdict(list)
    for f in findings:
        subjects[f.subject_key].append(f)
    subject_rows = []
    for key, items in subjects.items():
        nc = [f for f in items if _stated(f) and f.compliance == "non_compliant"]
        nr = [f for f in items if not _stated(f)]
        ok = [f for f in items if _stated(f) and f.compliance == "compliant"]
        worst = min((SEVERITY_RANK[f.severity] for f in nc), default=None)
        status = (
            "at_risk" if nc else "needs_review" if nr else "compliant" if ok else "not_assessed"
        )
        subject_rows.append(
            ReportSubjectRow(
                subject_key=key,
                subject_label=items[0].subject_label,
                document_id=next((f.document_id for f in items if f.document_id), None),
                findings=len(items),
                violated=len(nc),
                needs_review=len(nr),
                compliant=len(ok),
                worst_severity=(("high", "medium", "low")[worst] if worst is not None else None),
                status=status,  # type: ignore[arg-type]
            )
        )
    subject_rows.sort(
        key=lambda r: (
            {"at_risk": 0, "needs_review": 1, "compliant": 2, "not_assessed": 3}[r.status],
            SEVERITY_RANK.get(r.worst_severity or "low", 3),
            r.subject_label,
        )
    )

    actions: list[ReportAction] = []
    by_rule: dict[str, list[FindingRead]] = defaultdict(list)
    for f in order:
        by_rule[f.rule_key].append(f)
    for rule_key, items in sorted(
        by_rule.items(), key=lambda kv: (SEVERITY_RANK[kv[1][0].severity], kv[0])
    ):
        rule = rules_by_key.get(rule_key, {})
        remediation = rule.get("remediation") or f"Remediate: {items[0].rule_title}."
        names = sorted({f.subject_label for f in items})
        shown = ", ".join(names[:8]) + (f" and {len(names) - 8} more" if len(names) > 8 else "")
        actions.append(
            ReportAction(
                finding_ids=[f.id for f in items],
                priority=items[0].severity,
                action=f"{remediation.strip()} Affects: {shown}.",
            )
        )
    if review:
        worst_review = min(SEVERITY_RANK[f.severity] for f in review)
        actions.append(
            ReportAction(
                finding_ids=[f.id for f in review_sorted],
                priority=("high", "medium", "low")[worst_review],
                action=(
                    f"Review the {_plural(len(review), 'finding')} flagged for human review "
                    "and record a decision with a comment."
                ),
            )
        )

    limitations = [
        "Findings are machine-generated first-pass results; human decisions are shown as such.",
    ]
    if run.llm_is_mock:
        limitations.append(
            "The LLM steps used the deterministic MOCK provider (lexical heuristics, no real "
            "model): its verdicts, extractions and support checks are not legal analysis."
        )
    if run.config_json.get("rule_check_backend") == "mock":
        limitations.append(
            "Model-supported rules used the MOCK rule-check backend, not the trained model."
        )
    if budget.usage_source in ("mock_estimate", "mixed"):
        limitations.append("Token usage of mock calls is an estimate (characters / 4), cost $0.")
    if totals.get("budget_exhausted"):
        limitations.append(
            "The run budget was exhausted; some checks were not performed and need review."
        )
    complete = run.status == "completed"
    if not complete:
        limitations.append(f"The run is {run.status}; this report may be incomplete.")

    return ReportRead(
        run_id=run.id,
        workspace_id=run.workspace_id,
        generated_at=generated_at or datetime.now(UTC),
        writer_version=WRITER_VERSION,
        run_status=run.status,  # type: ignore[arg-type]
        rule_pack=RulePackRef(
            id=run.rule_pack_id,
            key=run.rule_pack_key,
            name=run.rule_pack_name,
            version=run.rule_pack_version,
        ),
        llm_provider=run.llm_provider,
        llm_model=run.llm_model,
        llm_is_mock=run.llm_is_mock,
        rule_check_backend=str(run.config_json.get("rule_check_backend", "service")),
        model_version=totals.get("model_version"),
        tau=totals.get("tau"),
        prompt_versions=dict(totals.get("prompt_versions", {})),
        budget=budget,
        complete=complete,
        summary=summary,
        rules=rule_rows,
        subjects=subject_rows,
        risks=risks,
        actions=actions,
        needs_review=needs_review,
        findings=findings,
        limitations=limitations,
    )


def _cell(text: str) -> str:
    return " ".join(text.replace("|", "\\|").split())


def render_markdown(report: ReportRead) -> str:
    """Markdown export. Quotes are copied from stored, re-checked citations."""
    findings: dict[uuid.UUID, FindingRead] = {f.id: f for f in report.findings}
    lines = [
        f"# Compliance report: {_cell(report.rule_pack.name)} {report.rule_pack.version}",
        "",
        f"- Run: `{report.run_id}` ({report.run_status})",
        f"- Generated: {report.generated_at.isoformat(timespec='seconds')} by {report.writer_version}",
        f"- LLM: {report.llm_provider} / {report.llm_model}"
        + (" (MOCK)" if report.llm_is_mock else ""),
        f"- Rule-check model: {report.model_version or 'n/a'} (backend {report.rule_check_backend},"
        f" tau {report.tau if report.tau is not None else 'n/a'})",
        f"- Usage: {report.budget.tokens_used} tokens, ${report.budget.usd_used:.6f}"
        f" ({report.budget.usage_source}); caps {report.budget.tokens_cap} tokens /"
        f" ${report.budget.usd_cap:.2f}",
        "",
        "## Executive summary",
        "",
    ]
    lines += [f"- {_cell(s.text)}" for s in report.summary]
    lines += ["", "## Risk ranking (verified or human-decided)", ""]
    if report.risks:
        lines += ["| # | Severity | Rule | Subject | Decided by |", "|---|---|---|---|---|"]
        lines += [
            f"| {r.rank} | {r.severity} | {_cell(r.rule_title)} | {_cell(r.subject_label)} |"
            f" {r.decided_by} |"
            for r in report.risks
        ]
    else:
        lines.append("No verified non-compliant findings.")
    lines += ["", "## Recommended actions", ""]
    lines += [f"- **{a.priority}**: {_cell(a.action)}" for a in report.actions] or ["- None."]
    lines += [
        "",
        "## Per rule",
        "",
        "| Rule | Severity | Satisfied | Violated | Not mentioned |"
        " Needs review | Compliant | Non-compliant |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report.rules:
        c, k = row.counts, row.compliance_counts
        lines.append(
            f"| {_cell(row.rule_title)} | {row.severity} | {c['satisfied']} | {c['violated']} |"
            f" {c['not_mentioned']} | {c['needs_review']} | {k['compliant']} |"
            f" {k['non_compliant']} |"
        )
    lines += [
        "",
        "## Per subject",
        "",
        "| Subject | Status | Non-compliant | Needs review | Compliant |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {_cell(s.subject_label)} | {s.status} | {s.violated} | {s.needs_review} |"
        f" {s.compliant} |"
        for s in report.subjects
    ]
    lines += ["", "## Needs human review", ""]
    if report.needs_review:
        for r in report.needs_review:
            f = findings[r.finding_id]
            reason = "source changed after verification" if f.source_changed else _cell(f.rationale)
            lines.append(
                f"- {_cell(r.rule_title)} / {_cell(r.subject_label)} "
                f"(machine: {f.machine_status} by {f.machine_decided_by}): {reason}"
            )
    else:
        lines.append("None.")
    lines += ["", "## Findings and citations", ""]
    for f in report.findings:
        lines.append(
            f"### {_cell(f.rule_title)} / {_cell(f.subject_label)}: {f.status}"
            f" ({f.compliance}, decided by {f.decided_by})"
        )
        if f.override:
            lines.append(
                f"- Human override by {f.override.user_email or 'a former member'}: "
                f"{f.override.previous_status} -> {f.override.new_status}. "
                f"Comment: {_cell(f.override.comment)}"
            )
        for cit in f.citations:
            where = cit.document_filename + (f", p. {cit.page_number}" if cit.page_number else "")
            flag: str = "verified" if cit.verified and cit.current == "valid" else cit.current
            if not cit.verified:
                flag = "unverified"
            lines.append(f"- [{_cell(where)}; {flag}] “{_cell(cit.quote)[:600]}”")
        lines.append("")
    lines += ["## Limitations", ""]
    lines += [f"- {_cell(item)}" for item in report.limitations]
    return "\n".join(lines).rstrip() + "\n"
