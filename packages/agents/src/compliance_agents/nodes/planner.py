"""Planner / orchestrator agent.

Reads the run's rule-pack snapshot and the document inventory, finds the vendor register
(a spreadsheet whose header has the pack's ``register.columns`` aliases) and expands:
- per-document rules into (rule x document) tasks for documents of matching type;
- cross-document rules into (rule x register row) tasks, filtered by ``subject_filter``.
A row whose filter column is empty is kept and later marked for review (never silently
dropped). Without a register, each cross-document rule gets one task that ends in
``needs_review``.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select

from compliance_agents.context import RunContext
from compliance_agents.nodes.common import CROSS_DOCUMENT, PER_DOCUMENT
from compliance_agents.prompting import prompt_versions
from compliance_agents.state import ReviewState, TaskRef
from compliance_core.db.models import Chunk, Document, DocumentStatus, TableRow, TableSheet

AGENT = "planner"
NO_REGISTER = "register:none"


def _norm(text: str) -> str:
    return " ".join(str(text).lower().replace("_", " ").split())


def register_roles(columns: list[str], aliases: dict[str, list[str]]) -> dict[str, str]:
    """Map register roles (vendor, contract_file, ...) to sheet column names."""
    by_norm = {_norm(c): c for c in columns}
    roles: dict[str, str] = {}
    for role, names in aliases.items():
        for alias in names:
            if _norm(alias) in by_norm:
                roles[role] = by_norm[_norm(alias)]
                break
    return roles


async def plan(ctx: RunContext, state: ReviewState) -> dict[str, Any]:
    await ctx.store.emit(AGENT, "agent_start", {"rule_pack": ctx.spec["key"]})
    wanted = [uuid.UUID(d) for d in ctx.config.get("document_ids", [])]
    errors: list[dict[str, Any]] = []
    async with ctx.sessions() as session:
        docs = list(
            await session.scalars(
                select(Document)
                .where(Document.workspace_id == ctx.workspace_id, Document.id.in_(wanted))
                .order_by(Document.created_at, Document.id)
            )
        )
        ready = [d for d in docs if d.status is DocumentStatus.READY]
        for gone in sorted(set(wanted) - {d.id for d in ready}):
            errors.append(
                {
                    "agent": AGENT,
                    "code": "document_unavailable",
                    "message": "a selected document was deleted or is not ready; skipped",
                    "task_key": str(gone),
                }
            )
        documents = [
            {
                "id": str(d.id),
                "filename": d.filename,
                "doc_type": d.doc_type.value if d.doc_type else None,
                "file_format": d.file_format,
            }
            for d in ready
        ]

        subjects: list[dict[str, Any]] = []
        aliases = ctx.register_columns
        spreadsheets = [d for d in ready if d.file_format in ("xlsx", "csv")]
        if aliases and spreadsheets:
            sheets = list(
                await session.scalars(
                    select(TableSheet)
                    .where(TableSheet.document_id.in_([d.id for d in spreadsheets]))
                    .order_by(TableSheet.document_id, TableSheet.sheet_index)
                )
            )
            names = {d.id: d.filename for d in spreadsheets}
            for sheet in sheets:
                roles = register_roles(sheet.columns, aliases)
                if "vendor" not in roles or "contract_file" not in roles:
                    continue
                rows = list(
                    await session.scalars(
                        select(TableRow)
                        .where(
                            TableRow.document_id == sheet.document_id,
                            TableRow.sheet_index == sheet.sheet_index,
                        )
                        .order_by(TableRow.row_index)
                    )
                )
                chunk_ids = (
                    dict(
                        (
                            await session.execute(
                                select(Chunk.table_row_id, Chunk.id).where(
                                    Chunk.table_row_id.in_([r.id for r in rows])
                                )
                            )
                        ).all()
                    )
                    if rows
                    else {}
                )
                for row in rows:
                    cells = {c["column"]: c for c in row.data_json.get("cells", [])}
                    values = {
                        role: str(cells[col]["display"]).strip() if col in cells else ""
                        for role, col in roles.items()
                    }
                    raw = {
                        role: cells[col]["value"] if col in cells else None
                        for role, col in roles.items()
                    }
                    if not values.get("vendor") and not values.get("contract_file"):
                        continue
                    subjects.append(
                        {
                            "subject_key": f"row:{row.id}",
                            "label": values.get("vendor")
                            or f"{names[sheet.document_id]} {sheet.sheet_name} row {row.row_index}",
                            "document_id": str(sheet.document_id),
                            "register_filename": names[sheet.document_id],
                            "row_id": str(row.id),
                            "row_number": row.row_index,
                            "chunk_id": str(chunk_ids[row.id]) if row.id in chunk_ids else None,
                            "char_start": row.char_start,
                            "char_end": row.char_end,
                            "values": values,
                            "raw": raw,
                        }
                    )

    needs_model = any(r.get("model_supported") for r in ctx.spec["rules"])
    info = await ctx.model.info() if needs_model else None
    tau_override = ctx.config.get("tau_override")
    tau = tau_override if tau_override is not None else (info.tau if info else None)
    model_info = {
        "needed": needs_model,
        "available": bool(info and info.available),
        "backend": ctx.model.backend,
        "model_version": info.model_version if info else None,
        "variant": info.variant if info else None,
        "tau": tau,
        "tau_source": "workspace override" if tau_override is not None else "model artifact (DEV)",
        "reason": info.reason if info else None,
    }

    tasks: list[TaskRef] = []
    for rule in ctx.spec["rules"]:
        types = set(rule["applies_to_doc_types"])
        if rule["scope"] == PER_DOCUMENT:
            for d in documents:
                if d["doc_type"] in types:
                    tasks.append(
                        TaskRef(
                            task_key=f"pd|{rule['key']}|{d['id']}",
                            kind=PER_DOCUMENT,
                            rule_key=rule["key"],
                            document_id=d["id"],
                            subject_key=f"doc:{d['id']}",
                            subject_label=str(d["filename"]),
                            position=len(tasks),
                        )
                    )
            continue
        spec = rule.get("cross_document") or {}
        selected = subjects
        flt = spec.get("subject_filter")
        if flt:
            allowed = {_norm(v) for v in flt["in"]}
            selected = [
                s
                for s in subjects
                if not s["values"].get(flt["column"])
                or _norm(s["values"][flt["column"]]) in allowed
            ]
        if not subjects:
            tasks.append(
                TaskRef(
                    task_key=f"cd|{rule['key']}|{NO_REGISTER}",
                    kind=CROSS_DOCUMENT,
                    rule_key=rule["key"],
                    document_id=None,
                    subject_key=NO_REGISTER,
                    subject_label="Vendor register (not found)",
                    position=len(tasks),
                )
            )
            continue
        for s in selected:
            tasks.append(
                TaskRef(
                    task_key=f"cd|{rule['key']}|{s['subject_key']}",
                    kind=CROSS_DOCUMENT,
                    rule_key=rule["key"],
                    document_id=s["document_id"],
                    subject_key=s["subject_key"],
                    subject_label=s["label"],
                    position=len(tasks),
                )
            )

    await ctx.store.upsert_tasks([dict(t) for t in tasks])
    versions = prompt_versions()
    await ctx.store.merge_totals(
        {
            "model_version": model_info["model_version"],
            "tau": tau,
            "tau_source": model_info["tau_source"],
            "model_available": model_info["available"],
            "prompt_versions": versions,
            "planned_tasks": len(tasks),
        }
    )
    for err in errors:
        await ctx.store.emit(AGENT, "error", err)
    await ctx.store.emit(
        AGENT,
        "result",
        {
            "documents": len(documents),
            "register_rows": len(subjects),
            "tasks": len(tasks),
            "per_document_tasks": sum(t["kind"] == PER_DOCUMENT for t in tasks),
            "cross_document_tasks": sum(t["kind"] == CROSS_DOCUMENT for t in tasks),
            "model_available": model_info["available"],
            "model_version": model_info["model_version"],
            "tau": tau,
        },
    )
    return {
        "run_id": str(ctx.run_id),
        "workspace_id": str(ctx.workspace_id),
        "rule_pack": {
            "key": ctx.spec["key"],
            "name": ctx.spec["name"],
            "version": ctx.spec["version"],
        },
        "documents": documents,
        "subjects": subjects,
        "tasks": tasks,
        "model_info": model_info,
        "prompt_versions": versions,
        "errors": errors,
        "phase": "planned",
        "budget": {"exhausted": False},
    }
