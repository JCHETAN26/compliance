"""Rule packs: built-in (global, read-only) and workspace packs (owner/reviewer editable).

Authorisation: any signed-in user may read built-in packs; workspace packs are visible to
members only (others get "not found"). Creating, importing and updating workspace packs
requires the owner or reviewer role. Review runs snapshot the rules they check, so
editing a pack never changes an existing run's results.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from compliance_core.db.models import Rule, RulePack, WorkspaceMember, WorkspaceRole
from compliance_core.review_schemas import RulePackRead, RulePackSummary, RuleRead
from compliance_core.rule_packs import (
    RulePackSpec,
    content_sha256,
    dump_yaml,
    load_builtin_specs,
    validate_spec,
)
from compliance_core.services.errors import (
    ConflictError,
    InvalidOperationError,
    NotFoundError,
    PermissionDeniedError,
)
from compliance_core.services.workspaces import require_role

logger = logging.getLogger(__name__)
WRITE_ROLES = frozenset({WorkspaceRole.OWNER, WorkspaceRole.REVIEWER})
PACK_NOT_FOUND = "Rule pack not found."
BUILTIN_SYNC_LOCK_ID = 7_362_918_402


def _rules(spec: RulePackSpec) -> list[Rule]:
    return [
        Rule(
            position=i,
            key=r.key,
            title=r.title,
            description=r.description,
            hypothesis_text=r.hypothesis_text,
            scope=r.scope,
            applies_to_doc_types=list(r.applies_to_doc_types),
            severity=r.severity,
            model_supported=r.model_supported,
            contractnli_key=r.contractnli_key,
            compliant_statuses=list(r.compliant_statuses)
            if r.compliant_statuses is not None
            else None,
            cross_document=r.cross_document.model_dump(mode="json", by_alias=True)
            if r.cross_document
            else None,
            remediation=r.remediation,
        )
        for i, r in enumerate(spec.rules)
    ]


def _apply(pack: RulePack, spec: RulePackSpec) -> None:
    pack.key = spec.key
    pack.name = spec.name
    pack.version = spec.version
    pack.description = spec.description
    pack.content_sha256 = content_sha256(spec)
    pack.register_json = (
        spec.register_.model_dump(mode="json", by_alias=True) if spec.register_ else None
    )
    pack.provenance_json = dict(spec.source) if spec.source else None


def pack_spec(pack: RulePack) -> RulePackSpec:
    """Rebuild the validated spec from stored rows (export and run snapshots)."""
    document: dict[str, Any] = {
        "key": pack.key,
        "name": pack.name,
        "version": pack.version,
        "description": pack.description,
        "rules": [
            {
                k: v
                for k, v in {
                    "key": r.key,
                    "title": r.title,
                    "description": r.description,
                    "hypothesis_text": r.hypothesis_text,
                    "scope": r.scope,
                    "applies_to_doc_types": r.applies_to_doc_types,
                    "severity": r.severity,
                    "model_supported": r.model_supported,
                    "contractnli_key": r.contractnli_key,
                    "compliant_statuses": r.compliant_statuses,
                    "cross_document": r.cross_document,
                    "remediation": r.remediation,
                }.items()
                if v is not None
            }
            for r in pack.rules
        ],
    }
    if pack.register_json is not None:
        document["register"] = pack.register_json
    if pack.provenance_json is not None:
        document["source"] = pack.provenance_json
    return validate_spec(document)


def _summary(pack: RulePack, role: WorkspaceRole | None) -> dict[str, Any]:
    return {
        "id": pack.id,
        "key": pack.key,
        "name": pack.name,
        "version": pack.version,
        "description": pack.description,
        "builtin": pack.workspace_id is None,
        "workspace_id": pack.workspace_id,
        "source": pack.source,
        "rule_count": len(pack.rules),
        "model_supported_count": sum(r.model_supported for r in pack.rules),
        "cross_document_count": sum(r.scope == "cross_document" for r in pack.rules),
        "content_sha256": pack.content_sha256,
        "created_at": pack.created_at,
        "updated_at": pack.updated_at,
        "editable": pack.workspace_id is not None and role in WRITE_ROLES,
    }


def _read(pack: RulePack, role: WorkspaceRole | None) -> RulePackRead:
    return RulePackRead(
        **_summary(pack, role),
        rules=[
            RuleRead(
                id=r.id,
                key=r.key,
                title=r.title,
                description=r.description,
                hypothesis_text=r.hypothesis_text,
                scope=r.scope,  # type: ignore[arg-type]
                applies_to_doc_types=r.applies_to_doc_types,
                severity=r.severity,  # type: ignore[arg-type]
                model_supported=r.model_supported,
                contractnli_key=r.contractnli_key,
                compliant_statuses=r.compliant_statuses,
                cross_document=r.cross_document,
                remediation=r.remediation,
            )
            for r in pack.rules
        ],
        register_=pack.register_json,
        provenance=pack.provenance_json,
    )


async def sync_builtin_packs(session: AsyncSession) -> int:
    """Insert or refresh the built-in packs from ``rule_packs/*.yaml`` (idempotent and
    serialised across processes). Returns the number of packs written."""
    specs = load_builtin_specs()
    await session.execute(
        text("SELECT pg_advisory_xact_lock(:lock_id)"), {"lock_id": BUILTIN_SYNC_LOCK_ID}
    )
    written = 0
    for spec in specs:
        existing = await session.scalar(
            select(RulePack)
            .options(selectinload(RulePack.rules))
            .where(
                RulePack.workspace_id.is_(None),
                RulePack.key == spec.key,
                RulePack.version == spec.version,
            )
        )
        sha = content_sha256(spec)
        if existing is not None and existing.content_sha256 == sha:
            continue
        if existing is None:
            pack = RulePack(workspace_id=None, source="builtin")
            _apply(pack, spec)
            pack.rules = _rules(spec)
            session.add(pack)
        else:
            logger.warning(
                "built-in rule pack content changed without a version bump; refreshing",
                extra={"rule_pack": spec.key, "version": spec.version},
            )
            await session.execute(delete(Rule).where(Rule.rule_pack_id == existing.id))
            _apply(existing, spec)
            existing.updated_at = func.now()
            await session.flush()
            existing.rules = _rules(spec)
        written += 1
    await session.commit()
    return written


async def _role(
    session: AsyncSession, workspace_id: uuid.UUID, user_id: uuid.UUID
) -> WorkspaceRole | None:
    return await session.scalar(
        select(WorkspaceMember.role).where(
            WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == user_id
        )
    )


async def load_pack(
    session: AsyncSession, pack_id: uuid.UUID, user_id: uuid.UUID, *, lock: bool = False
) -> tuple[RulePack, WorkspaceRole | None]:
    query = select(RulePack).options(selectinload(RulePack.rules)).where(RulePack.id == pack_id)
    if lock:
        query = query.with_for_update(of=RulePack)
    pack = await session.scalar(query)
    if pack is None:
        raise NotFoundError(PACK_NOT_FOUND)
    if pack.workspace_id is None:
        return pack, None
    role = await _role(session, pack.workspace_id, user_id)
    if role is None:
        raise NotFoundError(PACK_NOT_FOUND)
    return pack, role


async def list_packs(
    session: AsyncSession, user_id: uuid.UUID, workspace_id: uuid.UUID
) -> list[RulePackSummary]:
    role = await require_role(
        session, workspace_id, user_id, frozenset(WorkspaceRole), "view rule packs"
    )
    packs = await session.scalars(
        select(RulePack)
        .options(selectinload(RulePack.rules))
        .where((RulePack.workspace_id.is_(None)) | (RulePack.workspace_id == workspace_id))
        .order_by(RulePack.workspace_id.is_not(None), RulePack.key, RulePack.version.desc())
    )
    return [RulePackSummary(**_summary(p, role)) for p in packs]


async def get_pack(session: AsyncSession, pack_id: uuid.UUID, user_id: uuid.UUID) -> RulePackRead:
    pack, role = await load_pack(session, pack_id, user_id)
    return _read(pack, role)


async def create_pack(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    spec: RulePackSpec,
    source: str,
) -> RulePackRead:
    await require_role(session, workspace_id, user_id, WRITE_ROLES, "create rule packs")
    spec = validate_spec(spec.model_dump(mode="json", by_alias=True, exclude_none=True))
    pack = RulePack(workspace_id=workspace_id, source=source, created_by=user_id)
    _apply(pack, spec)
    pack.rules = _rules(spec)
    session.add(pack)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        raise ConflictError(
            f"This workspace already has rule pack {spec.key!r} version {spec.version}."
        ) from exc
    await session.commit()
    created, role = await load_pack(session, pack.id, user_id)
    return _read(created, role)


async def update_pack(
    session: AsyncSession, pack_id: uuid.UUID, user_id: uuid.UUID, spec: RulePackSpec
) -> RulePackRead:
    pack, role = await load_pack(session, pack_id, user_id, lock=True)
    if pack.workspace_id is None:
        raise PermissionDeniedError(
            "Built-in rule packs are read-only. Import a copy into the workspace to edit it."
        )
    if role not in WRITE_ROLES:
        raise PermissionDeniedError(
            f"Your role ({role.value if role else 'none'}) does not allow you to edit rule packs."
        )
    spec = validate_spec(spec.model_dump(mode="json", by_alias=True, exclude_none=True))
    if spec.key != pack.key:
        raise InvalidOperationError("A rule pack's key cannot be changed; import a new pack.")
    await session.execute(delete(Rule).where(Rule.rule_pack_id == pack.id))
    _apply(pack, spec)
    pack.updated_at = func.now()
    try:
        await session.flush()
        pack.rules = _rules(spec)
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        raise ConflictError(
            f"This workspace already has rule pack {spec.key!r} version {spec.version}."
        ) from exc
    await session.commit()
    session.expire_all()
    pack, role = await load_pack(session, pack_id, user_id)
    return _read(pack, role)


async def export_pack(
    session: AsyncSession, pack_id: uuid.UUID, user_id: uuid.UUID
) -> tuple[str, str]:
    pack, _ = await load_pack(session, pack_id, user_id)
    return f"{pack.key}-{pack.version}.yaml", dump_yaml(pack_spec(pack))
