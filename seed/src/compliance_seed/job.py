"""Idempotent demo seeding: demo user, demo workspace, vendor register and NDAs.

Re-running never resets anything a person may have changed:
- an existing demo user keeps its password (only a missing user is created);
- an existing demo workspace is reused; missing seed files are added, and files already
  present (same SHA-256 in the workspace) are left alone;
- seeding is refused in production.

Automatic seeding at worker start only happens when the demo workspace does not exist
yet (a clean install). ``make seed`` re-runs it explicitly.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from compliance_core.config import AppEnv, Settings
from compliance_core.db.models import User, Workspace
from compliance_core.downloads import download_verified, sha256_file
from compliance_core.files import ZipLimits, detect_format
from compliance_core.jobs import INGEST_DOCUMENT, QUEUE_NAME, ingest_job_id
from compliance_core.schemas import RegisterRequest, WorkspaceCreate
from compliance_core.services import auth as auth_service
from compliance_core.services import documents as document_service
from compliance_core.services import workspaces as workspace_service
from compliance_core.storage import LocalStorage
from compliance_seed.generate import generate_seed_files, load_manifest

logger = logging.getLogger("compliance_seed")
Enqueue = Callable[[uuid.UUID, int], Awaitable[None]]


class SeedRefusedError(RuntimeError):
    pass


@dataclass
class SeedSummary:
    user_created: bool = False
    workspace_created: bool = False
    workspace_id: uuid.UUID | None = None
    created: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)
    document_ids: list[uuid.UUID] = field(default_factory=list)


def dataset_path(settings: Settings) -> Path:
    return settings.dataset_cache_dir / "contractnli" / "contract-nli.zip"


def ensure_dataset(settings: Settings, *, force_download: bool = False) -> Path:
    """Return the verified ContractNLI archive, downloading it if allowed."""
    manifest = load_manifest()
    target = dataset_path(settings)
    if not target.exists() and not (settings.artifact_auto_download or force_download):
        raise SeedRefusedError(
            f"ContractNLI archive not found at {target}. Run `make fetch-artifacts` first "
            "(or allow ARTIFACT_AUTO_DOWNLOAD)."
        )
    return download_verified(
        manifest.dataset["url"],
        target,
        sha256=manifest.dataset["sha256"],
        max_bytes=int(manifest.dataset["max_bytes"]),
    )


async def demo_workspace_exists(session: AsyncSession) -> bool:
    demo = load_manifest().demo
    found = await session.scalar(
        select(Workspace.id)
        .join(User, User.id == Workspace.owner_id)
        .where(User.email == demo["email"], Workspace.name == demo["workspace"])
    )
    return found is not None


async def run_seed(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    storage: LocalStorage,
    enqueue: Enqueue,
) -> SeedSummary:
    if settings.app_env is AppEnv.PRODUCTION:
        raise SeedRefusedError("Demo seeding is disabled in production.")
    manifest = load_manifest()
    demo = manifest.demo
    archive = await asyncio.to_thread(ensure_dataset, settings)
    summary = SeedSummary()
    zip_limits = ZipLimits(
        max_entries=settings.max_zip_entries,
        max_uncompressed_bytes=settings.max_zip_uncompressed_bytes,
        max_ratio=settings.max_zip_compression_ratio,
    )
    with tempfile.TemporaryDirectory(prefix="seed-") as tmp:
        files = await asyncio.to_thread(generate_seed_files, archive, Path(tmp))
        async with session_factory() as session:
            user = await session.scalar(select(User).where(User.email == demo["email"]))
            if user is None:
                created_user = await auth_service.register_user(
                    session, RegisterRequest(email=demo["email"], password=demo["password"])
                )
                user_id = created_user.id
                summary.user_created = True
            else:
                user_id = user.id  # never touch an existing password
            workspace_id = await session.scalar(
                select(Workspace.id).where(
                    Workspace.owner_id == user_id, Workspace.name == demo["workspace"]
                )
            )
            if workspace_id is None:
                workspace = await workspace_service.create_workspace(
                    session, user_id, WorkspaceCreate(name=demo["workspace"])
                )
                workspace_id = workspace.id
                summary.workspace_created = True
            summary.workspace_id = workspace_id
            for item in files:
                # Uploads go through the same validation and registration path as the API.
                fmt = await asyncio.to_thread(detect_format, item.path, item.filename, zip_limits)
                digest = await asyncio.to_thread(sha256_file, item.path)
                registered = await document_service.register_upload(
                    session,
                    storage,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    filename=item.filename,
                    fmt=fmt,
                    staged_path=item.path,
                    sha256=digest,
                    size=item.path.stat().st_size,
                )
                summary.document_ids.append(registered.document.id)
                if registered.created:
                    summary.created.append(item.filename)
                    await enqueue(registered.document.id, 1)
                else:
                    summary.already_present.append(item.filename)
    logger.info(
        "demo seed complete",
        extra={
            "workspace_id": str(summary.workspace_id),
            "files_added": len(summary.created),
            "files_present": len(summary.already_present),
            "user_created": summary.user_created,
        },
    )
    return summary


async def seed_demo(ctx: dict[str, Any]) -> dict[str, Any]:
    """arq job wrapper around ``run_seed`` using the worker's resources."""
    deps = ctx["ingest"]
    redis = ctx["redis"]

    async def enqueue(document_id: uuid.UUID, generation: int) -> None:
        await redis.enqueue_job(
            INGEST_DOCUMENT,
            str(document_id),
            generation,
            _job_id=ingest_job_id(document_id, generation),
            _queue_name=QUEUE_NAME,
        )

    summary = await run_seed(deps.settings, deps.session_factory, deps.storage, enqueue)
    return {
        "workspace_id": str(summary.workspace_id),
        "created": summary.created,
        "already_present": summary.already_present,
        "user_created": summary.user_created,
    }
