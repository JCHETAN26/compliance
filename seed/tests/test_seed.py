# mypy: disallow-untyped-calls=False
# (fixtures are built with PyMuPDF/python-docx, which ship partial annotations)
"""Seed generator and idempotent demo seeding (real ContractNLI archive + Postgres)."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import uuid
import zipfile
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from sqlalchemy import func, select

from compliance_core.config import AppEnv, Settings
from compliance_core.db.models import Document, User, Workspace
from compliance_core.security.passwords import hash_password, verify_password
from compliance_core.services import auth as auth_service
from compliance_core.services import workspaces as workspace_service
from compliance_ingestion.spreadsheet import extract_xlsx
from compliance_seed.generate import generate_seed_files, load_manifest
from compliance_seed.job import SeedRefusedError, SeedSummary, run_seed


def test_manifest_documents_come_from_the_official_test_split(dataset_zip: Path) -> None:
    manifest = load_manifest()
    with zipfile.ZipFile(dataset_zip) as archive:
        ids = {
            split: {
                int(d["id"])
                for d in json.loads(archive.read(f"contract-nli/{split}.json"))["documents"]
            }
            for split in ("train", "dev", "test")
        }
    used = [v.contractnli_id for v in manifest.vendors if v.contractnli_id is not None]
    assert len(used) == len(set(used)) >= 9
    assert set(used) <= ids["test"]
    assert not set(used) & (ids["train"] | ids["dev"])
    assert sum(v.source_format == "scanned_pdf" for v in manifest.vendors) >= 1
    assert any(v.contractnli_id is None for v in manifest.vendors), (
        "a vendor with a missing contract"
    )


def test_generation_is_byte_identical_and_scan_is_image_only(
    dataset_zip: Path, tmp_path: Path
) -> None:
    first = generate_seed_files(dataset_zip, tmp_path / "a")
    second = generate_seed_files(dataset_zip, tmp_path / "b")
    digest = {f.filename: hashlib.sha256(f.path.read_bytes()).hexdigest() for f in first}
    assert digest == {f.filename: hashlib.sha256(f.path.read_bytes()).hexdigest() for f in second}
    scanned = [f for f in first if f.filename.endswith("_scanned.pdf")]
    assert scanned
    with pymupdf.open(scanned[0].path) as doc:
        assert doc.page_count >= 1
        for page in doc:
            assert page.get_text().strip() == "", "no text layer"
            assert page.get_images(), "each page is an image"
        assert "CC BY 4.0" in doc.metadata["subject"]


def test_register_matches_manifest(dataset_zip: Path, tmp_path: Path) -> None:
    files = generate_seed_files(dataset_zip, tmp_path)
    register = next(f for f in files if f.filename == "vendor_register.xlsx")
    sheet = extract_xlsx(register.path, max_rows=100)
    vendors = sheet.sheets[0]
    assert vendors.columns == [
        "Vendor name",
        "Data access level",
        "Contract file name",
        "Renewal date",
    ]
    assert vendors.header_row_number == 3, "header found below the title row"
    manifest = load_manifest()
    assert vendors.row_count == len(manifest.vendors)
    provided = {f.filename for f in files}
    missing = [v.contract_file for v in manifest.vendors if v.contract_file not in provided]
    assert missing == ["northwind_analytics_nda.pdf"]


@pytest.mark.integration
async def test_seed_is_idempotent_and_never_resets_credentials(
    api_app: Any, clean_state: None, test_settings: Settings, dataset_zip: Path
) -> None:
    ctx = api_app.state.ctx
    demo = load_manifest().demo
    async with ctx.session_factory() as session:
        session.add(User(email=demo["email"], password_hash=hash_password("changed-by-user")))
        await session.commit()
    queued: list[uuid.UUID] = []

    async def enqueue(document_id: uuid.UUID, generation: int) -> None:
        assert generation == 1
        queued.append(document_id)

    first = await run_seed(test_settings, ctx.session_factory, ctx.storage, enqueue)
    assert not first.user_created
    assert first.workspace_created
    assert len(first.created) == 11
    assert first.already_present == []
    assert len(queued) == 11
    second = await run_seed(test_settings, ctx.session_factory, ctx.storage, enqueue)
    assert not second.workspace_created
    assert second.created == []
    assert len(second.already_present) == 11
    assert len(queued) == 11, "nothing re-queued"
    async with ctx.session_factory() as session:
        user = await session.scalar(select(User).where(User.email == demo["email"]))
        assert user is not None
        assert verify_password(user.password_hash, "changed-by-user")
        workspaces = await session.scalar(
            select(func.count()).select_from(Workspace).where(Workspace.owner_id == user.id)
        )
        documents = await session.scalar(select(func.count()).select_from(Document))
    assert workspaces == 1
    assert documents == 11


def _rendezvous(monkeypatch: pytest.MonkeyPatch, module: Any, name: str) -> None:
    """Hold each call until two callers arrived (at most 2 s), widening the race window.

    Without serialization both seeds reach the create step after seeing nothing; with it
    the second seed never gets there while the first waits, so the timeout releases it.
    """
    real = getattr(module, name)
    arrived = 0
    both = asyncio.Event()

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal arrived
        arrived += 1
        if arrived >= 2:
            both.set()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(both.wait(), 2.0)
        return await real(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)


@pytest.mark.integration
@pytest.mark.parametrize("user_exists", [False, True])
async def test_concurrent_seeds_create_one_demo_workspace(
    api_app: Any,
    clean_state: None,
    test_settings: Settings,
    dataset_zip: Path,
    user_exists: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The worker's automatic seed job and `python -m compliance_seed --wait` (CI, `make
    # seed`) can run at the same time on a clean start (hosted run 39866518635).
    _rendezvous(monkeypatch, auth_service, "register_user")
    _rendezvous(monkeypatch, workspace_service, "create_workspace")
    ctx = api_app.state.ctx
    demo = load_manifest().demo
    if user_exists:
        async with ctx.session_factory() as session:
            session.add(User(email=demo["email"], password_hash=hash_password(demo["password"])))
            await session.commit()
    queued: list[uuid.UUID] = []

    async def enqueue(document_id: uuid.UUID, generation: int) -> None:
        queued.append(document_id)

    # return_exceptions: both seeds finish before teardown truncates the tables.
    results = await asyncio.gather(
        *(run_seed(test_settings, ctx.session_factory, ctx.storage, enqueue) for _ in range(2)),
        return_exceptions=True,
    )
    summaries = [r for r in results if isinstance(r, SeedSummary)]
    assert len(summaries) == 2, results
    assert sorted(s.workspace_created for s in summaries) == [False, True]
    assert sorted(s.user_created for s in summaries) == (
        [False, False] if user_exists else [False, True]
    )
    assert summaries[0].workspace_id == summaries[1].workspace_id
    assert summaries[0].document_ids == summaries[1].document_ids
    assert len(queued) == len(set(queued)) == 11, "each new document queued exactly once"
    async with ctx.session_factory() as session:
        workspaces = await session.scalar(select(func.count()).select_from(Workspace))
        documents = await session.scalar(select(func.count()).select_from(Document))
        users = await session.scalar(select(func.count()).select_from(User))
    assert (users, workspaces, documents) == (1, 1, 11)


async def test_seed_refused_in_production(tmp_path: Path) -> None:
    settings = Settings(app_env=AppEnv.PRODUCTION, jwt_secret="k" * 40, cookie_secure=True)  # type: ignore[arg-type]

    async def enqueue(*_: Any) -> None:
        raise AssertionError("must not enqueue")

    with pytest.raises(SeedRefusedError, match="production"):
        await run_seed(settings, None, None, enqueue)  # type: ignore[arg-type]
