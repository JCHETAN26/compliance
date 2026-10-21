# mypy: disallow-untyped-calls=False
# (fixtures are built with PyMuPDF/python-docx, which ship partial annotations)
"""Seed generator and idempotent demo seeding (real ContractNLI archive + Postgres)."""

from __future__ import annotations

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
from compliance_ingestion.spreadsheet import extract_xlsx
from compliance_seed.generate import generate_seed_files, load_manifest
from compliance_seed.job import SeedRefusedError, run_seed


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


async def test_seed_refused_in_production(tmp_path: Path) -> None:
    settings = Settings(app_env=AppEnv.PRODUCTION, jwt_secret="k" * 40, cookie_secure=True)  # type: ignore[arg-type]

    async def enqueue(*_: Any) -> None:
        raise AssertionError("must not enqueue")

    with pytest.raises(SeedRefusedError, match="production"):
        await run_seed(settings, None, None, enqueue)  # type: ignore[arg-type]
