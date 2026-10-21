"""Migration tests against a real Postgres with pgvector."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest
from sqlalchemy import make_url, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from compliance_core.db import migrate

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {"users", "workspaces", "workspace_members", "alembic_version"}


@pytest.fixture
async def engine(migrated_database: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(migrated_database)
    yield engine
    await engine.dispose()


async def _tables(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        return {row[0] for row in rows}


async def test_head_schema_present(engine: AsyncEngine) -> None:
    assert await _tables(engine) >= EXPECTED_TABLES
    async with engine.connect() as conn:
        assert await migrate.current_revision(conn) == migrate.head_revision()
        vector = await conn.scalar(text("SELECT extname FROM pg_extension WHERE extname='vector'"))
        assert vector == "vector"
        # The extension is usable, not merely registered.
        distance = await conn.scalar(text("SELECT '[1,0,0]'::vector <-> '[0,1,0]'::vector"))
        assert distance == pytest.approx(2**0.5)


async def test_upgrade_downgrade_roundtrip_on_fresh_database(
    migrated_database: str, ensure_database: Callable[[str], Awaitable[None]]
) -> None:
    # A dedicated database so the shared test schema used by API tests is never dropped.
    url = make_url(migrated_database).set(database="compliance_test_migrations")
    url_str = url.render_as_string(hide_password=False)
    await ensure_database(url_str)
    engine = create_async_engine(url_str)
    try:
        await asyncio.to_thread(migrate.downgrade, url_str, "base")
        assert not (await _tables(engine)) & (EXPECTED_TABLES - {"alembic_version"})
        await asyncio.to_thread(migrate.upgrade, url_str, "head")
        assert await _tables(engine) >= EXPECTED_TABLES
        await asyncio.to_thread(migrate.downgrade, url_str, "base")
        assert not (await _tables(engine)) & (EXPECTED_TABLES - {"alembic_version"})
        async with engine.connect() as conn:
            enum_exists = await conn.scalar(
                text("SELECT 1 FROM pg_type WHERE typname = 'workspace_role'")
            )
            assert enum_exists is None
            assert await migrate.current_revision(conn) is None
        # Upgrading twice is a no-op, not an error.
        await asyncio.to_thread(migrate.upgrade, url_str, "head")
        await asyncio.to_thread(migrate.upgrade, url_str, "head")
        assert await _tables(engine) >= EXPECTED_TABLES
    finally:
        await engine.dispose()


async def test_database_constraints_enforced(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        trans = await conn.begin()
        try:
            with pytest.raises(IntegrityError, match="ck_users_email_lowercase"):
                async with conn.begin_nested():
                    await conn.execute(
                        text(
                            "INSERT INTO users (email, password_hash) VALUES ('Mixed@Case.io', 'h')"
                        )
                    )
            user_id = await conn.scalar(
                text("INSERT INTO users (email, password_hash) VALUES ('a@b.io', 'h') RETURNING id")
            )
            with pytest.raises(IntegrityError, match="uq_users_email"):
                async with conn.begin_nested():
                    await conn.execute(
                        text("INSERT INTO users (email, password_hash) VALUES ('a@b.io', 'h')")
                    )
            with pytest.raises(IntegrityError, match="ck_workspaces_name_length"):
                async with conn.begin_nested():
                    await conn.execute(
                        text("INSERT INTO workspaces (name, owner_id) VALUES ('   ', :owner)"),
                        {"owner": user_id},
                    )
            workspace_id = await conn.scalar(
                text("INSERT INTO workspaces (name, owner_id) VALUES ('W', :o) RETURNING id"),
                {"o": user_id},
            )
            with pytest.raises(Exception, match="invalid input value for enum workspace_role"):
                async with conn.begin_nested():
                    await conn.execute(
                        text(
                            "INSERT INTO workspace_members (workspace_id, user_id, role) "
                            "VALUES (:w, :u, 'admin')"
                        ),
                        {"w": workspace_id, "u": user_id},
                    )
            # Owners cannot be deleted while they own a workspace.
            with pytest.raises(IntegrityError, match="fk_workspaces_owner_id_users"):
                async with conn.begin_nested():
                    await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": user_id})
        finally:
            await trans.rollback()
