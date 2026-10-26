"""The evaluator may only create/drop databases it manages, never anything else."""

from __future__ import annotations

from collections.abc import AsyncIterator

import asyncpg
import pytest
from sqlalchemy.engine import URL, make_url

from compliance_core.config import Settings
from evals.retrieval import corpus

pytestmark = pytest.mark.integration


@pytest.fixture
def server(test_settings: Settings) -> URL:
    return make_url(test_settings.database_url)


def url_for(server: URL, name: str) -> str:
    return server.set(database=name).render_as_string(hide_password=False)


async def connect(server: URL, database: str = "postgres") -> asyncpg.Connection:
    return await asyncpg.connect(
        user=server.username,
        password=server.password,
        host=server.host,
        port=server.port,
        database=database,
    )


async def drop(server: URL, name: str) -> None:
    conn = await connect(server)
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await conn.close()


async def exists(server: URL, name: str) -> bool:
    conn = await connect(server)
    try:
        return bool(await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name))
    finally:
        await conn.close()


@pytest.mark.parametrize(
    "name",
    [
        "compliance_test",
        "compliance",
        "postgres",
        'compliance_eval"; DROP DATABASE compliance_test; --',
        "compliance_eval-x",
        "compliance_evalX",
        "compliance_eval_UPPER",
        "compliance_eval_" + "a" * 31,
        "compliance_eval_",
        "",
    ],
)
async def test_unsafe_names_are_refused_before_any_sql(
    name: str, monkeypatch: pytest.MonkeyPatch, server: URL
) -> None:
    url = url_for(server, name) if name else url_for(server, "x").rsplit("/", 1)[0] + "/"
    monkeypatch.setenv("EVAL_DATABASE_URL", url)
    with pytest.raises(corpus.IsolationError):
        corpus.eval_database_url()

    async def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("no connection may be opened for an unsafe name")

    monkeypatch.setattr(corpus, "_connect", boom)
    with pytest.raises(corpus.IsolationError):
        await corpus.recreate_database(url)  # validated inside the destructive function
    with pytest.raises(corpus.IsolationError):
        await corpus.stored_fingerprint(url)
    assert await exists(server, "compliance_test")


@pytest.fixture
async def unmanaged(server: URL) -> AsyncIterator[str]:
    name = "compliance_eval_pytest_unmanaged"
    await drop(server, name)
    conn = await connect(server)
    await conn.execute(f'CREATE DATABASE "{name}"')
    await conn.close()
    db = await connect(server, name)
    await db.execute("CREATE TABLE sentinel (x int); INSERT INTO sentinel VALUES (42)")
    await db.close()
    yield name
    await drop(server, name)


async def test_existing_unmanaged_database_is_never_dropped(unmanaged: str, server: URL) -> None:
    url = url_for(server, unmanaged)
    assert await corpus.stored_fingerprint(url) is None  # treated as foreign, not stale
    with pytest.raises(corpus.IsolationError, match="not created by this evaluator"):
        await corpus.recreate_database(url)
    db = await connect(server, unmanaged)
    try:
        assert await db.fetchval("SELECT x FROM sentinel") == 42
    finally:
        await db.close()


async def test_managed_database_lifecycle_and_lock(server: URL) -> None:
    name = "compliance_eval_pytest_managed"
    url = url_for(server, name)
    await drop(server, name)
    try:
        await corpus.recreate_database(url)  # creates, marks, migrates
        assert await corpus.stored_fingerprint(url) is None  # marked, not yet loaded
        await corpus.recreate_database(url)  # managed -> may be recreated
        async with corpus.EvalLock(url):
            with pytest.raises(corpus.IsolationError, match="another evaluation run"):
                async with corpus.EvalLock(url):
                    pass
        async with corpus.EvalLock(url):  # released after the first run ended
            pass
    finally:
        await drop(server, name)
