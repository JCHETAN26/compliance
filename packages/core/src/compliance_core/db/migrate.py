"""Programmatic Alembic entry point: ``python -m compliance_core.db.migrate upgrade``."""

from __future__ import annotations

import argparse
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(database_url: str | None = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    if database_url:
        # ConfigParser interpolation treats '%' specially.
        cfg.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return cfg


def head_revision() -> str:
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    if head is None:  # pragma: no cover - there is always at least one migration
        raise RuntimeError("no migrations found")
    return head


async def current_revision(connection: AsyncConnection) -> str | None:
    exists = await connection.scalar(text("SELECT to_regclass('public.alembic_version')"))
    if exists is None:
        return None
    result = await connection.scalar(text("SELECT version_num FROM alembic_version"))
    return str(result) if result is not None else None


def upgrade(database_url: str | None = None, revision: str = "head") -> None:
    command.upgrade(alembic_config(database_url), revision)


def downgrade(database_url: str | None = None, revision: str = "base") -> None:
    command.downgrade(alembic_config(database_url), revision)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run database migrations")
    sub = parser.add_subparsers(dest="action", required=True)
    up = sub.add_parser("upgrade")
    up.add_argument("revision", nargs="?", default="head")
    down = sub.add_parser("downgrade")
    down.add_argument("revision")
    sub.add_parser("head")
    args = parser.parse_args()
    if args.action == "upgrade":
        upgrade(revision=args.revision)
    elif args.action == "downgrade":
        downgrade(revision=args.revision)
    else:
        print(head_revision())  # noqa: T201


if __name__ == "__main__":
    main()
