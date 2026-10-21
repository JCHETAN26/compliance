"""``python -m compliance_seed [--wait SECONDS]`` (what ``make seed`` runs in the worker).

Seeds the demo workspace (idempotent) and queues ingestion for new files. With
``--wait`` it then polls until every seeded document is ``ready`` or ``failed`` and
exits non-zero if any failed or the wait timed out.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid

from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy import select

from compliance_core.config import get_settings
from compliance_core.db.models import Document, DocumentStatus
from compliance_core.db.session import create_engine, create_session_factory
from compliance_core.jobs import INGEST_DOCUMENT, QUEUE_NAME, ingest_job_id
from compliance_core.storage import LocalStorage
from compliance_seed.job import SeedRefusedError, run_seed


async def _main(wait_seconds: int) -> int:
    settings = get_settings()
    engine = create_engine(settings)
    factory = create_session_factory(engine)
    queue = await create_pool(
        RedisSettings.from_dsn(settings.redis_url), default_queue_name=QUEUE_NAME
    )

    async def enqueue(document_id: uuid.UUID, generation: int) -> None:
        await queue.enqueue_job(
            INGEST_DOCUMENT,
            str(document_id),
            generation,
            _job_id=ingest_job_id(document_id, generation),
            _queue_name=QUEUE_NAME,
        )

    try:
        summary = await run_seed(settings, factory, LocalStorage(settings.storage_dir), enqueue)
        print(
            f"demo user {'created' if summary.user_created else 'already existed (password unchanged)'}; "
            f"workspace {summary.workspace_id} {'created' if summary.workspace_created else 'reused'}; "
            f"{len(summary.created)} file(s) added, {len(summary.already_present)} already present"
        )
        if wait_seconds <= 0:
            return 0
        deadline = time.monotonic() + wait_seconds
        while True:
            async with factory() as session:
                rows = (
                    await session.execute(
                        select(Document.filename, Document.status, Document.error).where(
                            Document.id.in_(summary.document_ids)
                        )
                    )
                ).all()
            pending = [
                r for r in rows if r.status in (DocumentStatus.UPLOADED, DocumentStatus.PROCESSING)
            ]
            if not pending:
                failed = [r for r in rows if r.status is DocumentStatus.FAILED]
                for row in sorted(rows, key=lambda r: r.filename):
                    print(
                        f"  {row.status.value:<10} {row.filename}"
                        + (f"  ({row.error})" if row.error else "")
                    )
                return 1 if failed else 0
            if time.monotonic() > deadline:
                print(
                    f"timed out waiting for {len(pending)} document(s) to finish", file=sys.stderr
                )
                return 1
            await asyncio.sleep(2)
    except SeedRefusedError as exc:
        print(f"seed refused: {exc}", file=sys.stderr)
        return 2
    finally:
        await queue.aclose()
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", type=int, default=0, help="seconds to wait for ingestion")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main(args.wait)))


if __name__ == "__main__":
    main()
