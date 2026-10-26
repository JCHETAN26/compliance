"""Background job names shared by the API (producer) and worker (consumer)."""

from __future__ import annotations

import uuid

QUEUE_NAME = "compliance:jobs"
INGEST_DOCUMENT = "ingest_document"
SEED_DEMO = "seed_demo"
RUN_REVIEW = "run_review"


def ingest_job_id(document_id: uuid.UUID, generation: int) -> str:
    """One job id per (document, generation): re-uploads/retries bump the generation, so
    a stale queued job for an older generation is recognised and ignored."""
    return f"ingest:{document_id.hex}:{generation}"


def review_job_id(run_id: uuid.UUID, generation: int, suffix: str = "") -> str:
    """One job id per (run, generation): pause/fail -> resume bumps the generation. The
    run's lease makes any duplicate job a harmless no-op; ``suffix`` lets the stale-run
    sweeper re-enqueue after a crash while an old job id is still remembered by arq."""
    return f"review:{run_id.hex}:{generation}{':' + suffix if suffix else ''}"
