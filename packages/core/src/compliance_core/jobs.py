"""Background job names shared by the API (producer) and worker (consumer)."""

from __future__ import annotations

import uuid

QUEUE_NAME = "compliance:jobs"
INGEST_DOCUMENT = "ingest_document"
SEED_DEMO = "seed_demo"


def ingest_job_id(document_id: uuid.UUID, generation: int) -> str:
    """One job id per (document, generation): re-uploads/retries bump the generation, so
    a stale queued job for an older generation is recognised and ignored."""
    return f"ingest:{document_id.hex}:{generation}"
