"""LangGraph checkpoint saver on the application's Postgres database.

Storage mirrors LangGraph's reference ``InMemorySaver`` (checkpoint rows, per-channel
versioned blobs, and pending task writes), so a run resumes exactly where it stopped:
completed tasks of an interrupted super-step are restored from their pending writes and
are not executed again.

The thread id is the review run id; rows cascade with the run. Values are serialized by
LangGraph's own serializer and hold only ids, statuses and small summaries (document
text, quotes and credentials are never part of graph state).

Writes are fenced: each write transaction re-checks the run's lease token under the run row
lock (taken first, like every other worker write), so a worker that lost its lease cannot overwrite the new owner's checkpoints.
Only the async API is implemented (the graph is always run with ``astream``/``ainvoke``).
"""

from __future__ import annotations

import random
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
    writes_sort_key,
)
from sqlalchemy import delete, select, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from compliance_core.db.models import (
    GraphCheckpoint,
    GraphCheckpointBlob,
    GraphCheckpointWrite,
    ReviewRun,
)
from compliance_core.services.review_events import LeaseLostError


class PostgresCheckpointSaver(BaseCheckpointSaver[str]):
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        lease_token: uuid.UUID | None = None,
    ) -> None:
        super().__init__()
        self._sessions = session_factory
        self._lease_token = lease_token

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _ids(config: RunnableConfig) -> tuple[uuid.UUID, str]:
        configurable = config["configurable"]
        return uuid.UUID(str(configurable["thread_id"])), str(configurable.get("checkpoint_ns", ""))

    async def _fence(self, session: AsyncSession, run_id: uuid.UUID) -> None:
        if self._lease_token is None:
            return
        holder = await session.scalar(
            select(ReviewRun.lease_token).where(ReviewRun.id == run_id).with_for_update()
        )
        if holder != self._lease_token:
            raise LeaseLostError(f"lease on run {run_id} is no longer held")

    async def _tuple(
        self, session: AsyncSession, row: GraphCheckpoint, config: RunnableConfig | None = None
    ) -> CheckpointTuple:
        checkpoint: Checkpoint = self.serde.loads_typed((row.checkpoint_type, row.checkpoint))
        versions = checkpoint.get("channel_versions", {})
        values: dict[str, Any] = {}
        if versions:
            blobs = await session.scalars(
                select(GraphCheckpointBlob).where(
                    GraphCheckpointBlob.run_id == row.run_id,
                    GraphCheckpointBlob.checkpoint_ns == row.checkpoint_ns,
                    tuple_(GraphCheckpointBlob.channel, GraphCheckpointBlob.version).in_(
                        [(channel, str(version)) for channel, version in versions.items()]
                    ),
                )
            )
            for blob in blobs:
                if blob.value_type == "empty":
                    continue
                values[blob.channel] = self.serde.loads_typed((blob.value_type, blob.value))
        writes = list(
            await session.scalars(
                select(GraphCheckpointWrite).where(
                    GraphCheckpointWrite.run_id == row.run_id,
                    GraphCheckpointWrite.checkpoint_ns == row.checkpoint_ns,
                    GraphCheckpointWrite.checkpoint_id == row.checkpoint_id,
                )
            )
        )
        writes.sort(key=lambda w: writes_sort_key(w.task_path, w.task_id, w.idx))
        thread = str(row.run_id)
        own = {
            "configurable": {
                "thread_id": thread,
                "checkpoint_ns": row.checkpoint_ns,
                "checkpoint_id": row.checkpoint_id,
            }
        }
        return CheckpointTuple(
            config=config if config and get_checkpoint_id(config) else own,  # type: ignore[arg-type]
            checkpoint={**checkpoint, "channel_values": values},
            metadata=self.serde.loads_typed((row.metadata_type, row.metadata_blob)),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread,
                        "checkpoint_ns": row.checkpoint_ns,
                        "checkpoint_id": row.parent_checkpoint_id,
                    }
                }
                if row.parent_checkpoint_id
                else None
            ),
            pending_writes=[
                (w.task_id, w.channel, self.serde.loads_typed((w.value_type, w.value)))
                for w in writes
            ],
        )

    # ------------------------------------------------------------------ async API

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        run_id, ns = self._ids(config)
        query = select(GraphCheckpoint).where(
            GraphCheckpoint.run_id == run_id, GraphCheckpoint.checkpoint_ns == ns
        )
        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id:
            query = query.where(GraphCheckpoint.checkpoint_id == checkpoint_id)
        else:
            query = query.order_by(GraphCheckpoint.checkpoint_id.desc()).limit(1)
        async with self._sessions() as session:
            row = await session.scalar(query)
            if row is None:
                return None
            return await self._tuple(session, row, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        query = select(GraphCheckpoint)
        if config is not None:
            run_id, ns = self._ids(config)
            query = query.where(GraphCheckpoint.run_id == run_id)
            if "checkpoint_ns" in config["configurable"]:
                query = query.where(GraphCheckpoint.checkpoint_ns == ns)
            if checkpoint_id := get_checkpoint_id(config):
                query = query.where(GraphCheckpoint.checkpoint_id == checkpoint_id)
        if before is not None and (before_id := get_checkpoint_id(before)):
            query = query.where(GraphCheckpoint.checkpoint_id < before_id)
        query = query.order_by(GraphCheckpoint.checkpoint_id.desc())
        remaining = limit
        async with self._sessions() as session:
            rows = list(await session.scalars(query))
            for row in rows:
                if remaining is not None and remaining <= 0:
                    break
                item = await self._tuple(session, row)
                if filter and not all(item.metadata.get(k) == v for k, v in filter.items()):
                    continue
                if remaining is not None:
                    remaining -= 1
                yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        run_id, ns = self._ids(config)
        stored = checkpoint.copy()
        values: dict[str, Any] = stored.pop("channel_values")  # type: ignore[misc]
        ctype, cblob = self.serde.dumps_typed(stored)
        mtype, mblob = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        async with self._sessions() as session, session.begin():
            await self._fence(session, run_id)
            for channel, version in new_versions.items():
                vtype, vblob = (
                    self.serde.dumps_typed(values[channel]) if channel in values else ("empty", b"")
                )
                await session.execute(
                    insert(GraphCheckpointBlob)
                    .values(
                        run_id=run_id,
                        checkpoint_ns=ns,
                        channel=channel,
                        version=str(version),
                        value_type=vtype,
                        value=vblob,
                    )
                    .on_conflict_do_nothing()
                )
            row = {
                "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
                "checkpoint_type": ctype,
                "checkpoint": cblob,
                "metadata_type": mtype,
                "metadata_blob": mblob,
            }
            await session.execute(
                insert(GraphCheckpoint)
                .values(run_id=run_id, checkpoint_ns=ns, checkpoint_id=checkpoint["id"], **row)
                .on_conflict_do_update(
                    index_elements=["run_id", "checkpoint_ns", "checkpoint_id"], set_=row
                )
            )
        return {
            "configurable": {
                "thread_id": str(run_id),
                "checkpoint_ns": ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        run_id, ns = self._ids(config)
        checkpoint_id = str(config["configurable"]["checkpoint_id"])
        async with self._sessions() as session, session.begin():
            await self._fence(session, run_id)
            for i, (channel, value) in enumerate(writes):
                idx = WRITES_IDX_MAP.get(channel, i)
                vtype, vblob = self.serde.dumps_typed(value)
                stmt = insert(GraphCheckpointWrite).values(
                    run_id=run_id,
                    checkpoint_ns=ns,
                    checkpoint_id=checkpoint_id,
                    task_id=task_id,
                    idx=idx,
                    channel=channel,
                    value_type=vtype,
                    value=vblob,
                    task_path=task_path,
                )
                keys = ["run_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx"]
                if idx >= 0:
                    await session.execute(stmt.on_conflict_do_nothing(index_elements=keys))
                else:
                    await session.execute(
                        stmt.on_conflict_do_update(
                            index_elements=keys,
                            set_={"channel": channel, "value_type": vtype, "value": vblob},
                        )
                    )

    async def adelete_thread(self, thread_id: str) -> None:
        run_id = uuid.UUID(str(thread_id))
        async with self._sessions() as session, session.begin():
            for table in (GraphCheckpointWrite, GraphCheckpointBlob, GraphCheckpoint):
                await session.execute(delete(table).where(table.run_id == run_id))

    def get_next_version(self, current: str | None, channel: None) -> str:
        current_v = 0 if current is None else int(str(current).split(".")[0])
        return f"{current_v + 1:032}.{random.random():016}"  # noqa: S311 - not security

    # ------------------------------------------------------------------ sync API

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        raise NotImplementedError("PostgresCheckpointSaver is async-only")

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        raise NotImplementedError("PostgresCheckpointSaver is async-only")

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        raise NotImplementedError("PostgresCheckpointSaver is async-only")

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        raise NotImplementedError("PostgresCheckpointSaver is async-only")
