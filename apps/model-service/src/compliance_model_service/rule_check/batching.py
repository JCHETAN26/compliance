"""Dynamic micro-batching of encoder rows across concurrent requests.

Each request submits its rows (token-id sequences). A single collector task takes the
first waiting chunk, then keeps collecting chunks from *any* request until either
``max_batch_rows`` rows are gathered or ``max_wait_ms`` has elapsed since the first one
arrived, and runs that batch once in the inference thread. Results are scattered back
to each request's future. A request larger than ``max_batch_rows`` is split into
chunks, which may share batches with other requests' chunks.

Admission is bounded by the rows the batcher actually *retains*: a chunk's rows count
against ``max_queue_rows`` from admission until the collector has finished with it
(run, skipped because its request was cancelled, or failed at shutdown). Cancelling a
request therefore does not free capacity while its rows still sit in the queue or in an
active forward pass. A request that would exceed the bound is rejected immediately
(``OverloadedError`` -> HTTP 503) instead of queueing without limit.

``close()`` fails every queued, carried and *active* chunk, so no caller can wait
forever on a forward pass that will never be delivered.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from compliance_model_service.rule_check.labels import NUM_LABELS

RunBatch = Callable[[list[list[int]]], tuple[np.ndarray, np.ndarray]]


class OverloadedError(Exception):
    pass


class BatcherClosedError(RuntimeError):
    pass


class InvalidModelOutputError(RuntimeError):
    pass


@dataclass
class _Chunk:
    request: int
    rows: list[list[int]]
    future: asyncio.Future[tuple[np.ndarray, np.ndarray]]


@dataclass
class BatchStats:
    batches: int = 0
    rows: int = 0
    max_rows_in_batch: int = 0
    multi_request_batches: int = 0
    rejected_requests: int = 0
    failed_batches: int = 0
    skipped_cancelled_chunks: int = 0
    size_histogram: Counter[int] = field(default_factory=Counter)

    def snapshot(self) -> dict[str, Any]:
        return {
            "batches": self.batches,
            "rows": self.rows,
            "max_rows_in_batch": self.max_rows_in_batch,
            "multi_request_batches": self.multi_request_batches,
            "rejected_requests": self.rejected_requests,
            "failed_batches": self.failed_batches,
            "skipped_cancelled_chunks": self.skipped_cancelled_chunks,
            "mean_rows_per_batch": round(self.rows / self.batches, 3) if self.batches else None,
        }


def check_outputs(evidence: np.ndarray, labels: np.ndarray, rows: int) -> None:
    """Never scatter misaligned or non-finite outputs back to requests."""
    if evidence.shape != (rows,) or labels.shape != (rows, NUM_LABELS):
        raise InvalidModelOutputError(
            f"model returned shapes {evidence.shape} / {labels.shape} for {rows} rows"
        )
    if not (np.isfinite(evidence).all() and np.isfinite(labels).all()):
        raise InvalidModelOutputError("model returned non-finite logits")


class MicroBatcher:
    def __init__(
        self,
        run_batch: RunBatch,
        *,
        max_batch_rows: int,
        max_wait_ms: float,
        max_queue_rows: int,
        executor: ThreadPoolExecutor,
    ) -> None:
        if max_batch_rows < 1 or max_queue_rows < max_batch_rows or max_wait_ms < 0:
            raise ValueError("invalid micro-batching limits")
        self._run_batch = run_batch
        self.max_batch_rows = max_batch_rows
        self.max_wait = max_wait_ms / 1000.0
        self.max_queue_rows = max_queue_rows
        self._executor = executor
        self._queue: asyncio.Queue[_Chunk] = asyncio.Queue()
        self._carry: _Chunk | None = None
        self._active: list[_Chunk] = []
        self._retained_rows = 0
        self._ids = itertools.count(1)
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self.stats = BatchStats()

    @property
    def retained_rows(self) -> int:
        """Rows admitted and not yet released (queued, carried or in a forward pass)."""
        return self._retained_rows

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._collect())

    async def close(self) -> None:
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        pending = [*self._active, *([self._carry] if self._carry else [])]
        self._active, self._carry = [], None
        while not self._queue.empty():
            pending.append(self._queue.get_nowait())
        for chunk in pending:
            self._release(chunk)
            if not chunk.future.done():
                chunk.future.set_exception(BatcherClosedError("model service is shutting down"))

    async def infer(self, rows: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
        """Evidence logits ``[n]`` and label logits ``[n, 3]`` for ``rows``, in order."""
        if self._closed or self._task is None:
            raise BatcherClosedError("micro-batcher is not running")
        if not rows:
            raise ValueError("no rows to infer")
        if self._retained_rows + len(rows) > self.max_queue_rows:
            self.stats.rejected_requests += 1
            raise OverloadedError
        self._retained_rows += len(rows)
        request = next(self._ids)
        loop = asyncio.get_running_loop()
        chunks = [
            _Chunk(
                request, [list(r) for r in rows[i : i + self.max_batch_rows]], loop.create_future()
            )
            for i in range(0, len(rows), self.max_batch_rows)
        ]
        for chunk in chunks:
            self._queue.put_nowait(chunk)
        # If this await is cancelled, the chunks' futures are cancelled with it; their
        # rows stay counted until the collector skips (and releases) them.
        results = await asyncio.gather(*(c.future for c in chunks))
        evidence = np.concatenate([r[0] for r in results])
        labels = np.concatenate([r[1] for r in results])
        return evidence, labels

    def _release(self, chunk: _Chunk) -> None:
        self._retained_rows -= len(chunk.rows)
        chunk.rows = []

    async def _next_batch(self) -> list[_Chunk]:
        loop = asyncio.get_running_loop()
        first = self._carry if self._carry is not None else await self._queue.get()
        self._carry = None
        # Collected chunks are tracked in ``_active`` immediately, so ``close()`` can fail
        # them even if it interrupts the collection wait.
        batch = self._active = [first]
        rows = len(first.rows)
        deadline = loop.time() + self.max_wait
        while rows < self.max_batch_rows:
            try:
                chunk = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    chunk = await asyncio.wait_for(self._queue.get(), timeout)
                except TimeoutError:
                    break
            if rows + len(chunk.rows) > self.max_batch_rows:
                self._carry = chunk
                break
            batch.append(chunk)
            rows += len(chunk.rows)
        return batch

    async def _collect(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._next_batch()
            live: list[_Chunk] = []
            for chunk in batch:
                if chunk.future.done():  # the request was cancelled while queued
                    self.stats.skipped_cancelled_chunks += 1
                    self._release(chunk)
                else:
                    live.append(chunk)
            if not live:
                continue
            self._active = live
            rows = [r for c in live for r in c.rows]
            try:
                evidence, labels = await loop.run_in_executor(self._executor, self._run_batch, rows)
                check_outputs(evidence, labels, len(rows))
            except Exception as exc:
                self.stats.failed_batches += 1
                self._finish(live, exc)
                continue
            self._record(live, len(rows))
            offset = 0
            for chunk in live:
                n = len(chunk.rows)
                if not chunk.future.done():
                    chunk.future.set_result(
                        (evidence[offset : offset + n], labels[offset : offset + n])
                    )
                offset += n
            self._finish(live, None)

    def _finish(self, chunks: list[_Chunk], error: Exception | None) -> None:
        for chunk in chunks:
            if error is not None and not chunk.future.done():
                chunk.future.set_exception(error)
            self._release(chunk)
        self._active = []

    def _record(self, chunks: list[_Chunk], rows: int) -> None:
        self.stats.batches += 1
        self.stats.rows += rows
        self.stats.max_rows_in_batch = max(self.stats.max_rows_in_batch, rows)
        self.stats.size_histogram[rows] += 1
        if len({c.request for c in chunks}) > 1:
            self.stats.multi_request_batches += 1
