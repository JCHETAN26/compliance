"""Dynamic micro-batching: merging, order, limits, failures, cancellation and shutdown."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from compliance_model_service.rule_check.batching import (
    BatcherClosedError,
    InvalidModelOutputError,
    MicroBatcher,
    OverloadedError,
)


class RecordingBackend:
    """Evidence logit = first token id; label logits = [len, 0, -len]. Records batches."""

    def __init__(self) -> None:
        self.batches: list[list[list[int]]] = []

    def __call__(self, rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        self.batches.append(rows)
        ev = np.array([float(r[0]) for r in rows], dtype=np.float32)
        lab = np.array([[len(r), 0.0, -len(r)] for r in rows], dtype=np.float32)
        return ev, lab


class BlockingBackend(RecordingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        self.entered.set()
        self.release.wait(timeout=10)
        return super().__call__(rows)


@pytest.fixture
async def pool() -> AsyncIterator[ThreadPoolExecutor]:
    executor = ThreadPoolExecutor(1)
    yield executor
    executor.shutdown(wait=True)


def _batcher(backend: RecordingBackend, pool: ThreadPoolExecutor, **kw: float) -> MicroBatcher:
    params: dict[str, float] = {"max_batch_rows": 8, "max_wait_ms": 30, "max_queue_rows": 64, **kw}
    return MicroBatcher(
        backend,
        max_batch_rows=int(params["max_batch_rows"]),
        max_wait_ms=params["max_wait_ms"],
        max_queue_rows=int(params["max_queue_rows"]),
        executor=pool,
    )


async def test_concurrent_requests_share_batches_and_keep_order(pool: ThreadPoolExecutor) -> None:
    backend = RecordingBackend()
    batcher = _batcher(backend, pool)
    batcher.start()
    requests = [[[i * 10 + j] for j in range(3)] for i in range(4)]  # 4 requests x 3 rows
    results = await asyncio.gather(*(batcher.infer(rows) for rows in requests))
    for rows, (ev, lab) in zip(requests, results, strict=True):
        assert ev.tolist() == [float(r[0]) for r in rows]
        assert lab.shape == (3, 3)
    assert batcher.stats.multi_request_batches >= 1
    assert max(len(b) for b in backend.batches) <= 8
    assert sum(len(b) for b in backend.batches) == 12
    await batcher.close()


async def test_large_request_is_split_into_bounded_chunks(pool: ThreadPoolExecutor) -> None:
    backend = RecordingBackend()
    batcher = _batcher(backend, pool, max_wait_ms=0)
    batcher.start()
    rows = [[i] for i in range(21)]
    ev, _ = await batcher.infer(rows)
    assert ev.tolist() == [float(i) for i in range(21)]
    assert all(len(b) <= 8 for b in backend.batches)
    assert batcher.retained_rows == 0
    await batcher.close()


async def test_admission_is_bounded(pool: ThreadPoolExecutor) -> None:
    backend = BlockingBackend()
    batcher = _batcher(backend, pool, max_queue_rows=10, max_wait_ms=0)
    batcher.start()
    first = asyncio.create_task(batcher.infer([[1]] * 8))
    await asyncio.to_thread(backend.entered.wait, 5)
    with pytest.raises(OverloadedError):
        await batcher.infer([[2]] * 3)  # 8 retained + 3 > 10
    assert batcher.stats.rejected_requests == 1
    backend.release.set()
    await first
    await batcher.close()


async def test_invalid_backend_output_fails_requests_not_misaligns(
    pool: ThreadPoolExecutor,
) -> None:
    def wrong(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(len(rows), dtype=np.float32), np.zeros((len(rows), 2), dtype=np.float32)

    def nan(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        return np.full(len(rows), np.nan, dtype=np.float32), np.zeros(
            (len(rows), 3), dtype=np.float32
        )

    for backend in (wrong, nan):
        batcher = MicroBatcher(
            backend, max_batch_rows=4, max_wait_ms=0, max_queue_rows=8, executor=pool
        )
        batcher.start()
        with pytest.raises(InvalidModelOutputError):
            await batcher.infer([[1], [2]])
        assert batcher.stats.failed_batches == 1
        assert batcher.retained_rows == 0
        await batcher.close()


async def test_close_fails_active_and_queued_requests(pool: ThreadPoolExecutor) -> None:
    backend = BlockingBackend()
    batcher = _batcher(backend, pool, max_wait_ms=0, max_batch_rows=4)
    batcher.start()
    active = asyncio.create_task(batcher.infer([[1]] * 4))
    await asyncio.to_thread(backend.entered.wait, 5)
    queued = asyncio.create_task(batcher.infer([[2]] * 2))
    await asyncio.sleep(0.01)
    await batcher.close()
    for task in (active, queued):
        with pytest.raises(BatcherClosedError):
            await asyncio.wait_for(task, 2)
    assert batcher.retained_rows == 0
    backend.release.set()  # let the worker thread finish; nothing may wait on it
    with pytest.raises(BatcherClosedError):
        await batcher.infer([[3]])


async def test_cancelled_requests_keep_capacity_until_released(pool: ThreadPoolExecutor) -> None:
    backend = BlockingBackend()
    batcher = _batcher(backend, pool, max_wait_ms=0, max_batch_rows=4, max_queue_rows=12)
    batcher.start()
    blocker = asyncio.create_task(batcher.infer([[0]] * 4))
    await asyncio.to_thread(backend.entered.wait, 5)
    for _ in range(10):  # repeated admit/cancel while the forward is blocked
        task = asyncio.create_task(batcher.infer([[1]] * 4))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)  # cancelled or OverloadedError
        assert batcher.retained_rows <= 12  # never more rows held than the bound
    backend.release.set()
    await blocker
    for _ in range(50):  # the collector releases the cancelled chunks
        if batcher.retained_rows == 0:
            break
        await asyncio.sleep(0.01)
    assert batcher.retained_rows == 0
    assert batcher.stats.skipped_cancelled_chunks >= 1
    assert batcher.stats.rejected_requests >= 1  # capacity was not freed by cancellation
    ev, _ = await batcher.infer([[7]])
    assert ev.tolist() == [7.0]
    await batcher.close()
