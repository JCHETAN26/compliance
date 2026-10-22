"""HNSW sweep grid validity: the skip rule matches pgvector, and the committed sweep
result covers exactly the valid grid with the invalid combination reported."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import asyncpg
import pytest
from sqlalchemy.engine import make_url

from conftest import TEST_DATABASE_URL
from evals.retrieval import sweep

RESULTS = Path(__file__).resolve().parents[2] / "results" / "hnsw_sweep.json"


def valid_builds() -> list[tuple[int, int]]:
    return [
        (m, efc)
        for m, efc in itertools.product(sweep.GRID_M, sweep.GRID_EF_CONSTRUCTION)
        if efc >= 2 * m
    ]


def test_grid_contains_the_production_index_and_one_invalid_combination() -> None:
    invalid = [
        (m, efc)
        for m, efc in itertools.product(sweep.GRID_M, sweep.GRID_EF_CONSTRUCTION)
        if efc < 2 * m
    ]
    assert invalid == [(32, 32)]
    assert (sweep.PRODUCTION["m"], sweep.PRODUCTION["ef_construction"]) in valid_builds()
    assert len(valid_builds()) * len(sweep.GRID_EF_SEARCH) == 66
    assert min(sweep.GRID_EF_SEARCH) >= sweep.K  # ef_search below k would truncate results


def test_committed_sweep_result_matches_the_valid_grid() -> None:
    payload = json.loads(RESULTS.read_text())
    measured = {(r["m"], r["ef_construction"], r["ef_search"]) for r in payload["results"]}
    expected = {
        (m, efc, ef) for (m, efc), ef in itertools.product(valid_builds(), sweep.GRID_EF_SEARCH)
    }
    assert measured == expected
    assert len(payload["results"]) == 66  # no duplicates
    assert [(s["m"], s["ef_construction"]) for s in payload["skipped_combinations"]] == [(32, 32)]
    assert all(r["plan_uses_hnsw"] for r in payload["results"])
    assert [r["ef_search"] for r in payload["results"] if r["production_setting"]] == list(
        sweep.GRID_EF_SEARCH
    )


@pytest.mark.integration
@pytest.mark.parametrize(("m", "efc", "accepted"), [(32, 32, False), (32, 64, True), (8, 16, True)])
async def test_skip_rule_matches_pgvector(m: int, efc: int, accepted: bool) -> None:
    """pgvector rejects ef_construction < 2*m and accepts the boundary. Runs in a rolled-back
    transaction on a temporary table, so nothing persists."""
    parsed = make_url(TEST_DATABASE_URL)
    conn = await asyncpg.connect(
        user=parsed.username,
        password=parsed.password,
        host=parsed.host,
        port=parsed.port,
        database=parsed.database,
    )
    try:
        tx = conn.transaction()
        await tx.start()
        try:
            await conn.execute("CREATE TEMP TABLE probe (embedding vector(3)) ON COMMIT DROP")
            statement = (
                "CREATE INDEX ON probe USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {m}, ef_construction = {efc})"
            )
            if accepted:
                await conn.execute(statement)
            else:
                with pytest.raises(asyncpg.InvalidParameterValueError, match="2 \\* m"):
                    await conn.execute(statement)
        finally:
            await tx.rollback()
    finally:
        await conn.close()
    assert (efc >= 2 * m) is accepted  # the sweep's skip rule agrees with the engine
