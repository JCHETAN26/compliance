"""Weighted RRF, deterministic ties and retrieval configuration validation."""

from __future__ import annotations

import pytest

from compliance_core.config import Settings
from compliance_retrieval.config import RetrievalConfig
from compliance_retrieval.fusion import Candidate, reciprocal_rank_fusion, single_method


def c(chunk_id: str, score: float) -> Candidate:
    return Candidate(chunk_id, score)


def test_rrf_scores_are_weighted_reciprocal_ranks() -> None:
    hits = reciprocal_rank_fusion(
        {
            "vector": [c("a", 0.9), c("b", 0.8), c("c", 0.7)],
            "keyword": [c("c", 3.0), c("a", 2.0)],
        },
        {"vector": 1.0, "keyword": 2.0},
        k=60,
    )
    by_id = {h.chunk_id: h for h in hits}
    assert by_id["a"].fused_score == pytest.approx(1 / 61 + 2 / 62)
    assert by_id["b"].fused_score == pytest.approx(1 / 62)
    assert by_id["c"].fused_score == pytest.approx(1 / 63 + 2 / 61)
    assert [h.chunk_id for h in hits] == ["c", "a", "b"]
    assert by_id["c"].ranks == {"vector": 3, "keyword": 1}
    assert by_id["c"].scores == {"vector": 0.7, "keyword": 3.0}
    assert by_id["b"].ranks == {"vector": 2}  # absent from keyword: no fabricated rank


def test_rrf_ranks_follow_scores_not_input_order() -> None:
    hits = reciprocal_rank_fusion(
        {"vector": [c("low", 0.1), c("high", 0.9)]}, {"vector": 1.0}, k=10
    )
    assert [(h.chunk_id, h.ranks["vector"]) for h in hits] == [("high", 1), ("low", 2)]


def test_ties_are_deterministic() -> None:
    # Equal fused scores: x (ranks 1 and 3) vs y (ranks 3 and 1) tie on score and best
    # rank and method count, so the chunk id decides; z has the same score from one method.
    ranked = {
        "vector": [c("x", 0.9), c("q", 0.8), c("y", 0.7)],
        "keyword": [c("y", 5.0), c("r", 4.0), c("x", 3.0)],
    }
    first = reciprocal_rank_fusion(ranked, {"vector": 1.0, "keyword": 1.0}, k=60)
    second = reciprocal_rank_fusion(
        dict(reversed(list(ranked.items()))), {"vector": 1.0, "keyword": 1.0}, k=60
    )
    assert [h.chunk_id for h in first] == [h.chunk_id for h in second]
    assert [h.chunk_id for h in first][:2] == ["x", "y"]
    # Equal native scores within a method are ordered by chunk id.
    tied = reciprocal_rank_fusion({"keyword": [c("b", 1.0), c("a", 1.0)]}, {"keyword": 1}, k=1)
    assert [(h.chunk_id, h.ranks["keyword"]) for h in tied] == [("a", 1), ("b", 2)]


def test_best_rank_breaks_equal_fused_scores() -> None:
    # p: rank 2 in both (2/62); s: rank 1 in vector, rank 4 in keyword (1/61+1/64 differs),
    # so construct an exact tie with k=1: p ranks (2,2) -> 2/3; s ranks (1, 5) -> 1/2+1/6=2/3.
    ranked = {
        "vector": [c("s", 9), c("p", 8), c("f1", 7), c("f2", 6), c("f3", 5)],
        "keyword": [c("g1", 9), c("p", 8), c("g2", 7), c("g3", 6), c("s", 5)],
    }
    hits = reciprocal_rank_fusion(ranked, {"vector": 1.0, "keyword": 1.0}, k=1)
    by_id = {h.chunk_id: h for h in hits}
    assert by_id["p"].fused_score == pytest.approx(by_id["s"].fused_score)
    order = [h.chunk_id for h in hits]
    assert order.index("s") < order.index("p")  # s has the better best rank (1 vs 2)


def test_duplicates_and_zero_weights_and_empty_inputs() -> None:
    hits = reciprocal_rank_fusion(
        {"vector": [c("a", 0.5), c("a", 0.9), c("b", 0.4)], "keyword": [c("z", 1.0)]},
        {"vector": 1.0, "keyword": 0.0},
        k=60,
    )
    assert [h.chunk_id for h in hits] == ["a", "b"]  # zero-weight method adds nothing
    assert hits[0].scores == {"vector": 0.9}
    assert hits[1].ranks == {"vector": 2}
    assert reciprocal_rank_fusion({}, {}, k=60) == []
    assert reciprocal_rank_fusion({"vector": []}, {"vector": 1.0}, k=60) == []
    with pytest.raises(ValueError, match="RRF constant"):
        reciprocal_rank_fusion({"vector": [c("a", 1)]}, {"vector": 1.0}, k=0)
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion({"vector": [c("a", 1)]}, {"vector": -1.0}, k=60)


def test_single_method_keeps_native_scores() -> None:
    hits = single_method("keyword", [c("b", 0.2), c("a", 0.7), c("b", 0.1)])
    assert [(h.chunk_id, h.fused_score, h.ranks) for h in hits] == [
        ("a", 0.7, {"keyword": 1}),
        ("b", 0.2, {"keyword": 2}),
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"top_k": 60, "candidate_k": 50},
        {"ef_search": 0},
        {"rrf_k": 0},
        {"vector_weight": -0.1},
        {"vector_weight": 0.0, "keyword_weight": 0.0},
        {"rerank_candidates": 0},
        {"rerank_candidates": 101},
        {"keyword_mode": "phrase"},
        {"statement_timeout_ms": 0},
    ],
)
def test_invalid_configs_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must"):
        RetrievalConfig(**kwargs)  # type: ignore[arg-type]


def test_config_from_settings_and_top_k_override(clean_env: None) -> None:
    settings = Settings(
        search_top_k=5,
        search_candidate_k=3,
        search_ef_search=40,
        search_rrf_k=20,
        search_vector_weight=0.5,
        search_keyword_weight=2.0,
        search_keyword_mode="websearch",
        search_rerank_candidates=4,
        search_statement_timeout_ms=900,
    )
    config = RetrievalConfig.from_settings(settings)
    assert config.candidate_k == 5  # never below top_k
    assert (config.ef_search, config.rrf_k, config.keyword_mode) == (40, 20, "websearch")
    assert (config.vector_weight, config.keyword_weight) == (0.5, 2.0)
    assert config.statement_timeout_ms == 900
    assert config.rerank_pool == 5  # at least top_k results are reranked
    wider = config.with_top_k(30)
    assert (wider.top_k, wider.candidate_k, wider.rerank_pool) == (30, 30, 30)
