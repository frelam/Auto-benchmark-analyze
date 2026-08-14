"""Tests for representative benchmark selection (design doc section 3)."""

from __future__ import annotations

import pandas as pd
import pytest

from benchmark_diagnosis.core.types import (
    ClusterPortfolio,
    CoverageEntry,
    Granularity,
    ScoringMethod,
)
from benchmark_diagnosis.representative_selection.portfolio_selector import (
    select_portfolios,
)


def make_entry(
    benchmark_id: str,
    cluster: str,
    *,
    breadth: float,
    reliability: float,
    profile: dict[str, float],
    saturated: bool = False,
    method: ScoringMethod = ScoringMethod.RULE_VERIFIED,
) -> CoverageEntry:
    """Build a CoverageEntry with the fields relevant to selection."""
    return CoverageEntry(
        benchmark_id=benchmark_id,
        primary_cluster=cluster,
        discrimination_profile=profile,
        coverage_breadth_score=breadth,
        reliability_score=reliability,
        saturated_flag=saturated,
        scoring_method=method,
        design_goal_tags=[],
        design_goal_agreement_score=0.0,
        granularity=Granularity.AGGREGATE_ONLY,
    )


def _build_two_cluster_coverage() -> list[CoverageEntry]:
    return [
        # cluster c1: one low-reliability and one saturated entry must be excluded
        make_entry(
            "b1",
            "c1",
            breadth=0.8,
            reliability=0.9,
            profile={"math": 1.0, "code": 0.2},
        ),
        make_entry(
            "b2",
            "c1",
            breadth=0.7,
            reliability=0.8,
            profile={"math": 0.9, "code": 1.0},
            method=ScoringMethod.LLM_JUDGE,
        ),
        make_entry(
            "b3",
            "c1",
            breadth=0.6,
            reliability=0.7,
            profile={"code": 1.0, "reason": 0.5},
            method=ScoringMethod.HYBRID,
        ),
        make_entry(
            "b4",
            "c1",
            breadth=0.9,
            reliability=0.2,
            profile={"math": 1.0},
        ),  # below min_reliability
        make_entry(
            "b5",
            "c1",
            breadth=0.85,
            reliability=0.9,
            profile={"math": 1.0},
            saturated=True,
        ),
        # cluster c2
        make_entry("b6", "c2", breadth=0.75, reliability=0.9, profile={"math": 1.0}),
        make_entry(
            "b7",
            "c2",
            breadth=0.65,
            reliability=0.8,
            profile={"code": 1.0},
            method=ScoringMethod.LLM_JUDGE,
        ),
        make_entry(
            "b8",
            "c2",
            breadth=0.55,
            reliability=0.7,
            profile={"reason": 1.0},
            method=ScoringMethod.HYBRID,
        ),
    ]


def test_select_portfolios_per_cluster_weights_and_exclusions() -> None:
    portfolios = select_portfolios(_build_two_cluster_coverage(), min_reliability=0.5)

    assert [p.cluster_id for p in portfolios] == ["c1", "c2"]
    anchors = {"c1": "b1", "c2": "b6"}
    for portfolio in portfolios:
        assert isinstance(portfolio, ClusterPortfolio)
        assert portfolio.benchmarks
        assert len(portfolio.benchmarks) <= 3
        ids = {item["benchmark_id"] for item in portfolio.benchmarks}
        assert "b4" not in ids  # reliability below min_reliability
        assert "b5" not in ids  # saturated
        assert anchors[portfolio.cluster_id] in ids  # rule_verified anchor present
        weights = [item["weight"] for item in portfolio.benchmarks]
        assert sum(weights) == pytest.approx(1.0)


def test_select_portfolios_correlation_skips_redundant() -> None:
    coverage = [
        make_entry("x1", "c3", breadth=0.8, reliability=0.9, profile={"a": 1.0}),
        make_entry(
            "x2",
            "c3",
            breadth=0.7,
            reliability=0.8,
            profile={"a": 1.0},
            method=ScoringMethod.LLM_JUDGE,
        ),
        make_entry(
            "x3",
            "c3",
            breadth=0.5,
            reliability=0.7,
            profile={"b": 1.0},
            method=ScoringMethod.HYBRID,
        ),
    ]
    corr = pd.DataFrame(
        [
            [1.0, 0.95, 0.2],
            [0.95, 1.0, 0.1],
            [0.2, 0.1, 1.0],
        ],
        index=["x1", "x2", "x3"],
        columns=["x1", "x2", "x3"],
    )

    portfolios = select_portfolios(coverage, correlation=corr, combo_size=3)

    assert len(portfolios) == 1
    ids = {item["benchmark_id"] for item in portfolios[0].benchmarks}
    assert "x2" not in ids  # 0.95 correlation with anchor exceeds max_corr 0.9
    assert {"x1", "x3"} <= ids
    assert sum(item["weight"] for item in portfolios[0].benchmarks) == pytest.approx(1.0)


def test_select_portfolios_empty_when_all_saturated() -> None:
    coverage = [
        make_entry(
            "y1", "c4", breadth=0.8, reliability=0.9, profile={"a": 1.0}, saturated=True
        ),
        make_entry(
            "y2", "c4", breadth=0.7, reliability=0.9, profile={"b": 1.0}, saturated=True
        ),
    ]

    portfolios = select_portfolios(coverage)

    # One ClusterPortfolio per cluster still appears, with no benchmarks chosen.
    assert [p.cluster_id for p in portfolios] == ["c4"]
    assert portfolios[0].benchmarks == []


def test_select_portfolios_single_candidate_weight_is_one() -> None:
    coverage = [
        make_entry("z1", "c5", breadth=0.9, reliability=0.9, profile={"a": 1.0}),
    ]

    portfolios = select_portfolios(coverage, combo_size=3)

    assert len(portfolios) == 1
    assert portfolios[0].benchmarks == [{"benchmark_id": "z1", "weight": 1.0}]
