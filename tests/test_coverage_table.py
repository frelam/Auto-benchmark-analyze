"""Tests for the capability-coverage table (factor + mIRT paths)."""

from __future__ import annotations

import numpy as np

from benchmark_diagnosis.capability_analysis.coverage_table import build_coverage_table
from benchmark_diagnosis.capability_analysis.factor_analysis import FactorModel
from benchmark_diagnosis.capability_analysis.mirt_fit import MIRTResult


def _meta():
    return {
        "math": {"declared_tags": ["math"], "scoring_method": "rule_verified"},
        "code": {"declared_tags": ["code"], "scoring_method": "rule_verified"},
        "world": {"declared_tags": ["world_knowledge"], "scoring_method": "rule_verified"},
    }


def test_factor_path():
    factor = FactorModel(
        loadings=np.array(
            [[0.9, 0.1], [0.8, 0.2], [0.2, 0.9]], dtype=np.float64
        ),
        clusters=[0, 0, 1],
        benchmark_ids=["math", "code", "world"],
        explained_variance_ratio=np.array([0.6, 0.3]),
        n_factors=2,
    )
    entries = build_coverage_table(factor=factor, benchmark_meta=_meta())
    by_id = {e.benchmark_id: e for e in entries}
    assert set(by_id) == {"math", "code", "world"}
    assert by_id["math"].primary_cluster == "cluster_0"
    assert by_id["world"].primary_cluster == "cluster_1"
    assert all(0.0 <= e.reliability_score <= 1.0 for e in entries)
    assert all(e.granularity.value == "aggregate_only" for e in entries)
    assert all(0.0 <= e.coverage_breadth_score <= 1.0 for e in entries)


def test_mirt_path():
    rng = np.random.default_rng(0)
    n_items = 20
    dims = 2
    mirt = MIRTResult(
        theta=rng.normal(size=(5, dims)),
        alpha=rng.normal(size=(n_items, dims)),
        beta=rng.normal(size=n_items),
        model_ids=[f"m{i}" for i in range(5)],
        item_ids=[f"i{i}" for i in range(n_items)],
        dims=dims,
    )
    item_benchmark = {f"i{i}": "math" if i < 10 else "code" for i in range(n_items)}
    entries = build_coverage_table(
        mirt=mirt, benchmark_meta=_meta(), item_benchmark=item_benchmark
    )
    by_id = {e.benchmark_id: e for e in entries}
    assert set(by_id) == {"math", "code"}
    assert all(e.granularity.value == "item_level" for e in entries)
    for e in entries:
        profile = e.discrimination_profile
        assert abs(sum(profile.values()) - 1.0) < 1e-6
        assert e.primary_cluster.startswith("dim_")


def test_requires_mirt_or_factor():
    import pytest

    with pytest.raises(ValueError):
        build_coverage_table()
