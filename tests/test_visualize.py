"""Tests for result-archive summary rendering and figure generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmark_diagnosis.reporting.report_generator import render_results_summary


def _coverage() -> list[dict]:
    return [
        {
            "benchmark_id": "mmlu",
            "primary_cluster": "cluster_0",
            "discrimination_profile": {"f0": 0.6, "f1": 0.4},
            "coverage_breadth_score": 0.92,
            "reliability_score": 0.23,
            "saturated_flag": False,
            "design_goal_agreement_score": 0.5,
            "design_goal_tags": ["world_knowledge", "reasoning"],
        },
        {
            "benchmark_id": "gsm8k",
            "primary_cluster": "cluster_0",
            "discrimination_profile": {"f0": 0.3, "f1": 0.7},
            "coverage_breadth_score": 0.84,
            "reliability_score": 0.21,
            "saturated_flag": True,
            "design_goal_agreement_score": 0.0,
            "design_goal_tags": ["math"],
        },
    ]


def test_render_results_summary_contains_sections() -> None:
    scores = pd.DataFrame({"mmlu": [0.66, 0.70], "gsm8k": [0.79, 0.80]})
    summary = render_results_summary(
        coverage=_coverage(),
        portfolios=[
            {
                "cluster_id": "cluster_0",
                "benchmarks": [{"benchmark_id": "mmlu", "weight": 1.0}],
            }
        ],
        curves=[{"benchmark_id": "mmlu", "kind": "params_dense", "points": []}],
        scores=scores,
        benchmark_names={"mmlu": "MMLU", "gsm8k": "GSM8K"},
        versions={"coverage_version": "coverage-1"},
        figure_names=["fig_curves_mmlu.png"],
        generated_at="2026-08-14",
    )
    assert "Offline Results" in summary
    assert "| models | 2 |" in summary
    assert "| benchmarks | 2 |" in summary
    assert "| scores | 4 |" in summary
    assert "| capability clusters | 1 |" in summary
    assert "| fitted expectation curves | 1 |" in summary
    assert "## Capability-coverage table" in summary
    assert "MMLU" in summary
    assert "## Representative portfolios" in summary
    assert "## Figures" in summary
    assert "![fig_curves_mmlu.png](figures/fig_curves_mmlu.png)" in summary


def test_cluster_map_embedding_uses_direction_not_magnitude() -> None:
    """Regression: the cluster map must not separate correlated benchmarks.

    Two benchmarks with identical loading *direction* but very different loading
    *magnitudes* are perfectly correlated and must land on the same map point.
    The old embedding (PCA on the abs/sum-normalized profile) pushed the
    strong-on-f0 benchmark far from a correlated weaker one — the bug that made
    math and gsm8k appear distant despite a 0.74 Spearman correlation.
    """
    from benchmark_diagnosis.reporting.visualize import _embed_cluster_map

    rows = [
        {"benchmark_id": "math_a", "primary_cluster": "c0",
         "factor_loadings": [0.9, 0.0, 0.0, 0.0, 0.0]},
        {"benchmark_id": "math_b", "primary_cluster": "c0",
         "factor_loadings": [0.45, 0.0, 0.0, 0.0, 0.0]},  # same direction, half magnitude
        {"benchmark_id": "code", "primary_cluster": "c1",
         "factor_loadings": [0.0, 0.9, 0.0, 0.0, 0.0]},
    ]
    coords = _embed_cluster_map(rows)
    by = {r["benchmark_id"]: coords[i] for i, r in enumerate(rows)}
    d_same = float(np.linalg.norm(by["math_a"] - by["math_b"]))
    d_diff = float(np.linalg.norm(by["math_a"] - by["code"]))
    assert d_same < 0.01   # identical direction -> identical normalized position
    assert d_same < d_diff


def test_cluster_map_embedding_falls_back_to_profile() -> None:
    """Without factor_loadings the map still renders from discrimination_profile."""
    from benchmark_diagnosis.reporting.visualize import _embed_cluster_map

    rows = [
        {"benchmark_id": "a", "primary_cluster": "c0",
         "discrimination_profile": {"f0": 0.8, "f1": 0.2}},
        {"benchmark_id": "b", "primary_cluster": "c1",
         "discrimination_profile": {"f0": 0.2, "f1": 0.8}},
    ]
    coords = _embed_cluster_map(rows)
    assert coords.shape == (2, 2)
    assert np.all(np.isfinite(coords))


def test_cluster_capability_labels() -> None:
    from benchmark_diagnosis.reporting.report_generator import cluster_capability_labels

    labels = cluster_capability_labels(_coverage())
    assert set(labels["cluster_0"]) == {"world_knowledge", "reasoning", "math"}


def test_render_figures_write_png(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from benchmark_diagnosis.reporting import visualize

    curves = [
        {
            "benchmark_id": "mmlu",
            "kind": "params_dense",
            "coefficients": {"slope": 0.5, "intercept": 0.0},
            "points": [[1.0, 0.4], [10.0, 0.6], [100.0, 0.8], [7.0, 0.5], [70.0, 0.7]],
        },
        {
            "benchmark_id": "mmlu",
            "kind": "time_frontier",
            "coefficients": {},
            "points": [[1500.0, 0.6], [1700.0, 0.7], [1900.0, 0.8]],
        },
    ]
    written = visualize.render_curves(curves, tmp_path, {"mmlu": "MMLU"})
    assert written and all(p.exists() for p in written)

    written = visualize.render_frontier_overview(curves, tmp_path)
    assert written and written[0].exists()

    written = visualize.render_coverage_profile(_coverage(), tmp_path)
    assert written and written[0].exists()

    written = visualize.render_coverage_metrics(_coverage(), tmp_path)
    assert written and written[0].exists()

    written = visualize.render_cluster_map(_coverage(), tmp_path)
    assert written and written[0].exists()

    rng = np.random.default_rng(0)
    scores = pd.DataFrame(
        {
            "mmlu": rng.uniform(0.40, 0.95, size=20),
            "gsm8k": rng.uniform(0.20, 0.98, size=20),
            "math": rng.uniform(0.05, 0.95, size=20),
        }
    )
    written = visualize.render_correlation(scores, tmp_path)
    assert written and written[0].exists()
