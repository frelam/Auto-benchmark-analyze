"""Tests for result-archive summary rendering and figure generation."""

from __future__ import annotations

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

    scores = pd.DataFrame({"mmlu": [0.66, 0.70, 0.68], "gsm8k": [0.79, 0.80, 0.82]})
    written = visualize.render_correlation(scores, tmp_path)
    assert written and written[0].exists()
