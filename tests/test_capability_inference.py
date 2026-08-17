"""Tests for correlation + bad-case driven capability-deficit inference."""

from __future__ import annotations

import pytest

from benchmark_diagnosis.recommendation.capability_inference import (
    FM_REINFORCEMENT,
    infer_capability_deficit,
)

_FM_MAP = {"context_ignoring": ["long_context", "reading_comprehension"]}


def _coverage(benchmarks=None) -> list[dict]:
    rows = benchmarks or [
        {
            "benchmark_id": "longbench_v2",
            "design_goal_tags": ["long_context", "reading_comprehension"],
            "reliability_score": 0.8,
            "saturated_flag": False,
            "design_goal_agreement_score": 0.9,
        },
        {
            "benchmark_id": "mmlu_pro",
            "design_goal_tags": ["world_knowledge", "reasoning"],
            "reliability_score": 0.7,
            "saturated_flag": False,
            "design_goal_agreement_score": 0.8,
        },
    ]
    return rows


def _verdict(benchmarks=None) -> list[dict]:
    rows = benchmarks or [
        {"benchmark_id": "longbench_v2", "weight": 0.834, "score": 38.0, "residual": -0.16},
        {"benchmark_id": "mmlu_pro", "weight": 0.049, "score": 50.0, "residual": 0.0},
    ]
    return rows


def test_empty_inputs() -> None:
    d = infer_capability_deficit([], [], {})
    assert d.strengths == {}
    assert d.magnitude == 0.0
    assert d.drivers == []
    assert d.narrative == ""


def test_basic_deficit() -> None:
    d = infer_capability_deficit(_coverage(), _verdict(), {}, _FM_MAP)
    assert d.strengths == {
        "long_context": 0.5,
        "reading_comprehension": 0.5,
    }
    assert d.magnitude > 0
    assert d.drivers == [("longbench_v2", 0.16)]
    assert "该簇缺失能力" in d.narrative
    assert "longbench_v2" in d.narrative


def test_residual_sign_positive_residual_means_no_shortfall() -> None:
    # residual = score - predicted; positive => above expectation => no deficit.
    verdict = [
        {"benchmark_id": "longbench_v2", "weight": 0.834, "score": 80.0, "residual": 0.3}
    ]
    d = infer_capability_deficit(_coverage(), verdict, {}, _FM_MAP)
    assert d.strengths == {}
    assert d.magnitude == 0.0


def test_score_fallback_when_no_residual() -> None:
    # No residual -> shortfall = 1 - normalized score (score 38 -> 0.62).
    verdict = [{"benchmark_id": "longbench_v2", "weight": 0.5, "score": 38.0}]
    d = infer_capability_deficit(_coverage(), verdict, {}, _FM_MAP)
    assert d.strengths == {
        "long_context": 0.5,
        "reading_comprehension": 0.5,
    }


def test_saturated_flag_halves_signal() -> None:
    coverage = [
        {
            "benchmark_id": "longbench_v2",
            "design_goal_tags": ["long_context"],
            "reliability_score": 1.0,
            "saturated_flag": True,
            "design_goal_agreement_score": 1.0,
        },
        {
            "benchmark_id": "mmlu_pro",
            "design_goal_tags": ["world_knowledge"],
            "reliability_score": 1.0,
            "saturated_flag": False,
            "design_goal_agreement_score": 1.0,
        },
    ]
    verdict = [
        {"benchmark_id": "longbench_v2", "weight": 0.5, "score": 40.0, "residual": -0.2},
        {"benchmark_id": "mmlu_pro", "weight": 0.5, "score": 40.0, "residual": -0.2},
    ]
    d = infer_capability_deficit(coverage, verdict, {}, _FM_MAP)
    # Halved saturated signal -> long_context gets 1/3 of the normalized mass.
    assert d.strengths["world_knowledge"] > d.strengths["long_context"]
    assert d.strengths["long_context"] == pytest.approx(1 / 3, abs=1e-3)
    assert d.strengths["world_knowledge"] == pytest.approx(2 / 3, abs=1e-3)


def test_failure_mode_reinforcement_brings_in_new_capability() -> None:
    # Only world_knowledge benchmark evidence; the bad case implies reasoning.
    coverage = [
        {
            "benchmark_id": "mmlu_pro",
            "design_goal_tags": ["world_knowledge", "reasoning"],
            "reliability_score": 0.7,
            "saturated_flag": False,
            "design_goal_agreement_score": 0.8,
        }
    ]
    verdict = [
        {"benchmark_id": "mmlu_pro", "weight": 0.5, "score": 40.0, "residual": -0.2}
    ]
    fm_map = {"reasoning_error": ["reasoning"], "hallucination": ["factuality"]}
    d = infer_capability_deficit(coverage, verdict, {"reasoning_error": 0.9}, fm_map)
    assert "reasoning" in d.strengths
    # The unreinforced mode (factuality) never appears.
    assert "factuality" not in d.strengths


def test_noise_floor_drops_weak_tags() -> None:
    coverage = [
        {
            "benchmark_id": "b_a",
            "design_goal_tags": ["math"],
            "reliability_score": 1.0,
            "saturated_flag": False,
            "design_goal_agreement_score": 1.0,
        },
        {
            "benchmark_id": "b_b",
            "design_goal_tags": ["math", "vision"],
            "reliability_score": 1.0,
            "saturated_flag": False,
            "design_goal_agreement_score": 1.0,
        },
    ]
    verdict = [
        {"benchmark_id": "b_a", "weight": 1.0, "score": 0.0, "residual": -1.0},
        {"benchmark_id": "b_b", "weight": 0.01, "score": 0.0, "residual": -1.0},
    ]
    d = infer_capability_deficit(coverage, verdict, {}, _FM_MAP)
    # vision signal is ~1% of total -> below floor; math keeps ~all the weight.
    assert "vision" not in d.strengths
    assert "math" in d.strengths


def test_driver_ordering() -> None:
    verdict = [
        {"benchmark_id": "mmlu_pro", "weight": 0.5, "score": 40.0, "residual": -0.2},
        {"benchmark_id": "longbench_v2", "weight": 0.5, "score": 40.0, "residual": -0.5},
    ]
    d = infer_capability_deficit(_coverage(), verdict, {}, _FM_MAP)
    assert d.drivers[0][0] == "longbench_v2"


def test_weights_scale_magnitude() -> None:
    light = infer_capability_deficit(
        _coverage(), _verdict([{"benchmark_id": "longbench_v2", "weight": 0.1, "score": 38.0, "residual": -0.16}]), {}, _FM_MAP
    )
    heavy = infer_capability_deficit(
        _coverage(), _verdict([{"benchmark_id": "longbench_v2", "weight": 0.9, "score": 38.0, "residual": -0.16}]), {}, _FM_MAP
    )
    assert heavy.magnitude > light.magnitude
    assert FM_REINFORCEMENT == 0.5
