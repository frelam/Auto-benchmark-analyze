"""Tests for expectation-curve fitting and residual judgment."""

from __future__ import annotations

import datetime as dt

from benchmark_diagnosis.config import CurvesConfig
from benchmark_diagnosis.core.schema import ModelRecord
from benchmark_diagnosis.evaluation_orchestration.expectation_curves import (
    fit_params_curve,
    fit_time_frontier,
    judge,
    predict,
)


def test_fit_params_curve_monotone():
    curve = fit_params_curve(
        "mmlu", "params_dense", params=[1.0, 10.0, 100.0], scores=[30.0, 50.0, 70.0]
    )
    assert curve is not None
    assert curve.kind == "params_dense"
    assert predict(curve, 10.0) > predict(curve, 1.0)
    assert predict(curve, 100.0) > predict(curve, 10.0)
    # predicts ~0.7 at 100 params (interpolated through the fitted line)
    assert abs(predict(curve, 100.0) - 0.7) < 0.02


def test_fit_params_curve_needs_two_points():
    assert fit_params_curve("x", "params_dense", [1.0], [0.5]) is None


def test_time_frontier_non_decreasing():
    dates = [dt.date(2023, 1, 1), dt.date(2023, 6, 1), dt.date(2024, 1, 1)]
    scores = [50.0, 80.0, 60.0]  # 60 is below prior max -> envelope stays 80
    curve = fit_time_frontier("mmlu", dates, scores)
    assert curve is not None
    ys = [p[1] for p in curve.points]
    assert all(ys[i] <= ys[i + 1] for i in range(len(ys) - 1))
    assert ys[-1] == 0.8


def _model(**kw):
    defaults = dict(
        model_id="test", name="test", arch_type="dense",
        total_params=10.0, active_params=10.0, release_date=None,
    )
    defaults.update(kw)
    return ModelRecord(**defaults)


def test_judge_underperforming():
    curve = fit_params_curve(
        "mmlu", "params_dense", params=[1.0, 10.0, 100.0], scores=[50.0, 60.0, 70.0]
    )
    config = CurvesConfig(percentile_threshold=25.0, z_threshold=-1.0, min_arch_points=5)
    # score well above the 10-param expectation -> in range
    good = judge(_model(), "mmlu", 80.0, [curve], config)
    assert not good["underperforming"]
    # score far below expectation -> underperforming
    bad = judge(_model(), "mmlu", 30.0, [curve], config)
    assert bad["underperforming"]
    assert bad["percentile"] < 25.0
