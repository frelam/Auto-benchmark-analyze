"""Expectation curves: scaling-law fits and "under-performing" judgment.

Two complementary curve families (design doc section 4.2):

* **Group A** (params -> score): log-linear fit of ``logit(score)`` vs
  ``log10(params)``, fitted separately for dense / MoE and a unified
  active-params curve.
* **Group B** (time -> score): the frontier envelope of best score over release
  date, so closed-source models without published parameter counts still count.

Residuals are converted to a local percentile / z-score (never absolute deltas)
so they are comparable across benchmarks.
"""

from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
from scipy.stats import percentileofscore
from sqlalchemy import select
from sqlalchemy.orm import Session

from benchmark_diagnosis.config import CurvesConfig
from benchmark_diagnosis.core.schema import ModelRecord, ScoreRecord
from benchmark_diagnosis.core.types import ExpectationCurve


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(x / (1.0 - x))


def _logistic(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def _to_unit(score: float) -> float:
    if score > 1.5:
        return score / 100.0
    return float(score)


def _epoch_days(date: dt.date | None) -> float | None:
    if date is None:
        return None
    return (date - dt.date(2020, 1, 1)).days


def fit_params_curve(
    benchmark_id: str,
    kind: str,
    params: list[float],
    scores: list[float],
) -> ExpectationCurve | None:
    """Fit a log-linear params->score curve. ``kind`` in {params_dense, params_moe,
    params_active}."""
    valid = [(p, _to_unit(s)) for p, s in zip(params, scores, strict=True) if p and p > 0]
    if len(valid) < 2:
        return None
    xs = [p for p, _ in valid]
    ys = [s for _, s in valid]
    with warnings.catch_warnings():
        # Sparse open-model coverage (a handful of published parameter counts)
        # can make the log-log design matrix near-singular; the fit is still
        # usable as a coarse trend, so silence the numerical noise.
        warnings.simplefilter("ignore")
        slope, intercept = np.polyfit(np.log10(xs), _logit(np.asarray(ys)), 1)
    return ExpectationCurve(
        benchmark_id=benchmark_id,
        kind=kind,
        coefficients={
            "slope": float(slope),
            "intercept": float(intercept),
            "x_space": "log10_params",
            "y_space": "logit",
        },
        points=[(float(x), float(y)) for x, y in valid],
    )


def fit_time_frontier(
    benchmark_id: str,
    dates: list[dt.date],
    scores: list[float],
) -> ExpectationCurve | None:
    """Fit the non-decreasing frontier envelope of score over release date."""
    valid = sorted(
        (_epoch_days(d), _to_unit(s))
        for d, s in zip(dates, scores, strict=True)
        if d is not None
    )
    if not valid:
        return None
    xs = [x for x, _ in valid]
    ys = [y for _, y in valid]
    running_max = np.maximum.accumulate(ys)
    return ExpectationCurve(
        benchmark_id=benchmark_id,
        kind="time_frontier",
        coefficients={},
        points=[(float(x), float(y)) for x, y in zip(xs, running_max, strict=True)],
    )


def fit_curves(session: Session) -> list[ExpectationCurve]:
    """Fit all expectation curves from the public reference data in the DB."""
    rows = session.execute(
        select(
            ScoreRecord.benchmark_id,
            ScoreRecord.score,
            ModelRecord.arch_type,
            ModelRecord.total_params,
            ModelRecord.active_params,
            ModelRecord.release_date,
        )
        .join(ModelRecord, ModelRecord.model_id == ScoreRecord.model_id)
        .where(ScoreRecord.item_id.is_(None), ScoreRecord.score.is_not(None))
    ).all()

    per_benchmark: dict[str, dict[str, list]] = {}
    for bid, score, arch, total, active, date in rows:
        bucket = per_benchmark.setdefault(
            bid,
            {
                "params_dense": [],
                "params_moe": [],
                "params_active": [],
                "dates": [],
                "scores": [],
            },
        )
        if arch == "dense" and total:
            bucket["params_dense"].append((total, score))
        elif arch == "moe" and total:
            bucket["params_moe"].append((total, score))
        if active:
            bucket["params_active"].append((active, score))
        if date is not None:
            bucket["dates"].append(date)
            bucket["scores"].append(score)

    curves: list[ExpectationCurve] = []
    for bid, b in per_benchmark.items():
        for kind in ("params_dense", "params_moe", "params_active"):
            pts = b[kind]
            if not pts:
                continue
            curve = fit_params_curve(bid, kind, [p for p, _ in pts], [s for _, s in pts])
            if curve is not None:
                curves.append(curve)
        frontier = fit_time_frontier(bid, b["dates"], b["scores"])
        if frontier is not None:
            curves.append(frontier)
    return curves


def predict(curve: ExpectationCurve, x: float) -> float:
    """Predict the expected score at parameter count / time ``x``."""
    if curve.kind.startswith("params"):
        slope = curve.coefficients["slope"]
        intercept = curve.coefficients["intercept"]
        return _logistic(slope * np.log10(x) + intercept)
    if curve.kind == "time_frontier":
        xs = [p[0] for p in curve.points]
        ys = [p[1] for p in curve.points]
        return float(np.interp(x, xs, ys, left=ys[0], right=ys[-1]))
    raise ValueError(f"unknown curve kind {curve.kind!r}")


def _pick_params_curve(
    curves: list[ExpectationCurve], arch: str, config: CurvesConfig
) -> ExpectationCurve | None:
    kind = f"params_{arch}"
    preferred = [c for c in curves if c.kind == kind]
    if preferred and len(preferred[0].points) >= config.min_arch_points:
        return preferred[0]
    fallback = [c for c in curves if c.kind == "params_active"]
    if fallback:
        return fallback[0]
    if preferred:
        return preferred[0]
    others = [c for c in curves if c.kind.startswith("params")]
    return max(others, key=lambda c: len(c.points), default=None)


def judge(
    model: ModelRecord,
    benchmark_id: str,
    score: float,
    curves: list[ExpectationCurve],
    config: CurvesConfig,
) -> dict:
    """Compute the local percentile / z-score of a model's score vs its curve.

    Returns a dict with keys: ``score``, ``score_unit``, ``curve_kind``,
    ``predicted``, ``residual``, ``percentile``, ``z_score``, ``frontier_gap``,
    ``underperforming``.
    """
    score_unit = _to_unit(score)
    relevant = [c for c in curves if c.benchmark_id == benchmark_id]
    result: dict = {
        "benchmark_id": benchmark_id,
        "score": score,
        "score_unit": score_unit,
        "curve_kind": None,
        "predicted": None,
        "residual": None,
        "percentile": 50.0,
        "z_score": 0.0,
        "frontier_gap": None,
        "underperforming": False,
    }

    params_curve = _pick_params_curve(relevant, model.arch_type, config)
    if params_curve is not None and model.total_params:
        x = model.total_params
        predicted = predict(params_curve, x)
        residual = score_unit - predicted
        ref_residuals = [y - predict(params_curve, px) for px, y in params_curve.points]
        ref_residuals = np.asarray(ref_residuals, dtype=np.float64)
        percentile = float(percentileofscore(ref_residuals, residual, kind="rank"))
        std = float(ref_residuals.std())
        z = float((residual - ref_residuals.mean()) / std) if std > 1e-12 else 0.0
        result.update(
            curve_kind=params_curve.kind,
            predicted=predicted,
            residual=residual,
            percentile=percentile,
            z_score=z,
        )

    frontier = next(
        (c for c in relevant if c.kind == "time_frontier"), None
    )
    if frontier is not None:
        today = _epoch_days(dt.date.today()) or 0.0
        frontier_score = predict(frontier, today)
        result["frontier_gap"] = frontier_score - score_unit

    result["underperforming"] = (
        result["percentile"] < config.percentile_threshold
        or result["z_score"] < config.z_threshold
    )
    return result
