"""Stage 7: execution feedback loop (design doc v2 section 9).

Every time a suggestion is actually executed (data mix adjusted, rejection
sampling run, synthesis pipeline built) and the benchmarks re-measured, log the
real outcome. Once enough logs accumulate, ``recalibrate`` re-estimates:

* per-suggestion-type **cost ratios** (actual_cost / predicted_cost),
* a global **gain scale** (actual_gain / predicted_gain),

smoothed against the identity prior and persisted as the versioned
``calibration`` asset. Stage 5 reads it, so priority ranking gets more accurate
the longer the tool runs — the "boosting with real feedback" idea applied at
the human diagnosis-fix loop level.
"""

from __future__ import annotations

import datetime as dt
import statistics
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from benchmark_diagnosis.core import db
from benchmark_diagnosis.core.schema import ExecutionLogRecord
from benchmark_diagnosis.intelligent_diagnosis.priority_scorer import Calibration

_CALIBRATION_ASSET = "calibration"
_EPS = 1e-9


def log_execution(
    session: Session,
    *,
    capability_id: str,
    suggestion_type: str,
    predicted_gain: float,
    actual_gain: float,
    predicted_cost: float,
    actual_cost: float,
    note: str | None = None,
) -> int:
    """Record one executed suggestion's outcome; returns the row id."""
    if predicted_cost <= 0:
        raise ValueError("predicted_cost must be > 0")
    row = ExecutionLogRecord(
        capability_id=capability_id,
        suggestion_type=suggestion_type,
        predicted_gain=float(predicted_gain),
        actual_gain=float(actual_gain),
        predicted_cost=float(predicted_cost),
        actual_cost=float(actual_cost),
        note=note,
    )
    session.add(row)
    session.commit()
    return row.id


def list_executions(
    session: Session,
    *,
    suggestion_type: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return recent execution logs as dicts (newest first)."""
    stmt = select(ExecutionLogRecord).order_by(ExecutionLogRecord.id.desc()).limit(limit)
    if suggestion_type is not None:
        stmt = (
            select(ExecutionLogRecord)
            .where(ExecutionLogRecord.suggestion_type == suggestion_type)
            .order_by(ExecutionLogRecord.id.desc())
            .limit(limit)
        )
    return [
        {
            "id": r.id,
            "capability_id": r.capability_id,
            "suggestion_type": r.suggestion_type,
            "predicted_gain": r.predicted_gain,
            "actual_gain": r.actual_gain,
            "predicted_cost": r.predicted_cost,
            "actual_cost": r.actual_cost,
            "created_at": r.created_at.isoformat(),
            "note": r.note,
        }
        for r in session.scalars(stmt)
    ]


def recalibrate(
    session: Session,
    *,
    smoothing: float = 0.5,
    min_logs_per_type: int = 3,
    cost_clamp: tuple[float, float] = (0.5, 3.0),
    gain_clamp: tuple[float, float] = (0.5, 2.0),
) -> Calibration:
    """Re-estimate cost ratios + gain scale from the execution log.

    Args:
        session: DB session.
        smoothing: Weight of the measured mean vs the identity prior
            (0 = keep prior, 1 = trust logs fully).
        min_logs_per_type: Logs below this per suggestion_type are ignored
            (keep the prior ratio).
        cost_clamp / gain_clamp: Bounds on the adjusted ratios.

    Returns:
        The new :class:`Calibration`, persisted as the ``calibration`` asset.
    """
    rows = list(session.scalars(select(ExecutionLogRecord)))
    per_type: dict[str, list[float]] = {}
    gain_ratios: list[float] = []
    for r in rows:
        per_type.setdefault(r.suggestion_type, []).append(
            r.actual_cost / max(r.predicted_cost, _EPS)
        )
        gain_ratios.append(r.actual_gain / max(r.predicted_gain, _EPS))

    lo_c, hi_c = cost_clamp
    lo_g, hi_g = gain_clamp
    costs: dict[str, float] = {}
    for stype, ratios in per_type.items():
        if len(ratios) < min_logs_per_type:
            continue
        mean_ratio = statistics.fmean(ratios)
        adjusted = 1.0 + smoothing * (mean_ratio - 1.0)
        costs[stype] = min(hi_c, max(lo_c, adjusted))

    gain_scale = 1.0
    if len(gain_ratios) >= min_logs_per_type:
        mean_gain = statistics.fmean(gain_ratios)
        gain_scale = min(hi_g, max(lo_g, 1.0 + smoothing * (mean_gain - 1.0)))

    calibration = Calibration(costs=costs, gain_scale=round(gain_scale, 4), n_logs=len(rows))
    db.save_asset(
        session,
        _CALIBRATION_ASSET,
        {
            "costs": costs,
            "gain_scale": calibration.gain_scale,
            "n_logs": len(rows),
            "updated_at": dt.datetime.utcnow().isoformat(),
        },
        note="feedback recalibration of Stage 5 cost table / gain scale",
    )
    return calibration


def load_calibration(session: Session) -> Calibration:
    """Load the latest calibration asset, or the identity calibration."""
    payload = db.load_latest_asset(session, _CALIBRATION_ASSET)
    if not isinstance(payload, dict):
        return Calibration()
    return Calibration(
        costs={str(k): float(v) for k, v in (payload.get("costs") or {}).items()},
        gain_scale=float(payload.get("gain_scale") or 1.0),
        n_logs=int(payload.get("n_logs") or 0),
    )
