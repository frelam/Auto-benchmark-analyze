"""Decide whether a cluster needs drill-down diagnosis (design doc 4.3)."""

from __future__ import annotations

from benchmark_diagnosis.config import CurvesConfig


def should_drilldown(residual: dict, config: CurvesConfig) -> bool:
    """Return True when a cluster is judged "under-performing".

    The residual dict carries ``percentile`` (local residual percentile, higher
    is better), ``z_score`` (lower is worse) and ``underperforming`` (an earlier
    explicit judgment). Any single failing signal triggers the drill-down.

    Args:
        residual: Dict with keys ``percentile`` (float), ``z_score`` (float)
            and ``underperforming`` (bool).
        config: Curve judgment thresholds (section 4.2.4).

    Returns:
        True if the cluster should enter section-5 drill-down diagnosis.
    """
    percentile = float(residual.get("percentile", 100.0))
    z_score = float(residual.get("z_score", 0.0))
    underperforming = bool(residual.get("underperforming", False))
    return (
        underperforming
        or z_score < config.z_threshold
        or percentile < config.percentile_threshold
    )
