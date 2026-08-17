"""Correlation + bad-case driven capability-deficit inference (engine input).

Given a cluster's low-scoring benchmarks and the coverage table — which records
which declared capability tags each benchmark exercises and how reliably — this
module estimates *which capabilities the model is missing* and how strongly. The
benchmark->capability link IS the correlation structure: benchmarks that co-vary
across models were grouped into a cluster, and each benchmark's declared
design-goal tags name the capabilities its score reflects.

Failure-mode fractions from bad-case analysis reinforce the relevant capabilities
through a hint map (``failure_mode_capability_map`` in the experience base), so
the deficit profile is driven by benchmark evidence *and* per-case evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# How much a failure-mode fraction counts relative to benchmark-level evidence.
# The benchmark signal sums to ``magnitude``; each bad-case tag adds
# ``frac * magnitude * FM_REINFORCEMENT``, so a dominant failure mode can raise
# the associated capability but never outweigh all benchmark evidence combined.
FM_REINFORCEMENT = 0.5

# Tags below this normalized strength are treated as noise.
NOISE_FLOOR = 0.05


@dataclass
class CapabilityDeficit:
    """Which capabilities the model is missing, and the evidence behind it."""

    strengths: dict[str, float]  # normalized capability_tag -> strength (sums ~1)
    magnitude: float  # absolute shortfall scale (unnormalized signal sum)
    drivers: list[tuple[str, float]] = field(default_factory=list)  # (benchmark, shortfall)
    narrative: str = ""


def infer_capability_deficit(
    coverage: list[dict[str, Any]],
    verdict_benchmarks: list[dict[str, Any]],
    failure_modes: dict[str, float] | None = None,
    fm_capability_map: dict[str, list[str]] | None = None,
) -> CapabilityDeficit:
    """Infer the capability-deficit profile for one cluster.

    Args:
        coverage: Coverage asset rows (each a dict with ``benchmark_id``,
            ``design_goal_tags``, ``reliability_score``, ``saturated_flag``,
            ``design_goal_agreement_score``).
        verdict_benchmarks: Per-benchmark verdicts ``{benchmark_id, weight,
            score, residual}`` (residual optional; falls back to
            ``1 - normalized score`` when absent).
        failure_modes: Mapping ``failure_mode_id -> fraction`` from bad-case
            analysis (may be empty).
        fm_capability_map: Mapping ``failure_mode_id -> [capability_tags]`` from
            the experience base (may be None / empty).

    Returns:
        A :class:`CapabilityDeficit`. Empty ``verdict_benchmarks`` or a zero
        signal total yields ``strengths={}``, ``magnitude=0.0``.
    """
    by_benchmark = {row["benchmark_id"]: row for row in coverage}
    deficit: dict[str, float] = {}
    drivers: list[tuple[str, float]] = []
    total_signal = 0.0

    for b in verdict_benchmarks:
        if b.get("score") is None:
            continue
        row = by_benchmark.get(b["benchmark_id"])
        if row is None:
            continue
        shortfall = _shortfall(b)
        if shortfall <= 0:
            continue
        weight = float(b.get("weight") or 0.0)
        reliability = float(row.get("reliability_score") or 1.0)
        agreement = float(row.get("design_goal_agreement_score") or 1.0)
        saturation_penalty = 0.5 if row.get("saturated_flag") else 1.0
        signal = weight * shortfall * reliability * agreement * saturation_penalty
        if signal <= 0:
            continue
        total_signal += signal
        drivers.append((b["benchmark_id"], shortfall))
        for tag in row.get("design_goal_tags") or []:
            deficit[tag] = deficit.get(tag, 0.0) + signal

    # Bad-case evidence reinforces the capabilities implied by each failure mode.
    for mode, fraction in (failure_modes or {}).items():
        for tag in (fm_capability_map or {}).get(mode, []):
            deficit[tag] = deficit.get(tag, 0.0) + fraction * total_signal * FM_REINFORCEMENT

    if not deficit:
        return CapabilityDeficit(strengths={}, magnitude=0.0, drivers=[], narrative="")

    total = sum(deficit.values())
    strengths = {tag: value / total for tag, value in deficit.items()}
    strengths = {tag: s for tag, s in strengths.items() if s >= NOISE_FLOOR}
    if strengths:
        renormalize = sum(strengths.values())
        strengths = {tag: s / renormalize for tag, s in strengths.items()}
    drivers.sort(key=lambda pair: pair[1], reverse=True)
    narrative = _narrative(strengths, drivers)
    return CapabilityDeficit(
        strengths=strengths, magnitude=total, drivers=drivers, narrative=narrative
    )


def _shortfall(benchmark: dict[str, Any]) -> float:
    """How far below expectation this benchmark scores (0-1 units).

    Uses the fitted-curve residual when available (true shortfall vs. the
    expectation for this model's size/time); otherwise falls back to how far the
    raw score is below a perfect 1.0.
    """
    residual = benchmark.get("residual")
    if residual is not None:
        # residual = score_unit - predicted; a negative residual means the model
        # is below its expectation curve, so the shortfall is -residual.
        return max(0.0, -float(residual))
    score = float(benchmark["score"])
    norm = score / 100.0 if score > 1.5 else score
    return max(0.0, 1.0 - norm)


def _narrative(
    strengths: dict[str, float], drivers: list[tuple[str, float]]
) -> str:
    """Deterministic Chinese narrative for the report / reason chain."""
    if not strengths:
        return ""
    top = "、".join(
        f"{tag}({strength:.2f})"
        for tag, strength in sorted(strengths.items(), key=lambda kv: kv[1], reverse=True)
    )
    drives = "、".join(f"{bid}({shortfall:.2f})" for bid, shortfall in drivers[:3])
    return f"该簇缺失能力：{top}；主要驱动：{drives}"
