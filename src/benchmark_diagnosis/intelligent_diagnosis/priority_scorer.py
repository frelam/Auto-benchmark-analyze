"""Stage 5: priority scoring — ExpectedGain / Cost / Priority (design v2 §7).

The core idea (SliceTeller-style "repair cost vs expected gain" ranking): a
capability may drag down *several* benchmarks, so its expected gain sums the
shortfall over every source benchmark that flagged it, not just one:

    gap_b,c      = max(0, peer_mean_sub_acc - sub_acc) * item_weight_share   (fine)
                 = max(0, -residual_b)                                       (coarse)
    ExpectedGain = sum_b weight_b * gap_b,c
    Cost         = base_cost[suggestion_type] * calibrated_ratio
    Priority     = confidence_numeric * ExpectedGain / Cost

Only benchmarks with a *measured* shortfall (the candidate's sources) count:
a benchmark merely tagged with the capability but at/above expectation
contributes zero gap. Small-sample honesty: ``gain_lower`` recomputes the
fine-mode gain with the Wilson *upper* bound of the sub-accuracy (the model's
true accuracy could be higher than measured, so the true gap — and the gain —
could be smaller), so a few lucky items cannot inflate a candidate's rank. The
``calibration`` dict (from the feedback module) scales both gains and costs
using real execution outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmark_diagnosis.intelligent_diagnosis.confidence_fusion import (
    confidence_numeric,
)
from benchmark_diagnosis.intelligent_diagnosis.types import (
    CandidateCapability,
    ConfidenceLevel,
    FusedVerdict,
    PriorityItem,
    SuggestionType,
)


@dataclass
class PriorityConfig:
    """Stage 5 thresholds."""

    min_gain: float = 0.0  # below this the item is dropped from the ranking
    cost_clamp: tuple[float, float] = (0.5, 3.0)


@dataclass
class Calibration:
    """Feedback-driven scaling (Stage 7 output, stored as versioned asset)."""

    costs: dict[str, float] = field(default_factory=dict)  # suggestion_type -> ratio
    gain_scale: float = 1.0
    n_logs: int = 0


def _benchmark_weights(portfolios: list[dict[str, Any]]) -> dict[str, float]:
    """benchmark_id -> portfolio weight (defaults 1.0 when absent)."""
    weights: dict[str, float] = {}
    for pf in portfolios:
        for b in pf.get("benchmarks") or []:
            weights[str(b["benchmark_id"])] = float(b.get("weight") or 1.0)
    return weights


def score_priorities(
    candidates: list[CandidateCapability],
    verdicts: dict[str, FusedVerdict],
    *,
    portfolios: list[dict[str, Any]] | None = None,
    base_costs: dict[SuggestionType, float] | None = None,
    calibration: Calibration | None = None,
    item_totals: dict[str, int] | None = None,
    config: PriorityConfig | None = None,
) -> list[PriorityItem]:
    """Rank fused candidates by expected gain per unit cost.

    The footprint of a candidate is its Stage-1 *source* benchmarks — the ones
    with a measured below-expectation shortfall. Benchmarks merely tagged with
    the capability but at/above expectation contribute zero gap (fixing the
    capability cannot lift them), so they are not summed.

    Args:
        candidates: Stage 1 candidates (carry per-benchmark sub-accuracy /
            shortfall evidence).
        verdicts: ``capability_id -> FusedVerdict`` from Stage 4.
        portfolios: Cluster portfolios (benchmark weights).
        base_costs: Base cost table keyed by :class:`SuggestionType` (defaults:
            packaged ``cost_table.yaml`` loaded by the orchestrator).
        calibration: Stage 7 calibration (cost ratios + gain scale).
        item_totals: ``benchmark_id -> total item count`` (fine-mode share).
        config: Thresholds.

    Returns:
        Priority items sorted by priority (desc). Low-confidence verdicts are
        excluded from the ranking (they go to human review).
    """
    config = config or PriorityConfig()
    base_costs = base_costs or {}
    calibration = calibration or Calibration()
    weights = _benchmark_weights(portfolios or [])
    item_totals = item_totals or {}

    items: list[PriorityItem] = []
    for cand in candidates:
        verdict = verdicts.get(cand.capability_id)
        if verdict is None or verdict.confidence == ConfidenceLevel.LOW:
            continue

        gain, gain_lower = _expected_gain(cand, weights, item_totals)
        gain *= calibration.gain_scale
        gain_lower *= calibration.gain_scale

        stype = verdict.suggestion_type
        base = base_costs.get(stype, 1.0)
        ratio = calibration.costs.get(stype.value, 1.0)
        lo, hi = config.cost_clamp
        cost = base * min(hi, max(lo, ratio))
        priority = confidence_numeric(verdict.confidence) * gain / cost if cost > 0 else 0.0

        if gain < config.min_gain:
            continue
        items.append(
            PriorityItem(
                capability_id=cand.capability_id,
                suggestion_type=stype,
                confidence=verdict.confidence,
                expected_gain=round(gain, 4),
                gain_lower=round(gain_lower, 4),
                cost=round(cost, 4),
                priority=round(priority, 4),
                sources=sorted(cand.sources),
            )
        )

    items.sort(key=lambda item: item.priority, reverse=True)
    return items


def _expected_gain(
    cand: CandidateCapability,
    weights: dict[str, float],
    item_totals: dict[str, int],
) -> tuple[float, float]:
    """(gain, gain_lower) for one candidate over its source benchmark footprint.

    Coarse mode: the per-benchmark shortfall recorded at Stage 1
    (residual-based gap). Fine mode: ``(peer_mean - sub_accuracy) * item_share``,
    with the conservative variant using the Wilson upper bound of the model's
    accuracy.
    """
    sub_by_bench: dict[str, Any] = {sa.benchmark_id: sa for sa in cand.sub_accuracies}
    gain = 0.0
    gain_lower = 0.0
    for bid in cand.sources:
        weight = weights.get(bid, 1.0)
        sa = sub_by_bench.get(bid)
        if sa is not None and cand.evidence_mode == "fine":
            total = max(1, item_totals.get(bid, sa.n_items))
            share = sa.n_items / total
            gap = max(0.0, sa.peer_mean - sa.sub_accuracy) * share
            # conservative gain: the model's true accuracy could be as high as
            # the Wilson upper bound, so the true gap could be smaller
            gap_lower = max(0.0, sa.peer_mean - sa.wilson_ub) * share
        else:
            gap = cand.source_scores.get(bid, cand.screening_score)
            gap_lower = gap
        gain += weight * gap
        gain_lower += weight * gap_lower
    return gain, gain_lower
