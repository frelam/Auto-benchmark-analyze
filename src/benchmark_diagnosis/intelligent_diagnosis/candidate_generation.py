"""Stage 1: candidate deficit capability generation (design doc v2 section 3).

For every below-expectation benchmark from Stage 0, flag the capabilities it
exercises:

* **fine mode** — item-level capability tags are available (``item_capabilities``
  from the benchmark tagger): compute per-capability sub-accuracy of the target
  model and compare it against the same items' historical peer distribution
  (percentile, rank method). Statistical guards: minimum item count and minimum
  peer count; small samples use the Wilson lower bound as a conservative
  estimate and get a screening-score penalty.
* **coarse mode** — only benchmark-level tags exist: all resolved tags enter the
  candidate set with a discounted screening score.

The OR-union happens here: one capability flagged by several benchmarks keeps
the max screening score and records every source benchmark (Stage 5 needs the
footprint).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from benchmark_diagnosis.intelligent_diagnosis.capability_taxonomy import (
    CapabilityTaxonomy,
)
from benchmark_diagnosis.intelligent_diagnosis.types import (
    CandidateCapability,
    SubAccuracy,
)

_WILSON_Z = 1.96  # 95% two-sided


def wilson_lower_bound(successes: float, n: int, z: float = _WILSON_Z) -> float:
    """Wilson score interval lower bound for a proportion (conservative, small-n safe)."""
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    margin = z * np.sqrt(max(0.0, (p * (1.0 - p) + z * z / (4.0 * n)) / n))
    return float(max(0.0, (center - margin) / denom))


def wilson_upper_bound(successes: float, n: int, z: float = _WILSON_Z) -> float:
    """Wilson score interval upper bound for a proportion."""
    if n <= 0:
        return 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    margin = z * np.sqrt(max(0.0, (p * (1.0 - p) + z * z / (4.0 * n)) / n))
    return float(min(1.0, (center + margin) / denom))


@dataclass
class CandidateConfig:
    """Stage 1 thresholds (mirrors config.diagnosis; defaults kept here for tests)."""

    percentile_threshold: float = 25.0
    min_items_per_capability: int = 8
    min_peers: int = 5
    low_support_penalty: float = 0.5  # screening score multiplier when guards fail
    coarse_penalty: float = 0.5  # coarse evidence is weaker: cap score at this


def _resolve_tag(taxonomy: CapabilityTaxonomy, tag: str) -> str | None:
    """Map a legacy/flat tag to a taxonomy node id via exact id or alias."""
    if taxonomy.get(tag) is not None:
        return tag
    for node in taxonomy.nodes.values():
        if tag in node.aliases:
            return node.id
    return None


def _target_sub_accuracy(
    items: pd.DataFrame,
    target_model_id: str,
    capability_items: list[str],
) -> tuple[float, int, list[float], float]:
    """Target sub-accuracy, item count, peer sub-accuracies, peer mean.

    ``items``: item-level rows (model_id, benchmark_id, item_id, correct) for one
    benchmark; ``capability_items``: item ids tagged with the capability.
    """
    tagged = items[items["item_id"].isin(capability_items)]
    if tagged.empty:
        return 0.0, 0, [], 0.0
    target = tagged[tagged["model_id"] == target_model_id]
    n_items = int(tagged["item_id"].nunique())
    if target.empty:
        return 0.0, n_items, [], 0.0
    sub_acc = float(target["correct"].astype(float).mean())
    peers = (
        tagged[tagged["model_id"] != target_model_id]
        .groupby("model_id")["correct"]
        .mean()
        .astype(float)
    )
    peer_list = [float(v) for v in peers]
    peer_mean = float(peers.mean()) if len(peer_list) else 0.0
    return sub_acc, n_items, peer_list, peer_mean


def _fine_candidates_for_benchmark(
    benchmark_id: str,
    items: pd.DataFrame,
    target_model_id: str,
    item_capabilities: dict[str, list[str]],
    config: CandidateConfig,
) -> list[SubAccuracy]:
    """Per-capability sub-accuracy evidence within one benchmark (fine mode)."""
    by_item: dict[str, list[str]] = {}
    for item_id, tags in item_capabilities.items():
        for tag in tags:
            by_item.setdefault(item_id, []).append(tag)

    # capability -> [item ids] for items that appear in this benchmark's rows
    cap_items: dict[str, list[str]] = {}
    for item_id in items["item_id"].unique():
        for tag in by_item.get(item_id, []):
            cap_items.setdefault(tag, []).append(item_id)

    out: list[SubAccuracy] = []
    for capability_id, cap_item_ids in cap_items.items():
        sub_acc, n_items, peer_list, peer_mean = _target_sub_accuracy(
            items, target_model_id, cap_item_ids
        )
        if n_items == 0 or not peer_list:
            continue
        percentile = float(
            percentileofscore(peer_list, sub_acc, kind="rank")
        )
        low_support = (
            n_items < config.min_items_per_capability
            or len(peer_list) < config.min_peers
        )
        out.append(
            SubAccuracy(
                benchmark_id=benchmark_id,
                capability_id=capability_id,
                sub_accuracy=sub_acc,
                peer_mean=peer_mean,
                peer_percentile=percentile,
                n_items=n_items,
                wilson_lb=wilson_lower_bound(sub_acc * n_items, n_items),
                wilson_ub=wilson_upper_bound(sub_acc * n_items, n_items),
                low_support=low_support,
            )
        )
    return out


def _coarse_candidates_for_benchmark(
    benchmark_id: str,
    verdict: dict[str, Any],
    coverage_row: dict[str, Any] | None,
    taxonomy: CapabilityTaxonomy,
    config: CandidateConfig,
) -> list[CandidateCapability]:
    """Benchmark-level tag evidence (coarse mode)."""
    tags = (coverage_row or {}).get("design_goal_tags") or []
    residual = verdict.get("residual")
    shortfall = max(0.0, -float(residual)) if residual is not None else (
        max(0.0, 1.0 - float(verdict.get("score_unit") or verdict.get("score") or 0.0) / 100.0)
    )
    score = min(1.0, shortfall * config.coarse_penalty)
    candidates: list[CandidateCapability] = []
    for tag in tags:
        cid = _resolve_tag(taxonomy, tag)
        if cid is None:
            # Unresolved legacy tag: keep it so evidence is not silently lost,
            # but flag it as unmatched (coarse evidence stays weak).
            cid = tag
        candidates.append(
            CandidateCapability(
                capability_id=cid,
                screening_score=score,
                evidence_mode="coarse",
                sources=[benchmark_id],
                source_scores={benchmark_id: score},
            )
        )
    return candidates


def generate_candidates(
    verdict_benchmarks: list[dict[str, Any]],
    *,
    taxonomy: CapabilityTaxonomy,
    target_model_id: str,
    item_scores: pd.DataFrame | None = None,
    item_capabilities: dict[str, list[str]] | None = None,
    coverage: list[dict[str, Any]] | None = None,
    config: CandidateConfig | None = None,
) -> list[CandidateCapability]:
    """Stage 1: OR-union of flagged capabilities across under-performing benchmarks.

    Args:
        verdict_benchmarks: Per-benchmark verdicts from ``expectation_curves.judge``
            (``benchmark_id``, ``score``, ``residual``, ``underperforming``,
            optional ``weight``). Only ``underperforming`` ones are screened.
        taxonomy: Hierarchical capability taxonomy (tag resolution).
        target_model_id: The evaluated model's id (its rows in ``item_scores``).
        item_scores: Historical item-level DataFrame with columns
            ``model_id`` / ``benchmark_id`` / ``item_id`` / ``correct``. When
            present *and* ``item_capabilities`` non-empty, fine mode is used.
        item_capabilities: ``item_id -> [capability_id]`` from the benchmark
            tagger; empty/None degrades every benchmark to coarse mode.
        coverage: Coverage asset rows (``benchmark_id``, ``design_goal_tags``)
            for coarse mode.
        config: Stage 1 thresholds.

    Returns:
        Candidate capabilities sorted by screening score (desc).
    """
    config = config or CandidateConfig()
    coverage_by_id = {row["benchmark_id"]: row for row in (coverage or [])}
    by_capability: dict[str, CandidateCapability] = {}

    def _merge(cand: CandidateCapability) -> None:
        existing = by_capability.get(cand.capability_id)
        if existing is None:
            by_capability[cand.capability_id] = cand
            return
        existing.screening_score = max(existing.screening_score, cand.screening_score)
        for src in cand.sources:
            if src not in existing.sources:
                existing.sources.append(src)
        existing.source_scores.update(cand.source_scores)
        existing.sub_accuracies.extend(cand.sub_accuracies)
        existing.evidence_mode = (
            "fine" if existing.evidence_mode == "fine" or cand.evidence_mode == "fine"
            else "coarse"
        )
        existing.low_support = existing.low_support or cand.low_support

    fine_available = item_scores is not None and not item_scores.empty and bool(item_capabilities)

    for verdict in verdict_benchmarks:
        if not verdict.get("underperforming"):
            continue
        bid = verdict["benchmark_id"]
        score = float(verdict.get("score") or 0.0)
        if score <= 0 and verdict.get("score") is not None:
            continue  # no data for this benchmark in this run

        if fine_available and item_scores is not None:
            bench_items = item_scores[item_scores["benchmark_id"] == bid]
            if not bench_items.empty:
                sub_accs = _fine_candidates_for_benchmark(
                    bid,
                    bench_items,
                    target_model_id,
                    item_capabilities or {},
                    config,
                )
                for sa in sub_accs:
                    if sa.peer_percentile >= config.percentile_threshold:
                        continue  # not below expectation on this capability
                    support = 1.0 if not sa.low_support else config.low_support_penalty
                    _merge(
                        CandidateCapability(
                            capability_id=sa.capability_id,
                            screening_score=min(
                                1.0, (1.0 - sa.peer_percentile / 100.0) * support
                            ),
                            evidence_mode="fine",
                            sources=[bid],
                            sub_accuracies=[sa],
                            low_support=sa.low_support,
                        )
                    )
                continue  # fine evidence consumed this benchmark

        for cand in _coarse_candidates_for_benchmark(
            bid, verdict, coverage_by_id.get(bid), taxonomy, config
        ):
            _merge(cand)

    result = list(by_capability.values())
    result.sort(key=lambda c: c.screening_score, reverse=True)
    return result
