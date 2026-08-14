"""Build the versioned capability-coverage table (design doc section 2.6).

Given either an mIRT fit (item-level) or a factor-analysis fit (aggregate-only),
produce one :class:`CoverageEntry` per benchmark quantifying: primary cluster,
discrimination profile, coverage breadth, reliability, saturation, and agreement
with the benchmark's declared design goal.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from benchmark_diagnosis.capability_analysis.design_goal_validation import (
    agreement_score,
)
from benchmark_diagnosis.capability_analysis.factor_analysis import FactorModel
from benchmark_diagnosis.capability_analysis.mirt_fit import MIRTResult
from benchmark_diagnosis.core.types import CoverageEntry, Granularity, ScoringMethod


def _to_unit(values: np.ndarray) -> np.ndarray:
    """Normalize scores that look like 0-100 into 0-1."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size and arr.max() > 1.5:
        arr = arr / 100.0
    return arr


def _entropy(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    w = np.abs(w)
    total = w.sum()
    if total <= 0:
        return 0.0
    p = w / total
    p = p[p > 1e-12]
    n = max(1, len(weights))
    return float(-(p * np.log(p)).sum() / np.log(n))


def _scoring_method(meta: Any, benchmark_id: str) -> ScoringMethod:
    raw = None
    if isinstance(meta, dict):
        raw = meta.get("scoring_method")
    elif meta is not None:
        raw = getattr(meta, "scoring_method", None)
    try:
        return ScoringMethod(raw)
    except ValueError:
        return ScoringMethod.RULE_VERIFIED


def _declared_tags(meta: Any, benchmark_id: str) -> list[str]:
    if isinstance(meta, dict):
        return list(meta.get("declared_tags") or [])
    if meta is not None:
        return list(getattr(meta, "declared_tags", None) or [])
    return []


def _cluster_tag_map(
    benchmark_ids: list[str], clusters: list[int], meta: dict[str, Any]
) -> dict[str, set[str]]:
    """Map primary-cluster label -> union of declared tags of its benchmarks."""
    out: dict[str, set[str]] = defaultdict(set)
    for bid, label in zip(benchmark_ids, clusters, strict=True):
        out[f"cluster_{label}"] |= set(_declared_tags(meta.get(bid), bid))
    return dict(out)


def build_coverage_table(
    *,
    mirt: MIRTResult | None = None,
    factor: FactorModel | None = None,
    benchmark_meta: dict[str, Any] | None = None,
    item_benchmark: dict[str, str] | None = None,
    scores: pd.DataFrame | None = None,
) -> list[CoverageEntry]:
    """Build coverage entries from an mIRT fit (preferred) or factor model.

    Args:
        mirt: Item-level mIRT result; when provided, produces item-level entries.
        factor: Aggregate factor-analysis result (used when ``mirt`` is None).
        benchmark_meta: Mapping benchmark_id -> object with ``declared_tags`` /
            ``scoring_method`` attributes (or plain dicts).
        item_benchmark: Mapping item_id -> benchmark_id (required for mIRT path).
        scores: Model x benchmark matrix, used to compute the saturation flag.

    Returns:
        A list of :class:`CoverageEntry`, one per benchmark.
    """
    meta = benchmark_meta or {}
    score_frame = scores

    if mirt is not None:
        return _from_mirt(mirt, meta, item_benchmark or {}, score_frame)
    if factor is not None:
        return _from_factor(factor, meta, score_frame)
    raise ValueError("provide either `mirt` or `factor`")


def _saturation(score_frame: pd.DataFrame | None, benchmark_id: str) -> bool:
    if score_frame is None or benchmark_id not in score_frame.columns:
        return False
    col = score_frame[benchmark_id].dropna()
    if col.empty:
        return False
    top = float(_to_unit(col.values).max())
    return top >= 0.95


def _from_mirt(
    mirt: MIRTResult,
    meta: dict[str, Any],
    item_benchmark: dict[str, str],
    score_frame: pd.DataFrame | None,
) -> list[CoverageEntry]:
    by_benchmark: dict[str, list[int]] = defaultdict(list)
    for idx, item_id in enumerate(mirt.item_ids):
        by_benchmark[item_benchmark.get(item_id, "unknown")].append(idx)

    abs_alpha = np.abs(mirt.alpha)  # (n_items, dims)
    signed_alpha = np.asarray(mirt.alpha, dtype=np.float64)
    theta_mean = mirt.theta.mean(axis=0)

    # Precompute per-benchmark raw reliability for later min-max normalization.
    raw_rel: dict[str, float] = {}
    profiles: dict[str, np.ndarray] = {}
    signed_profiles: dict[str, np.ndarray] = {}
    for bid, idxs in by_benchmark.items():
        prof = abs_alpha[idxs].mean(axis=0)
        profiles[bid] = prof
        signed_profiles[bid] = signed_alpha[idxs].mean(axis=0)
        # Fisher information per item at mean ability: ||alpha||^2 * p(1-p).
        info = 0.0
        for i in idxs:
            logit = float(theta_mean @ mirt.alpha[i] - mirt.beta[i])
            p = 1.0 / (1.0 + np.exp(-logit))
            info += float(np.dot(mirt.alpha[i], mirt.alpha[i])) * p * (1 - p)
        raw_rel[bid] = info / len(idxs) if idxs else 0.0

    rel_min = min(raw_rel.values()) if raw_rel else 0.0
    rel_max = max(raw_rel.values()) if raw_rel else 1.0

    # Primary dimension per benchmark for cluster labeling.
    primary_dim: dict[str, int] = {
        bid: int(np.argmax(prof)) for bid, prof in profiles.items()
    }
    cluster_tag_map: dict[str, set[str]] = defaultdict(set)
    for bid in profiles:
        cluster_tag_map[f"dim_{primary_dim[bid]}"] |= set(_declared_tags(meta.get(bid), bid))

    entries: list[CoverageEntry] = []
    for bid in sorted(profiles):
        prof = profiles[bid]
        norm = prof.sum() or 1.0
        profile_dict = {f"dim_{i}": float(v / norm) for i, v in enumerate(prof)}
        cluster = f"dim_{primary_dim[bid]}"
        own_tags = _declared_tags(meta.get(bid), bid)
        cluster_tags = sorted(cluster_tag_map[cluster])  # full union
        agree = agreement_score(own_tags, cluster_tags)
        reliability = 1.0 if rel_max == rel_min else (raw_rel[bid] - rel_min) / (rel_max - rel_min)
        entries.append(
            CoverageEntry(
                benchmark_id=bid,
                primary_cluster=cluster,
                discrimination_profile=profile_dict,
                factor_loadings=[float(v) for v in signed_profiles[bid]],
                coverage_breadth_score=_entropy(prof),
                reliability_score=float(reliability),
                saturated_flag=_saturation(score_frame, bid),
                scoring_method=_scoring_method(meta.get(bid), bid),
                design_goal_tags=own_tags,
                design_goal_agreement_score=agree,
                granularity=Granularity.ITEM_LEVEL,
            )
        )
    return entries


def _from_factor(
    factor: FactorModel,
    meta: dict[str, Any],
    score_frame: pd.DataFrame | None,
) -> list[CoverageEntry]:
    loadings = np.abs(factor.loadings)  # (n_benchmarks, n_factors)
    cluster_map = _cluster_tag_map(factor.benchmark_ids, factor.clusters, meta)

    entries: list[CoverageEntry] = []
    for i, bid in enumerate(factor.benchmark_ids):
        load = loadings[i]  # signed; keep sign for the map embedding
        norm = load.sum() or 1.0
        profile = {f"f{j}": float(v / norm) for j, v in enumerate(load)}
        cluster = f"cluster_{factor.clusters[i]}"
        own_tags = _declared_tags(meta.get(bid), bid)
        cluster_tags = sorted(cluster_map[cluster])  # full union: coherent clusters score high
        communality = float(np.clip((load**2).sum(), 0.0, 1.0))
        entries.append(
            CoverageEntry(
                benchmark_id=bid,
                primary_cluster=cluster,
                discrimination_profile=profile,
                factor_loadings=[float(v) for v in factor.loadings[i]],
                coverage_breadth_score=_entropy(load),
                reliability_score=communality,
                saturated_flag=_saturation(score_frame, bid),
                scoring_method=_scoring_method(meta.get(bid), bid),
                design_goal_tags=own_tags,
                design_goal_agreement_score=agreement_score(own_tags, cluster_tags),
                granularity=Granularity.AGGREGATE_ONLY,
            )
        )
    return entries
