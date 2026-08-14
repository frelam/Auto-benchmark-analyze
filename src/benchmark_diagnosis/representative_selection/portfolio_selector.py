"""Representative benchmark selection (design doc section 3).

For each capability cluster, greedily pick a small combo of benchmarks that
maximizes combined coverage breadth while preferring a rule_verified anchor and
dropping candidates that are too highly correlated with what was already
selected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark_diagnosis.core.types import (
    ClusterPortfolio,
    CoverageEntry,
    ScoringMethod,
)


def select_portfolios(
    coverage: list[CoverageEntry],
    *,
    correlation: pd.DataFrame | None = None,
    min_reliability: float = 0.0,
    max_corr: float = 0.9,
    combo_size: int = 3,
) -> list[ClusterPortfolio]:
    """Build a representative benchmark combo per capability cluster.

    Args:
        coverage: Capability-coverage rows (one per benchmark).
        correlation: Optional benchmark_id x benchmark_id correlation matrix
            used to drop redundant candidates.
        min_reliability: Minimum reliability_score for a candidate to qualify.
        max_corr: Drop a candidate whose correlation with any already-selected
            benchmark exceeds this value.
        combo_size: Maximum number of benchmarks per combo.

    Returns:
        One :class:`ClusterPortfolio` per cluster, sorted by cluster_id.
    """
    by_cluster: dict[str, list[CoverageEntry]] = {}
    for entry in coverage:
        by_cluster.setdefault(entry.primary_cluster, []).append(entry)

    portfolios: list[ClusterPortfolio] = []
    for cluster_id in sorted(by_cluster):
        selected = _select_for_cluster(
            by_cluster[cluster_id],
            correlation=correlation,
            min_reliability=min_reliability,
            max_corr=max_corr,
            combo_size=combo_size,
        )
        portfolios.append(ClusterPortfolio(cluster_id=cluster_id, benchmarks=selected))
    return portfolios


def _select_for_cluster(
    entries: list[CoverageEntry],
    *,
    correlation: pd.DataFrame | None,
    min_reliability: float,
    max_corr: float,
    combo_size: int,
) -> list[dict[str, float | str]]:
    """Greedy combo selection for a single cluster (weights normalized to 1.0)."""
    candidates = [
        entry
        for entry in entries
        if not entry.saturated_flag and entry.reliability_score >= min_reliability
    ]
    if not candidates:
        return []

    dims = sorted(
        {key for entry in candidates for key in entry.discrimination_profile}
    )
    profiles: dict[str, np.ndarray] = {
        entry.benchmark_id: np.array(
            [entry.discrimination_profile.get(key, 0.0) for key in dims],
            dtype=float,
        )
        for entry in candidates
    }

    remaining = list(candidates)
    selected: list[CoverageEntry] = []
    gains: list[float] = []

    first = _first_pick(remaining)
    selected.append(first)
    gains.append(first.coverage_breadth_score)
    remaining.remove(first)

    while len(selected) < combo_size and remaining:
        best: CoverageEntry | None = None
        best_gain = -np.inf
        for candidate in remaining:
            if _has_high_corr(candidate.benchmark_id, selected, correlation, max_corr):
                continue
            gain = _marginal_gain(candidate, selected, profiles)
            if gain > best_gain:
                best, best_gain = candidate, gain
        if best is None or best_gain <= 0.0:
            break
        selected.append(best)
        gains.append(best_gain)
        remaining.remove(best)

    total = float(np.sum(gains))
    if total <= 0.0:
        weights = [1.0 / len(selected)] * len(selected)
    else:
        weights = [float(gain) / total for gain in gains]

    return [
        {"benchmark_id": entry.benchmark_id, "weight": weight}
        for entry, weight in zip(selected, weights, strict=True)
    ]


def _first_pick(candidates: list[CoverageEntry]) -> CoverageEntry:
    """Pick the anchor: a rule_verified candidate when available, else the one
    with the highest coverage breadth."""
    verified = [
        entry
        for entry in candidates
        if entry.scoring_method == ScoringMethod.RULE_VERIFIED
    ]
    pool = verified if verified else candidates
    return max(pool, key=lambda entry: entry.coverage_breadth_score)


def _marginal_gain(
    candidate: CoverageEntry,
    selected: list[CoverageEntry],
    profiles: dict[str, np.ndarray],
) -> float:
    """Breadth-scaled novelty of a candidate against the selected set."""
    candidate_vec = profiles[candidate.benchmark_id]
    max_sim = max(
        _cosine(candidate_vec, profiles[entry.benchmark_id]) for entry in selected
    )
    return candidate.coverage_breadth_score * (1.0 - max_sim)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, treating a zero vector as orthogonal to everything."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _has_high_corr(
    candidate_id: str,
    selected: list[CoverageEntry],
    correlation: pd.DataFrame | None,
    max_corr: float,
) -> bool:
    """True when any selected benchmark correlates with ``candidate_id`` above
    ``max_corr`` (unknown/missing correlations never disqualify)."""
    if correlation is None:
        return False
    for entry in selected:
        value = _corr_value(correlation, candidate_id, entry.benchmark_id)
        if value is not None and value > max_corr:
            return True
    return False


def _corr_value(correlation: pd.DataFrame, a: str, b: str) -> float | None:
    """Look up a correlation entry; None when the pair is absent or NaN."""
    if a not in correlation.index or b not in correlation.columns:
        return None
    value = correlation.loc[a, b]
    if pd.isna(value):
        return None
    return float(value)
