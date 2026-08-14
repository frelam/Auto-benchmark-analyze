"""Cluster-level screening against representative portfolios (design doc 4.1).

For each capability cluster we evaluate its representative benchmark combo and
aggregate the raw scores into one weighted cluster score.
"""

from __future__ import annotations

from benchmark_diagnosis.core.types import ClusterPortfolio


def cluster_scores(
    raw_scores: dict[str, float],
    portfolios: list[ClusterPortfolio],
) -> dict[str, float]:
    """Aggregate raw benchmark scores into weighted per-cluster scores.

    Each portfolio's weights are renormalized over the benchmarks actually
    present in ``raw_scores``, so a missing benchmark does not silently change
    the cluster's effective total weight.

    Args:
        raw_scores: Mapping ``benchmark_id -> score``.
        portfolios: Representative benchmark combos, one per capability cluster.

    Returns:
        Mapping ``cluster_id -> weighted_score``. Clusters whose portfolios have
        no evaluable benchmarks are omitted.
    """
    result: dict[str, float] = {}
    for portfolio in portfolios:
        present = [
            (entry, raw_scores[entry["benchmark_id"]])
            for entry in portfolio.benchmarks
            if entry["benchmark_id"] in raw_scores
        ]
        if not present:
            continue
        weights = [float(entry.get("weight", 1.0)) for entry, _ in present]
        total = sum(weights)
        if total <= 0:
            # Degenerate all-zero weights: fall back to equal weighting.
            weights = [1.0] * len(weights)
            total = len(weights)
        result[portfolio.cluster_id] = sum(
            (weight / total) * score
            for weight, (_, score) in zip(weights, present, strict=True)
        )
    return result
