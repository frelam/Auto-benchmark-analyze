"""Design-goal validation (design doc section 2.5).

Cross-checks a benchmark's officially declared task tags against the tags
aggregated over its statistical cluster; a low agreement signals that the
benchmark may not measure what it claims.
"""

from __future__ import annotations


def agreement_score(declared_tags: list[str], cluster_tags: list[str]) -> float:
    """Jaccard similarity between declared and statistically-aggregated tags.

    Args:
        declared_tags: Tags the benchmark officially declares for itself.
        cluster_tags: Tags aggregated over the benchmark's statistical cluster,
            excluding the benchmark itself.

    Returns:
        A float in [0, 1]; 0.0 when either list is empty.
    """
    if not declared_tags or not cluster_tags:
        return 0.0
    declared = set(declared_tags)
    cluster = set(cluster_tags)
    if not declared or not cluster:
        return 0.0
    return len(declared & cluster) / len(declared | cluster)
