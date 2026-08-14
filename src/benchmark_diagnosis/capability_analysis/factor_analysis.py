"""Aggregate factor analysis fallback (design doc section 2.1).

When item-level IRT data is unavailable, benchmarks degrade to their aggregate
scores and we fall back to PCA + KMeans to recover a latent capability space.
This module produces the per-benchmark cluster labels and factor loadings that
feed the capability-coverage table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_MIN_NON_NULL = 2  # benchmarks must have at least this many scores to enter PCA
_VARIANCE_THRESHOLD = 0.8
_MAX_AUTO_FACTORS = 8
_DEFAULT_FACTORS = 2


@dataclass
class FactorModel:
    """Fitted aggregate factor structure for one benchmark set."""

    loadings: np.ndarray  # shape (n_benchmarks, n_factors)
    clusters: list[int]  # KMeans label per benchmark (same order as benchmark_ids)
    benchmark_ids: list[str]
    explained_variance_ratio: np.ndarray
    n_factors: int


def fit_factor_model(scores: pd.DataFrame, n_factors: int | None = None) -> FactorModel:
    """Fit PCA + KMeans on aggregate model x benchmark scores.

    Benchmarks with fewer than ``_MIN_NON_NULL`` non-null scores are dropped;
    remaining NaNs are imputed with the column mean, each benchmark column is
    z-scored, then PCA is run. ``n_factors`` is auto-chosen (smallest k in
    [2 .. min(8, n_benchmarks)] whose cumulative explained variance ratio
    reaches 0.8, defaulting to 2) unless given explicitly.

    Args:
        scores: DataFrame indexed by model_id with one column per benchmark.
        n_factors: Desired number of factors; auto-selected when None.

    Returns:
        A :class:`FactorModel` with loadings, cluster labels and variance ratios.

    Raises:
        ValueError: If fewer than 2 usable benchmarks remain, or n_factors < 2.
    """
    if scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("scores must be a non-empty model x benchmark DataFrame")

    keep = scores.columns[scores.notna().sum(axis=0) >= _MIN_NON_NULL]
    df = scores[keep]
    if df.shape[1] < 2:
        raise ValueError(
            "at least 2 benchmarks with >= 2 non-null scores are required for PCA"
        )
    df = df.fillna(df.mean())
    benchmark_ids = list(df.columns)

    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    pca = PCA()
    pca.fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    max_k = min(_MAX_AUTO_FACTORS, len(benchmark_ids), len(cum_var))

    if n_factors is None:
        k = _DEFAULT_FACTORS
        for candidate in range(_DEFAULT_FACTORS, max_k + 1):
            if cum_var[candidate - 1] >= _VARIANCE_THRESHOLD:
                k = candidate
                break
    else:
        if n_factors < _DEFAULT_FACTORS:
            raise ValueError(f"n_factors must be >= {_DEFAULT_FACTORS}")
        k = min(n_factors, max_k)

    loadings = pca.components_[:k].T  # (n_benchmarks, n_factors)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(loadings)

    return FactorModel(
        loadings=loadings,
        clusters=[int(label) for label in cluster_labels],
        benchmark_ids=benchmark_ids,
        explained_variance_ratio=pca.explained_variance_ratio_[:k],
        n_factors=k,
    )
