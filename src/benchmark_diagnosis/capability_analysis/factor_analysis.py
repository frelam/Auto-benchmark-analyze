"""Aggregate factor analysis fallback (design doc section 2.1).

When item-level IRT data is unavailable, benchmarks degrade to their aggregate
scores and we fall back to PCA + KMeans to recover a latent capability space.
This module produces the per-benchmark cluster labels and factor loadings that
feed the capability-coverage table.

The aggregate scores across a model population are heavily missing-not-at-random:
hard / agentic benchmarks (AIME, SWE-bench, ...) are only reported for recent
frontier models, and easy benchmarks (GSM8K, HumanEval) saturate near the
ceiling. Two consequences:

* Mean-imputing the missing cells fabricates covariance between benchmarks that
  share the same *missingness* rather than the same *capability*.
* Pearson correlation on saturated scales understates genuine relationships.

So instead of PCA on a mean-imputed matrix we work directly on the **pairwise
complete-observation Spearman correlation matrix** (rank-based, so ceiling
effects are removed) and recover the loadings by eigendecomposition. This is
equivalent to PCA on the standardized data when the matrix is complete, but
stays stable in the realistic sparse case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

_MIN_NON_NULL = 2  # benchmarks must have at least this many scores to enter
_MIN_SHARED = 5  # benchmark pairs must share this many models to correlate
_VARIANCE_THRESHOLD = 0.8
_MIN_EXTRA_EIGENVAL = 0.85  # retain an extra factor while it explains >= this variance
_MAX_AUTO_FACTORS = 8
_DEFAULT_FACTORS = 2
_EPS = 1e-12


@dataclass
class FactorModel:
    """Fitted aggregate factor structure for one benchmark set."""

    loadings: np.ndarray  # shape (n_benchmarks, n_factors)
    clusters: list[int]  # KMeans label per benchmark (same order as benchmark_ids)
    benchmark_ids: list[str]
    explained_variance_ratio: np.ndarray
    n_factors: int


def fit_factor_model(scores: pd.DataFrame, n_factors: int | None = None) -> FactorModel:
    """Fit a rank-based factor model on aggregate model x benchmark scores.

    Benchmarks with fewer than ``_MIN_NON_NULL`` non-null scores are dropped.
    The pairwise-complete **Spearman** correlation matrix (each pair requires at
    least ``_MIN_SHARED`` co-observed models) is then eigendecomposed to obtain
    per-benchmark loadings, and KMeans groups the benchmarks in loading space.

    ``n_factors`` is auto-chosen: smallest k in [2 .. min(8, n_benchmarks)] whose
    cumulative explained variance reaches ``_VARIANCE_THRESHOLD``, extended while
    the next eigenvalue still explains meaningful variance (>= ``_MIN_EXTRA_EIGENVAL``).
    Spearman rank correlation makes the fit robust to the ceiling effect of
    saturated benchmarks (GSM8K, HumanEval, ...) and pairwise deletion avoids the
    fabricated covariance that mean-imputation of missing cells would introduce.

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
    # Canonical column order so the KMeans grouping is invariant to the input
    # frame's column ordering (k-means++ seeds initial centroids by row index).
    df = df[sorted(df.columns)]
    benchmark_ids = list(df.columns)

    # Pairwise-complete Spearman correlation; pairs with too few co-observed
    # models yield NaN and are neutralized (0.0) before the eigendecomposition.
    corr = df.corr(method="spearman", min_periods=_MIN_SHARED)
    corr_matrix = corr.fillna(0.0).to_numpy(dtype=np.float64).copy()
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2.0

    eigen_values, eigen_vectors = np.linalg.eigh(corr_matrix)
    order = np.argsort(eigen_values)[::-1]
    eigen_values, eigen_vectors = eigen_values[order], eigen_vectors[:, order]
    # A pairwise correlation matrix can be mildly indefinite under missingness;
    # clipping negative eigenvalues keeps the loadings real and stable.
    eigen_values = np.maximum(eigen_values, 0.0)

    max_k = min(_MAX_AUTO_FACTORS, len(benchmark_ids))
    if n_factors is None:
        k = _auto_select(eigen_values, max_k)
    else:
        if n_factors < _DEFAULT_FACTORS:
            raise ValueError(f"n_factors must be >= {_DEFAULT_FACTORS}")
        k = min(n_factors, max_k)

    # Loadings = eigenvectors scaled by sqrt(eigenvalue): the correlation between
    # each benchmark and each latent factor (matches PCA-on-correlation semantics).
    loadings = eigen_vectors[:, :k] * np.sqrt(eigen_values[:k])

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(loadings)

    total_var = float(eigen_values.sum()) or 1.0
    return FactorModel(
        loadings=loadings,
        clusters=[int(label) for label in cluster_labels],
        benchmark_ids=benchmark_ids,
        explained_variance_ratio=eigen_values[:k] / total_var,
        n_factors=k,
    )


def _auto_select(eigen_values: np.ndarray, max_k: int) -> int:
    """Choose the number of retained factors from the eigenvalue spectrum.

    Keeps at least ``_DEFAULT_FACTORS``, then extends while either (a) the
    cumulative explained variance has not yet reached ``_VARIANCE_THRESHOLD`` or
    (b) the next factor still explains meaningful variance on its own
    (``_MIN_EXTRA_EIGENVAL``, a relaxed Kaiser-style cutoff). This keeps the
    latent space coarse enough to be readable while preserving factors that carry
    a distinct, informative dimension (e.g. a dedicated agentic/code factor).
    """
    total = float(eigen_values.sum()) or 1.0
    k = _DEFAULT_FACTORS
    while k < max_k:
        cumvar = float(eigen_values[:k].sum()) / total
        next_is_weak = eigen_values[k] < _MIN_EXTRA_EIGENVAL
        if cumvar >= _VARIANCE_THRESHOLD and next_is_weak:
            break
        k += 1
    return k
