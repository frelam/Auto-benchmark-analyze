"""Tests for aggregate factor analysis fallback (design doc section 2.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmark_diagnosis.capability_analysis.design_goal_validation import (
    agreement_score,
)
from benchmark_diagnosis.capability_analysis.factor_analysis import (
    FactorModel,
    fit_factor_model,
)


def test_fit_factor_model_shapes_and_ordering() -> None:
    rng = np.random.default_rng(0)
    models = [f"model_{i:02d}" for i in range(30)]
    benchmarks = [f"bench_{i:02d}" for i in range(12)]
    scores = pd.DataFrame(
        rng.normal(size=(len(models), len(benchmarks))),
        index=models,
        columns=benchmarks,
    )

    model = fit_factor_model(scores)

    assert isinstance(model, FactorModel)
    assert model.benchmark_ids == benchmarks
    assert model.loadings.shape == (len(benchmarks), model.n_factors)
    assert len(model.clusters) == len(benchmarks)
    assert len(model.explained_variance_ratio) == model.n_factors
    assert 2 <= model.n_factors <= min(8, len(benchmarks))
    assert all(0 <= label < model.n_factors for label in model.clusters)


def test_fit_factor_model_explicit_n_factors() -> None:
    rng = np.random.default_rng(0)
    scores = pd.DataFrame(
        rng.normal(size=(40, 10)),
        index=[f"m{i}" for i in range(40)],
        columns=[f"b{i}" for i in range(10)],
    )

    model = fit_factor_model(scores, n_factors=3)

    assert model.n_factors == 3
    assert model.loadings.shape == (10, 3)
    assert len(model.explained_variance_ratio) == 3


def test_fit_factor_model_drops_sparse_columns_and_imputes_nan() -> None:
    rng = np.random.default_rng(0)
    benchmarks = [f"b{i}" for i in range(10)]
    scores = pd.DataFrame(
        rng.normal(size=(30, len(benchmarks))),
        index=[f"m{i}" for i in range(30)],
        columns=benchmarks,
    )
    # b9 has a single non-null score -> dropped before PCA.
    scores["b9"] = np.nan
    scores.iloc[0, scores.columns.get_loc("b9")] = 0.5
    # b5 has one NaN -> imputed with the column mean, still kept.
    scores.iloc[5, scores.columns.get_loc("b5")] = np.nan

    model = fit_factor_model(scores)

    assert model.benchmark_ids == benchmarks[:9]
    assert model.loadings.shape[0] == 9
    assert len(model.clusters) == 9


def test_fit_factor_model_rejects_insufficient_benchmarks() -> None:
    rng = np.random.default_rng(0)
    scores = pd.DataFrame(
        rng.normal(size=(20, 5)),
        index=[f"m{i}" for i in range(20)],
        columns=[f"b{i}" for i in range(5)],
    )
    scores["b4"] = np.nan  # only one non-null value -> dropped
    scores.iloc[0, scores.columns.get_loc("b4")] = 0.1

    model = fit_factor_model(scores)
    assert model.benchmark_ids == [f"b{i}" for i in range(4)]

    # A DataFrame whose only column is too sparse cannot be factor-analyzed.
    with pytest.raises(ValueError):
        fit_factor_model(scores[["b4"]])


def test_rank_factor_keeps_code_benchmarks_together() -> None:
    """MNAR missingness + ceiling saturation must not split a capability family.

    Regression test for the cluster-map accuracy fix: coding/agentic benchmarks
    (the ``code_*`` family) share a genuine covariance pattern but are only
    reported for recent models (missing-not-at-random), while the easy ``sat_*``
    benchmarks saturate near the ceiling. Mean-imputation + Pearson PCA splits
    the code family apart; pairwise Spearman eigendecomposition keeps it intact.
    """
    rng = np.random.default_rng(7)
    # Shared latent factor: "general" ability.
    general = rng.normal(size=40)
    # Code-specific factor, only expressed (observed) for recent models.
    code_latent = rng.normal(size=40)
    recent = np.zeros(40, dtype=bool)
    recent[20:] = True  # only the 20 most recent models report agentic benchmarks

    def _b(gen_w: float, code_w: float, base: float) -> list[float]:
        vals = base + gen_w * general + code_w * code_latent + rng.normal(0, 0.3, 40)
        return np.clip(vals, 1.0, 99.0).tolist()

    df = pd.DataFrame(
        {
            "sat_0": _b(1.0, 0.0, 60.0),  # saturated, general-only (top ~98)
            "sat_1": _b(1.0, 0.0, 60.0),
            "code_a": _b(0.8, 1.0, 20.0),  # code, reported for everyone
            "code_b": _b(0.8, 1.0, 20.0),
        }
    )
    df.loc[~recent, ["code_b"]] = np.nan  # MNAR: code_b missing for old models
    # Saturate the easy benchmarks.
    df["sat_0"] = df["sat_0"].clip(upper=97.0)
    df["sat_1"] = df["sat_1"].clip(upper=97.0)

    model = fit_factor_model(df, n_factors=2)
    cluster_of = dict(zip(model.benchmark_ids, model.clusters, strict=True))
    assert cluster_of["code_a"] == cluster_of["code_b"]


def test_clusters_invariant_to_benchmark_column_order() -> None:
    """KMeans groupings must not depend on the input frame's column order.

    k-means++ seeds initial centroids by row index, so reordering the benchmark
    columns (which permutes the loading rows) can silently change the clustering.
    fit_factor_model canonicalizes column order so results are stable.
    """
    rng = np.random.default_rng(11)
    n = 30
    general = rng.normal(size=n)
    math_spec = rng.normal(size=n)
    code_spec = rng.normal(size=n)

    def _mk(gw: float, mw: float, cw: float) -> list[float]:
        vals = 50.0 + gw * general + mw * math_spec + cw * code_spec
        return (vals + rng.normal(0, 0.4, n)).clip(0.0, 100.0).tolist()

    cols: dict[str, list[float]] = {}
    for i in range(3):
        cols[f"math_{i}"] = _mk(1.0, 1.0, 0.0)
        cols[f"code_{i}"] = _mk(1.0, 0.0, 1.0)
        cols[f"gen_{i}"] = _mk(1.0, 0.0, 0.0)
    df = pd.DataFrame(cols, index=[f"m{i}" for i in range(n)])

    def grouping(m: FactorModel) -> list[list[str]]:
        g: dict[int, list[str]] = {}
        for bid, label in zip(m.benchmark_ids, m.clusters, strict=True):
            g.setdefault(label, []).append(bid)
        return sorted(sorted(v) for v in g.values())

    m_a = fit_factor_model(df, n_factors=3)
    m_b = fit_factor_model(df[list(reversed(df.columns))], n_factors=3)
    assert grouping(m_a) == grouping(m_b)


def test_agreement_score() -> None:
    assert agreement_score([], ["a"]) == 0.0
    assert agreement_score(["a", "b"], []) == 0.0
    assert agreement_score(["a", "b", "c"], ["b", "c", "d"]) == pytest.approx(2 / 4)
    assert agreement_score(["a"], ["a"]) == 1.0
    assert agreement_score(["a", "b"], ["c", "d"]) == 0.0
