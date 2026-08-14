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


def test_agreement_score() -> None:
    assert agreement_score([], ["a"]) == 0.0
    assert agreement_score(["a", "b"], []) == 0.0
    assert agreement_score(["a", "b", "c"], ["b", "c", "d"]) == pytest.approx(2 / 4)
    assert agreement_score(["a"], ["a"]) == 1.0
    assert agreement_score(["a", "b"], ["c", "d"]) == 0.0
