"""Tests for the item-level multidimensional IRT fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from benchmark_diagnosis.capability_analysis.mirt_fit import (  # noqa: E402
    fit_mirt,
    predict_proba,
    select_dimensions,
)


def _make_triples(n_models: int = 10, n_items: int = 40, dims: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(n_models, dims))
    alpha = rng.normal(size=(n_items, dims))
    beta = rng.normal(size=n_items)
    rows = []
    for m in range(n_models):
        for i in range(n_items):
            logit = float(theta[m] @ alpha[i] - beta[i])
            p = 1.0 / (1.0 + np.exp(-logit))
            rows.append(
                {"model_id": f"m{m}", "item_id": f"i{i}", "correct": int(rng.random() < p)}
            )
    return pd.DataFrame(rows), theta


def test_fit_mirt_shapes_and_auc():
    triples, _ = _make_triples()
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(triples))
    heldout = triples.iloc[perm[:100]].reset_index(drop=True)

    result = fit_mirt(triples, dims=2, epochs=300, heldout=heldout)

    assert result.theta.shape == (10, 2)
    assert result.alpha.shape == (40, 2)
    assert result.beta.shape == (40,)
    assert result.dims == 2
    assert result.model_ids == [f"m{i}" for i in range(10)]
    assert result.heldout_auc is not None and result.heldout_auc > 0.6


def test_predict_proba_in_range():
    triples, _ = _make_triples()
    result = fit_mirt(triples, dims=2, epochs=150)
    proba = predict_proba(result, triples.head(20))
    assert proba.shape == (20,)
    assert np.all((proba >= 0) & (proba <= 1))


def test_select_dimensions_returns_valid_dims():
    triples, _ = _make_triples(n_models=6, n_items=20)
    best_dims, result = select_dimensions(triples, candidate_dims=(2, 3), epochs=150)
    assert best_dims in (2, 3)
    assert result.dims == best_dims
    assert result.theta.shape == (6, best_dims)
