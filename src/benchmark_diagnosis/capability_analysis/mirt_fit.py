"""Multidimensional Item Response Theory (mIRT) — the item-level capability model.

This is the core "special" module (design doc section 2.1). We estimate, jointly
over all (model, item, correct) triples:

* ``theta``  — per-model capability vector (d-dimensional),
* ``alpha``  — per-item discrimination vector (d-dimensional),
* ``beta``   — per-item difficulty (scalar),

with ``P(correct) = sigmoid(theta · alpha − beta)``. This is exactly a two-tower
scoring model; we implement it directly in PyTorch (``torch`` is an optional
extra). A Gaussian prior on ``theta`` (L2 regularization) fixes the otherwise
scale-degenerate product ``theta · alpha``.

``torch`` is imported lazily so the package works without it; when item-level data
is unavailable, callers fall back to :mod:`factor_analysis`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MIRTResult:
    theta: np.ndarray  # (n_models, dims)
    alpha: np.ndarray  # (n_items, dims)
    beta: np.ndarray  # (n_items,)
    model_ids: list[str]
    item_ids: list[str]
    dims: int
    heldout_auc: float | None = None


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "mIRT requires torch; install with `pip install -e '.[mirt]'`"
        ) from exc
    return torch


def _encode_triples(
    triples: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Convert a triples DataFrame into integer indices + id lists."""
    models = pd.unique(triples["model_id"]).tolist()
    items = pd.unique(triples["item_id"]).tolist()
    model_idx = {m: i for i, m in enumerate(models)}
    item_idx = {it: i for i, it in enumerate(items)}
    m_idx = np.array([model_idx[m] for m in triples["model_id"]], dtype=np.int64)
    i_idx = np.array([item_idx[it] for it in triples["item_id"]], dtype=np.int64)
    y = np.asarray(triples["correct"], dtype=np.float32)
    return m_idx, i_idx, y, models, items


def fit_mirt(
    triples: pd.DataFrame,
    dims: int,
    *,
    lr: float = 0.05,
    epochs: int = 500,
    theta_reg: float = 0.1,
    alpha_reg: float = 0.01,
    device: str | None = None,
    seed: int = 0,
    heldout: pd.DataFrame | None = None,
) -> MIRTResult:
    """Fit the two-tower mIRT model to item-level triples.

    Args:
        triples: DataFrame with columns ``model_id``, ``item_id``, ``correct``.
        dims: Number of latent capability dimensions.
        lr: Adam learning rate.
        epochs: Number of optimization epochs.
        theta_reg: L2 prior strength on ``theta`` (fixes scale).
        alpha_reg: L2 strength on ``alpha`` (regularizes discrimination).
        device: torch device string (auto if None).
        seed: Random seed for initialization.
        heldout: Optional DataFrame of held-out triples for AUC evaluation.

    Returns:
        A :class:`MIRTResult` with the estimated parameters.
    """
    torch = _require_torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    m_idx, i_idx, y, models, items = _encode_triples(triples)
    n_models, n_items = len(models), len(items)

    theta = torch.nn.Parameter(torch.randn(n_models, dims, device=device) * 0.1)
    alpha = torch.nn.Parameter(torch.randn(n_items, dims, device=device) * 0.1)
    beta = torch.nn.Parameter(torch.zeros(n_items, device=device))

    m_idx_t = torch.as_tensor(m_idx, device=device)
    i_idx_t = torch.as_tensor(i_idx, device=device)
    y_t = torch.as_tensor(y, device=device)

    optimizer = torch.optim.Adam([theta, alpha, beta], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = (theta[m_idx_t] * alpha[i_idx_t]).sum(dim=1) - beta[i_idx_t]
        loss = loss_fn(logits, y_t)
        loss = loss + theta_reg * (theta**2).sum() + alpha_reg * (alpha**2).sum()
        loss.backward()
        optimizer.step()

    result = MIRTResult(
        theta=theta.detach().cpu().numpy(),
        alpha=alpha.detach().cpu().numpy(),
        beta=beta.detach().cpu().numpy(),
        model_ids=models,
        item_ids=items,
        dims=dims,
    )
    if heldout is not None and not heldout.empty:
        result.heldout_auc = _auc(result, heldout)
    return result


def predict_proba(result: MIRTResult, triples: pd.DataFrame) -> np.ndarray:
    """Return P(correct) for each row of a triples DataFrame."""
    model_idx = {m: i for i, m in enumerate(result.model_ids)}
    item_idx = {it: i for i, it in enumerate(result.item_ids)}
    theta = result.theta
    alpha = result.alpha
    beta = result.beta
    out = np.zeros(len(triples), dtype=np.float64)
    for k, row in triples.iterrows():
        m = model_idx.get(row["model_id"])
        it = item_idx.get(row["item_id"])
        if m is None or it is None:
            out[k] = 0.5
            continue
        logit = float(theta[m] @ alpha[it] - beta[it])
        out[k] = 1.0 / (1.0 + np.exp(-logit))
    return out


def _auc(result: MIRTResult, triples: pd.DataFrame) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:  # pragma: no cover
        return float("nan")
    proba = predict_proba(result, triples)
    y = np.asarray(triples["correct"], dtype=np.float32)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, proba))


def select_dimensions(
    triples: pd.DataFrame,
    candidate_dims: tuple[int, ...] = (2, 4, 6, 8),
    *,
    heldout_frac: float = 0.2,
    seed: int = 0,
    **fit_kwargs,
) -> tuple[int, MIRTResult]:
    """Choose the best latent dimension via held-out AUC (design doc section 2.2).

    Returns ``(best_dims, result)`` where ``result`` is the model refit on the full
    triples with ``best_dims``.
    """
    if triples.empty:
        raise ValueError("cannot fit mIRT on empty triples")

    rng = np.random.default_rng(seed)
    n = len(triples)
    n_hold = max(1, int(round(n * heldout_frac)))
    perm = rng.permutation(n)
    hold_idx = perm[:n_hold]
    train = triples.iloc[perm[n_hold:]].reset_index(drop=True)
    held = triples.iloc[hold_idx].reset_index(drop=True)

    best_dims: int | None = None
    best_auc = -np.inf
    for d in candidate_dims:
        if d > len(pd.unique(train["item_id"])):
            continue
        res = fit_mirt(train, d, seed=seed, **fit_kwargs)
        auc = _auc(res, held)
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_dims = d

    if best_dims is None:
        best_dims = candidate_dims[0]

    final = fit_mirt(triples, best_dims, seed=seed, heldout=held, **fit_kwargs)
    return best_dims, final
