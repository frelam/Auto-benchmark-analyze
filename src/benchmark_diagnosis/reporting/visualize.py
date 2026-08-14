"""Render offline-analysis artifacts as publication-ready PNG figures.

This module turns the versioned offline assets (capability-coverage table,
representative portfolios, expectation curves) plus the model x benchmark score
matrix into a set of figures intended for a README / report homepage:

* per-benchmark expectation curves (params -> score log-linear fit, plus the
  time -> score frontier envelope);
* a single overview figure overlaying every benchmark's time frontier;
* a capability-coverage heatmap and a coverage-metrics bar chart;
* a benchmark x benchmark correlation heatmap.

Matplotlib is an optional dependency (``pip install -e ".[plot]"``); importing
this module without it raises a clear error.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

try:  # matplotlib is an optional extra — fail loudly but clearly.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise RuntimeError(
        "matplotlib is required for visualization; install it with "
        "`pip install -e '.[plot]'`"
    ) from exc

_EPOCH = dt.date(2020, 1, 1)
_DPI = 130

# Distinctive but colorblind-friendly palette (Okabe-Ito-ish).
_C_DENSE = "#0072B2"
_C_MOE = "#D55E00"
_C_FRONTIER = "#009E73"
_C_FIT = "#CC79A7"
_CMAP = LinearSegmentedColormap.from_list(
    "bd", ["#f7fbff", "#6baed6", "#08306b"]
)


def _score100(value: float | np.ndarray) -> float | np.ndarray:
    """Convert a unit score (0-1) back to a 0-100 scale for display."""
    arr = np.asarray(value, dtype=np.float64)
    out = np.where(arr <= 1.5, arr * 100.0, arr)
    return float(out) if arr.ndim == 0 else out


def _logistic(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _epoch_date(days: float) -> dt.date:
    return _EPOCH + dt.timedelta(days=float(days))


def render_curves(
    curves: list[dict],
    out_dir: Path,
    benchmark_names: dict[str, str] | None = None,
) -> list[Path]:
    """One figure per benchmark: params fit (left) + time frontier (right)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = benchmark_names or {}
    written: list[Path] = []

    by_benchmark: dict[str, list[dict]] = {}
    for curve in curves:
        by_benchmark.setdefault(curve["benchmark_id"], []).append(curve)

    for benchmark_id in sorted(by_benchmark):
        group = by_benchmark[benchmark_id]
        title = names.get(benchmark_id, benchmark_id)
        fig, (ax_p, ax_t) = plt.subplots(1, 2, figsize=(11, 4.2))

        # --- left: params -> score (log-x) ---
        param_kinds = [c for c in group if c["kind"].startswith("params")]
        for curve in param_kinds:
            points = [(float(x), _score100(y)) for x, y in curve["points"]]
            if not points:
                continue
            color = _C_MOE if curve["kind"] == "params_moe" else _C_DENSE
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax_p.scatter(xs, ys, color=color, s=38, alpha=0.8, zorder=3,
                         label=curve["kind"].removeprefix("params_"))
            coef = curve.get("coefficients") or {}
            if "slope" in coef and "intercept" in coef and len(xs) >= 5:
                grid = np.logspace(np.log10(min(xs)), np.log10(max(xs)), 100)
                fit = _score100(_logistic(
                    coef["slope"] * np.log10(grid) + coef["intercept"]
                ))
                ax_p.plot(grid, fit, color=color, lw=2, alpha=0.9, zorder=2)
        ax_p.set_xscale("log")
        ax_p.set_xlabel("Parameters (B, log scale)")
        ax_p.set_ylabel("Score (0-100)")
        ax_p.set_title("Params -> score fit")
        ax_p.grid(True, which="both", alpha=0.25)
        if param_kinds:
            ax_p.legend(frameon=False, fontsize=8)

        # --- right: time -> score frontier ---
        frontier = next((c for c in group if c["kind"] == "time_frontier"), None)
        if frontier is not None:
            points = sorted(
                (_epoch_date(x), _score100(y)) for x, y in frontier["points"]
            )
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax_t.scatter(xs, ys, color=_C_FRONTIER, s=30, alpha=0.7, zorder=3,
                         label="best observed")
            ax_t.step(xs, ys, where="post", color=_C_FRONTIER, lw=2, zorder=2,
                      label="frontier envelope")
            ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax_t.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3))
            ax_t.set_xlabel("Release date")
            ax_t.set_ylabel("Score (0-100)")
            ax_t.set_title("SOTA frontier over time")
            ax_t.grid(True, alpha=0.25)
            ax_t.legend(frameon=False, fontsize=8)

        fig.suptitle(title, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = out_dir / f"fig_curves_{benchmark_id}.png"
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def render_frontier_overview(curves: list[dict], out_dir: Path) -> list[Path]:
    """Overlay every benchmark's time frontier on a single axes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frontiers = [c for c in curves if c["kind"] == "time_frontier"]
    if not frontiers:
        return []

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for curve in sorted(frontiers, key=lambda c: c["benchmark_id"]):
        points = sorted((_epoch_date(x), _score100(y)) for x, y in curve["points"])
        ax.step([p[0] for p in points], [p[1] for p in points], where="post",
                lw=2, alpha=0.9, label=curve["benchmark_id"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5))
    ax.set_xlabel("Release date")
    ax.set_ylabel("Frontier score (0-100)")
    ax.set_title("State-of-the-art frontier across benchmarks")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path = out_dir / "fig_frontier_overview.png"
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return [path]


def render_coverage_profile(coverage: list[dict], out_dir: Path) -> list[Path]:
    """Heatmap of the normalized discrimination profile (benchmark x dimension)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not coverage:
        return []

    dims = sorted({d for row in coverage for d in row["discrimination_profile"]})
    rows = sorted(coverage, key=lambda r: (r["primary_cluster"], r["benchmark_id"]))
    labels = [r["benchmark_id"] for r in rows]
    matrix = np.array(
        [[r["discrimination_profile"].get(d, 0.0) for d in dims] for r in rows]
    )

    fig, ax = plt.subplots(figsize=(0.6 + 0.9 * len(dims), 0.6 + 0.32 * len(rows)))
    im = ax.imshow(matrix, aspect="auto", cmap=_CMAP, vmin=0.0, vmax=matrix.max() or 1.0)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    # Color spine to delimit primary clusters.
    cluster_boundaries = [i for i in range(1, len(rows))
                          if rows[i]["primary_cluster"] != rows[i - 1]["primary_cluster"]]
    for b in cluster_boundaries:
        ax.axhline(b - 0.5, color="white", lw=1.6)
    for i, r in enumerate(rows):
        ax.annotate(r["primary_cluster"], (len(dims) + 0.4, i), va="center",
                    fontsize=7, color="#444444")
    fig.colorbar(im, ax=ax, shrink=0.85, label="normalized loading")
    ax.set_title("Capability-coverage profile (benchmark x latent dimension)")
    fig.tight_layout()
    path = out_dir / "fig_coverage_profile.png"
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return [path]


def render_coverage_metrics(coverage: list[dict], out_dir: Path) -> list[Path]:
    """Bar chart of breadth / reliability / agreement per benchmark."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not coverage:
        return []

    rows = sorted(coverage, key=lambda r: r["benchmark_id"])
    labels = [r["benchmark_id"] for r in rows]
    breadth = [r["coverage_breadth_score"] for r in rows]
    reliability = [r["reliability_score"] for r in rows]
    agreement = [r["design_goal_agreement_score"] for r in rows]

    x = np.arange(len(rows))
    width = 0.26
    fig, ax = plt.subplots(figsize=(0.6 + 0.42 * len(rows), 5))
    ax.bar(x - width, breadth, width, label="coverage breadth", color=_C_DENSE)
    ax.bar(x, reliability, width, label="reliability", color=_C_MOE)
    ax.bar(x + width, agreement, width, label="design-goal agreement",
           color=_C_FRONTIER)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score (0-1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Coverage quality metrics per benchmark")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "fig_coverage_metrics.png"
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return [path]


def render_cluster_map(
    coverage: list[dict],
    out_dir: Path,
    benchmark_names: dict[str, str] | None = None,
) -> list[Path]:
    """Scatter benchmarks in factor space, colored by cluster + labeled by capability.

    The 2D embedding is a PCA of each benchmark's discrimination profile, so
    proximity means "these benchmarks' scores co-vary across models". A cluster
    is therefore a group of *correlated capabilities* — the kind that tend to
    transfer together during training — and is labeled with its dominant tags.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(coverage) < 2:
        return []

    from sklearn.decomposition import PCA

    from benchmark_diagnosis.reporting.report_generator import (
        cluster_capability_labels,
    )

    names = benchmark_names or {}
    rows = sorted(coverage, key=lambda r: (r["primary_cluster"], r["benchmark_id"]))
    dims = sorted({d for row in rows for d in row["discrimination_profile"]})
    matrix = np.array(
        [[row["discrimination_profile"].get(d, 0.0) for d in dims] for row in rows]
    )
    if matrix.shape[1] >= 2:
        coords = PCA(n_components=2, random_state=0).fit_transform(matrix)
    else:
        coords = np.column_stack([matrix[:, 0], np.zeros(matrix.shape[0])])

    cluster_labels = cluster_capability_labels(coverage)
    clusters = sorted({row["primary_cluster"] for row in rows})
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(10.5, 7))
    for ci, cluster in enumerate(clusters):
        idxs = [i for i, row in enumerate(rows) if row["primary_cluster"] == cluster]
        color = cmap(ci % 10)
        ax.scatter(
            coords[idxs, 0], coords[idxs, 1], s=150, color=color, alpha=0.85,
            edgecolors="white", linewidths=1.0, zorder=3,
            label=f"{cluster}: {', '.join(cluster_labels.get(cluster, []))}",
        )
        for i in idxs:
            ax.annotate(
                names.get(rows[i]["benchmark_id"], rows[i]["benchmark_id"]),
                (coords[i, 0], coords[i, 1]),
                fontsize=8, xytext=(7, 5), textcoords="offset points", zorder=4,
            )

    ax.set_xlabel("Factor component 1")
    ax.set_ylabel("Factor component 2")
    ax.set_title(
        "Proximity = score co-variation across models  =>  correlated "
        "(co-trainable) capabilities",
        fontsize=9, pad=12,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(title="cluster -> dominant capabilities", frameon=False, fontsize=8)
    fig.suptitle("Benchmark capability clusters", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "fig_cluster_map.png"
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return [path]


def render_correlation(scores, out_dir: Path) -> list[Path]:
    """Pearson correlation heatmap over benchmarks (min 2 shared models)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if scores is None or scores.shape[1] < 2:
        return []

    corr = scores.corr(min_periods=2)
    corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if corr.empty or corr.shape[0] < 2:
        return []

    labels = list(corr.columns)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.6 + 0.42 * n, 0.6 + 0.42 * n))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            value = corr.values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(value) > 0.55 else "#222222")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Pearson r")
    ax.set_title("Benchmark score correlation")
    fig.tight_layout()
    path = out_dir / "fig_benchmark_correlation.png"
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return [path]
