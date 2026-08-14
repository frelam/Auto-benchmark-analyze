"""Cheap, label-based aggregation slicing (design doc section 5.1).

The first diagnosis step: re-aggregate the evaluation log by the benchmark's
declared subcategory / task-type labels to localize which sub-capability is
dragging the cluster down, before spending LLM budget on failure-mode analysis.
"""

from __future__ import annotations

import pandas as pd


def _to_binary(correct: object) -> float:
    """Map a ``correct`` cell to 1.0 / 0.0.

    Truthy values (True, 1, non-empty strings, ...) map to 1.0 and falsy values
    (False, 0, ...) to 0.0. Missing values (NaN / None) are treated as falsy so
    absent item scores never inflate a subcategory mean.
    """
    if pd.isna(correct):
        return 0.0
    return 1.0 if correct else 0.0


def slice_by_subcategory(
    items: pd.DataFrame,
    *,
    subcategory_col: str = "subcategory",
    correct_col: str = "correct",
) -> dict[str, float]:
    """Aggregate item-level rows into per-subcategory mean correctness.

    Args:
        items: DataFrame of item-level rows. Must contain ``subcategory_col``
            and ``correct_col`` columns.
        subcategory_col: Name of the column holding the subcategory label.
        correct_col: Name of the column holding the per-item correctness value
            (boolean, numeric 0/1, etc.).

    Returns:
        Mapping ``{subcategory: mean correct}`` sorted by value ascending, so the
        worst subcategory comes first.
    """
    binary = items[correct_col].map(_to_binary)
    frame = items[subcategory_col].to_frame(name="_subcategory")
    frame["_correct"] = binary
    means = frame.groupby("_subcategory")["_correct"].mean()
    return {str(key): float(value) for key, value in means.sort_values(ascending=True).items()}
