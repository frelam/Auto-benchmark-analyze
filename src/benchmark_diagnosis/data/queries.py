"""Read-side helpers: materialize matrices/triples used by analysis modules."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from benchmark_diagnosis.core.schema import (
    BenchmarkRecord,
    ModelRecord,
    ScoreRecord,
)


def scores_matrix(session: Session) -> pd.DataFrame:
    """Return a model x benchmark aggregate-score matrix.

    Duplicate (model, benchmark) aggregate rows are averaged. Index = model_id,
    columns = benchmark_id.
    """
    rows = session.execute(
        select(
            ScoreRecord.model_id,
            ScoreRecord.benchmark_id,
            ScoreRecord.score,
        ).where(ScoreRecord.item_id.is_(None), ScoreRecord.score.is_not(None))
    ).all()
    df = pd.DataFrame(rows, columns=["model_id", "benchmark_id", "score"])
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(
        index="model_id", columns="benchmark_id", values="score", aggfunc="mean"
    )


def item_triples(session: Session) -> pd.DataFrame:
    """Return item-level (model_id, benchmark_id, item_id, correct) triples."""
    rows = session.execute(
        select(
            ScoreRecord.model_id,
            ScoreRecord.benchmark_id,
            ScoreRecord.item_id,
            ScoreRecord.correct,
        ).where(ScoreRecord.item_id.is_not(None), ScoreRecord.correct.is_not(None))
    ).all()
    return pd.DataFrame(rows, columns=["model_id", "benchmark_id", "item_id", "correct"])


def list_models(session: Session) -> list[ModelRecord]:
    return list(session.scalars(select(ModelRecord).order_by(ModelRecord.model_id)))


def list_benchmarks(session: Session) -> list[BenchmarkRecord]:
    return list(session.scalars(select(BenchmarkRecord).order_by(BenchmarkRecord.benchmark_id)))
