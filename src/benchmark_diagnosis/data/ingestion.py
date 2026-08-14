"""Ingestion of models / benchmarks / scores into the persistent store.

Inputs may come from a structured JSON (see ``seed/seed_reference.json`` for the
shape), from a CSV, or from an lm-evaluation-harness ``results.json``.
"""

from __future__ import annotations

import datetime as dt
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from benchmark_diagnosis.core.schema import BenchmarkRecord, ModelRecord, ScoreRecord

_SEED_PATH = Path(files("benchmark_diagnosis")) / "data" / "seed" / "seed_reference.json"


def ingest_models(session: Session, models: list[dict[str, Any]]) -> int:
    n = 0
    for m in models:
        existing = session.get(ModelRecord, m["model_id"])
        if existing is not None:
            continue
        session.add(
            ModelRecord(
                model_id=m["model_id"],
                name=m.get("name", m["model_id"]),
                arch_type=m.get("arch_type", "dense"),
                total_params=m.get("total_params"),
                active_params=m.get("active_params"),
                train_compute=m.get("train_compute"),
                release_date=_parse_date(m.get("release_date")),
            )
        )
        n += 1
    session.commit()
    return n


def ingest_benchmarks(session: Session, benchmarks: list[dict[str, Any]]) -> int:
    n = 0
    for b in benchmarks:
        existing = session.get(BenchmarkRecord, b["benchmark_id"])
        if existing is not None:
            continue
        session.add(
            BenchmarkRecord(
                benchmark_id=b["benchmark_id"],
                name=b.get("name", b["benchmark_id"]),
                scoring_method=b.get("scoring_method", "rule_verified"),
                declared_tags=b.get("declared_tags"),
                version=b.get("version"),
                source_url=b.get("source_url"),
            )
        )
        n += 1
    session.commit()
    return n


def ingest_scores(
    session: Session,
    scores: list[dict[str, Any]],
    *,
    eval_date: dt.datetime | None = None,
) -> int:
    n = 0
    for s in scores:
        if s.get("score") is None and s.get("correct") is None:
            continue
        session.add(
            ScoreRecord(
                model_id=s["model_id"],
                benchmark_id=s["benchmark_id"],
                item_id=s.get("item_id"),
                score=s.get("score"),
                correct=s.get("correct"),
                eval_date=eval_date or dt.datetime.utcnow(),
            )
        )
        n += 1
    session.commit()
    return n


def ingest_json(session: Session, payload: dict[str, Any]) -> dict[str, int]:
    """Ingest a payload with optional ``benchmarks`` / ``models`` / ``scores`` keys."""
    counts = {
        "models": ingest_models(session, payload.get("models", [])),
        "benchmarks": ingest_benchmarks(session, payload.get("benchmarks", [])),
        "scores": ingest_scores(session, payload.get("scores", [])),
    }
    return counts


def load_seed(session: Session) -> dict[str, int]:
    """Load the packaged seed reference data (idempotent)."""
    payload = json.loads(_SEED_PATH.read_text(encoding="utf-8"))
    return ingest_json(session, payload)


def ingest_lm_eval_results(
    session: Session,
    results: dict[str, Any],
    model_id: str,
    benchmark_id: str,
    *,
    eval_date: dt.datetime | None = None,
) -> int:
    """Ingest an lm-evaluation-harness ``results.json`` payload.

    Each key under ``results`` is treated as a benchmark/subtask; its primary
    accuracy metric is stored as an aggregate score. If ``samples`` (per-example
    correctness) is present, item-level records are stored too.
    """
    n = 0
    for task, metrics in results.get("results", {}).items():
        value = primary_metric(metrics)
        if value is None:
            continue
        session.add(
            ScoreRecord(
                model_id=model_id,
                benchmark_id=f"{benchmark_id}/{task}",
                score=value,
                eval_date=eval_date or dt.datetime.utcnow(),
            )
        )
        n += 1

    for sample in results.get("samples", []):
        session.add(
            ScoreRecord(
                model_id=model_id,
                benchmark_id=f"{benchmark_id}/{sample.get('task', 'unknown')}",
                item_id=sample.get("doc_id") or sample.get("id"),
                correct=bool(sample.get("correct")) if sample.get("correct") is not None else None,
                eval_date=eval_date or dt.datetime.utcnow(),
            )
        )
        n += 1

    session.commit()
    return n


def primary_metric(metrics: dict[str, Any]) -> float | None:
    """Pick the headline accuracy metric from an lm-eval metrics dict.

    Metric keys look like ``"acc,none"`` or ``"exact_match,strict-match"``; we
    prefer the more stringent metric then fall back to ``acc``.
    """
    priority = ("exact_match", "pass@1", "acc", "f1")
    for prefix in priority:
        for key, value in metrics.items():
            if key.split(",")[0] == prefix and isinstance(value, (int, float)):
                return float(value)
    return None


def _parse_date(value: str | None) -> dt.date | None:
    if value is None:
        return None
    return dt.date.fromisoformat(value)
