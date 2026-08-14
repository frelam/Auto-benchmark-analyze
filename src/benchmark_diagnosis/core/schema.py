"""Persistent tables (design doc section 1.2) plus a versioned asset table.

All offline-analysis artifacts (coverage table, cluster registry, expectation
curves) are stored as versioned JSON blobs in :class:`Asset` so every diagnosis
report can trace exactly which asset versions it used.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ModelRecord(Base):
    __tablename__ = "model_registry"

    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    arch_type: Mapped[str] = mapped_column(String, nullable=False)  # dense | moe
    total_params: Mapped[float | None] = mapped_column(Float, nullable=True)  # billions
    active_params: Mapped[float | None] = mapped_column(Float, nullable=True)
    train_compute: Mapped[float | None] = mapped_column(Float, nullable=True)  # FLOPs
    release_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)


class BenchmarkRecord(Base):
    __tablename__ = "benchmark_registry"

    benchmark_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    scoring_method: Mapped[str] = mapped_column(String, nullable=False)
    declared_tags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    version: Mapped[str | None] = mapped_column(String, nullable=True)
    source_url: Mapped[str | None] = mapped_column(String, nullable=True)


class ItemRecord(Base):
    """Optional item-level registry (subcategory tags for slicing, section 5.1)."""

    __tablename__ = "item_registry"

    item_id: Mapped[str] = mapped_column(String, primary_key=True)
    benchmark_id: Mapped[str] = mapped_column(
        String, ForeignKey("benchmark_registry.benchmark_id"), nullable=False
    )
    declared_subcategory: Mapped[str | None] = mapped_column(String, nullable=True)


class ScoreRecord(Base):
    """model x benchmark result. `item_id`/`correct` populated for item-level data."""

    __tablename__ = "score_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(
        String, ForeignKey("model_registry.model_id"), index=True, nullable=False
    )
    benchmark_id: Mapped[str] = mapped_column(
        String, ForeignKey("benchmark_registry.benchmark_id"), index=True, nullable=False
    )
    item_id: Mapped[str | None] = mapped_column(String, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)  # aggregate score
    correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # item-level
    eval_date: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )


class Asset(Base):
    """Versioned offline-analysis artifact (coverage / portfolio / curves)."""

    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_type: Mapped[str] = mapped_column(String, index=True, nullable=False)
    version_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
