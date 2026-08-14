"""Database session management and versioned-asset helpers."""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from benchmark_diagnosis.core.schema import Asset, Base

ASSET_TYPES = ("coverage", "portfolio", "curves")


def make_engine(db_path: str | Path) -> Engine:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", future=True)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, future=True, expire_on_commit=False)


def new_version_id(prefix: str) -> str:
    """Deterministic-ish version id: prefix + UTC timestamp + short uuid."""
    stamp = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def save_asset(
    session: Session,
    asset_type: str,
    payload: dict | list,
    *,
    note: str | None = None,
) -> str:
    """Persist an artifact and return its version id for traceability."""
    if asset_type not in ASSET_TYPES:
        raise ValueError(f"unknown asset_type {asset_type!r}; expected one of {ASSET_TYPES}")
    version_id = new_version_id(asset_type)
    session.add(
        Asset(asset_type=asset_type, version_id=version_id, note=note, payload=payload)
    )
    session.commit()
    return version_id


def load_latest_asset(session: Session, asset_type: str) -> dict | list | None:
    """Return the newest payload of ``asset_type``, or None if none exists."""
    asset = _latest_asset(session, asset_type)
    return asset.payload if asset is not None else None


def latest_version_id(session: Session, asset_type: str) -> str | None:
    """Return the version id of the newest ``asset_type`` asset, or None."""
    asset = _latest_asset(session, asset_type)
    return asset.version_id if asset is not None else None


def _latest_asset(session: Session, asset_type: str) -> Asset | None:
    stmt = (
        select(Asset)
        .where(Asset.asset_type == asset_type)
        .order_by(Asset.id.desc())
        .limit(1)
    )
    return session.scalar(stmt)


def load_asset_by_version(session: Session, asset_type: str, version_id: str) -> dict | list | None:
    stmt = select(Asset).where(
        Asset.asset_type == asset_type, Asset.version_id == version_id
    )
    asset = session.scalar(stmt)
    return asset.payload if asset is not None else None


def iter_assets(session: Session, asset_type: str) -> Iterator[Asset]:
    stmt = select(Asset).where(Asset.asset_type == asset_type).order_by(Asset.id)
    yield from session.scalars(stmt)
