"""End-to-end pipeline test: seed -> offline assets -> diagnosis report."""

from __future__ import annotations

import pytest

from benchmark_diagnosis.config import load_config
from benchmark_diagnosis.core import db
from benchmark_diagnosis.core.schema import ModelRecord
from benchmark_diagnosis.data import ingestion
from benchmark_diagnosis.pipeline import build_offline, diagnose_model


@pytest.fixture()
def seeded_session(tmp_path):
    settings = load_config()
    settings.storage.db_path = str(tmp_path / "test.db")
    engine = db.make_engine(settings.storage.db_path)
    db.init_db(engine)
    session = db.session_factory(engine)()
    ingestion.load_seed(session)
    yield session, settings
    session.close()


def test_build_offline_and_diagnose(seeded_session):
    session, settings = seeded_session
    versions = build_offline(session, settings)
    assert set(versions) == {"coverage_version", "portfolio_version", "curves_version"}
    for key, version_id in versions.items():
        asset_type = key.removesuffix("_version")
        assert db.load_asset_by_version(session, asset_type, version_id) is not None

    model = ModelRecord(
        model_id="llama-3-8b",
        name="Llama-3-8B",
        arch_type="dense",
        total_params=8.0,
        active_params=8.0,
    )
    # Include scores that overlap the current representative portfolios (math in
    # the general cluster, swe_bench in the code cluster) so cluster verdicts
    # actually render; scores only count when their benchmark is in a portfolio.
    raw_scores = {"mmlu": 66.6, "math": 55.0, "swe_bench": 18.0}
    report = diagnose_model(session, model, raw_scores, settings)

    assert report["model"]["model_id"] == "llama-3-8b"
    assert report["versions"]["coverage_version"] == versions["coverage_version"]
    assert "clusters" in report
    assert len(report["clusters"]) >= 1
    for cluster in report["clusters"]:
        assert "percentile" in cluster and "underperforming" in cluster
        assert "recommendations" in cluster


def test_diagnose_without_offline_assets_returns_empty(seeded_session):
    session, settings = seeded_session
    model = ModelRecord(
        model_id="x", name="x", arch_type="dense", total_params=1.0, active_params=1.0
    )
    # No offline assets built -> portfolios empty -> no clusters, no crash.
    report = diagnose_model(session, model, {"mmlu": 50.0}, settings)
    assert report["clusters"] == []
