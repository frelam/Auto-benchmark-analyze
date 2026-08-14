"""Tests for deploy command building, cluster screening, and drill-down triggers."""

from __future__ import annotations

import pytest

from benchmark_diagnosis.config import CurvesConfig, ServingConfig
from benchmark_diagnosis.core.types import ClusterPortfolio
from benchmark_diagnosis.evaluation_orchestration import deploy
from benchmark_diagnosis.evaluation_orchestration.deploy import (
    serve_command,
    wait_until_ready,
)
from benchmark_diagnosis.evaluation_orchestration.drilldown_trigger import (
    should_drilldown,
)
from benchmark_diagnosis.evaluation_orchestration.screening_runner import (
    cluster_scores,
)


def _portfolio(cluster_id: str, benchmarks: list[dict]) -> ClusterPortfolio:
    return ClusterPortfolio(cluster_id=cluster_id, benchmarks=benchmarks)


# --------------------------------------------------------------------------- deploy


def test_serve_command_includes_model_id_and_port():
    cfg = ServingConfig(
        host="0.0.0.0", port=8000, tensor_parallel_size=2, gpu_memory_utilization=0.8
    )
    cmd = serve_command("Qwen/Qwen2.5-7B", cfg)
    assert cmd[:2] == ["vllm", "serve"]
    assert "Qwen/Qwen2.5-7B" in cmd
    assert cmd[cmd.index("--port") + 1] == "8000"
    assert cmd[cmd.index("--host") + 1] == "0.0.0.0"
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.8"


def test_serve_command_served_model_name_matches_model_id():
    cmd = serve_command("my-model", ServingConfig())
    assert cmd[cmd.index("--served-model-name") + 1] == "my-model"


def test_serve_command_max_model_len_and_extra_args():
    cfg = ServingConfig(max_model_len=8192, extra_args=["--enable-prefix-caching"])
    cmd = serve_command("m", cfg)
    assert cmd[cmd.index("--max-model-len") + 1] == "8192"
    assert cmd[-1] == "--enable-prefix-caching"


def test_serve_command_max_model_len_omitted_when_none():
    cmd = serve_command("m", ServingConfig())
    assert "--max-model-len" not in cmd


def test_wait_until_ready_returns_true_on_200(monkeypatch):
    class FakeResponse:
        status_code = 200

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            return FakeResponse()

    monkeypatch.setattr(deploy.httpx, "Client", FakeClient)
    assert wait_until_ready("http://host:8000/v1", timeout_seconds=5.0, interval=0.05)


def test_wait_until_ready_times_out_on_errors(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            raise deploy.httpx.TransportError("connection refused")

    monkeypatch.setattr(deploy.httpx, "Client", FakeClient)
    assert (
        wait_until_ready("http://host:8000/v1", timeout_seconds=0.2, interval=0.05)
        is False
    )


# ------------------------------------------------------------------ cluster_scores


def test_cluster_scores_weighted_sum():
    portfolios = [
        _portfolio(
            "c1",
            [
                {"benchmark_id": "a", "weight": 0.25},
                {"benchmark_id": "b", "weight": 0.75},
            ],
        )
    ]
    raw = {"a": 100.0, "b": 0.0}
    assert cluster_scores(raw, portfolios) == {"c1": 25.0}


def test_cluster_scores_renormalizes_when_benchmark_missing():
    portfolios = [
        _portfolio(
            "c1",
            [
                {"benchmark_id": "a", "weight": 0.5},
                {"benchmark_id": "missing", "weight": 0.5},
            ],
        )
    ]
    raw = {"a": 80.0}
    # "missing" is dropped; "a"'s weight renormalizes to 1.0 -> score 80.0.
    assert cluster_scores(raw, portfolios) == {"c1": 80.0}


def test_cluster_scores_renormalizes_across_multiple_present():
    portfolios = [
        _portfolio(
            "c1",
            [
                {"benchmark_id": "a", "weight": 1.0},
                {"benchmark_id": "b", "weight": 1.0},
                {"benchmark_id": "missing", "weight": 1.0},
            ],
        )
    ]
    raw = {"a": 40.0, "b": 60.0}
    # Present weights 1,1 renormalize to 0.5,0.5 -> 0.5*40 + 0.5*60 = 50.0.
    assert cluster_scores(raw, portfolios) == pytest.approx({"c1": 50.0})


def test_cluster_scores_zero_weights_fall_back_to_equal():
    portfolios = [
        _portfolio(
            "c1",
            [
                {"benchmark_id": "a", "weight": 0.0},
                {"benchmark_id": "b", "weight": 0.0},
            ],
        )
    ]
    raw = {"a": 10.0, "b": 30.0}
    assert cluster_scores(raw, portfolios) == pytest.approx({"c1": 20.0})


def test_cluster_scores_empty_portfolio_omitted():
    portfolios = [_portfolio("c1", [{"benchmark_id": "missing", "weight": 1.0}])]
    assert cluster_scores({"a": 1.0}, portfolios) == {}


def test_cluster_scores_multiple_clusters():
    portfolios = [
        _portfolio("c1", [{"benchmark_id": "a", "weight": 1.0}]),
        _portfolio("c2", [{"benchmark_id": "b", "weight": 1.0}]),
    ]
    raw = {"a": 10.0, "b": 20.0}
    assert cluster_scores(raw, portfolios) == {"c1": 10.0, "c2": 20.0}


# ------------------------------------------------------------- should_drilldown


def _cfg() -> CurvesConfig:
    return CurvesConfig(z_threshold=-1.0, percentile_threshold=25.0)


def test_should_drilldown_underperforming_flag_triggers():
    assert (
        should_drilldown(
            {"percentile": 90.0, "z_score": 5.0, "underperforming": True}, _cfg()
        )
        is True
    )


def test_should_drilldown_z_score_below_threshold_triggers():
    assert (
        should_drilldown(
            {"percentile": 90.0, "z_score": -1.5, "underperforming": False}, _cfg()
        )
        is True
    )


def test_should_drilldown_z_score_at_boundary_does_not_trigger():
    assert (
        should_drilldown(
            {"percentile": 90.0, "z_score": -1.0, "underperforming": False}, _cfg()
        )
        is False
    )


def test_should_drilldown_percentile_below_threshold_triggers():
    assert (
        should_drilldown(
            {"percentile": 24.0, "z_score": 0.0, "underperforming": False}, _cfg()
        )
        is True
    )


def test_should_drilldown_percentile_at_boundary_does_not_trigger():
    assert (
        should_drilldown(
            {"percentile": 25.0, "z_score": 0.0, "underperforming": False}, _cfg()
        )
        is False
    )


def test_should_drilldown_healthy_does_not_trigger():
    assert (
        should_drilldown(
            {"percentile": 80.0, "z_score": 1.2, "underperforming": False}, _cfg()
        )
        is False
    )
