"""Tests for the end-to-end runner (unified ``run`` path)."""

from __future__ import annotations

import json

import pytest

from benchmark_diagnosis.config import load_config
from benchmark_diagnosis.core import db
from benchmark_diagnosis.data import ingestion
from benchmark_diagnosis.runner import (
    RunRequest,
    build_run_request,
    execute_run,
)

SCORES = {"mmlu_pro": 50.0, "math": 40.0, "swe_bench": 18.0}


@pytest.fixture()
def run_env(tmp_path):
    settings = load_config()
    settings.storage.db_path = str(tmp_path / "test.db")
    settings.run.output.dir = str(tmp_path / "out")
    engine = db.make_engine(settings.storage.db_path)
    db.init_db(engine)
    session = db.session_factory(engine)()
    ingestion.load_seed(session)
    session.close()
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(SCORES), encoding="utf-8")
    return settings, tmp_path, scores_path


class _NoServer:
    """Subprocess-like stub that fails loudly if deploy/eval is attempted."""

    def __init__(self) -> None:
        raise AssertionError("weights deploy must not run for the scores path")

    def terminate(self) -> None:
        pass

    def wait(self) -> None:
        pass


class _StubProc:
    """Subprocess-like stub recording terminate()/wait() for teardown checks."""

    def __init__(self) -> None:
        self.terminated = False
        self.waited = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self) -> None:
        self.waited = True


def _harness_results() -> dict:
    return {
        "results": {
            "mmlu_pro": {"acc,none": 0.5},
            "math": {"acc,none": 0.4},
            "swe_bench": {"pass@1": 0.18},
        }
    }


def test_execute_run_scores_path_writes_artifacts(run_env):
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model="llama-3-8b", source="scores", scores_file=scores_path
    )
    result = execute_run(
        settings, request, deploy_weights=_NoServer, wait_ready=lambda _: True
    )

    assert result.mode == "full"
    assert result.advisor_mode == "rules"  # no analyst LLM configured
    assert result.report["scores"] == SCORES
    assert result.report_path.exists()
    assert result.metrics_path.exists()

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["mode"] == "full"
    assert metrics["advisor_mode"] == "rules"
    assert metrics["model"]["model_id"] == "llama-3-8b"
    assert len(metrics["clusters"]) >= 1

    report_text = result.report_path.read_text(encoding="utf-8")
    assert "**mode**: full | **advisor**: rules" in report_text
    assert "## Summary" in report_text


def test_execute_run_scores_path_renders_figures(run_env):
    pytest.importorskip("matplotlib")
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model="llama-3-8b", source="scores", scores_file=scores_path
    )
    result = execute_run(settings, request, deploy_weights=_NoServer)
    assert len(result.figure_paths) == 3
    assert all(p.exists() for p in result.figure_paths)
    report_text = result.report_path.read_text(encoding="utf-8")
    assert "## Figures" in report_text


def test_execute_run_analyze_mode_skips_recommendations(run_env):
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model="llama-3-8b", mode="analyze", source="scores", scores_file=scores_path
    )
    result = execute_run(settings, request, deploy_weights=_NoServer)
    assert result.report["mode"] == "analyze"
    assert result.report["clusters"]
    # analyze mode skips Stage 6 suggestions: training list empty.
    suggestions = (result.report["diagnosis"].get("suggestions") or {})
    assert (suggestions.get("training") or []) == []


def test_execute_run_weights_path_deploys_and_tears_down(run_env, monkeypatch):
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model="llama-3-8b", source="weights", weights="meta-llama/Llama-3-8B"
    )
    proc = _StubProc()
    called: list[str] = []

    def fake_deploy(cmd):
        called.append("deploy")
        assert cmd[0] == "vllm"
        return proc

    result = execute_run(
        settings,
        request,
        deploy_weights=fake_deploy,
        wait_ready=lambda _: True,
        run_harness=lambda cmd: _harness_results(),
    )
    assert called == ["deploy"]
    assert proc.terminated and proc.waited  # torn down in finally
    assert result.served is True
    assert result.report["scores"] == {"mmlu_pro": 0.5, "math": 0.4, "swe_bench": 0.18}
    assert result.report_path.exists()


def test_execute_run_deploy_failure_tears_down(run_env):
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model="llama-3-8b", source="weights", weights="meta-llama/Llama-3-8B"
    )
    proc = _StubProc()

    with pytest.raises(RuntimeError, match="did not become ready"):
        execute_run(
            settings,
            request,
            deploy_weights=lambda cmd: proc,
            wait_ready=lambda _: False,
        )
    assert proc.terminated and proc.waited


def test_execute_run_non_overlapping_scores_no_crash(run_env):
    settings, tmp_path, scores_path = run_env
    off_portfolio = tmp_path / "off.json"
    off_portfolio.write_text(
        json.dumps({"gsm8k": 79.0, "humaneval": 60.0}), encoding="utf-8"
    )
    request = RunRequest(
        model="llama-3-8b", source="scores", scores_file=off_portfolio
    )
    result = execute_run(settings, request, deploy_weights=_NoServer)
    assert result.report["clusters"] == []
    assert result.report["scores"] == {"gsm8k": 79.0, "humaneval": 60.0}


def test_execute_run_fails_fast_on_forced_llm_without_model(run_env):
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model="llama-3-8b",
        source="scores",
        scores_file=scores_path,
        advisor_mode="llm_rules",
    )
    with pytest.raises(ValueError, match="requires an analyst LLM"):
        execute_run(settings, request, deploy_weights=_NoServer)


def test_execute_run_rejects_unknown_mode(run_env):
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model="llama-3-8b", mode="bogus", source="scores", scores_file=scores_path
    )
    with pytest.raises(ValueError, match="unknown mode"):
        execute_run(settings, request, deploy_weights=_NoServer)


def test_build_run_request_from_config_scores(run_env):
    settings, tmp_path, scores_path = run_env
    settings.run.model.name = "llama-3-8b"
    settings.run.model.source = None
    settings.run.model.scores_file = str(scores_path)
    request = build_run_request(settings)
    assert request.source == "scores"
    assert request.scores_file == scores_path
    assert request.model == "llama-3-8b"
    assert request.mode == "full"


def test_build_run_request_weights_defaults_model(run_env):
    settings, tmp_path, scores_path = run_env
    request = build_run_request(settings, model_id="meta-llama/Llama-3-8B")
    assert request.source == "weights"
    assert request.model == "meta-llama/Llama-3-8B"
    assert request.weights == "meta-llama/Llama-3-8B"


def test_build_run_request_cli_overrides_config_source(run_env):
    settings, tmp_path, scores_path = run_env
    settings.run.model.name = "llama-3-8b"
    settings.run.model.base_url = "http://example:8000/v1"
    # CLI --scores shadows the config's endpoint entirely (no ambiguity error).
    request = build_run_request(settings, scores=str(scores_path))
    assert request.source == "scores"
    assert request.base_url is None


def test_build_run_request_mutual_exclusion_cli(run_env):
    settings, tmp_path, scores_path = run_env
    with pytest.raises(ValueError, match="ambiguous model source"):
        build_run_request(settings, scores=str(scores_path), base_url="http://x")


def test_build_run_request_mutual_exclusion_config(run_env):
    settings, tmp_path, scores_path = run_env
    settings.run.model.scores_file = str(scores_path)
    settings.run.model.base_url = "http://x"
    with pytest.raises(ValueError, match="ambiguous run.model source"):
        build_run_request(settings)


def test_build_run_request_no_source_raises(run_env):
    settings, tmp_path, scores_path = run_env
    with pytest.raises(ValueError, match="no model source"):
        build_run_request(settings)


def test_build_run_request_rejects_bogus_mode(run_env):
    settings, tmp_path, scores_path = run_env
    with pytest.raises(ValueError, match="unknown mode"):
        build_run_request(settings, scores=str(scores_path), mode="bogus")
