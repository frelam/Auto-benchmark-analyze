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
    detect_served_model,
    execute_run,
    read_scores_model,
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


def test_execute_run_benchmark_mode_writes_scores_and_skips_diagnosis(run_env):
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model="llama-3-8b", mode="benchmark", source="weights",
        model_path="meta-llama/Llama-3-8B",
    )
    proc = _StubProc()
    result = execute_run(
        settings, request,
        deploy_weights=lambda cmd: proc,
        wait_ready=lambda _: True,
        run_harness=lambda cmd: _harness_results(),
    )
    assert result.mode == "benchmark"
    assert result.figure_paths == []
    assert proc.terminated and proc.waited  # weights still torn down
    # scores.json is written and shaped as the --scores input format.
    assert result.report_path == result.metrics_path
    assert result.report_path.exists()
    written = json.loads(result.report_path.read_text(encoding="utf-8"))
    # scores.json carries the _model metadata key (the explicit --model value
    # here) so a follow-up `run --scores <path>` picks up the model name
    # without needing --model.
    assert written == {
        "mmlu_pro": 0.5, "math": 0.4, "swe_bench": 0.18,
        "_model": "llama-3-8b",
    }
    # no diagnosis artifacts: clusters absent, report carries only scores.
    assert "clusters" not in result.report
    assert result.report["scores"] == {"mmlu_pro": 0.5, "math": 0.4, "swe_bench": 0.18}


def test_build_run_request_accepts_benchmark_mode(run_env):
    settings, tmp_path, scores_path = run_env
    request = build_run_request(
        settings, model_path="meta-llama/Llama-3-8B", mode="benchmark"
    )
    assert request.mode == "benchmark"
    assert request.source == "weights"


def test_execute_run_weights_path_deploys_and_tears_down(run_env, monkeypatch):
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model="llama-3-8b", source="weights", model_path="meta-llama/Llama-3-8B"
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
        model="llama-3-8b", source="weights", model_path="meta-llama/Llama-3-8B"
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


def test_build_run_request_benchmarks_cli_overrides_config(run_env):
    settings, tmp_path, scores_path = run_env
    settings.run.model.benchmarks = ["mmlu_pro", "gsm8k"]
    request = build_run_request(
        settings, model_path="meta-llama/Llama-3-8B", benchmarks=["math", "swe_bench"]
    )
    assert request.benchmarks == ["math", "swe_bench"]


def test_build_run_request_benchmarks_from_config(run_env):
    settings, tmp_path, scores_path = run_env
    settings.run.model.benchmarks = ["mmlu_pro", "math"]
    request = build_run_request(settings, model_path="meta-llama/Llama-3-8B")
    assert request.benchmarks == ["mmlu_pro", "math"]


def test_build_run_request_benchmarks_ignored_on_scores_path(run_env):
    settings, tmp_path, scores_path = run_env
    # CLI --benchmarks on the scores path must be ignored (scores file carries
    # its own benchmark set); otherwise the user would get a confusing "subset"
    # that has no effect on the eval (which is skipped).
    request = build_run_request(
        settings,
        model="llama-3-8b",
        scores=str(scores_path),
        benchmarks=["mmlu_pro", "math"],
    )
    assert request.source == "scores"
    assert request.benchmarks is None


def test_build_run_request_benchmarks_empty_list_is_unset(run_env):
    settings, tmp_path, scores_path = run_env
    settings.run.model.benchmarks = []
    request = build_run_request(settings, model_path="meta-llama/Llama-3-8B")
    assert request.benchmarks is None


def test_execute_run_benchmarks_subset_limits_eval_tasks(run_env):
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model="llama-3-8b",
        source="weights",
        model_path="meta-llama/Llama-3-8B",
        benchmarks=["mmlu_pro", "math", "swe_bench"],
    )
    captured: list[list[str]] = []

    def fake_harness(cmd):
        captured.append(cmd)
        return _harness_results()

    execute_run(
        settings,
        request,
        deploy_weights=lambda cmd: _StubProc(),
        wait_ready=lambda _: True,
        run_harness=fake_harness,
    )
    # Only the requested subset is sent to the harness.
    assert len(captured) == 1
    tasks_idx = captured[0].index("--tasks") + 1
    tasks = captured[0][tasks_idx].split(",")
    assert set(tasks) == {"mmlu_pro", "math", "swe_bench"}


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
    # --model is optional for weights: auto-derived as the basename of model_path.
    request = build_run_request(settings, model_path="meta-llama/Llama-3-8B")
    assert request.source == "weights"
    assert request.model == "Llama-3-8B"
    assert request.model_path == "meta-llama/Llama-3-8B"


def test_build_run_request_weights_local_path_derives_model(run_env):
    settings, tmp_path, scores_path = run_env
    # A local filesystem path also derives a clean model name from its basename.
    request = build_run_request(settings, model_path="/data/models/my-model/")
    assert request.source == "weights"
    assert request.model == "my-model"
    assert request.model_path == "/data/models/my-model/"


def test_build_run_request_weights_explicit_model_wins(run_env):
    settings, tmp_path, scores_path = run_env
    # An explicit --model still takes precedence over the derived basename.
    request = build_run_request(
        settings, model="llama-3-8b", model_path="meta-llama/Llama-3-8B"
    )
    assert request.model == "llama-3-8b"


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


# --- model-name auto-derivation for endpoint / scores paths ----------------


def test_build_run_request_endpoint_allows_missing_model(run_env):
    """--model is optional on the endpoint path (auto-detected in execute_run)."""
    settings, tmp_path, scores_path = run_env
    request = build_run_request(settings, base_url="http://example:8000/v1")
    assert request.source == "endpoint"
    assert request.base_url == "http://example:8000/v1"
    assert request.model is None  # resolved later in execute_run


def test_build_run_request_scores_allows_missing_model(run_env):
    """--model is optional on the scores path (read from _model in execute_run)."""
    settings, tmp_path, scores_path = run_env
    request = build_run_request(settings, scores=str(scores_path))
    assert request.source == "scores"
    assert request.scores_file == scores_path
    assert request.model is None  # resolved later in execute_run


def test_read_scores_model_returns_metadata_key(tmp_path):
    """read_scores_model reads the optional _model metadata field."""
    p = tmp_path / "scores.json"
    p.write_text(
        json.dumps({"_model": "llama-3-8b", "_note": "x", "mmlu_pro": 50.0}),
        encoding="utf-8",
    )
    assert read_scores_model(p) == "llama-3-8b"


def test_read_scores_model_returns_none_when_absent(tmp_path):
    p = tmp_path / "scores.json"
    p.write_text(json.dumps({"mmlu_pro": 50.0}), encoding="utf-8")
    assert read_scores_model(p) is None


def test_read_scores_model_returns_none_on_invalid_json(tmp_path):
    p = tmp_path / "scores.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert read_scores_model(p) is None


def test_detect_served_model_returns_first_id(monkeypatch):
    """detect_served_model probes /v1/models and returns the first model id."""
    import benchmark_diagnosis.runner as runner_mod

    class _Resp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "served-model-name"}, {"id": "other"}]}

    class _Client:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            assert url == "http://example:8000/v1/models"
            return _Resp()

    monkeypatch.setattr(runner_mod.httpx, "Client", _Client)
    assert detect_served_model("http://example:8000/v1") == "served-model-name"


def test_detect_served_model_returns_none_on_non_200(monkeypatch):
    import benchmark_diagnosis.runner as runner_mod

    class _Resp:
        status_code = 503

        def json(self):
            return {}

    class _Client:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            return _Resp()

    monkeypatch.setattr(runner_mod.httpx, "Client", _Client)
    assert detect_served_model("http://example:8000/v1") is None


def test_detect_served_model_returns_none_on_http_error(monkeypatch):
    import httpx

    import benchmark_diagnosis.runner as runner_mod

    class _Client:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(runner_mod.httpx, "Client", _Client)
    assert detect_served_model("http://example:8000/v1") is None


def test_execute_run_scores_path_resolves_model_from_metadata(run_env):
    """When --model is omitted on the scores path, _model is read from the file."""
    settings, tmp_path, scores_path = run_env
    # Write a scores file carrying _model metadata.
    scores_path.write_text(
        json.dumps({"_model": "from-file", **SCORES}), encoding="utf-8"
    )
    request = RunRequest(model=None, source="scores", scores_file=scores_path)
    result = execute_run(settings, request, deploy_weights=_NoServer)
    assert result.mode == "full"
    # The model name from _model propagates into the metrics.
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["model"]["model_id"] == "from-file"


def test_execute_run_scores_path_falls_back_to_file_stem(run_env):
    """No --model and no _model key: the model name falls back to the file stem."""
    settings, tmp_path, scores_path = run_env
    # scores_path.name == "scores.json" -> stem "scores"
    request = RunRequest(model=None, source="scores", scores_file=scores_path)
    result = execute_run(settings, request, deploy_weights=_NoServer)
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["model"]["model_id"] == "scores"


def test_execute_run_endpoint_path_auto_detects_model(run_env, monkeypatch):
    """On the endpoint path with --model omitted, /v1/models is probed."""
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model=None, source="endpoint", base_url="http://example:8000/v1"
    )
    monkeypatch.setattr(
        "benchmark_diagnosis.runner.detect_served_model",
        lambda url, **kw: "auto-detected",
    )
    result = execute_run(
        settings, request,
        deploy_weights=_NoServer,
        wait_ready=lambda _: True,
        run_harness=lambda cmd: _harness_results(),
    )
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["model"]["model_id"] == "auto-detected"


def test_execute_run_endpoint_path_fallback_when_detection_fails(run_env, monkeypatch):
    """If /v1/models is unreachable, the endpoint path falls back to a default."""
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model=None, source="endpoint", base_url="http://example:8000/v1"
    )
    monkeypatch.setattr(
        "benchmark_diagnosis.runner.detect_served_model",
        lambda url, **kw: None,
    )
    result = execute_run(
        settings, request,
        deploy_weights=_NoServer,
        wait_ready=lambda _: True,
        run_harness=lambda cmd: _harness_results(),
    )
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["model"]["model_id"] == "served-model"


def test_execute_run_benchmark_mode_endpoint_writes_model_metadata(run_env, monkeypatch):
    """benchmark mode on the endpoint path also writes _model into scores.json."""
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model=None, mode="benchmark", source="endpoint",
        base_url="http://example:8000/v1",
    )
    monkeypatch.setattr(
        "benchmark_diagnosis.runner.detect_served_model",
        lambda url, **kw: "qwen-2.5-72b",
    )
    result = execute_run(
        settings, request,
        deploy_weights=_NoServer,
        wait_ready=lambda _: True,
        run_harness=lambda cmd: _harness_results(),
    )
    written = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert written["_model"] == "qwen-2.5-72b"
    assert written["mmlu_pro"] == 0.5
