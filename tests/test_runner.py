"""Tests for the end-to-end runner (unified ``run`` path)."""

from __future__ import annotations

import json

import pytest

from benchmark_diagnosis.config import load_config
from benchmark_diagnosis.core import db
from benchmark_diagnosis.data import ingestion
from benchmark_diagnosis.runner import (
    RunRequest,
    _collect_scores,
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
    settings.diagnosis.enabled = True
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
    settings.diagnosis.enabled = True
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
    settings.diagnosis.enabled = True
    request = RunRequest(
        model="llama-3-8b", mode="analyze", source="scores", scores_file=scores_path
    )
    result = execute_run(settings, request, deploy_weights=_NoServer)
    assert result.report["mode"] == "analyze"
    assert result.report["clusters"]
    # analyze mode trims the final suggestion write-up (rule engine: the
    # capability-dataset suggestions are omitted, diagnosis kept).
    block = result.report["diagnosis"]
    assert block["engine"] == "rule"
    assert block["rule_base"]["missing_capabilities"]
    assert block["rule_base"]["dataset_suggestions"] == []


def test_execute_run_benchmark_mode_writes_scores_and_skips_diagnosis(run_env):
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model="llama-3-8b", mode="benchmark", source="weights",
        weights="meta-llama/Llama-3-8B",
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
    assert written == {"mmlu_pro": 0.5, "math": 0.4, "swe_bench": 0.18}
    # no diagnosis artifacts: clusters absent, report carries only scores.
    assert "clusters" not in result.report
    assert result.report["scores"] == {"mmlu_pro": 0.5, "math": 0.4, "swe_bench": 0.18}


def test_execute_run_passes_evaluation_config_to_harness(run_env, tmp_path):
    settings, tmp_path, scores_path = run_env
    settings.evaluation.tokenizer = "/models/qwen3"
    settings.evaluation.max_gen_toks = 16384
    settings.evaluation.timeout = 7200
    settings.evaluation.confirm_run_unsafe_code = True
    settings.evaluation.apply_chat_template = True
    # Point the harness at a non-existent venv so the chat-template patch
    # resolves nowhere — tests must never write into the real lm_eval install.
    settings.evaluation.harness_cmd = str(tmp_path / "venv" / "bin" / "lm_eval")
    request = RunRequest(
        model="llama-3-8b", mode="benchmark", source="weights",
        weights="meta-llama/Llama-3-8B",
    )
    seen: list[list[str]] = []

    def fake_harness(cmd):
        seen.append(cmd)
        return _harness_results()

    execute_run(
        settings, request,
        deploy_weights=lambda cmd: _StubProc(),
        wait_ready=lambda _: True,
        run_harness=fake_harness,
    )
    cmd = seen[0]
    args = cmd[cmd.index("--model_args") + 1]
    assert "--apply_chat_template" in cmd
    assert "tokenizer=/models/qwen3" in args
    assert "tokenizer_backend=huggingface" in args
    assert "max_gen_toks=16384" in args
    assert "timeout=7200" in args
    assert "--confirm_run_unsafe_code" in cmd


def test_build_run_request_accepts_benchmark_mode(run_env):
    settings, tmp_path, scores_path = run_env
    request = build_run_request(
        settings, model_id="meta-llama/Llama-3-8B", mode="benchmark"
    )
    assert request.mode == "benchmark"
    assert request.source == "weights"


def test_execute_run_weights_path_deploys_and_tears_down(run_env, monkeypatch):
    settings, tmp_path, scores_path = run_env
    settings.diagnosis.enabled = True
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
    settings.diagnosis.enabled = True
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
        settings, model_id="meta-llama/Llama-3-8B", benchmarks=["math", "swe_bench"]
    )
    assert request.benchmarks == ["math", "swe_bench"]


def test_build_run_request_benchmarks_from_config(run_env):
    settings, tmp_path, scores_path = run_env
    settings.run.model.benchmarks = ["mmlu_pro", "math"]
    request = build_run_request(settings, model_id="meta-llama/Llama-3-8B")
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
    request = build_run_request(settings, model_id="meta-llama/Llama-3-8B")
    assert request.benchmarks is None


def test_execute_run_benchmarks_subset_limits_eval_tasks(run_env):
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model="llama-3-8b",
        source="weights",
        weights="meta-llama/Llama-3-8B",
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
    # Only the requested subset reaches the harness — one command per benchmark
    # (evaluated one at a time so each dataset's score prints on completion).
    # Benchmark ids are translated to lm-eval task names (math -> hendrycks_math);
    # non-evaluable ids (swe_bench) pass through and fail in the harness.
    tasks = set()
    for cmd in captured:
        tasks_idx = cmd.index("--tasks") + 1
        tasks.update(cmd[tasks_idx].split(","))
    assert len(captured) == 3
    assert set(tasks) == {"mmlu_pro", "hendrycks_math", "swe_bench"}


def test_execute_run_translates_aliases_and_patches_chat_templates(run_env, tmp_path):
    """math/longbench_v2 must reach the harness as hendrycks_math/longbench2,
    and with apply_chat_template the venv bbh template is adapted in place."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "lm_eval").write_text("#!/bin/sh\n", encoding="utf-8")
    site = venv / "lib" / "python3.11" / "site-packages"
    bbh_yaml = site / "lm_eval" / "tasks" / "bbh" / "cot_fewshot" / "_cot_fewshot_template_yaml"
    bbh_yaml.parent.mkdir(parents=True)
    bbh_yaml.write_text(
        "generation_kwargs:\n  max_gen_toks: 1024\n  until:\n"
        '    - "</s>"\n    - "Q"\n    - "\\n\\n"\n  do_sample: false\n',
        encoding="utf-8",
    )

    settings, tmp_path, scores_path = run_env
    settings.evaluation.harness_cmd = str(venv / "bin" / "lm_eval")
    settings.evaluation.apply_chat_template = True
    settings.evaluation.tokenizer = "/models/qwen3"
    settings.evaluation.max_gen_toks = 16384
    request = RunRequest(
        model="qwen3-4b",
        source="weights",
        weights="qwen3-4b",
        benchmarks=["math", "longbench_v2", "bbh"],
    )
    captured: list[list[str]] = []

    def fake_harness(cmd):
        captured.append(cmd)
        return {"results": {"bbh": {"exact_match,none": 0.5}}}

    execute_run(
        settings,
        request,
        deploy_weights=lambda cmd: _StubProc(),
        wait_ready=lambda _: True,
        run_harness=fake_harness,
    )
    names = {cmd[cmd.index("--tasks") + 1] for cmd in captured}
    assert names == {"hendrycks_math", "longbench2", "bbh"}
    patched = bbh_yaml.read_text(encoding="utf-8")
    assert 'until:\n    - "<|im_end|>"' in patched
    assert "max_gen_toks: 16384" in patched
    assert '"Q"' not in patched


def test_execute_run_repeats_sampling_forwards_to_harness(run_env):
    settings, tmp_path, scores_path = run_env
    # Random-sampling eval: each generative prompt is sampled `repeats` times
    # and the harness averages the metric over the samples.
    settings.evaluation.repeats = 5
    settings.evaluation.gen_kwargs = {"temperature": 0.7, "top_p": 0.95}
    request = RunRequest(
        model="llama-3-8b", source="weights", weights="meta-llama/Llama-3-8B"
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
    # Sampling flags forward on every per-benchmark command.
    assert captured
    for cmd in captured:
        assert cmd[cmd.index("--repeats") + 1] == "5"
        assert cmd[cmd.index("--gen_kwargs") + 1] == (
            "temperature=0.7,top_p=0.95"
        )


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


# ------------------------------------------------------------------ diagnosis gating


def test_execute_run_diagnosis_disabled_by_default_writes_artifacts(run_env):
    """Default config: diagnosis is OFF, but eval artifacts are always archived."""
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model="llama-3-8b", source="scores", scores_file=scores_path
    )
    result = execute_run(settings, request, deploy_weights=_NoServer)

    assert result.report["diagnosed"] is False
    assert "clusters" not in result.report
    out = tmp_path / "out"
    assert (out / "scores.json").exists()
    assert (out / "eval_results.json").exists()
    assert (out / "eval_summary.md").exists()
    assert (out / "bad_cases" / "README.md").exists()
    assert not (out / "report.md").exists()  # no diagnosis report


def test_execute_run_diagnose_request_enables_rule_engine(run_env):
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model="llama-3-8b", source="scores", scores_file=scores_path,
        diagnose=True,
    )
    result = execute_run(settings, request, deploy_weights=_NoServer)
    assert result.report["engine"] == "rule"
    assert result.report["diagnosis"]["engine"] == "rule"
    assert result.report["diagnosis"]["rule_base"]["missing_capabilities"]


def test_execute_run_engine_llm_agent_fails_fast_without_config(run_env):
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model="llama-3-8b", source="scores", scores_file=scores_path,
        diagnose=True, engine="llm_agent",
    )
    with pytest.raises(ValueError, match="llm_agent"):
        execute_run(settings, request, deploy_weights=_NoServer)


def test_build_run_request_diagnose_defaults(run_env):
    settings, tmp_path, scores_path = run_env
    request = build_run_request(settings, model="llama-3-8b", scores=str(scores_path))
    assert request.diagnose is False  # diagnosis.enabled default
    assert request.engine == "rule"
    settings.diagnosis.enabled = True
    request = build_run_request(
        settings, model="llama-3-8b", scores=str(scores_path)
    )
    assert request.diagnose is True
    request = build_run_request(
        settings, model="llama-3-8b", scores=str(scores_path),
        diagnose=False, engine="llm_agent",
    )
    assert request.diagnose is False
    assert request.engine == "llm_agent"


def test_collect_scores_routes_bfcl_to_its_own_backend(run_env):
    """A BFCL-only benchmark runs on the native harness, not lm-eval, and its
    scores merge into the tool-namespace raw_scores (bfcl + bfcl.<category>)."""
    settings, tmp_path, scores_path = run_env
    request = RunRequest(
        model="llama-3-8b",
        source="weights",
        weights="meta-llama/Llama-3-8B",
        benchmarks=["bfcl"],
    )
    harness_calls = []
    bfcl_calls = []

    def fake_bfcl(model, base_url, **kwargs):
        bfcl_calls.append((model, base_url, kwargs))
        return {"bfcl": 0.72, "bfcl.simple": 0.85, "bfcl.multi_turn": 0.6}

    raw, results = _collect_scores(
        request,
        base_url="http://127.0.0.1:8000/v1",
        portfolio_ids=set(),
        run_harness=lambda cmd: harness_calls.append(cmd) or {},
        settings=settings,
        run_bfcl=fake_bfcl,
    )

    assert harness_calls == []  # lm-eval must not be invoked for bfcl
    assert len(bfcl_calls) == 1
    model, base_url, kwargs = bfcl_calls[0]
    assert model == "llama-3-8b"
    assert base_url == "http://127.0.0.1:8000/v1"
    assert kwargs["model_type"] == settings.evaluation.bfcl_model_type
    assert kwargs["categories"] == settings.evaluation.bfcl_categories
    assert raw == {"bfcl": 0.72, "bfcl.simple": 0.85, "bfcl.multi_turn": 0.6}
    assert results["bfcl"] == {"bfcl": 0.72, "bfcl.simple": 0.85, "bfcl.multi_turn": 0.6}
