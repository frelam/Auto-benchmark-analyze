"""Tests for the lm-evaluation-harness bridge."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from benchmark_diagnosis.evaluation_orchestration import harness_bridge
from benchmark_diagnosis.evaluation_orchestration.harness_bridge import (
    build_command,
    parse_results,
    run_eval,
)


def test_build_command_local_completions_path():
    cmd = build_command("my-model", ["arc_easy", "mmlu"], base_url="http://x:8000/v1")
    assert cmd[:3] == ["lm_eval", "--model", "local-completions"]
    assert cmd[cmd.index("--model_args") + 1] == (
        "model=my-model,base_url=http://x:8000/v1"
    )
    assert cmd[cmd.index("--tasks") + 1] == "arc_easy,mmlu"


def test_build_command_hf_path():
    cmd = build_command(
        "meta-llama/Llama-2-7b", ["gsm8k"], num_fewshot=5, limit=10
    )
    assert cmd[cmd.index("--model") + 1] == "hf"
    assert cmd[cmd.index("--model_args") + 1] == "pretrained=meta-llama/Llama-2-7b"
    assert cmd[cmd.index("--num_fewshot") + 1] == "5"
    assert cmd[cmd.index("--limit") + 1] == "10"


def test_build_command_log_samples_on_by_default():
    cmd = build_command("m", ["t"], base_url="http://x")
    assert "--log_samples" in cmd


def test_build_command_log_samples_disabled():
    cmd = build_command("m", ["t"], base_url="http://x", log_samples=False)
    assert "--log_samples" not in cmd


def test_build_command_batch_size_and_output_dir(tmp_path):
    out = tmp_path / "run1"
    cmd = build_command("m", ["t"], batch_size="8", output_dir=out)
    assert cmd[cmd.index("--batch_size") + 1] == "8"
    assert cmd[cmd.index("--output_path") + 1] == str(out)


def test_build_command_default_batch_size_auto():
    cmd = build_command("m", ["t"])
    assert cmd[cmd.index("--batch_size") + 1] == "auto"


def test_build_command_custom_harness_cmd():
    cmd = build_command(
        "m", ["t"], base_url="http://x", harness_cmd="python -m lm_eval"
    )
    assert cmd[0] == "python -m lm_eval"


def test_parse_results_missing_returns_empty(tmp_path):
    assert parse_results(tmp_path / "nope.json") == {}


def test_parse_results_roundtrip(tmp_path):
    p = tmp_path / "results.json"
    payload = {"results": {"arc_easy": {"acc,none": 0.7}}}
    p.write_text(json.dumps(payload), encoding="utf-8")
    assert parse_results(p) == payload


def test_run_eval_success_loads_results(tmp_path, monkeypatch):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    cmd = build_command("m", ["t"], base_url="http://x", output_dir=out_dir)

    def fake_run(argv, **kwargs):
        (out_dir / "results.json").write_text(
            json.dumps({"results": {"t": {"acc,none": 0.5}}}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harness_bridge.subprocess, "run", fake_run)
    result = run_eval(cmd, cwd=tmp_path)
    assert result["results"]["t"]["acc,none"] == 0.5


def test_run_eval_no_output_path_returns_empty(monkeypatch):
    cmd = build_command("m", ["t"], base_url="http://x")

    def fake_run(argv, **kwargs):
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harness_bridge.subprocess, "run", fake_run)
    assert run_eval(cmd) == {}


def test_run_eval_failure_raises_runtime_error(monkeypatch):
    cmd = build_command("m", ["t"], base_url="http://x")

    def fake_run(argv, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1, cmd=argv, stderr="cuda out of memory\n"
        )

    monkeypatch.setattr(harness_bridge.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="cuda out of memory"):
        run_eval(cmd)
