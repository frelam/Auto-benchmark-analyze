"""Tests for the lm-evaluation-harness bridge."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from benchmark_diagnosis.evaluation_orchestration import harness_bridge
from benchmark_diagnosis.evaluation_orchestration.harness_bridge import (
    build_command,
    extract_scores,
    parse_results,
    run_eval,
)


def test_build_command_local_completions_path():
    cmd = build_command("my-model", ["arc_easy", "mmlu"], base_url="http://x:8000/v1")
    assert cmd[:3] == ["lm_eval", "--model", "local-completions"]
    # lm-eval 0.4.12 POSTs directly to base_url, so the bridge normalizes a
    # /v1 root to the full completions endpoint.
    assert cmd[cmd.index("--model_args") + 1] == (
        "model=my-model,base_url=http://x:8000/v1/completions"
    )
    assert cmd[cmd.index("--tasks") + 1] == "arc_easy,mmlu"


def test_build_command_local_completions_keeps_full_url():
    cmd = build_command("m", ["t"], base_url="http://x:8000/v1/completions")
    assert "base_url=http://x:8000/v1/completions" in cmd[
        cmd.index("--model_args") + 1
    ]


def test_build_command_apply_chat_template_forces_local_tokenizer():
    cmd = build_command(
        "m",
        ["gsm8k"],
        base_url="http://x:8000/v1",
        tokenizer="/models/m",
        apply_chat_template=True,
    )
    # --apply_chat_template must render the template client-side: force the
    # huggingface backend, otherwise the remote tokenizer would silently skip
    # templating and the model would receive bare prompts.
    args = cmd[cmd.index("--model_args") + 1]
    assert "tokenizer_backend=huggingface" in args
    assert "tokenizer=/models/m" in args
    assert "--apply_chat_template" in cmd
    # wire protocol stays the raw completions endpoint (loglikelihood relies
    # on echo+logprobs which /v1/chat/completions does not provide)
    assert "base_url=http://x:8000/v1/completions" in args


def test_build_command_apply_chat_template_without_tokenizer_raises():
    with pytest.raises(ValueError, match="requires a local HF tokenizer"):
        build_command(
            "m", ["gsm8k"], base_url="http://x:8000/v1", apply_chat_template=True
        )


def test_build_command_apply_chat_template_off_by_default():
    cmd = build_command("m", ["t"], base_url="http://x", tokenizer="/models/m")
    assert "--apply_chat_template" not in cmd
    assert "tokenizer_backend=huggingface" not in cmd[
        cmd.index("--model_args") + 1
    ]


def test_build_command_local_completions_extra_args():
    cmd = build_command(
        "m",
        ["t"],
        base_url="http://x:8000/v1",
        tokenizer="/models/m",
        max_gen_toks=1024,
        num_concurrent=16,
        max_length=32768,
        confirm_run_unsafe_code=True,
    )
    args = cmd[cmd.index("--model_args") + 1]
    assert "tokenizer=/models/m" in args
    assert "max_gen_toks=1024" in args
    assert "num_concurrent=16" in args
    assert "max_length=32768" in args
    assert "--confirm_run_unsafe_code" in cmd


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


def test_parse_results_falls_back_to_model_subdir(tmp_path):
    # lm-eval >= 0.4.12 writes <output_dir>/<model>/results_<ts>.json
    sub = tmp_path / "model-id"
    sub.mkdir()
    (sub / "results_2026-01-01T00-00-00.000000.json").write_text(
        json.dumps({"results": {"t": {"acc,none": 0.9}}}), encoding="utf-8"
    )
    assert parse_results(tmp_path / "results.json")["results"]["t"]["acc,none"] == 0.9


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


def test_extract_scores_prefers_stricter_metric():
    results = {
        "results": {
            "task_exact": {"acc,none": 0.5, "exact_match,strict-match": 0.42},
            "task_pass": {"acc,none": 0.6, "pass@1": 0.55},
            "task_acc": {"acc,none": 0.71},
            "task_skip": {"bleu,none": 0.3},  # no recognized metric -> dropped
        }
    }
    scores = extract_scores(results)
    assert scores == {"task_exact": 0.42, "task_pass": 0.55, "task_acc": 0.71}


def test_extract_scores_empty_results():
    assert extract_scores({}) == {}
    assert extract_scores({"results": {}}) == {}


def test_run_eval_failure_raises_runtime_error(monkeypatch):
    cmd = build_command("m", ["t"], base_url="http://x")

    def fake_run(argv, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1, cmd=argv, stderr="cuda out of memory\n"
        )

    monkeypatch.setattr(harness_bridge.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="cuda out of memory"):
        run_eval(cmd)
