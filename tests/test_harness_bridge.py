"""Tests for the lm-evaluation-harness bridge."""

from __future__ import annotations

import json
import time

import pytest

from benchmark_diagnosis.evaluation_orchestration import harness_bridge
from benchmark_diagnosis.evaluation_orchestration.harness_bridge import (
    build_command,
    extract_scores,
    extract_task_results,
    fmt_duration,
    parse_results,
    run_eval,
    task_headline,
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


def test_build_command_repeats_and_gen_kwargs():
    cmd = build_command(
        "m",
        ["gsm8k"],
        base_url="http://x",
        repeats=5,
        gen_kwargs={"temperature": 0.7, "top_p": 0.95},
    )
    assert cmd[cmd.index("--repeats") + 1] == "5"
    assert cmd[cmd.index("--gen_kwargs") + 1] == "temperature=0.7,top_p=0.95"


def test_build_command_repeats_one_is_single_pass():
    # repeats=1 (and None) keep the default single greedy pass.
    for repeats in (None, 1):
        cmd = build_command("m", ["t"], base_url="http://x", repeats=repeats)
        assert "--repeats" not in cmd


def test_build_command_gen_kwargs_none_omits_flag():
    cmd = build_command("m", ["t"], base_url="http://x", gen_kwargs=None)
    assert "--gen_kwargs" not in cmd


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

    class FakePopen:
        def __init__(self, argv, **kwargs):
            (out_dir / "results.json").write_text(
                json.dumps({"results": {"t": {"acc,none": 0.5}}}), encoding="utf-8"
            )
            self.stdout = ["task t: done\n"]

        def wait(self):
            return 0

    monkeypatch.setattr(harness_bridge.subprocess, "Popen", FakePopen)
    result = run_eval(cmd, cwd=tmp_path)
    assert result["results"]["t"]["acc,none"] == 0.5


def test_run_eval_streams_output_to_stdout(tmp_path, monkeypatch, capsys):
    cmd = build_command("m", ["t"], base_url="http://x", output_dir=tmp_path)

    class FakePopen:
        def __init__(self, argv, **kwargs):
            self.stdout = ["Building contexts for t...\n", "Requesting API: 50%|\n"]

        def wait(self):
            return 0

    monkeypatch.setattr(harness_bridge.subprocess, "Popen", FakePopen)
    run_eval(cmd, cwd=tmp_path)
    out = capsys.readouterr().out
    # lm-eval progress lines must reach the caller live (no capture until exit)
    assert "Building contexts for t..." in out
    assert "Requesting API: 50%|" in out


def test_run_eval_no_output_path_returns_empty(monkeypatch):
    cmd = build_command("m", ["t"], base_url="http://x")

    class FakePopen:
        def __init__(self, argv, **kwargs):
            self.stdout = []

        def wait(self):
            return 0

    monkeypatch.setattr(harness_bridge.subprocess, "Popen", FakePopen)
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

    class FakePopen:
        def __init__(self, argv, **kwargs):
            self.stdout = ["loading task t...\n", "cuda out of memory\n"]

        def wait(self):
            return 1

    monkeypatch.setattr(harness_bridge.subprocess, "Popen", FakePopen)
    with pytest.raises(RuntimeError, match="cuda out of memory"):
        run_eval(cmd)


def test_build_command_timeout_passthrough():
    cmd = build_command("m", ["t"], base_url="http://x", timeout=3600)
    args = cmd[cmd.index("--model_args") + 1]
    # V100 fix: lm-eval's default 300s aiohttp timeout kills long generations
    assert ",timeout=3600" in args


def test_build_command_timeout_omitted_when_none():
    cmd = build_command("m", ["t"], base_url="http://x")
    args = cmd[cmd.index("--model_args") + 1]
    assert "timeout=" not in args


# --- live status line -------------------------------------------------------


def test_parse_progress_line_condenses_tqdm_bar():
    line = "Requesting API:  99%|█████████▊| 534/541 [1:22:09<00:56,  8.11s/it]"
    assert harness_bridge._parse_progress_line(line) == (
        "Requesting API: 534/541 (99%) · 1:22:09"
    )
    assert harness_bridge._is_progress_line(line) is True
    assert harness_bridge._is_progress_line("Setting up task: 100%|██| 30/30 [00:01<00:00, 17.62it/s]") is True
    # plain log lines are not progress bars
    assert harness_bridge._parse_progress_line("Building contexts for t...") is None
    assert harness_bridge._is_progress_line("Building contexts for t...") is False


def test_fmt_duration():
    assert fmt_duration(42) == "42s"
    assert fmt_duration(302) == "5:02"
    assert fmt_duration(5029) == "1:23:49"


def test_run_eval_live_filters_progress_bars_and_ticks(tmp_path, monkeypatch, capsys):
    cmd = build_command("m", ["t"], base_url="http://x", output_dir=tmp_path)

    def _stream():
        yield "Building contexts for t...\n"
        yield "Requesting API:  50%|██████████     | 270/541 [1:02:09<00:56,  8.11s/it]\n"
        time.sleep(0.15)  # give the status ticker time to fire
        yield "Requesting API: 100%|██████████████| 541/541 [1:23:03<00:00,  9.21s/it]\n"
        yield "task t: done\n"

    class FakePopen:
        def __init__(self, argv, **kwargs):
            self.stdout = _stream()

        def wait(self):
            return 0

    monkeypatch.setattr(harness_bridge.subprocess, "Popen", FakePopen)
    run_eval(cmd, live=True, label="t", tick=0.05)
    out = capsys.readouterr().out
    # raw tqdm bar lines are condensed, not streamed line by line
    assert "Requesting API:  50%|" not in out
    assert "Requesting API: 100%|" not in out
    # real log lines still stream live
    assert "Building contexts for t..." in out
    assert "task t: done" in out
    # the live status line carries the task label + parsed request progress
    assert "⏳ t" in out
    assert "Requesting API: 270/541 (50%)" in out


def test_run_eval_live_still_streams_without_bars(tmp_path, monkeypatch, capsys):
    cmd = build_command("m", ["t"], base_url="http://x", output_dir=tmp_path)

    class FakePopen:
        def __init__(self, argv, **kwargs):
            self.stdout = ["loading task t...\n", "task t: done\n"]

        def wait(self):
            return 0

    monkeypatch.setattr(harness_bridge.subprocess, "Popen", FakePopen)
    run_eval(cmd, live=True, tick=0.05)
    out = capsys.readouterr().out
    assert "loading task t..." in out
    assert "task t: done" in out


# --- per-task score extraction ----------------------------------------------


def test_extract_task_results_aliased_and_group_entries():
    results = {
        "results": {
            "hendrycks_math": {"acc,none": 0.4},
            "math": {"exact_match,strict-match": 0.35},  # same benchmark id, aliased
            "longbench2_2wikimqa": {"acc,none": 0.5},  # group subtask entries
            "longbench2_dureader": {"acc,none": 0.6},
            "other_task": {"acc,none": 0.9},  # unrelated -> dropped
        }
    }
    assert extract_task_results(results, "hendrycks_math") == [
        ("hendrycks_math", "acc", 0.4),
        ("math", "exact_match", 0.35),
    ]
    assert extract_task_results(results, "longbench2") == [
        ("longbench2_2wikimqa", "acc", 0.5),
        ("longbench2_dureader", "acc", 0.6),
    ]
    assert extract_task_results(results, "other_task") == [("other_task", "acc", 0.9)]


def test_extract_task_results_ignores_unrelated_and_metricless():
    results = {
        "results": {
            "mmlu_pro": {"acc,none": 0.5},
            "gsm8k": {"bleu,none": 0.3},  # no recognized metric -> dropped
        }
    }
    assert extract_task_results(results, "mmlu_pro") == [("mmlu_pro", "acc", 0.5)]
    assert extract_task_results(results, "gsm8k") == []


def test_task_headline_prefers_group_aggregate():
    results = {
        "results": {
            "longbench2_2wikimqa": {"acc,none": 0.5},
            "longbench2_dureader": {"acc,none": 0.6},
        },
        "groups": {"longbench2": {"acc,none": 0.55}},
    }
    assert task_headline(results, "longbench2") == ("acc", 0.55)


def test_task_headline_falls_back_to_subtask_mean():
    results = {
        "results": {
            "longbench2_2wikimqa": {"acc,none": 0.5},
            "longbench2_dureader": {"acc,none": 0.7},
        }
    }
    assert task_headline(results, "longbench2") == ("acc", 0.6)


def test_task_headline_none_when_no_attributable_score():
    results = {"results": {"some_other_task": {"acc,none": 0.9}}}
    assert task_headline(results, "longbench2") is None
    assert task_headline({}, "mmlu") is None
