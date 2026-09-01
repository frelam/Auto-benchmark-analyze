"""Tests for the chat-template compatibility patches (bbh / humaneval / ifeval).

The fixtures replicate the exact lm-eval 0.4.12 template contents audited on
2026-08-24 (``.venv/lib/python3.11/site-packages/lm_eval/tasks/``), so the
patch logic is verified against the real shapes: block-style ``until`` lists,
``max_gen_toks: 1024`` under ``generation_kwargs``, and humaneval's
``filter_fn: !function utils.build_predictions``.
"""

from __future__ import annotations

import ast
import re

import yaml

from benchmark_diagnosis.config import load_config
from benchmark_diagnosis.evaluation_orchestration.chat_template_patch import (
    find_lm_eval_tasks_dir,
    patch_task_templates_for_chat,
    prepare_chat_template_eval,
)

BBH_TEMPLATE = """dataset_path: SaylorTwift/bbh
output_type: generate_until
test_split: test
doc_to_target: "{{target}}"
target_delimiter: ""
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
generation_kwargs:
  max_gen_toks: 1024
  until:
    - "</s>"
    - "Q"
    - "\\n\\n"
  do_sample: false
  temperature: 0.0
filter_list:
  - name: "get-answer"
    filter:
      - function: "regex"
        regex_pattern: "(?<=the answer is )(.*)(?=.)"
      - function: "take_first"
num_fewshot: 3
metadata:
  version: 4.0
"""

HUMANEVAL_YAML = """task: humaneval
dataset_path: openai/openai_humaneval
unsafe_code: true
output_type: generate_until
test_split: test
doc_to_text: "{{prompt}}"
doc_to_target: "{{test}}\\ncheck({{entry_point}})"
metric_list:
  - metric: !function utils.pass_at_k
    aggregation: mean
    higher_is_better: true
    k: [1]
generation_kwargs:
  until:
    - "\\nclass"
    - "\\ndef"
    - "\\n#"
    - "\\nif"
    - "\\nprint"
  max_gen_toks: 1024
  do_sample: false
repeats: 1
num_fewshot: 0
filter_list:
  - name: "create_test"
    filter:
      - function: "custom"
        filter_fn: !function utils.build_predictions
metadata:
  version: 1.0
"""

IFEVAL_YAML = """output_type: generate_until
test_split: train
num_fewshot: 0
doc_to_text: prompt
doc_to_target: 0
generation_kwargs:
  until: []
  do_sample: false
  temperature: 0.0
  max_gen_toks: 1280
process_results: !function utils.process_results
"""

HUMANEVAL_UTILS = """def pass_at_k(references, predictions, k=None):
    return {}


def build_predictions(resps, docs):
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]
"""

MMLU_DEFAULT = """dataset_path: cais/mmlu
test_split: test
fewshot_split: dev
fewshot_config:
  sampler: first_n
output_type: multiple_choice
doc_to_text: "{{question.strip()}}\\nA. {{choices[0]}}\\nB. {{choices[1]}}\\nC. {{choices[2]}}\\nD. {{choices[3]}}\\nAnswer:"
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: answer
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""

IFEVAL_UTILS = """from typing import Dict, Optional, Union

from lm_eval.tasks.ifeval import instructions_registry


def process_results(doc, results):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )
    response = results[0]
    return test_instruction_following(inp, response)
"""


def _write_tasks_dir(tmp_path, *, humaneval=True, ifeval=True, bbh=True, mmlu=True):
    """Replicate ``lm_eval/tasks`` for the registered templates + utils."""
    tasks = tmp_path / "lm_eval" / "tasks"
    if bbh:
        p = tasks / "bbh" / "cot_fewshot" / "_cot_fewshot_template_yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(BBH_TEMPLATE, encoding="utf-8")
    if mmlu:
        p = tasks / "mmlu" / "default" / "_default_template_yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(MMLU_DEFAULT, encoding="utf-8")
    if humaneval:
        p = tasks / "humaneval" / "humaneval.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(HUMANEVAL_YAML, encoding="utf-8")
        (tasks / "humaneval" / "utils.py").write_text(
            HUMANEVAL_UTILS, encoding="utf-8"
        )
    if ifeval:
        p = tasks / "ifeval" / "ifeval.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(IFEVAL_YAML, encoding="utf-8")
        (tasks / "ifeval" / "utils.py").write_text(IFEVAL_UTILS, encoding="utf-8")
    return tasks


def _make_venv(tmp_path):
    """A venv-like tree: <venv>/bin/lm_eval + lib/python3.11/site-packages."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "lm_eval").write_text("#!/bin/sh\n", encoding="utf-8")
    site = venv / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True)
    return venv, site
def test_find_lm_eval_tasks_dir_from_harness_cmd(tmp_path):
    venv, site = _make_venv(tmp_path)
    tasks = site / "lm_eval" / "tasks"
    tasks.mkdir(parents=True)
    found = find_lm_eval_tasks_dir(str(venv / "bin" / "lm_eval"))
    assert found == tasks


def test_find_lm_eval_tasks_dir_missing_exe_returns_none(tmp_path):
    assert find_lm_eval_tasks_dir(str(tmp_path / "nope" / "bin" / "lm_eval")) is None


def test_patch_bbh_template_until_and_max_gen_toks(tmp_path):
    tasks = _write_tasks_dir(tmp_path, humaneval=False, ifeval=False)
    logs = patch_task_templates_for_chat(tasks, ["bbh"], max_gen_toks=16384)
    text = (tasks / "bbh" / "cot_fewshot" / "_cot_fewshot_template_yaml").read_text(
        encoding="utf-8"
    )
    assert '  until:\n    - "<|im_end|>"\n' in text
    assert '"Q"' not in text
    assert '"\\n\\n"' not in text
    assert "max_gen_toks: 16384" in text
    assert any("until -> [<|im_end|>]" in line for line in logs)
    assert any("max_gen_toks -> 16384" in line for line in logs)
    # idempotent: second run reports no further changes and rewrites nothing.
    before = text
    logs2 = patch_task_templates_for_chat(tasks, ["bbh"], max_gen_toks=16384)
    assert (tasks / "bbh" / "cot_fewshot" / "_cot_fewshot_template_yaml").read_text(
        encoding="utf-8"
    ) == before
    assert any("已 patch (幂等)" in line for line in logs2)


def test_patch_humaneval_rewires_filter_and_appends_utils(tmp_path):
    tasks = _write_tasks_dir(tmp_path, bbh=False, ifeval=False)
    logs = patch_task_templates_for_chat(tasks, ["humaneval"], max_gen_toks=16384)
    yaml_text = (tasks / "humaneval" / "humaneval.yaml").read_text(encoding="utf-8")
    assert "filter_fn: !function utils.build_predictions_chat" in yaml_text
    utils_text = (tasks / "humaneval" / "utils.py").read_text(encoding="utf-8")
    assert "def build_predictions_chat(resps, docs):" in utils_text
    assert any("build_predictions_chat" in line for line in logs)
    # idempotent.
    logs2 = patch_task_templates_for_chat(tasks, ["humaneval"], max_gen_toks=16384)
    assert (tasks / "humaneval" / "utils.py").read_text(encoding="utf-8") == utils_text
    assert any("已 patch (幂等)" in line for line in logs2)


def test_build_predictions_chat_strips_think_and_fences(tmp_path):
    """The appended utils function must yield runnable code from chat output."""
    tasks = _write_tasks_dir(tmp_path, bbh=False, ifeval=False)
    patch_task_templates_for_chat(tasks, ["humaneval"])
    source = (tasks / "humaneval" / "utils.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_predictions_chat"
    )
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<test>", "exec"), ns)  # noqa: S102
    build = ns["build_predictions_chat"]

    prompt = "def add(a, b):\n    \"\"\"doc\"\"\"\n"
    resp = "<think>this is a long CoT</think>\n\n```python\ndef add(a, b):\n    return a + b\n```\n"
    assert build([[resp]], [{"prompt": prompt}]) == [
        [prompt + "def add(a, b):\n    return a + b"]
    ]
    # without fences / think: pass through cleanly.
    plain = "def add(a, b):\n    return a + b"
    assert build([[plain]], [{"prompt": prompt}]) == [[prompt + plain]]


def test_patch_ifeval_only_retargets_max_gen_toks(tmp_path):
    tasks = _write_tasks_dir(tmp_path, bbh=False, humaneval=False)
    logs = patch_task_templates_for_chat(tasks, ["ifeval"], max_gen_toks=16384)
    text = (tasks / "ifeval" / "ifeval.yaml").read_text(encoding="utf-8")
    assert "until: []" in text  # 无碍, 保持不变
    assert "max_gen_toks: 16384" in text
    assert any("max_gen_toks -> 16384" in line for line in logs)


def test_patch_ifeval_utils_strips_think_before_verification(tmp_path):
    tasks = _write_tasks_dir(tmp_path, bbh=False, humaneval=False)
    logs = patch_task_templates_for_chat(tasks, ["ifeval"], max_gen_toks=16384)
    utils = tasks / "ifeval" / "utils.py"
    source = utils.read_text(encoding="utf-8")
    # process_results now sanitizes before checking, and the helper is appended.
    assert "response = _strip_chat_think(results[0])" in source
    assert "def _strip_chat_think(text):" in source
    assert any("_strip_chat_think" in line for line in logs)
    # idempotent.
    before = source
    logs2 = patch_task_templates_for_chat(tasks, ["ifeval"], max_gen_toks=16384)
    assert utils.read_text(encoding="utf-8") == before
    assert any("已 patch (幂等)" in line for line in logs2)

    # verify the appended helper runs and strips the think block in place.
    tree = ast.parse(source)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_strip_chat_think"
    )
    ns: dict = {"_ifeval_re": __import__("re")}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<test>", "exec"), ns)  # noqa: S102
    strip = ns["_strip_chat_think"]
    raw = " thinking\nLet me reason...\n response\n\nThe text is 42 words <|im_end|>"
    assert strip(raw) == "The text is 42 words"


def test_patch_skips_unregistered_tasks_and_missing_files(tmp_path):
    tasks = _write_tasks_dir(tmp_path, bbh=False, humaneval=False, ifeval=False, mmlu=False)
    assert patch_task_templates_for_chat(tasks, ["gsm8k", "mmlu_pro"]) == []
    logs = patch_task_templates_for_chat(tasks, ["bbh"], max_gen_toks=16384)
    assert len(logs) == 1 and "WARNING" in logs[0]


def test_patch_mmlu_rewrites_to_generative_and_extracts_letter(tmp_path):
    tasks = _write_tasks_dir(
        tmp_path, bbh=False, humaneval=False, ifeval=False, mmlu=True
    )
    target = tasks / "mmlu" / "default" / "_default_template_yaml"
    original = target.read_text(encoding="utf-8")
    assert "output_type: multiple_choice" in original  # fixture sanity

    logs = patch_task_templates_for_chat(tasks, ["mmlu"], max_gen_toks=16384)
    text = target.read_text(encoding="utf-8")
    assert "output_type: generate_until" in text
    assert 'until:\n    - "<|im_end|>"' in text
    assert "max_gen_toks: 16384" in text
    assert "do_sample: false" in text
    assert "metric: exact_match" in text
    assert "ignore_case: true" in text
    assert 'regex_pattern: "(?i)\\\\b([a-d])\\\\b"' in text
    assert any("生成式+解析" in line for line in logs)

    # 生成的 YAML 必须可被 lm-eval 解析, 正则需命中自然答案中的字母。
    cfg = yaml.safe_load(text)
    assert cfg["output_type"] == "generate_until"
    assert cfg["generation_kwargs"]["until"] == ["<|im_end|>"]
    rx = cfg["filter_list"][0]["filter"][0]["regex_pattern"]
    assert re.fullmatch(rx, "B")  # 裸字母
    assert re.search(rx, "The answer is C because")  # 自然句里的答案字母
    assert not re.search(rx, "Paris, 1919")  # 无 A-D 字母时不误抽取
    assert "['A', 'B', 'C', 'D'][answer]" in cfg["doc_to_target"]

    # idempotent: second run rewrites nothing and reports it.
    before = text
    logs2 = patch_task_templates_for_chat(tasks, ["mmlu"], max_gen_toks=16384)
    assert target.read_text(encoding="utf-8") == before
    assert any("已 patch (幂等)" in line for line in logs2)


def test_prepare_chat_template_eval_gating(tmp_path):
    venv, site = _make_venv(tmp_path)
    _write_tasks_dir(site, humaneval=False, ifeval=False)
    tasks_dir = site / "lm_eval" / "tasks"

    settings = load_config()
    settings.evaluation.harness_cmd = str(venv / "bin" / "lm_eval")
    settings.evaluation.apply_chat_template = True
    settings.evaluation.max_gen_toks = 16384
    logs = prepare_chat_template_eval(settings, ["bbh", "gsm8k"])
    patched = (tasks_dir / "bbh" / "cot_fewshot" / "_cot_fewshot_template_yaml").read_text(
        encoding="utf-8"
    )
    assert 'until:\n    - "<|im_end|>"' in patched
    assert logs

    # apply_chat_template off -> no patch, no log.
    settings.evaluation.apply_chat_template = False
    assert prepare_chat_template_eval(settings, ["bbh"]) == []

    # unresolvable harness_cmd -> warning, not an exception.
    settings.evaluation.apply_chat_template = True
    settings.evaluation.harness_cmd = str(tmp_path / "missing" / "bin" / "lm_eval")
    warn_logs = prepare_chat_template_eval(settings, ["bbh"])
    assert len(warn_logs) == 1 and "WARNING" in warn_logs[0]


def test_build_predictions_chat_addon_is_valid_python(tmp_path):
    """The appended utils source must parse as a standalone module."""
    tasks = _write_tasks_dir(tmp_path, bbh=False, ifeval=False)
    patch_task_templates_for_chat(tasks, ["humaneval"])
    source = (tasks / "humaneval" / "utils.py").read_text(encoding="utf-8")
    ast.parse(source)


def test_find_lm_eval_tasks_dir_fallback_import():
    # Bare command name resolves through the in-process lm_eval install
    # (tool + harness share one venv in the deployed layout).
    found = find_lm_eval_tasks_dir("lm_eval")
    assert found is not None and found.name == "tasks" and found.is_dir()
