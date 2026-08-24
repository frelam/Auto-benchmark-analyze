"""Chat-model template compatibility patches for the pinned lm-eval (0.4.12).

背景 (2026-08-24 实测): Qwen3 系 chat 模型先输出 ``<think>...</think>``
再输出最终答案，而 lm-eval 0.4.12 里多个生成式任务的模板 ``until`` /
``max_gen_toks`` 是按 base 模型（无思考过程）设计的，chat 评测时分数失真：

* ``bbh`` (cot_fewshot，27 个子任务 include 同一份
  ``bbh/cot_fewshot/_cot_fewshot_template_yaml``): until 含 ``"\\n\\n"``，
  恰好在 ``</think>`` 与答案之间截断生成 → get-answer 正则永远抠不到
  ``the answer is`` → 21/27 子任务 0.0、整体 ~1%；模板 ``max_gen_toks: 1024``
  还会压过 model_args 里配置的更大值（task 级 gen_kwargs 优先）。
* ``humaneval``: until 含 ``"\\ndef"`` / ``"\\nif"`` 等，think 块后的换行
  即触发截断，函数体根本没生成；且即使不截断，think 块会进入
  ``build_predictions`` 拼出的代码 → 语法错误直接 0 分；模板
  ``max_gen_toks: 1024`` 同样压过配置值。
* ``ifeval``: ``until: []`` 无碍（harness 会自动补 eos），但模板
  ``max_gen_toks: 1280`` 会截断长思考 + 长输出。

已审计、无需修补的任务：``gsm8k``（until 已含 ``<|im_end|>``，无模板
max_gen_toks）、``aime24``/``aime25``（until 已含 ``<|im_end|>`` +
``<|eot_id|>``，max_gen_toks 32768）、``hendrycks_math`` 7 个子任务
（until ``["Problem:"]`` 对 chat 输出无害，\boxed 正则全文本扫描）、
``mmlu``/``mmlu_pro``/``longbench2``/``arc_challenge``/``hellaswag``/
``mmmlu``（multiple_choice / loglikelihood，不走生成）。

修补只在本工具以 ``apply_chat_template=True`` 评估 chat 模型、且任务列表
命中注册表时触发（见 :func:`prepare_chat_template_eval`），并保持幂等：
已修补的模板再次运行直接跳过。所有函数 best-effort——定位不到 venv 时
返回警告日志，绝不中断评测。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark_diagnosis.config import Settings

# lm-eval 任务/组名 -> lm_eval/tasks/ 下的模板文件（相对路径）。
# 27 个 bbh_cot_fewshot 子任务共享同一份模板，patch 一次全覆盖。
_TASK_TEMPLATES: dict[str, str] = {
    "bbh": "bbh/cot_fewshot/_cot_fewshot_template_yaml",
    "humaneval": "humaneval/humaneval.yaml",
    "ifeval": "ifeval/ifeval.yaml",
}

_HUMANEVAL_UTILS = "humaneval/utils.py"
# 附加到 humaneval/utils.py 的 chat 版 build_predictions 标记（幂等判据）。
_HUMANEVAL_MARKER = "# auto chat-template patch: build_predictions_chat"
# 默认 chat EOS（Qwen3 系；非 Qwen 时可在调用处覆盖）。
_DEFAULT_CHAT_EOS = "<|im_end|>"

# 附加到 humaneval/utils.py 的独立函数源码。生成文本里去掉 <think> 块与
# markdown 围栏后再拼回 prompt，否则 think 块会让 check() 语法错误。
_HUMANEVAL_UTILS_ADDON = f'''
{_HUMANEVAL_MARKER}
def build_predictions_chat(resps, docs):
    """build_predictions 的 chat 版: 去掉 <think>…</think> 与 ``` 围栏。

    chat 模型(如 Qwen3)在最终答案前输出思考块, 甚至把代码包进 markdown
    围栏; 原样拼回 prompt 会导致语法错误。此函数只保留纯净的代码续写。
    """
    import re as _re

    def _clean(r):
        r = _re.sub(r"<think>.*?</think>", "", r, flags=_re.DOTALL)
        r = r.replace("<|im_end|>", "")
        if "```" in r:
            i, j = r.find("```"), r.rfind("```")
            if j > i:
                newline = r.find("\\n", i)
                start = newline + 1 if newline != -1 else i + 3
                r = r[start:j]
        return r.strip(" \\n")

    return [[doc["prompt"] + _clean(r) for r in resp] for resp, doc in zip(resps, docs)]
'''


def find_lm_eval_tasks_dir(harness_cmd: str) -> Path | None:
    """从 harness 可执行文件推导 lm_eval/tasks 目录，定位不到返回 None。

    优先按绝对路径解析 ``<venv>/bin/lm_eval`` -> ``<venv>/lib/python3.*/
    site-packages/lm_eval/tasks``；``harness_cmd`` 为裸命令名（默认
    ``lm_eval``）时退回本进程的 ``import lm_eval``（工具与 harness 共用
    同一 venv 的部署场景）。
    """
    exe = Path(harness_cmd).expanduser()
    if exe.is_absolute():
        if not exe.exists():
            return None  # 配置的路径不存在: 不猜测、不写文件
        venv = exe.parent.parent
        for lib in (venv / "lib", venv / "lib64"):
            candidates = sorted(
                lib.glob("python3*/site-packages/lm_eval/tasks")
            )
            if candidates:
                return candidates[-1]
        return None
    # 裸命令名 (默认 "lm_eval"): 工具与 harness 共用同一 venv 的部署场景。
    try:
        import lm_eval  # noqa: PLC0415 - fallback 只在裸命令名时触发

        tasks = Path(lm_eval.__file__).resolve().parent / "tasks"
        return tasks if tasks.is_dir() else None
    except ImportError:
        return None


def patch_task_templates_for_chat(
    tasks_dir: str | Path,
    task_names: Iterable[str],
    max_gen_toks: int | None = None,
    eos: str = _DEFAULT_CHAT_EOS,
) -> list[str]:
    """幂等地把注册表内任务模板适配为 chat 模型，返回人类可读日志行。

    Args:
        tasks_dir: ``lm_eval/tasks`` 目录（:func:`find_lm_eval_tasks_dir`）。
        task_names: 本次评估的 lm-eval 任务名；不在注册表内的忽略。
        max_gen_toks: 把模板 ``max_gen_toks`` 抬到该值（None 则保留模板值，
            仅修补 until）。
        eos: 替换后的唯一停止符（默认 Qwen3 chat 的 ``<|im_end|>``）。

    Returns:
        每份被检查文件的日志行（含幂等跳过）；不会抛异常。
    """
    root = Path(tasks_dir)
    logs: list[str] = []
    for task in task_names:
        rel = _TASK_TEMPLATES.get(task)
        if rel is None:
            continue
        target = root / rel
        if not target.is_file():
            logs.append(
                f"[chat-patch] WARNING: 未找到 lm_eval/tasks/{rel}, 跳过"
            )
            continue
        logs.extend(_patch_yaml(target, max_gen_toks, eos, rel))
        if task == "humaneval":
            logs.extend(_patch_humaneval_utils(root / _HUMANEVAL_UTILS))
    return logs


def prepare_chat_template_eval(
    settings: Settings, lm_eval_tasks: list[str]
) -> list[str]:
    """按 settings 定位 venv 并对本次任务列表执行 chat 模板修补（best-effort）。

    返回日志行由调用方打印；定位不到 lm_eval 目录时返回一条警告而不是
    抛异常（模板未修补只会分数失真，不应中断评测）。
    """
    if not settings.evaluation.apply_chat_template:
        return []
    tasks_dir = find_lm_eval_tasks_dir(settings.evaluation.harness_cmd)
    if tasks_dir is None:
        return [
            "[chat-patch] WARNING: 无法定位 lm_eval/tasks (harness_cmd="
            f"{settings.evaluation.harness_cmd!r}); bbh/humaneval 等 chat "
            "模板适配被跳过, 相关分数可能失真"
        ]
    return patch_task_templates_for_chat(
        tasks_dir, lm_eval_tasks, settings.evaluation.max_gen_toks
    )


def _patch_yaml(
    path: Path, max_gen_toks: int | None, eos: str, rel: str
) -> list[str]:
    """把一份任务模板的 until 换成 [eos]、max_gen_toks 抬到配置值。幂等。"""
    text = path.read_text(encoding="utf-8")
    changes: list[str] = []
    modified = False

    # until: 整块替换为 [eos]（模板的 \n\n / "Q" 等对 chat 输出是致命停止符）。
    # 注意 0.4.12 模板里 max_gen_toks 位于 until 之前，不能锚定
    # "generation_kwargs:\n  until:" 的紧邻顺序；直接锚定 "  until:" 行本身。
    if f'  until:\n    - "{eos}"' not in text:
        until_m = re.search(r"^  until:\n(?:    - .*\n)+", text, re.MULTILINE)
        if until_m:
            text = text.replace(
                until_m.group(0),
                f'  until:\n    - "{eos}"\n',
            )
            changes.append(f"until -> [{eos}]")
            modified = True

    # max_gen_toks: task 级模板值压过 model_args，长思考会被截断。
    if max_gen_toks is not None:
        want = f"max_gen_toks: {max_gen_toks}"
        gen_m = re.search(r"max_gen_toks: \d+", text)
        if gen_m:
            if gen_m.group(0) != want:
                text = text.replace(gen_m.group(0), want)
                changes.append(f"max_gen_toks -> {max_gen_toks}")
                modified = True
        else:
            changes.append(
                f"模板无 max_gen_toks (回落 model_args={max_gen_toks})"
            )

    # humaneval 专属: 把 filter 指向 chat 版 build_predictions。
    if path.name == "humaneval.yaml" and "utils.build_predictions_chat" not in text:
        text = text.replace(
            "filter_fn: !function utils.build_predictions",
            "filter_fn: !function utils.build_predictions_chat",
        )
        changes.append("filter -> build_predictions_chat")
        modified = True

    if modified:
        path.write_text(text, encoding="utf-8")
        return [f"[chat-patch] {rel}: {', '.join(changes)}"]
    return [f"[chat-patch] {rel}: 已 patch (幂等)"]


def _patch_humaneval_utils(utils_path: Path) -> list[str]:
    """给 humaneval/utils.py 追加 chat 版 build_predictions_chat。幂等。"""
    if not utils_path.is_file():
        return ["[chat-patch] WARNING: 未找到 humaneval/utils.py, 跳过"]
    text = utils_path.read_text(encoding="utf-8")
    if _HUMANEVAL_MARKER in text:
        return ["[chat-patch] humaneval/utils.py: 已 patch (幂等)"]
    utils_path.write_text(text.rstrip("\n") + _HUMANEVAL_UTILS_ADDON, encoding="utf-8")
    return ["[chat-patch] humaneval/utils.py: 追加 build_predictions_chat"]
