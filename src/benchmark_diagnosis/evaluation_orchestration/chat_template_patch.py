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
  ``max_gen_toks: 1280`` 会截断长思考 + 长输出；且校验器把整条生成串当正文
  判断（词数/段落/起始/JSON 等），思考块会污染分数，故额外在 ``utils.py``
  的 ``process_results`` 里剥掉  thinking...response 块再送校验。

已审计、无需修补的任务：``gsm8k``（until 已含 ``<|im_end|>``，无模板
max_gen_toks）、``aime24``/``aime25``（until 已含 ``<|im_end|>`` +
``<|eot_id|>``，max_gen_toks 32768）、``hendrycks_math`` 7 个子任务
（until ``["Problem:"]`` 对 chat 输出无害，\boxed 正则全文本扫描）。
``mmlu`` 虽属 ``multiple_choice``/loglikelihood，但 chat 模型下在 assistant
标记后接裸选项字母算续写概率、分布被思考/填充占满，系统性失真，故改写为
 ``generate_until`` 生成自然答案加正则抽字母判分（见 ``_patch_mmlu_default``）。
``mmlu_pro``/``longbench2``/``arc_challenge``/``hellaswag``/``mmmlu`` 为
同类 latent 问题，超出本次修补范围。

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
    "mmlu": "mmlu/default/_default_template_yaml",
}

_HUMANEVAL_UTILS = "humaneval/utils.py"
# 附加到 humaneval/utils.py 的 chat 版 build_predictions 标记（幂等判据）。
_HUMANEVAL_MARKER = "# auto chat-template patch: build_predictions_chat"
# ifeval 任务模板同目录下的 utils.py（process_results 所在）。
_IFEVAL_UTILS = "ifeval/utils.py"
_IFEVAL_MARKER = "# auto chat-template patch: ifeval strip think block"
# MMLU 生成式改写：lm-eval 默认模板 doing multiple_choice/loglikelihood，
# chat 模型下在 assistant 标记后接裸选项字母算续写概率，首 token 分布被思考/
# 填充占满，系统性失真。改写为 generate_until 让模型生成自然答案，再用正则
# 抽出 A-D 字母经 exact_match 判分。57 个子任务都 include 这份共享模板。
_MMLU_DEFAULT_TPL = "mmlu/default/_default_template_yaml"
_MMLU_MARKER = "# auto chat-template patch: mmlu generative"
# config 未设 max_gen_toks 时 MMLU 生成回落的默认值（长思考+答案）。
_DEFAULT_MC_GEN_TOKS = 16384
# __MAX_TOKS__ / __EOS__ 用替换填充，jinja 花括号保持字面量。
_MMLU_GEN_YAML = f'''{_MMLU_MARKER}
dataset_path: cais/mmlu
test_split: test
fewshot_split: dev
fewshot_config:
  sampler: first_n
output_type: generate_until
doc_to_text: "{{{{question.strip()}}}}\\nA. {{{{choices[0]}}}}\\nB. {{{{choices[1]}}}}\\nC. {{{{choices[2]}}}}\\nD. {{{{choices[3]}}}}\\nAnswer:"
doc_to_target: "{{{{ ['A', 'B', 'C', 'D'][answer] }}}}"
target_delimiter: ""
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
generation_kwargs:
  max_gen_toks: __MAX_TOKS__
  until:
    - "__EOS__"
  do_sample: false
  temperature: 0.0
filter_list:
  - name: "mmlu-answer"
    filter:
      - function: "regex"
        regex_pattern: "(?i)\\\\b([a-d])\\\\b"
      - function: "take_first"
metadata:
  version: 1.0
'''

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

# 附加到 ifeval/utils.py 的独立函数源码 + process_results 改造：IFEval 的
# 校验器把整条生成串当正文去检查指令（词数/段落/起始/JSON 等），chat 模型
# 的思考块会污染这些判定，故在送校验前先剥掉  thinking...response 与 EOS。
_IFEVAL_UTILS_ADDON = f'''{_IFEVAL_MARKER}
import re as _ifeval_re


def _strip_chat_think(text):
    """去除 chat 思考块 ( thinking...response ) 与末尾 EOS 标记。

    保留正文(含 markdown 与指令要求的原文)，仅移除前置推理与 token 结尾，
    避免 IFEval 的词数/段落/起始/JSON 等指令误判思考块。
    """
    text = _ifeval_re.sub(r" thinking.*? response", "", text, flags=_ifeval_re.DOTALL)
    text = text.replace("<|im_end|>", "").replace("</s>", "")
    return text.strip()
'''

_PROCESS_RESULTS_ASSIGN = "    response = results[0]\n"


def _patch_ifeval_utils(utils_path: Path) -> list[str]:
    """给 ifeval/utils.py 注入 _strip_chat_think 并在 process_results 调用。幂等。"""
    if not utils_path.is_file():
        return ["[chat-patch] WARNING: 未找到 ifeval/utils.py, 跳过"]
    text = utils_path.read_text(encoding="utf-8")
    if _IFEVAL_MARKER in text:
        return ["[chat-patch] ifeval/utils.py: 已 patch (幂等)"]
    if _PROCESS_RESULTS_ASSIGN not in text:
        return ["[chat-patch] WARNING: ifeval/utils.py 结构不符, 跳过"]
    text = text.replace(
        _PROCESS_RESULTS_ASSIGN,
        "    response = _strip_chat_think(results[0])\n",
    )
    # addon 定义放在文件顶部(紧跟在 import 之后的模块级)，确保可用。
    addon = "\n" + _IFEVAL_UTILS_ADDON
    utils_path.write_text(text + addon, encoding="utf-8")
    return ["[chat-patch] ifeval/utils.py: 注入 _strip_chat_think"]


def _patch_mmlu_default(
    path: Path, eos: str, max_gen_toks: int | None
) -> list[str]:
    """把 mmlu 默认模板改写为生成式+解析字母判分。幂等。

    原模板是 ``output_type: multiple_choice`` 的对数似然打分：context 以
    chat assistant 标记结尾、continuation 为裸选项字母，chat 模型在该位置的
    首 token 分布被思考/填充占满，续写概率不反映真实选择。改写为
    ``generate_until`` 让模型生成自然答案，再用正则抽出 A-D 字母经
    ``exact_match`` 判分。57 个子任务都 include 这份共享模板，改一次全覆盖。
    """
    if not path.is_file():
        return [
            "[chat-patch] WARNING: 未找到 mmlu/default/_default_template_yaml, "
            "跳过"
        ]
    text = path.read_text(encoding="utf-8")
    if _MMLU_MARKER in text:
        return ["[chat-patch] mmlu/default/_default_template_yaml: 已 patch (幂等)"]
    toks = max_gen_toks or _DEFAULT_MC_GEN_TOKS
    gen = _MMLU_GEN_YAML.replace("__MAX_TOKS__", str(toks)).replace("__EOS__", eos)
    path.write_text(gen, encoding="utf-8")
    return [
        "[chat-patch] mmlu/default/_default_template_yaml: 改写为生成式+解析"
        f" (max_gen_toks={toks}, until={[eos]}, filter=mmlu-answer)"
    ]


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
        if task == "mmlu":
            # 生成式改写（native loglikelihood 对 chat 模型系统性失真）。
            logs.extend(_patch_mmlu_default(target, eos, max_gen_toks))
            continue
        logs.extend(_patch_yaml(target, max_gen_toks, eos, rel))
        if task == "humaneval":
            logs.extend(_patch_humaneval_utils(root / _HUMANEVAL_UTILS))
        elif task == "ifeval":
            logs.extend(_patch_ifeval_utils(root / _IFEVAL_UTILS))
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
