"""LLM-agent-base diagnosis engine (design section 2.2).

Runs on top of the rule-base result: the harness agent adopts the rule-base
conclusions (2.1), performs bad-case analysis, and then iterates an
*analysis -> verification -> new hypothesis* loop — verifying hypotheses by
executing the eval tool (``eval-task``) on datasets or by re-analyzing bad
cases — until it reaches a final conclusion.

Integration contract (harness-agnostic, file-based):

* The tool prepares a **case pack** directory (``<run output>/agent_run/``):

  .. code-block:: text

      agent_run/
      ├── input/            # scores.json, rule_base_result.json, bad_cases/*.jsonl,
      │                     # context.md, evaluation_target.json
      ├── output/           # the harness writes conclusion.json / conclusion.md here
      └── skill/            # copy of the benchmark-diagnosis skill (SKILL.md workflow)

* ``diagnosis.llm_agent.harness_cmd`` launches the harness agent with the case
  pack: the ``{case_pack}`` placeholder is replaced with the pack path, and the
  path is always exported as ``BMD_CASE_PACK`` (so wrappers need no placeholder).
* The harness agent runs the skill workflow and writes
  ``<pack>/output/conclusion.json`` (schema below). The tool polls for it while
  the harness runs and stops early on ``status: "final"``.
* If the harness exits / finishes without a final conclusion and
  ``diagnosis.llm_agent.interact_cmd`` is configured, the tool writes a
  follow-up message file (``<pack>/input/followup_<round>.json`` with the draft
  conclusion + "keep iterating" instruction) and invokes ``interact_cmd``
  (``{case_pack}`` / ``{message}`` placeholders, always exported as
  ``BMD_CASE_PACK`` / ``BMD_MESSAGE``). Up to ``max_rounds`` rounds total.
* A per-round timeout (``timeout_seconds``) kills a stuck harness.

Conclusion schema (written by the harness, read by the tool):

.. code-block:: json

    {
      "status": "final" | "draft",
      "round": 2,
      "summary": "...",
      "conclusions": [{"capability_id": "...", "confidence": "high|medium|low",
                       "evidence": "...", "verified_by": ["eval:math", "bad_case:..."]}],
      "suggestions": [{"capability_id": "...", "action": "...", "datasets": [],
                       "expected_gain": 0.1}],
      "bad_case_analysis": {"n_cases": 12, "root_causes": {...}}
    }
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_diagnosis.config import Settings, resolve_diagnosis_engine
from benchmark_diagnosis.core.schema import ModelRecord

DEFAULT_EVAL_CMD = "benchmark-diagnosis eval-task"
_POLL_INTERVAL = 2.0

_PACKAGED_SKILL = (
    Path(__file__).resolve().parent / "skill" / "benchmark-diagnosis"
)


@dataclass
class AgentRunResult:
    """Outcome of the harness loop (JSON-serializable via asdict)."""

    engine: str = "llm_agent"
    case_pack: str = ""
    rounds: int = 0
    max_rounds: int = 0
    timed_out: bool = False
    concluded: bool = False
    conclusion: dict[str, Any] | None = None
    harness_cmd: str = ""
    interact_cmd: str = ""


# ------------------------------------------------------------------ case pack


def prepare_case_pack(
    output_dir: str | Path,
    *,
    model: ModelRecord,
    raw_scores: dict[str, float],
    rule_result: dict[str, Any],
    bad_cases_dir: str | Path | None,
    base_url: str | None,
    config: Settings,
) -> Path:
    """Assemble the case pack the harness agent works on; returns its path."""
    pack = Path(output_dir) / "agent_run"
    inp = pack / "input"
    out = pack / "output"
    shutil.rmtree(pack, ignore_errors=True)
    inp.mkdir(parents=True)
    out.mkdir(parents=True)

    (inp / "scores.json").write_text(
        json.dumps(raw_scores, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (inp / "rule_base_result.json").write_text(
        json.dumps(rule_result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    bad_in = inp / "bad_cases"
    if bad_cases_dir and Path(bad_cases_dir).exists():
        bad_in.mkdir(parents=True)
        for src in sorted(Path(bad_cases_dir).glob("*.jsonl")):
            shutil.copy2(src, bad_in / src.name)

    target = {
        "available": bool(base_url),
        "base_url": base_url,
        "model": model.model_id,
        "model_name": model.name,
        "arch_type": model.arch_type,
        "total_params": model.total_params,
        "active_params": model.active_params,
    }
    (inp / "evaluation_target.json").write_text(
        json.dumps(target, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    eval_cmd = config.diagnosis.llm_agent.eval_cmd or DEFAULT_EVAL_CMD
    (inp / "context.md").write_text(
        _render_context(model, base_url, eval_cmd, pack),
        encoding="utf-8",
    )

    skill_src = _resolve_skill_dir(config)
    if skill_src is not None:
        shutil.copytree(skill_src, pack / "skill", dirs_exist_ok=True)
    return pack


def _resolve_skill_dir(config: Settings) -> Path | None:
    """Locate the benchmark-diagnosis skill directory (SKILL.md workflow)."""
    candidate = config.diagnosis.llm_agent.skill_path
    if candidate:
        p = Path(candidate)
        return p if (p / "SKILL.md").exists() else None
    if _PACKAGED_SKILL.exists() and (_PACKAGED_SKILL / "SKILL.md").exists():
        return _PACKAGED_SKILL
    return None


def _render_context(
    model: ModelRecord, base_url: str | None, eval_cmd: str, pack: Path
) -> str:
    release = model.release_date.isoformat() if model.release_date else "unknown"
    endpoint = base_url or "（无评测端点：本次为分数文件路径，只能做 bad-case 验证）"
    return f"""# Benchmark 诊断上下文（case pack: {pack}）

## 被诊断模型
- model_id: {model.model_id}（{model.name}）
- arch: {model.arch_type} / total_params: {model.total_params} / active_params: {model.active_params}
- release_date: {release}

## 评测端点（数据集验证用）
- base_url: {endpoint}
- 可用: {bool(base_url)}
- eval 工具命令: `{eval_cmd} --task <benchmark_id> --limit <N> --base-url <url> --model <model> --out <dir>`
  （跑一个数据集的子集并写出 <dir>/scores.json + <dir>/bad_cases/，用于验证假设；端点不可用时跳过数据集验证，只用 bad case 验证）

## 输入文件
- `input/rule_base_result.json` — rule-base 诊断结论（2.1 结果，必须采纳为起点）
- `input/scores.json` — 各 benchmark 分数
- `input/bad_cases/<benchmark>.jsonl` — 错题（question/gold/model_output/metrics）
- `input/evaluation_target.json` — 端点与模型元信息

## 工作流（详见 skill/SKILL.md）
1. 采纳 rule-base 结论 → 2. bad case 分析 → 3. 假设-验证循环（数据集评测或 bad case
   重分析验证）→ 4. 把最终结论写入 `output/conclusion.json`（status 必须为 "final"，
   schema 见 SKILL.md）。
"""


# ------------------------------------------------------------------ execution


def _expand(template: str, case_pack: Path, message: Path | None = None) -> list[str]:
    text = template.replace("{case_pack}", str(case_pack))
    if message is not None:
        text = text.replace("{message}", str(message))
    return shlex.split(text)


def _launch(cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
    )


def _stream(proc: subprocess.Popen) -> None:
    """Print the harness's stdout live until the process ends."""
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    except Exception:  # noqa: BLE001 - streaming must never crash the runner
        pass


def _read_conclusion(pack: Path) -> dict[str, Any] | None:
    path = pack / "output" / "conclusion.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_followup(pack: Path, round_no: int, draft: dict[str, Any]) -> Path:
    message = {
        "round": round_no,
        "instruction": (
            "上一轮结论尚未确认（status 不是 final）。请基于 rule-base 结论、"
            "bad case 和上一轮验证结果继续迭代：提出新假设 → 用 eval 工具或 "
            "bad case 分析验证 → 更新结论，直到你确信为止，然后把最终结论写回 "
            "output/conclusion.json（status: \"final\"）。"
        ),
        "previous_draft": draft,
    }
    path = pack / "input" / f"followup_{round_no}.json"
    path.write_text(json.dumps(message, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_llm_agent_diagnosis(
    *,
    session: Any,
    model: ModelRecord,
    raw_scores: dict[str, float],
    config: Settings,
    rule_result: dict[str, Any],
    output_dir: str | Path,
    base_url: str | None,
    bad_cases_dir: str | Path | None,
    popen: Callable[[list[str], dict[str, str]], subprocess.Popen] | None = None,
) -> AgentRunResult:
    """Run the harness loop and collect the final conclusion (2.2).

    Args:
        session: DB session (kept for interface symmetry; assets already read).
        model: The evaluated model.
        raw_scores: ``benchmark_id -> score``.
        config: Settings (``diagnosis.llm_agent.*``).
        rule_result: The rule-base block (2.1 result) the agent must adopt.
        output_dir: Run output directory (case pack lives under it).
        base_url: Served model endpoint for dataset verification (None when
            the run used a scores file — verification degrades to bad cases).
        bad_cases_dir: The archived ``bad_cases/`` directory to copy in.
        popen: Injectable process launcher (tests). Signature ``(cmd, env)``.

    Returns:
        The agent run outcome (rounds, concluded flag, conclusion).

    Raises:
        ValueError: when the llm-agent path is not fully configured
            (delegates to :func:`resolve_diagnosis_engine`).
    """
    resolve_diagnosis_engine(config, "llm_agent")  # fail fast on bad config
    agent_cfg = config.diagnosis.llm_agent
    assert agent_cfg.harness_cmd and agent_cfg.interact_cmd  # enforced above

    popen = popen or _launch
    pack = prepare_case_pack(
        output_dir,
        model=model,
        raw_scores=raw_scores,
        rule_result=rule_result,
        bad_cases_dir=bad_cases_dir,
        base_url=base_url,
        config=config,
    )
    result = AgentRunResult(
        case_pack=str(pack),
        max_rounds=agent_cfg.max_rounds,
        harness_cmd=agent_cfg.harness_cmd,
        interact_cmd=agent_cfg.interact_cmd,
    )

    for round_no in range(1, agent_cfg.max_rounds + 1):
        result.rounds = round_no
        env = dict(os.environ)
        env["BMD_CASE_PACK"] = str(pack)

        if round_no == 1:
            cmd = _expand(agent_cfg.harness_cmd, pack)
            print(f"[llm-agent] round {round_no}: launching harness: {' '.join(cmd)}")
            proc = popen(cmd, env)
            streamer = threading.Thread(target=_stream, args=(proc,), daemon=True)
            streamer.start()
            deadline = time.monotonic() + agent_cfg.timeout_seconds
            timed_out = False
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    timed_out = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
                conclusion = _read_conclusion(pack)
                if conclusion and conclusion.get("status") == "final":
                    proc.terminate()
                    break
                time.sleep(_POLL_INTERVAL)
            proc.wait()
            result.timed_out = timed_out
        else:
            conclusion = _read_conclusion(pack)
            if not conclusion or conclusion.get("status") != "final":
                message = _write_followup(pack, round_no, conclusion or {})
                env["BMD_MESSAGE"] = str(message)
                cmd = _expand(agent_cfg.interact_cmd, pack, message)
                print(
                    f"[llm-agent] round {round_no}: interacting: {' '.join(cmd)}"
                )
                proc = popen(cmd, env)
                try:
                    proc.wait(timeout=agent_cfg.timeout_seconds)
                except subprocess.TimeoutExpired:
                    print(
                        f"[llm-agent] interact command timed out after "
                        f"{agent_cfg.timeout_seconds}s; terminating."
                    )
                    result.timed_out = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                # Give the harness a short grace period to write its updated
                # conclusion after the interaction command returns.
                deadline = time.monotonic() + agent_cfg.timeout_seconds
                while time.monotonic() < deadline:
                    conclusion = _read_conclusion(pack)
                    if conclusion and conclusion.get("status") == "final":
                        break
                    time.sleep(_POLL_INTERVAL)

        conclusion = _read_conclusion(pack)
        if conclusion and conclusion.get("status") == "final":
            result.concluded = True
            result.conclusion = conclusion
            break
        if round_no == agent_cfg.max_rounds:
            result.conclusion = conclusion or None

    print(
        f"[llm-agent] done: rounds={result.rounds}, "
        f"concluded={result.concluded}, timed_out={result.timed_out}"
    )
    return result
