"""Stage 6: concrete suggestion write-up (design doc v2 section 8).

Stage 4/5 already decided *what type* of fix and *how urgent*. This stage only
*instantiates parameters* and writes the evidence trail:

* **deterministic mode** (no LLM): per-suggestion-type templates filled with the
  measured evidence (probe percentile, pass@1/pass@k, bad-case ratios);
* **LLM mode**: the analyst LLM is constrained to the fixed schema and the
  evidence set — every number it cites must exist in the evidence (numeric
  groundedness), otherwise the deterministic version is used.

Output is grouped: training suggestions (priority-ranked) vs
non-training / build-probe / human-review, never mixed.
"""

from __future__ import annotations

import re
from typing import Any

from benchmark_diagnosis.intelligent_diagnosis.types import (
    NON_TRAINING_TYPES,
    CaseAnalysis,
    FusedVerdict,
    PriorityItem,
    ProbeResult,
    Suggestion,
    SuggestionType,
)

# ---------------------------------------------------------------- templates

_TEMPLATES: dict[SuggestionType, str] = {
    SuggestionType.REJECTION_SAMPLING: (
        "以该能力上表现最好的 checkpoint 为 teacher，对 {capability} 相关题型做 "
        "K={k} 采样，规则 + RM 联合过滤，预计新增 N 条蒸馏数据加入下一轮 SFT"
    ),
    SuggestionType.TARGETED_SYNTHESIS: (
        "该能力基本缺失：构造 / 收集针对 {capability} 的高质量训练数据（或引入 "
        "更强 teacher 蒸馏），并人工抽检质量后加入训练"
    ),
    SuggestionType.DATA_REWEIGHTING: (
        "调整 {capability} 相关数据在 SFT/RL 配比中的采样权重（当前占比 "
        "{share}，低于阈值），先做小步长消融再全量"
    ),
    SuggestionType.COMPOSITIONAL_CURRICULUM: (
        "单项能力达标但组合场景掉链子：构造多技能组合的训练数据 / 环境，"
        "按课程顺序训练（{capability} 与其他能力的组合）"
    ),
    SuggestionType.NON_TRAINING_FIX: (
        "输出格式 / 解析问题：检查 prompt 模板、输出解析器、reward shaping，"
        "无需训练改动（{capability}）"
    ),
    SuggestionType.EVAL_INFRA_FIX: (
        "评测脚本 / 抽取逻辑疑似误判：标记相关 benchmark / 题目复查，"
        "路由给评测基建（{capability}）"
    ),
    SuggestionType.BUILD_PROBE_FIRST: (
        "该能力缺少 probe 或 probe 未评测：先建设 / 运行窄口径 probe 集，"
        "确认能力缺口后再决定训练动作（{capability}）"
    ),
    SuggestionType.HUMAN_REVIEW: (
        "证据冲突或一致率过低：转人工复核，不自动进入建议生成（{capability}）"
    ),
}


def _evidence_lines(
    probe: ProbeResult | None,
    case: CaseAnalysis | None,
) -> list[str]:
    lines: list[str] = []
    if probe is not None:
        if probe.probe_benchmark_id and probe.percentile is not None:
            lines.append(
                f"probe {probe.probe_benchmark_id}: percentile={probe.percentile:.1f}"
            )
        if probe.pass_1 is not None and probe.pass_k is not None:
            lines.append(f"pass@1={probe.pass_1:.2f}, pass@k={probe.pass_k:.2f}")
    if case is not None and case.cases_analyzed:
        lines.append(
            f"bad-case 分析（{case.cases_analyzed} 条）：content={case.content_error_ratio:.0%}, "
            f"format={case.format_error_ratio:.0%}, grading={case.grading_artifact_ratio:.0%}, "
            f"标签一致率={case.tag_agreement:.0%}"
        )
    return lines


def write_suggestions_deterministic(
    items: list[PriorityItem],
    *,
    probe_results: dict[str, ProbeResult] | None = None,
    case_analyses: dict[str, CaseAnalysis] | None = None,
    fused: dict[str, FusedVerdict] | None = None,
    passk_k: int = 16,
) -> list[Suggestion]:
    """Fill the templates with measured evidence (no LLM needed)."""
    probe_results = probe_results or {}
    case_analyses = case_analyses or {}
    fused = fused or {}
    suggestions: list[Suggestion] = []
    for item in items:
        probe = probe_results.get(item.capability_id)
        case = case_analyses.get(item.capability_id)
        verdict = fused.get(item.capability_id)
        share = (
            f"{verdict.data_share:.0%}"
            if verdict is not None and verdict.data_share is not None
            else "占比未知"
        )
        template = _TEMPLATES[item.suggestion_type]
        action = template.format(capability=item.capability_id, k=passk_k, share=share)
        suggestions.append(
            Suggestion(
                capability_id=item.capability_id,
                suggestion_type=item.suggestion_type,
                confidence=item.confidence,
                priority_score=item.priority,
                concrete_action=action,
                supporting_evidence=_evidence_lines(probe, case),
                expected_gain=item.expected_gain,
                source="deterministic",
            )
        )
    return suggestions


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_PASSK_RE = re.compile(r"pass@\d+", re.IGNORECASE)


def _cites_unknown_numbers(text: str, allowed: set[float]) -> bool:
    """True when ``text`` cites a number that is not in the evidence set.

    ``pass@N`` indices are configuration markers, not citations, so they are
    normalized away before number extraction.
    """
    cleaned = _PASSK_RE.sub("pass@", text)
    for token in _NUMBER_RE.findall(cleaned):
        value = float(token)
        if any(abs(value - allowed_value) < 1e-6 for allowed_value in allowed):
            continue
        return True
    return False


def _llm_suggestions(
    llm: Any,
    items: list[PriorityItem],
    *,
    probe_results: dict[str, ProbeResult],
    case_analyses: dict[str, CaseAnalysis],
    fused: dict[str, FusedVerdict],
    passk_k: int = 16,
) -> list[Suggestion]:
    """LLM instantiation of concrete actions; groundedness-checked."""
    suggestions: list[Suggestion] = []
    for item in items:
        probe = probe_results.get(item.capability_id)
        case = case_analyses.get(item.capability_id)
        evidence = _evidence_lines(probe, case)
        allowed_numbers = {
            item.expected_gain,
            item.priority,
            item.cost,
            item.gain_lower,
            float(passk_k),  # the pass@k sampling budget is evidence context
        }
        for line in evidence:
            allowed_numbers.update(float(t) for t in _NUMBER_RE.findall(line))

        prompt = _build_llm_prompt(item, evidence, fused.get(item.capability_id))
        try:
            data = llm.complete_json(prompt)
        except Exception:  # noqa: BLE001 - LLM failure falls back to templates
            data = None
        if not isinstance(data, dict):
            continue
        action = data.get("concrete_action")
        if not isinstance(action, str) or not action.strip():
            continue
        supporting = data.get("supporting_evidence")
        if not isinstance(supporting, list):
            supporting = []
        joined = action + " " + " ".join(str(s) for s in supporting)
        if _cites_unknown_numbers(joined, allowed_numbers):
            continue  # ungrounded -> fall back to the deterministic version
        suggestions.append(
            Suggestion(
                capability_id=item.capability_id,
                suggestion_type=item.suggestion_type,
                confidence=item.confidence,
                priority_score=item.priority,
                concrete_action=action,
                supporting_evidence=[str(s) for s in supporting] or evidence,
                expected_gain=item.expected_gain,
                source="llm",
                raw=data,
            )
        )
    return suggestions


def _build_llm_prompt(
    item: PriorityItem,
    evidence: list[str],
    verdict: FusedVerdict | None,
) -> list[dict[str, str]]:
    evidence_block = "\n".join(f"- {line}" for line in evidence) or "- （无）"
    verdict_block = (
        f"confidence={verdict.confidence.value}, suggestion_type={verdict.suggestion_type.value}"
        if verdict is not None
        else "confidence=?, suggestion_type=?"
    )
    system = (
        "You instantiate a training/engineering action from a fixed diagnosis. "
        "Do NOT invent datasets, numbers, or checkpoints that are not in the "
        "evidence. Reply with JSON: {\"concrete_action\": str, "
        "\"supporting_evidence\": [str], \"expected_gain\": number}."
    )
    user = (
        f"CAPABILITY: {item.capability_id}\n"
        f"VERDICT: {verdict_block}\n"
        f"PRIORITY: {item.priority}, EXPECTED_GAIN: {item.expected_gain}, "
        f"COST: {item.cost}\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        "Write the concrete_action (2-4 sentences, executable parameters only), "
        "supporting_evidence (cite only evidence above), expected_gain (use the "
        "given value)."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def write_suggestions(
    items: list[PriorityItem],
    *,
    llm: Any | None = None,
    probe_results: dict[str, ProbeResult] | None = None,
    case_analyses: dict[str, CaseAnalysis] | None = None,
    fused: dict[str, FusedVerdict] | None = None,
    passk_k: int = 16,
) -> dict[str, list[Suggestion]]:
    """Produce grouped suggestions (training / non_training / build_probe_first / human_review).

    LLM-mode suggestions are used when grounded; every item without a grounded
    LLM write-up gets its deterministic template version.
    """
    deterministic = write_suggestions_deterministic(
        items,
        probe_results=probe_results,
        case_analyses=case_analyses,
        fused=fused,
        passk_k=passk_k,
    )
    by_id = {s.capability_id: s for s in deterministic}

    if llm is not None:
        llm_suggestions = _llm_suggestions(
            llm,
            items,
            probe_results=probe_results or {},
            case_analyses=case_analyses or {},
            fused=fused or {},
            passk_k=passk_k,
        )
        for s in llm_suggestions:
            by_id[s.capability_id] = s

    grouped: dict[str, list[Suggestion]] = {
        "training": [],
        "non_training": [],
        "build_probe_first": [],
        "human_review": [],
    }
    for item in items:
        suggestion = by_id.get(item.capability_id)
        if suggestion is None:
            continue
        if item.suggestion_type == SuggestionType.HUMAN_REVIEW:
            grouped["human_review"].append(suggestion)
        elif item.suggestion_type in NON_TRAINING_TYPES:
            if item.suggestion_type == SuggestionType.BUILD_PROBE_FIRST:
                grouped["build_probe_first"].append(suggestion)
            else:
                grouped["non_training"].append(suggestion)
        else:
            grouped["training"].append(suggestion)
    return grouped
