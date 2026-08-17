"""Recommendation engine: turns a capability deficit into executable actions.

This is the tool's recommendation *algorithm*. It is designed to compose three
evidence sources, per the product goal (rules + LLM + human experience):

* **Stage 1 (rules, deterministic)** — score the tool-maintained experience-base
  interventions against the diagnosed capability deficit by
  ``tag overlap x evidence strength x empirical outcome lift`` and emit concrete
  datasets / hyperparameters with a reason chain. Never needs an LLM.
* **Stage 2 (LLM)** — when an analyst LLM is configured, constrain it to select
  and order the Stage-1 candidates (it re-ranks the maintained interventions, it
  never invents datasets / knobs), then ground-check its picks.
* **Stage 3 (human)** — the experience base itself is curated human expert
  knowledge, and ``record_outcome()`` is the human feedback loop that lets
  measured deltas accumulate and re-rank future recommendations.

The empty-deficit path falls back to the legacy tag logic (failure modes ->
benchmark declared tags -> sub-capability) so runs without correlation data or
bad cases behave exactly as before.
"""

from __future__ import annotations

from typing import Any

from benchmark_diagnosis.config import RecommendationConfig
from benchmark_diagnosis.core.types import DiagnosisResult, Recommendation
from benchmark_diagnosis.recommendation.experience_base import (
    ExperienceBase,
    Intervention,
)
from benchmark_diagnosis.recommendation.groundedness_check import check_groundedness
from benchmark_diagnosis.recommendation.rule_base.loader import (
    EVIDENCE_STRENGTH_ORDER,
    Rule,
)
from benchmark_diagnosis.recommendation.synthesizer import synthesize

# Outcome-lift window: 1 + LIFT_GAIN * mean_delta, clamped to [LIFT_MIN, LIFT_MAX].
LIFT_GAIN = 5.0
LIFT_MIN = 0.8
LIFT_MAX = 1.5


def recommend(
    diagnosis: DiagnosisResult,
    experience_base: ExperienceBase,
    config: RecommendationConfig,
    llm: Any | None,
    *,
    rules: list[Rule] | None = None,
    benchmark_tags: dict[str, list[str]] | None = None,
    retrieved: list[dict] | None = None,
) -> list[Recommendation]:
    """Produce grounded, executable recommendations for one cluster diagnosis.

    Args:
        diagnosis: Cluster diagnosis (``capability_deficit`` drives the search;
            empty deficit falls back to legacy tag logic).
        experience_base: The tool-maintained knowledge base (datasets, knobs,
            expected effects, outcomes).
        config: Recommendation configuration (``max_actions``).
        llm: Optional analyst LLM; when present, Stage 2 re-ranks the candidates.
        rules: Optional rule base (used for legacy fallback + linked-rule
            provenance).
        benchmark_tags: Optional benchmark_id -> declared tags (legacy fallback).
        retrieved: Optional external snippets (passed to the LLM synthesizer).

    Returns:
        Ordered :class:`Recommendation` list (datasets / hyperparameters /
        expected effect / reason chain populated). Empty when no intervention
        matches the diagnosis.
    """
    max_actions = max(1, config.max_actions)
    tags, strengths = _diagnosis_tags(diagnosis, benchmark_tags)
    ranked = _score_interventions(experience_base, tags, strengths)
    if not ranked:
        return []

    top = [intervention for intervention, _ in ranked[:max_actions]]
    if llm is None:
        return [_intervention_rec(intervention, diagnosis, config, rules) for intervention in top]

    # Stage 2: let the LLM re-rank the maintained candidates; ground-check it.
    candidates = [intervention for intervention, _ in ranked[: max_actions * 2]]
    try:
        synth = synthesize(
            llm,
            diagnosis,
            rules or [],
            retrieved or [],
            interventions=candidates,
        )
        evidence = _evidence(diagnosis, rules, candidates)
        if not check_groundedness(synth, evidence):
            mapped = _synth_to_recs(synth, candidates, rules or [], diagnosis, config)
            if mapped:
                return mapped[:max_actions]
    except Exception:  # noqa: BLE001 - any LLM failure degrades to Stage 1
        pass
    return [_intervention_rec(intervention, diagnosis, config, rules) for intervention in top]


def _diagnosis_tags(
    diagnosis: DiagnosisResult,
    benchmark_tags: dict[str, list[str]] | None,
) -> tuple[list[str], dict[str, float] | None]:
    """Resolve the search tags, preferring the inferred capability deficit.

    Returns ``(tags, strengths)`` where ``strengths`` is the deficit profile or
    None on the legacy fallback (each overlapping tag then weighs 1.0).
    """
    deficit = diagnosis.capability_deficit or {}
    if deficit:
        ordered = sorted(deficit, key=lambda tag: deficit[tag], reverse=True)
        return ordered, deficit

    tags = list(diagnosis.failure_modes.keys())
    if not tags and diagnosis.sub_capability:
        tags = list((benchmark_tags or {}).get(diagnosis.sub_capability) or [])
    if not tags and diagnosis.sub_capability:
        tags = [diagnosis.sub_capability]
    return tags, None


def _score_interventions(
    base: ExperienceBase,
    tags: list[str],
    strengths: dict[str, float] | None,
) -> list[tuple[Intervention, float]]:
    """Score interventions by tag overlap x evidence x outcome lift (desc)."""
    tag_set = set(tags)
    scored: list[tuple[Intervention, float]] = []
    for intervention in base.interventions:
        overlap = set(intervention.applicable_tags) & tag_set
        if not overlap:
            continue
        tag_score = sum((strengths or {}).get(tag, 1.0) for tag in overlap)
        evidence_w = EVIDENCE_STRENGTH_ORDER.get(intervention.evidence_strength, 1) / 3.0
        mean = intervention.mean_delta
        lift_w = 1.0 if mean is None else min(LIFT_MAX, max(LIFT_MIN, 1.0 + LIFT_GAIN * mean))
        scored.append((intervention, tag_score * evidence_w * lift_w))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def _intervention_rec(
    intervention: Intervention,
    diagnosis: DiagnosisResult,
    config: RecommendationConfig,
    rules: list[Rule] | None,
    *,
    action_text: str | None = None,
) -> Recommendation:
    """Build a Recommendation from one experience-base intervention."""
    rule = _linked_rule(rules or [], intervention.linked_rule_id)
    return Recommendation(
        rule_id=intervention.linked_rule_id or intervention.intervention_id,
        source=f"experience:{intervention.intervention_id}",
        evidence_strength=intervention.evidence_strength,
        action=action_text or intervention.title,
        validation_experiment=_default_validation(intervention.category),
        datasets=_serialize_datasets(intervention),
        hyperparameters=_serialize_hyperparameters(intervention),
        expected_effect=intervention.expected_effect,
        reason_chain=_reason_chain(intervention, diagnosis, rule),
        intervention_id=intervention.intervention_id,
    )


def _synth_to_recs(
    synth: dict[str, Any],
    candidates: list[Intervention],
    rules: list[Rule],
    diagnosis: DiagnosisResult,
    config: RecommendationConfig,
) -> list[Recommendation]:
    """Map grounded LLM actions back to Recommendations (LLM re-ranks only)."""
    by_id = {i.intervention_id: i for i in candidates}
    rule_by_id = {rule.rule_id: rule for rule in rules}
    recs: list[Recommendation] = []
    for action in synth.get("actions", []):
        if not isinstance(action, dict):
            continue
        intervention = by_id.get(action.get("intervention_id"))
        if intervention is not None:
            recs.append(
                _intervention_rec(
                    intervention,
                    diagnosis,
                    config,
                    rules,
                    action_text=action.get("action") or None,
                )
            )
            continue
        rule = rule_by_id.get(action.get("rule_id"))
        if rule is not None:
            recs.append(_rule_rec(rule, diagnosis, config))
    return recs


def _rule_rec(rule: Rule, diagnosis: DiagnosisResult, config: RecommendationConfig) -> Recommendation:
    """Legacy rule-only Recommendation (kept for the external-evidence path)."""
    return Recommendation(
        rule_id=rule.rule_id,
        source=f"rule_base:{rule.rule_id}",
        evidence_strength=rule.evidence_strength,
        action=rule.description,
        validation_experiment=_default_validation(rule.category),
        reason_chain=[f"关联规则 {rule.rule_id} 命中诊断标签"],
    )


def _evidence(
    diagnosis: DiagnosisResult,
    rules: list[Rule] | None,
    candidates: list[Intervention],
) -> dict[str, Any]:
    rules = rules or []
    numbers = set(diagnosis.failure_modes.values())
    if diagnosis.quantified_gap is not None:
        numbers.add(diagnosis.quantified_gap)
    return {
        "rule_ids": {rule.rule_id for rule in rules},
        "sources": {f"rule_base:{rule.rule_id}" for rule in rules}
        | {f"experience:{i.intervention_id}" for i in candidates},
        "numbers": numbers,
        "intervention_ids": {i.intervention_id for i in candidates},
    }


def _linked_rule(rules: list[Rule], rule_id: str | None) -> Rule | None:
    if rule_id is None:
        return None
    for rule in rules:
        if rule.rule_id == rule_id:
            return rule
    return None


def _reason_chain(
    intervention: Intervention,
    diagnosis: DiagnosisResult,
    rule: Rule | None,
) -> list[str]:
    """Human-readable evidence trail: deficit -> intervention -> history."""
    reason: list[str] = []
    if diagnosis.deficit_narrative:
        reason.append(diagnosis.deficit_narrative)
    elif diagnosis.sub_capability:
        reason.append(f"簇 {diagnosis.cluster_id} 低分能力：{diagnosis.sub_capability}")
    reason.append(
        f"匹配干预 {intervention.intervention_id}：{intervention.title}"
    )
    mean = intervention.mean_delta
    if mean is not None:
        reason.append(
            f"历史 {len(intervention.outcomes)} 次实验平均提升 {mean * 100:+.1f}%（分数变化）"
        )
    if rule is not None:
        reason.append(f"关联规则 {rule.rule_id}（{rule.evidence_strength} 证据）")
    return reason


def _serialize_datasets(intervention: Intervention) -> list[dict[str, str]]:
    return [
        {
            "name": dataset.name,
            "rationale": dataset.rationale,
            "expected_effect": dataset.expected_effect,
            "source": dataset.source,
        }
        for dataset in intervention.datasets
    ]


def _serialize_hyperparameters(intervention: Intervention) -> list[dict[str, str]]:
    return [
        {
            "knob": param.knob,
            "direction": param.direction,
            "typical_range": param.typical_range,
            "rationale": param.rationale,
        }
        for param in intervention.hyperparameters
    ]


def _default_validation(category: str) -> str:
    return (
        "在小规模子集上做消融：固定其余训练配置，仅应用该改动，"
        "对比目标 benchmark 的簇级得分变化（建议 >=1% 且跨 2 个随机种子稳定）。"
    )
