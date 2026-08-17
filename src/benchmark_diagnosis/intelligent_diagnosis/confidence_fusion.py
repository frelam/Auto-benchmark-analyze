"""Stage 4: diagnosis confidence fusion + suggestion-type rules (design v2 §6).

Fuses Stage 1-3 signals into ``confidence`` and ``suggestion_type``. Two
mechanisms:

* **evidence score** — ``w_probe * probe_state + w_content * content_ratio +
  w_agreement * tag_agreement``, emitted for transparency;
* **rule tables** — the v1 decision table, hardened with (a) a hard cap when a
  key evidence channel is missing, (b) conflict detection (contradictory
  signals -> Low + human review), and (c) a precedence-ordered suggestion-type
  ladder: grading > format > probe-missing > compositional > reweighting >
  pass@k decision > synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_diagnosis.intelligent_diagnosis.types import (
    CONFIDENCE_NUMERIC,
    CaseAnalysis,
    ConfidenceLevel,
    FusedVerdict,
    ProbeResult,
    ProbeState,
    SuggestionType,
)


@dataclass
class FusionConfig:
    """Stage 4 thresholds (defaults mirror config.diagnosis)."""

    content_high_threshold: float = 0.6
    agreement_high_threshold: float = 0.6
    dominant_ratio: float = 0.5  # format/grading dominant routing
    low_signal_floor: float = 0.3  # below this, content evidence is "absent"
    agreement_low_threshold: float = 0.3  # LLM disagrees with the hypothesis
    data_share_low_threshold: float = 0.1  # share of capability data in the mix
    passk_gap_threshold: float = 0.5
    pass1_high_threshold: float = 0.5
    w_probe: float = 0.4
    w_content: float = 0.35
    w_agreement: float = 0.25


def _probe_score(state: ProbeState) -> float:
    return {
        ProbeState.CONFIRMED: 1.0,
        ProbeState.NOT_CONFIRMED: 0.5,
        ProbeState.NO_PROBE: 0.0,
        ProbeState.PENDING_EVAL: 0.0,
    }[state]


def _content_high(case: CaseAnalysis | None, config: FusionConfig) -> bool:
    return case is not None and case.content_error_ratio > config.content_high_threshold


def _agreement_high(case: CaseAnalysis | None, config: FusionConfig) -> bool:
    return case is not None and case.tag_agreement > config.agreement_high_threshold


def _no_case_evidence(case: CaseAnalysis | None) -> bool:
    return case is None or case.cases_analyzed == 0


def _conflict(probe: ProbeResult, case: CaseAnalysis | None, config: FusionConfig) -> bool:
    """Contradictory evidence: probe speaks but cases say nothing conclusive."""
    if _no_case_evidence(case):
        return False
    if probe.state not in (ProbeState.CONFIRMED, ProbeState.NOT_CONFIRMED):
        return False
    all_low = (
        case.content_error_ratio <= config.low_signal_floor
        and case.format_error_ratio <= config.low_signal_floor
        and case.grading_artifact_ratio <= config.low_signal_floor
        and case.label_noise_ratio <= config.low_signal_floor
    )
    return all_low


def _pick_suggestion_type(
    probe: ProbeResult,
    case: CaseAnalysis | None,
    config: FusionConfig,
    *,
    data_share: float | None,
) -> SuggestionType:
    """Precedence-ordered suggestion-type ladder (design doc v2 section 6)."""
    if case is not None:
        if case.grading_artifact_ratio > config.dominant_ratio:
            return SuggestionType.EVAL_INFRA_FIX
        if case.format_error_ratio > config.dominant_ratio:
            return SuggestionType.NON_TRAINING_FIX

    if probe.state in (ProbeState.NO_PROBE, ProbeState.PENDING_EVAL):
        return SuggestionType.BUILD_PROBE_FIRST

    if probe.state == ProbeState.NOT_CONFIRMED:
        return (
            SuggestionType.COMPOSITIONAL_CURRICULUM
            if _content_high(case, config)
            else SuggestionType.HUMAN_REVIEW
        )

    # CONFIRMED below.
    if (
        data_share is not None
        and data_share < config.data_share_low_threshold
        and _content_high(case, config)
    ):
        return SuggestionType.DATA_REWEIGHTING

    passk_gap = (
        probe.passk_gap_ratio is not None
        and probe.passk_gap_ratio > config.passk_gap_threshold
        and probe.pass_1 is not None
        and probe.pass_1 < config.pass1_high_threshold
    )
    if passk_gap:
        return SuggestionType.REJECTION_SAMPLING
    return SuggestionType.TARGETED_SYNTHESIS


def fuse(
    probe: ProbeResult,
    case: CaseAnalysis | None,
    *,
    data_share: float | None = None,
    config: FusionConfig | None = None,
) -> FusedVerdict:
    """Fuse one candidate's Stage 2/3 evidence into a verdict.

    Args:
        probe: Stage 2 probe result for the capability.
        case: Stage 3 guided case analysis (None when no LLM / no bad cases).
        data_share: Optional fraction of this capability's data in the current
            SFT/RL mix (enables ``data_reweighting``); None disables the rule.
        config: Thresholds.

    Returns:
        A :class:`FusedVerdict`.
    """
    config = config or FusionConfig()
    probe_score = _probe_score(probe.state)
    content_score = case.content_error_ratio if case is not None else 0.0
    agreement_score = case.tag_agreement if case is not None else 0.0
    evidence_score = (
        config.w_probe * probe_score
        + config.w_content * content_score
        + config.w_agreement * agreement_score
    )

    # --- confidence ---------------------------------------------------------
    confidence: ConfidenceLevel
    rationale: list[str] = []
    needs_human_review = False

    if _conflict(probe, case, config):
        confidence = ConfidenceLevel.LOW
        needs_human_review = True
        rationale.append("probe 与 bad-case 证据冲突（内容/格式/评测 均无主导信号）")
    elif probe.state == ProbeState.CONFIRMED and _content_high(case, config) and _agreement_high(case, config):
        confidence = ConfidenceLevel.HIGH
        rationale.append("probe 复测确认 + 内容错误主导 + LLM 标签一致")
    elif probe.state == ProbeState.NOT_CONFIRMED and _content_high(case, config) and _agreement_high(case, config):
        confidence = ConfidenceLevel.MEDIUM_HIGH
        rationale.append("probe 未复测出低分但源头 benchmark 持续低分 + 内容错误主导 → 组合能力缺陷")
    elif _no_case_evidence(case):
        confidence = ConfidenceLevel.MEDIUM
        rationale.append("缺少 bad-case 分析证据，置信度封顶 Medium")
        if probe.state in (ProbeState.NO_PROBE, ProbeState.PENDING_EVAL):
            rationale.append("且 probe 证据缺失，需先建设/评测 probe 集")
    elif not _content_high(case, config):
        confidence = ConfidenceLevel.MEDIUM
        rationale.append("内容错误占比不高，按主导失败模式路由（非训练问题候选）")
        if case is not None and case.tag_agreement < config.agreement_low_threshold:
            needs_human_review = True
            rationale.append("LLM 标签一致率低，诊断可能跑偏")
    else:
        confidence = ConfidenceLevel.MEDIUM
        rationale.append("证据部分缺失或组合未达 High 门槛")
        if case is not None and case.tag_agreement < config.agreement_low_threshold:
            confidence = ConfidenceLevel.LOW
            needs_human_review = True
            rationale.append("LLM 标签一致率过低，转人工复核")

    suggestion_type = _pick_suggestion_type(probe, case, config, data_share=data_share)
    if confidence == ConfidenceLevel.LOW:
        # Conflicting / untrustworthy evidence: never auto-generate an action.
        suggestion_type = SuggestionType.HUMAN_REVIEW
        needs_human_review = True

    return FusedVerdict(
        capability_id=probe.capability_id,
        confidence=confidence,
        suggestion_type=suggestion_type,
        evidence_score=round(evidence_score, 4),
        needs_human_review=needs_human_review,
        rationale="；".join(rationale) or "规则判定",
        probe_state=probe.state,
        data_share=data_share,
    )


def confidence_numeric(level: ConfidenceLevel) -> float:
    """Map a confidence tier to its priority-scoring weight (Low = 0)."""
    return CONFIDENCE_NUMERIC[level]
