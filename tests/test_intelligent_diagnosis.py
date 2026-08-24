"""Tests for the intelligent diagnosis pipeline (design doc v2, stages 1-7)."""

from __future__ import annotations

import pandas as pd
import pytest

from benchmark_diagnosis.config import load_config
from benchmark_diagnosis.core import db
from benchmark_diagnosis.data import ingestion, queries
from benchmark_diagnosis.intelligent_diagnosis.candidate_generation import (
    CandidateConfig,
    generate_candidates,
    wilson_lower_bound,
)
from benchmark_diagnosis.intelligent_diagnosis.capability_taxonomy import (
    CapabilityTaxonomy,
    TaxonomyValidationError,
    load_taxonomy,
)
from benchmark_diagnosis.intelligent_diagnosis.confidence_fusion import (
    FusionConfig,
    fuse,
)
from benchmark_diagnosis.intelligent_diagnosis.feedback import (
    list_executions,
    load_calibration,
    log_execution,
    recalibrate,
)
from benchmark_diagnosis.intelligent_diagnosis.guided_case_analyzer import (
    GuidedCaseAnalyzer,
)
from benchmark_diagnosis.intelligent_diagnosis.priority_scorer import (
    Calibration,
    score_priorities,
)
from benchmark_diagnosis.intelligent_diagnosis.probe_registry import (
    ProbeConfig,
    ProbeRegistry,
    estimate_pass_at_k,
    split_todos,
    verify_candidates,
)
from benchmark_diagnosis.intelligent_diagnosis.suggestion_writer import (
    write_suggestions,
)
from benchmark_diagnosis.intelligent_diagnosis.types import (
    CaseAnalysis,
    ConfidenceLevel,
    PassKStats,
    ProbeResult,
    ProbeState,
    SuggestionType,
)
from benchmark_diagnosis.pipeline import build_offline, diagnose_model


@pytest.fixture(scope="module")
def taxonomy() -> CapabilityTaxonomy:
    return load_taxonomy()


@pytest.fixture()
def registry(taxonomy) -> ProbeRegistry:
    return ProbeRegistry.load(taxonomy=taxonomy)


class FakeLLM:
    """Duck-typed analyst LLM: ``complete_json(messages)`` with injectable response."""

    def __init__(self, response=None) -> None:
        self._response = response
        self.calls = 0

    def complete_json(self, messages):
        self.calls += 1
        if callable(self._response):
            return self._response(messages)
        return self._response


class ExplodingLLM:
    def complete_json(self, messages):
        raise RuntimeError("analyst endpoint unavailable")


def _cases(count: int) -> list[dict]:
    return [
        {"question": f"q{i}", "model_output": f"out{i}", "gold": f"gold{i}"}
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# capability taxonomy
# ---------------------------------------------------------------------------


def test_taxonomy_loads_and_resolves_aliases(taxonomy) -> None:
    assert taxonomy.get("math") is None  # legacy flat id is an alias, not a node
    assert taxonomy.get("reasoning.math") is not None
    assert "math" in taxonomy.get("reasoning.math").aliases
    # depth is derived from the parent chain
    assert taxonomy.get("reasoning.math.multi_step").level == 3


def test_taxonomy_validation_errors() -> None:
    with pytest.raises(TaxonomyValidationError):
        CapabilityTaxonomy.from_dict(
            {
                "version": "1",
                "capabilities": [
                    {"id": "a", "name": "A"},
                    {"id": "a", "name": "A2"},
                ],
            }
        )
    with pytest.raises(TaxonomyValidationError):
        CapabilityTaxonomy.from_dict(
            {
                "version": "1",
                "capabilities": [
                    {"id": "a", "name": "A", "parent": "ghost"},
                ],
            }
        )
    with pytest.raises(TaxonomyValidationError):
        CapabilityTaxonomy.from_dict(
            {
                "version": "1",
                "capabilities": [
                    {"id": "a", "name": "A", "parent": "b"},
                    {"id": "b", "name": "B", "parent": "a"},
                ],
            }
        )


def test_taxonomy_within_and_near_miss(taxonomy) -> None:
    assert taxonomy.is_within("reasoning.math", "reasoning.math.calculation")
    assert taxonomy.is_within("reasoning.math", "reasoning.math")
    assert not taxonomy.is_within("reasoning.math", "reasoning.logical")
    assert taxonomy.near_miss("reasoning.math", "reasoning.logical")
    assert taxonomy.near_miss("reasoning.math.calculation", "reasoning.math")
    assert not taxonomy.near_miss("reasoning.math", "reasoning.math.calculation")


# ---------------------------------------------------------------------------
# Stage 1: candidate generation
# ---------------------------------------------------------------------------


def test_wilson_lower_bound_is_conservative() -> None:
    # 10/10 successes: 95% Wilson lower bound is ~0.72, not 1.0 (sampling honesty)
    assert wilson_lower_bound(10, 10) == pytest.approx(0.7225, abs=0.01)
    lb = wilson_lower_bound(1, 10)
    assert 0.0 < lb < 0.05  # 1/10 successes -> very low conservative bound
    assert wilson_lower_bound(0, 0) == 0.0


def test_candidate_generation_fine_mode(taxonomy) -> None:
    # target gets 1/10 on the tagged items; peers get ~9/10 -> below expectation
    items = pd.DataFrame(
        {
            "model_id": ["target"] * 10 + ["peer_a"] * 10 + ["peer_b"] * 10,
            "benchmark_id": ["gsm8k"] * 30,
            "item_id": [f"it{i % 10}" for i in range(10)] * 3,
            "correct": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            + [1] * 10
            + [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    tags = {f"it{i}": ["reasoning.math.calculation"] for i in range(10)}
    candidates = generate_candidates(
        [{"benchmark_id": "gsm8k", "score": 10.0, "residual": -0.3, "underperforming": True}],
        taxonomy=taxonomy,
        target_model_id="target",
        item_scores=items,
        item_capabilities=tags,
        coverage=[{"benchmark_id": "gsm8k", "design_goal_tags": ["math", "reasoning"]}],
    )
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.capability_id == "reasoning.math.calculation"
    assert cand.evidence_mode == "fine"
    assert cand.sources == ["gsm8k"]
    assert 0.0 < cand.screening_score <= 1.0
    # peer mean ~0.9 vs target 0.1 -> large gap, percentile well below threshold
    assert cand.sub_accuracies[0].peer_percentile < 25.0
    assert cand.sub_accuracies[0].wilson_lb < cand.sub_accuracies[0].sub_accuracy


def test_candidate_generation_or_union_across_benchmarks(taxonomy) -> None:
    rows = []
    for model, corrects in [
        ("target", [0] * 10),
        ("peer_a", [1] * 10),
        ("peer_b", [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
    ]:
        for i, c in enumerate(corrects):
            rows.append(
                {"model_id": model, "benchmark_id": "gsm8k", "item_id": f"g{i}", "correct": c}
            )
    for model, corrects in [("target", [0] * 10), ("peer_a", [1] * 10), ("peer_b", [1] * 10)]:
        for i, c in enumerate(corrects):
            rows.append(
                {"model_id": model, "benchmark_id": "math", "item_id": f"m{i}", "correct": c}
            )
    items = pd.DataFrame(rows)
    tags = {f"g{i}": ["reasoning.math.calculation"] for i in range(10)} | {
        f"m{i}": ["reasoning.math.calculation"] for i in range(10)
    }
    verdicts = [
        {"benchmark_id": "gsm8k", "score": 10.0, "residual": -0.2, "underperforming": True},
        {"benchmark_id": "math", "score": 10.0, "residual": -0.4, "underperforming": True},
        {"benchmark_id": "mmlu", "score": 50.0, "residual": 0.1, "underperforming": False},
    ]
    candidates = generate_candidates(
        verdicts,
        taxonomy=taxonomy,
        target_model_id="target",
        item_scores=items,
        item_capabilities=tags,
        coverage=[{"benchmark_id": "mmlu", "design_goal_tags": ["world_knowledge"]}],
    )
    assert len(candidates) == 1
    cand = candidates[0]
    assert set(cand.sources) == {"gsm8k", "math"}  # OR union
    # screening score is the max across the two source benchmarks
    alone = []
    for verdict in verdicts[:2]:
        alone.extend(
            generate_candidates(
                [verdict],
                taxonomy=taxonomy,
                target_model_id="target",
                item_scores=items,
                item_capabilities=tags,
                coverage=[],
            )
        )
    assert cand.screening_score == pytest.approx(max(c.screening_score for c in alone))


def test_candidate_generation_coarse_mode_with_alias_resolution(taxonomy) -> None:
    candidates = generate_candidates(
        [{"benchmark_id": "gsm8k", "score": 20.0, "residual": -0.4, "underperforming": True}],
        taxonomy=taxonomy,
        target_model_id="target",
        item_scores=None,
        item_capabilities=None,
        coverage=[
            {"benchmark_id": "gsm8k", "design_goal_tags": ["math", "reasoning"]}
        ],
    )
    ids = {c.capability_id for c in candidates}
    assert "reasoning.math" in ids  # "math" alias resolved
    assert "reasoning" in ids
    assert all(c.evidence_mode == "coarse" for c in candidates)
    # coarse penalty caps the screening score
    assert all(c.screening_score <= 0.5 + 1e-9 for c in candidates)


def test_candidate_generation_skips_in_range_benchmarks(taxonomy) -> None:
    candidates = generate_candidates(
        [
            {"benchmark_id": "gsm8k", "score": 90.0, "residual": 0.2, "underperforming": False},
        ],
        taxonomy=taxonomy,
        target_model_id="target",
        item_scores=None,
        item_capabilities=None,
        coverage=[{"benchmark_id": "gsm8k", "design_goal_tags": ["math"]}],
    )
    assert candidates == []


def test_candidate_generation_low_support_penalty(taxonomy) -> None:
    items = pd.DataFrame(
        {
            "model_id": ["target"] * 3 + ["peer_a"] * 3 + ["peer_b"] * 3,
            "benchmark_id": ["gsm8k"] * 9,
            "item_id": [f"it{i % 3}" for i in range(3)] * 3,
            "correct": [0] * 3 + [1] * 3 + [1] * 3,
        }
    )
    tags = {f"it{i % 3}": ["reasoning.math.calculation"] for i in range(3)}
    candidates = generate_candidates(
        [{"benchmark_id": "gsm8k", "score": 10.0, "residual": -0.3, "underperforming": True}],
        taxonomy=taxonomy,
        target_model_id="target",
        item_scores=items,
        item_capabilities=tags,
        config=CandidateConfig(min_items_per_capability=8, min_peers=5),
    )
    assert len(candidates) == 1
    assert candidates[0].low_support is True
    # penalty halves the severity (score = (1 - percentile/100) * 0.5 <= 0.5)
    assert candidates[0].screening_score <= 0.5 + 1e-9


# ---------------------------------------------------------------------------
# Stage 2: probe verification
# ---------------------------------------------------------------------------


def test_estimate_pass_at_k_known_values() -> None:
    assert estimate_pass_at_k(0, 10, 8) == 0.0
    assert estimate_pass_at_k(10, 10, 8) == 1.0
    assert estimate_pass_at_k(4, 10, 8) == pytest.approx(1.0)  # C(6,8)=0
    assert 0.0 < estimate_pass_at_k(3, 10, 5) < 1.0


def _candidate(capability_id: str, score: float = 0.8) -> list:
    from benchmark_diagnosis.intelligent_diagnosis.types import CandidateCapability

    return [
        CandidateCapability(
            capability_id=capability_id,
            screening_score=score,
            evidence_mode="coarse",
            sources=["source_bench"],
        )
    ]


def test_probe_verify_confirmed_not_confirmed_no_probe(registry) -> None:
    config = ProbeConfig(percentile_threshold=25.0, min_peers=2)
    # aime24 is a probe for reasoning.math.multi_step
    cands = _candidate("reasoning.math.multi_step")
    confirmed = verify_candidates(
        cands,
        registry=registry,
        probe_scores={"aime24": 0.05},
        peer_scores={"aime24": [0.5, 0.6, 0.7, 0.8]},
        config=config,
    )[0]
    assert confirmed.state == ProbeState.CONFIRMED
    assert confirmed.percentile < 25.0

    not_confirmed = verify_candidates(
        cands,
        registry=registry,
        probe_scores={"aime24": 0.75},
        peer_scores={"aime24": [0.5, 0.6, 0.7, 0.8]},
        config=config,
    )[0]
    assert not_confirmed.state == ProbeState.NOT_CONFIRMED

    no_probe = verify_candidates(
        _candidate("safety"),
        registry=registry,
        probe_scores={},
        peer_scores={},
        config=config,
    )[0]
    assert no_probe.state == ProbeState.NO_PROBE


def test_probe_verify_pending_eval_and_ancestor_fallback(registry) -> None:
    config = ProbeConfig(min_peers=2)
    # no probe registered, but the ancestor instruction_following has ifeval
    cands = _candidate("instruction_following.multi_constraint")
    pending = verify_candidates(
        cands,
        registry=registry,
        probe_scores={},
        peer_scores={"ifeval": [0.4, 0.5]},
        config=config,
    )[0]
    assert pending.state == ProbeState.PENDING_EVAL  # registered but not evaluated

    confirmed_via_ancestor = verify_candidates(
        cands,
        registry=registry,
        probe_scores={"ifeval": 0.1},
        peer_scores={"ifeval": [0.4, 0.5, 0.6, 0.7]},
        config=config,
    )[0]
    assert confirmed_via_ancestor.state == ProbeState.CONFIRMED
    assert confirmed_via_ancestor.note  # mentions the via-ancestor resolution


def test_probe_verify_passk_gap_ratio(registry) -> None:
    config = ProbeConfig(min_peers=2, min_passk_samples=8)
    result = verify_candidates(
        _candidate("reasoning.math.calculation"),
        registry=registry,
        probe_scores={"gsm8k": 0.2},
        peer_scores={"gsm8k": [0.5, 0.6, 0.7, 0.8]},
        passk_stats={
            "reasoning.math.calculation": PassKStats(
                capability_id="reasoning.math.calculation",
                pass_1=0.2,
                pass_k=0.7,
                k=16,
                samples=10,
            )
        },
        config=config,
    )[0]
    assert result.pass_1 == pytest.approx(0.2)
    assert result.passk_gap_ratio == pytest.approx((0.7 - 0.2) / 0.2)


def test_probe_verify_min_peers_guard(registry) -> None:
    config = ProbeConfig(min_peers=5)
    result = verify_candidates(
        _candidate("reasoning.math.calculation"),
        registry=registry,
        probe_scores={"gsm8k": 0.1},
        peer_scores={"gsm8k": [0.5, 0.6]},  # only 2 peers
        config=config,
    )[0]
    assert result.state == ProbeState.PENDING_EVAL  # cannot judge reliably


def test_probe_split_todos() -> None:
    build, pending = split_todos(
        [
            ProbeResult(capability_id="a", state=ProbeState.NO_PROBE),
            ProbeResult(capability_id="b", state=ProbeState.PENDING_EVAL),
            ProbeResult(capability_id="c", state=ProbeState.CONFIRMED),
        ]
    )
    assert build == ["a"]
    assert pending == ["b"]


# ---------------------------------------------------------------------------
# Stage 3: guided case analysis
# ---------------------------------------------------------------------------


def _case_response(cause="content_error", tag="reasoning.math.calculation", n=10):
    def response(_messages):
        return {
            "classifications": [
                {
                    "case_index": i,
                    "root_cause_type": cause,
                    "capability_tag": tag,
                    "evidence": "ev",
                }
                for i in range(n)
            ]
        }

    return response


def test_guided_analysis_aggregation(taxonomy) -> None:
    llm = FakeLLM(_case_response(cause="content_error", tag="reasoning.math.calculation"))
    result = GuidedCaseAnalyzer(llm, taxonomy).analyze(
        "reasoning.math.calculation", _cases(10)
    )
    assert result.cases_analyzed == 10
    assert result.content_error_ratio == 1.0
    assert result.tag_agreement == 1.0
    assert result.saturated is False


def test_guided_analysis_ratio_mix_and_relabel(taxonomy) -> None:
    def response(_messages):
        return {
            "classifications": [
                {"case_index": i, "root_cause_type": "format_error",
                 "capability_tag": "reasoning.math.calculation", "evidence": "e"}
                if i % 2 == 0
                else {"case_index": i, "root_cause_type": "content_error",
                      "capability_tag": "reasoning.math.multi_step", "evidence": "e"}
                for i in range(10)
            ]
        }

    result = GuidedCaseAnalyzer(FakeLLM(response), taxonomy).analyze(
        "reasoning.math.calculation", _cases(10)
    )
    assert result.format_error_ratio == pytest.approx(0.5)
    assert result.content_error_ratio == pytest.approx(0.5)
    # half the cases relabeled to a *sibling* capability (multi_step shares the
    # reasoning.math parent): not agreement, but a near miss
    assert result.tag_agreement == pytest.approx(0.5)
    assert result.near_miss_ratio == pytest.approx(0.5)


def test_guided_analysis_saturation_stop(taxonomy) -> None:
    def response(_messages):
        return {
            "classifications": [
                {"case_index": i, "root_cause_type": "content_error",
                 "capability_tag": "reasoning.math.calculation", "evidence": "e"}
                for i in range(10)
            ]
        }

    llm = FakeLLM(response)
    result = GuidedCaseAnalyzer(llm, taxonomy).analyze(
        "reasoning.math.calculation", _cases(100), saturation_window=9
    )
    # one batch establishes the (cause, tag) pair, 9 following identical cases
    # cross the window -> stop at 10 instead of sampling all 100
    assert result.cases_analyzed == 10
    assert result.saturated is True
    assert llm.calls == 1


def test_guided_analysis_robust_to_llm_failure(taxonomy) -> None:
    result = GuidedCaseAnalyzer(ExplodingLLM(), taxonomy).analyze(
        "reasoning.math.calculation", _cases(10)
    )
    assert result.cases_analyzed == 0
    assert result.content_error_ratio == 0.0


def test_guided_analysis_invalid_entries_skipped(taxonomy) -> None:
    llm = FakeLLM(
        {
            "classifications": [
                {"case_index": 0, "root_cause_type": "made_up", "capability_tag": "x", "evidence": ""},
                {"case_index": 1, "root_cause_type": "content_error", "capability_tag": "not_a_capability", "evidence": ""},
                {"case_index": 2, "root_cause_type": "content_error", "capability_tag": "math", "evidence": "e"},
            ]
        }
    )
    result = GuidedCaseAnalyzer(llm, taxonomy).analyze(
        "reasoning.math.calculation", _cases(3)
    )
    # only case 2 parsed; "math" alias resolves to reasoning.math, the *parent*
    # of the hypothesis -> a near miss (coarser granularity), not agreement
    assert result.cases_analyzed == 1
    assert result.tag_agreement == 0.0
    assert result.near_miss_ratio == 1.0


# ---------------------------------------------------------------------------
# Stage 4: confidence fusion
# ---------------------------------------------------------------------------


def _probe(state: ProbeState, pass1=None, passk=None, gap=None) -> ProbeResult:
    return ProbeResult(
        capability_id="c",
        state=state,
        percentile=10.0 if state == ProbeState.CONFIRMED else 60.0,
        pass_1=pass1,
        pass_k=passk,
        passk_gap_ratio=gap,
    )


def _case(content=0.8, fmt=0.0, grading=0.0, agree=0.8) -> CaseAnalysis:
    return CaseAnalysis(
        cases_analyzed=10,
        content_error_ratio=content,
        format_error_ratio=fmt,
        grading_artifact_ratio=grading,
        tag_agreement=agree,
    )


def test_fusion_high_confidence_confirmed(taxonomy) -> None:
    verdict = fuse(_probe(ProbeState.CONFIRMED), _case())
    assert verdict.confidence == ConfidenceLevel.HIGH
    assert verdict.suggestion_type == SuggestionType.TARGETED_SYNTHESIS


def test_fusion_passk_gap_selects_rejection_sampling() -> None:
    verdict = fuse(
        _probe(ProbeState.CONFIRMED, pass1=0.2, passk=0.7, gap=2.5),
        _case(),
    )
    assert verdict.suggestion_type == SuggestionType.REJECTION_SAMPLING


def test_fusion_not_confirmed_is_compositional() -> None:
    verdict = fuse(_probe(ProbeState.NOT_CONFIRMED), _case())
    assert verdict.confidence == ConfidenceLevel.MEDIUM_HIGH
    assert verdict.suggestion_type == SuggestionType.COMPOSITIONAL_CURRICULUM


def test_fusion_no_probe_caps_confidence() -> None:
    verdict = fuse(_probe(ProbeState.NO_PROBE), _case())
    assert verdict.confidence == ConfidenceLevel.MEDIUM
    assert verdict.suggestion_type == SuggestionType.BUILD_PROBE_FIRST


def test_fusion_routes_format_and_grading() -> None:
    fmt = fuse(_probe(ProbeState.CONFIRMED), _case(content=0.2, fmt=0.7))
    assert fmt.suggestion_type == SuggestionType.NON_TRAINING_FIX
    grading = fuse(_probe(ProbeState.CONFIRMED), _case(content=0.2, grading=0.8))
    assert grading.suggestion_type == SuggestionType.EVAL_INFRA_FIX


def test_fusion_data_reweighting_rule() -> None:
    verdict = fuse(
        _probe(ProbeState.CONFIRMED),
        _case(),
        data_share=0.02,
        config=FusionConfig(data_share_low_threshold=0.1),
    )
    assert verdict.suggestion_type == SuggestionType.DATA_REWEIGHTING


def test_fusion_conflict_goes_to_human_review() -> None:
    verdict = fuse(_probe(ProbeState.CONFIRMED), _case(content=0.1, fmt=0.1, grading=0.1))
    assert verdict.confidence == ConfidenceLevel.LOW
    assert verdict.needs_human_review is True
    assert verdict.suggestion_type == SuggestionType.HUMAN_REVIEW


def test_fusion_missing_case_evidence_caps() -> None:
    verdict = fuse(_probe(ProbeState.CONFIRMED), None)
    assert verdict.confidence == ConfidenceLevel.MEDIUM


def test_fusion_low_agreement_downgrades() -> None:
    verdict = fuse(_probe(ProbeState.CONFIRMED), _case(agree=0.1))
    assert verdict.confidence == ConfidenceLevel.LOW
    assert verdict.needs_human_review is True


# ---------------------------------------------------------------------------
# Stage 5: priority scoring
# ---------------------------------------------------------------------------

from benchmark_diagnosis.intelligent_diagnosis.types import (  # noqa: E402
    CandidateCapability,
    FusedVerdict,
    SubAccuracy,
)


def _cand_with_subacc(peer_mean=0.9, sub_acc=0.3, share_items=4, total_items=10) -> CandidateCapability:
    return CandidateCapability(
        capability_id="reasoning.math.calculation",
        screening_score=0.6,
        evidence_mode="fine",
        sources=["gsm8k"],
        sub_accuracies=[
            SubAccuracy(
                benchmark_id="gsm8k",
                capability_id="reasoning.math.calculation",
                sub_accuracy=sub_acc,
                peer_mean=peer_mean,
                peer_percentile=10.0,
                n_items=share_items,
                wilson_lb=wilson_lower_bound(sub_acc * share_items, share_items),
            )
        ],
    )


def _fused(confidence=ConfidenceLevel.HIGH, stype=SuggestionType.TARGETED_SYNTHESIS) -> FusedVerdict:
    return FusedVerdict(
        capability_id="reasoning.math.calculation",
        confidence=confidence,
        suggestion_type=stype,
        evidence_score=0.8,
    )


def test_priority_formula_and_ordering(taxonomy) -> None:
    cand = _cand_with_subacc()
    base_costs = {
        SuggestionType.TARGETED_SYNTHESIS: 4.0,
        SuggestionType.REJECTION_SAMPLING: 1.0,
    }
    verdicts = {"reasoning.math.calculation": _fused()}
    items = score_priorities(
        [cand],
        verdicts,
                        portfolios=[{"cluster_id": "c1", "benchmarks": [{"benchmark_id": "gsm8k", "weight": 2.0}]}],
        base_costs=base_costs,
        item_totals={"gsm8k": 10},
    )
    assert len(items) == 1
    item = items[0]
    # gap = (0.9 - 0.3) * (4/10) = 0.24; gain = 2.0 * 0.24 = 0.48
    assert item.expected_gain == pytest.approx(0.48, abs=1e-6)
    assert item.gain_lower < item.expected_gain
    # priority = 1.0 * 0.48 / 4.0
    assert item.priority == pytest.approx(0.48 / 4.0, abs=1e-6)
    assert item.sources == ["gsm8k"]


def test_priority_low_confidence_excluded(taxonomy) -> None:
    verdicts = {
        "reasoning.math.calculation": _fused(
            confidence=ConfidenceLevel.LOW, stype=SuggestionType.HUMAN_REVIEW
        )
    }
    items = score_priorities(
        [_cand_with_subacc()],
        verdicts,
                        base_costs={SuggestionType.HUMAN_REVIEW: 0.5},
    )
    assert items == []


def test_priority_calibration_scales_gain_and_cost(taxonomy) -> None:
    cand = _cand_with_subacc()
    base_costs = {SuggestionType.TARGETED_SYNTHESIS: 4.0}
    cal = Calibration(
        costs={"targeted_synthesis": 2.0},  # real cost was 2x the prediction
        gain_scale=0.8,  # real gains were 80% of predictions
        n_logs=10,
    )
    items = score_priorities(
        [cand],
        {"reasoning.math.calculation": _fused()},
                        portfolios=[{"cluster_id": "c1", "benchmarks": [{"benchmark_id": "gsm8k", "weight": 2.0}]}],
        base_costs=base_costs,
        calibration=cal,
        item_totals={"gsm8k": 10},
    )
    item = items[0]
    assert item.cost == pytest.approx(8.0, abs=1e-6)  # 4.0 * 2.0
    assert item.expected_gain == pytest.approx(0.48 * 0.8, abs=1e-6)


# ---------------------------------------------------------------------------
# Stage 6: suggestion writer
# ---------------------------------------------------------------------------


def test_suggestion_deterministic_templates_and_grouping(taxonomy) -> None:
    cand = _cand_with_subacc()
    fused = {
        "reasoning.math.calculation": _fused(
            stype=SuggestionType.REJECTION_SAMPLING
        )
    }
    items = score_priorities(
        [cand],
        fused,
                        portfolios=[{"cluster_id": "c1", "benchmarks": [{"benchmark_id": "gsm8k", "weight": 1.0}]}],
        base_costs={SuggestionType.REJECTION_SAMPLING: 1.0},
        item_totals={"gsm8k": 10},
    )
    probe = _probe(ProbeState.CONFIRMED, pass1=0.2, passk=0.7, gap=2.5)
    grouped = write_suggestions(
        items,
        probe_results={"reasoning.math.calculation": probe},
        case_analyses={"reasoning.math.calculation": _case()},
        fused=fused,
    )
    assert len(grouped["training"]) == 1
    suggestion = grouped["training"][0]
    assert suggestion.capability_id == "reasoning.math.calculation"
    assert "采样" in suggestion.concrete_action
    assert any("pass@1=0.20" in line for line in suggestion.supporting_evidence)


def test_suggestion_grouping_routes_non_training(taxonomy) -> None:
    from benchmark_diagnosis.intelligent_diagnosis.types import CandidateCapability

    cand = CandidateCapability(
        capability_id="instruction_following.format",
        screening_score=0.6,
        evidence_mode="coarse",
        sources=["ifeval"],
        source_scores={"ifeval": 0.6},
    )
    fused = {
        "instruction_following.format": FusedVerdict(
            capability_id="instruction_following.format",
            confidence=ConfidenceLevel.MEDIUM,
            suggestion_type=SuggestionType.NON_TRAINING_FIX,
            evidence_score=0.5,
        )
    }
    items = score_priorities(
        [cand],
        fused,
                        base_costs={SuggestionType.NON_TRAINING_FIX: 0.5},
    )
    grouped = write_suggestions(items, fused=fused)
    assert grouped["training"] == []
    assert len(grouped["non_training"]) == 1


def test_suggestion_llm_mode_groundedness(taxonomy) -> None:
    cand = _cand_with_subacc()
    fused = {"reasoning.math.calculation": _fused()}
    items = score_priorities(
        [cand],
        fused,
                        base_costs={SuggestionType.TARGETED_SYNTHESIS: 4.0},
        item_totals={"gsm8k": 10},
    )
    probe = _probe(ProbeState.CONFIRMED, pass1=0.2, passk=0.7, gap=2.5)

    # grounded response -> llm source
    grounded = FakeLLM(
        {
            "concrete_action": "用当前 checkpoint 对该题型做 K=16 采样",
            "supporting_evidence": ["pass@1=0.20, pass@8=0.70"],
            "expected_gain": 0.48,
        }
    )
    grouped = write_suggestions(
        items,
        llm=grounded,
        probe_results={"reasoning.math.calculation": probe},
        case_analyses={"reasoning.math.calculation": _case()},
        fused=fused,
    )
    assert grouped["training"][0].source == "llm"

    # ungrounded response (invents a number) -> deterministic fallback
    ungrounded = FakeLLM(
        {
            "concrete_action": "引入 500 万条合成数据（该数字不在证据中）",
            "supporting_evidence": [],
            "expected_gain": 99.9,
        }
    )
    grouped = write_suggestions(
        items,
        llm=ungrounded,
        probe_results={"reasoning.math.calculation": probe},
        case_analyses={"reasoning.math.calculation": _case()},
        fused=fused,
    )
    assert grouped["training"][0].source == "deterministic"


# ---------------------------------------------------------------------------
# Stage 7: feedback loop
# ---------------------------------------------------------------------------


def _feedback_session(tmp_path):
    engine = db.make_engine(tmp_path / "feedback.db")
    db.init_db(engine)
    factory = db.session_factory(engine)
    return factory()


def test_feedback_log_and_list(tmp_path) -> None:
    session = _feedback_session(tmp_path)
    row_id = log_execution(
        session,
        capability_id="reasoning.math.calculation",
        suggestion_type="targeted_synthesis",
        predicted_gain=0.4,
        actual_gain=0.35,
        predicted_cost=4.0,
        actual_cost=6.0,
        note="synthesized 10k items",
    )
    rows = list_executions(session)
    assert len(rows) == 1
    assert rows[0]["id"] == row_id
    assert rows[0]["note"] == "synthesized 10k items"
    with pytest.raises(ValueError):
        log_execution(
            session,
            capability_id="c",
            suggestion_type="x",
            predicted_gain=0.1,
            actual_gain=0.1,
            predicted_cost=0.0,
            actual_cost=1.0,
        )
    session.close()


def test_feedback_recalibrate_ratios_and_clamp(tmp_path) -> None:
    session = _feedback_session(tmp_path)
    for i in range(4):
        log_execution(
            session,
            capability_id=f"cap{i}",
            suggestion_type="targeted_synthesis",
            predicted_gain=0.4,
            actual_gain=0.4 + 0.05 * i,
            predicted_cost=4.0,
            actual_cost=8.0,  # real cost 2x predicted
        )
    cal = recalibrate(session, smoothing=1.0)  # trust logs fully
    assert cal.n_logs == 4
    assert cal.costs["targeted_synthesis"] == pytest.approx(2.0, abs=1e-6)
    # gains: mean ratio = (1.0+1.125+1.25+1.375)/4 = 1.1875
    assert cal.gain_scale == pytest.approx(1.1875, abs=1e-6)
    loaded = load_calibration(session)
    assert loaded.costs["targeted_synthesis"] == pytest.approx(2.0, abs=1e-6)

    # too few logs -> prior kept
    cal2 = recalibrate(session, smoothing=1.0, min_logs_per_type=10)
    assert cal2.costs == {}
    session.close()


def test_feedback_recalibrate_smoothing_and_clamp(tmp_path) -> None:
    session = _feedback_session(tmp_path)
    for _ in range(4):
        log_execution(
            session,
            capability_id="c",
            suggestion_type="rejection_sampling",
            predicted_gain=0.3,
            actual_gain=0.3,
            predicted_cost=1.0,
            actual_cost=10.0,  # extreme: 10x
        )
    cal = recalibrate(session, smoothing=0.5)
    # 1 + 0.5*(10-1) = 5.5 -> clamped to 3.0
    assert cal.costs["rejection_sampling"] == pytest.approx(3.0, abs=1e-6)
    session.close()


# ---------------------------------------------------------------------------
# Orchestrator + pipeline integration (stages 1-7 end to end)
# ---------------------------------------------------------------------------


@pytest.fixture()
def seeded_db(tmp_path):
    engine = db.make_engine(tmp_path / "intelligent.db")
    db.init_db(engine)
    factory = db.session_factory(engine)
    session = factory()
    ingestion.load_seed(session)
    config = load_config()
    build_offline(session, config)
    yield session, config
    session.close()


def test_orchestrator_end_to_end(seeded_db, registry, taxonomy) -> None:
    session, config = seeded_db
    # pick a model with a params curve (residual-based judgment needs params)
    model = next(m for m in queries.list_models(session) if m.total_params)
    matrix = queries.scores_matrix(session)
    raw_scores: dict[str, float] = {}
    for bid in matrix.columns:
        value = float(matrix.loc[model.model_id, bid])
        # depress math-ish benchmarks so they fall below expectation
        if bid in {"gsm8k", "math", "aime24"}:
            value = max(5.0, value - 0.35)
        raw_scores[str(bid)] = value

    from benchmark_diagnosis.evaluation_orchestration.expectation_curves import judge
    from benchmark_diagnosis.intelligent_diagnosis.orchestrator import (
        run_intelligent_diagnosis,
    )
    from benchmark_diagnosis.pipeline import _revive_curves

    curves = _revive_curves(db.load_latest_asset(session, "curves") or [])
    verdicts = [
        judge(model, bid, score, curves, config.curves)
        for bid, score in raw_scores.items()
    ]
    verdicts = [
        {**v, "weight": 1.0} for v in verdicts if v["score"] is not None
    ]

    block = run_intelligent_diagnosis(
        session=session,
        verdict_benchmarks=verdicts,
        model=model,
        raw_scores=raw_scores,
        config=config,
                registry=registry,
        mode="full",
    )
    assert set(block) >= {
        "candidates",
        "probe_results",
        "verdicts",
        "priorities",
        "suggestions",
        "probe_todo",
        "calibration",
    }
    assert len(block["candidates"]) > 0
    for cand in block["candidates"]:
        assert cand["capability_id"]
        assert 0.0 <= cand["screening_score"] <= 1.0
    for probe in block["probe_results"]:
        assert probe["state"] in {s.value for s in ProbeState}
    for verdict in block["verdicts"]:
        assert verdict["confidence"] in {c.value for c in ConfidenceLevel}
    assert isinstance(block["suggestions"], dict)
    for group in ("training", "non_training", "build_probe_first", "human_review"):
        assert group in block["suggestions"]
    assert set(block["probe_todo"]) == {"build_probe_set", "pending_eval"}

    # full block must be JSON-serializable (metrics.json path)
    import json

    json.dumps(block)
    session.close()


def test_pipeline_diagnose_model_unified(seeded_db) -> None:
    session, config = seeded_db
    model = next(m for m in queries.list_models(session) if m.total_params)
    matrix = queries.scores_matrix(session)
    raw_scores: dict[str, float] = {}
    for bid in matrix.columns:
        value = float(matrix.loc[model.model_id, bid])
        if bid in {"gsm8k", "math", "aime24"}:
            value = max(5.0, value - 0.35)
        raw_scores[str(bid)] = value

    report = diagnose_model(
        session,
        model,
        raw_scores,
        config,
        mode="full",
        advisor_mode="rules",
        engine="legacy",
    )
    block = report["diagnosis"]
    assert block["candidates"]  # non-empty under-performing capability set
    assert report["clusters"]  # Stage 0 cluster verdicts still render
    session.close()
