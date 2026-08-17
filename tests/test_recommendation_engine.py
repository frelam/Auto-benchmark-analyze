"""Tests for the recommendation engine (deterministic scoring + LLM re-rank)."""

from __future__ import annotations

from benchmark_diagnosis.config import RecommendationConfig
from benchmark_diagnosis.core.types import DiagnosisResult
from benchmark_diagnosis.recommendation import engine
from benchmark_diagnosis.recommendation.experience_base import (
    DatasetSuggestion,
    ExperienceBase,
    HyperparameterSuggestion,
    Intervention,
    Outcome,
    load_experience_base,
)


class FakeLLM:
    """Duck-typed analyst LLM returning a canned JSON dict."""

    def __init__(self, response=None) -> None:
        self._response = response
        self.calls = 0

    def complete_json(self, messages):
        self.calls += 1
        if callable(self._response):
            return self._response(messages)
        return self._response


def _intervention(
    iid: str,
    tags: list[str],
    *,
    evidence: str = "low",
    outcomes: list[Outcome] | None = None,
    linked_rule: str | None = None,
) -> Intervention:
    return Intervention(
        intervention_id=iid,
        title=f"{iid} title",
        applicable_tags=tags,
        category="data_mix",
        expected_effect=f"{iid} expected effect",
        evidence_strength=evidence,
        source_type="human_expert",
        linked_rule_id=linked_rule,
        datasets=[
            DatasetSuggestion(f"{iid}-ds", f"{iid} rationale", f"{iid} effect", "human_expert")
        ],
        hyperparameters=[
            HyperparameterSuggestion(f"{iid}-knob", "up", "1-2", f"{iid} rationale")
        ],
        outcomes=list(outcomes or []),
    )


def _base() -> ExperienceBase:
    return ExperienceBase(
        version=1,
        interventions=[
            _intervention("EXP-math-a", ["math"]),
            _intervention("EXP-math-b", ["math"], outcomes=[Outcome(0.02)]),
            _intervention("EXP-ctx-a", ["long_context"]),
            _intervention("EXP-read-a", ["reading_comprehension"], evidence="medium"),
            # failure-mode tag (like the packaged seed) so the legacy fallback
            # that passes failure-mode ids as tags can match.
            _intervention("EXP-calc", ["calculation_error"]),
        ],
        failure_mode_capability_map={"calculation_error": ["math"]},
    )


def _diagnosis(**overrides) -> DiagnosisResult:
    defaults = dict(
        cluster_id="C1",
        sub_capability="math",
        failure_modes={},
        quantified_gap=0.12,
        capability_deficit={"math": 0.8, "reasoning": 0.2},
        deficit_narrative="该簇缺失能力：math(0.80)、reasoning(0.20)",
    )
    defaults.update(overrides)
    return DiagnosisResult(**defaults)


# ---------------------------------------------------------------------------
# deterministic Stage 1 (no LLM)
# ---------------------------------------------------------------------------


def test_deterministic_populates_executable_fields() -> None:
    recs = engine.recommend(_diagnosis(), _base(), RecommendationConfig(max_actions=2), None)
    assert len(recs) == 2
    rec = recs[0]
    assert rec.source.startswith("experience:")
    assert rec.datasets and rec.datasets[0]["name"].endswith("-ds")
    assert rec.hyperparameters and rec.hyperparameters[0]["knob"].endswith("-knob")
    assert rec.expected_effect.endswith("expected effect")
    assert rec.intervention_id is not None
    assert any("该簇缺失能力" in r for r in rec.reason_chain)
    assert any("匹配干预" in r for r in rec.reason_chain)


def test_deterministic_outcome_lift_reorders() -> None:
    # EXP-math-b has a measured +2% outcome -> ranks above the same-tag rival.
    recs = engine.recommend(_diagnosis(), _base(), RecommendationConfig(max_actions=4), None)
    ids = [r.intervention_id for r in recs]
    assert ids.index("EXP-math-b") < ids.index("EXP-math-a")


def test_deterministic_reason_chain_includes_outcome_history() -> None:
    recs = engine.recommend(
        _diagnosis(capability_deficit={"math": 1.0}), _base(), RecommendationConfig(), None
    )
    rec = next(r for r in recs if r.intervention_id == "EXP-math-b")
    assert any("平均提升 +2.0%" in r for r in rec.reason_chain)


def test_empty_deficit_falls_back_to_failure_modes() -> None:
    diagnosis = _diagnosis(
        capability_deficit={},
        deficit_narrative="",
        failure_modes={"calculation_error": 0.8},
    )
    recs = engine.recommend(diagnosis, _base(), RecommendationConfig(max_actions=3), None)
    # failure-mode ids are passed as tags -> the intervention carrying the
    # failure-mode tag matches.
    assert [r.intervention_id for r in recs] == ["EXP-calc"]


def test_empty_deficit_falls_back_to_benchmark_tags() -> None:
    diagnosis = _diagnosis(
        sub_capability="longbench_v2",
        capability_deficit={},
        deficit_narrative="",
        failure_modes={},
    )
    benchmark_tags = {"longbench_v2": ["long_context"]}
    recs = engine.recommend(
        diagnosis,
        _base(),
        RecommendationConfig(max_actions=3),
        None,
        benchmark_tags=benchmark_tags,
    )
    assert [r.intervention_id for r in recs] == ["EXP-ctx-a"]


def test_empty_deficit_falls_back_to_sub_capability() -> None:
    diagnosis = _diagnosis(
        sub_capability="reading_comprehension",
        capability_deficit={},
        deficit_narrative="",
        failure_modes={},
    )
    recs = engine.recommend(diagnosis, _base(), RecommendationConfig(max_actions=3), None)
    assert [r.intervention_id for r in recs] == ["EXP-read-a"]


def test_no_matching_intervention_returns_empty() -> None:
    recs = engine.recommend(
        _diagnosis(capability_deficit={"vision": 1.0}), _base(), RecommendationConfig(), None
    )
    assert recs == []


def test_max_actions_caps_output() -> None:
    recs = engine.recommend(_diagnosis(), _base(), RecommendationConfig(max_actions=1), None)
    assert len(recs) == 1


# ---------------------------------------------------------------------------
# LLM Stage 2 re-rank
# ---------------------------------------------------------------------------


def _llm_diagnosis() -> DiagnosisResult:
    return _diagnosis(capability_deficit={"math": 1.0})


def test_llm_rerank_uses_grounded_actions() -> None:
    def response(_messages):
        return {
            "diagnosis_basis": "gap 0.12 deficit [math]",
            "actions": [
                {
                    "intervention_id": "EXP-math-b",
                    "source": "experience:EXP-math-b",
                    "action": "LLM-picked math-b",
                }
            ],
        }

    recs = engine.recommend(_llm_diagnosis(), _base(), RecommendationConfig(), FakeLLM(response))
    assert [r.intervention_id for r in recs] == ["EXP-math-b"]
    assert recs[0].action == "LLM-picked math-b"


def test_llm_hallucinated_intervention_falls_back() -> None:
    def response(_messages):
        return {
            "diagnosis_basis": "gap 0.12",
            "actions": [
                {
                    "intervention_id": "EXP-madeup-999",
                    "source": "experience:EXP-madeup-999",
                    "action": "fake",
                }
            ],
        }

    recs = engine.recommend(_llm_diagnosis(), _base(), RecommendationConfig(), FakeLLM(response))
    # Grounding fails -> Stage 1 deterministic (EXP-math-b first, has lift).
    assert recs[0].intervention_id == "EXP-math-b"
    assert all(r.intervention_id != "EXP-madeup-999" for r in recs)


def test_llm_failure_falls_back_to_stage1() -> None:
    def response(_messages):
        raise RuntimeError("llm down")

    recs = engine.recommend(_llm_diagnosis(), _base(), RecommendationConfig(), FakeLLM(response))
    assert recs[0].intervention_id == "EXP-math-b"


def test_llm_malformed_json_falls_back() -> None:
    recs = engine.recommend(_llm_diagnosis(), _base(), RecommendationConfig(), FakeLLM([1, 2, 3]))
    assert recs[0].intervention_id == "EXP-math-b"


# ---------------------------------------------------------------------------
# integration with the packaged experience base
# ---------------------------------------------------------------------------


def test_packaged_base_supports_long_context_deficit() -> None:
    base = load_experience_base()
    diagnosis = _diagnosis(
        sub_capability="longbench_v2",
        capability_deficit={"long_context": 0.6, "reading_comprehension": 0.4},
        failure_modes={"context_ignoring": 0.6},
    )
    recs = engine.recommend(diagnosis, base, RecommendationConfig(max_actions=3), None)
    assert recs
    assert recs[0].intervention_id == "EXP-longctx-001"
    assert any(d["name"].startswith("LongBench") for d in recs[0].datasets)
    assert any("rope_scaling" in h["knob"] for h in recs[0].hyperparameters)
