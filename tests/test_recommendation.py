"""Tests for recommendation retrieval, synthesizer, and groundedness check."""

from __future__ import annotations

import pytest

from benchmark_diagnosis.config import RecommendationConfig
from benchmark_diagnosis.core.types import DiagnosisResult
from benchmark_diagnosis.recommendation.groundedness_check import check_groundedness
from benchmark_diagnosis.recommendation.retrieval import NullRetriever, get_retriever
from benchmark_diagnosis.recommendation.rule_base.loader import (
    load_validated_rules,
    match_rules,
)
from benchmark_diagnosis.recommendation.synthesizer import synthesize


class FakeLLM:
    """Duck-typed analyst LLM returning a canned response."""

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
        raise RuntimeError("no analyst model configured")


def _diagnosis(**overrides) -> DiagnosisResult:
    defaults = dict(
        cluster_id="math-reasoning",
        sub_capability="math",
        failure_modes={"reasoning_error": 0.6, "calculation_error": 0.4},
        quantified_gap=0.12,
    )
    defaults.update(overrides)
    return DiagnosisResult(**defaults)


@pytest.fixture()
def rule_base():
    return load_validated_rules()


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


def test_get_retriever_returns_null_retriever() -> None:
    retriever = get_retriever(RecommendationConfig())
    assert isinstance(retriever, NullRetriever)


def test_null_retriever_returns_empty() -> None:
    retriever = NullRetriever()
    assert retriever.retrieve("math reasoning gap") == []
    assert retriever.retrieve("anything", max_results=3) == []


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------


def test_synthesize_fallback_on_llm_failure(rule_base) -> None:
    _taxonomy, rules = rule_base
    diagnosis = _diagnosis()
    out = synthesize(ExplodingLLM(), diagnosis, rules, [])

    assert "diagnosis_basis" in out and "actions" in out
    assert len(out["actions"]) >= 1
    assert len(out["actions"]) <= 3

    expected_ids = [
        r.rule_id
        for r, _score in match_rules(rules, list(diagnosis.failure_modes))[:3]
    ]
    assert [a["rule_id"] for a in out["actions"]] == expected_ids
    for action in out["actions"]:
        assert action["source"] == f"rule_base:{action['rule_id']}"
        assert action["action"]
        assert action["validation_experiment"]


def test_synthesize_fallback_with_empty_failure_modes(rule_base) -> None:
    _taxonomy, rules = rule_base
    diagnosis = _diagnosis(failure_modes={})
    out = synthesize(ExplodingLLM(), diagnosis, rules, [])
    assert out["actions"] == []
    assert "diagnosis_basis" in out


def test_synthesize_llm_path_passes_through(rule_base) -> None:
    _taxonomy, rules = rule_base

    def response(_messages):
        return {
            "diagnosis_basis": "math reasoning gap 0.12 dominated by reasoning_error",
            "actions": [
                {
                    "rule_id": "R-reward-001",
                    "source": "rule_base:R-reward-001",
                    "action": "introduce process reward",
                    "validation_experiment": "ablate on 500 math items",
                }
            ],
        }

    llm = FakeLLM(response)
    out = synthesize(llm, _diagnosis(), rules, [])
    assert out["diagnosis_basis"].startswith("math reasoning gap")
    assert out["actions"][0]["rule_id"] == "R-reward-001"
    assert out["actions"][0]["source"] == "rule_base:R-reward-001"
    assert llm.calls == 1


def test_synthesize_llm_malformed_falls_back(rule_base) -> None:
    _taxonomy, rules = rule_base
    out = synthesize(FakeLLM([1, 2, 3]), _diagnosis(), rules, [])
    assert "diagnosis_basis" in out and "actions" in out
    assert all(a["source"].startswith("rule_base:") for a in out["actions"])


def test_synthesize_llm_missing_actions_normalized(rule_base) -> None:
    _taxonomy, rules = rule_base
    out = synthesize(FakeLLM({"diagnosis_basis": "some basis"}), _diagnosis(), rules, [])
    assert out["diagnosis_basis"] == "some basis"
    assert out["actions"] == []


# ---------------------------------------------------------------------------
# groundedness_check
# ---------------------------------------------------------------------------


def _action(rule_id: str | None, source: str) -> dict:
    return {
        "rule_id": rule_id,
        "source": source,
        "action": "do something",
        "validation_experiment": "run an ablation",
    }


def test_groundedness_clean(rule_base) -> None:
    synthesis = {
        "diagnosis_basis": "gap 0.12 dominated by reasoning_error fraction 0.6",
        "actions": [
            _action("R-reward-001", "rule_base:R-reward-001"),
            _action(None, "external:arxiv-42"),
        ],
    }
    evidence = {
        "rule_ids": {"R-reward-001"},
        "sources": {"rule_base:R-reward-001"},
        "numbers": {0.12, 0.6},
    }
    assert check_groundedness(synthesis, evidence) == []


def test_groundedness_catches_fabricated_rule_id_and_source(rule_base) -> None:
    synthesis = {
        "diagnosis_basis": "gap 0.12",
        "actions": [_action("R-FAKE-99", "rule_base:R-FAKE-99")],
    }
    evidence = {
        "rule_ids": {"R-reward-001"},
        "sources": {"rule_base:R-reward-001"},
        "numbers": {0.12},
    }
    violations = check_groundedness(synthesis, evidence)
    assert len(violations) == 2
    assert any("R-FAKE-99" in v for v in violations)


def test_groundedness_catches_fabricated_number(rule_base) -> None:
    synthesis = {"diagnosis_basis": "improves by 0.99 points", "actions": []}
    evidence = {"rule_ids": set(), "sources": set(), "numbers": {0.12, 0.6}}
    violations = check_groundedness(synthesis, evidence)
    assert len(violations) == 1
    assert "0.99" in violations[0]


def test_groundedness_tolerates_numbers_within_precision(rule_base) -> None:
    synthesis = {"diagnosis_basis": "gap 0.3500001", "actions": []}
    evidence = {"rule_ids": set(), "sources": set(), "numbers": {0.35}}
    assert check_groundedness(synthesis, evidence) == []


def test_groundedness_tolerates_missing_evidence_keys(rule_base) -> None:
    synthesis = {"diagnosis_basis": "no numbers here", "actions": []}
    assert check_groundedness(synthesis, {}) == []
