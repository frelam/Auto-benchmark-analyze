"""Tests for diagnosis.label_slicing and diagnosis.failure_mode_analyst."""

from __future__ import annotations

import pandas as pd
import pytest

from benchmark_diagnosis.diagnosis.failure_mode_analyst import classify_failures
from benchmark_diagnosis.diagnosis.label_slicing import slice_by_subcategory
from benchmark_diagnosis.recommendation.rule_base.loader import load_validated_rules


class FakeLLM:
    """Duck-typed stand-in for LLMClient exposing ``complete_json(messages)``."""

    def __init__(self, response=None) -> None:
        self._response = response
        self.calls = 0

    def complete_json(self, messages):
        self.calls += 1
        if callable(self._response):
            return self._response(messages)
        return self._response


class ExplodingLLM:
    """Analyst LLM that raises (e.g. no endpoint reachable)."""

    def complete_json(self, messages):
        raise RuntimeError("analyst endpoint unavailable")


def _make_cases(count: int) -> list[dict]:
    return [
        {"question": f"q{i}", "model_output": f"out{i}", "gold": f"gold{i}"}
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# slice_by_subcategory
# ---------------------------------------------------------------------------


def test_slice_by_subcategory_means_and_order() -> None:
    items = pd.DataFrame(
        {
            "subcategory": ["math", "math", "math", "code", "code"],
            "correct": [True, True, False, True, False],
        }
    )
    result = slice_by_subcategory(items)
    assert result["math"] == pytest.approx(2 / 3)
    assert result["code"] == pytest.approx(0.5)
    # Worst subcategory first.
    assert list(result.keys()) == ["code", "math"]


def test_slice_by_subcategory_numeric_and_string_truthiness() -> None:
    items = pd.DataFrame(
        {
            "subcategory": ["a", "a", "b", "b", "b"],
            "correct": [1, 0, "yes", "", 0],
        }
    )
    result = slice_by_subcategory(items)
    assert result["a"] == pytest.approx(0.5)  # 1 -> 1.0, 0 -> 0.0
    assert result["b"] == pytest.approx(1 / 3)  # "yes" truthy, "" and 0 falsy


def test_slice_by_subcategory_missing_correct_is_falsy() -> None:
    items = pd.DataFrame(
        {"subcategory": ["a", "a", "a"], "correct": [True, False, None]}
    )
    result = slice_by_subcategory(items)
    assert result["a"] == pytest.approx(1 / 3)


def test_slice_by_subcategory_custom_columns() -> None:
    items = pd.DataFrame(
        {"subcat_label": ["x", "x", "y"], "is_correct": [False, False, True]}
    )
    result = slice_by_subcategory(items, subcategory_col="subcat_label", correct_col="is_correct")
    assert result["x"] == pytest.approx(0.0)
    assert result["y"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# classify_failures
# ---------------------------------------------------------------------------


@pytest.fixture()
def taxonomy_and_rules():
    return load_validated_rules()


def test_classify_failures_fractions(taxonomy_and_rules) -> None:
    taxonomy, _rules = taxonomy_and_rules

    def response(_messages):
        return {
            "classifications": [
                {
                    "case_index": i,
                    "failure_mode": "reasoning_error" if i % 2 == 0 else "calculation_error",
                }
                for i in range(10)
            ]
        }

    llm = FakeLLM(response)
    result = classify_failures(llm, taxonomy, _make_cases(10))
    assert result == {"reasoning_error": 0.5, "calculation_error": 0.5}
    assert llm.calls == 1


def test_classify_failures_empty_cases(taxonomy_and_rules) -> None:
    taxonomy, _rules = taxonomy_and_rules
    llm = FakeLLM({"classifications": []})
    assert classify_failures(llm, taxonomy, []) == {}
    assert llm.calls == 0


def test_classify_failures_sample_size_caps(taxonomy_and_rules) -> None:
    taxonomy, _rules = taxonomy_and_rules

    def response(_messages):
        return {
            "classifications": [
                {"case_index": i, "failure_mode": "reasoning_error"} for i in range(10)
            ]
        }

    llm = FakeLLM(response)
    # 25 cases but sample_size 10 => only first 10 classified, fractions over 10.
    result = classify_failures(llm, taxonomy, _make_cases(25), sample_size=10)
    assert result == {"reasoning_error": 1.0}
    assert llm.calls == 1


def test_classify_failures_invalid_id_maps_to_unclassified(taxonomy_and_rules) -> None:
    taxonomy, _rules = taxonomy_and_rules
    llm = FakeLLM(
        {
            "classifications": [
                {"case_index": i, "failure_mode": "made_up_mode"} for i in range(10)
            ]
        }
    )
    result = classify_failures(llm, taxonomy, _make_cases(5))
    assert result == {"unclassified": 1.0}


def test_classify_failures_malformed_json_is_robust(taxonomy_and_rules) -> None:
    taxonomy, _rules = taxonomy_and_rules
    result = classify_failures(ExplodingLLM(), taxonomy, _make_cases(4))
    assert result == {"unclassified": 1.0}


def test_classify_failures_non_dict_response_is_robust(taxonomy_and_rules) -> None:
    taxonomy, _rules = taxonomy_and_rules
    llm = FakeLLM([1, 2, 3])  # valid JSON, wrong shape
    result = classify_failures(llm, taxonomy, _make_cases(4))
    assert result == {"unclassified": 1.0}
