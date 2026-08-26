"""Tests for the native BFCL evaluation backend (categorization + harness wrap)."""

from __future__ import annotations

import pytest

from benchmark_diagnosis.evaluation_orchestration.bfcl_eval import (
    BFCL_BENCHMARK_ID,
    categorize_scores,
    is_bfcl_benchmark,
    run_bfcl,
)


class _Handler:
    """In-memory stand-in for ``bfcl_eval.handler`` recording the call."""

    calls: list[dict] = []

    def __init__(self, result, *, drop_base_url: bool = False):
        self.result = result
        self.drop_base_url = drop_base_url

    def run_inference_and_eval(self, **kwargs):
        type(self).calls.append(kwargs)
        return self.result


@pytest.mark.parametrize(
    "result,expected",
    [
        (
            {"ast_acc": 0.72, "other_categories": [["simple", 0.85], ["multi_turn", 0.6]]},
            {"bfcl": 0.72, "bfcl.simple": 0.85, "bfcl.multi_turn": 0.6},
        ),
        # v4-style overall key
        ({"overall_acc": 0.68}, {"bfcl": 0.68}),
        # missing overall -> no top-level entry
        ({"other_categories": [["simple", 0.9]]}, {"bfcl.simple": 0.9}),
        # malformed rows are skipped, never raised
        ({"ast_acc": "not-a-number", "other_categories": [[None, 0.1], ["ok", "nan"]]}, {}),
        # non-mapping result is tolerated
        (None, {}),
    ],
)
def test_categorize_scores(result, expected):
    assert categorize_scores(result) == expected


def test_is_bfcl_benchmark_only_matches_bfcl():
    assert is_bfcl_benchmark("bfcl") is True
    assert is_bfcl_benchmark("math") is False
    assert is_bfcl_benchmark("bfcl.simple") is False


def test_run_bfcl_passes_endpoint_and_parses(monkeypatch):
    handler = _Handler(
        {"ast_acc": 0.8, "other_categories": [["parallel", 0.7]]},
        drop_base_url=False,
    )
    monkeypatch.setattr(
        "benchmark_diagnosis.evaluation_orchestration.bfcl_eval._import_bfcl",
        lambda: handler,
    )
    scores = run_bfcl(
        "my-model", "http://127.0.0.1:8000/v1", categories=["simple"],
        model_type="vllm", api_key="EMPTY",
    )
    assert scores == {"bfcl": 0.8, "bfcl.parallel": 0.7}
    call = _Handler.calls[-1]
    assert call["model"] == "my-model"
    assert call["base_url"] == "http://127.0.0.1:8000/v1"
    assert call["model_type"] == "vllm"
    assert call["test_categories"] == ["simple"]


def test_run_bfcl_backs_off_base_url_on_typeerror(monkeypatch):
    calls: list[dict] = []

    def backend(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1 and "base_url" in kwargs:
            raise TypeError("fn() got an unexpected keyword argument 'base_url'")
        return {"ast_acc": 0.5}

    monkeypatch.setattr(
        "benchmark_diagnosis.evaluation_orchestration.bfcl_eval._import_bfcl",
        lambda: type("H", (), {"run_inference_and_eval": staticmethod(backend)}),
    )
    assert run_bfcl("m", "u", model_type="vllm") == {"bfcl": 0.5}
    # first attempt carried base_url and was rejected; the retry dropped it.
    assert "base_url" in calls[0]
    assert "base_url" not in calls[-1]
    assert calls[-1]["model"] == "m"


def test_run_bfcl_reports_missing_package(monkeypatch):
    def boom():
        raise RuntimeError("BFCL evaluation requested but the official `bfcl-eval` harness is not installed")
    monkeypatch.setattr(
        "benchmark_diagnosis.evaluation_orchestration.bfcl_eval._import_bfcl",
        boom,
    )
    with pytest.raises(RuntimeError, match="bfcl-eval"):
        run_bfcl("m", "u")