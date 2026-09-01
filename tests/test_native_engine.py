"""Tests for the generic native evaluation backend (native_engine.py).

These stay offline: the dataset load and the endpoint client are injected /
stubbed, so no network or `datasets` package is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark_diagnosis.evaluation_orchestration.native_engine import (
    NATIVE_SPECS,
    is_native_benchmark,
    run_native,
)
from benchmark_diagnosis.evaluation_orchestration.task_registry import (
    NATIVE_INTERACTIVE_IDS,
    evaluable_backend,
)


NON_NATIVE = {"arena_hard", "livecodebench", "codeforces", "livebench", "multiif"}


def test_native_ids_registered_and_route_to_native_backend():
    """Every spec id is registered and routed to the native backend, not lm-eval."""
    for bid in NON_NATIVE:
        assert bid in NATIVE_SPECS, bid
        assert is_native_benchmark(bid)
        # They are deliberately not lm-eval tasks.
        assert evaluable_backend(bid) == "native"


@pytest.mark.parametrize("bid", sorted(NON_NATIVE))
def test_every_native_id_is_evaluable_via_registry(bid):
    assert bid in NATIVE_INTERACTIVE_IDS


def test_unknown_native_id_raises():
    with pytest.raises(ValueError, match="unknown native benchmark"):
        run_native(
            "model",
            "http://localhost:8000/v1",
            "not_a_benchmark",
            client_factory=_stub_client_factory,
        )


class _StubClient:
    def __init__(self):
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return "model completion"


def _stub_client_factory(base_url, model, api_key, max_tokens=None, timeout_seconds=120.0):
    return _StubClient()


def test_run_native_archives_and_reports_missing_external(monkeypatch, tmp_path):
    """Without the external scorer, cases are archived but scores stay empty."""
    monkeypatch.setattr(
        "benchmark_diagnosis.evaluation_orchestration.native_engine._load_spec_rows",
        lambda spec, limit: [
            {"idx": 0, "prompt": "p0"},
            {"idx": 1, "prompt": "p1"},
        ],
    )
    scores, results = run_native(
        "model",
        "http://localhost:8000/v1",
        "multiif",
        limit=5,
        output_dir=Path(tmp_path),
        client_factory=_stub_client_factory,
    )
    assert scores == {}  # no external multi_turn scorer supplied yet
    task = results["native"]["multiif"]
    assert task["cases"] == 2
    assert task["missing_external"] == ["multi_turn"]
    archive = Path(task["archived"])
    assert archive.name == "native_multiif_generations.jsonl"
    lines = archive.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "model completion" in lines[0]


def test_run_native_scores_when_external_scorer_and_score_cases_supplied(monkeypatch, tmp_path):
    """Once a spec's external component + score_cases exist, scores are produced."""
    spec = dict(NATIVE_SPECS["multiif"])
    spec["requires"] = ()
    spec["score_cases"] = lambda cases, ext: {"multiif": 0.5}
    monkeypatch.setitem(NATIVE_SPECS, "multiif", spec)
    monkeypatch.setattr(
        "benchmark_diagnosis.evaluation_orchestration.native_engine._load_spec_rows",
        lambda spec, limit: [{"idx": 0, "prompt": "p"}],
    )
    scores, results = run_native("model", "http://localhost:8000/v1", "multiif")
    assert scores == {"multiif": 0.5}
    assert results["native"]["multiif"]["cases"] == 1