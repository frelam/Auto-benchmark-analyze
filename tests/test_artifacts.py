"""Tests for evaluation artifact archiving (scores + bad cases for humans)."""

from __future__ import annotations

import json

from benchmark_diagnosis.evaluation_orchestration.artifacts import (
    extract_bad_cases,
    find_sample_files,
    load_samples,
    write_eval_artifacts,
)


def _sample(task: str, doc_id: int, metric: float | None, *, gold: str = "42", output: str = "42") -> dict:
    sample = {
        "doc": {"question": f"q{doc_id}", "gold": gold},
        "doc_id": doc_id,
        "sample_index": doc_id,
        "filtered_resps": [output],
        "resps": [[output]],
    }
    if metric is not None:
        sample["metrics"] = {"exact_match": metric}
    return sample


def test_find_sample_files_discovers_newest_per_task(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "samples_math.jsonl").write_text("", encoding="utf-8")
    old = tmp_path / "sub" / "samples_math.jsonl"
    old.write_text("", encoding="utf-8")
    files = find_sample_files(tmp_path)
    assert set(files) == {"math"}
    assert files["math"] == [old]  # newest wins


def test_extract_bad_cases_classifies_failed_passed_unclear(tmp_path):
    path = tmp_path / "samples_math.jsonl"
    lines = [
        _sample("math", 0, 0.0),          # failed via metric
        _sample("math", 1, 1.0),          # passed via metric
        _sample("math", 2, None, gold="42", output="wrong"),  # failed via fallback
        _sample("math", 3, None, gold="42", output="42"),     # passed via fallback
    ]
    path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in lines) + "\n",
        encoding="utf-8",
    )
    bad = extract_bad_cases({"math": [path]})
    assert list(bad) == ["math"]
    cases = bad["math"]
    assert [c["doc_id"] for c in cases] == [0, 2]
    assert cases[0]["question"] == "q0"
    assert cases[0]["gold"] == "42"
    assert cases[0]["model_output"] == "42"
    assert cases[0]["metrics"] == {"exact_match": 0.0}


def test_write_eval_artifacts_layout(tmp_path):
    sample_file = tmp_path / "samples_math.jsonl"
    sample_file.write_text(
        json.dumps(_sample("math", 0, 0.0), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    results = {"results": {"math": {"acc,none": 0.5}}}
    out = tmp_path / "out"
    artifacts = write_eval_artifacts(
        out,
        {"math": 0.5},
        results,
        {"math": [sample_file]},
        benchmark_names={"math": "Math"},
        judgments={"math": {"curve_kind": "params_dense", "percentile": 12.0, "z_score": -2.0}},
    )

    scores = json.loads(artifacts.scores_path.read_text(encoding="utf-8"))
    assert scores == {"math": 0.5}

    detail = json.loads(artifacts.results_path.read_text(encoding="utf-8"))
    assert detail["math"]["n_bad_cases"] == 1
    assert detail["math"]["n_samples"] == 1
    assert detail["math"]["metrics"] == {"acc": 0.5}
    assert detail["math"]["judgment"]["percentile"] == 12.0

    summary = artifacts.summary_path.read_text(encoding="utf-8")
    assert "| math Math | 0.500 |" in summary
    assert "params_dense" in summary

    assert (out / "bad_cases" / "math.jsonl").exists()
    assert (out / "bad_cases" / "math.md").exists()
    assert (out / "bad_cases" / "README.md").exists()
    md = (out / "bad_cases" / "math.md").read_text(encoding="utf-8")
    assert "q0" in md and "Case 1" in md


def test_write_eval_artifacts_empty_scores_path(tmp_path):
    """The --scores path archives scores without any sample logs."""
    out = tmp_path / "out"
    artifacts = write_eval_artifacts(out, {"gsm8k": 79.0}, {})
    assert json.loads(artifacts.scores_path.read_text(encoding="utf-8")) == {"gsm8k": 79.0}
    assert (out / "bad_cases" / "README.md").exists()
    assert not list((out / "bad_cases").glob("*.jsonl"))


def test_load_samples_skips_bad_lines(tmp_path):
    path = tmp_path / "samples_x.jsonl"
    path.write_text('{"a": 1}\nnot-json\n{"b": 2}\n', encoding="utf-8")
    samples = load_samples(path)
    assert [s for s in samples] == [{"a": 1}, {"b": 2}]
