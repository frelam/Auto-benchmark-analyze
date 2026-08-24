"""Evaluation artifact archiving: scores + bad cases written for humans.

After every benchmark run the runner calls :func:`write_eval_artifacts`, which
turns the raw harness payload plus its ``samples_*.jsonl`` item logs into a
self-contained, human-inspectable directory:

* ``scores.json``        — ``{benchmark_id: score}``, exactly the ``--scores``
  input shape, so it can be fed back into a follow-up run;
* ``eval_results.json``  — machine-readable detail per benchmark (metric,
  sample count, bad-case count, expectation-curve judgment);
* ``eval_summary.md``    — the human overview (scores table + bad-case stats);
* ``bad_cases/<benchmark_id>.jsonl`` — every failed sample, full fidelity
  (question / gold / model output / sample metrics / raw doc);
* ``bad_cases/<benchmark_id>.md``   — the same cases in a readable form;
* ``bad_cases/README.md`` — index over the archived benchmarks.

Bad cases are harvested from the harness's ``--log_samples`` output
(``samples_*.jsonl`` under the eval output dir). A sample counts as failed when
its primary sample metric is below 0.5; when the sample carries no metric a
normalized exact-match fallback against ``gold`` is used, and samples that
cannot be classified either way are skipped (they are not assumed to be bad).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark_diagnosis.data.ingestion import primary_metric, primary_metric_name
from benchmark_diagnosis.evaluation_orchestration.task_registry import to_benchmark_id

# Doc fields commonly used by lm-eval tasks for the question and the gold answer.
_QUESTION_KEYS = ("question", "problem", "input", "prompt", "ctx", "text")
_GOLD_KEYS = ("gold", "answer", "target", "label", "output", "completion")


@dataclass
class EvalArtifacts:
    """Paths of the archived evaluation artifacts."""

    scores_path: Path
    results_path: Path
    summary_path: Path
    bad_case_dir: Path
    bad_cases: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


# --------------------------------------------------------------------------- bad cases


def find_sample_files(output_dir: str | Path) -> dict[str, list[Path]]:
    """Map task name -> newest ``samples_*.jsonl`` files under ``output_dir``.

    lm-eval writes one samples file per evaluated task; the exact location
    varies across versions (``<output_dir>/samples_<task>.jsonl`` or a
    per-model subdirectory), so both layouts are scanned and the newest file
    per task wins.
    """
    root = Path(output_dir)
    if not root.exists():
        return {}
    found: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("samples_*.jsonl"), key=lambda p: p.stat().st_mtime):
        task = path.name[len("samples_") : -len(".jsonl")]
        found.setdefault(task, []).append(path)
    return {task: [paths[-1]] for task, paths in found.items()}


def load_samples(path: str | Path) -> list[dict[str, Any]]:
    """Read a ``samples_*.jsonl`` file into a list of sample dicts."""
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _sample_failed(sample: dict[str, Any]) -> bool | None:
    """Classify one sample: True failed / False passed / None unclassifiable."""
    metrics = sample.get("metrics") or {}
    metric_name = primary_metric_name(metrics)
    value = primary_metric(metrics)
    if metric_name is not None and value is not None:
        return float(value) < 0.5
    resps = sample.get("filtered_resps") or []
    doc = sample.get("doc") or {}
    gold = _first_field(doc, _GOLD_KEYS)
    if resps and gold is not None:
        output = resps[0] if isinstance(resps, list) else str(resps)
        return _normalize(output) != _normalize(str(gold))
    return None


def _normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _first_field(doc: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in doc and doc[key] is not None:
            return doc[key]
    return None


def _case_from_sample(
    sample: dict[str, Any], task: str, benchmark_id: str
) -> dict[str, Any] | None:
    """Build a bad-case record from a failed sample (None when unclassifiable)."""
    failed = _sample_failed(sample)
    if failed is not True:
        return None
    doc = sample.get("doc") or {}
    resps = sample.get("filtered_resps") or []
    output = resps[0] if isinstance(resps, list) and resps else ""
    if isinstance(output, list):  # nested response list
        output = " ".join(str(x) for x in output)
    return {
        "benchmark_id": benchmark_id,
        "task": task,
        "doc_id": sample.get("doc_id"),
        "sample_index": sample.get("sample_index"),
        "question": _first_field(doc, _QUESTION_KEYS) or "",
        "gold": _first_field(doc, _GOLD_KEYS) or "",
        "model_output": str(output),
        "metrics": sample.get("metrics") or {},
        "doc": doc,
    }


def extract_bad_cases(
    sample_files: dict[str, list[Path]], tasks: list[str] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Extract failed samples per benchmark id from the harness sample logs.

    Args:
        sample_files: ``task -> [paths]`` from :func:`find_sample_files`.
        tasks: Optional whitelist of evaluated task names; when None every
            discovered task is considered.

    Returns:
        ``benchmark_id -> [bad case dicts]`` (order: sample file order).
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for task, paths in sample_files.items():
        if tasks is not None and task not in tasks:
            continue
        bid = to_benchmark_id(task)
        for path in paths:
            for sample in load_samples(path):
                case = _case_from_sample(sample, task, bid)
                if case is not None:
                    out.setdefault(bid, []).append(case)
    return out


# --------------------------------------------------------------------------- writing


def _truncate(text: str, limit: int) -> str:
    text = str(text).replace("\r", "")
    return text if len(text) <= limit else text[:limit] + "…"


def _render_cases_markdown(cases: list[dict[str, Any]]) -> str:
    lines = [f"# Bad cases — {cases[0]['benchmark_id']}", ""]
    for idx, case in enumerate(cases, start=1):
        lines += [
            f"## Case {idx}",
            "",
            f"- doc_id: {case.get('doc_id')}",
            f"- sample_index: {case.get('sample_index')}",
            f"- metrics: {json.dumps(case.get('metrics') or {}, ensure_ascii=False)}",
            "",
            "**Question**",
            "",
            _truncate(case.get("question", ""), 800),
            "",
            "**Model output**",
            "",
            "```text",
            _truncate(case.get("model_output", ""), 1000),
            "```",
            "",
            "**Gold**",
            "",
            _truncate(case.get("gold", ""), 500),
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


def write_eval_artifacts(
    output_dir: str | Path,
    raw_scores: dict[str, float],
    results: dict[str, Any],
    sample_files: dict[str, list[Path]] | None = None,
    *,
    benchmark_names: dict[str, str] | None = None,
    judgments: dict[str, dict[str, Any]] | None = None,
) -> EvalArtifacts:
    """Write scores / eval detail / summary / bad cases under ``output_dir``.

    Args:
        output_dir: Where the artifacts land (the run output dir).
        raw_scores: ``benchmark_id -> score``.
        results: Parsed harness results payload (may be empty for the
            ``--scores`` path, which archives scores without sample logs).
        sample_files: ``task -> [paths]`` from :func:`find_sample_files`
            (default: discovered under the run's eval output dir).
        benchmark_names: Optional ``benchmark_id -> display name``.
        judgments: Optional per-benchmark expectation judgments
            (``curve_kind`` / ``percentile`` / ``z_score`` / ``predicted`` /
            ``residual``) to enrich ``eval_results.json``.

    Returns:
        The written artifact paths.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    tasks = list((results.get("results") or {}).keys()) or None
    bad_cases = extract_bad_cases(sample_files or {}, tasks=tasks)

    scores_path = root / "scores.json"
    scores_path.write_text(
        json.dumps(raw_scores, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    detail: dict[str, Any] = {}
    for bid, score in raw_scores.items():
        cases = bad_cases.get(bid, [])
        entry: dict[str, Any] = {
            "score": score,
            "n_bad_cases": len(cases),
            "bad_cases_file": f"bad_cases/{bid}.jsonl" if cases else None,
            "judgment": (judgments or {}).get(bid) or {},
        }
        metrics_by_task: dict[str, float] = {}
        for task, metrics in (results.get("results") or {}).items():
            if to_benchmark_id(task) == bid:
                value = primary_metric(metrics)
                if value is not None:
                    metrics_by_task[primary_metric_name(metrics) or "metric"] = value
        entry["metrics"] = metrics_by_task
        detail[bid] = entry

    # Sample counts per benchmark (sum over the task files that map to it).
    for task, paths in (sample_files or {}).items():
        bid = to_benchmark_id(task)
        if bid in detail and paths:
            n = sum(len(load_samples(p)) for p in paths)
            detail[bid]["n_samples"] = detail[bid].get("n_samples", 0) + n

    results_path = root / "eval_results.json"
    results_path.write_text(
        json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    bad_dir = root / "bad_cases"
    bad_dir.mkdir(parents=True, exist_ok=True)
    index_lines = ["# Bad-case archive", "", "| benchmark | cases | file |", "|---|---|---|"]
    for bid in sorted(bad_cases):
        cases = bad_cases[bid]
        jsonl_path = bad_dir / f"{bid}.jsonl"
        jsonl_path.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n",
            encoding="utf-8",
        )
        md_path = bad_dir / f"{bid}.md"
        md_path.write_text(_render_cases_markdown(cases), encoding="utf-8")
        name = (benchmark_names or {}).get(bid, "")
        index_lines.append(f"| {bid} {name} | {len(cases)} | `bad_cases/{bid}.jsonl` |")
    if bad_cases:
        index_lines += [
            "",
            "> 每条记录含 question / gold / model_output / metrics，可手工分析；",
            "> 也可以直接喂给 llm-agent 诊断的 bad-case 分析步骤。",
        ]
    else:
        index_lines += ["", "> 本跑次没有可归档的 bad case（无样本日志或全部通过）。"]
    (bad_dir / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    summary = _render_summary(raw_scores, detail, benchmark_names or {})
    summary_path = root / "eval_summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    return EvalArtifacts(
        scores_path=scores_path,
        results_path=results_path,
        summary_path=summary_path,
        bad_case_dir=bad_dir,
        bad_cases=bad_cases,
    )


def _render_summary(
    raw_scores: dict[str, float],
    detail: dict[str, Any],
    benchmark_names: dict[str, str],
) -> str:
    lines = [
        "# Evaluation summary",
        "",
        f"benchmarks scored: **{len(raw_scores)}**",
        "",
        "| benchmark | score | metric | samples | bad cases | judgment |",
        "|---|---|---|---|---|---|",
    ]
    for bid in sorted(raw_scores):
        entry = detail.get(bid, {})
        name = benchmark_names.get(bid, "")
        metrics = entry.get("metrics") or {}
        metric = ",".join(f"{k}={v:.3f}" for k, v in sorted(metrics.items())) or "—"
        n_samples = entry.get("n_samples", "—")
        n_bad = entry.get("n_bad_cases", 0)
        j = entry.get("judgment") or {}
        judgment = ""
        if j.get("curve_kind"):
            judgment = (
                f"{j['curve_kind']} · p{j.get('percentile')} · "
                f"z={j.get('z_score')}"
            )
        lines.append(f"| {bid} {name} | {raw_scores[bid]:.3f} | {metric} | {n_samples} | {n_bad} | {judgment} |")
    lines += [
        "",
        "Bad cases 逐条见 `bad_cases/`；机器可读明细见 `eval_results.json`。",
        "下一轮可用 `run --scores scores.json` 直接接续诊断。",
        "",
    ]
    return "\n".join(lines)
