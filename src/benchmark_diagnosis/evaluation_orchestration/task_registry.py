"""Map the tool's benchmark ids to lm-evaluation-harness task names.

The seed/coverage layer uses benchmark ids (e.g. ``math``, ``mmmlu``,
``longbench_v2``) that match lm-eval task names of the era the tool was
designed against. The pinned lm-eval (0.4.12) renamed or moved several of
them; this registry is the single source of truth for the translation:

* ``TASK_ALIASES``: benchmark_id -> lm-eval task name where they differ.
* ``is_evaluable`` / ``to_lm_eval_task``: availability + translation used by
  the runner (eval task list) and the offline asset build (portfolio
  selection only considers evaluable benchmarks, so every portfolio member
  can actually be run).

Benchmarks with no lm-eval 0.4.12 equivalent (``simpleqa``, ``swe_bench``,
``livecodebench``, ``frontiermath``, ``bigcodebench``, ``tau2_bench``,
``terminal_bench``, ``humaneval_plus``) are not evaluable here; they are
skipped from portfolio selection with a warning rather than failing the run.
"""

from __future__ import annotations

# benchmark_id -> lm-eval task name (only where they differ).
TASK_ALIASES: dict[str, str] = {
    "math": "hendrycks_math",
    "math_500": "hendrycks_math500",
    "longbench_v2": "longbench2",
    "gpqa": "gpqa_diamond_zeroshot",
}

# Benchmark ids whose lm-eval task name equals the id itself.
_EVALUABLE_IDENTITY = {
    "aime24",
    "aime25",
    "arc_challenge",
    "bbh",
    "gsm8k",
    "hellaswag",
    "humaneval",
    "ifeval",
    "mmlu",
    "mmlu_pro",
    "mmmlu",
}

# Reverse map: lm-eval task name -> benchmark_id.
_TASK_TO_ID = {task: bid for bid, task in TASK_ALIASES.items()}


def to_lm_eval_task(benchmark_id: str) -> str | None:
    """Return the lm-eval task name for a benchmark id, or None if missing."""
    if benchmark_id in TASK_ALIASES:
        return TASK_ALIASES[benchmark_id]
    if benchmark_id in _EVALUABLE_IDENTITY:
        return benchmark_id
    return None


def to_lm_eval_task_list(benchmark_ids: list[str]) -> list[str]:
    """Translate benchmark ids to lm-eval task names (unknown ids pass through).

    The evaluation path (``run`` / ``eval`` / ``eval-task``) passes task names
    straight to the harness CLI, which only knows lm-eval names (``math`` ->
    ``hendrycks_math``, ``longbench_v2`` -> ``longbench2``, ...). Without this
    translation a portfolio/CLI benchmark id that differs from the harness name
    fails with ``Tasks not found`` before evaluation starts.
    """
    return [to_lm_eval_task(bid) or bid for bid in benchmark_ids]


def is_evaluable(benchmark_id: str) -> bool:
    return to_lm_eval_task(benchmark_id) is not None


def to_benchmark_id(task_name: str) -> str:
    """Map an lm-eval task name back to the tool's benchmark id."""
    return _TASK_TO_ID.get(task_name, task_name)
