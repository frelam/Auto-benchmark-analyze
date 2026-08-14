"""Best-effort prefetch of lm-evaluation-harness tasks for the portfolio benchmarks.

Run via ``bash scripts/install.sh --download-data`` (or directly). It warms up
the task registry so the first real evaluation validates task names immediately,
and downloads whatever dataset components the harness pulls eagerly.

The harness's Python API drifts across versions, so every interaction is
defensive: a missing ``lm-eval`` or an unknown task is a *warning only* and the
script always exits 0 — datasets still download lazily at eval time if a task
cannot be prefetched here.
"""

from __future__ import annotations

import argparse
import sys

# benchmark_id (as used by this repo's representative portfolios) -> lm-eval task
# names to try, in order of preference. Names are best-effort and version-aware.
TASK_MAP: dict[str, list[str]] = {
    "mmlu_pro": ["mmlu_pro"],
    "mmlu": ["mmlu"],
    "gsm8k": ["gsm8k"],
    "math": ["math", "minerva_math"],
    "simpleqa": ["simpleqa"],
    "ifeval": ["ifeval"],
    "livecodebench": ["livecodebench"],
    "humaneval": ["humaneval", "human_eval"],
    "arc_easy": ["arc_easy"],
    "arc_challenge": ["arc_challenge"],
    "frontiermath": ["frontiermath"],
    "swe_bench": ["swe_bench"],
    "tau2_bench": ["tau2_bench"],
    "terminal_bench": ["terminal_bench"],
    "longbench_v2": ["longbench_v2"],
    "mmmlu": ["mmmlu", "mmmlu_pro"],
}


def _load_task_manager() -> tuple[object | None, str | None]:
    """Return a TaskManager (or None with a reason) without crashing on API drift."""
    try:
        from lm_eval.tasks import TaskManager
    except ImportError:
        return None, "lm-eval is not installed (pip install -e '.[eval]')"
    try:
        return TaskManager(), None
    except Exception as exc:  # noqa: BLE001 - defensive against API drift
        return None, f"could not create TaskManager: {exc}"


def prefetch(tm: object, benchmarks: list[str]) -> tuple[int, list[str]]:
    """Attempt to load each benchmark's lm-eval task; return (loaded, missed)."""
    loaded = 0
    missed: list[str] = []
    for bench in benchmarks:
        for task in TASK_MAP.get(bench, [bench]):
            try:
                tm.load_task_or_group(task)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - unknown task / name drift
                print(f"  warn  {bench:>14} -> {task}: {type(exc).__name__}")
                continue
            print(f"  ok    {bench:>14} -> {task}")
            loaded += 1
            break
        else:
            missed.append(bench)
    return loaded, missed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Best-effort prefetch of lm-eval tasks for the portfolio benchmarks."
    )
    parser.add_argument(
        "--benchmarks",
        help="Comma-separated benchmark ids to prefetch (default: all known ids).",
    )
    args = parser.parse_args(argv)

    tm, err = _load_task_manager()
    if tm is None:
        print(f"[warn] {err}; skipping dataset prefetch.")
        return 0

    if args.benchmarks:
        targets = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    else:
        targets = sorted(TASK_MAP)
    print(f"Prefetching {len(targets)} benchmark(s)...")
    loaded, missed = prefetch(tm, targets)
    if missed:
        print(f"[warn] {len(missed)} benchmark(s) not prefetched: {', '.join(missed)}")
    print(
        f"[done] loaded {loaded} task(s). Any not-yet-downloaded datasets will "
        "still download lazily at first evaluation."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
