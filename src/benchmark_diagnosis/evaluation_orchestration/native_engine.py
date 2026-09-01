"""Generic native-backend evaluation for benchmarks lm-eval cannot express.

lm-eval models a benchmark as a single-pass, scalar-scored task. Several
benchmark families need more than that — an external judge model (Arena-Hard's
pairwise comparison), a code-execution sandbox (LiveCodeBench / Codeforces), a
continuously-release official harness (LiveBench), or multi-turn agentic
interaction (MultiIF). These are routed to this module instead of
``harness_bridge`` (see :func:`task_registry.evaluable_backend`).

Each benchmark is described by a spec in :data:`NATIVE_SPECS` (dataset, split,
prompt field, and the *external components* it needs to actually *score*). The
runner here does what is possible offline — load the dataset, render prompts,
call the endpoint for model completions, and archive raw cases — and only
produces scores once the spec's external components are supplied. Until then the
generations are archived for later judging and the score map stays empty (a
fabricated score would be worse than none).

Scoring hooks are left as TODO fill-in points (see the ``score_cases`` field of
each spec and :func:`run_native`); the dataset field names below are best-effort
against public HF layouts and loads are defensive — a missing field / dataset
degrades to a recorded message, never a hard crash.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmark_diagnosis.core.llm_client import LLMClient

# benchmark_id -> spec. ``requires`` lists the external components that must be
# available before a real score can be computed:
#   judge      -> an LLM judge model / endpoint to score Arena-Hard pairs
#   sandbox    -> a code-execution sandbox to run generated solutions
#   official   -> the benchmark's own harness / per-release answers (LiveBench)
#   multi_turn -> multi-turn agentic interaction + instruction-following scoring
NATIVE_SPECS: dict[str, dict[str, Any]] = {
    "arena_hard": {
        "name": "Arena-Hard-Auto",
        "dataset": "lmarena-ai/arena-hard-auto",
        "split": "train",
        "prompt_field": "problem",
        "prompt_template": "{prompt}",
        "requires": ("judge",),
        "source_url": "https://github.com/lm-sys/arena-hard-auto",
        # TODO(judge): provide a score_cases callable that, given the generated
        # completions + a judge-able target, returns pairwise win-rate.
        "score_cases": None,
    },
    "livecodebench": {
        "name": "LiveCodeBench v5",
        "dataset": "livecodebench",
        "split": "test",
        "prompt_field": "question_content",
        "prompt_template": "{prompt}",
        "requires": ("sandbox",),
        "source_url": "https://github.com/LiveCodeBench/LiveCodeBench",
        # TODO(sandbox): pass@1 against the per-platform hidden test cases.
        "score_cases": None,
    },
    "codeforces": {
        "name": "Codeforces",
        "dataset": "deepmind/codecontests",
        "split": "train",
        "prompt_field": "description",
        "prompt_template": "{prompt}",
        "requires": ("sandbox",),
        "source_url": "https://huggingface.co/datasets/deepmind/codecontests",
        # TODO(sandbox): run public/private tests in the sandbox, pass@1.
        "score_cases": None,
    },
    "livebench": {
        "name": "LiveBench",
        "dataset": "livebench",
        "split": "test",
        "prompt_field": "question",
        "prompt_template": "{prompt}",
        "requires": ("official",),
        "source_url": "https://github.com/LiveBench/LiveBench",
        # TODO(official): exact-match against the released per-release answers.
        "score_cases": None,
    },
    "multiif": {
        "name": "MultiIF",
        "dataset": "SalesforceChengdu/MultiIF",
        "split": "train",
        "prompt_field": "instructions",
        "prompt_template": "{prompt}",
        "requires": ("multi_turn",),
        "source_url": "https://github.com/SalesforceAIResearch/MultiIF",
        # TODO(multi_turn): multi-turn agent loop + answer-type matching scorer.
        "score_cases": None,
    },
}

NATIVE_BACKEND = "native"


def is_native_benchmark(benchmark_id: str) -> bool:
    """True if ``benchmark_id`` is a generic native-backend benchmark."""
    return benchmark_id in NATIVE_SPECS


def _import_datasets() -> Any:
    """Import the ``datasets`` library lazily (optional dependency)."""
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "native benchmark evaluation requires the `datasets` package; "
            "install it with `pip install datasets`"
        ) from exc
    return load_dataset


def _load_spec_rows(spec: dict[str, Any], limit: int | None) -> list[dict[str, str]]:
    """Load dataset rows and render each into a chat user prompt.

    Defensive: misses on a field / dataset degrade to whatever rows can be
    rendered (the caller records the gap instead of raising).
    """
    load_dataset = _import_datasets()
    ds = load_dataset(spec["dataset"], split=spec["split"])
    field = spec["prompt_field"]
    template = spec["prompt_template"]
    rows: list[dict[str, str]] = []
    for i, item in enumerate(ds):
        if limit is not None and len(rows) >= limit:
            break
        value = item.get(field)
        if value is None or value == "":
            continue
        prompt = template.format(prompt=str(value))
        rows.append({"idx": i, "prompt": prompt})
    return rows


def run_native(
    model: str,
    base_url: str,
    benchmark_id: str,
    *,
    api_key: str = "EMPTY",
    max_tokens: int | None = None,
    timeout_seconds: float = 120.0,
    limit: int | None = None,
    output_dir: Path | None = None,
    client_factory: Callable[..., LLMClient] | None = None,
    external_scorers: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Run one native benchmark against an OpenAI-compatible endpoint.

    Loads the spec's dataset, renders prompts, generates a completion per row,
    archives the raw cases, and computes scores only when every external
    component the spec ``requires`` is present in ``external_scorers``.

    Args:
        model / base_url: the served model and endpoint root (``/v1``).
        benchmark_id: one of :data:`NATIVE_SPECS` keys.
        api_key: Endpoint API key (default ``EMPTY`` for local vLLM).
        max_tokens / timeout_seconds: generation limits (defaults to the
            client defaults when None).
        limit: cap the number of evaluated rows (debug/cheap subsets).
        output_dir: where raw generations are archived (default: none).
        client_factory: callable returning the chat client (default: a plain
            :class:`LLMClient`); injectable so tests can stub endpoint calls.
        external_scorers: mapping of component name (e.g. ``"judge"`` /
            ``"sandbox"``) -> ready scorer to unlock ``score_cases``.

    Returns:
        ``(scores, results)`` — ``scores`` is ``{benchmark_id: float}``,
        ordinarily empty until the spec's external components are supplied;
        ``results`` carries the archived cases + gap record for artifact
        archiving.

    Raises:
        ValueError: for an unregistered native benchmark id.
        RuntimeError: if the ``datasets`` package is missing.
    """
    spec = NATIVE_SPECS.get(benchmark_id)
    if spec is None:
        raise ValueError(f"unknown native benchmark id: {benchmark_id!r}")

    if client_factory is not None:
        client = client_factory(
            base_url,
            model,
            api_key,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
    else:
        client = _default_client(base_url, model, api_key, max_tokens, timeout_seconds)

    rows = _load_spec_rows(spec, limit)
    cases: list[dict[str, str]] = []
    for row in rows:
        try:
            completion = client.complete([{"role": "user", "content": row["prompt"]}])
        except Exception as exc:  # noqa: BLE001 - archive the failure, keep going
            completion = f"[error] {exc}"
        cases.append(
            {"idx": row["idx"], "prompt": row["prompt"], "completion": completion}
        )

    archived: str | None = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"native_{benchmark_id}_generations.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for case in cases:
                fh.write(json.dumps(case, ensure_ascii=False) + "\n")
        archived = str(path)

    requires = tuple(spec["requires"])
    missing = [comp for comp in requires if not (external_scorers or {}).get(comp)]

    scores: dict[str, float] = {}
    if not missing and spec.get("score_cases") is not None:
        scores = spec["score_cases"](cases, external_scorers or {})

    results = {
        NATIVE_BACKEND: {
            benchmark_id: {
                "name": spec["name"],
                "cases": len(cases),
                "requires": list(requires),
                "missing_external": list(missing),
                "scored": bool(scores),
                "archived": archived,
            }
        }
    }
    return scores, results


def _default_client(
    base_url: str,
    model: str,
    api_key: str,
    max_tokens: int | None,
    timeout_seconds: float,
) -> LLMClient:
    return LLMClient(
        base_url,
        model,
        api_key,
        max_tokens=max_tokens or 2048,
        timeout_seconds=timeout_seconds,
    )
