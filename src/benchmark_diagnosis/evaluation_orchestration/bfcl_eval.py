"""Native BFCL (Berkeley Function Calling Leaderboard) evaluation backend.

BFCL (Gorilla project, UC Berkeley) measures **function/tool calling**: given a
request and a set of typed tool schemas, the model must decide which tool to
call (or to decline) and emit the correct call — scored by AST matching and
(multi-turn) executable outcomes. It is a multi-turn agentic benchmark with its
own harness; it does **not** map onto lm-eval's single-pass task model, so the
tool's eval path routes a BFCL-only benchmark id (``bfcl``) here instead of
``harness_bridge.build_command`` / ``run_eval``.

This module wraps the official ``bfcl-eval`` PyPI distribution (the Gorilla
harness, https://github.com/ShishirPatil/gorilla; ``pip install '.[bfcl]'``).
The package is imported lazily so this repo works without it until a BFCL
benchmark is actually requested, and results are normalized into the tool's
``{benchmark_id: score}`` convention — ``bfcl`` = overall accuracy plus one
``bfcl.<category>`` entry per reported sub-category.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Benchmark id (tool namespace) evaluated by this backend. Registered in the
# seed/lookup layer and routed to ``evaluable_backend == "bfcl"``.
BFCL_BENCHMARK_ID = "bfcl"


def is_bfcl_benchmark(benchmark_id: str) -> bool:
    """True if ``benchmark_id`` is a BFCL-only benchmark (handled by this backend)."""
    return benchmark_id == BFCL_BENCHMARK_ID


def _import_bfcl() -> Any:
    """Import and return the official ``bfcl_eval.handler`` module lazily."""
    try:
        from bfcl_eval import handler  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "BFCL evaluation requested but the official `bfcl-eval` harness is "
            "not installed. Install it with `pip install -e '.[bfcl]'` (or "
            "`pip install bfcl-eval` from https://pypi.org/project/bfcl-eval/)."
        ) from exc
    return handler


def run_bfcl(
    model: str,
    base_url: str,
    *,
    categories: list[str] | None = None,
    model_type: str = "vllm",
    output_dir: str | None = None,
    api_key: str = "EMPTY",
) -> dict[str, float]:
    """Run BFCL against an OpenAI-compatible endpoint and return tool-namespace scores.

    Args:
        model: Served model name (the value reported on the endpoint).
        base_url: OpenAI-compatible endpoint root (e.g. ``http://host:8000/v1``),
            i.e. the vLLM/SGLang server this tool deploys.
        categories: Optional subset of BFCL categories to evaluate (None => the
            harness's default set; category ids are version-specific).
        model_type: How the harness talks to the endpoint — ``vllm`` (default,
            matches this tool's vLLM deployment) or ``openai``.
        output_dir: Where the harness writes per-response results (default: none).
        api_key: Endpoint API key (default ``EMPTY`` for local vLLM).

    Returns:
        ``{benchmark_id: score}`` mapped into the tool namespace: overall
        accuracy under ``bfcl`` plus one ``bfcl.<category>`` per sub-category.

    Raises:
        RuntimeError: if the official ``bfcl-eval`` package is not installed.
    """
    handler = _import_bfcl()
    kwargs: dict[str, Any] = {
        "model": model,
        "model_type": model_type,
        "api_key": api_key,
        "test_categories": categories,
        "base_url": base_url,
    }
    if output_dir:
        kwargs["output_dir"] = output_dir

    # ``base_url`` (pointing at a pre-existing OpenAI-compatible server) was
    # added to the harness in later releases; older installs only accept the
    # core args. Back off to those rather than failing the whole run when the
    # harness rejects an unexpected keyword (keyword-arg mismatches only).
    try:
        result = handler.run_inference_and_eval(**kwargs)
    except TypeError as exc:
        msg = str(exc)
        if "unexpected keyword argument" not in msg:
            raise
        for key in set(kwargs) - {"model", "model_type", "api_key", "test_categories"}:
            kwargs.pop(key, None)
        result = handler.run_inference_and_eval(**kwargs)
    return categorize_scores(result)


def categorize_scores(result: Mapping[str, Any]) -> dict[str, float]:
    """Flatten a BFCL harness result dict into ``{benchmark_id: score}``.

    The harness result is version-specific; this reader tolerates a missing
    overall key (``ast_acc``) and skips malformed category rows. Never raises —
    missing values simply do not produce a score entry.
    """
    scores: dict[str, float] = {}
    overall = result.get("ast_acc") if isinstance(result, Mapping) else None
    if overall is None:
        overall = result.get("overall_acc") if isinstance(result, Mapping) else None
    if overall is not None:
        try:
            scores[BFCL_BENCHMARK_ID] = float(overall)
        except (TypeError, ValueError):
            pass
    for entry in (result.get("other_categories") if isinstance(result, Mapping) else None) or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        name, metric = entry[0], entry[1]
        if name is None or not isinstance(metric, (int, float)):
            continue
        scores[f"{BFCL_BENCHMARK_ID}.{name}"] = float(metric)
    return scores