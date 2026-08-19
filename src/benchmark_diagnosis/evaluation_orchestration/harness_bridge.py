"""Bridge to the EleutherAI lm-evaluation-harness CLI (design doc section 4.1).

We invoke the harness as a subprocess and parse its ``results.json`` rather than
importing its Python API — its API changes often, while the CLI is the stable
contract (ARCHITECTURE.md section 1).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from benchmark_diagnosis.data.ingestion import primary_metric


def build_command(
    model: str,
    tasks: list[str],
    *,
    base_url: str | None = None,
    num_fewshot: int | None = None,
    batch_size: str = "auto",
    limit: int | None = None,
    output_dir: str | Path | None = None,
    log_samples: bool = True,
    harness_cmd: str = "lm_eval",
    tokenizer: str | None = None,
    max_gen_toks: int | None = None,
    num_concurrent: int | None = None,
    max_length: int | None = None,
    confirm_run_unsafe_code: bool = False,
) -> list[str]:
    """Build the lm-evaluation-harness CLI argv for one evaluation run.

    With ``base_url`` the harness talks to an OpenAI-compatible endpoint via the
    ``local-completions`` model type (e.g. a vLLM deployment or a user-provided
    inference IP); without one, ``model`` is treated as a HuggingFace id and the
    ``hf`` model type is used.

    Args:
        model: Model id. Served by the endpoint when ``base_url`` is given,
            otherwise a HuggingFace id.
        tasks: Benchmark task ids passed to ``--tasks`` (comma-joined).
        base_url: OpenAI-compatible endpoint root, or None for a HuggingFace id.
        num_fewshot: Override the registered few-shot count (None => default).
        batch_size: Harness batch size (default ``"auto"``).
        limit: Cap examples per task (debug only; None => unlimited).
        output_dir: Directory the harness writes ``results.json`` into.
        log_samples: Append ``--log_samples`` so per-example results are emitted.
        harness_cmd: The harness entrypoint (default ``"lm_eval"``).
        tokenizer: Local HF tokenizer path / id for the ``local-completions``
            model type (the harness otherwise loads a tokenizer named after the
            served model id, which fails for locally-served models).
        max_gen_toks: Max generation tokens for generative tasks (the harness
            default of 256 truncates CoT answers on math-style benchmarks).

    Returns:
        The full argv list ready for :func:`subprocess.run`.
    """
    if base_url is not None:
        model_type = "local-completions"
        # lm-eval 0.4.12's local-completions model type POSTs the payload
        # directly to ``base_url`` (no path appended), so pass the full
        # completions endpoint even though the CLI convention is a /v1 root.
        api_base = base_url.rstrip("/")
        if not api_base.endswith("/completions"):
            api_base += "/completions"
        model_args = f"model={model},base_url={api_base}"
        if tokenizer:
            model_args += f",tokenizer={tokenizer}"
        if max_gen_toks:
            model_args += f",max_gen_toks={max_gen_toks}"
        if num_concurrent:
            model_args += f",num_concurrent={num_concurrent}"
        if max_length:
            model_args += f",max_length={max_length}"
    else:
        model_type = "hf"
        model_args = f"pretrained={model}"

    cmd: list[str] = [
        harness_cmd,
        "--model",
        model_type,
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
    ]
    if confirm_run_unsafe_code:
        cmd.append("--confirm_run_unsafe_code")
    if log_samples:
        cmd.append("--log_samples")
    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]
    if batch_size:
        cmd += ["--batch_size", batch_size]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if output_dir is not None:
        cmd += ["--output_path", str(output_dir)]
    return cmd


def run_eval(cmd: list[str], *, cwd: str | Path | None = None) -> dict:
    """Run the harness command and load its ``results.json``.

    Args:
        cmd: argv produced by :func:`build_command`.
        cwd: Working directory for the subprocess (e.g. the repo root).

    Returns:
        The parsed results payload, or ``{}`` if no ``--output_path`` was given
        or the results file is missing.

    Raises:
        RuntimeError: if the subprocess exits non-zero (stderr tail attached).
    """
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            f"evaluation failed (exit {exc.returncode}): {tail[-2000:]}"
        ) from exc

    output_dir = _output_dir_from_cmd(cmd)
    if output_dir is None:
        return {}
    return parse_results(output_dir / "results.json")


def parse_results(path: str | Path) -> dict:
    """Load a harness ``results.json`` file into a dict.

    lm-eval >= 0.4.12 writes results into a per-model subdirectory
    (``<output_dir>/<model>/results_<timestamp>.json``) instead of
    ``<output_dir>/results.json``; when the direct file is missing we fall
    back to the newest ``results_*.json`` anywhere under the output dir.

    Args:
        path: Path to a ``results.json`` produced by lm-evaluation-harness.

    Returns:
        The parsed payload as a dict, or ``{}`` when the file is missing or
        cannot be read/decoded.
    """
    p = Path(path)
    if not p.exists():
        candidates = sorted(
            p.parent.glob("*/results_*.json"), key=lambda f: f.stat().st_mtime
        )
        if not candidates:
            return {}
        p = candidates[-1]
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def extract_scores(results: dict[str, Any]) -> dict[str, float]:
    """Flatten a harness ``results`` dict into ``{task: headline_score}``.

    For each task the most stringent available metric wins (``exact_match`` >
    ``pass@1`` > ``acc`` > ``f1``), matching :func:`ingestion.primary_metric`.

    Args:
        results: Parsed lm-evaluation-harness results payload (``results`` key).

    Returns:
        Mapping ``task_id -> score`` for every task with a computable metric.
    """
    out: dict[str, float] = {}
    for task, metrics in (results.get("results") or {}).items():
        value = primary_metric(metrics)
        if value is not None:
            out[task] = value
    return out


def _output_dir_from_cmd(cmd: list[str]) -> Path | None:
    """Extract the ``--output_path`` value from a harness argv, if present."""
    try:
        idx = cmd.index("--output_path")
    except ValueError:
        return None
    if idx + 1 < len(cmd):
        return Path(cmd[idx + 1])
    return None
