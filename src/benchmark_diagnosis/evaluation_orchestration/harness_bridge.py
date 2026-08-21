"""Bridge to the EleutherAI lm-evaluation-harness CLI (design doc section 4.1).

We invoke the harness as a subprocess and parse its ``results.json`` rather than
importing its Python API — its API changes often, while the CLI is the stable
contract (ARCHITECTURE.md section 1).

:func:`run_eval` streams the harness output live. In ``live`` mode the noisy
tqdm progress bars (``Requesting API: 99%|...|`` — one line per update when
stdout is a pipe) are filtered out and condensed into a single status line that
rewrites in place on a TTY and ticks periodically when piped, so long evals
never look frozen and per-dataset progress is readable.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from benchmark_diagnosis.data.ingestion import primary_metric, primary_metric_name
from benchmark_diagnosis.evaluation_orchestration.task_registry import to_benchmark_id


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
    timeout: int | None = None,
    confirm_run_unsafe_code: bool = False,
    apply_chat_template: bool = False,
    repeats: int | None = None,
    gen_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """Build the lm-evaluation-harness CLI argv for one evaluation run.

    With ``base_url`` the harness talks to an OpenAI-compatible endpoint via the
    ``local-completions`` model type (e.g. a vLLM deployment or a user-provided
    inference IP); without one, ``model`` is treated as a HuggingFace id and the
    ``hf`` model type is used.

    With ``apply_chat_template`` the harness wraps every prompt with the
    tokenizer's chat template *before* sending it (lm-eval's ``--apply_chat_template``
    machinery: task contexts become a single user message, generations add the
    assistant turn marker). The served model therefore always receives
    chat-formatted text even though the wire protocol stays the raw
    ``/v1/completions`` endpoint — required for chat/RL-trained models that
    degrade on bare prompts. The template comes from the local HF tokenizer
    (``tokenizer`` arg); ``tokenizer_backend=huggingface`` is forced so the
    template is rendered client-side instead of falling back to the remote
    tokenizer (which cannot render it).

    Args:
        model: Model id. Served by the endpoint when ``base_url`` is given,
            otherwise a HuggingFace id.
        tasks: Benchmark task ids passed to ``--tasks`` (comma-joined).
        base_url: OpenAI-compatible endpoint root, or None for a HuggingFace id.
        num_fewshot: Override the registered few-shot count (None => default).
        batch_size: Harness batch size (default 'auto').
        limit: Cap examples per task (debug only; None => unlimited).
        output_dir: Directory the harness writes results.json into.
        log_samples: Append ``--log_samples`` so per-example results are emitted.
        harness_cmd: The harness entrypoint (default 'lm_eval').
        tokenizer: Local HF tokenizer path / id for the ``local-completions``
            model type (the harness otherwise loads a tokenizer named after the
            served model id, which fails for locally-served models).
        max_gen_toks: Max generation tokens for generative tasks (the harness
            default of 256 truncates CoT answers on math-style benchmarks).
        apply_chat_template: Wrap every prompt in the tokenizer's chat template
            before sending. Requires ``tokenizer`` (a local HF tokenizer whose
            chat_template.jinja / tokenizer_config.json provides the template).
        repeats: Sample each generative prompt this many times (the harness
            averages mean-aggregated metrics over the samples). None/1 keeps
            the single greedy pass; requires matching ``gen_kwargs`` for the
            runs to actually differ.
        gen_kwargs: Sampling kwargs forwarded to ``--gen_kwargs`` (e.g.
            ``{"temperature": 0.7, "top_p": 0.95}``).

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
        if apply_chat_template:
            # Force the local HF tokenizer: the remote tokenizer backend cannot
            # render the chat template (it returns the raw message list), which
            # would silently skip templating.
            model_args += ",tokenizer_backend=huggingface"
        if tokenizer:
            model_args += f",tokenizer={tokenizer}"
        if max_gen_toks:
            model_args += f",max_gen_toks={max_gen_toks}"
        if num_concurrent:
            model_args += f",num_concurrent={num_concurrent}"
        if max_length:
            model_args += f",max_length={max_length}"
        if timeout:
            # lm-eval's aiohttp ClientTimeout default is 300s — far too short
            # for slow endpoints (V100 eager decode at ~3.4 tok/s/request
            # under 16-way concurrency: an 8192-token generation needs
            # ~2400s). Without a raised timeout, requests die with
            # TimeoutError, tenacity retries exhaust, and every other
            # in-flight request collapses with "Connector is closed".
            model_args += f",timeout={timeout}"
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
    if apply_chat_template:
        if tokenizer is None:
            raise ValueError(
                "apply_chat_template=True requires a local HF tokenizer "
                "(pass `tokenizer`) — the remote tokenizer backend cannot "
                "render the chat template."
            )
        cmd.append("--apply_chat_template")
    if confirm_run_unsafe_code:
        cmd.append("--confirm_run_unsafe_code")
    if repeats is not None and repeats > 1:
        cmd += ["--repeats", str(repeats)]
    if gen_kwargs:
        # lm-eval parses --gen_kwargs as key=value pairs; join them with
        # commas (e.g. "temperature=0.7,top_p=0.95").
        cmd += ["--gen_kwargs", ",".join(f"{k}={v}" for k, v in gen_kwargs.items())]
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


def run_eval(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    live: bool = False,
    label: str | None = None,
    tick: float = 30.0,
) -> dict:
    """Run the harness command and load its ``results.json``.

    The harness output is streamed to stdout live — lm-eval's per-task
    INFO logs and "Requesting API" progress bars reach the caller instead
    of being hidden until the process exits (long evals otherwise look
    frozen). A bounded output tail is kept for the error report.

    With ``live=True`` the tqdm progress-bar lines are filtered out of the
    stream and condensed into one status line (:class:`_LiveStatus`): on a TTY
    it rewrites a single line in place (updated on every bar tick, at least
    once per ``tick`` seconds), and when piped it emits one line per tick so
    progress still lands in logs. Non-bar lines (lm-eval INFO/ERROR logs) are
    streamed as usual, and the status line is ended cleanly before each of
    them. ``label`` names the running task in the status line.

    Args:
        cmd: argv produced by :func:`build_command`.
        cwd: Working directory for the subprocess (e.g. the repo root).
        live: Filter tqdm progress bars into a single live status line.
        label: Task name shown in the live status line (``live=True`` only).
        tick: Seconds between status refreshes when no bar update arrives.

    Returns:
        The parsed results payload, or ``{}`` if no ``--output_path`` was given
        or the results file is missing.

    Raises:
        RuntimeError: if the subprocess exits non-zero (output tail attached).
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
    )
    assert proc.stdout is not None
    status = _LiveStatus(label or "evaluating", tick=tick) if live else None
    tail: deque[str] = deque(maxlen=200)  # bounded tail for the error report
    try:
        for line in proc.stdout:
            if status is not None:
                progress = _parse_progress_line(line)
                if progress is not None:
                    status.update(progress)
                    continue
                if _is_progress_line(line):
                    continue  # unparseable bar line: drop it, the status shows liveness
                status.before_emit()
            sys.stdout.write(line)
            sys.stdout.flush()
            tail.append(line)
    finally:
        if status is not None:
            status.close()
    rc = proc.wait()
    if rc != 0:
        tail_text = "".join(tail).strip()
        raise RuntimeError(
            f"evaluation failed (exit {rc}): {tail_text[-2000:]}"
        )

    output_dir = _output_dir_from_cmd(cmd)
    if output_dir is None:
        return {}
    return parse_results(output_dir / "results.json")


# --- live status line -------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# tqdm bar lines look like:  Requesting API:  99%|█████████▊| 534/541 [1:22:09<00:56,  8.11s/it]
_PROGRESS_RE = re.compile(
    r"^(?P<desc>.*?):\s*(?P<pct>\d+)%\|.*?\|\s*(?P<cur>\d+)/(?P<total>\d+)"
    r" \[(?P<elapsed>[\d:]+)<(?P<remain>[^,]*),.*?\]\s*$"
)
# Generic "this is a progress bar" hint: percentage + count/range + rate.
_PROGRESS_HINT_RE = re.compile(r"\d+%\|.*\|\s*\d+/\d+ \[.*(?:it/s|s/it)\]")


def fmt_duration(seconds: float) -> str:
    """Format seconds compactly: ``1:22:09`` / ``5:03`` / ``42s``."""
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    if minutes:
        return f"{minutes}:{secs:02d}"
    return f"{secs}s"


def _clean_ansi(line: str) -> str:
    return _ANSI_RE.sub("", line).strip()


def _parse_progress_line(line: str) -> str | None:
    """Parse a tqdm bar line into a compact status string, or None.

    ``"Requesting API:  99%|...| 534/541 [1:22:09<00:56, 8.11s/it]"`` becomes
    ``"Requesting API: 534/541 (99%) · 1:22:09"``.
    """
    clean = _clean_ansi(line)
    match = _PROGRESS_RE.match(clean)
    if match is None:
        return None
    pct = int(match.group("pct"))
    cur = int(match.group("cur"))
    total = int(match.group("total"))
    return (
        f"{match.group('desc')}: {cur}/{total} ({pct}%)"
        f" · {match.group('elapsed')}"
    )


def _is_progress_line(line: str) -> bool:
    """True when ``line`` looks like a tqdm progress bar (rate suffix)."""
    return bool(_PROGRESS_HINT_RE.search(_clean_ansi(line)))


class _LiveStatus:
    """One-line live status for a running harness task.

    A daemon ticker thread refreshes the line every ``tick`` seconds. On a TTY
    the line rewrites in place (``\\r`` + ANSI erase); when piped each refresh
    emits a full line so progress still reaches logs. ``update`` may be called
    from the output reader with freshly parsed bar progress — on a TTY the
    status then redraws immediately (at most once per second). Callers must
    call :meth:`close` once the subprocess ends and :meth:`before_emit` before
    printing any other output line.
    """

    def __init__(self, label: str, tick: float = 30.0) -> None:
        self._label = label
        self._tick = tick if tick > 0 else 30.0
        self._lock = threading.Lock()
        self._progress = ""
        self._started = time.monotonic()
        self._tty = sys.stdout.isatty()
        self._line_open = False
        self._last_render = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, progress: str) -> None:
        """Refresh the status with newly parsed bar progress."""
        with self._lock:
            self._progress = progress
            if self._tty and time.monotonic() - self._last_render >= 1.0:
                self._render_locked()

    def before_emit(self) -> None:
        """End the status line before a real output line is printed."""
        if self._tty and self._line_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._line_open = False

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.before_emit()

    def _loop(self) -> None:
        while not self._stop.wait(self._tick):
            with self._lock:
                self._render_locked()

    def _render_locked(self) -> None:
        elapsed = time.monotonic() - self._started
        text = (
            f"⏳ {self._label} · {self._progress or 'running'}"
            f" · {fmt_duration(elapsed)}"
        )
        self._last_render = time.monotonic()
        if self._tty:
            sys.stdout.write("\r\x1b[2K" + text)
            sys.stdout.flush()
            self._line_open = True
        else:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()


# --- per-task score extraction ----------------------------------------------


def extract_task_results(
    results: dict[str, Any], task: str
) -> list[tuple[str, str, float]]:
    """Scores attributable to one evaluated task, as ``(entry, metric, value)``.

    A single-task run's ``results.json`` may carry more than the task's own
    entry: group tasks (``longbench2``, ``mmmlu``) expand into ``task_*``
    subtask entries, and the registry aliases benchmark ids to task names
    (``math`` -> ``hendrycks_math``). An entry belongs to ``task`` when its
    name equals the task, starts with ``task_``, or maps back to the same
    benchmark id.

    Args:
        results: Parsed lm-evaluation-harness results payload.
        task: The lm-eval task name that was evaluated.

    Returns:
        ``(entry_name, metric_name, score)`` tuples for every attributable
        entry with a computable primary metric, in payload order.
    """
    bid = to_benchmark_id(task)
    out: list[tuple[str, str, float]] = []
    for name, metrics in (results.get("results") or {}).items():
        if not (
            name == task
            or name.startswith(task + "_")
            or to_benchmark_id(name) == bid
        ):
            continue
        metric = primary_metric_name(metrics)
        value = primary_metric(metrics)
        if metric is not None and value is not None:
            out.append((name, metric, value))
    return out


def task_headline(results: dict[str, Any], task: str) -> tuple[str, float] | None:
    """Headline ``(metric, score)`` for one evaluated task, or None.

    Group tasks have no single score in their subtask entries; lm-eval
    aggregates them in the payload's ``groups`` section (weighted by sample
    count), which is preferred when present. Falls back to the arithmetic mean
    over the task's own entries so a group still reports one number.

    Args:
        results: Parsed lm-evaluation-harness results payload.
        task: The lm-eval task name that was evaluated.

    Returns:
        ``(metric_name, score)``, or ``None`` when the task produced no
        attributable, computable score.
    """
    groups = results.get("groups") or {}
    group_metrics = groups.get(task) if isinstance(groups, dict) else None
    if isinstance(group_metrics, dict):
        metric = primary_metric_name(group_metrics)
        value = primary_metric(group_metrics)
        if metric is not None and value is not None:
            return (metric, value)
    entries = extract_task_results(results, task)
    if not entries:
        return None
    mean = sum(value for _, _, value in entries) / len(entries)
    return (entries[0][1], mean)


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
