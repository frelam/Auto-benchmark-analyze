"""End-to-end run orchestration: one command from model source to report.

The runner is the single entry point behind the ``run`` CLI command. It owns
the full lifecycle:

* ensure offline assets exist (ingest seed / build coverage + portfolio +
  curves);
* resolve where the model comes from — auto-deploy weights with vLLM, reuse an
  existing OpenAI-compatible endpoint, or read a JSON scores file (no eval);
* run lm-evaluation-harness and flatten results to scores;
* diagnose the scores (analyze vs full mode; LLM+rules vs rules advisor);
* render charts, write ``metrics.json`` + a Markdown report;
* tear down any server we launched.

Diagnosis always runs the unified stages 1-7 intelligent pipeline (preceded by
a Stage 0 cluster-verdict aggregation). The subprocess boundaries (deploy,
readiness probe, harness run) are dependency injectable so tests can drive the
whole flow without vLLM or lm_eval installed.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.table import Table
from sqlalchemy import select
from sqlalchemy.orm import Session

from benchmark_diagnosis.config import Settings, resolve_advisor_mode
from benchmark_diagnosis.core import db
from benchmark_diagnosis.core.schema import ModelRecord
from benchmark_diagnosis.data import ingestion
from benchmark_diagnosis.evaluation_orchestration.deploy import (
    serve_command,
    wait_until_ready,
)
from benchmark_diagnosis.evaluation_orchestration.harness_bridge import (
    build_command,
    extract_scores,
    run_eval,
)
from benchmark_diagnosis.pipeline import build_offline, diagnose_model
from benchmark_diagnosis.reporting.report_generator import render_json, render_markdown

console = Console()


@dataclass
class RunRequest:
    """A fully-resolved end-to-end run (CLI args merged over the ``run`` profile).

    ``source`` selects the model source: ``weights`` (auto-deploy via vLLM),
    ``endpoint`` (existing inference service IP) or ``scores`` (JSON file, no
    eval). ``mode`` picks analyze-only vs analysis + recommendations.
    """

    model: str | None = None
    mode: str = "full"
    source: Literal["weights", "endpoint", "scores"] | None = None
    weights: str | None = None
    base_url: str | None = None
    scores_file: Path | None = None
    arch: str | None = None
    params: float | None = None
    release_date: str | None = None
    advisor_mode: str = "auto"
    output: Path | None = None


@dataclass
class RunResult:
    """Outcome of :func:`execute_run` — the report plus written artifacts."""

    report: dict[str, Any]
    report_path: Path
    metrics_path: Path
    figure_paths: list[Path]
    mode: str
    advisor_mode: str
    served: bool


def build_run_request(
    settings: Settings,
    *,
    model: str | None = None,
    model_id: str | None = None,
    base_url: str | None = None,
    scores: str | Path | None = None,
    mode: str | None = None,
    advisor_mode: str | None = None,
    arch: str | None = None,
    params: float | None = None,
    release_date: str | None = None,
    output: str | Path | None = None,
) -> RunRequest:
    """Merge CLI arguments over the config's ``run`` profile into a run request.

    Precedence is **CLI argument > ``settings.run.model.*``**. The model source
    is derived from whichever of scores / base_url / weights is present; passing
    more than one on the CLI, or configuring more than one source field, raises
    ``ValueError`` (the sources are mutually exclusive).

    Args:
        settings: Resolved settings.
        model: Model name reported in the output (defaults to the weights id for
            a weights run, else ``run.model.name``).
        model_id: HuggingFace weights id / local path (source ``weights``).
        base_url: Existing OpenAI-compatible endpoint (source ``endpoint``).
        scores: JSON ``{benchmark_id: score}`` file (source ``scores``; no eval).
        mode: ``"analyze"`` or ``"full"`` (default ``run.mode``).
        advisor_mode: ``auto`` / ``llm_rules`` / ``rules`` (default
            ``recommendation.advisor_mode``).
        arch / params / release_date: New-model metadata.
        output: Report output path (defaults to ``run.output.dir/report.md``).

    Returns:
        A fully-resolved :class:`RunRequest`.

    Raises:
        ValueError: on an ambiguous or missing model source, or an invalid mode.
    """
    cfg = settings.run.model

    cli_provided = [
        kind
        for kind, value in (
            ("scores", scores is not None),
            ("endpoint", base_url is not None),
            ("weights", model_id is not None),
        )
        if value
    ]
    if len(cli_provided) > 1:
        raise ValueError(
            "ambiguous model source: provide exactly one of "
            "--scores / --base-url / --model-id"
        )

    if cli_provided:
        source: str | None = cli_provided[0]
        resolved_scores = Path(scores) if scores is not None else None
        resolved_base_url = base_url
        resolved_weights = model_id
    else:
        cfg_provided = [
            kind
            for kind, value in (
                ("scores", cfg.scores_file is not None),
                ("endpoint", cfg.base_url is not None),
                ("weights", cfg.weights is not None),
            )
            if value
        ]
        if len(cfg_provided) > 1:
            raise ValueError(f"ambiguous run.model source in config: {cfg_provided}")
        source = cfg_provided[0] if cfg_provided else cfg.source
        resolved_scores = Path(cfg.scores_file) if cfg.scores_file else None
        resolved_base_url = cfg.base_url
        resolved_weights = cfg.weights

    if source is None:
        raise ValueError(
            "no model source: pass --scores / --base-url / --model-id, "
            "or set run.model.source in the config"
        )
    if source == "scores" and resolved_scores is None:
        raise ValueError(
            "source=scores requires a scores file (--scores or run.model.scores_file)"
        )
    if source == "endpoint" and resolved_base_url is None:
        raise ValueError(
            "source=endpoint requires a base_url (--base-url or run.model.base_url)"
        )
    if source == "weights" and resolved_weights is None:
        raise ValueError(
            "source=weights requires weights (--model-id or run.model.weights)"
        )

    resolved_mode = mode or settings.run.mode
    if resolved_mode not in ("analyze", "full"):
        raise ValueError(f"unknown mode {resolved_mode!r}; expected analyze|full")
    if advisor_mode not in (None, "auto", "llm_rules", "rules"):
        raise ValueError(
            f"unknown advisor_mode {advisor_mode!r}; expected auto|llm_rules|rules"
        )

    resolved_model = model or cfg.name
    if resolved_model is None and source == "weights":
        resolved_model = resolved_weights
    if resolved_model is None:
        raise ValueError("a model name is required: pass --model (or set run.model.name)")

    return RunRequest(
        model=resolved_model,
        mode=resolved_mode,
        source=source,
        weights=resolved_weights,
        base_url=resolved_base_url,
        scores_file=resolved_scores,
        arch=arch or cfg.arch,
        params=params if params is not None else cfg.params,
        release_date=release_date or cfg.release_date,
        advisor_mode=advisor_mode or settings.recommendation.advisor_mode,
        output=Path(output) if output is not None else None,
    )


def execute_run(
    settings: Settings,
    request: RunRequest,
    *,
    deploy_weights: Callable[[list[str]], Any] | None = None,
    wait_ready: Callable[[str], bool] | None = None,
    run_harness: Callable[[list[str]], dict[str, Any]] | None = None,
) -> RunResult:
    """Run one end-to-end diagnosis and write report + metrics + figures.

    Args:
        settings: Resolved settings (offline assets are ensured against this DB).
        request: The resolved run request.
        deploy_weights: Callable launching a served model from ``vllm serve``
            argv; must return a subprocess-like handle with ``terminate()`` /
            ``wait()`` (default: :func:`subprocess.Popen`).
        wait_ready: Callable polling an OpenAI-compatible ``/v1`` endpoint until
            it answers (default: :func:`deploy.wait_until_ready`).
        run_harness: Callable running the harness argv and returning parsed
            results (default: :func:`harness_bridge.run_eval`).

    Returns:
        A :class:`RunResult` with the report dict and written artifact paths.

    Raises:
        ValueError: if ``request`` has an invalid mode / advisor mode.
        RuntimeError: if a weights deployment never becomes ready.
    """
    mode = request.mode
    if mode not in ("analyze", "full"):
        raise ValueError(f"unknown mode {mode!r}; expected analyze|full")
    advisor_mode = resolve_advisor_mode(settings, request.advisor_mode)

    deploy_weights = deploy_weights or _launch_server
    wait_ready = wait_ready or wait_until_ready
    run_harness = run_harness or run_eval

    session = _open_session(settings)
    proc: Any = None
    base_url: str | None = None
    try:
        ensure_offline(session, settings)
        portfolio_ids = set(portfolio_benchmarks(session))

        source = request.source
        if source == "weights":
            base_url = f"http://{settings.serving.host}:{settings.serving.port}/v1"
            cmd = serve_command(request.weights, settings.serving)
            console.print("[cyan]Deploying weights...[/] " + " ".join(cmd))
            proc = deploy_weights(cmd)
            if not wait_ready(base_url):
                raise RuntimeError(f"served model did not become ready at {base_url}")
        elif source == "endpoint":
            base_url = request.base_url
        elif source == "scores":
            pass
        else:
            raise ValueError(
                "no model source in request; expected weights|endpoint|scores"
            )

        raw_scores = _collect_scores(request, base_url, portfolio_ids, run_harness, settings)

        if raw_scores and not (portfolio_ids & set(raw_scores)):
            console.print(
                "[yellow]Warning: none of the scores overlap the representative "
                "portfolio; cluster analysis will be empty. Overlap needs one of: "
                f"{sorted(portfolio_ids)}[/]"
            )

        model_record = get_or_build_model(
            session, request.model, request.arch, request.params, request.release_date
        )
        report = diagnose_model(
            session,
            model_record,
            raw_scores,
            settings,
            mode=mode,
            advisor_mode=advisor_mode,
        )
        report["scores"] = raw_scores

        # Artifacts land next to the report so the Markdown figure links hold
        # wherever the report is written (config run.output.dir when no --output).
        report_path = request.output or Path(settings.run.output.dir) / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        figure_paths = _render_figures(
            report, raw_scores, report_path.parent / "figures", portfolio_ids
        )
        report["figures"] = [_relpath(f, report_path.parent) for f in figure_paths]

        metrics_path = report_path.parent / "metrics.json"
        metrics_path.write_text(render_json(report), encoding="utf-8")
        report_path.write_text(render_markdown(report), encoding="utf-8")

        console.print(
            f"[green]Report:[/] {report_path}\n"
            f"[green]Metrics:[/] {metrics_path}\n"
            f"[green]Figures:[/] {len(figure_paths)} written"
        )
        return RunResult(
            report=report,
            report_path=report_path,
            metrics_path=metrics_path,
            figure_paths=figure_paths,
            mode=mode,
            advisor_mode=advisor_mode,
            served=source == "weights",
        )
    finally:
        session.close()
        if proc is not None:
            console.print("[yellow]Stopping server...[/]")
            _terminate(proc)


def ensure_offline(session: Session, settings: Settings) -> None:
    """Idempotently build seed data + offline assets (coverage/portfolio/curves)."""
    if db.load_latest_asset(session, "portfolio") is None:
        if session.scalar(select(ModelRecord.model_id).limit(1)) is None:
            ingestion.load_seed(session)
        console.print("[cyan]Building offline assets (coverage/portfolio/curves)...[/]")
        build_offline(session, settings)


def portfolio_benchmarks(session: Session) -> list[str]:
    """Benchmark ids referenced by the latest representative portfolios."""
    portfolios = db.load_latest_asset(session, "portfolio") or []
    benchmarks: set[str] = set()
    for pf in portfolios:
        for b in pf.get("benchmarks", []):
            benchmarks.add(b["benchmark_id"])
    return sorted(benchmarks)


def get_or_build_model(
    session: Session,
    model_id: str,
    arch: str | None,
    params: float | None,
    release_date: str | None,
) -> ModelRecord:
    """Return the registered model record, or a transient one for a new model."""
    existing = session.get(ModelRecord, model_id)
    if existing is not None:
        return existing
    return ModelRecord(
        model_id=model_id,
        name=model_id,
        arch_type=arch or "dense",
        total_params=params,
        active_params=params,
        release_date=dt.date.fromisoformat(release_date) if release_date else None,
    )


def _collect_scores(
    request: RunRequest,
    base_url: str | None,
    portfolio_ids: set[str],
    run_harness: Callable[[list[str]], dict[str, Any]],
    settings: Settings,
) -> dict[str, float]:
    """Load scores from a file, or evaluate the portfolio and flatten the results."""
    if request.source == "scores":
        scores_path = request.scores_file
        payload = json.loads(scores_path.read_text(encoding="utf-8"))
        raw: dict[str, float] = {}
        for key, value in payload.items():
            if key.startswith("_"):
                continue  # metadata convention (e.g. _note), not a benchmark score
            try:
                raw[key] = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"invalid score for '{key}' in {scores_path}: {value!r} "
                    "(expected a number)"
                ) from None
        console.print(f"[green]Loaded {len(raw)} score(s) from {scores_path}[/]")
        return raw

    tasks = sorted(portfolio_ids)
    if not tasks:
        console.print("[yellow]No portfolio benchmarks to evaluate; skipping eval.[/]")
        return {}
    cmd = build_command(
        request.model,
        tasks,
        base_url=base_url,
        num_fewshot=settings.evaluation.num_fewshot,
        batch_size=settings.evaluation.batch_size,
        limit=settings.evaluation.limit,
        output_dir=Path(settings.evaluation.output_dir),
        harness_cmd=settings.evaluation.harness_cmd,
    )
    console.print("[cyan]Evaluating...[/] " + " ".join(cmd))
    raw_scores = extract_scores(run_harness(cmd))
    _print_scores(raw_scores)
    return raw_scores


def _render_figures(
    report: dict[str, Any],
    raw_scores: dict[str, float],
    fig_dir: Path,
    portfolio_ids: set[str],
) -> list[Path]:
    """Render the model scores / clusters / gap charts, degrading without matplotlib."""
    try:
        from benchmark_diagnosis.reporting import visualize
    except (ImportError, RuntimeError):
        console.print(
            "[yellow]matplotlib not installed; skipping charts. "
            "Install with `pip install -e '.[plot]'`.[/]"
        )
        return []
    return visualize.render_model_report(report, raw_scores, fig_dir, portfolio_ids=portfolio_ids)


def _open_session(settings: Settings) -> Session:
    engine = db.make_engine(settings.storage.db_path)
    db.init_db(engine)
    factory = db.session_factory(engine)
    return factory()


def _launch_server(cmd: list[str]) -> subprocess.Popen:
    """Launch a serving subprocess (the default ``deploy_weights``)."""
    return subprocess.Popen(cmd)


def _terminate(proc: Any) -> None:
    """Best-effort termination for a launched server (already exited is fine)."""
    try:
        proc.terminate()
        proc.wait()
    except Exception:
        pass


def _relpath(path: Path, anchor: Path) -> str:
    """Return ``path`` relative to ``anchor`` (POSIX separators for Markdown)."""
    try:
        return path.resolve().relative_to(anchor.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _print_scores(scores: dict[str, float]) -> None:
    if not scores:
        return
    table = Table(title="Evaluation scores")
    table.add_column("task", style="cyan")
    table.add_column("score", justify="right")
    for task, score in sorted(scores.items()):
        table.add_row(task, f"{score:.4f}")
    console.print(table)
