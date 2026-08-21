"""Command-line interface for benchmark-diagnosis.

Commands take input in exactly two forms — a YAML config (``--config``) or CLI
arguments after the command. The ``run`` command funnels through
:mod:`benchmark_diagnosis.runner`, which auto-deploys model weights or reuses
an inference endpoint, evaluates, diagnoses, and writes report + metrics +
figures in one shot. Diagnosis always runs the unified stages 1-7 intelligent
pipeline (preceded by a Stage 0 cluster-verdict aggregation).
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from benchmark_diagnosis.config import Settings, load_config
from benchmark_diagnosis.core import db
from benchmark_diagnosis.core.schema import ModelRecord
from benchmark_diagnosis.data import ingestion, queries
from benchmark_diagnosis.evaluation_orchestration.deploy import (
    serve_command,
    wait_until_ready,
)
from benchmark_diagnosis.evaluation_orchestration.harness_bridge import (
    build_command,
    extract_scores,
    run_eval,
)
from benchmark_diagnosis.pipeline import build_offline
from benchmark_diagnosis.reporting import visualize as viz
from benchmark_diagnosis.reporting.report_generator import render_results_summary
from benchmark_diagnosis.runner import (
    RunResult,
    build_run_request,
    ensure_offline,
    execute_run,
    portfolio_benchmarks,
)

app = typer.Typer(help="One-stop LLM evaluation, low-score diagnosis, and advice.")
console = Console()


@app.callback()
def main(
    ctx: typer.Context,
    config: Path = typer.Option(None, "--config", help="Path to a YAML config overriding defaults."),
    db_path: str = typer.Option(None, "--db", help="Override the SQLite DB path."),
) -> None:
    """benchmark-diagnosis CLI (config deep-merges over config/default.yaml)."""
    settings = load_config(config)
    if db_path:
        settings.storage.db_path = db_path
    ctx.obj = settings


def _session(settings: Settings) -> tuple[Session, Engine]:
    engine = db.make_engine(settings.storage.db_path)
    db.init_db(engine)
    factory = db.session_factory(engine)
    return factory(), engine


def _ensure_offline(settings: Settings) -> None:
    """Make sure seed data + offline assets exist, building them if missing."""
    session, _ = _session(settings)
    try:
        ensure_offline(session, settings)
    finally:
        session.close()


def _run_or_exit(settings: Settings, **kwargs: Any) -> RunResult:
    """Merge CLI args into a request and execute it, surfacing errors cleanly."""
    try:
        request = build_run_request(settings, **kwargs)
        return execute_run(settings, request)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc


def _print_result(result: RunResult) -> None:
    if result.mode == "benchmark":
        scores = result.report.get("scores", {}) or {}
        console.print(f"[green]Scores written to {result.report_path}[/]")
        console.print(
            f"[cyan]{len(scores)} benchmark(s) scored[/] (mode=benchmark, "
            f"served={result.served})"
        )
        return
    clusters = result.report.get("clusters", [])
    n_under = sum(1 for c in clusters if c.get("underperforming"))
    console.print(f"[green]Report written to {result.report_path}[/]")
    console.print(f"[green]Metrics written to {result.metrics_path}[/]")
    console.print(
        f"[cyan]{len(clusters)} cluster(s) analysed, {n_under} under-performing[/] "
        f"(mode={result.mode}, advisor={result.advisor_mode}, "
        f"figures={len(result.figure_paths)})"
    )


@app.command()
def ingest(
    ctx: typer.Context,
    seed: bool = typer.Option(False, "--seed", help="Load the packaged seed reference data."),
    file: Path = typer.Option(None, "--file", help="JSON file with benchmarks/models/scores."),
) -> None:
    """Ingest models / benchmarks / scores into the store."""
    settings: Settings = ctx.obj
    session, _ = _session(settings)
    try:
        if seed:
            counts = ingestion.load_seed(session)
        elif file is not None:
            payload = json.loads(file.read_text(encoding="utf-8"))
            counts = ingestion.ingest_json(session, payload)
        else:
            console.print("[red]Provide --seed or --file.[/]")
            raise typer.Exit(code=1)
        console.print(f"[green]Ingested:[/] {counts}")
    finally:
        session.close()


@app.command(name="build-offline")
def build_offline_cmd(ctx: typer.Context) -> None:
    """Build versioned offline assets: coverage table, portfolio, curves."""
    settings: Settings = ctx.obj
    session, _ = _session(settings)
    try:
        if session.scalar(select(ModelRecord.model_id).limit(1)) is None:
            ingestion.load_seed(session)
        versions = build_offline(session, settings)
        console.print("[green]Offline assets built:[/]")
        for k, v in versions.items():
            console.print(f"  {k}: {v}")
    finally:
        session.close()


@app.command()
def visualize(
    ctx: typer.Context,
    out: Path = typer.Option(
        Path("results"), "--out", help="Directory for figures + JSON archive."
    ),
) -> None:
    """Render the latest offline assets as charts and archive them (JSON + summary)."""
    settings: Settings = ctx.obj
    _ensure_offline(settings)

    session, _ = _session(settings)
    try:
        coverage = db.load_latest_asset(session, "coverage") or []
        portfolios = db.load_latest_asset(session, "portfolio") or []
        curves = db.load_latest_asset(session, "curves") or []
        scores = queries.scores_matrix(session)
        benchmark_names = {
            b.benchmark_id: b.name for b in queries.list_benchmarks(session)
        }
        versions = {
            "coverage_version": db.latest_version_id(session, "coverage"),
            "portfolio_version": db.latest_version_id(session, "portfolio"),
            "curves_version": db.latest_version_id(session, "curves"),
        }
    finally:
        session.close()

    fig_dir = out / "figures"
    fig_paths: list[Path] = []
    fig_paths += viz.render_curves(curves, fig_dir, benchmark_names)
    fig_paths += viz.render_frontier_overview(curves, fig_dir)
    fig_paths += viz.render_coverage_profile(coverage, fig_dir)
    fig_paths += viz.render_coverage_metrics(coverage, fig_dir)
    fig_paths += viz.render_cluster_map(coverage, fig_dir, benchmark_names)
    fig_paths += viz.render_correlation(scores, fig_dir)

    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("coverage", coverage), ("portfolio", portfolios),
                       ("curves", curves)):
        (out / f"{name}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    figure_names = [p.name for p in fig_paths]
    summary = render_results_summary(
        coverage=coverage,
        portfolios=portfolios,
        curves=curves,
        scores=scores,
        benchmark_names=benchmark_names,
        versions=versions,
        figure_names=figure_names,
        generated_at=dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )
    (out / "README.md").write_text(summary, encoding="utf-8")

    console.print(f"[green]Wrote {len(fig_paths)} figure(s) + archive to {out}/[/]")


@app.command()
def deploy(
    ctx: typer.Context,
    model_id: str = typer.Argument(..., help="HuggingFace model id or local weights path."),
    launch: bool = typer.Option(False, "--launch", help="Actually launch the server and wait."),
) -> None:
    """Print (or --launch) the serving command for a model's weights."""
    settings: Settings = ctx.obj
    cmd = serve_command(model_id, settings.serving)
    console.print("[cyan]Serving command:[/]")
    console.print("  " + " ".join(cmd))
    if not launch:
        return
    base_url = f"http://{settings.serving.host}:{settings.serving.port}/v1"
    console.print(f"[cyan]Launching {model_id} ...[/]")
    proc = subprocess.Popen(cmd)
    try:
        if wait_until_ready(base_url):
            console.print(f"[green]Ready at {base_url}[/]")
            console.print("[dim]Press Ctrl+C to stop the server.[/]")
            proc.wait()
        else:
            console.print("[red]Server did not become ready in time.[/]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping server...[/]")
        proc.terminate()
        proc.wait()


@app.command()
def eval_model(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Model name sent to the API."),
    base_url: str = typer.Option(..., "--base-url", help="OpenAI-compatible /v1 endpoint."),
    tasks: str = typer.Option(None, "--tasks", help="Comma-separated task list (default: portfolio)."),
) -> None:
    """Evaluate a served model against a benchmark list."""
    settings: Settings = ctx.obj
    if tasks is None:
        session, _ = _session(settings)
        try:
            task_list = portfolio_benchmarks(session)
        finally:
            session.close()
    else:
        task_list = tasks.split(",")
    output_dir = Path(settings.evaluation.output_dir)
    cmd = build_command(
        model,
        task_list,
        base_url=base_url,
        num_fewshot=settings.evaluation.num_fewshot,
        batch_size=settings.evaluation.batch_size,
        limit=settings.evaluation.limit,
        output_dir=output_dir,
        harness_cmd=settings.evaluation.harness_cmd,
        tokenizer=settings.evaluation.tokenizer,
        max_gen_toks=settings.evaluation.max_gen_toks,
        num_concurrent=settings.evaluation.num_concurrent,
        max_length=settings.evaluation.max_length,
        timeout=settings.evaluation.timeout,
        confirm_run_unsafe_code=settings.evaluation.confirm_run_unsafe_code,
        apply_chat_template=settings.evaluation.apply_chat_template,
        repeats=settings.evaluation.repeats,
        gen_kwargs=settings.evaluation.gen_kwargs,
    )
    console.print("[cyan]Running:[/] " + " ".join(cmd))
    results = run_eval(cmd)
    scores = extract_scores(results)
    _print_scores(scores)


@app.command()
def run(
    ctx: typer.Context,
    model: str = typer.Option(
        None, "--model",
        help="Model name reported in the output (defaults to --model-id for weights).",
    ),
    model_id: str = typer.Option(
        None, "--model-id", help="HF weights id or local path to auto-deploy (source: weights).",
    ),
    base_url: str = typer.Option(
        None, "--base-url", help="Existing OpenAI-compatible endpoint / inference service IP.",
    ),
    scores: Path = typer.Option(
        None, "--scores", help="JSON {benchmark_id: score} file; skip evaluation (source: scores).",
    ),
    mode: str = typer.Option(
        None, "--mode",
        help="benchmark (eval only, writes scores.json) | analyze (eval + diagnosis, "
             "no recommendations) | full (eval + diagnosis + recommendations).",
    ),
    advisor_mode: str = typer.Option(
        None, "--advisor-mode", help="auto | llm_rules | rules (default auto).",
    ),
    arch: str = typer.Option(None, "--arch", help="dense|moe (new model)."),
    params: float = typer.Option(None, "--params", help="Parameter count in billions."),
    release_date: str = typer.Option(None, "--release-date", help="ISO date."),
    benchmarks: str = typer.Option(
        None, "--benchmarks",
        help="Comma-separated benchmark ids to evaluate (subset of the representative "
             "portfolio; e.g. mmlu_pro,math,swe_bench). Saves time by skipping the rest. "
             "Only effective on the evaluation path; ignored with --scores.",
    ),
    output: Path = typer.Option(
        None, "--output", help="Report output path (default: config run.output.dir/report.md).",
    ),
) -> None:
    """One command: deploy or reuse an endpoint, evaluate, diagnose, advise.

    Exactly one model source must be provided — ``--model-id`` (auto-deploys via
    vLLM), ``--base-url`` (reuse an inference service), or ``--scores`` (a JSON
    scores file, no evaluation). Source, mode, and advisor can also come from a
    YAML config via ``--config``. ``--mode`` picks how far the pipeline goes:
    ``benchmark`` (eval only, writes ``scores.json`` — feed it back with
    ``--scores``), ``analyze`` (eval + diagnosis, no recommendations), or
    ``full`` (eval + diagnosis + recommendations, default). The diagnosis in
    ``analyze``/``full`` always runs the unified stages 1-7 intelligent
    pipeline; ``--mode analyze`` skips Stage 6 suggestions. ``--benchmarks``
    narrows the evaluation to a subset of the representative portfolio.
    """
    settings: Settings = ctx.obj
    benchmark_list = (
        [b.strip() for b in benchmarks.split(",") if b.strip()]
        if benchmarks is not None
        else None
    )
    result = _run_or_exit(
        settings,
        model=model,
        model_id=model_id,
        base_url=base_url,
        scores=scores,
        mode=mode,
        advisor_mode=advisor_mode,
        arch=arch,
        params=params,
        release_date=release_date,
        benchmarks=benchmark_list,
        output=output,
    )
    _print_result(result)


feedback_app = typer.Typer(
    help="Stage 7 feedback loop: log executed suggestions and recalibrate cost/gain estimates.",
)
app.add_typer(feedback_app, name="feedback")


@feedback_app.command("log")
def feedback_log(
    ctx: typer.Context,
    capability: str = typer.Argument(..., help="Capability id the suggestion targeted."),
    suggestion_type: str = typer.Argument(..., help="rejection_sampling | targeted_synthesis | ..."),
    predicted_gain: float = typer.Option(..., "--predicted-gain", help="ExpectedGain predicted by Stage 5."),
    actual_gain: float = typer.Option(..., "--actual-gain", help="Measured gain after execution."),
    predicted_cost: float = typer.Option(..., "--predicted-cost", help="Cost predicted by Stage 5."),
    actual_cost: float = typer.Option(..., "--actual-cost", help="Real effort spent."),
    note: str = typer.Option(None, "--note", help="Free-form note."),
) -> None:
    """Record one executed suggestion's real outcome (feeds recalibration)."""
    from benchmark_diagnosis.intelligent_diagnosis.feedback import log_execution

    settings: Settings = ctx.obj
    session, _ = _session(settings)
    try:
        row_id = log_execution(
            session,
            capability_id=capability,
            suggestion_type=suggestion_type,
            predicted_gain=predicted_gain,
            actual_gain=actual_gain,
            predicted_cost=predicted_cost,
            actual_cost=actual_cost,
            note=note,
        )
        console.print(f"[green]Logged execution #{row_id}.[/]")
    finally:
        session.close()


@feedback_app.command("list")
def feedback_list(
    ctx: typer.Context,
    suggestion_type: str = typer.Option(None, "--suggestion-type"),
    limit: int = typer.Option(100, "--limit"),
) -> None:
    """List recent execution logs."""
    from benchmark_diagnosis.intelligent_diagnosis.feedback import list_executions

    settings: Settings = ctx.obj
    session, _ = _session(settings)
    try:
        rows = list_executions(session, suggestion_type=suggestion_type, limit=limit)
    finally:
        session.close()
    if not rows:
        console.print("[dim]No execution logs yet.[/]")
        return
    table = Table(title="Execution logs")
    for col in ("id", "capability", "type", "pred_gain", "act_gain", "pred_cost", "act_cost", "created"):
        table.add_column(col)
    for row in rows:
        table.add_row(
            str(row["id"]), row["capability_id"], row["suggestion_type"],
            f"{row['predicted_gain']:.3f}", f"{row['actual_gain']:.3f}",
            f"{row['predicted_cost']:.3f}", f"{row['actual_cost']:.3f}",
            row["created_at"],
        )
    console.print(table)


@feedback_app.command("recalibrate")
def feedback_recalibrate(
    ctx: typer.Context,
    smoothing: float = typer.Option(0.5, "--smoothing", help="Weight of measured vs prior (0..1)."),
) -> None:
    """Re-estimate Stage 5 cost ratios + gain scale from the execution log."""
    from benchmark_diagnosis.intelligent_diagnosis.feedback import recalibrate

    settings: Settings = ctx.obj
    session, _ = _session(settings)
    try:
        cal = recalibrate(session, smoothing=smoothing)
    finally:
        session.close()
    console.print(
        f"[green]Recalibrated from {cal.n_logs} execution log(s):[/] "
        f"gain_scale={cal.gain_scale}"
    )
    for stype, ratio in sorted(cal.costs.items()):
        console.print(f"  {stype}: cost ratio {ratio:.2f}")


def _print_scores(scores: dict[str, float]) -> None:
    table = Table(title="Evaluation scores")
    table.add_column("task", style="cyan")
    table.add_column("score", justify="right")
    for task, score in sorted(scores.items()):
        table.add_row(task, f"{score:.4f}")
    console.print(table)


if __name__ == "__main__":
    app()
