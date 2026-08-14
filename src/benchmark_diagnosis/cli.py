"""Command-line interface for benchmark-diagnosis."""

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
from benchmark_diagnosis.data import ingestion
from benchmark_diagnosis.evaluation_orchestration.deploy import (
    serve_command,
    wait_until_ready,
)
from benchmark_diagnosis.evaluation_orchestration.harness_bridge import (
    build_command,
    run_eval,
)
from benchmark_diagnosis.pipeline import build_offline, diagnose_model
from benchmark_diagnosis.reporting.report_generator import write_report

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
    session, engine = _session(settings)
    try:
        if db.load_latest_asset(session, "portfolio") is None:
            if session.scalar(select(ModelRecord.model_id).limit(1)) is None:
                ingestion.load_seed(session)
            console.print("[cyan]Building offline assets (coverage/portfolio/curves)...[/]")
            build_offline(session, settings)
    finally:
        session.close()


def _get_or_build_model(
    session: Session,
    model_id: str,
    arch: str | None,
    params: float | None,
    release_date: str | None,
) -> ModelRecord:
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
    task_list = tasks.split(",") if tasks else _portfolio_benchmarks(settings)
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
    )
    console.print("[cyan]Running:[/] " + " ".join(cmd))
    results = run_eval(cmd)
    scores = _extract_scores(results)
    _print_scores(scores)


@app.command()
def diagnose(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Model id (registered or new)."),
    base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible endpoint to evaluate first."),
    scores_file: Path = typer.Option(None, "--scores", help="JSON {benchmark_id: score} instead of evaluating."),
    arch: str = typer.Option(None, "--arch", help="dense|moe (for a new model)."),
    params: float = typer.Option(None, "--params", help="Parameter count in billions."),
    release_date: str = typer.Option(None, "--release-date", help="ISO date."),
    output: Path = typer.Option(Path("data/report.md"), "--output", help="Report output path."),
) -> None:
    """Evaluate (optional) then diagnose and write a report."""
    settings: Settings = ctx.obj
    _ensure_offline(settings)

    raw_scores: dict[str, float]
    if scores_file is not None:
        raw_scores = json.loads(scores_file.read_text(encoding="utf-8"))
    elif base_url is not None:
        task_list = _portfolio_benchmarks(settings)
        cmd = build_command(
            model, task_list, base_url=base_url,
            num_fewshot=settings.evaluation.num_fewshot,
            batch_size=settings.evaluation.batch_size,
            limit=settings.evaluation.limit,
            output_dir=Path(settings.evaluation.output_dir),
            harness_cmd=settings.evaluation.harness_cmd,
        )
        console.print("[cyan]Evaluating...[/] " + " ".join(cmd))
        raw_scores = _extract_scores(run_eval(cmd))
    else:
        console.print("[red]Provide --base-url or --scores.[/]")
        raise typer.Exit(code=1)

    session, _ = _session(settings)
    try:
        model_record = _get_or_build_model(session, model, arch, params, release_date)
        report = diagnose_model(session, model_record, raw_scores, settings)
    finally:
        session.close()

    write_report(report, output)
    console.print(f"[green]Report written to {output}[/]")


@app.command()
def run(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Model name / HF id."),
    model_id: str = typer.Option(None, "--model-id", help="HF weights id to deploy first."),
    base_url: str = typer.Option(None, "--base-url", help="Existing OpenAI-compatible endpoint."),
    arch: str = typer.Option(None, "--arch", help="dense|moe (new model)."),
    params: float = typer.Option(None, "--params", help="Parameter count in billions."),
    output: Path = typer.Option(Path("data/report.md"), "--output", help="Report output path."),
) -> None:
    """End-to-end: deploy weights (optional) -> evaluate -> diagnose -> report."""
    settings: Settings = ctx.obj
    _ensure_offline(settings)

    if base_url is None:
        if model_id is None:
            console.print("[red]Provide --model-id (weights) or --base-url (endpoint).[/]")
            raise typer.Exit(code=1)
        base_url = f"http://{settings.serving.host}:{settings.serving.port}/v1"
        cmd = serve_command(model_id, settings.serving)
        console.print("[cyan]Deploying weights...[/] " + " ".join(cmd))
        proc = subprocess.Popen(cmd)
        try:
            if not wait_until_ready(base_url):
                console.print("[red]Server not ready in time.[/]")
                raise typer.Exit(code=1)
        except Exception:
            proc.terminate()
            raise
    else:
        proc = None

    try:
        task_list = _portfolio_benchmarks(settings)
        cmd = build_command(
            model, task_list, base_url=base_url,
            num_fewshot=settings.evaluation.num_fewshot,
            batch_size=settings.evaluation.batch_size,
            limit=settings.evaluation.limit,
            output_dir=Path(settings.evaluation.output_dir),
            harness_cmd=settings.evaluation.harness_cmd,
        )
        console.print("[cyan]Evaluating...[/] " + " ".join(cmd))
        raw_scores = _extract_scores(run_eval(cmd))

        session, _ = _session(settings)
        try:
            model_record = _get_or_build_model(session, model, arch, params, None)
            report = diagnose_model(session, model_record, raw_scores, settings)
        finally:
            session.close()
        write_report(report, output)
        console.print(f"[green]Report written to {output}[/]")
    finally:
        if proc is not None:
            console.print("[yellow]Stopping server...[/]")
            proc.terminate()
            proc.wait()


def _portfolio_benchmarks(settings: Settings) -> list[str]:
    session, _ = _session(settings)
    try:
        portfolios = db.load_latest_asset(session, "portfolio") or []
        benchmarks: set[str] = set()
        for pf in portfolios:
            for b in pf.get("benchmarks", []):
                benchmarks.add(b["benchmark_id"])
        return sorted(benchmarks)
    finally:
        session.close()


def _extract_scores(results: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for task, metrics in (results.get("results") or {}).items():
        value = ingestion._primary_metric(metrics)
        if value is not None:
            out[task] = value
    return out


def _print_scores(scores: dict[str, float]) -> None:
    table = Table(title="Evaluation scores")
    table.add_column("task", style="cyan")
    table.add_column("score", justify="right")
    for task, score in sorted(scores.items()):
        table.add_row(task, f"{score:.4f}")
    console.print(table)


if __name__ == "__main__":
    app()
