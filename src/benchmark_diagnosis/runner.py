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
from sqlalchemy import select
from sqlalchemy.orm import Session

from benchmark_diagnosis.config import Settings, resolve_advisor_mode
from benchmark_diagnosis.core import db
from benchmark_diagnosis.core.schema import ModelRecord
from benchmark_diagnosis.data import ingestion, queries
from benchmark_diagnosis.evaluation_orchestration.deploy import (
    serve_command,
    wait_until_ready,
)
from benchmark_diagnosis.evaluation_orchestration.expectation_curves import judge
from benchmark_diagnosis.evaluation_orchestration.harness_bridge import (
    build_command,
    extract_scores,
    run_eval,
    task_headline,
)
from benchmark_diagnosis.pipeline import _revive_curves, build_offline, diagnose_model
from benchmark_diagnosis.reporting.report_generator import render_json, render_markdown

console = Console()


@dataclass
class RunRequest:
    """A fully-resolved end-to-end run (CLI args merged over the ``run`` profile).

    ``source`` selects the model source: ``weights`` (auto-deploy via vLLM),
    ``endpoint`` (existing inference service IP) or ``scores`` (JSON file, no
    eval). ``mode`` picks how far the pipeline goes: ``benchmark`` (evaluation
    only), ``analyze`` (evaluation + diagnosis, no final suggestion
    write-up), or ``full`` (evaluation + diagnosis + suggestions).

    ``diagnose`` gates the diagnosis engines: evaluation always archives its
    results + bad cases, but diagnosis only runs when ``diagnose`` is true
    (CLI ``--diagnose`` or ``diagnosis.enabled: true``) and ``mode`` is
    ``analyze``/``full``. ``engine`` selects the path: ``rule`` (default,
    deterministic) or ``llm_agent`` (rule base + harness loop).
    """

    model: str | None = None
    mode: str = "full"
    source: Literal["weights", "endpoint", "scores"] | None = None
    weights: str | None = None
    base_url: str | None = None
    scores_file: Path | None = None
    benchmarks: list[str] | None = None
    arch: str | None = None
    params: float | None = None
    release_date: str | None = None
    advisor_mode: str = "auto"
    output: Path | None = None
    diagnose: bool | None = None
    engine: str | None = None


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
    benchmarks: list[str] | None = None,
    output: str | Path | None = None,
    diagnose: bool | None = None,
    engine: str | None = None,
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
        mode: ``"benchmark"`` (eval only), ``"analyze"`` (eval + diagnosis, no
            recommendations), or ``"full"`` (default ``run.mode``).
        advisor_mode: ``auto`` / ``llm_rules`` / ``rules`` (default
            ``recommendation.advisor_mode``).
        arch / params / release_date: New-model metadata.
        benchmarks: Optional subset of the representative portfolio to evaluate
            (e.g. ``["mmlu_pro", "math"]``). CLI ``--benchmarks`` (comma-
            separated) overrides ``run.model.benchmarks``. Only effective on
            the evaluation path (``weights`` / ``endpoint``); ignored when
            ``source=scores`` (the scores file carries its own benchmark set).
        output: Report output path (defaults to ``run.output.dir/report.md``).
        diagnose: Enable the diagnosis engines (CLI ``--diagnose``); default
            ``settings.diagnosis.enabled`` (which defaults to False).
        engine: Diagnosis path ``rule`` / ``llm_agent``; default
            ``settings.diagnosis.engine`` (``rule``).

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
    if resolved_mode not in ("benchmark", "analyze", "full"):
        raise ValueError(
            f"unknown mode {resolved_mode!r}; expected benchmark|analyze|full"
        )
    if advisor_mode not in (None, "auto", "llm_rules", "rules"):
        raise ValueError(
            f"unknown advisor_mode {advisor_mode!r}; expected auto|llm_rules|rules"
        )
    if engine not in (None, "rule", "llm_agent"):
        raise ValueError(
            f"unknown diagnosis engine {engine!r}; expected rule|llm_agent"
        )

    resolved_model = model or cfg.name
    if resolved_model is None and source == "weights":
        resolved_model = resolved_weights
    if resolved_model is None:
        raise ValueError("a model name is required: pass --model (or set run.model.name)")

    # Benchmark subset: CLI wins over config; an empty list reads as "no
    # constraint" so a YAML `benchmarks: []` is the same as unset. Ignored for
    # the scores path — the scores file already carries its own benchmark set.
    resolved_benchmarks = benchmarks if benchmarks is not None else cfg.benchmarks
    if resolved_benchmarks is not None and not resolved_benchmarks:
        resolved_benchmarks = None
    if resolved_benchmarks is not None and source == "scores":
        console.print(
            "[yellow]--benchmarks is ignored on the scores path "
            "(the scores file carries its own benchmark set).[/]"
        )
        resolved_benchmarks = None

    return RunRequest(
        model=resolved_model,
        mode=resolved_mode,
        source=source,
        weights=resolved_weights,
        base_url=resolved_base_url,
        scores_file=resolved_scores,
        benchmarks=resolved_benchmarks,
        arch=arch or cfg.arch,
        params=params if params is not None else cfg.params,
        release_date=release_date or cfg.release_date,
        advisor_mode=advisor_mode or settings.recommendation.advisor_mode,
        output=Path(output) if output is not None else None,
        diagnose=diagnose if diagnose is not None else settings.diagnosis.enabled,
        engine=engine or settings.diagnosis.engine,
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
    if mode not in ("benchmark", "analyze", "full"):
        raise ValueError(f"unknown mode {mode!r}; expected benchmark|analyze|full")
    # benchmark mode stops after evaluation; no advisor / diagnosis is needed.
    advisor_mode = (
        resolve_advisor_mode(settings, request.advisor_mode)
        if mode != "benchmark"
        else "n/a"
    )

    # Diagnosis is opt-in: evaluation always archives its results + bad cases,
    # but the engines only run when requested (CLI --diagnose / config
    # diagnosis.enabled) and the mode goes past evaluation. ``request.diagnose``
    # is None when the caller built the request directly (no CLI merge), so the
    # config default applies.
    diagnose_requested = (
        settings.diagnosis.enabled if request.diagnose is None else request.diagnose
    )
    diagnose = bool(diagnose_requested) and mode in ("analyze", "full")
    engine: str | None = None
    if diagnose:
        from benchmark_diagnosis.config import resolve_diagnosis_engine

        engine = resolve_diagnosis_engine(settings, request.engine)
    elif diagnose_requested and mode == "benchmark":
        console.print(
            "[yellow]--diagnose is ignored in benchmark mode: evaluation artifacts "
            "are archived; re-run with --mode analyze/full to diagnose.[/]"
        )

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

        raw_scores, results_payload = _collect_scores(
            request, base_url, portfolio_ids, run_harness, settings
        )

        if raw_scores and not (portfolio_ids & set(raw_scores)):
            console.print(
                "[yellow]Warning: none of the scores overlap the representative "
                "portfolio; cluster analysis will be empty. Overlap needs one of: "
                f"{sorted(portfolio_ids)}[/]"
            )

        # Evaluation artifacts (scores + bad cases) are archived on every run,
        # diagnosis or not — this is the human-analyzable eval record.
        output_dir = (
            request.output or Path(settings.run.output.dir) / "report.md"
        ).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        model_record = get_or_build_model(
            session, request.model, request.arch, request.params, request.release_date
        )
        artifacts = _archive_eval_artifacts(
            session, output_dir, raw_scores, results_payload, settings, model_record
        )

        if mode == "benchmark":
            # Benchmark-only: scores.json is shaped exactly like the ``--scores``
            # input so a follow-up ``run --scores <path>`` completes the
            # diagnosis; bad cases / summary live next to it.
            scores_path = output_dir / "scores.json"
            console.print(
                f"[green]Scores:[/] {scores_path} ({len(raw_scores)} benchmarks) "
                f"— feed back with `run --scores {scores_path}`"
            )
            console.print(
                f"[green]Artifacts:[/] {artifacts.summary_path} "
                f"({sum(len(v) for v in artifacts.bad_cases.values())} bad case(s) "
                f"in {artifacts.bad_case_dir})"
            )
            return RunResult(
                report={"scores": raw_scores, "mode": "benchmark"},
                report_path=scores_path,
                metrics_path=scores_path,
                figure_paths=[],
                mode="benchmark",
                advisor_mode=advisor_mode,
                served=source == "weights",
            )

        if not diagnose:
            console.print(
                "[yellow]Diagnosis disabled[/] (enable with --diagnose or "
                "diagnosis.enabled: true). Evaluation artifacts written:"
            )
            console.print(
                f"[green]Summary:[/] {artifacts.summary_path}\n"
                f"[green]Results:[/] {artifacts.results_path}\n"
                f"[green]Bad cases:[/] {artifacts.bad_case_dir}"
            )
            return RunResult(
                report={"scores": raw_scores, "mode": mode, "diagnosed": False},
                report_path=artifacts.summary_path,
                metrics_path=artifacts.results_path,
                figure_paths=[],
                mode=mode,
                advisor_mode=advisor_mode,
                served=source == "weights",
            )

        assert engine is not None
        report = diagnose_model(
            session,
            model_record,
            raw_scores,
            settings,
            mode=mode,
            advisor_mode=advisor_mode,
            engine=engine,
            base_url=base_url,
            bad_cases_dir=artifacts.bad_case_dir,
            output_dir=output_dir,
        )
        report["scores"] = raw_scores

        # Artifacts land next to the report so the Markdown figure links hold
        # wherever the report is written (config run.output.dir when no --output).
        report_path = output_dir / "report.md"

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
    """Idempotently build seed data + offline assets (coverage/portfolio/curves).

    The build runs when ANY of the four asset types (coverage / portfolio /
    curves / experience) is missing — a stale database from an older version
    (e.g. built before the experience asset existed) is completed rather than
    silently leaving the diagnosis engines without their tables.
    """
    missing = [
        asset_type
        for asset_type in ("coverage", "portfolio", "curves", "experience")
        if db.load_latest_asset(session, asset_type) is None
    ]
    if missing:
        if session.scalar(select(ModelRecord.model_id).limit(1)) is None:
            ingestion.load_seed(session)
        console.print(
            "[cyan]Building offline assets "
            f"(coverage/portfolio/curves/experience; missing: {missing})...[/]"
        )
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
) -> tuple[dict[str, float], dict[str, Any]]:
    """Load scores from a file, or evaluate the portfolio and flatten the results.

    Returns ``(raw_scores, results_payload)`` — the second element is the
    parsed harness results dict (empty for the ``--scores`` path) and feeds
    the evaluation-artifact archiving (bad-case extraction).
    """
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
        return raw, {}

    tasks = _resolve_eval_tasks(portfolio_ids, request.benchmarks)
    if not tasks:
        console.print("[yellow]No benchmarks to evaluate; skipping eval.[/]")
        return {}, {}

    def _cmd(task: str) -> list[str]:
        return build_command(
            request.model,
            [task],
            base_url=base_url,
            num_fewshot=settings.evaluation.num_fewshot,
            batch_size=settings.evaluation.batch_size,
            limit=settings.evaluation.limit,
            output_dir=Path(settings.evaluation.output_dir),
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

    if request.benchmarks:
        console.print(f"[cyan]Evaluating {len(tasks)} benchmark(s)[/] (subset: {','.join(tasks)})")
    else:
        console.print(f"[cyan]Evaluating {len(tasks)} benchmark(s)[/] (full portfolio)")

    # Evaluate one dataset per harness run and print its score as soon as it
    # finishes, instead of waiting for the whole portfolio (lm-eval only writes
    # a results.json once every task completes). Per-run payloads are merged so
    # downstream bad-case archiving sees the same combined results as before.
    raw_scores: dict[str, float] = {}
    results: dict[str, Any] = {}
    for i, task in enumerate(tasks, 1):
        cmd = _cmd(task)
        console.print(f"[cyan][{i}/{len(tasks)}] Evaluating {task}[/]")
        console.print("[dim]  " + " ".join(cmd) + "[/]")
        task_results = run_harness(cmd)
        entries = task_results.get("results") or {}
        results.setdefault("results", {}).update(entries)
        groups = task_results.get("groups")
        if isinstance(groups, dict):
            results.setdefault("groups", {}).update(groups)
        raw_scores.update(extract_scores(task_results))
        _print_task_score(task, task_results)
    return raw_scores, results


def _archive_eval_artifacts(
    session: Session,
    output_dir: Path,
    raw_scores: dict[str, float],
    results: dict[str, Any],
    settings: Settings,
    model: ModelRecord,
) -> Any:
    """Write scores / eval detail / summary / bad cases under ``output_dir``.

    Runs on every evaluation (diagnosis or not) so the run leaves a
    self-contained, human-analyzable record. Enriches ``eval_results.json``
    with expectation-curve judgments when the curves asset exists (best
    effort: failures degrade to an unjudged record, never crash the run).
    """
    from benchmark_diagnosis.evaluation_orchestration.artifacts import (
        find_sample_files,
        write_eval_artifacts,
    )

    judgments: dict[str, dict[str, Any]] = {}
    try:
        curves = _revive_curves(db.load_latest_asset(session, "curves") or [])
        for bid, score in raw_scores.items():
            judgments[bid] = judge(model, bid, score, curves, settings.curves)
    except Exception:  # noqa: BLE001 - archiving must never fail the run
        judgments = {}
    benchmark_names = {
        b.benchmark_id: b.name for b in queries.list_benchmarks(session)
    }
    sample_files = find_sample_files(settings.evaluation.output_dir)
    return write_eval_artifacts(
        output_dir,
        raw_scores,
        results,
        sample_files,
        benchmark_names=benchmark_names,
        judgments=judgments,
    )


def _resolve_eval_tasks(
    portfolio_ids: set[str], benchmarks: list[str] | None
) -> list[str]:
    """Pick the evaluation task list, honoring an optional benchmark subset.

    With ``benchmarks=None`` the full representative portfolio is evaluated.
    With a subset, the intersection is evaluated and any requested ids not in
    the portfolio are reported (they are still evaluated — the harness knows the
    task — but they will not feed the cluster analysis downstream).
    """
    if not benchmarks:
        return sorted(portfolio_ids)
    off = [b for b in benchmarks if b not in portfolio_ids]
    if off:
        console.print(
            f"[yellow]Warning: {len(off)} of {len(benchmarks)} requested "
            f"benchmark(s) are not in the representative portfolio and will be "
            f"evaluated but excluded from cluster analysis: {off}[/]"
        )
    return sorted(benchmarks)


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


def _print_task_score(task: str, task_results: dict[str, Any]) -> None:
    """Print a single dataset's score the moment its harness run returns."""
    headline = task_headline(task_results, task)
    if headline is None:
        console.print(f"[green]✔ {task}[/]  (no computable metric)")
        return
    metric, score = headline
    console.print(f"[green]✔ {task}[/]  {metric} = {score:.4f}")
