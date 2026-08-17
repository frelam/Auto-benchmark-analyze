"""End-to-end orchestration: offline asset build + online diagnosis/recommendation.

This is the glue layer. It composes the analysis modules and persists every
offline artifact as a versioned asset so each report is reproducible.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import math
from typing import Any

from sqlalchemy.orm import Session

from benchmark_diagnosis.capability_analysis.coverage_table import build_coverage_table
from benchmark_diagnosis.capability_analysis.factor_analysis import fit_factor_model
from benchmark_diagnosis.capability_analysis.mirt_fit import select_dimensions
from benchmark_diagnosis.config import Settings, resolve_advisor_mode
from benchmark_diagnosis.core import db
from benchmark_diagnosis.core.llm_client import LLMClient
from benchmark_diagnosis.core.schema import ModelRecord
from benchmark_diagnosis.core.types import ClusterPortfolio, ExpectationCurve
from benchmark_diagnosis.data import queries
from benchmark_diagnosis.evaluation_orchestration.drilldown_trigger import (
    should_drilldown,
)
from benchmark_diagnosis.evaluation_orchestration.expectation_curves import (
    fit_curves,
    judge,
)
from benchmark_diagnosis.evaluation_orchestration.screening_runner import cluster_scores
from benchmark_diagnosis.recommendation.experience_base import load_experience_base
from benchmark_diagnosis.representative_selection.portfolio_selector import (
    select_portfolios,
)

_MIN_MODELS_FOR_MIRT = 2
_MIN_ITEMS_FOR_MIRT = 5


def _jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses/enums/dates/numpy to JSON-friendly types."""
    if dataclasses.is_dataclass(obj):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _revive_curves(data: list[dict]) -> list[ExpectationCurve]:
    curves = []
    for d in data or []:
        d = dict(d)
        d["points"] = [tuple(p) for p in d.get("points", [])]
        curves.append(ExpectationCurve(**d))
    return curves


def _revive_portfolios(data: list[dict]) -> list[ClusterPortfolio]:
    return [
        ClusterPortfolio(cluster_id=d["cluster_id"], benchmarks=d["benchmarks"])
        for d in (data or [])
    ]


def _build_coverage(session: Session, config: Settings) -> list:
    """Build the coverage table, preferring mIRT over factor analysis."""
    scores = queries.scores_matrix(session)
    meta = {b.benchmark_id: b for b in queries.list_benchmarks(session)}
    triples = queries.item_triples(session)

    can_mirt = (
        not triples.empty
        and triples["model_id"].nunique() >= _MIN_MODELS_FOR_MIRT
        and triples["item_id"].nunique() >= _MIN_ITEMS_FOR_MIRT
    )
    if can_mirt:
        try:
            _, mirt = select_dimensions(triples)
            item_bench = (
                triples.drop_duplicates("item_id")
                .set_index("item_id")["benchmark_id"]
                .to_dict()
            )
            return build_coverage_table(
                mirt=mirt, benchmark_meta=meta, item_benchmark=item_bench, scores=scores
            )
        except Exception:  # noqa: BLE001 - any mIRT failure falls back gracefully
            pass

    if scores.empty:
        return []
    factor = fit_factor_model(scores)
    return build_coverage_table(factor=factor, benchmark_meta=meta, scores=scores)


def build_offline(session: Session, config: Settings) -> dict[str, str]:
    """Build and persist the versioned offline assets (design doc layers 2-4).

    Returns the version ids of the coverage table, portfolio, curves, and the
    tool-maintained experience base (dataset / hyperparameter knowledge).
    """
    coverage = _build_coverage(session, config)
    coverage_version = db.save_asset(
        session, "coverage", _jsonable(coverage), note="capability coverage table"
    )

    portfolios = select_portfolios(coverage) if coverage else []
    portfolio_version = db.save_asset(session, "portfolio", _jsonable(portfolios))

    curves = fit_curves(session)
    curves_version = db.save_asset(session, "curves", _jsonable(curves))

    experience_base = load_experience_base(
        session, config.recommendation.experience_path
    )
    experience_version = db.save_asset(
        session,
        "experience",
        _jsonable(experience_base.interventions),
        note="tool-maintained experience base (datasets + hyperparameters + outcomes)",
    )

    return {
        "coverage_version": coverage_version,
        "portfolio_version": portfolio_version,
        "curves_version": curves_version,
        "experience_version": experience_version,
    }


def _judge_cluster(
    model: ModelRecord,
    portfolio: ClusterPortfolio,
    raw_scores: dict[str, float],
    curves: list[ExpectationCurve],
    config: Settings,
) -> dict[str, Any] | None:
    """Weight-aggregate per-benchmark residual judgments into a cluster verdict."""
    judges: list[tuple[float, dict]] = []
    benchmarks: list[dict] = []
    for b in portfolio.benchmarks:
        bid = b["benchmark_id"]
        if bid not in raw_scores:
            continue
        j = judge(model, bid, raw_scores[bid], curves, config.curves)
        judges.append((float(b["weight"]), j))
        benchmarks.append(
            {
                "benchmark_id": bid,
                "weight": b["weight"],
                "score": raw_scores.get(bid),
                "residual": j.get("residual"),
                "percentile": j.get("percentile"),
                "z_score": j.get("z_score"),
                "underperforming": j.get("underperforming", False),
            }
        )

    if not judges:
        return None

    total_w = sum(w for w, _ in judges)
    weighted_pct = sum(w * j["percentile"] for w, j in judges) / total_w
    weighted_z = sum(w * j["z_score"] for w, j in judges) / total_w
    under = should_drilldown(
        {"percentile": weighted_pct, "z_score": weighted_z, "underperforming": False},
        config.curves,
    )

    # Quantified gap = how far the model falls short of its expectation curve
    # (predicted - score, unit scale), aggregated over scored benchmarks whose
    # residual is computable (skip closed-source models without params curves).
    gap_judges = [(w, j["residual"]) for w, j in judges if j.get("residual") is not None]
    quantified_gap = (
        sum(w * (-residual) for w, residual in gap_judges)
        / sum(w for w, _ in gap_judges)
        if gap_judges
        else None
    )

    return {
        "cluster_id": portfolio.cluster_id,
        "percentile": weighted_pct,
        "z_score": weighted_z,
        "underperforming": under,
        "quantified_gap": quantified_gap,
        "benchmarks": benchmarks,
    }


def _make_analyst_llm(config: Settings) -> LLMClient | None:
    if not config.llm.model:
        return None
    return LLMClient(
        base_url=config.llm.base_url,
        model=config.llm.model,
        api_key=config.llm.api_key,
        max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature,
        timeout_seconds=config.llm.timeout_seconds,
    )


def diagnose_model(
    session: Session,
    model: ModelRecord,
    raw_scores: dict[str, float],
    config: Settings,
    *,
    cases: list[dict] | None = None,
    mode: str = "full",
    advisor_mode: str = "auto",
) -> dict[str, Any]:
    """Run the unified diagnosis pipeline and return a report dict.

    The pipeline is the stages 1-7 chain (design doc v2), preceded by a Stage 0
    cluster-verdict aggregation that feeds per-benchmark verdicts into Stage 1.
    Stages 1-7 degrade gracefully when their preconditions are absent (no probe
    -> NO_PROBE list, no LLM -> deterministic Stage 6, no item tags -> coarse
    Stage 1, no cases -> Stage 3 skipped).

    Args:
        session: DB session (offline assets must already exist).
        model: Target model (transient ModelRecord is fine; need arch/params/date).
        raw_scores: Mapping benchmark_id -> score for the evaluated model.
        config: Settings.
        cases: Optional failed-case samples (``{question, model_output, gold}``)
            for Stage 3 guided bad-case analysis. When None, case evidence empty.
        mode: ``"full"`` (analysis + Stage 6 suggestions) or ``"analyze"``
            (analysis only; Stage 6 suggestions are omitted).
        advisor_mode: Requested advisor mode (``auto``/``llm_rules``/``rules``),
            resolved against ``config`` via :func:`resolve_advisor_mode`.
            ``rules`` disables the analyst LLM (Stage 3 + Stage 6 LLM skipped).

    Returns:
        A structured report dict suitable for ``reporting.render_markdown``. The
        ``clusters`` field carries Stage 0 per-cluster verdicts; the ``diagnosis``
        field carries the unified stages 1-7 output (candidates, verdicts,
        priorities, suggestions).
    """
    if mode not in ("analyze", "full"):
        raise ValueError(f"unknown mode {mode!r}; expected analyze|full")
    advisor_mode = resolve_advisor_mode(config, advisor_mode)

    portfolios = _revive_portfolios(db.load_latest_asset(session, "portfolio") or [])
    curves = _revive_curves(db.load_latest_asset(session, "curves") or [])
    coverage_version = _latest_version(session, "coverage")
    portfolio_version = _latest_version(session, "portfolio")
    curves_version = _latest_version(session, "curves")
    experience_version = _latest_version(session, "experience")

    llm = None if advisor_mode == "rules" else _make_analyst_llm(config)
    c_scores = cluster_scores(raw_scores, portfolios)

    # ---- Stage 0: per-cluster verdict aggregation (also feeds Stage 1) ----
    clusters: list[dict] = []
    per_benchmark_verdicts: dict[str, dict] = {}
    for pf in portfolios:
        verdict = _judge_cluster(model, pf, raw_scores, curves, config)
        if verdict is None:
            continue
        for b in pf.benchmarks:
            bid = b["benchmark_id"]
            if bid in raw_scores and bid not in per_benchmark_verdicts:
                per_benchmark_verdicts[bid] = judge(
                    model, bid, raw_scores[bid], curves, config.curves
                )
        clusters.append(
            {
                "cluster_id": pf.cluster_id,
                "score": c_scores.get(pf.cluster_id),
                "percentile": verdict["percentile"],
                "z_score": verdict["z_score"],
                "underperforming": verdict["underperforming"],
                "benchmarks": verdict["benchmarks"],
                "diagnosis": {
                    "sub_capability": _lowest_benchmark(pf, raw_scores),
                    "quantified_gap": verdict.get("quantified_gap"),
                },
            }
        )

    # ---- Stages 1-7: unified intelligent diagnosis ----
    from benchmark_diagnosis.intelligent_diagnosis.orchestrator import (
        run_intelligent_diagnosis,
    )

    diagnosis_block = run_intelligent_diagnosis(
        session=session,
        verdict_benchmarks=list(per_benchmark_verdicts.values()),
        model=model,
        raw_scores=raw_scores,
        config=config,
        llm=llm,
        cases=cases,
        mode=mode,
    )

    return {
        "model": {
            "model_id": model.model_id,
            "name": model.name,
            "arch_type": model.arch_type,
            "total_params": model.total_params,
            "active_params": model.active_params,
            "release_date": model.release_date.isoformat() if model.release_date else None,
        },
        "generated_at": dt.datetime.utcnow().isoformat(),
        "mode": mode,
        "advisor_mode": advisor_mode,
        "versions": {
            "coverage_version": coverage_version,
            "portfolio_version": portfolio_version,
            "curves_version": curves_version,
            "experience_version": experience_version,
        },
        "clusters": clusters,
        "diagnosis": diagnosis_block,
    }


def _lowest_benchmark(
    portfolio: ClusterPortfolio, raw_scores: dict[str, float]
) -> str | None:
    scored = [(b["benchmark_id"], raw_scores[b["benchmark_id"]]) for b in portfolio.benchmarks
              if b["benchmark_id"] in raw_scores]
    if not scored:
        return None
    return min(scored, key=lambda kv: kv[1])[0]


def _latest_version(session: Session, asset_type: str) -> str | None:
    return db.latest_version_id(session, asset_type)
