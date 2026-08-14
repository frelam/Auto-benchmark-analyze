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
from benchmark_diagnosis.core.types import (
    ClusterPortfolio,
    DiagnosisResult,
    ExpectationCurve,
    Recommendation,
)
from benchmark_diagnosis.data import queries
from benchmark_diagnosis.diagnosis.failure_mode_analyst import classify_failures
from benchmark_diagnosis.evaluation_orchestration.drilldown_trigger import (
    should_drilldown,
)
from benchmark_diagnosis.evaluation_orchestration.expectation_curves import (
    fit_curves,
    judge,
)
from benchmark_diagnosis.evaluation_orchestration.screening_runner import cluster_scores
from benchmark_diagnosis.recommendation.groundedness_check import check_groundedness
from benchmark_diagnosis.recommendation.retrieval import get_retriever
from benchmark_diagnosis.recommendation.rule_base.loader import (
    load_validated_rules,
    match_rules,
)
from benchmark_diagnosis.recommendation.synthesizer import synthesize
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

    Returns the version ids of the coverage table, portfolio, and curves.
    """
    coverage = _build_coverage(session, config)
    coverage_version = db.save_asset(
        session, "coverage", _jsonable(coverage), note="capability coverage table"
    )

    portfolios = select_portfolios(coverage) if coverage else []
    portfolio_version = db.save_asset(session, "portfolio", _jsonable(portfolios))

    curves = fit_curves(session)
    curves_version = db.save_asset(session, "curves", _jsonable(curves))

    return {
        "coverage_version": coverage_version,
        "portfolio_version": portfolio_version,
        "curves_version": curves_version,
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
    for b in portfolio.benchmarks:
        bid = b["benchmark_id"]
        if bid not in raw_scores:
            continue
        j = judge(model, bid, raw_scores[bid], curves, config.curves)
        judges.append((float(b["weight"]), j))

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
        "benchmarks": [
            {
                "benchmark_id": b["benchmark_id"],
                "weight": b["weight"],
                "score": raw_scores.get(b["benchmark_id"]),
            }
            for b in portfolio.benchmarks
        ],
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


def _recommend_cluster(
    taxonomy,
    rules,
    diagnosis: DiagnosisResult,
    config: Settings,
    llm: LLMClient | None,
    benchmark_tags: dict[str, list[str]] | None = None,
) -> list[Recommendation]:
    tags = list(diagnosis.failure_modes.keys())
    if not tags:
        tags = list((benchmark_tags or {}).get(diagnosis.sub_capability) or [])
    if not tags and diagnosis.sub_capability:
        tags = [diagnosis.sub_capability]
    matched = match_rules(rules, tags)
    if not matched:
        return []

    top_rules = [r for r, _ in matched[:3]]
    retrieved = get_retriever(config.recommendation).retrieve(
        ",".join(tags), config.recommendation.max_external_sources
    )

    evidence = {
        "rule_ids": {r.rule_id for r in top_rules},
        "sources": {f"rule_base:{r.rule_id}" for r in top_rules},
        "numbers": set(diagnosis.failure_modes.values()),
    }

    if llm is not None:
        try:
            synth = synthesize(llm, diagnosis, top_rules, retrieved)
            if not check_groundedness(synth, evidence):
                return _actions_to_recs(synth.get("actions", []), top_rules)
        except Exception:  # noqa: BLE001 - LLM failure degrades to rule-based
            pass

    return [
        Recommendation(
            rule_id=r.rule_id,
            source=f"rule_base:{r.rule_id}",
            evidence_strength=r.evidence_strength,
            action=r.description,
            validation_experiment=_default_validation(r.category),
        )
        for r in top_rules
    ]


def _actions_to_recs(actions: list[dict], rules) -> list[Recommendation]:
    by_id = {r.rule_id: r for r in rules}
    recs = []
    for a in actions:
        rid = a.get("rule_id")
        rule = by_id.get(rid)
        recs.append(
            Recommendation(
                rule_id=rid,
                source=a.get("source", f"rule_base:{rid}" if rid else "external"),
                evidence_strength=rule.evidence_strength if rule else "low",
                action=a.get("action", ""),
                validation_experiment=a.get("validation_experiment", ""),
            )
        )
    return recs


def _default_validation(category: str) -> str:
    return (
        "在小规模子集上做消融：固定其余训练配置，仅应用该改动，"
        "对比目标 benchmark 的簇级得分变化（建议 >=1% 且跨 2 个随机种子稳定）。"
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
    """Run the online diagnosis/recommendation chain and return a report dict.

    Args:
        session: DB session (offline assets must already exist).
        model: Target model (transient ModelRecord is fine; need arch/params/date).
        raw_scores: Mapping benchmark_id -> score for the evaluated model.
        config: Settings.
        cases: Optional failed-case samples ({question, model_output, gold}) for
            LLM-as-analyst classification. When None, failure modes are empty.
        mode: ``"full"`` (analysis + recommendations) or ``"analyze"``
            (analysis only; recommendations are omitted).
        advisor_mode: Requested advisor mode (``auto``/``llm_rules``/``rules``),
            resolved against ``config`` via :func:`resolve_advisor_mode`.

    Returns:
        A structured report dict suitable for ``reporting.render_markdown``.
    """
    if mode not in ("analyze", "full"):
        raise ValueError(f"unknown mode {mode!r}; expected analyze|full")
    advisor_mode = resolve_advisor_mode(config, advisor_mode)

    portfolios = _revive_portfolios(db.load_latest_asset(session, "portfolio") or [])
    curves = _revive_curves(db.load_latest_asset(session, "curves") or [])
    coverage_version = _latest_version(session, "coverage")
    portfolio_version = _latest_version(session, "portfolio")
    curves_version = _latest_version(session, "curves")

    taxonomy, rules = load_validated_rules()
    llm = None if advisor_mode == "rules" else _make_analyst_llm(config)
    benchmark_tags = {
        b.benchmark_id: (b.declared_tags or []) for b in queries.list_benchmarks(session)
    }

    c_scores = cluster_scores(raw_scores, portfolios)

    clusters: list[dict] = []
    for pf in portfolios:
        verdict = _judge_cluster(model, pf, raw_scores, curves, config)
        if verdict is None:
            continue

        diagnosis = DiagnosisResult(
            cluster_id=pf.cluster_id,
            sub_capability=_lowest_benchmark(pf, raw_scores),
            failure_modes={},
            quantified_gap=verdict.get("quantified_gap"),
        )
        if verdict["underperforming"] and cases and llm is not None:
            diagnosis.failure_modes = classify_failures(
                llm, taxonomy, cases, sample_size=config.diagnosis.sample_size
            )

        recs = (
            []
            if mode == "analyze"
            else _recommend_cluster(taxonomy, rules, diagnosis, config, llm, benchmark_tags)
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
                    "sub_capability": diagnosis.sub_capability,
                    "failure_modes": diagnosis.failure_modes,
                    "quantified_gap": diagnosis.quantified_gap,
                },
                "recommendations": [_jsonable(r) for r in recs],
            }
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
        },
        "clusters": clusters,
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
