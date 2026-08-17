"""Intelligent diagnosis orchestrator: wires stages 1-7 into one report block.

Entry point: :func:`run_intelligent_diagnosis`. It composes the stage modules
and reuses Stage-0 assets from the DB (coverage / portfolio / curves / item
scores), so the output slots into ``pipeline.diagnose_model`` as the
``intelligent_diagnosis`` report section.

Everything is JSON-serializable at the boundary; dataclasses are converted via
:func:`_jsonable` so the report can be written to ``metrics.json``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import math
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Session

from benchmark_diagnosis.config import DiagnosisConfig, Settings
from benchmark_diagnosis.core import db
from benchmark_diagnosis.core.llm_client import LLMClient
from benchmark_diagnosis.core.schema import ModelRecord
from benchmark_diagnosis.data import queries
from benchmark_diagnosis.intelligent_diagnosis.candidate_generation import (
    CandidateConfig,
    generate_candidates,
)
from benchmark_diagnosis.intelligent_diagnosis.capability_taxonomy import (
    CapabilityTaxonomy,
    load_taxonomy,
)
from benchmark_diagnosis.intelligent_diagnosis.confidence_fusion import (
    FusionConfig,
    fuse,
)
from benchmark_diagnosis.intelligent_diagnosis.feedback import (
    load_calibration,
)
from benchmark_diagnosis.intelligent_diagnosis.guided_case_analyzer import (
    analyze_cases,
)
from benchmark_diagnosis.intelligent_diagnosis.priority_scorer import (
    PriorityConfig,
    score_priorities,
)
from benchmark_diagnosis.intelligent_diagnosis.probe_registry import (
    ProbeConfig,
    ProbeRegistry,
    split_todos,
    verify_candidates,
)
from benchmark_diagnosis.intelligent_diagnosis.suggestion_writer import (
    write_suggestions,
)
from benchmark_diagnosis.intelligent_diagnosis.types import (
    CaseAnalysis,
    FusedVerdict,
    PassKStats,
    SuggestionType,
)

_PACKAGE_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_COST_TABLE_PATH = _PACKAGE_DATA / "cost_table.yaml"


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


def _load_base_costs(path: str | Path | None = None) -> dict[SuggestionType, float]:
    p = Path(path) if path is not None else DEFAULT_COST_TABLE_PATH
    with open(p, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    costs: dict[SuggestionType, float] = {}
    for key, value in (raw.get("costs") or {}).items():
        try:
            stype = SuggestionType(key)
        except ValueError:
            continue
        costs[stype] = float(value)
    return costs


def _stage_configs(config: Settings) -> tuple[CandidateConfig, ProbeConfig, FusionConfig, PriorityConfig]:
    d: DiagnosisConfig = config.diagnosis
    candidate_cfg = CandidateConfig(
        percentile_threshold=config.curves.percentile_threshold,
        min_items_per_capability=getattr(d, "min_items_per_capability", 8),
        min_peers=getattr(d, "min_peers", 5),
    )
    probe_cfg = ProbeConfig(
        percentile_threshold=config.curves.percentile_threshold,
        min_peers=getattr(d, "min_peers", 5),
        min_passk_samples=getattr(d, "min_passk_samples", 8),
        passk_gap_threshold=getattr(d, "passk_gap_threshold", 0.5),
        pass1_high_threshold=getattr(d, "pass1_high_threshold", 0.5),
    )
    fusion_cfg = FusionConfig()
    priority_cfg = PriorityConfig()
    return candidate_cfg, probe_cfg, fusion_cfg, priority_cfg


def _peer_scores(session: Session) -> dict[str, list[float]]:
    """benchmark_id -> historical peer scores (aggregate, item_id IS NULL)."""
    matrix = queries.scores_matrix(session)
    out: dict[str, list[float]] = {}
    for bid in matrix.columns:
        values = [float(v) for v in matrix[bid].dropna() if v is not None]
        if values:
            out[str(bid)] = values
    return out


def run_intelligent_diagnosis(
    *,
    session: Session,
    verdict_benchmarks: list[dict[str, Any]],
    model: ModelRecord,
    raw_scores: dict[str, float],
    config: Settings,
    taxonomy: CapabilityTaxonomy | None = None,
    registry: ProbeRegistry | None = None,
    llm: LLMClient | None = None,
    cases: list[dict] | None = None,
    cases_by_benchmark: dict[str, list[dict]] | None = None,
    probe_scores: dict[str, float] | None = None,
    passk_stats: dict[str, PassKStats] | None = None,
    item_capabilities: dict[str, list[str]] | None = None,
    data_shares: dict[str, float] | None = None,
    mode: str = "full",
) -> dict[str, Any]:
    """Run the full stage 1-7 chain and return the report block.

    Args:
        session: DB session (Stage-0 assets + historical scores must exist).
        verdict_benchmarks: Per-benchmark verdicts from ``expectation_curves.judge``
            (benchmark_id / score / residual / underperforming / optional weight).
        model: The evaluated model.
        raw_scores: ``benchmark_id -> score`` for the evaluated model.
        config: Settings (diagnosis thresholds).
        taxonomy / registry: Optional overrides (defaults: packaged seed data).
        llm: Optional analyst LLM (Stage 3 + Stage 6 LLM mode).
        cases: Failed cases for guided analysis (used for every candidate when
            ``cases_by_benchmark`` is not provided).
        cases_by_benchmark: ``benchmark_id -> [failed cases]``; candidates sample
            from their source benchmarks.
        probe_scores: Optional ``benchmark_id -> score`` for probe benchmarks
            evaluated this run (defaults to the subset of ``raw_scores``).
        passk_stats: Optional ``capability_id -> PassKStats`` (pass@1/pass@k).
        item_capabilities: Optional ``item_id -> [capability_id]`` (fine mode).
        data_shares: Optional ``capability_id -> data fraction`` (Stage 4 rule).
        mode: ``"full"`` includes Stage 6 suggestions.

    Returns:
        Report block dict with keys: candidates, probe_results, verdicts,
        priorities, suggestions (full mode), probe_todo, versions.
    """
    taxonomy = taxonomy or load_taxonomy(config.diagnosis.taxonomy_path)
    registry = registry or ProbeRegistry.load(
        config.diagnosis.probe_registry_path, taxonomy=taxonomy
    )
    candidate_cfg, probe_cfg, fusion_cfg, priority_cfg = _stage_configs(config)
    coverage = db.load_latest_asset(session, "coverage") or []
    portfolios = db.load_latest_asset(session, "portfolio") or []
    item_scores = queries.item_triples(session)
    peer_scores = _peer_scores(session)
    item_totals: dict[str, int] = {}
    if not item_scores.empty:
        item_totals = {
            str(bid): int(n)
            for bid, n in item_scores.groupby("benchmark_id")["item_id"].nunique().items()
        }

    # ------------------------------------------------------------- Stage 1
    candidates = generate_candidates(
        verdict_benchmarks,
        taxonomy=taxonomy,
        target_model_id=model.model_id,
        item_scores=item_scores if not item_scores.empty else None,
        item_capabilities=item_capabilities,
        coverage=coverage,
        config=candidate_cfg,
    )

    # ------------------------------------------------------------- Stage 2
    probe_scores = dict(probe_scores or {})
    for bid, score in raw_scores.items():
        probe_scores.setdefault(bid, score)
    probe_results = verify_candidates(
        candidates,
        registry=registry,
        probe_scores=probe_scores,
        peer_scores=peer_scores,
        passk_stats=passk_stats,
        config=probe_cfg,
    )
    probe_by_cap = {r.capability_id: r for r in probe_results}

    # ------------------------------------------------------------- Stage 3
    case_analyses: dict[str, CaseAnalysis] = {}
    if llm is not None and mode in ("analyze", "full"):
        for cand in candidates:
            bench_cases: list[dict] = []
            if cases_by_benchmark:
                for src in cand.sources:
                    bench_cases.extend(cases_by_benchmark.get(src, []))
            if not bench_cases:
                bench_cases = list(cases or [])
            case_analyses[cand.capability_id] = analyze_cases(
                llm,
                taxonomy,
                cand.capability_id,
                bench_cases,
                sample_size=config.diagnosis.sample_size,
                saturation_window=getattr(config.diagnosis, "saturation_window", 20),
            )

    # ------------------------------------------------------------- Stage 4
    fused: dict[str, FusedVerdict] = {}
    for cand in candidates:
        fused[cand.capability_id] = fuse(
            probe_by_cap[cand.capability_id],
            case_analyses.get(cand.capability_id),
            data_share=(data_shares or {}).get(cand.capability_id),
            config=fusion_cfg,
        )

    # ------------------------------------------------------------- Stage 5
    calibration = load_calibration(session)
    base_costs = _load_base_costs(config.diagnosis.cost_table_path)
    priorities = score_priorities(
        candidates,
        fused,
        portfolios=portfolios,
        base_costs=base_costs,
        calibration=calibration,
        item_totals=item_totals,
        config=priority_cfg,
    )

    # ------------------------------------------------------------- Stage 6
    suggestions: dict[str, list[Any]] = {}
    if mode == "full":
        grouped = write_suggestions(
            priorities,
            llm=llm,
            probe_results=probe_by_cap,
            case_analyses=case_analyses,
            fused=fused,
        )
        suggestions = {key: _jsonable(value) for key, value in grouped.items()}

    build_probe, eval_pending = split_todos(probe_results)

    block: dict[str, Any] = {
        "taxonomy_version": taxonomy.version,
        "probe_registry_version": registry.version,
        "candidates": _jsonable(candidates),
        "probe_results": _jsonable(probe_results),
        "verdicts": _jsonable(list(fused.values())),
        "priorities": _jsonable(priorities),
        "suggestions": suggestions,
        "probe_todo": {
            "build_probe_set": build_probe,
            "pending_eval": eval_pending,
        },
        "calibration": _jsonable(calibration),
    }
    return block
