"""Rule-base diagnosis engine (design section 2.1) — deterministic, statistics-only.

Pipeline (all steps deterministic, no LLM):

1. **Low-score dataset filtering** — every scored benchmark is judged against
   the expectation curves: the *same-parameter-count / same-active-parameter-
   count* percentile (``params_dense`` / ``params_moe`` / ``params_active``
   curves, with a time-frontier fallback). Benchmarks below the configured
   percentile/z thresholds are the low-score datasets.
2. **Dataset -> capability mapping** — the coverage asset records which
   capabilities each benchmark exercises (``design_goal_tags``). Each low
   benchmark's shortfall (how far below its curve it sits, weighted by
   coverage reliability / agreement / saturation) is credited to its tags,
   resolved through the capability taxonomy (alias + ancestor roll-up).
3. **Missing-capability filtering** — candidates are filtered: evidence below
   a noise floor is dropped; a capability whose evidence is fully explained by
   a more specific descendant is collapsed away (keep the leaf, not the
   generic ancestor). The survivors are the *missing capabilities*.
4. **Capability -> dataset suggestions** — the experience base maps each
   capability to concrete training datasets + hyperparameters (the
   capability-dataset table); the probe registry and coverage table add
   verification/evaluation datasets worth adding. Suggestions are ranked by
   the capability's evidence and gap.

The result is a JSON-serializable block that the runner embeds in the report
and that the llm-agent engine adopts as its starting conclusion (2.2).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from benchmark_diagnosis.config import Settings
from benchmark_diagnosis.core import db
from benchmark_diagnosis.core.schema import ModelRecord
from benchmark_diagnosis.evaluation_orchestration.expectation_curves import judge
from benchmark_diagnosis.intelligent_diagnosis.capability_taxonomy import (
    CapabilityTaxonomy,
    load_taxonomy,
)
from benchmark_diagnosis.intelligent_diagnosis.probe_registry import ProbeRegistry

# Evidence below this fraction of the strongest capability is treated as noise.
NOISE_FLOOR = 0.05
# Drop an ancestor when a descendant carries at least this share of its
# evidence (the ancestor is then "explained by" the more specific capability).
COLLAPSE_RATIO = 0.8
# Capability tags that resolve to nothing in the taxonomy are still surfaced
# verbatim (unknown vocabulary should not silently vanish from the report).
KEEP_UNRESOLVED = True


@dataclass
class RuleBaseResult:
    """The full rule-base diagnosis output (JSON-serializable via asdict)."""

    engine: str = "rule"
    generated_at: str = ""
    low_score_benchmarks: list[dict[str, Any]] = field(default_factory=list)
    missing_capabilities: list[dict[str, Any]] = field(default_factory=list)
    dataset_suggestions: list[dict[str, Any]] = field(default_factory=list)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _shortfall(benchmark: dict[str, Any]) -> float:
    """How far below expectation this benchmark sits, in 0-1 units."""
    residual = benchmark.get("residual")
    if residual is not None:
        return max(0.0, -float(residual))
    score = float(benchmark.get("score", 0.0))
    norm = score / 100.0 if score > 1.5 else score
    return max(0.0, 1.0 - norm)


def _curve_basis(curve_kind: str | None) -> str:
    """Human label for the expectation basis a judgment used."""
    return {
        "params_dense": "同等参数量(dense)",
        "params_moe": "同等参数量(MoE)",
        "params_active": "同等激活参数量",
        "time_frontier": "时间前沿",
    }.get(curve_kind or "", curve_kind or "无预期曲线")


def filter_low_score_benchmarks(
    model: ModelRecord,
    raw_scores: dict[str, float],
    curves: list[Any],
    config: Settings,
) -> list[dict[str, Any]]:
    """Step 1: benchmarks scoring below the same-params/active-params percentile."""
    low: list[dict[str, Any]] = []
    for bid in sorted(raw_scores):
        j = judge(model, bid, raw_scores[bid], curves, config.curves)
        if not j["underperforming"]:
            continue
        low.append(
            {
                "benchmark_id": bid,
                "score": raw_scores[bid],
                "curve_kind": j["curve_kind"],
                "curve_basis": _curve_basis(j["curve_kind"]),
                "percentile": j["percentile"],
                "z_score": j["z_score"],
                "predicted": j["predicted"],
                "residual": j["residual"],
                "shortfall": _shortfall(j),
            }
        )
    low.sort(key=lambda b: b["percentile"])
    return low


def infer_missing_capabilities(
    low_benchmarks: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    taxonomy: CapabilityTaxonomy,
) -> list[dict[str, Any]]:
    """Steps 2-3: dataset->capability mapping + missing-capability filtering.

    Evidence per capability: each low benchmark credits its resolved
    design-goal tags with ``shortfall * reliability * agreement *
    saturation_penalty``; the credit also rolls up to ancestor capabilities
    (via the taxonomy). Filtering: noise floor, then collapse ancestors whose
    evidence is fully explained by a more specific descendant.
    """
    by_benchmark = {row["benchmark_id"]: row for row in coverage}
    evidence: dict[str, float] = {}
    sources: dict[str, dict[str, dict[str, Any]]] = {}

    def _credit(cid: str, b: dict[str, Any], signal: float) -> None:
        evidence[cid] = evidence.get(cid, 0.0) + signal
        prev = sources.setdefault(
            cid,
            {},
        ).setdefault(
            b["benchmark_id"],
            {
                "benchmark_id": b["benchmark_id"],
                "percentile": b["percentile"],
                "shortfall": b["shortfall"],
            },
        )
        # A benchmark can reach a capability both directly (its tag) and via
        # ancestor roll-up of a more specific tag; keep the strongest entry.
        if b["shortfall"] > prev["shortfall"]:
            prev.update(shortfall=b["shortfall"], percentile=b["percentile"])

    for b in low_benchmarks:
        bid = b["benchmark_id"]
        row = by_benchmark.get(bid)
        if row is None:
            continue
        reliability = float(row.get("reliability_score") or 1.0)
        agreement = float(row.get("design_goal_agreement_score") or 1.0)
        saturation_penalty = 0.5 if row.get("saturated_flag") else 1.0
        signal = b["shortfall"] * reliability * agreement * saturation_penalty
        if signal <= 0:
            continue
        resolved: list[str] = []
        for tag in row.get("design_goal_tags") or []:
            cid = taxonomy.resolve(str(tag))
            if cid is not None:
                resolved.append(cid)
            elif KEEP_UNRESOLVED:
                resolved.append(str(tag))  # unknown vocabulary, surfaced verbatim
        for cid in resolved:
            _credit(cid, b, signal)
            for anc in taxonomy.ancestors(cid):
                _credit(anc, b, signal)

    if not evidence:
        return []

    max_ev = max(evidence.values())
    # Filter 1: noise floor.
    survivors = {
        cid: ev for cid, ev in evidence.items() if ev >= NOISE_FLOOR * max_ev
    }
    # Filter 2: collapse ancestors fully explained by a descendant (keep the
    # most specific capability that still carries its own evidence).
    for cid in list(survivors):
        if cid not in taxonomy.nodes:
            continue
        for desc in taxonomy.descendants(cid):
            if survivors.get(desc, 0.0) >= COLLAPSE_RATIO * survivors[cid]:
                del survivors[cid]
                break

    out: list[dict[str, Any]] = []
    for cid, ev in sorted(survivors.items(), key=lambda kv: kv[1], reverse=True):
        node = taxonomy.get(cid)
        entry: dict[str, Any] = {
            "capability_id": cid,
            "name": node.name if node else cid,
            "level": node.level if node else 0,
            "evidence": round(ev / max_ev, 4),
            "sources": [
                {
                    "benchmark_id": s["benchmark_id"],
                    "percentile": round(s["percentile"], 1),
                    "shortfall": round(s["shortfall"], 4),
                }
                for s in sorted(
                    sources.get(cid, {}).values(),
                    key=lambda s: s["shortfall"],
                    reverse=True,
                )
            ],
        }
        if node and node.parent:
            entry["parent"] = node.parent
        out.append(entry)
    return out


def suggest_datasets(
    missing: list[dict[str, Any]],
    *,
    session: Session | None,
    coverage: list[dict[str, Any]],
    taxonomy: CapabilityTaxonomy,
    config: Settings,
) -> list[dict[str, Any]]:
    """Step 4: capability -> dataset suggestions (experience + probe tables).

    For each missing capability the experience base supplies training datasets
    + hyperparameter knobs (matched by capability tag, exact first then
    ancestor/descendant); the probe registry supplies narrow verification
    benchmarks; the coverage table lists benchmarks tagged with the capability
    as candidate evaluation additions. Entries without any experience match
    are still returned with an empty ``datasets`` list (the probe/coverage
    suggestions remain actionable).
    """
    interventions = _load_interventions(session)
    registry = ProbeRegistry.load(config.diagnosis.probe_registry_path, taxonomy=taxonomy)
    tagged_benchmarks: dict[str, list[str]] = {}
    for row in coverage:
        for tag in row.get("design_goal_tags") or []:
            cid = taxonomy.resolve(str(tag)) or str(tag)
            tagged_benchmarks.setdefault(cid, []).append(row["benchmark_id"])

    out: list[dict[str, Any]] = []
    for cap in missing:
        cid = cap["capability_id"]
        entry: dict[str, Any] = {
            "capability_id": cid,
            "gap": cap["evidence"],
            "datasets": [],
            "hyperparameters": [],
            "verification_benchmarks": [
                {"benchmark_id": p.benchmark_id, "note": p.note or "窄口径 probe"}
                for p in registry.probes_for(cid)
            ],
            "tagged_benchmarks": sorted(set(tagged_benchmarks.get(cid, []))),
            "evidence_strength": None,
            "intervention_id": None,
            "note": "",
        }
        best = _best_intervention(cid, interventions, taxonomy)
        if best is not None:
            intervention, closeness = best
            entry["datasets"] = _jsonable(intervention.get("datasets") or [])
            entry["hyperparameters"] = _jsonable(
                intervention.get("hyperparameters") or []
            )
            entry["evidence_strength"] = intervention.get("evidence_strength")
            entry["intervention_id"] = intervention.get("intervention_id")
            entry["note"] = (
                intervention.get("title") + "（来源："
                + (intervention.get("source_type") or "experience")
                + f"，匹配度 {'精确' if closeness == 0 else '层级'}）"
            )
        out.append(entry)
    return out


def _load_interventions(session: Session | None) -> list[dict[str, Any]]:
    if session is None:
        return []
    return db.load_latest_asset(session, "experience") or []


def _best_intervention(
    capability_id: str,
    interventions: list[dict[str, Any]],
    taxonomy: CapabilityTaxonomy,
) -> tuple[dict[str, Any], int] | None:
    """Best experience intervention for a capability (0=exact, 1=ancestor/desc).

    An intervention's ``applicable_tags`` (flat vocabulary) are resolved
    through the taxonomy; a resolved tag equal to the capability is an exact
    match, a tag that is an ancestor or descendant of it is a hierarchical
    match. Exact matches win; ties keep the first (stable) intervention.
    """
    best: tuple[dict[str, Any], int] | None = None
    for intervention in interventions:
        for tag in intervention.get("applicable_tags") or []:
            cid = taxonomy.resolve(str(tag))
            if cid is None:
                continue
            if cid == capability_id:
                closeness = 0
            elif cid in taxonomy.ancestors(capability_id) or cid in taxonomy.descendants(capability_id):
                closeness = 1
            else:
                continue
            if best is None or closeness < best[1]:
                best = (intervention, closeness)
                if closeness == 0:
                    return best
    return best


def run_rule_base(
    *,
    session: Session,
    model: ModelRecord,
    raw_scores: dict[str, float],
    config: Settings,
    coverage: list[dict[str, Any]] | None = None,
    curves: list[Any] | None = None,
) -> RuleBaseResult:
    """Run the full rule-base pipeline (2.1) and return the result block.

    Args:
        session: DB session (offline assets: coverage / curves / experience).
        model: The evaluated model (arch / params drive the same-params curves).
        raw_scores: ``benchmark_id -> score``.
        config: Settings (curves thresholds, taxonomy/probe paths).
        coverage / curves: Optional asset overrides (defaults: latest assets).

    Returns:
        A :class:`RuleBaseResult` with low-score benchmarks, missing
        capabilities, and dataset suggestions.
    """
    from benchmark_diagnosis.pipeline import _revive_curves

    coverage = coverage if coverage is not None else (db.load_latest_asset(session, "coverage") or [])
    curves = (
        curves
        if curves is not None
        else _revive_curves(db.load_latest_asset(session, "curves") or [])
    )
    taxonomy = load_taxonomy(config.diagnosis.taxonomy_path)

    low = filter_low_score_benchmarks(model, raw_scores, curves, config)
    missing = infer_missing_capabilities(low, coverage, taxonomy)
    suggestions = suggest_datasets(
        missing,
        session=session,
        coverage=coverage,
        taxonomy=taxonomy,
        config=config,
    )

    return RuleBaseResult(
        generated_at=dt.datetime.utcnow().isoformat(),
        low_score_benchmarks=_jsonable(low),
        missing_capabilities=_jsonable(missing),
        dataset_suggestions=_jsonable(suggestions),
    )
