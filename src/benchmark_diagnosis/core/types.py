"""Shared enums and dataclasses used across pipeline stages.

These are the *in-memory* contracts between modules. Persistent tables live in
:mod:`benchmark_diagnosis.core.schema`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ScoringMethod(str, Enum):
    RULE_VERIFIED = "rule_verified"
    LLM_JUDGE = "llm_judge"
    HYBRID = "hybrid"


class ArchType(str, Enum):
    DENSE = "dense"
    MOE = "moe"


class Granularity(str, Enum):
    ITEM_LEVEL = "item_level"
    AGGREGATE_ONLY = "aggregate_only"


@dataclass
class CoverageEntry:
    """One row of the versioned capability-coverage table (design doc section 2.6)."""

    benchmark_id: str
    primary_cluster: str
    discrimination_profile: dict[str, float]
    factor_loadings: list[float]  # signed loadings (shape n_factors); the map embeds these
    coverage_breadth_score: float
    reliability_score: float
    saturated_flag: bool
    scoring_method: ScoringMethod
    design_goal_tags: list[str]
    design_goal_agreement_score: float
    granularity: Granularity


@dataclass
class ClusterPortfolio:
    """Representative benchmark combo for one capability cluster (section 3)."""

    cluster_id: str
    benchmarks: list[dict[str, float | str]] = field(default_factory=list)
    # ^ list of {"benchmark_id": str, "weight": float}


@dataclass
class ExpectationCurve:
    """A fitted expectation curve for one (benchmark, group) pair.

    ``kind`` is one of ``params_dense`` / ``params_moe`` / ``params_active`` /
    ``time_frontier``. ``coefficients`` are the fitted params; ``points`` holds the
    raw (x, y) data used for the fit so residuals can be recomputed.
    """

    benchmark_id: str
    kind: str
    coefficients: dict[str, float]
    points: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class DiagnosisResult:
    """Structured diagnosis output (section 5)."""

    cluster_id: str
    sub_capability: str | None
    failure_modes: dict[str, float] = field(default_factory=dict)
    representative_cases: list[dict[str, Any]] = field(default_factory=list)
    quantified_gap: float | None = None
    # Correlation + bad-case driven "which capabilities are missing" (engine core).
    capability_deficit: dict[str, float] = field(default_factory=dict)
    deficit_narrative: str = ""


@dataclass
class Recommendation:
    """A grounded optimization recommendation (section 6).

    ``datasets`` / ``hyperparameters`` hold the executable specifics the
    recommendation engine produces; ``reason_chain`` is the human-readable
    evidence trail (benchmark correlation -> missing capability -> intervention).
    """

    rule_id: str | None
    source: str  # rule_base:<id> or experience:<id> or external:<url>
    evidence_strength: str
    action: str
    validation_experiment: str
    datasets: list[dict[str, Any]] = field(default_factory=list)
    hyperparameters: list[dict[str, Any]] = field(default_factory=list)
    expected_effect: str = ""
    reason_chain: list[str] = field(default_factory=list)
    intervention_id: str | None = None


@dataclass
class ReportTrace:
    """Version trace attached to every generated report (section 2.5 / 4.2.5)."""

    coverage_version: str | None = None
    portfolio_version: str | None = None
    curves_version: str | None = None
    generated_at: str | None = None
