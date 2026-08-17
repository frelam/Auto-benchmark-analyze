"""Intelligent benchmark diagnosis: stage 1-7 shared data contracts.

Every stage of the v2 diagnosis pipeline
(see ``docs/intelligent-diagnosis-v2-design.md``) exchanges these in-memory
dataclasses; persistent storage lives in the versioned ``assets`` table and the
``execution_logs`` table (``benchmark_diagnosis.core.schema``).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class ProbeState(str, enum.Enum):
    """Stage 2 probe verification outcome for one candidate capability.

    ``NO_PROBE`` and ``PENDING_EVAL`` both cap confidence (missing evidence);
    they are kept distinct so the follow-up action differs: build a probe set
    vs. just evaluate the registered probe.
    """

    CONFIRMED = "CONFIRMED"
    NOT_CONFIRMED = "NOT_CONFIRMED"
    NO_PROBE = "NO_PROBE"  # no probe registered for this capability
    PENDING_EVAL = "PENDING_EVAL"  # probe registered but not evaluated yet


class RootCauseType(str, enum.Enum):
    """Stage 3 per-case root-cause vocabulary (fixed, never free-form)."""

    CONTENT_ERROR = "content_error"
    FORMAT_ERROR = "format_error"
    GRADING_ARTIFACT = "grading_artifact"
    LABEL_NOISE = "label_noise"
    AMBIGUOUS = "ambiguous"


class ConfidenceLevel(str, enum.Enum):
    """Stage 4 diagnosis confidence tiers."""

    HIGH = "High"
    MEDIUM_HIGH = "Medium-High"
    MEDIUM = "Medium"
    LOW = "Low"


class SuggestionType(str, enum.Enum):
    """Stage 4 suggestion types; Cost table and Stage 6 templates key on these."""

    REJECTION_SAMPLING = "rejection_sampling"
    TARGETED_SYNTHESIS = "targeted_synthesis"
    DATA_REWEIGHTING = "data_reweighting"
    COMPOSITIONAL_CURRICULUM = "compositional_curriculum"
    NON_TRAINING_FIX = "non_training_fix"
    EVAL_INFRA_FIX = "eval_infra_fix"
    BUILD_PROBE_FIRST = "build_probe_first"
    HUMAN_REVIEW = "human_review"


#: Confidence tier -> numeric weight used in Priority scoring (Low = 0: excluded).
CONFIDENCE_NUMERIC: dict[ConfidenceLevel, float] = {
    ConfidenceLevel.HIGH: 1.0,
    ConfidenceLevel.MEDIUM_HIGH: 0.75,
    ConfidenceLevel.MEDIUM: 0.5,
    ConfidenceLevel.LOW: 0.0,
}

#: Suggestion types that are NOT model-training interventions; they are routed
#: to engineering / eval infra and never mixed into the training priority list.
NON_TRAINING_TYPES = {
    SuggestionType.NON_TRAINING_FIX,
    SuggestionType.EVAL_INFRA_FIX,
    SuggestionType.BUILD_PROBE_FIRST,
    SuggestionType.HUMAN_REVIEW,
}


@dataclass
class CapabilityNode:
    """One node of the hierarchical capability taxonomy."""

    id: str
    name: str
    parent: str | None
    description: str = ""
    level: int = 1
    aliases: list[str] = field(default_factory=list)


@dataclass
class SubAccuracy:
    """Per-(benchmark, capability) item-level accuracy evidence (Stage 1 fine)."""

    benchmark_id: str
    capability_id: str
    sub_accuracy: float
    peer_mean: float
    peer_percentile: float
    n_items: int
    wilson_lb: float
    low_support: bool = False
    wilson_ub: float = 1.0  # upper bound: conservative-gain estimate uses this


@dataclass
class CandidateCapability:
    """Stage 1 output: one candidate deficit capability with screening evidence."""

    capability_id: str
    screening_score: float  # severity in [0, 1]
    evidence_mode: str  # "fine" | "coarse"
    sources: list[str] = field(default_factory=list)  # benchmark ids (OR union)
    sub_accuracies: list[SubAccuracy] = field(default_factory=list)
    source_scores: dict[str, float] = field(default_factory=dict)  # benchmark -> gap
    low_support: bool = False


@dataclass
class PassKStats:
    """Optional pass@1 / pass@k evidence for one capability (Stage 2)."""

    capability_id: str
    pass_1: float
    pass_k: float
    k: int = 16
    samples: int = 0
    probe_benchmark_id: str | None = None


@dataclass
class ProbeResult:
    """Stage 2 output for one candidate capability."""

    capability_id: str
    state: ProbeState
    probe_benchmark_id: str | None = None
    percentile: float | None = None  # model percentile among historical peers on the probe
    z_score: float | None = None
    pass_1: float | None = None
    pass_k: float | None = None
    passk_gap_ratio: float | None = None  # (pass_k - pass_1) / max(pass_1, eps)
    note: str = ""


@dataclass
class CaseVerdict:
    """Stage 3 per-case LLM verdict."""

    case_index: int
    root_cause_type: RootCauseType
    capability_tag: str
    evidence: str = ""


@dataclass
class CaseAnalysis:
    """Stage 3 aggregated statistics over guided bad-case analysis."""

    cases_analyzed: int = 0
    content_error_ratio: float = 0.0
    format_error_ratio: float = 0.0
    grading_artifact_ratio: float = 0.0
    label_noise_ratio: float = 0.0
    ambiguous_ratio: float = 0.0
    tag_agreement: float = 0.0  # fraction of cases whose tag is within the hypothesis
    near_miss_ratio: float = 0.0  # sibling/ancestor tags: hypothesis plausible but off
    saturated: bool = False
    per_tag_counts: dict[str, int] = field(default_factory=dict)
    per_cause_counts: dict[str, int] = field(default_factory=dict)

    @property
    def classified_fraction(self) -> float:
        """Fraction of sampled cases with a valid verdict (unparsed -> ambiguous)."""
        return self.cases_analyzed


@dataclass
class FusedVerdict:
    """Stage 4 output: confidence + suggestion type for one candidate."""

    capability_id: str
    confidence: ConfidenceLevel
    suggestion_type: SuggestionType
    evidence_score: float
    needs_human_review: bool = False
    rationale: str = ""
    probe_state: ProbeState = ProbeState.NO_PROBE
    data_share: float | None = None


@dataclass
class PriorityItem:
    """Stage 5 output: one ranked (capability, action) candidate."""

    capability_id: str
    suggestion_type: SuggestionType
    confidence: ConfidenceLevel
    expected_gain: float
    gain_lower: float  # Wilson-lower-bound version of the gain (uncertainty-aware)
    cost: float
    priority: float
    sources: list[str] = field(default_factory=list)


@dataclass
class Suggestion:
    """Stage 6 output: concrete, evidence-backed action write-up."""

    capability_id: str
    suggestion_type: SuggestionType
    confidence: ConfidenceLevel
    priority_score: float
    concrete_action: str
    supporting_evidence: list[str] = field(default_factory=list)
    expected_gain: float = 0.0
    root_cause_type: str = ""
    source: str = "deterministic"  # "deterministic" | "llm"
    raw: dict[str, Any] = field(default_factory=dict)
