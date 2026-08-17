"""Tool-maintained experience base — the recommendation engine's data core.

This is the *maintained knowledge* the recommendation algorithm reads: dataset
recommendations, hyperparameter knobs, expected effects, and accumulated
outcomes for each diagnosed capability/failure-mode deficit. Users never edit it
in normal use — the packaged ``experience.yaml`` seeds the base and the tool
persists it as a versioned ``experience`` DB asset that :func:`record_outcome`
extends with measured deltas over time, so the base learns from every run.

Loading composes seed + persistence: parse the packaged YAML (canonical
interventions), then overlay any recorded outcomes found in the latest
``experience`` asset (when one exists). A fresh DB therefore behaves identically
to a mature one — the engine never depends on an asset existing.
"""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from benchmark_diagnosis.core import db
from benchmark_diagnosis.recommendation.rule_base.loader import (
    EVIDENCE_STRENGTH_ORDER,
    load_rules,
    load_taxonomy,
)

_PKG_ROOT = files("benchmark_diagnosis")

SOURCE_TYPES = {"human_expert", "rule_base", "paper", "outcome_measured"}


@dataclass
class DatasetSuggestion:
    """A concrete dataset the tool recommends adding to the training mix."""

    name: str
    rationale: str
    expected_effect: str
    source: str = ""


@dataclass
class HyperparameterSuggestion:
    """One hyperparameter knob to change, with direction and a typical range."""

    knob: str
    direction: str
    typical_range: str
    rationale: str


@dataclass
class Outcome:
    """A measured delta from an intervention trial (the feedback loop)."""

    delta: float
    note: str = ""
    run_id: str | None = None


@dataclass
class Intervention:
    """One actionable intervention: what to add (datasets) and how to tune."""

    intervention_id: str
    title: str
    applicable_tags: list[str]
    category: str
    expected_effect: str
    evidence_strength: str
    source_type: str
    datasets: list[DatasetSuggestion] = field(default_factory=list)
    hyperparameters: list[HyperparameterSuggestion] = field(default_factory=list)
    linked_rule_id: str | None = None
    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def mean_delta(self) -> float | None:
        """Mean measured outcome delta, or None when no outcomes exist."""
        if not self.outcomes:
            return None
        return sum(o.delta for o in self.outcomes) / len(self.outcomes)


@dataclass
class ExperienceBase:
    """The full tool-maintained knowledge base."""

    version: int
    interventions: list[Intervention]
    failure_mode_capability_map: dict[str, list[str]] = field(default_factory=dict)

    def by_id(self) -> dict[str, Intervention]:
        return {i.intervention_id: i for i in self.interventions}


def _default_experience_path() -> Path:
    return Path(_PKG_ROOT) / "recommendation" / "experience_base" / "experience.yaml"


def load_experience_base(
    session: Any | None = None,
    path: str | Path | None = None,
) -> ExperienceBase:
    """Load the experience base, overlaying recorded outcomes from the DB asset.

    Args:
        session: Optional DB session. When provided, outcomes recorded by
            :func:`record_outcome` are merged back onto the packaged seed.
        path: Optional path to an alternate experience YAML (default: packaged).

    Returns:
        A validated :class:`ExperienceBase`.

    Raises:
        ValueError: if the YAML (or its overlay) violates the taxonomy.
    """
    data = yaml.safe_load(_resolve(path).read_text(encoding="utf-8")) or {}
    base = _parse(data)
    base = _overlay_asset(base, session)
    errors = validate_experience_base(base)
    if errors:
        raise ValueError(
            "experience base validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    return base


def validate_experience_base(base: ExperienceBase) -> list[str]:
    """Return a list of validation errors (empty => valid).

    Reuses the taxonomy for tag / lever / evidence-strength checks and the rule
    base for ``linked_rule_id`` provenance, mirroring ``loader.validate_rules``.
    """
    taxonomy = load_taxonomy()
    rule_ids = {r.rule_id for r in load_rules()}
    errors: list[str] = []

    seen: set[str] = set()
    for intervention in base.interventions:
        iid = intervention.intervention_id
        if iid in seen:
            errors.append(f"duplicate intervention_id {iid!r}")
        seen.add(iid)

        bad_tags = [t for t in intervention.applicable_tags if t not in taxonomy.valid_tags]
        if bad_tags:
            errors.append(f"{iid}: unknown tags {bad_tags}")
        if not intervention.applicable_tags:
            errors.append(f"{iid}: empty applicable_tags")
        if intervention.category not in taxonomy.training_levers:
            errors.append(
                f"{iid}: category {intervention.category!r} not in training_levers"
            )
        if intervention.evidence_strength not in EVIDENCE_STRENGTH_ORDER:
            errors.append(
                f"{iid}: bad evidence_strength {intervention.evidence_strength!r}"
            )
        if intervention.source_type not in SOURCE_TYPES:
            errors.append(f"{iid}: bad source_type {intervention.source_type!r}")
        if intervention.linked_rule_id and intervention.linked_rule_id not in rule_ids:
            errors.append(
                f"{iid}: linked_rule_id {intervention.linked_rule_id!r} not in rules"
            )

        names = [d.name for d in intervention.datasets]
        if len(names) != len(set(names)):
            errors.append(f"{iid}: duplicate dataset names")
        knobs = [h.knob for h in intervention.hyperparameters]
        if len(knobs) != len(set(knobs)):
            errors.append(f"{iid}: duplicate hyperparameter knobs")

    for mode in base.failure_mode_capability_map:
        if mode not in taxonomy.failure_mode_ids:
            errors.append(f"failure_mode_capability_map: unknown mode {mode!r}")
    return errors


def record_outcome(
    session: Any,
    intervention_id: str,
    delta: float,
    note: str = "",
    *,
    base: ExperienceBase | None = None,
) -> str:
    """Append a measured outcome delta and persist a new experience asset version.

    Args:
        session: DB session (the new versioned asset is written here).
        intervention_id: Target intervention; must exist in the base.
        delta: Measured score delta (signed, e.g. +0.012 = +1.2 percentage points).
        note: Optional human note about the trial.
        base: Optional pre-loaded base (default: :func:`load_experience_base`).

    Returns:
        The new ``experience`` asset version id.

    Raises:
        ValueError: if ``intervention_id`` is unknown.
    """
    base = base or load_experience_base(session)
    intervention = base.by_id().get(intervention_id)
    if intervention is None:
        raise ValueError(f"unknown intervention_id {intervention_id!r}")
    outcome = Outcome(delta=delta, note=note, run_id=uuid.uuid4().hex[:8])
    intervention.outcomes.append(outcome)
    payload = [_to_dict(i) for i in base.interventions]
    return db.save_asset(
        session, "experience", payload, note=f"record outcome for {intervention_id}"
    )


def _resolve(path: str | Path | None) -> Path:
    return Path(path) if path else _default_experience_path()


def _parse(data: dict[str, Any]) -> ExperienceBase:
    fm_map = {
        str(k): [str(t) for t in (v or [])]
        for k, v in (data.get("failure_mode_capability_map") or {}).items()
    }
    interventions = [_parse_intervention(item) for item in data.get("interventions", [])]
    return ExperienceBase(
        version=int(data.get("version", 1)),
        interventions=interventions,
        failure_mode_capability_map=fm_map,
    )


def _parse_intervention(item: Any) -> Intervention:
    return Intervention(
        intervention_id=item["intervention_id"],
        title=item["title"],
        applicable_tags=list(item.get("applicable_tags", [])),
        category=item["category"],
        expected_effect=item.get("expected_effect", ""),
        evidence_strength=item.get("evidence_strength", "low"),
        source_type=item.get("source_type", "human_expert"),
        datasets=[_parse_dataset(d) for d in item.get("datasets", [])],
        hyperparameters=[_parse_hyperparameter(h) for h in item.get("hyperparameters", [])],
        linked_rule_id=item.get("linked_rule_id"),
        outcomes=[_parse_outcome(o) for o in item.get("outcomes", [])],
    )


def _parse_dataset(item: Any) -> DatasetSuggestion:
    return DatasetSuggestion(
        name=item["name"],
        rationale=item.get("rationale", ""),
        expected_effect=item.get("expected_effect", ""),
        source=item.get("source", ""),
    )


def _parse_hyperparameter(item: Any) -> HyperparameterSuggestion:
    return HyperparameterSuggestion(
        knob=item["knob"],
        direction=item.get("direction", ""),
        typical_range=item.get("typical_range", ""),
        rationale=item.get("rationale", ""),
    )


def _parse_outcome(item: Any) -> Outcome:
    return Outcome(
        delta=float(item["delta"]),
        note=item.get("note", ""),
        run_id=item.get("run_id"),
    )


def _to_dict(intervention: Intervention) -> dict[str, Any]:
    return {
        "intervention_id": intervention.intervention_id,
        "title": intervention.title,
        "applicable_tags": intervention.applicable_tags,
        "category": intervention.category,
        "expected_effect": intervention.expected_effect,
        "evidence_strength": intervention.evidence_strength,
        "source_type": intervention.source_type,
        "linked_rule_id": intervention.linked_rule_id,
        "datasets": [dataclasses.asdict(d) for d in intervention.datasets],
        "hyperparameters": [dataclasses.asdict(h) for h in intervention.hyperparameters],
        "outcomes": [dataclasses.asdict(o) for o in intervention.outcomes],
    }


def _overlay_asset(base: ExperienceBase, session: Any | None) -> ExperienceBase:
    """Merge recorded outcomes from the latest ``experience`` asset onto the seed.

    A mature asset may also carry interventions added by newer tool versions that
    the seed lacks; those are appended. The asset never removes seed content.
    """
    if session is None:
        return base
    payload = db.load_latest_asset(session, "experience")
    if not isinstance(payload, list):
        return base

    merged: list[Intervention] = list(base.interventions)
    by_id = {i.intervention_id: i for i in merged}
    for item in payload:
        if not isinstance(item, dict) or not item.get("intervention_id"):
            continue
        iid = item["intervention_id"]
        if iid in by_id:
            by_id[iid].outcomes = [_parse_outcome(o) for o in item.get("outcomes", [])]
        else:
            merged.append(_parse_intervention(item))
    return ExperienceBase(
        version=base.version,
        interventions=merged,
        failure_mode_capability_map=base.failure_mode_capability_map,
    )
