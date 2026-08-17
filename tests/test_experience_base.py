"""Tests for the tool-maintained experience base (seed + outcome feedback loop)."""

from __future__ import annotations

import pytest

from benchmark_diagnosis.core import db
from benchmark_diagnosis.recommendation.experience_base import (
    ExperienceBase,
    HyperparameterSuggestion,
    Intervention,
    Outcome,
    load_experience_base,
    record_outcome,
    validate_experience_base,
)
from benchmark_diagnosis.recommendation.rule_base.loader import load_validated_rules

_MINIMAL_YAML = """
version: 2
failure_mode_capability_map:
  reasoning_error: [reasoning, math]
interventions:
  - intervention_id: EXP-minimal-001
    title: minimal
    applicable_tags: [reasoning]
    category: data_mix
    expected_effect: "+1"
    evidence_strength: low
    source_type: human_expert
    datasets:
      - {name: GSM8K, rationale: "reasoning steps", expected_effect: "+1"}
    hyperparameters:
      - {knob: temperature, direction: lower, typical_range: "0.0-0.3"}
"""


def _base(**overrides) -> ExperienceBase:
    defaults: dict = dict(
        version=1,
        interventions=[
            Intervention(
                intervention_id="EXP-a",
                title="a",
                applicable_tags=["math"],
                category="data_mix",
                expected_effect="+1",
                evidence_strength="low",
                source_type="human_expert",
            )
        ],
        failure_mode_capability_map={"calculation_error": ["math"]},
    )
    defaults.update(overrides)
    return ExperienceBase(**defaults)


@pytest.fixture()
def session(tmp_path):
    engine = db.make_engine(tmp_path / "test.db")
    db.init_db(engine)
    return db.session_factory(engine)()


# ---------------------------------------------------------------------------
# load / parse / validate
# ---------------------------------------------------------------------------


def test_load_packaged_base_is_valid() -> None:
    base = load_experience_base()
    assert base.version >= 1
    assert base.interventions
    assert base.failure_mode_capability_map
    assert validate_experience_base(base) == []


def test_load_custom_path(tmp_path) -> None:
    path = tmp_path / "minimal_experience.yaml"
    path.write_text(_MINIMAL_YAML, encoding="utf-8")
    base = load_experience_base(path=path)
    assert base.version == 2
    assert base.by_id()["EXP-minimal-001"].datasets[0].name == "GSM8K"
    assert base.failure_mode_capability_map == {"reasoning_error": ["reasoning", "math"]}


def test_validate_rejects_unknown_tag() -> None:
    intervention = Intervention(
        intervention_id="EXP-bad",
        title="bad",
        applicable_tags=["not_a_capability"],
        category="data_mix",
        expected_effect="",
        evidence_strength="low",
        source_type="human_expert",
    )
    errors = validate_experience_base(_base(interventions=[intervention]))
    assert any("unknown tags" in e for e in errors)


def test_validate_rejects_duplicate_id() -> None:
    errors = validate_experience_base(_base(interventions=_base().interventions * 2))
    assert any("duplicate intervention_id" in e for e in errors)


def test_validate_rejects_bad_category() -> None:
    intervention = Intervention(
        intervention_id="EXP-bad",
        title="bad",
        applicable_tags=["math"],
        category="not_a_lever",
        expected_effect="",
        evidence_strength="low",
        source_type="human_expert",
    )
    errors = validate_experience_base(_base(interventions=[intervention]))
    assert any("not in training_levers" in e for e in errors)


def test_validate_rejects_unknown_linked_rule() -> None:
    intervention = Intervention(
        intervention_id="EXP-bad",
        title="bad",
        applicable_tags=["math"],
        category="data_mix",
        expected_effect="",
        evidence_strength="low",
        source_type="human_expert",
        linked_rule_id="R-does-not-exist",
    )
    errors = validate_experience_base(_base(interventions=[intervention]))
    assert any("linked_rule_id" in e for e in errors)


def test_validate_rejects_duplicate_datasets() -> None:
    from benchmark_diagnosis.recommendation.experience_base import DatasetSuggestion

    intervention = Intervention(
        intervention_id="EXP-bad",
        title="bad",
        applicable_tags=["math"],
        category="data_mix",
        expected_effect="",
        evidence_strength="low",
        source_type="human_expert",
        datasets=[DatasetSuggestion("GSM8K", "", ""), DatasetSuggestion("GSM8K", "", "")],
    )
    errors = validate_experience_base(_base(interventions=[intervention]))
    assert any("duplicate dataset names" in e for e in errors)


def test_validate_packaged_rules_are_referenced() -> None:
    """Every packaged linked_rule_id must exist in the rule base."""
    _taxonomy, rules = load_validated_rules()
    rule_ids = {r.rule_id for r in rules}
    base = load_experience_base()
    linked = {i.linked_rule_id for i in base.interventions if i.linked_rule_id}
    assert linked <= rule_ids


# ---------------------------------------------------------------------------
# outcome feedback loop
# ---------------------------------------------------------------------------


def test_mean_delta_none_without_outcomes() -> None:
    intervention = _base().interventions[0]
    assert intervention.mean_delta is None


def test_mean_delta_averages_outcomes() -> None:
    intervention = Intervention(
        intervention_id="EXP-x",
        title="x",
        applicable_tags=["math"],
        category="data_mix",
        expected_effect="",
        evidence_strength="low",
        source_type="human_expert",
        outcomes=[Outcome(0.02), Outcome(0.01)],
    )
    assert intervention.mean_delta == pytest.approx(0.015)


def test_record_outcome_round_trip(session) -> None:
    base = load_experience_base(session)
    before = len(base.interventions)
    version_id = record_outcome(session, "EXP-longctx-001", 0.031, note="first trial")

    # A new experience asset version was written and the latest load sees it.
    assert db.load_asset_by_version(session, "experience", version_id) is not None
    reloaded = load_experience_base(session)
    assert len(reloaded.interventions) == before
    intervention = reloaded.by_id()["EXP-longctx-001"]
    assert intervention.mean_delta is not None
    assert intervention.outcomes[-1].delta == pytest.approx(0.031)
    assert intervention.outcomes[-1].note == "first trial"
    assert intervention.outcomes[-1].run_id


def test_record_outcome_unknown_id_raises(session) -> None:
    with pytest.raises(ValueError, match="unknown intervention_id"):
        record_outcome(session, "EXP-nope-999", 0.01)


def test_overlay_appends_interventions_from_asset(session) -> None:
    fresh = load_experience_base()  # no session -> seed only
    seed_ids = {i.intervention_id for i in fresh.interventions}

    extra = Intervention(
        intervention_id="EXP-added-001",
        title="added by newer version",
        applicable_tags=["safety"],
        category="data_mix",
        expected_effect="",
        evidence_strength="low",
        source_type="human_expert",
        hyperparameters=[HyperparameterSuggestion("k", "up", "1-2", "r")],
    )
    payload = [_to_dict(fresh.interventions[0]), _to_dict(extra)]
    db.save_asset(session, "experience", payload)

    loaded = load_experience_base(session)
    assert seed_ids <= {i.intervention_id for i in loaded.interventions}
    assert "EXP-added-001" in loaded.by_id()
    assert loaded.by_id()["EXP-added-001"].hyperparameters[0].knob == "k"


def _to_dict(intervention: Intervention) -> dict:
    import dataclasses

    return dataclasses.asdict(intervention)
