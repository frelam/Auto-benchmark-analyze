"""Load and validate the taxonomy and experience rule base.

The rule base is a *controlled* YAML asset: every rule's ``applicable_tags`` and
``category`` must fall within the taxonomy, otherwise the loader raises. This is
the user-augmentation interface from design doc section 6.1 — users edit
``rules.yaml`` and re-run validation rather than free-typing tags.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

_PKG_ROOT = files("benchmark_diagnosis")

EVIDENCE_STRENGTH_ORDER = {"high": 3, "medium": 2, "low": 1}


@dataclass
class Taxonomy:
    capabilities: list[str]
    training_levers: list[str]
    failure_modes: list[dict[str, Any]]  # each: {id, description, levers}

    @property
    def failure_mode_ids(self) -> set[str]:
        return {fm["id"] for fm in self.failure_modes}

    @property
    def valid_tags(self) -> set[str]:
        """All labels a rule may reference: capabilities + failure modes."""
        return set(self.capabilities) | self.failure_mode_ids


@dataclass
class Rule:
    rule_id: str
    title: str
    applicable_tags: list[str]
    category: str
    description: str
    source_type: str
    evidence_strength: str
    deprecated: bool = False


def _default_taxonomy_path() -> Path:
    return Path(_PKG_ROOT) / "recommendation" / "rule_base" / "taxonomy.yaml"


def _default_rules_path() -> Path:
    return Path(_PKG_ROOT) / "recommendation" / "rule_base" / "rules.yaml"


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    """Load the taxonomy; raise :class:`ValueError` on malformed content."""
    p = Path(path) if path else _default_taxonomy_path()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    required = ("capabilities", "training_levers", "failure_modes")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"taxonomy missing sections: {missing}")
    return Taxonomy(
        capabilities=list(data["capabilities"]),
        training_levers=list(data["training_levers"]),
        failure_modes=list(data["failure_modes"]),
    )


def load_rules(path: str | Path | None = None) -> list[Rule]:
    """Load raw rules (no taxonomy validation); see :func:`validate_rules`."""
    p = Path(path) if path else _default_rules_path()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    rules: list[Rule] = []
    for item in data.get("rules", []):
        rules.append(
            Rule(
                rule_id=item["rule_id"],
                title=item["title"],
                applicable_tags=list(item.get("applicable_tags", [])),
                category=item["category"],
                description=item.get("description", ""),
                source_type=item.get("source_type", "community"),
                evidence_strength=item.get("evidence_strength", "low"),
                deprecated=bool(item.get("deprecated", False)),
            )
        )
    return rules


def validate_rules(taxonomy: Taxonomy, rules: list[Rule]) -> list[str]:
    """Return a list of validation errors (empty => valid)."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    for rule in rules:
        if rule.rule_id in seen_ids:
            errors.append(f"duplicate rule_id {rule.rule_id!r}")
        seen_ids.add(rule.rule_id)

        bad_tags = [t for t in rule.applicable_tags if t not in taxonomy.valid_tags]
        if bad_tags:
            errors.append(f"{rule.rule_id}: unknown tags {bad_tags}")
        if not rule.applicable_tags:
            errors.append(f"{rule.rule_id}: empty applicable_tags")
        if rule.category not in taxonomy.training_levers:
            errors.append(
                f"{rule.rule_id}: category {rule.category!r} not in training_levers"
            )
        if rule.source_type not in {"internal_validated", "external_paper", "community"}:
            errors.append(f"{rule.rule_id}: bad source_type {rule.source_type!r}")
        if rule.evidence_strength not in EVIDENCE_STRENGTH_ORDER:
            errors.append(f"{rule.rule_id}: bad evidence_strength {rule.evidence_strength!r}")
    return errors


def load_validated_rules(path: str | Path | None = None) -> tuple[Taxonomy, list[Rule]]:
    """Load taxonomy + rules and validate; raises on first violation."""
    taxonomy = load_taxonomy(path)
    rules = load_rules(path)
    errors = validate_rules(taxonomy, rules)
    if errors:
        raise ValueError("rule base validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    return taxonomy, rules


def match_rules(rules: list[Rule], tags: list[str]) -> list[tuple[Rule, int]]:
    """Score rules by tag overlap, tie-broken by evidence strength.

    Returns (rule, score) sorted descending, including only rules that share at
    least one tag with ``tags``.
    """
    tag_set = set(tags)
    scored: list[tuple[Rule, int]] = []
    for rule in rules:
        if rule.deprecated:
            continue
        overlap = len(set(rule.applicable_tags) & tag_set)
        if overlap == 0:
            continue
        strength = EVIDENCE_STRENGTH_ORDER.get(rule.evidence_strength, 0)
        scored.append((rule, overlap * 10 + strength))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
