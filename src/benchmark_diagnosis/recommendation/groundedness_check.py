"""Post-hoc hallucination check on synthesized recommendations (design doc section 6.4).

The check verifies, automatically and independent of prompt wording, that every
cited rule id / external source actually exists in the evidence set used for this
diagnosis, and that every number appearing in the diagnosis basis matches one of
the passed numbers. Items that fail the check must be dropped or rewritten — the
pipeline never trusts a "please don't fabricate" prompt alone.
"""

from __future__ import annotations

import re
from typing import Any

# Matches standalone integers and decimal numbers in the diagnosis-basis prose.
# Token-boundary lookarounds keep identifier digits (cluster "C1", "V2", "GSM8K",
# model "8b") out of the check — only numbers that read as quantities count.
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")

_TOLERANCE = 1e-6


def check_groundedness(synthesis: dict, evidence: dict) -> list[str]:
    """Return violation strings for a synthesis that is not fully grounded.

    Args:
        synthesis: Output of :func:`synthesize` with ``diagnosis_basis`` prose and
            a list of ``actions`` (each ``{rule_id, intervention_id, source,
            action, validation_experiment}``).
        evidence: ``{"rule_ids": set[str], "sources": set[str],
            "numbers": set[float]}`` plus the optional ``"intervention_ids":
            set[str]`` — everything the LLM was allowed to reference.

    Returns:
        A list of human-readable violation strings; an empty list means the
        synthesis is fully grounded.
    """
    violations: list[str] = []
    rule_ids = _as_set(evidence.get("rule_ids"))
    sources = _as_set(evidence.get("sources"))
    numbers = _as_set(evidence.get("numbers"))
    intervention_ids = _as_set(evidence.get("intervention_ids"))

    actions = synthesis.get("actions", [])
    if not isinstance(actions, list):
        actions = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        rule_id = action.get("rule_id")
        if rule_id is not None:
            if rule_id not in rule_ids:
                violations.append(
                    f"action[{index}]: rule_id {rule_id!r} not in evidence"
                )
            source = action.get("source")
            if (
                isinstance(source, str)
                and source.startswith("rule_base:")
                and source not in sources
            ):
                violations.append(f"action[{index}]: source {source!r} not in evidence")
        intervention_id = action.get("intervention_id")
        if intervention_id is not None:
            if intervention_id not in intervention_ids:
                violations.append(
                    f"action[{index}]: intervention_id {intervention_id!r} not in evidence"
                )
            source = action.get("source")
            if (
                isinstance(source, str)
                and source.startswith("experience:")
                and source not in sources
            ):
                violations.append(f"action[{index}]: source {source!r} not in evidence")

    basis = synthesis.get("diagnosis_basis", "")
    if isinstance(basis, str):
        for match in _NUMBER_RE.findall(basis):
            value = float(match)
            if not any(abs(value - number) <= _TOLERANCE for number in numbers):
                violations.append(
                    f"diagnosis_basis: number {value!r} not in evidence"
                )
    return violations


def _as_set(value: Any) -> set[Any]:
    """Coerce an evidence field to a set, tolerating lists/sets/None."""
    if value is None:
        return set()
    if isinstance(value, set):
        return value
    if isinstance(value, (list, tuple, frozenset)):
        return set(value)
    return {value}
