"""LLM synthesis of grounded optimization recommendations (design doc section 6.3).

The LLM is restricted to *selecting, ordering, combining and explaining* from the
provided evidence (diagnosis + matched internal rules + retrieved snippets) — it
never freely generates rule ids. A deterministic rule-based synthesis is used as
a fallback whenever the LLM is unavailable or returns unusable JSON, so the
pipeline never produces an empty recommendation.
"""

from __future__ import annotations

from typing import Any

from benchmark_diagnosis.core.types import DiagnosisResult
from benchmark_diagnosis.recommendation.rule_base.loader import Rule, match_rules

_MAX_FALLBACK_ACTIONS = 3

_FALLBACK_VALIDATION = (
    "Run a small-scale ablation: pick the worst sub-capability cluster, apply the "
    "action on a bounded subset (e.g. 1k-2k sampled examples) for a short RL/SFT "
    "run, and compare accuracy / quantified gap before and after against a "
    "held-out eval set of the same cluster."
)


def synthesize(
    llm: Any,
    diagnosis: DiagnosisResult,
    rules: list[Rule],
    retrieved: list[dict],
) -> dict:
    """Synthesize a grounded recommendation dict from diagnosis + rule evidence.

    Args:
        llm: Duck-typed analyst LLM exposing ``complete_json(messages) -> Any``.
        diagnosis: Structured diagnosis (cluster, failure-mode distribution,
            representative cases, quantified gap).
        rules: Matched internal rules (id + title + description + evidence
            strength) the LLM may select from.
        retrieved: External snippets (may be empty).

    Returns:
        A normalized dict with:
            ``diagnosis_basis``: prose restating the diagnosis with key numbers;
            ``actions``: list of ``{rule_id, source, action, validation_experiment}``.
        If the LLM call raises or returns unusable JSON, a deterministic
        rule-based synthesis (one action per top matched rule, up to 3) is
        returned instead.
    """
    messages = _build_synthesis_messages(diagnosis, rules, retrieved)
    try:
        raw = llm.complete_json(messages)
    except Exception:
        raw = None

    if raw is None or not isinstance(raw, dict):
        return _deterministic_synthesis(diagnosis, rules)
    return _normalize_synthesis(raw, diagnosis)


def _build_synthesis_messages(
    diagnosis: DiagnosisResult,
    rules: list[Rule],
    retrieved: list[dict],
) -> list[dict[str, str]]:
    """Build the system+user prompt constraining the LLM to the evidence set."""
    rules_text = "\n".join(
        f"- [{rule.rule_id}] ({rule.evidence_strength}) {rule.title}\n  {rule.description}"
        for rule in rules
    )
    retrieved_text = "\n".join(f"- {_format_snippet(snippet)}" for snippet in retrieved) or "(none)"
    system = (
        "You select, order, combine and explain optimization recommendations for an "
        "LLM training team. Base every action ONLY on the provided diagnosis, "
        "internal rules, and retrieved snippets; never invent rule ids, sources or "
        "numbers. Reply with JSON of the form "
        '{"diagnosis_basis": "<prose restating the diagnosis with its key '
        'numbers>", "actions": [{"rule_id": "<id or null>", "source": '
        '"rule_base:<id>" or "external:<url>", "action": "<text>", '
        '"validation_experiment": "<small-scale validation design>"}]}.'
    )
    user = (
        f"DIAGNOSIS:\n{_build_diagnosis_basis(diagnosis)}\n\n"
        f"MATCHED RULES (id, evidence_strength, title, description):\n{rules_text}\n\n"
        f"RETRIEVED EXTERNAL SNIPPETS:\n{retrieved_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _format_snippet(snippet: Any) -> str:
    """Render one retrieved snippet dict (or any value) for the prompt."""
    if isinstance(snippet, dict):
        title = snippet.get("title", "")
        body = snippet.get("snippet", snippet.get("text", ""))
        return f"{title}: {body}" if title else str(body)
    return str(snippet)


def _build_diagnosis_basis(diagnosis: DiagnosisResult) -> str:
    """Deterministic prose restating the diagnosis with its key numbers."""
    parts = [f"cluster {diagnosis.cluster_id}"]
    if diagnosis.sub_capability:
        parts.append(f"sub-capability {diagnosis.sub_capability}")
    if diagnosis.quantified_gap is not None:
        parts.append(f"quantified gap {diagnosis.quantified_gap:.3f}")
    distribution = ", ".join(
        f"{mode}={fraction:.3f}"
        for mode, fraction in sorted(
            diagnosis.failure_modes.items(), key=lambda item: item[1], reverse=True
        )
    )
    if distribution:
        parts.append(f"failure modes [{distribution}]")
    return "; ".join(parts)


def _normalize_synthesis(raw: dict, diagnosis: DiagnosisResult) -> dict:
    """Coerce a possibly-malformed LLM response into the required dict shape."""
    basis = raw.get("diagnosis_basis")
    if not isinstance(basis, str) or not basis.strip():
        basis = _build_diagnosis_basis(diagnosis)

    actions: list[dict] = []
    raw_actions = raw.get("actions")
    if isinstance(raw_actions, list):
        for item in raw_actions:
            normalized = _normalize_action(item)
            if normalized is not None:
                actions.append(normalized)
    return {"diagnosis_basis": basis, "actions": actions}


def _normalize_action(action: Any) -> dict | None:
    """Normalize one action entry, filling defaults for missing fields."""
    if not isinstance(action, dict):
        return None
    rule_id = action.get("rule_id")
    if rule_id is not None and not isinstance(rule_id, str):
        rule_id = None

    source = action.get("source")
    if not isinstance(source, str) or not source:
        source = f"rule_base:{rule_id}" if rule_id else "external:llm"

    action_text = action.get("action")
    if not isinstance(action_text, str) or not action_text:
        action_text = ""

    validation = action.get("validation_experiment")
    if not isinstance(validation, str) or not validation:
        validation = ""

    return {
        "rule_id": rule_id,
        "source": source,
        "action": action_text,
        "validation_experiment": validation,
    }


def _deterministic_synthesis(diagnosis: DiagnosisResult, rules: list[Rule]) -> dict:
    """No-LLM fallback: one action per top matched rule, up to 3."""
    tags = [mode for mode, fraction in diagnosis.failure_modes.items() if fraction > 0]
    top_rules = match_rules(rules, tags)
    actions = [
        {
            "rule_id": rule.rule_id,
            "source": f"rule_base:{rule.rule_id}",
            "action": rule.description,
            "validation_experiment": _FALLBACK_VALIDATION,
        }
        for rule, _score in top_rules[:_MAX_FALLBACK_ACTIONS]
    ]
    return {"diagnosis_basis": _build_diagnosis_basis(diagnosis), "actions": actions}
