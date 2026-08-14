"""LLM-as-analyst failure-mode classification (design doc section 5.2).

When label slicing cannot localize the problem, we sample failed cases and ask
an analyst LLM to classify each into EXACTLY ONE failure mode drawn from the
fixed taxonomy (never free-form labels), so cross-run trends aggregate.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from benchmark_diagnosis.recommendation.rule_base.loader import Taxonomy

# Cases sent to the analyst LLM per chat round; the response maps each case by
# its per-batch index (0..BATCH_SIZE-1).
_BATCH_SIZE = 10

UNCLASSIFIED = "unclassified"


def classify_failures(
    llm: Any,
    taxonomy: Taxonomy,
    cases: list[dict],
    *,
    sample_size: int = 50,
) -> dict[str, float]:
    """Classify sampled failed cases into taxonomy failure modes.

    Args:
        llm: Duck-typed analyst LLM exposing ``complete_json(messages) -> Any``.
        taxonomy: Validated :class:`Taxonomy` whose ``failure_modes`` provide the
            allowed ids (``{id, description, levers}``) and ``failure_mode_ids``
            the valid set.
        cases: Item-level failed cases, each with ``question``, ``model_output``
            and ``gold`` keys.
        sample_size: Maximum number of cases to classify (sampled as the first
            ``sample_size`` items for determinism).

    Returns:
        Mapping ``{failure_mode_id: fraction}`` over the sampled cases, including
        ``"unclassified"`` for cases whose LLM answer was invalid or absent.
        Only ids with fraction > 0 are returned; empty ``cases`` yields ``{}``.
    """
    sampled = cases[:sample_size]
    if not sampled:
        return {}

    counts: Counter[str] = Counter()
    for start in range(0, len(sampled), _BATCH_SIZE):
        batch = sampled[start : start + _BATCH_SIZE]
        ids = _classify_batch(llm, taxonomy, batch)
        counts.update(ids)

    total = len(sampled)
    return {
        mode_id: count / total
        for mode_id, count in counts.items()
        if count > 0
    }


def _build_classification_messages(taxonomy: Taxonomy, batch: list[dict]) -> list[dict[str, str]]:
    """Build a system+user chat pair constraining the model to taxonomy ids."""
    mode_lines = "\n".join(
        f"- {fm['id']}: {fm.get('description', '')}" for fm in taxonomy.failure_modes
    )
    case_lines = "\n".join(
        f"[{index}] question: {case.get('question', '')}\n"
        f"    model_output: {case.get('model_output', '')}\n"
        f"    gold: {case.get('gold', '')}"
        for index, case in enumerate(batch)
    )
    system = (
        "You are an LLM failure analyst. Classify each model output into EXACTLY "
        "ONE failure mode id chosen from the taxonomy below. Never invent ids. "
        'Reply with JSON of the form {"classifications": '
        '[{"case_index": 0, "failure_mode": "<id>"}, ...]} covering every case.'
    )
    user = f"TAXONOMY:\n{mode_lines}\n\nCASES:\n{case_lines}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _classify_batch(llm: Any, taxonomy: Taxonomy, batch: list[dict]) -> list[str]:
    """Classify one batch; any failure to parse maps to ``unclassified``."""
    try:
        data = llm.complete_json(_build_classification_messages(taxonomy, batch))
    except Exception:
        return [UNCLASSIFIED] * len(batch)
    single = len(batch) == 1
    return [
        _extract_failure_mode(data, index, taxonomy.failure_mode_ids, single=single)
        for index in range(len(batch))
    ]


def _extract_failure_mode(
    data: Any,
    case_index: int,
    valid_ids: set[str],
    *,
    single: bool = False,
) -> str:
    """Extract a valid failure-mode id for ``case_index`` from the LLM response."""
    if not isinstance(data, dict):
        return UNCLASSIFIED

    classifications = data.get("classifications")
    if isinstance(classifications, list):
        for entry in classifications:
            if (
                isinstance(entry, dict)
                and entry.get("case_index") == case_index
                and isinstance(entry.get("failure_mode"), str)
                and entry["failure_mode"] in valid_ids
            ):
                return entry["failure_mode"]
        return UNCLASSIFIED

    # Tolerate the single-case shape {"failure_mode": "<id>"} when we asked for one case.
    if single:
        mode = data.get("failure_mode")
        if isinstance(mode, str) and mode in valid_ids:
            return mode
    return UNCLASSIFIED
