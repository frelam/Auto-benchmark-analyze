"""Re-score generative lm-eval samples whose ``get-answer`` filter dropped them.

Chat / RL-trained models (e.g. Qwen3-tool-rl templated via
``apply_chat_template``) emit answers wrapped in a reasoning + ``response``
block (`...\\nresponse\\n\\nA: <answer>`) and often conclude in prose such as
``Answer: invalid`` instead of the exact phrasing lm-eval's ``get-answer``
regex expects. The harness then marks those samples ``filtered_resps ==
["[invalid]"]`` and scores them 0, badly understating real capability.

This module re-derives an ``exact_match`` accuracy straight from the raw
``resps`` in each ``samples_*.jsonl``. Answer typing is strict to avoid the
false positives of naive substring matching:

* ``option``  — the gold is one token/letter (e.g. ``(B)``); the chosen option
  must literally appear parenthesized, or be the trailing token of the answer.
* ``enumerated`` — the gold is one of {yes/no, true/false, valid/invalid};
  the *last* token of the final answer must equal the gold (so ``not valid``
  is never miscounted).
* ``free-form`` (nums, lists, sequences) — normalized exact equality, falling
  back to an order-preserving token-subsequence match for long list answers.

Only tasks whose samples are dominated by ``["[invalid]"]`` filtered outputs
are re-scored, so tasks lm-eval parsed correctly are left untouched.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Marker(s) that introduce the final answer inside the model output.
_ANSWER_MARKER = re.compile(
    r"(?:So the answer is|[Aa]nswer)\s*[:：]?\s*([^\n<|]+)"
    r"|(?:\bA\s*[:：]\s*([^\n<|]+))",
    re.DOTALL,
)
_ENUM_SET = {"yes", "no", "true", "false", "valid", "invalid"}


def _response_block(raw: str) -> str:
    """Return the model's final ``response`` block, trimming the reasoning block."""
    idx = raw.rfind("response")
    return raw[idx + len("response"):] if idx != -1 else raw


def extract_final_answer(raw: str) -> str:
    """Extract the final answer phrase from a raw model generation.

    Uses the *last* answer marker in the response block; the generated tail
    after that marker is the answer phrase.
    """
    body = _response_block(raw)
    matches = [m for m in _ANSWER_MARKER.finditer(body)]
    if matches:
        last = matches[-1]
        candidate = last.group(1) or last.group(2)
        return (candidate or "").strip()
    return body.strip()


def _normalize(text: str) -> str:
    """Lowercase, keep letters/digits only, collapse whitespace."""
    text = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def _answer_kind(target: str) -> str:
    t = target.strip()
    if re.fullmatch(r"\(?[a-z0-9]\)?", t, re.IGNORECASE):
        return "option"
    if t.lower() in _ENUM_SET:
        return "enumerated"
    return "free"


def _option_letter(target: str) -> str:
    """The option token/letter itself, e.g. ``(B)`` -> ``b``, ``2`` -> ``2``."""
    m = re.search(r"[A-Za-z0-9]", target)
    return m.group(0).lower() if m else target.strip().lower()


def _matches(extracted: str, target: str) -> bool:
    """Strict per-type match between a final answer phrase and the gold."""
    if not extracted or not target:
        return False
    kind = _answer_kind(target)
    if kind == "option":
        letter = _option_letter(target)
        if not letter:
            return False
        if re.search(rf"\(\s*{re.escape(letter)}\s*\)", extracted, re.IGNORECASE):
            return True
        return _normalize(extracted).split()[-1:] == [letter]
    if kind == "enumerated":
        last = _normalize(extracted).split()
        if not last:
            return False
        if last[-1] != target.strip().lower():
            return False
        # "not valid" / "not true" is a negation, never a match for the bare gold.
        return not (len(last) >= 2 and last[-2] == "not")
    # free-form
    e, t = _normalize(extracted), _normalize(target)
    if not e or not t:
        return False
    if e == t:
        return True
    ew, tw = e.split(), t.split()
    if not tw:
        return False
    # order-preserving subsequence (tolerates list wrapping / extra punctuation)
    it = iter(ew)
    return all(any(w == tok for w in it) for tok in tw) if tw else False


def re_score_samples(samples: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Return ``(n_correct, n, n_invalid)`` over one task's samples."""
    n_correct = n = n_invalid = 0
    for sample in samples:
        target = sample.get("target")
        if target is None:
            doc = sample.get("doc") or {}
            target = doc.get("target") or doc.get("gold")
        if target is None:
            continue
        n += 1
        if (sample.get("filtered_resps") or [None]) == ["[invalid]"]:
            n_invalid += 1
        agents = (sample.get("resps") or [[None]])[0] or [None]
        good = any(
            _matches(extract_final_answer(str(a)), str(target))
            for a in agents if a is not None
        )
        n_correct += int(good)
    return n_correct, n, n_invalid


def _base_task(filepath: Path) -> str:
    """``samples_<TASK>_<ts>.jsonl`` -> ``<TASK>`` (strip timestamp suffix)."""
    stem = Path(filepath).name[len("samples_"):-len(".jsonl")]
    return re.sub(r"_\d{4}-\d{2}-\d{2}T.*$", "", stem)


def reparse_scores(
    sample_files: dict[str, list[Path]],
    raw_scores: dict[str, float],
    *,
    min_invalid_fraction: float = 0.3,
) -> dict[str, float]:
    """Re-derive exact_match scores from sample logs where the filter dropped answers.

    Args:
        sample_files: ``task -> [paths]`` from
            :func:`artifacts.find_sample_files <benchmark_diagnosis.evaluation_orchestration.artifacts.find_sample_files>`.
        raw_scores: ``task_id -> score`` (the lm-eval headline metric).
        min_invalid_fraction: tasks where fewer than this share of samples were
            ``["[invalid]"]`` filtered are left untouched (lm-eval parsed them).

    Returns:
        ``{task_id: corrected_score}`` for every re-scored task plus any group
        aggregate recomputed from them (e.g. ``bbh`` <- mean of its subtasks).
        Values equal a raw score when the task was left untouched.
    """
    corrected: dict[str, float] = {}
    re_scored_ids: set[str] = set()
    re_scored: dict[str, float] = {}

    for task, paths in sample_files.items():
        for path in paths:
            base = _base_task(path)
            if base not in raw_scores:
                continue
            samples = _load_samples(path)
            n_correct, n, n_invalid = re_score_samples(samples)
            if n <= 0:
                continue
            invalid_frac = n_invalid / n
            if invalid_frac < min_invalid_fraction:
                continue  # lm-eval parsed these; keep its score
            corrected[base] = n_correct / n
            re_scored_ids.add(base)
            re_scored[base] = n_correct / n

    # Recompute group aggregates (e.g. ``bbh``) as the mean over re-scored
    # subtask ids that share the group as a prefix.
    for tid in raw_scores:
        if tid in re_scored_ids:
            continue
        members = [rs for rs in re_scored_ids if rs.startswith(tid + "_")]
        if members:
            corrected[tid] = sum(re_scored[m] for m in members) / len(members)

    return corrected


def _load_samples(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out