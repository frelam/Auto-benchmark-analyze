"""Tests for the strict answer-reparse scorer."""

from __future__ import annotations

import json

import pytest

from benchmark_diagnosis.evaluation_orchestration.answer_reparse import (
    extract_final_answer,
    re_score_samples,
    reparse_scores,
)

CHAT_STYLE = (
    " thinking\nLet me reason this out.\n response\n\nA: (B) 12/25/1937"
)


def _sample(raw: str, target: str, gold: str | None = None, filtered=None):
    return {
        "target": target,
        "doc": {"gold": gold} if gold is not None else {},
        "resps": [[raw]],
        "filtered_resps": filtered if filtered is not None else ["[invalid]"],
    }


def test_extract_final_answer_reads_after_response_block():
    assert extract_final_answer(CHAT_STYLE) == "(B) 12/25/1937"
    assert extract_final_answer("response\n\nA: (C)") == "(C)"


def test_option_letter_match_parenthesized():
    samples = [
        _sample("A: (B) 12/25/1937", "(B)", filtered=["[invalid]"]),
        _sample("A: (A)", "(A)", filtered=["[invalid]"]),
    ]
    n_correct, n, _ = re_score_samples(samples)
    assert n == 2
    assert n_correct == 2


def test_option_mismatch_not_counted():
    samples = [_sample("A: (A)", "(B)", filtered=["[invalid]"])]
    n_correct, n, _ = re_score_samples(samples)
    assert n_correct == 0


def test_enumerated_uses_last_token():
    # "Answer: not valid" must NOT match target "valid".
    bad = [_sample("A: not valid", "valid", filtered=["[invalid]"])]
    assert re_score_samples(bad)[0] == 0
    good = [_sample("A: valid", "valid", filtered=["[invalid]"])]
    assert re_score_samples(good)[0] == 1


def test_enumerated_accepts_leading_verdict_in_sentence():
    # sports_understanding gold is yes/no but the model reasons in prose
    # ("A: Yes. John ... hockey"), so the verdict is the first token.
    cases = [
        ("A: Yes.", "yes"),
        ("A: No.", "no"),
        ("A: Yes. John Carlson is a hockey player.", "yes"),
        ("A: No. As of my knowledge this is wrong.", "no"),
    ]
    good = [_sample(resp, gold) for resp, gold in cases]
    assert re_score_samples(good)[0] == len(cases)
    assert re_score_samples(good)[2] == len(cases)  # all were [invalid] before


def test_free_form_keeps_brackets_as_data():
    # dyck_languages gold is a closing-bracket sequence; brackets must not be
    # stripped, otherwise the answer normalizes to empty and never matches.
    cases = [
        ("A: ] ]", "] ]"),
        ("A: ] ] >", "] ] >"),
        ("A: ) >", ") >"),
    ]
    bad = [_sample("A: ] ] ] >", "] ] >", filtered=["[invalid]"])]
    good = [_sample(resp, gold, filtered=["[invalid]"]) for resp, gold in cases]
    assert re_score_samples(good)[0] == len(cases)
    assert re_score_samples(bad)[0] == 0


def test_free_form_exact_and_subsequence():
    exact = [_sample("A: 42", "42", filtered=["[invalid]"])]
    assert re_score_samples(exact)[0] == 1
    subseq = [_sample("A: [3, 1, 2]", "3 1 2", filtered=["[invalid]"])]
    assert re_score_samples(subseq)[0] == 1
    wrong = [_sample("A: 5", "42", filtered=["[invalid]"])]
    assert re_score_samples(wrong)[0] == 0


def test_reparse_scores_leaves_well_parsed_unmodified(tmp_path):
    # A task lm-eval parsed correctly (no [invalid]) must be left untouched.
    task = "bbh_cot_fewshot_boolean_expressions"
    sample_file = tmp_path / f"samples_{task}_2026-01-01T00-00-00.000000.jsonl"
    sample_file.write_text(
        json.dumps(_sample("A: yes", "yes", filtered=["yes"])) + "\n"
        * 20,
        encoding="utf-8",
    )
    sample_files = {f"{task}_2026-01-01T00-00-00.000000": [sample_file]}
    raw = {"bbh_cot_fewshot_boolean_expressions": 0.868}
    out = reparse_scores(sample_files, raw)
    # untouched tasks are not keys in the returned (rewritten-only) map, so the
    # caller's merge keeps the raw value.
    assert task not in out


def test_reparse_scores_rewrites_invalid_dominated_and_aggregate(tmp_path):
    tasks = ["bbh_cot_fewshot_date_understanding", "bbh_cot_fewshot_navigate"]
    raw = {t: 0.05 for t in tasks}
    raw["bbh"] = 0.05
    sample_files: dict[str, list] = {}
    for t in tasks:
        p = tmp_path / f"samples_{t}_2026-01-01T00-00-00.000000.jsonl"
        p.write_text(
            json.dumps(_sample("A: (B)", "(B)")) + "\n"
            * 20,
            encoding="utf-8",
        )
        sample_files[f"{t}_2026-01-01T00-00-00.000000"] = [p]
    out = reparse_scores(sample_files, raw)
    assert out["bbh_cot_fewshot_date_understanding"] == 1.0
    assert out["bbh_cot_fewshot_navigate"] == 1.0
    # aggregate recomputed as the mean over its re-scored subtasks
    assert out["bbh"] == 1.0