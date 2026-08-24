"""Diagnosis engines: rule base (2.1) and LLM agent base (2.2).

``rule_base`` is the deterministic statistics-only path; ``llm_agent`` runs
the rule base first, then drives a harness agent through bad-case analysis and
an analysis->verification->hypothesis loop (the workflow ships as a skill in
``skill/benchmark-diagnosis/``).
"""

from benchmark_diagnosis.diagnosis_engine.llm_agent import (
    AgentRunResult,
    run_llm_agent_diagnosis,
)
from benchmark_diagnosis.diagnosis_engine.rule_base import (
    RuleBaseResult,
    run_rule_base,
)

__all__ = [
    "AgentRunResult",
    "RuleBaseResult",
    "run_llm_agent_diagnosis",
    "run_rule_base",
]
