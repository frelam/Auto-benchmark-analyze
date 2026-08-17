"""Intelligent benchmark diagnosis: stages 1-7 (design doc v2).

Package layout mirrors the design stages:

* ``capability_taxonomy`` — hierarchical, versioned capability vocabulary;
* ``candidate_generation`` — Stage 1 candidate deficit capabilities;
* ``probe_registry`` — Stage 2 single-capability probe verification;
* ``guided_case_analyzer`` — Stage 3 guided bad-case analysis;
* ``confidence_fusion`` — Stage 4 confidence + suggestion-type rules;
* ``priority_scorer`` — Stage 5 ExpectedGain / Cost / Priority;
* ``suggestion_writer`` — Stage 6 concrete suggestion write-up;
* ``feedback`` — Stage 7 execution feedback + recalibration;
* ``orchestrator`` — wires the stages into one report block.
"""

from benchmark_diagnosis.intelligent_diagnosis.orchestrator import (
    run_intelligent_diagnosis,
)

__all__ = ["run_intelligent_diagnosis"]
