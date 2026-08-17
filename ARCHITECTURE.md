# Architecture & Technical Decisions

This repo implements the system specified in `benchmark-diagnosis-tool-design.md`
(seven layers: data → capability coverage → representative selection → evaluation
orchestration → diagnosis → recommendation → reporting). This document records the
concrete technical decisions, biased toward **reusing mature open-source tools**
and only writing the parts that are unique to this project.

## 1. Evaluation bridge — reuse `lm-evaluation-harness`

- **Decision**: use [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) as the evaluation engine; we only write a thin bridge (`evaluation_orchestration/harness_bridge.py`).
- **Why**: it is the de-facto standard, has a large benchmark registry, and natively
  supports OpenAI-compatible endpoints via the `local-completions` /
  `local-chat-completions` model types with a `base_url` argument — exactly matching
  the requirement "the user only gives an inference-service IP".
- **Bridge mechanism**: we invoke it as a **subprocess** (`lm_eval ...`) and parse its
  `results.json`, rather than importing its internal Python API. Its API changes often;
  the CLI is the stable contract.
- **Item-level data**: `--log_samples` / `samples_*.jsonl` gives per-example results,
  which feed the mIRT module (section 2 of the design doc).

## 2. Weight deployment — reuse `vLLM`

- **Decision**: `evaluation_orchestration/deploy.py` shells out to `vllm serve <model_id>`
  to stand up an OpenAI-compatible server on a configured port; the bridge then evaluates
  it through the *same* `base_url` path used for a user-provided endpoint.
- **Why**: vLLM is the most widely supported serving engine and exposes OpenAI-compatible
  `/v1` endpoints, so "deploy weights" and "user gives an IP" collapse into one uniform
  evaluation path. `serving.engine` is configurable (SGLang as a future option).
- **Health check**: poll `/v1/models` until ready before running the harness.

## 3. Data layer — SQLite + SQLAlchemy 2.0

- **Decision**: SQLAlchemy 2.0 ORM over SQLite (zero-config, swap to Postgres later).
- Tables mirror design-doc section 1.2 (`model_registry`, `benchmark_registry`,
  `item_registry`, `score_records`) plus an `assets` table for **versioned** offline
  artifacts (coverage table, cluster registry, expectation curves). Every report records
  the asset versions it used (section 2.5 / 4.2.5 traceability).

## 4. Capability coverage — mIRT with a factor-analysis fallback

- **Decision**: the primary method is the two-tower multidimensional IRT model from the
  design doc (section 2.1), implemented directly in PyTorch (`torch` is an **optional**
  extra). There is no maintained multi-dimensional IRT library (`py-irt` is 1-D), so this
  is one of the "special" parts we write ourselves. When item-level data is absent, we
  fall back to aggregate factor analysis (PCA + clustering) — the Phase-1 simplification.
- **Dimension selection**: held-out item-prediction AUC over candidate `d` (section 2.2).
- **Judge-bias control**: `llm_judge` benchmarks are projected into the space built from
  `rule_verified` items (section 2.3).

## 5. Expectation curves — two complementary fits (section 4.2)

- **Group A** (params → score): log-linear fit of `logit(score)` vs `log(params)`, fitted
  separately for dense/MoE and a unified active-params curve.
- **Group B** (time → score): frontier envelope over release date, so closed-source models
  with unknown parameter counts are still included.
- Residuals are converted to a **local percentile / z-score** (not absolute deltas) and
  compared against configurable thresholds (default p25 / z < −1).

## 6. Diagnosis — fixed taxonomy, LLM-as-analyst

- `diagnosis/label_slicing.py` aggregates eval logs by declared subcategory first (cheap).
- `diagnosis/failure_mode_analyst.py` classifies sampled failures into a **fixed taxonomy**
  (never free-form labels) so cross-run trends aggregate.

## 7. Recommendation — grounded, not free-form

- Experience rule base (`recommendation/rule_base/`) is a controlled YAML with a
  validation loader (section 6.1 user-augmentation interface).
- The LLM only *selects, orders, combines, explains* — never freely generates — and a
  `groundedness_check.py` post-processor verifies every cited rule id and number exists in
  the evidence set (section 6.4).

## 8. Intelligent diagnosis — stages 1-7 (design doc v2)

- `intelligent_diagnosis/` implements the v2 pipeline (see
  `docs/intelligent-diagnosis-v2-design.md`): candidate generation →
  probe verification → guided bad-case analysis → confidence fusion →
  priority scoring → suggestion write-up → feedback loop.
- **Reuses** the Stage-0 assets (`curves`/`coverage`/`portfolio`), `judge()`
  below-expectation logic, and the historical item-level scores from the DB;
  item-level capability tags (`item_capabilities`) come from the benchmark
  tagger and degrade to benchmark-level (coarse) evidence when absent.
- **Statistical guards**: minimum item/peer counts, Wilson intervals for
  small-sample sub-accuracy, rank-based percentiles, unbiased pass@k
  (Chen et al.), saturation-stopped bad-case sampling.
- **Feedback loop** (`feedback.py`): `execution_logs` table + versioned
  `calibration` asset; `recalibrate()` re-estimates the Stage-5 cost ratios
  and gain scale from real outcomes (DataChef-style "use last round's effect
  to fix next round's estimate", applied at the human fix loop level).

## Directory map (design doc section 7)

```
src/benchmark_diagnosis/
├── core/            # config, schema, db, types, llm_client
├── data/            # ingestion + queries + seed reference data
├── capability_analysis/   # factor_analysis, mirt_fit, coverage_table, design_goal_validation
├── representative_selection/  # portfolio_selector
├── evaluation_orchestration/  # harness_bridge, deploy, screening_runner, expectation_curves, drilldown_trigger
├── diagnosis/       # label_slicing, failure_mode_analyst
├── recommendation/  # rule_base, retrieval, synthesizer, groundedness_check
├── intelligent_diagnosis/  # stages 1-7: candidate_generation, probe_registry,
│                           # guided_case_analyzer, confidence_fusion,
│                           # priority_scorer, suggestion_writer, feedback, orchestrator
└── reporting/       # report_generator
```

## Dependency tiers

- **core** (always installed): numpy, scipy, pandas, scikit-learn, pydantic, typer,
  rich, pyyaml, sqlalchemy, httpx.
- **`[mirt]`**: torch (item-level IRT).
- **`[eval]`**: lm-eval (evaluation bridge).
- **`[serve]`**: vllm (weight deployment — install in the GPU environment).
- **`[dev]`**: pytest, pytest-xdist, ruff.
