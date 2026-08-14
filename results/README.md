# Benchmark Diagnosis — Offline Results

_Generated 2026-08-14 07:31 UTC from seed reference data (approximate public leaderboard values; see `_note` in the seed)._

## Overview

| metric | value |
|---|---|
| models | 48 |
| benchmarks | 14 |
| scores | 469 |
| capability clusters | 3 |
| fitted expectation curves | 54 |

**Asset versions** (every diagnosis report traces back to these):

- `coverage_version`: `coverage-20260814073142-e97f81`
- `portfolio_version`: `portfolio-20260814073142-3f1582`
- `curves_version`: `curves-20260814073143-cee715`

## Capability-coverage table

| benchmark | cluster | breadth | reliability | saturated | design-goal agreement | tags |
|---|---|---|---|---|---|---|
| `AIME 2024` | `cluster_0` | 0.92 | 0.16 |  | 0.00 | math, reasoning |
| `ARC-Challenge` | `cluster_1` | 0.81 | 0.23 | ⚠️ | 0.00 | commonsense_knowledge, reasoning |
| `GPQA Diamond` | `cluster_0` | 0.84 | 0.15 |  | 0.00 | world_knowledge, reasoning |
| `GSM8K` | `cluster_1` | 0.88 | 0.23 | ⚠️ | 0.00 | math, reasoning |
| `HellaSwag` | `cluster_1` | 0.74 | 0.15 | ⚠️ | 0.00 | commonsense_knowledge, reading_comprehension |
| `HumanEval` | `cluster_1` | 0.94 | 0.21 | ⚠️ | 0.00 | code |
| `HumanEval+` | `cluster_0` | 0.90 | 0.25 |  | 0.00 | code |
| `IFEval` | `cluster_0` | 0.95 | 0.21 |  | 0.00 | instruction_following |
| `LiveCodeBench` | `cluster_0` | 0.67 | 0.16 |  | 0.00 | code, reasoning |
| `MATH` | `cluster_0` | 0.80 | 0.15 |  | 0.00 | math, reasoning |
| `MMLU` | `cluster_1` | 0.83 | 0.23 |  | 0.00 | world_knowledge, reading_comprehension, reasoning |
| `MMLU-Pro` | `cluster_0` | 0.92 | 0.12 |  | 0.00 | world_knowledge, reasoning |
| `SWE-bench Verified` | `cluster_2` | 0.89 | 0.38 |  | 0.00 | code, agentic_tool_use |
| `tau2-Bench` | `cluster_2` | 0.90 | 0.36 |  | 0.00 | agentic_tool_use, instruction_following |

## Capability clusters

_Clusters group benchmarks whose scores co-vary across models — i.e. correlated capabilities that tend to improve together during training._

- **cluster_0**: reasoning, math, world_knowledge
- **cluster_1**: reasoning, commonsense_knowledge, reading_comprehension
- **cluster_2**: agentic_tool_use, code, instruction_following

## Representative portfolios

- **cluster_0**: `ifeval` (0.82), `livecodebench` (0.14), `mmlu_pro` (0.04)
- **cluster_1**: `mmlu` (1.00)
- **cluster_2**: `tau2_bench` (1.00), `swe_bench` (0.00)

## Figures

### fig_curves_aime24.png

![fig_curves_aime24.png](figures/fig_curves_aime24.png)

### fig_curves_arc_challenge.png

![fig_curves_arc_challenge.png](figures/fig_curves_arc_challenge.png)

### fig_curves_gpqa.png

![fig_curves_gpqa.png](figures/fig_curves_gpqa.png)

### fig_curves_gsm8k.png

![fig_curves_gsm8k.png](figures/fig_curves_gsm8k.png)

### fig_curves_hellaswag.png

![fig_curves_hellaswag.png](figures/fig_curves_hellaswag.png)

### fig_curves_humaneval.png

![fig_curves_humaneval.png](figures/fig_curves_humaneval.png)

### fig_curves_humaneval_plus.png

![fig_curves_humaneval_plus.png](figures/fig_curves_humaneval_plus.png)

### fig_curves_ifeval.png

![fig_curves_ifeval.png](figures/fig_curves_ifeval.png)

### fig_curves_livecodebench.png

![fig_curves_livecodebench.png](figures/fig_curves_livecodebench.png)

### fig_curves_math.png

![fig_curves_math.png](figures/fig_curves_math.png)

### fig_curves_mmlu.png

![fig_curves_mmlu.png](figures/fig_curves_mmlu.png)

### fig_curves_mmlu_pro.png

![fig_curves_mmlu_pro.png](figures/fig_curves_mmlu_pro.png)

### fig_curves_swe_bench.png

![fig_curves_swe_bench.png](figures/fig_curves_swe_bench.png)

### fig_curves_tau2_bench.png

![fig_curves_tau2_bench.png](figures/fig_curves_tau2_bench.png)

### fig_frontier_overview.png

![fig_frontier_overview.png](figures/fig_frontier_overview.png)

### fig_coverage_profile.png

![fig_coverage_profile.png](figures/fig_coverage_profile.png)

### fig_coverage_metrics.png

![fig_coverage_metrics.png](figures/fig_coverage_metrics.png)

### fig_cluster_map.png

![fig_cluster_map.png](figures/fig_cluster_map.png)

### fig_benchmark_correlation.png

![fig_benchmark_correlation.png](figures/fig_benchmark_correlation.png)
