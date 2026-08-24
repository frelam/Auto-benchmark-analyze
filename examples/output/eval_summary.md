# Evaluation summary

benchmarks scored: **8**

| benchmark | score | metric | samples | bad cases | judgment |
|---|---|---|---|---|---|
| gsm8k GSM8K | 79.000 | — | — | 0 | params_dense · p55.0 · z=0.43109649763422747 |
| humaneval HumanEval | 60.000 | — | — | 0 | params_dense · p70.0 · z=0.5826603677232809 |
| longbench_v2 LongBench V2 | 38.000 | — | — | 0 | params_active · p0.0 · z=-2.5635317748282946 |
| math MATH | 40.000 | — | — | 0 | params_dense · p60.0 · z=0.4935939850958813 |
| mmlu MMLU | 66.000 | — | — | 0 | params_dense · p25.0 · z=-0.20300944411252303 |
| mmlu_pro MMLU-Pro | 50.000 | — | — | 0 | params_dense · p0.0 · z=-5.927644681416815 |
| simpleqa SimpleQA | 24.000 | — | — | 0 | params_active · p0.0 · z=-2.8654292721844468 |
| swe_bench SWE-bench Verified | 18.000 | — | — | 0 | params_active · p23.076923076923077 · z=-0.8118418121798199 |

Bad cases 逐条见 `bad_cases/`；机器可读明细见 `eval_results.json`。
下一轮可用 `run --scores scores.json` 直接接续诊断。
