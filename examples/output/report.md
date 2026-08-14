# Benchmark Diagnosis Report — Llama-3-8B

- **model_id**: `llama-3-8b`
- **arch**: dense | total_params: 8.000 | active_params: 8.000
- **generated_at**: 2026-08-14T10:03:27.804677
- **mode**: full | **advisor**: rules
- **asset versions**: coverage=`coverage-20260814084434-3e725b` portfolio=`portfolio-20260814084434-ecae13` curves=`curves-20260814084434-ee521f`

## Summary — 3 capability cluster(s)

| cluster | weighted score | percentile | z-score | verdict |
|---|---|---|---|---|
| `cluster_0` | 38.666 | 0.0 | -2.75 | ⚠️ under-performing |
| `cluster_1` | 24.454 | 1.7 | -2.77 | ⚠️ under-performing |
| `cluster_2` | 18.000 | 23.1 | -0.81 | ⚠️ under-performing |

## Cluster `cluster_0`

### Benchmarks

| benchmark | weight | score |
|---|---|---|
| `longbench_v2` | 0.834 | 38.000 |
| `frontiermath` | 0.117 | — |
| `mmlu_pro` | 0.049 | 50.000 |

### Diagnosis

- sub_capability: `longbench_v2`
- quantified_gap: 0.16

### Recommendations

**1. [R-reading-001] 增加长文档理解、多跳抽取与摘要类样本，配合长上下文训练策略，改善信息 定位与利用。** _(source: rule_base:R-reading-001, evidence: medium)_
   - *Validation*: 在小规模子集上做消融：固定其余训练配置，仅应用该改动，对比目标 benchmark 的簇级得分变化（建议 >=1% 且跨 2 个随机种子稳定）。

**2. [R-longctx-001] 对长上下文任务采用 RoPE 外推 / 位置内插 / 长上下文继续预训练，并构造 长文档信息定位与长链推理样本，改善长输入下的信息利用。** _(source: rule_base:R-longctx-001, evidence: medium)_
   - *Validation*: 在小规模子集上做消融：固定其余训练配置，仅应用该改动，对比目标 benchmark 的簇级得分变化（建议 >=1% 且跨 2 个随机种子稳定）。

---

## Cluster `cluster_1`

### Benchmarks

| benchmark | weight | score |
|---|---|---|
| `simpleqa` | 0.935 | 24.000 |
| `ifeval` | 0.037 | — |
| `math` | 0.027 | 40.000 |

### Diagnosis

- sub_capability: `simpleqa`
- quantified_gap: 0.25

### Recommendations

**1. [R-factuality-001] 提高含证据链、引用溯源、事实核查的语料与 SFT 样本配比，配合"不确定即声明 不确定"的拒答 / 反事实样本，抑制幻觉，提升短问答事实准确率。** _(source: rule_base:R-factuality-001, evidence: high)_
   - *Validation*: 在小规模子集上做消融：固定其余训练配置，仅应用该改动，对比目标 benchmark 的簇级得分变化（建议 >=1% 且跨 2 个随机种子稳定）。

---

## Cluster `cluster_2`

### Benchmarks

| benchmark | weight | score |
|---|---|---|
| `swe_bench` | 0.926 | 18.000 |
| `tau2_bench` | 0.051 | — |
| `livecodebench` | 0.023 | — |

### Diagnosis

- sub_capability: `swe_bench`
- quantified_gap: 0.11

### Recommendations

**1. [R-code-001] 提高高质量代码（含单测与执行结果反馈）预训练 / SFT 数据占比，并在 RL 阶段 引入单元测试 / 执行通过率作为 reward，强化"可运行"而非"看着对"。** _(source: rule_base:R-code-001, evidence: high)_
   - *Validation*: 在小规模子集上做消融：固定其余训练配置，仅应用该改动，对比目标 benchmark 的簇级得分变化（建议 >=1% 且跨 2 个随机种子稳定）。

---

## Figures

![figures/fig_model_scores.png](figures/fig_model_scores.png)

![figures/fig_model_clusters.png](figures/fig_model_clusters.png)

![figures/fig_model_gaps.png](figures/fig_model_gaps.png)
