# Benchmark Diagnosis Report — Llama-3-8B

- **model_id**: `llama-3-8b`
- **arch**: dense | total_params: 8.000 | active_params: 8.000
- **generated_at**: 2026-08-17T01:27:54.380435
- **mode**: full | **advisor**: rules
- **asset versions**: coverage=`coverage-20260817012754-16df85` portfolio=`portfolio-20260817012754-01b3ba` curves=`curves-20260817012754-9e6692` experience=`experience-20260817012754-e06761`

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
| `mmlu_pro` | 0.049 | 50.000 |

### Diagnosis

- sub_capability: `longbench_v2`
- quantified_gap: 0.16

**Missing-capability profile** (from benchmark correlation + bad-case analysis):

| capability | deficit strength |
|---|---|
| `long_context` | 0.500 |
| `reading_comprehension` | 0.500 |

> 该簇缺失能力：long_context(0.50)、reading_comprehension(0.50)；主要驱动：mmlu_pro(0.24)、longbench_v2(0.16)


### Recommendations

**1. [R-longctx-001] 长上下文能力：长文档语料 + 位置编码外推** _(source: experience:EXP-longctx-001, evidence: high)_
   - *Expected effect*: LongBench V2 / 长文档 benchmark 提升 10~25 分位
   - *Datasets to add*:
     - `LongBench 训练版 / L-Eval 长文 QA` — 覆盖 16 类长文本任务，可拆分为指令微调样本，直接对齐评测分布
     - `书级长文档语料（BookSum 类 / 自建 32k-128k 文档）` — 长上下文继续预训练需要连贯长文档，而非拼接的短文本
   - *Hyperparameter adjustments*:
     - `rope_scaling / max_position_embeddings` → extend (typical range: 4k -> 32k-128k（linear 或 YaRN 外推）)
     - `long-context continued-pretraining tokens` → increase (typical range: 50B~200B tokens)
   - *Reasoning*:
     - 该簇缺失能力：long_context(0.50)、reading_comprehension(0.50)；主要驱动：mmlu_pro(0.24)、longbench_v2(0.16)
     - 匹配干预 EXP-longctx-001：长上下文能力：长文档语料 + 位置编码外推
     - 历史 1 次实验平均提升 +2.1%（分数变化）
     - 关联规则 R-longctx-001（medium 证据）
   - *Validation*: 在小规模子集上做消融：固定其余训练配置，仅应用该改动，对比目标 benchmark 的簇级得分变化（建议 >=1% 且跨 2 个随机种子稳定）。

**2. [R-reading-001] 阅读理解：多跳抽取与摘要样本增强** _(source: experience:EXP-reading-001, evidence: medium)_
   - *Expected effect*: 阅读理解类 benchmark 提升 5~15 分位
   - *Datasets to add*:
     - `SQuAD 2.0 / RACE` — 基础抽取式与选择式阅读理解，覆盖拒答（无答案）场景
     - `HotpotQA` — 多跳推理式阅读理解，训练跨段落信息融合
   - *Hyperparameter adjustments*:
     - `长文/抽取类样本在 SFT 中的配比` → increase (typical range: 提升至总 SFT 的 5%~15%)
   - *Reasoning*:
     - 该簇缺失能力：long_context(0.50)、reading_comprehension(0.50)；主要驱动：mmlu_pro(0.24)、longbench_v2(0.16)
     - 匹配干预 EXP-reading-001：阅读理解：多跳抽取与摘要样本增强
     - 关联规则 R-reading-001（medium 证据）
   - *Validation*: 在小规模子集上做消融：固定其余训练配置，仅应用该改动，对比目标 benchmark 的簇级得分变化（建议 >=1% 且跨 2 个随机种子稳定）。

---

## Cluster `cluster_1`

### Benchmarks

| benchmark | weight | score |
|---|---|---|
| `simpleqa` | 0.935 | 24.000 |
| `math` | 0.027 | 40.000 |

### Diagnosis

- sub_capability: `simpleqa`
- quantified_gap: 0.25

**Missing-capability profile** (from benchmark correlation + bad-case analysis):

| capability | deficit strength |
|---|---|
| `factuality` | 1.000 |

> 该簇缺失能力：factuality(1.00)；主要驱动：simpleqa(0.26)


### Recommendations

**1. [R-factuality-001] 事实性：证据链语料 + 不确定即声明不确定** _(source: experience:EXP-factuality-001, evidence: high)_
   - *Expected effect*: SimpleQA / 短问答事实准确率提升 10~20 分位
   - *Datasets to add*:
     - `SimpleQA / HaluEval` — 短问答事实性与幻觉检测样本，训练"有依据才回答"
     - `TruthfulQA + 引用溯源 SFT 语料` — 反事实 + 引用格式样本，配合"不确定即拒答"模板
   - *Hyperparameter adjustments*:
     - `"不确定即拒答" SFT 样本配比` → increase (typical range: 5%~15%)
     - `幻觉成对 DPO（编造 vs 拒答）` → enable (typical range: β ∈ [0.1, 0.3])
   - *Reasoning*:
     - 该簇缺失能力：factuality(1.00)；主要驱动：simpleqa(0.26)
     - 匹配干预 EXP-factuality-001：事实性：证据链语料 + 不确定即声明不确定
     - 历史 1 次实验平均提升 +1.5%（分数变化）
     - 关联规则 R-factuality-001（high 证据）
   - *Validation*: 在小规模子集上做消融：固定其余训练配置，仅应用该改动，对比目标 benchmark 的簇级得分变化（建议 >=1% 且跨 2 个随机种子稳定）。

---

## Cluster `cluster_2`

### Benchmarks

| benchmark | weight | score |
|---|---|---|
| `swe_bench` | 0.926 | 18.000 |

### Diagnosis

- sub_capability: `swe_bench`
- quantified_gap: 0.11

**Missing-capability profile** (from benchmark correlation + bad-case analysis):

| capability | deficit strength |
|---|---|
| `code` | 0.500 |
| `agentic_tool_use` | 0.500 |

> 该簇缺失能力：code(0.50)、agentic_tool_use(0.50)；主要驱动：swe_bench(0.11)


### Recommendations

**1. [R-code-001] 代码能力：可执行验证 + 代码语料增强** _(source: experience:EXP-code-001, evidence: high)_
   - *Expected effect*: SWE-bench / LiveCodeBench 提升 10~20 分位
   - *Datasets to add*:
     - `CodeContests + BigCodeBench` — 竞赛级与指令级代码样本，含单测，强调"可运行"
     - `SWE-bench 轨迹（问题→修复→单测）` — 真实仓库级问题解决轨迹，训练定位与修复
   - *Hyperparameter adjustments*:
     - `执行通过率 reward（单元测试 / pass@k）` → enable (typical range: RL 阶段以 test 通过为 reward)
     - `代码语料在预训练中的配比` → increase (typical range: 提升至 15%~25%)
   - *Reasoning*:
     - 该簇缺失能力：code(0.50)、agentic_tool_use(0.50)；主要驱动：swe_bench(0.11)
     - 匹配干预 EXP-code-001：代码能力：可执行验证 + 代码语料增强
     - 历史 1 次实验平均提升 +1.8%（分数变化）
     - 关联规则 R-code-001（high 证据）
   - *Validation*: 在小规模子集上做消融：固定其余训练配置，仅应用该改动，对比目标 benchmark 的簇级得分变化（建议 >=1% 且跨 2 个随机种子稳定）。

**2. [R-sft-002] Agent / 工具调用：函数调用轨迹 SFT** _(source: experience:EXP-agent-001, evidence: medium)_
   - *Expected effect*: τ-bench / 工具调用类 benchmark 提升 10~20 分位
   - *Datasets to add*:
     - `τ-bench / ToolBench 轨迹` — 多轮工具调用轨迹（含错误调用与纠正），训练收到工具返回后正确利用
     - `Gorilla 函数调用样本` — API 文档感知的函数调用数据，覆盖常见 API
   - *Hyperparameter adjustments*:
     - `工具调用格式一致性` → enforce (typical range: 统一 function-call schema / JSON)
     - `任务成功率 RL reward` → enable (typical range: 以多轮任务是否成功为 reward)
   - *Reasoning*:
     - 该簇缺失能力：code(0.50)、agentic_tool_use(0.50)；主要驱动：swe_bench(0.11)
     - 匹配干预 EXP-agent-001：Agent / 工具调用：函数调用轨迹 SFT
     - 关联规则 R-sft-002（medium 证据）
   - *Validation*: 在小规模子集上做消融：固定其余训练配置，仅应用该改动，对比目标 benchmark 的簇级得分变化（建议 >=1% 且跨 2 个随机种子稳定）。

---

## Figures

![figures/fig_model_scores.png](figures/fig_model_scores.png)

![figures/fig_model_clusters.png](figures/fig_model_clusters.png)

![figures/fig_model_gaps.png](figures/fig_model_gaps.png)
