# Benchmark Diagnosis Report — Llama-3-8B

- **model_id**: `llama-3-8b`
- **arch**: dense | total_params: 8.000 | active_params: 8.000
- **generated_at**: 2026-08-24T04:35:13.460117
- **mode**: full | **advisor**: rules | **engine**: rule
- **asset versions**: coverage=`coverage-20260824043513-dc410a` portfolio=`portfolio-20260824043513-6cea43` curves=`curves-20260824043513-4724d5` experience=`experience-20260824043513-2c8139`

## Summary — 2 capability cluster(s)

| cluster | weighted score | percentile | z-score | verdict |
|---|---|---|---|---|
| `cluster_0` | 38.666 | 0.0 | -2.75 | ⚠️ under-performing |
| `cluster_1` | 65.541 | 25.6 | -0.19 | ✅ in range |

## Cluster `cluster_0`

### Benchmarks

| benchmark | weight | score |
|---|---|---|
| `longbench_v2` | 0.916 | 38.000 |
| `mmlu_pro` | 0.054 | 50.000 |

- sub_capability: `longbench_v2` · quantified_gap: 0.16

---

## Cluster `cluster_1`

### Benchmarks

| benchmark | weight | score |
|---|---|---|
| `mmlu` | 0.964 | 66.000 |
| `math` | 0.017 | 40.000 |

- sub_capability: `math` · quantified_gap: 0.02

---

## 诊断（rule base）

engine: `rule` — 规则路径：低分筛选 → 数据集-能力映射 → 缺失能力 → 能力-数据集建议。

### 低分数据集（低于同等参数量 / 激活参数量分位）— 4

| benchmark | score | 判定基准 | percentile | z-score | shortfall |
|---|---|---|---|---|---|
| `longbench_v2` | 38.000 | 同等激活参数量 | 0.0 | -2.56 | 0.16 |
| `mmlu_pro` | 50.000 | 同等参数量(dense) | 0.0 | -5.93 | 0.24 |
| `simpleqa` | 24.000 | 同等激活参数量 | 0.0 | -2.87 | 0.26 |
| `swe_bench` | 18.000 | 同等激活参数量 | 23.1 | -0.81 | 0.11 |

### 缺失能力（过滤后）— 8

| 能力 | 证据 | 低分来源 |
|---|---|---|
| `knowledge` 知识 | 1.00 | `simpleqa`(p0.0)、`mmlu_pro`(p0.0) |
| `knowledge.world` 世界知识 | 0.74 | `mmlu_pro`(p0.0) |
| `reasoning` 推理 | 0.74 | `mmlu_pro`(p0.0) |
| `code` 代码 | 0.42 | `swe_bench`(p23.1) |
| `agent.tool_call` 工具调用 | 0.42 | `swe_bench`(p23.1) |
| `long_context` 长上下文 | 0.41 | `longbench_v2`(p0.0) |
| `language.reading` 阅读理解 | 0.41 | `longbench_v2`(p0.0) |
| `knowledge.factual` 事实性与幻觉抑制 | 0.26 | `simpleqa`(p0.0) |

### 能力-数据集建议 — 8

#### `knowledge`（gap=1.00 · 事实性：证据链语料 + 不确定即声明不确定（来源：human_expert，匹配度 层级））

建议补充数据集：
- **SimpleQA / HaluEval** — 短问答事实性与幻觉检测样本，训练"有依据才回答"（预期效果: 降低无依据编造）
- **TruthfulQA + 引用溯源 SFT 语料** — 反事实 + 引用格式样本，配合"不确定即拒答"模板（预期效果: 抑制幻觉、提升引用溯源）

调参建议：
- `"不确定即拒答" SFT 样本配比` increase （5%~15%）— 教会模型在依据不足时声明不确定而非强行生成
- `幻觉成对 DPO（编造 vs 拒答）` enable （β ∈ [0.1, 0.3]）— 直接校正拒答边界，降低过度自信生成

#### `knowledge.world`（gap=0.74 · 世界 / 常识知识：知识语料 + 事实一致性过滤（来源：human_expert，匹配度 精确））

建议补充数据集：
- **百科 / 开放域 QA（Natural Questions 类）** — 覆盖事实性知识与检索式问答（预期效果: 提升知识回忆准确率）
- **ARC / HellaSwag 增强样本** — 常识推理与物理直觉样本（预期效果: 提升常识知识）

调参建议：
- `知识语料去重 + 事实一致性过滤` enable （minhash 去重 + 语义冲突剔除）— 重复与矛盾语料会放大知识错误
- `知识类语料在预训练中的配比` increase （提升至 25%~35%）— 知识容量与语料规模强相关

验证数据集：`mmlu`

#### `reasoning`（gap=0.74 · 数学推理：CoT 语料 + 过程监督（来源：human_expert，匹配度 精确））

建议补充数据集：
- **GSM8K + MATH（含逐步解答）** — 多步算术与竞赛级数学推理基础语料（预期效果: 提升推理链完整度与符号计算）
- **PRM800K 过程标注** — 逐步骤正确性标签，用于训练过程奖励模型（PRM）（预期效果: 为过程监督提供训练信号）
- **OpenWebMath / Proof-Pile-2** — 高质量数学预训练语料，补充数学背景知识（预期效果: 增强数学直觉与公式能力）

调参建议：
- `reward_model` prm_over_orm （过程监督 PRM 替换结果-only ORM）— 中间步骤错误在结果级 reward 中不可见
- `采样温度 / self-consistency 预算` increase （T ∈ [0.6, 1.0]，投票 5~16 次）— 多步推理可借采样投票摊薄单次路径错误

#### `code`（gap=0.42 · 代码能力：可执行验证 + 代码语料增强（来源：human_expert，匹配度 精确））

建议补充数据集：
- **CodeContests + BigCodeBench** — 竞赛级与指令级代码样本，含单测，强调"可运行"（预期效果: 提升代码正确性与边界处理）
- **SWE-bench 轨迹（问题→修复→单测）** — 真实仓库级问题解决轨迹，训练定位与修复（预期效果: 提升 agent 式代码修复能力）

调参建议：
- `执行通过率 reward（单元测试 / pass@k）` enable （RL 阶段以 test 通过为 reward）— 强化"可运行"而非"看着对"
- `代码语料在预训练中的配比` increase （提升至 15%~25%）— 代码语料不足导致语法与 API 知识薄弱

#### `agent.tool_call`（gap=0.42 · Agent / 工具调用：函数调用轨迹 SFT（来源：human_expert，匹配度 精确））

建议补充数据集：
- **τ-bench / ToolBench 轨迹** — 多轮工具调用轨迹（含错误调用与纠正），训练收到工具返回后正确利用（预期效果: 降低参数错误与信息忽略）
- **Gorilla 函数调用样本** — API 文档感知的函数调用数据，覆盖常见 API（预期效果: 提升工具选择与参数生成）

调参建议：
- `工具调用格式一致性` enforce （统一 function-call schema / JSON）— 格式不一致会破坏 agent 循环的解析环节
- `任务成功率 RL reward` enable （以多轮任务是否成功为 reward）— 轨迹级成功信号强于单步格式正确

验证数据集：`tau2_bench`

#### `long_context`（gap=0.41 · 长上下文能力：长文档语料 + 位置编码外推（来源：human_expert，匹配度 精确））

建议补充数据集：
- **LongBench 训练版 / L-Eval 长文 QA** — 覆盖 16 类长文本任务，可拆分为指令微调样本，直接对齐评测分布（预期效果: 提升长文档信息定位与多跳抽取）
- **书级长文档语料（BookSum 类 / 自建 32k-128k 文档）** — 长上下文继续预训练需要连贯长文档，而非拼接的短文本（预期效果: 让模型在真实长文中学会利用远处依赖）

调参建议：
- `rope_scaling / max_position_embeddings` extend （4k -> 32k-128k（linear 或 YaRN 外推））— 直接外推可复用预训练权重；YaRN 适合超长输入
- `long-context continued-pretraining tokens` increase （50B~200B tokens）— 仅位置外推不足以激活长文能力，需长文继续预训练

验证数据集：`longbench_v2`

#### `language.reading`（gap=0.41 · 长上下文能力：长文档语料 + 位置编码外推（来源：human_expert，匹配度 精确））

建议补充数据集：
- **LongBench 训练版 / L-Eval 长文 QA** — 覆盖 16 类长文本任务，可拆分为指令微调样本，直接对齐评测分布（预期效果: 提升长文档信息定位与多跳抽取）
- **书级长文档语料（BookSum 类 / 自建 32k-128k 文档）** — 长上下文继续预训练需要连贯长文档，而非拼接的短文本（预期效果: 让模型在真实长文中学会利用远处依赖）

调参建议：
- `rope_scaling / max_position_embeddings` extend （4k -> 32k-128k（linear 或 YaRN 外推））— 直接外推可复用预训练权重；YaRN 适合超长输入
- `long-context continued-pretraining tokens` increase （50B~200B tokens）— 仅位置外推不足以激活长文能力，需长文继续预训练

验证数据集：`mmlu`

#### `knowledge.factual`（gap=0.26 · 事实性：证据链语料 + 不确定即声明不确定（来源：human_expert，匹配度 精确））

建议补充数据集：
- **SimpleQA / HaluEval** — 短问答事实性与幻觉检测样本，训练"有依据才回答"（预期效果: 降低无依据编造）
- **TruthfulQA + 引用溯源 SFT 语料** — 反事实 + 引用格式样本，配合"不确定即拒答"模板（预期效果: 抑制幻觉、提升引用溯源）

调参建议：
- `"不确定即拒答" SFT 样本配比` increase （5%~15%）— 教会模型在依据不足时声明不确定而非强行生成
- `幻觉成对 DPO（编造 vs 拒答）` enable （β ∈ [0.1, 0.3]）— 直接校正拒答边界，降低过度自信生成

验证数据集：`simpleqa`


## Figures

![figures/fig_model_scores.png](figures/fig_model_scores.png)

![figures/fig_model_clusters.png](figures/fig_model_clusters.png)

![figures/fig_model_gaps.png](figures/fig_model_gaps.png)
