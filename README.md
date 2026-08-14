# benchmark-diagnosis

一站式大模型评测 + benchmark 低分归因诊断 + 优化建议系统。

围绕一个 LLM 模型（无论你只有**推理服务 IP**，还是只有**权重文件**），系统自动完成三件事：

1. **评测**：桥接业界标准评测工具，跑一组经过"能力覆盖分析"精选的 benchmark；
2. **诊断**：判断每个能力簇"是否不及预期"，并切片定位低分子类、用固定 taxonomy 分析失败模式；
3. **建议**：基于经验规则库 + 外部检索 + LLM 综合，输出**可追溯、可验证**的优化建议。

完整设计见 `benchmark-diagnosis-tool-design.md`，技术选型见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

---

## 三个核心能力

| 能力 | 做法 | 复用 vs 自研 |
|---|---|---|
| 评测 LLM | 桥接 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)，`local-completions` 直连 OpenAI 兼容端点 | **复用**，只写薄桥接 |
| 权重部署 | `vllm serve <model_id>` 起 OpenAI 兼容服务，与"给 IP"收敛成同一条评测路径 | **复用** vLLM |
| 能力覆盖分析 | 多维 IRT（双塔打分模型，PyTorch 可选）；无 item 数据时回退因子分析（PCA+聚类） | **自研**（核心特色） |
| 预期曲线 | 参数量→分 log-linear + 时间→前沿包络，残差转局部百分位/z-score 判"不及预期" | **自研** |
| 归因诊断 | 标签切片 + LLM-as-analyst（固定 taxonomy，禁止自由命名） | **自研** |
| 优化建议 | 经验规则库（YAML + 校验）+ LLM 综合 + 幻觉校验（引用/数字必须落在证据集内） | **自研** |

## 系统架构（七层）

```
数据层 → 能力覆盖分析层(离线) → 代表性 benchmark 选择层(离线)
      → 评测编排层 → 归因诊断层 → 优化建议生成层 → 报告层
```

- **离线层**（定期重跑，产出版本化静态资产）：能力覆盖表、簇级代表性组合、预期曲线。每次产出新版本号、不覆盖历史，报告可追溯。
- **在线层**（每评测一个模型跑一次）：读取最新离线资产 → 评测 → 诊断 → 建议 → 报告。

---

## 运行结果（预生成）

仓库随附一份基于内置 seed 跑出的离线结果（`results/`），可直接查看系统产出的图表与资产：

- 完整归档：**[`results/README.md`](results/README.md)**（覆盖表 + 代表性组合 + 全部图表 + 资产版本号）
- 原始资产：`results/coverage.json` / `results/portfolio.json` / `results/curves.json`

### 能力聚类图（哪些能力在训练时相关）

![能力聚类](results/figures/fig_cluster_map.png)

> 空间距离 = 各 benchmark 分数在 48 个模型上的**协同变化程度**（pairwise-complete 秩相关 → 特征分解 → KMeans）。落点来自**单位化因子载荷**的二维投影：距离近似反映两两相关，而不是单个 benchmark 在主导因子上的载荷强度（修正了早前版本"高相关任务被载荷强度拉远"的问题）。同簇 = 训练时往往一起提升（可迁移 / 共训）。当前 5 簇：
>
> - **前沿数学 / 强推理**：AIME 2024/2025、FrontierMath、GPQA、LongBench V2、MATH-500、MMLU-Pro；
> - **通用知识 / 中等数学**：ARC-Challenge、GSM8K、Hellaswag、IFEval、MATH、MMLU、SimpleQA —— `math` 与 `gsm8k` 同簇（Spearman ρ ≈ 0.74）；
> - **代码 / agentic**：BigCodeBench、HumanEval(+)、LiveCodeBench、SWE-bench、τ²-bench —— `livecodebench` 与 `swe_bench` 同簇（ρ ≈ 0.85）；
> - **终端 agentic**：Terminal-Bench（独立簇）；
> - **多模态**：MMMU（独立簇）。
>
> 已补充 2026 主流数据集：AIME 2025、FrontierMath、MATH-500、BigCodeBench（仅 Instruct-FULL 难度档）、Terminal-Bench（仅 v1.0）、SimpleQA、MMMU、LongBench V2。分数 provenance 在 seed 的 `_note` 中标注为 `verified`（已核对）或 `estimated`（按同模型已知排位标定）。

### 各 benchmark 的 SOTA 前沿（时间维度）

![SOTA 前沿](results/figures/fig_frontier_overview.png)

### 能力覆盖画像（benchmark × 潜在维度）

![能力覆盖画像](results/figures/fig_coverage_profile.png)

### 预期曲线示例：MMLU（参数量拟合 + 时间前沿）

![MMLU 预期曲线](results/figures/fig_curves_mmlu.png)

### benchmark 分数相关性

![相关性](results/figures/fig_benchmark_correlation.png)

> 这些图由 `benchmark-diagnosis visualize` 生成；重跑会覆盖 `results/`，版本号随之更新。

---

## 安装

```bash
pip install -e .                 # 核心：编排、诊断、建议
pip install -e ".[eval]"         # + 评测桥接（lm-eval）
pip install -e ".[serve]"        # + 权重部署（vLLM，GPU 环境）
pip install -e ".[mirt]"         # + item 级多维 IRT（torch）
pip install -e ".[plot]"         # + 图表渲染（`visualize` 命令）
pip install -e ".[dev]"          # 开发依赖（pytest / ruff）
```

> 需要 Python ≥ 3.10。评测桥接和部署依赖较重，建议在 GPU 环境安装 `[eval,serve,mirt]`。

---

## 快速开始（30 秒）

```bash
# 1. 载入 seed 参考数据（48 个模型 × 22 个 benchmark，公开 leaderboard 近似值）
benchmark-diagnosis ingest --seed

# 2. 离线构建：能力覆盖表 + 代表性组合 + 预期曲线
benchmark-diagnosis build-offline

# 3. 生成图表 + 归档结果（曲线 / 覆盖分析 / 相关性，见 results/）
benchmark-diagnosis visualize --out results

# 4. 给定一个推理服务 IP，跑精选 benchmark 组合
benchmark-diagnosis eval-model --model my-model --base-url http://<ip>:8000/v1

# 5. 诊断 + 建议 + 报告（需在 config 里配分析用 LLM，见下文）
benchmark-diagnosis diagnose --model my-model --base-url http://<ip>:8000/v1
```

---

## 使用例子

### 例子 0：配置

默认配置在 `config/default.yaml`。复制一份，`--config` 传入即可深度覆盖；环境变量 `BMD_<SECTION>__<FIELD>` 优先级最高。

```bash
cp config/default.yaml my-config.yaml
# 编辑 my-config.yaml，配分析用 LLM（失败模式分析 / 综合生成用到）
```

`my-config.yaml` 中关键片段：

```yaml
llm:
  base_url: http://localhost:8000/v1   # 分析用 LLM 的 OpenAI 兼容端点
  api_key: EMPTY
  model: qwen2.5-32b-instruct          # 你的分析模型
```

### 例子 1：载入数据

```bash
# 用内置 seed（48 个模型 × 22 个 benchmark，公开 leaderboard 近似值）
benchmark-diagnosis ingest --seed
# 输出: Ingested: {'models': 48, 'benchmarks': 22, 'scores': 599}

# 或载入你自己的数据（结构见 data/seed/seed_reference.json）
benchmark-diagnosis ingest --file my_data.json
```

### 例子 2：构建离线资产

```bash
benchmark-diagnosis build-offline
# 输出:
#   coverage_version: coverage-20260814063812-430b2c
#   portfolio_version: portfolio-20260814063812-097a03
#   curves_version: curves-20260814063812-e16657
```

这三个版本号是版本化资产，之后每次诊断报告都会带上，保证可追溯。重跑会生成新版本号，不覆盖历史。

再用 `benchmark-diagnosis visualize --out results` 把这些资产渲染成图表并归档（见上文「运行结果」）。

### 例子 3：部署权重（有模型权重时）

```bash
# 打印部署命令（dry-run）
benchmark-diagnosis deploy meta-llama/Llama-3-8B-Instruct
# 输出:
#   vllm serve meta-llama/Llama-3-8B-Instruct --host 0.0.0.0 --port 8000 \
#     --served-model-name meta-llama/Llama-3-8B-Instruct --tensor-parallel-size 1 \
#     --gpu-memory-utilization 0.9

# 或直接拉起服务并等待就绪（Ctrl+C 停止）
benchmark-diagnosis deploy meta-llama/Llama-3-8B-Instruct --launch
```

### 例子 4：评测（只有推理服务 IP 时）

```bash
benchmark-diagnosis eval-model \
    --model my-model \
    --base-url http://<ip>:8000/v1
# 默认跑"代表性组合"里的 benchmark；也可手动指定：
benchmark-diagnosis eval-model --model my-model --base-url http://<ip>:8000/v1 \
    --tasks gsm8k,humaneval
```

### 例子 5：诊断 + 报告（用分数文件，不重复评测）

```bash
echo '{"mmlu": 66.6, "gsm8k": 79.6, "humaneval": 33.4, "math": 16.0}' > scores.json

benchmark-diagnosis diagnose \
    --model llama-3-8b \
    --scores scores.json \
    --output report.md
```

报告（节选）：

```markdown
# Benchmark Diagnosis Report — Llama-3-8B
- asset versions: coverage=`coverage-…-430b2c` portfolio=`portfolio-…-097a03` curves=`curves-…-e16657`

## Summary — 2 capability cluster(s)
| cluster   | weighted score | percentile | z-score | verdict          |
|-----------|----------------|-----------|---------|------------------|
| cluster_0 | 34.111         | 74.8      | 0.58    | ✅ in range      |
| cluster_1 | 16.000         | 100.0     | 0.00    | ✅ in range      |

## Cluster cluster_1
### Benchmarks
| benchmark | weight | score |
|-----------|--------|-------|
| math      | 1.000  | 16.000|

### Recommendations
1. [R-math-001] 提高高质量数学（含带解题过程的）语料…引入过程监督 / 可验证器 reward…
   (source: rule_base:R-math-001, evidence: high)
   - Validation: 在小规模子集上做消融…对比目标 benchmark 的簇级得分变化…
```

判定"不及预期"的簇会触发细分诊断（标签切片 + LLM 失败模式分析），并在报告里给出对应建议。

### 例子 6：一键端到端

```bash
# 有权重：自动部署 → 评测 → 诊断 → 报告
benchmark-diagnosis run \
    --model llama-3-8b \
    --model-id meta-llama/Llama-3-8B-Instruct

# 只有 IP：跳过部署
benchmark-diagnosis run --model my-model --base-url http://<ip>:8000/v1
```

---

## 目录结构

```
src/benchmark_diagnosis/
├── core/                 # config、schema、db、types、llm_client
├── data/                 # ingestion + queries + seed 参考数据
├── capability_analysis/  # factor_analysis / mirt_fit / coverage_table / design_goal_validation
├── representative_selection/  # portfolio_selector
├── evaluation_orchestration/  # harness_bridge / deploy / screening_runner / expectation_curves / drilldown_trigger
├── diagnosis/            # label_slicing / failure_mode_analyst
├── recommendation/       # rule_base / retrieval / synthesizer / groundedness_check
└── reporting/            # report_generator
```

## 扩展点

- **经验规则库增补**：直接编辑 `src/benchmark_diagnosis/recommendation/rule_base/rules.yaml`，`applicable_tags` 和 `category` 必须落在 `taxonomy.yaml` 定义的词汇表内，加载器会自动校验，防止标签体系发散。
- **外部检索 RAG**：`recommendation/retrieval.py` 目前返回 `NullRetriever`（MVP 范围）。实现 `Retriever` 协议即可接入论文/技术报告检索，作为规则库未覆盖时的补充来源。
- **数据集接入**：`data/ingestion.py` 支持 JSON 与 lm-eval `results.json`，新数据源只需新增一个 ingest 函数。

## 测试

```bash
pytest -q              # 80 个测试
ruff check src tests   # 风格检查
```

## 说明

- `data/seed/seed_reference.json` 是**近似值**（公开 leaderboard 近似数，覆盖到 2026 年初的前沿模型：GPT-4o/o1/o3、Claude 3.5/3.7、Gemini 2.0/2.5、DeepSeek-V3/R1/V3.1/V3.2/V4、Qwen2.5/Qwen3/Qwen3-Coder、GLM-4.5/4.6、Kimi-K2、Llama-3.1/3.3 等；22 个 benchmark 含 SWE-bench / τ²-bench 等 agentic benchmark，并补充了 2026 主流数据集：AIME 2025、FrontierMath、MATH-500、BigCodeBench、Terminal-Bench、SimpleQA、MMMU、LongBench V2，难度档位已做隔离（BigCodeBench 仅 Instruct-FULL、Terminal-Bench 仅 v1.0）），仅用于 bootstrap 预期曲线，生产前请用真实数据替换。
- 参数量不可得的闭源模型，预期曲线自动走"时间→前沿包络"维度判定，不会被漏掉。
