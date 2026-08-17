# benchmark-diagnosis

一站式大模型评测 + benchmark 低分归因诊断 + 优化建议系统。

围绕一个 LLM 模型（无论你只有**推理服务 IP**，还是只有**权重文件**），系统自动完成三件事：

1. **评测**：桥接业界标准评测工具，跑一组经过"能力覆盖分析"精选的 benchmark；
2. **诊断**：判断每个能力簇"是否不及预期"，并切片定位低分子类、用固定 taxonomy 分析失败模式；
3. **建议**：根据 benchmark 相关性 + bad case 归因出"缺什么能力"，从工具维护的经验库（具体数据集 + 调参 knob + 历史效果）给出**可执行**建议，可选 LLM 重排，输出**可追溯、可验证**的优化建议。

完整设计见 `benchmark-diagnosis-tool-design.md`，统一诊断管线（Stages 1-7）设计见 [`docs/intelligent-diagnosis-v2-design.md`](docs/intelligent-diagnosis-v2-design.md)，技术选型见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

---

## 运行结果（预生成）

仓库随附一份基于内置 seed 跑出的离线结果（`results/`），可直接查看系统产出的图表与资产：

- 完整归档：**[`results/README.md`](results/README.md)**（覆盖表 + 代表性组合 + 全部图表 + 资产版本号）
- 原始资产：`results/coverage.json` / `results/portfolio.json` / `results/curves.json`

### 能力聚类图（哪些能力在训练时相关）

![能力聚类](results/figures/fig_cluster_map.png)

### 预期曲线示例：SWE-bench（参数量拟合 + 时间前沿）

![SWE-bench 预期曲线](results/figures/fig_curves_swe_bench.png)

---

## 安装

```bash
bash scripts/install.sh                    # 核心 + 评测桥接（lm-eval）+ 图表渲染
bash scripts/install.sh --with-gpu         # + vLLM 权重部署 + IRT（GPU 环境）
bash scripts/install.sh --download-data    # 可选：额外预下载评测数据集
```

> 需要 Python ≥ 3.10。脚本会创建 `.venv` 并完成安装，之后用 `.venv/bin/benchmark-diagnosis`（或激活 venv 后直接 `benchmark-diagnosis`）执行命令。参数说明见 `bash scripts/install.sh --help`。

---

## 快速开始（30 秒）

有模型权重（需 GPU）：

```bash
benchmark-diagnosis run --model llama-3-8b --model-id meta-llama/Llama-3-8B-Instruct
```

只有推理服务 IP：

```bash
benchmark-diagnosis run --model my-model --base-url http://<ip>:8000/v1
```

一条 `run` 自动完成：部署（如有权重）→ 评测 → 诊断 → 建议 → 写报告（`report.md` + `metrics.json` + `figures/`）。数据库为空时自动载入 seed 并构建离线资产，无需手动准备。

不想上 GPU、想先离线看效果，或想手动重建离线资产：

```bash
benchmark-diagnosis --config examples/run-from-scores.yaml run   # 用分数文件跑全流程（无需 GPU）
benchmark-diagnosis visualize --out results                      # 把离线资产渲染成图表归档到 results/
```

### 统一诊断管线（Stages 1-7）

`run` 一条命令就是完整的 v2 诊断管线（无需开关）：
候选能力生成（item 级 sub-accuracy / benchmark 级标签）→ probe 单项复测（含 pass@1/pass@k）
→ 引导式 bad case 分析（content/format/grading 归因）→ 置信度融合（High/Medium/Low）
→ 优先级排序（ExpectedGain / Cost）→ 具体建议文案，报告内含"诊断"章节：

```bash
benchmark-diagnosis --config examples/run-from-scores.yaml run
```

只想看归因分析、不要 Stage 6 的建议文案时，加 `--mode analyze` 跳过建议写作。

执行建议后，用反馈回路校准 Stage 5 的 Cost/Gain 估计（跑得越久排得越准）：

```bash
benchmark-diagnosis feedback log reasoning.math.calculation rejection_sampling \
  --predicted-gain 0.4 --actual-gain 0.35 --predicted-cost 1.0 --actual-cost 1.5
benchmark-diagnosis feedback recalibrate        # 重估 cost 比例 + gain 缩放
benchmark-diagnosis feedback list               # 查看执行记录
```

---

## 使用例子

站在你的角度：你手里只有一个**模型权重**，或一个**推理服务 IP**。想看到「评测 → 低分归因 → 优化建议」的完整结果，最小命令是 `run` 一条：

### 场景 A：只有模型权重（需 GPU）

```bash
benchmark-diagnosis run --model llama-3-8b --model-id meta-llama/Llama-3-8B-Instruct
```

自动完成：拉取权重 → vLLM 部署并等待就绪 → 跑代表性 benchmark 组合 → 诊断 → 建议 → 写报告。

### 场景 B：只有推理服务 IP

```bash
benchmark-diagnosis run --model my-model --base-url http://<ip>:8000/v1
```

跳过部署，直接对已有服务评测。

> **没有 GPU 也想先看效果？** 用一份 JSON 分数文件代替真实评测，30 秒出结果——下面的输出就是这条命令实际生成的（`examples/output/`，可复现）：
> ```bash
> benchmark-diagnosis --config examples/run-from-scores.yaml run
> ```

### 输出结果

跑完 `run`，命令行打印摘要，同时生成 `report.md`（完整诊断报告）、`metrics.json`（机器可读）与 `figures/`（三张图）。你最关心的两样东西：

**① 各 benchmark 得分**（蓝色 = 代表性组合，灰色 = 额外补充的分数）：

<img src="examples/output/figures/fig_model_scores.png" alt="评测得分" width="640">

**② 诊断与建议** —— 一眼看出：哪些 benchmark 分数偏低 → 归因出缺什么能力 → 该补什么数据集 / 调什么参数：

| 能力簇 | 低分 benchmark（分数） | 缺失能力（归因） | 建议：新增数据集 | 建议：调参 |
|---|---|---|---|---|
| 长文本 · 数学推理 | `longbench_v2`（38） | `long_context` / `reading_comprehension` | LongBench/L-Eval 长文 QA；书级 32k-128k 长文档语料 | `rope_scaling`（YaRN 外推）4k→32k-128k；长上下文继续预训练 tokens 提升 |
| 事实性 · 指令遵循 | `simpleqa`（24） | `factuality` | SimpleQA/HaluEval；TruthfulQA + 引用溯源 SFT 语料 | "不确定即拒答" SFT 配比 5~15%；幻觉成对 DPO β∈[0.1,0.3] |
| 代码 · Agent | `swe_bench`（18） | `code` / `agentic_tool_use` | CodeContests + BigCodeBench；SWE-bench 修复轨迹 | 执行通过率 reward；代码语料配比 15~25%；工具调用轨迹 SFT + 任务成功率 RL |

> 每条建议来自**工具维护的经验库**（`experience:<id>`，具体数据集 + 调参 knob + 历史效果），附缺失能力归因链（benchmark 相关性 + bad case → 能力 → 干预）与验证实验设计；完整版与可追溯的资产版本号（含 `experience` 资产）见 `report.md`。规则库与 LLM 可进一步参与（`advisor_mode`）。
