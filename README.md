# benchmark-diagnosis

一站式大模型评测 + benchmark 低分归因诊断 + 优化建议系统。

围绕一个 LLM 模型（无论你只有**推理服务 IP**，还是只有**权重文件**），系统自动完成三件事：

1. **评测**：桥接业界标准评测工具，跑一组经过"能力覆盖分析"精选的 benchmark，并把分数 + bad case 归档为人类可分析的目录；
2. **诊断**（默认关闭，`--diagnose` 开启）：rule base 统计路径筛低分数据集、过滤缺失能力、按能力-数据集表给补充建议；或 llm agent base 路径在 harness 里做 bad case 分析与结论验证循环；
3. **建议**：根据能力缺失 + 经验库（具体数据集 + 调参 knob + 历史效果）给出**可执行**建议，输出**可追溯、可验证**的优化建议。

完整设计见 `benchmark-diagnosis-tool-design.md`，诊断双路径设计见 [`docs/diagnosis-engines-design.md`](docs/diagnosis-engines-design.md)（v2 stages 1-7 设计见 [`docs/intelligent-diagnosis-v2-design.md`](docs/intelligent-diagnosis-v2-design.md)，作为 legacy 保留），技术选型见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

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

一条 `run` 命令完成"评测 + 归档"，诊断按需开启。先选**模型来源**（三选一），再用 `--mode` 控制**跑到哪一步**（默认 `full`）：

| 模型来源 | 参数 | 是否需要 GPU |
|---|---|---|
| 模型权重 | `--model-id meta-llama/Llama-3-8B-Instruct` | 是（自动 vLLM 部署 → 评测） |
| 推理服务 IP | `--base-url http://<ip>:8000/v1` | 否（直接评测已有服务） |
| 已有 benchmark 分数 | `--scores scores.json` | 否（跳过评测；`{benchmark_id: 分数}` JSON） |

| `--mode` | 跑到哪一步 | 产出 |
|---|---|---|
| `benchmark` | 只评测 | `scores.json` + `eval_summary.md` + `bad_cases/`（不诊断） |
| `analyze` | 评测 + 诊断 | + 诊断报告（归因，无最终建议文案） |
| `full`（默认） | 评测 + 诊断 | + `report.md` 完整版（含建议） |

> **诊断默认不开启**：评测结果和 bad case 每次都归档（人类可手工分析），但诊断
> 引擎只在 `--diagnose`（或配置 `diagnosis.enabled: true`）时运行。诊断分两条路径：
> **rule base**（默认，确定性统计）与 **llm agent base**（`--engine llm_agent`，
> 需配置 `diagnosis.llm_agent` 的开关 + harness 启动/交互命令）。三种来源互斥：
> 传两个会被拒绝并报清晰错误。来源 / mode / advisor 也能写进 YAML 用 `--config`
> 传，CLI 参数覆盖配置。数据库为空时自动载入 seed 并构建离线资产，无需手动准备。

### ① 有权重或已部署服务 → 跑完整流程

```bash
# 有权重：自动部署 → 评测 → 诊断 → 建议
benchmark-diagnosis run --model llama-3-8b --model-id meta-llama/Llama-3-8B-Instruct

# 已有推理服务 IP：跳过部署，直接评测 → 诊断 → 建议
benchmark-diagnosis run --model my-model --base-url http://<ip>:8000/v1
```

### ② 只想跑 benchmark（拿分数 + bad case，先不诊断）

加 `--mode benchmark`：只评测，把分数写到 `scores.json`，之后用 `--scores` 接着跑诊断。评测结果和错题会同时归档到输出目录。

```bash
benchmark-diagnosis run --model my-model --base-url http://<ip>:8000/v1 --mode benchmark
# → data/run_output/scores.json + eval_summary.md + bad_cases/
```

省时间只跑几个 benchmark：用 `--benchmarks` 逗号分隔指定子集（代表性 portfolio 的子集）。

```bash
benchmark-diagnosis run --model my-model --base-url http://<ip>:8000/v1 \
  --mode benchmark --benchmarks mmlu_pro,math,swe_bench
```

> `--benchmarks` 只在评测路径（`--model-id` / `--base-url`）生效；`--scores` 路径会忽略它（分数文件自带 benchmark 列表，会提前提示）。每样本数 cap 用 `evaluation.limit` 配置（debug 用）。

### ③ 已有 benchmark 分数 → 只跑诊断

用 `--scores` 跳过评测，加 `--diagnose` 直接诊断（默认 `full` 含建议；只要归因不要建议就加 `--mode analyze`）：

```bash
benchmark-diagnosis run --model my-model --scores scores.json --diagnose
benchmark-diagnosis run --model my-model --scores scores.json --diagnose --mode analyze   # 只要归因，不要建议
```

> 没 GPU 想先看效果？用一份 JSON 分数文件代替真实评测，30 秒出结果（下面的输出就来自这条命令）：
> ```bash
> benchmark-diagnosis --config examples/run-from-scores.yaml run --diagnose
> ```

### ④ 跑完看什么、怎么拿结论

跑完写到 `<output dir>/`（默认 `data/run_output/`，用 `--output` 或 `run.output.dir` 改）：

| 文件 | 内容 |
|---|---|
| `scores.json` | 各 benchmark 分数（`{benchmark_id: score}`，可直接喂回 `--scores`） |
| `eval_summary.md` | 评测概览（分数 + 判定基准 + bad case 统计），**每次评测都有** |
| `eval_results.json` | 机器可读评测明细（metric / 样本数 / 错题数 / 预期曲线判定） |
| `bad_cases/` | 逐 benchmark 的错题归档（`.jsonl` 机器可读 + `.md` 人读），**人类可手工分析** |
| `report.md` | 诊断报告（仅开启诊断时生成） |
| `metrics.json` | 同内容，机器可读（仅开启诊断时生成） |
| `figures/` | 图表（仅开启诊断时生成） |

**最关心的结论在 `report.md` 的"诊断"章节**：rule base 路径给出低分数据集（低于同等参数量 / 激活参数量分位）→ 缺失能力 → 建议补充的数据集 / 调参；命令行也打印摘要（N 个簇、M 个 under-performing）。

### 诊断管线与两条路径

诊断默认关闭（`--diagnose` 或 `diagnosis.enabled: true` 开启），`--engine` 选择路径：

- **rule base（默认，`diagnosis.engine: rule`）**：确定性统计——按"同等参数量 /
  同等激活参数量"预期曲线分位筛选低分数据集 → 数据集-能力映射表 → 过滤出缺失能力
  （噪声下限 + 祖先折叠）→ 按能力-数据集表（经验库）提示补充哪些数据集、调什么参数。
  完整设计见 [`docs/diagnosis-engines-design.md`](docs/diagnosis-engines-design.md)。
- **llm agent base（`--engine llm_agent`）**：先跑 rule base 采纳其结论，再做
  bad case 分析，然后在 harness（如 DeepSeek Harness）里循环"分析得结论 → 用
  `eval-task` 数据集评测或 bad case 分析验证 → 得新猜想"，直到最终结论。
  需要配置 `diagnosis.llm_agent`（开关 + harness 启动/交互命令）；工作流以
  [skill](skills/benchmark-diagnosis/SKILL.md) 形式提供：

```yaml
diagnosis:
  enabled: true
  engine: llm_agent
  llm_agent:
    enabled: true
    harness_cmd:  "dsh profile run my-diagnosis --case-pack {case_pack}"   # 启动命令
    interact_cmd: "dsh profile interact my-diagnosis --message {message}"  # 交互命令
```

harness 内可执行 `benchmark-diagnosis eval-task --task <id> --limit <N> --base-url <url> --model <m> --out <dir>` 验证假设（跑数据集子集并归档分数 + bad case）。

反馈回路（v2 保留，作用于经验库）：执行建议后校准 Cost/Gain 估计：

```bash
benchmark-diagnosis feedback log reasoning.math.calculation rejection_sampling \
  --predicted-gain 0.4 --actual-gain 0.35 --predicted-cost 1.0 --actual-cost 1.5
benchmark-diagnosis feedback recalibrate        # 重估 cost 比例 + gain 缩放
benchmark-diagnosis feedback list               # 查看执行记录
```

想手动重建离线资产 / 出图表归档：

```bash
benchmark-diagnosis visualize --out results      # 把离线资产渲染成图表归档到 results/
```

---

## 使用例子

站在你的角度：你手里只有一个**模型权重**，或一个**推理服务 IP**。想看到「评测 → 低分归因 → 优化建议」的完整结果，最小命令是 `run` 一条：

### 场景 A：只有模型权重（需 GPU）

```bash
benchmark-diagnosis run --model llama-3-8b --model-id meta-llama/Llama-3-8B-Instruct
```

自动完成：拉取权重 → vLLM 部署并等待就绪 → 跑代表性 benchmark 组合 → 归档分数 + bad case。想要诊断就加 `--diagnose`（默认 rule base；`--engine llm_agent` 走 harness 路径）。

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

跑完 `run`（开了 `--diagnose`），命令行打印摘要，同时生成 `report.md`（完整诊断报告）、`metrics.json`（机器可读）与 `figures/`（三张图）；不开诊断时每次评测也会生成 `scores.json` / `eval_summary.md` / `bad_cases/` 供人工分析。你最关心的两样东西：

**① 各 benchmark 得分**（蓝色 = 代表性组合，灰色 = 额外补充的分数）：

<img src="examples/output/figures/fig_model_scores.png" alt="评测得分" width="640">

**② 诊断与建议** —— 一眼看出：哪些 benchmark 分数偏低 → 归因出缺什么能力 → 该补什么数据集 / 调什么参数：

| 能力簇 | 低分 benchmark（分数） | 缺失能力（归因） | 建议：新增数据集 | 建议：调参 |
|---|---|---|---|---|
| 长文本 · 数学推理 | `longbench_v2`（38） | `long_context` / `reading_comprehension` | LongBench/L-Eval 长文 QA；书级 32k-128k 长文档语料 | `rope_scaling`（YaRN 外推）4k→32k-128k；长上下文继续预训练 tokens 提升 |
| 事实性 · 指令遵循 | `simpleqa`（24） | `factuality` | SimpleQA/HaluEval；TruthfulQA + 引用溯源 SFT 语料 | "不确定即拒答" SFT 配比 5~15%；幻觉成对 DPO β∈[0.1,0.3] |
| 代码 · Agent | `swe_bench`（18） | `code` / `agentic_tool_use` | CodeContests + BigCodeBench；SWE-bench 修复轨迹 | 执行通过率 reward；代码语料配比 15~25%；工具调用轨迹 SFT + 任务成功率 RL |

> 每条建议来自**工具维护的经验库**（`experience:<id>`，具体数据集 + 调参 knob + 历史效果），附缺失能力归因链（benchmark 相关性 + bad case → 能力 → 干预）与验证实验设计；完整版与可追溯的资产版本号（含 `experience` 资产）见 `report.md`。规则库与 LLM 可进一步参与（`advisor_mode`）。
