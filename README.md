# benchmark-diagnosis

一站式大模型评测 + 低分归因诊断 + 优化建议系统。给一个模型（**推理服务 IP** / **权重文件** / **分数文件**，三选一），自动完成：

**评测 → 能力簇诊断（量化差距、失败模式）→ 可追溯优化建议**，并产出报告 + 指标 + 图表。

技术选型见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

---

## 安装

```bash
bash scripts/install.sh                  # CPU 安全（评测 lm-eval + 图表 matplotlib）
bash scripts/install.sh --with-gpu       # 追加 vLLM 部署 + torch
bash scripts/install.sh --download-data  # 追加预取 benchmark 数据集（尽力而为）
```

> 需 Python ≥ 3.10，装到仓库 `.venv/` 内，之后 `source .venv/bin/activate`。

---

## 一个执行例子（无需 GPU / 端点）

```bash
bash examples/run_example.sh
# 等价于：benchmark-diagnosis --config examples/run-from-scores.yaml run
```

（入参的另一种形式：命令后直接跟参数，效果相同）

```bash
benchmark-diagnosis run --model llama-3-8b --scores examples/scores.json \
  --arch dense --params 8 --release-date 2024-04-01
```

输入 `examples/scores.json`：一份 `{benchmark_id: score}` 的分数文件（`mmlu_pro`/`math`/`swe_bench`/`simpleqa`/`longbench_v2` 等）。

## 输出结果

### ① 命令行输出

```
Building offline assets (coverage/portfolio/curves)...
Loaded 8 score(s) from examples/scores.json
Report: examples/output/report.md
Metrics: examples/output/metrics.json
Figures: 3 written
Report written to examples/output/report.md
Metrics written to examples/output/metrics.json
3 cluster(s) analysed, 3 under-performing (mode=full, advisor=rules, figures=3)
```

### ② 报告（`examples/output/report.md` 节选）

```markdown
# Benchmark Diagnosis Report — Llama-3-8B
- **mode**: full | **advisor**: rules
- **asset versions**: coverage=`coverage-…` portfolio=`portfolio-…` curves=`curves-…`

## Summary — 3 capability cluster(s)
| cluster   | weighted score | percentile | z-score | verdict            |
|-----------|----------------|-----------|---------|--------------------|
| cluster_0 | 38.666         | 0.0        | -2.75   | ⚠️ under-performing |
| cluster_1 | 24.454         | 1.7        | -2.77   | ⚠️ under-performing |
| cluster_2 | 18.000         | 23.1       | -0.81   | ⚠️ under-performing |

## Cluster cluster_1
### Benchmarks
| benchmark | weight | score |
|-----------|--------|-------|
| simpleqa  | 0.935  | 24.000|
| ifeval    | 0.037  | —     |
| math      | 0.027  | 40.000|
### Diagnosis
- sub_capability: `simpleqa`
- quantified_gap: 0.25          # 比预期曲线低了 25 分
### Recommendations
1. [R-factuality-001] 提高含证据链、引用溯源、事实核查的语料与 SFT 样本配比…
   (source: rule_base:R-factuality-001, evidence: high)
   - Validation: 在小规模子集上做消融…对比目标 benchmark 的簇级得分变化…
```

### ③ 图表（`examples/output/figures/`，自动生成并内嵌到报告）

![得分柱状图](examples/output/figures/fig_model_scores.png)

![簇判定柱状图](examples/output/figures/fig_model_clusters.png)

![量化差距图](examples/output/figures/fig_model_gaps.png)

---

## 七种使用场景

| # | 场景 | 说明 |
|---|------|------|
| 1 | **两种入参** | 设置要么写 YAML（`--config`），要么命令后跟参数；参数优先 |
| 2 | **模型来源** | `--model-id` 权重→自动 vLLM 部署；`--base-url` IP→自动选评测路径；`--scores` 文件→跳过评测 |
| 3 | **两种模式** | `--mode analyze` 评测+分析（无建议）；`full` 评测+分析+建议（默认） |
| 4 | **Advisor 自动选择** | 配了 `llm.model` → LLM+规则；否则自动退回纯规则；强配 LLM 没模型会在评测前快速失败 |
| 5 | **自动图表** | 每次运行自动产出 `report.md` + `metrics.json` + 3 张图（得分/簇判定/量化差距） |
| 6 | **自动安装** | `scripts/install.sh` 一键装包，`--download-data` 可选预取数据集 |
| 7 | **可执行例子** | `examples/` 一键运行 + 已提交输出（见上） |

**命令速查：**

```bash
# 有权重：自动部署 → 评测 → 分析 → 建议
benchmark-diagnosis run --model my-model --model-id meta-llama/Llama-3.1-8B
# 只有 IP
benchmark-diagnosis run --model my-model --base-url http://10.0.0.5:8000/v1
# 只有分数
benchmark-diagnosis run --model my-model --scores scores.json
# 只要分析不要建议
benchmark-diagnosis run --model my-model --base-url http://10.0.0.5:8000/v1 --mode analyze
```

> 低层命令仍在：`deploy`（打印/拉起部署）、`eval-model`（只评测）、`diagnose`（`run` 的端点/分数子集，不部署）。代表性组合之外的分数（如 `mmlu`/`gsm8k`）会展示但不参与簇判定。

---

## 测试

```bash
.venv/bin/python -m pytest -q        # 112 个测试
.venv/bin/python -m ruff check src tests scripts
```

## 说明

- `data/seed/seed_reference.json` 是**近似值**（公开 leaderboard 近似数，覆盖至 2026 年初前沿模型，22 个 benchmark 含 SWE-bench / τ²-bench / AIME 2025 / FrontierMath / SimpleQA / MMMU / LongBench V2 等），仅用于 bootstrap 预期曲线，生产前请用真实数据替换。
- 参数量不可得的闭源模型，预期曲线自动走"时间→前沿包络"维度判定，不会被漏掉。
- 目录结构与扩展点见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。
