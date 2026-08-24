---
name: benchmark-diagnosis
description: Use when asked to run LLM-agent-base benchmark diagnosis — adopt the rule-base conclusions, analyze bad cases, and iterate a hypothesis→verification→new-hypothesis loop (using the eval tool) until a final, evidence-backed conclusion is written to the case pack's output/conclusion.json
---

# LLM-Agent-Base Benchmark 诊断工作流

你在为 `benchmark-diagnosis` 执行 **llm agent base 诊断**（设计文档 2.2）。case pack
目录（环境变量 `BMD_CASE_PACK`，或任务描述中的路径）里有全部输入，你的工作是把诊断
推进到**可验证的最终结论**并写回 `output/conclusion.json`。

## 输入（case pack 结构）

```
<case_pack>/
├── input/
│   ├── scores.json               # 各 benchmark 分数
│   ├── rule_base_result.json     # rule-base 诊断结论（2.1 结果）
│   ├── bad_cases/<benchmark>.jsonl  # 错题（question/gold/model_output/metrics）
│   ├── evaluation_target.json    # 评测端点（base_url）与模型元信息
│   └── context.md                # 上下文与命令速查
├── output/                       # 你的输出目录：conclusion.json / conclusion.md
└── skill/SKILL.md                # 本工作流
```

## 工作流

### Step 1 — 采纳 rule-base 结论（必须）

读取 `input/rule_base_result.json`：`low_score_benchmarks`（低分数据集）与
`missing_capabilities`（缺失能力）是**起点结论**，不得丢弃。除非你有强证据推翻，
否则最终结论必须覆盖它们。

### Step 2 — bad case 分析

对每个低分 benchmark，读 `input/bad_cases/<benchmark>.jsonl`（每个低分 benchmark
至少抽样 10 条，不足则全看）。对每条 bad case 判定：

- `root_cause_type`: `content_error`（内容/推理错）| `format_error`（格式错）|
  `grading_artifact`（判分问题）| `label_noise`（标签噪声）| `ambiguous`（题意含糊）
- `capability_tag`: 该题失败暴露的能力（用 rule_base 的 `missing_capabilities`
  词汇，允许细化为其子能力）

统计：各 root cause 占比、各能力占比、与 rule-base 结论一致/冲突的地方。
**结论：rule-base 的缺失能力是否成立？哪些能力其实没问题（如 gradding artifact）？**

### Step 3 — 假设-验证循环（最多 max_rounds 轮）

对每个候选假设（如"缺失 `reasoning.math.multi_step`"），用下列手段**验证**：

- **数据集评测（首选，端点可用时）**：执行 eval 工具
  `benchmark-diagnosis eval-task --task <benchmark_id> --limit <N> --base-url <base_url> --model <model_id> --out <case_pack>/output/eval_round_<i>`
  然后读 `<out>/scores.json` 与 `<out>/bad_cases/` 判断假设是否被支持。用
  `--limit` 控制成本（如 100~300 条）；要完整验证就去掉 `--limit`。
- **bad case 重分析**：端点不可用时（`evaluation_target.json` 的
  `available: false`），回到 Step 2 的 bad case 上做更细的对照分析（如
  对比 gold 与输出、检查是否格式/判分问题）。

每一轮结束更新结论：确认的假设进入 `conclusions`（标注 `verified_by`），被推翻的
写明原因；若验证产生新猜想（如"实际缺的是 X 的子能力 Y"），下一轮验证它。

### Step 4 — 写最终结论

把最终结论写入 `<case_pack>/output/conclusion.json`（**status 必须为 "final"**），
并附一份人读版 `conclusion.md`。schema：

```json
{
  "status": "final",
  "round": 2,
  "summary": "一句话总结",
  "conclusions": [
    {
      "capability_id": "reasoning.math.multi_step",
      "confidence": "high | medium | low",
      "evidence": "为什么认为是这个能力",
      "verified_by": ["eval:math:limit=200", "bad_case:math:12/15 为计算错误"]
    }
  ],
  "suggestions": [
    {
      "capability_id": "reasoning.math.multi_step",
      "action": "补充多步数学推理训练数据",
      "datasets": ["MathScale", "GSM8K 难子集"],
      "expected_gain": 0.05
    }
  ],
  "bad_case_analysis": {
    "n_cases": 15,
    "root_causes": {"content_error": 0.8, "format_error": 0.2}
  }
}
```

## 约束

1. **结论必须可追溯**：每条 conclusion 的 `evidence` 引用具体 benchmark 分数、
   bad case 样本或 eval 结果；禁止凭空断言。
2. **先采纳后修正**：rule-base 结论是基线；推翻它需要 bad case 或 eval 的明确证据。
3. **验证优先**：端点可用时，能跑 eval 验证的假设一定要跑（用 `--limit` 控制成本）。
4. **conflict 显式化**：bad case 分析若与 rule-base 结论冲突（如低分是判分问题而非
   能力缺失），在 `summary` 里明确说明，并把结论改为 `confidence: low` + 人工复核。
5. 若你在多轮中被要求继续（收到 `input/followup_*.json`），说明上一轮结论未被确认：
   基于 `previous_draft` 继续迭代，直到你能写出 `status: "final"` 的结论为止。
