# Benchmark 智能诊断与优化建议 — 设计文档 v2（已实现）

> 本文是 `benchmark_diagnosis_suggestion_design.md`（v1）的优化落地版：保留 v1 的
> 7 阶段骨架，修正其实现层面的模糊点，补充统计严谨性与业界做法，并给出本仓库
> `src/benchmark_diagnosis/intelligent_diagnosis/` 中每一模块的对应实现。
> 代码模块：`intelligent_diagnosis/{types, capability_taxonomy, candidate_generation,
> probe_registry, guided_case_analyzer, confidence_fusion, priority_scorer,
> suggestion_writer, feedback, orchestrator}.py`。

---

## 0. 与 v1 相比的关键改进

| # | v1 的模糊点 | v2 的处理 | 依据 |
|---|---|---|---|
| 1 | `NO_PROBE` 把"没有 probe"与"有 probe 但没跑"混为一谈 | 拆成 `NO_PROBE`（进"待建设 probe 集"）与 `PENDING_EVAL`（进"待评测 probe 清单"），后续动作不同 | CheckList 的 MFT 思路：每个能力配 1~2 个窄口径最小功能测试 |
| 2 | sub-accuracy 显著性检验未给样本量约束 | 最小 item 数 / 最小 peer 数门槛；小样本用 **Wilson 下界**做保守估计；百分位用 rank 方法 | 小样本均值噪声大，v1 承认"样本量通常不够拟合"，那就用区间估计而非点估计 |
| 3 | pass@1 vs pass@k 只有定性描述 | 给出 `passk_gap_ratio` 与阈值；k 用 Chen et al. 无偏估计；要求最小采样数 | "Hard or Just Unreached?"：pass@1 低而 pass@k 高 = 采样盲区（能力存在但触发不稳定），这正是拒绝采样的修复对象 |
| 4 | 置信度判定表是离散格子，冲突时无机制 | 证据加权分数 + 冲突检测：任一输入缺失即降档；矛盾组合显式转人工 | 多重独立证据原则：缺一路验证就封顶 |
| 5 | suggestion_type 判定表多行可能同时命中 | 给**优先级序**：grading 路由 > format 路由 > probe 路由 > 组合能力 > 数据配比 > passk 二选一 | 工程侧修复成本最低，先路由出去，避免误报训练问题 |
| 6 | ExpectedGain 只有点估计 | 增加 Wilson 下界版 `gain_lower`，排序用下界，避免小样本虚高 | SliceTeller SliceBoosting：按"修复 slice 的性价比"排序 |
| 7 | Cost 表"可迭代"但无机制 | `feedback` 模块：执行日志入库 → 按 suggestion_type 平滑校准 Cost、校准 gain 缩放因子，落盘为 versioned asset | DataChef：用上一轮真实执行效果校准下一轮估计（本文把该思想落在"人工诊断-修复动作"层） |
| 8 | item 级打标是"优先做" | 定义为 versioned asset（`benchmark_tagger` 离线产出），代码支持 fine/coarse 双模式自动降级 | 无 item 标签时证据强度打折，v1 已同意 |

---

## 1. 总体 Pipeline（与 v1 相同骨架）

```
[Stage 0 已有] 相关性聚类 + 代表性 benchmark 初筛（below-expectation 判定）
        ▼
[Stage 1] 候选能力生成 candidate_generation.generate_candidates()
        ▼
[Stage 2] 单项能力复测 probe_registry.verify_candidates()
        ▼
[Stage 3] 引导式 Bad Case 分析 guided_case_analyzer.analyze_cases()
        ▼
[Stage 4] 置信度融合 + 建议类型判定 confidence_fusion.fuse()
        ▼
[Stage 5] 优先级排序 priority_scorer.score_priorities()
        ▼
[Stage 6] 建议文案生成 suggestion_writer.write_suggestions()
        ▼
[Stage 7] 执行反馈回收 feedback.{log_execution, recalibrate}()
```

依赖的已有资产（复用本仓库 Stage 0 产物，不重复建设）：

- `curves` asset：expectation curve，`judge()` 产出 per-benchmark 的 residual / percentile / z-score；
- `coverage` asset：benchmark 级能力标签（`design_goal_tags`）与可靠性分；
- `portfolio` asset：簇 → 代表性 benchmark 及其权重；
- 历史模型 item 级分数（`score_records.item_id`）：Stage 1 fine 模式的 peer 参照；
- `benchmark_tagger` 离线产物（`item_capabilities: item_id -> [capability_id]`）：Stage 1 fine 模式的 item 标签，缺失时自动降级 coarse。

---

## 2. 能力分类体系（Stage 0~6 共享语言）

v1 要求"有层级、可版本管理"。实现为 `data/capability_taxonomy.yaml`：

```yaml
version: 1
capabilities:
  - id: reasoning.math.multi_step
    name: 多步数学推理
    parent: reasoning.math
    description: ...
```

关键点：

- `parent` 必须存在（root 的 parent 为 null），loader 校验：id 唯一、parent 闭环、层级 `level` 由深度推导；
- 提供 `ancestors(id)` / `descendants(id)` / `is_within(hypothesis, tag)`：Stage 3 的"LLM 改标一致性"用 `is_within` 判定（LLM 标成假设能力的子能力算一致，标成兄弟能力算 near-miss 并单独统计）；
- 与已有 `recommendation/rule_base/taxonomy.yaml`（扁平 failure modes）**并存不冲突**：那是失败模式词表，这是能力本体；Stage 3 的 `capability_tag` 从本 taxonomy 取，`root_cause_type` 是另一维度。

## 3. Stage 1：候选能力生成（`candidate_generation.py`）

输入：below-expectation benchmark 的 verdict 列表（`judge()` 输出 + portfolio 权重）、
coverage asset、历史 item 分数（peer 参照）、可选 item 能力标签。

**fine 模式（有 item 标签）**：

1. 对 benchmark `b` 内每个能力 `c`：`sub_acc_b,c = mean(correct over items tagged c)`;
2. peer 参照：同一批历史模型在这些 item 上的 sub-accuracy 分布（`percentileofscore(..., kind="rank")`）；
3. **统计门槛**（默认值见 config）：`n_items >= min_items_per_capability`（默认 8）、
   `n_peers >= min_peers`（默认 5），任一不满足 → 该项标 `low_support`，screening_score 打折 0.5；
4. `screening_score = clamp((1 - percentile/100) * support_penalty, 0, 1)`，同时记录
   `sub_acc`、`peer_mean`、`wilson_lb`（模型 sub-acc 的 Wilson 下界，供 Stage 5 用）；
5. 只保留 `percentile < percentile_threshold`（与 Stage 0 同阈值）的能力。

**coarse 模式（无 item 标签）**：benchmark 所有 `design_goal_tags` 进候选，
`screening_score = max(0, -residual) * 0.5`，`evidence_mode = "coarse"`。

**OR 汇总**：`D_candidate = ⋃`；同一能力多来源取 max(screening_score)，记录
`sources = [benchmark_id,...]`（Stage 5 的 footprint 用）。

## 4. Stage 2：单项能力复测（`probe_registry.py`）

注册表 `data/probe_registry.yaml`：`capability_id -> [{benchmark_id, note}]`（每能力 1~2 个窄口径 probe）。

判定（对每个候选能力）：

- 注册表无条目 → `NO_PROBE`（输出到"待建设 probe 集"）；
- 有条目但本轮没有该 probe 的分数 → `PENDING_EVAL`（输出到"待评测 probe 清单"，置信度封顶同 NO_PROBE）；
- 有条目且有分数：peer 百分位（历史模型在该 probe 上的分数分布）`< threshold`
  → `CONFIRMED`；否则 `NOT_CONFIRMED`（源头 benchmark 仍低分 → 转"组合能力缺陷"候选）。

**pass@1 / pass@k**（可选输入 `passk_stats`，`samples` 为采样数）：

- `passk_gap_ratio = (pass_k - pass_1) / max(pass_1, eps)`，`gap = ratio > passk_gap_threshold`（默认 0.5）
  且 `pass_1 < pass1_high_threshold`（默认 0.5）且 `samples >= min_passk_samples`（默认 8）；
- 该信号进入 Stage 4：`gap=True → rejection_sampling`，否则 `targeted_synthesis`。

## 5. Stage 3：引导式 Bad Case 分析（`guided_case_analyzer.py`）

与 v1 一致：LLM 拿到候选能力假设，做"验证 + 细化"而非开放式归纳。每条 case 输出：

```json
{"case_index": 0, "root_cause_type": "content_error|format_error|grading_artifact|label_noise|ambiguous",
 "capability_tag": "<taxonomy id，允许改标>", "evidence": "..."}
```

聚合输出：

- `content_error_ratio` / `format_error_ratio` / `grading_artifact_ratio` / `label_noise_ratio` / `ambiguous_ratio`（按已分类 case 计数）；
- `tag_agreement`：`capability_tag` 与假设能力满足 `is_within` 的比例；`near_miss_ratio` 单独统计；
- **饱和停止**：`saturation_window`（默认 20）条连续无新 (root_cause_type, capability_tag) 组合即停止扩大采样，返回 `cases_analyzed`。

无 LLM 时返回空分析（Stage 4 对缺失证据自动降档），与现有 `classify_failures` 的降级语义一致。

## 6. Stage 4：置信度融合 + 建议类型判定（`confidence_fusion.py`）

**置信度**：先算证据分 `evidence = w1*probe(1/0.5/0) + w2*content_ratio + w3*tag_agreement`
（权重可配），再按 v1 表映射档位，另加两条硬规则：

- 任一关键输入缺失（无 probe 且无 case 分析）→ 上限 `Medium`；
- 冲突组合（如 CONFIRMED 但 content_error_ratio 低且 format 也不高）→ `Low` + `needs_human_review=True`。

档位 → 数值：`High=1.0, Medium-High=0.75, Medium=0.5, Low=0.0`（Low 不进排序）。

**suggestion_type**（按优先级序判定，命中即返回）：

| 序 | 条件 | suggestion_type |
|---|---|---|
| 1 | `grading_artifact_ratio > 0.5` | `eval_infra_fix`（评测基建，非模型问题） |
| 2 | `format_error_ratio > 0.5` | `non_training_fix`（prompt 模板/输出解析器/reward shaping） |
| 3 | probe ∈ {NO_PROBE, PENDING_EVAL} | `build_probe_first`（不直接给训练建议） |
| 4 | NOT_CONFIRMED 且 content 高 | `compositional_curriculum`（组合能力缺陷） |
| 5 | CONFIRMED 且提供 data_share 且偏低 | `data_reweighting` |
| 6 | CONFIRMED 且 pass@k gap | `rejection_sampling` |
| 7 | CONFIRMED | `targeted_synthesis` |
| 8 | 其余 | `human_review` |

## 7. Stage 5：优先级排序（`priority_scorer.py`）

```
gap_b,c    = max(0, peer_mean_sub_acc_b,c − sub_acc_b,c) × item_weight_share_b,c   # fine
           = max(0, −residual_b) × 1.0                                            # coarse
ExpectedGain(c)  = Σ_b weight_b × gap_b,c          （b 遍历候选 c 的 source benchmarks，
                  即有实测短差的 benchmark；仅打标但未低于预期的 benchmark 贡献为 0）
gain_lower(c)    = 用 sub-accuracy 的 Wilson 上界（wilson_ub）替代 sub_acc 的同一公式
                   （小样本防虚高：模型真实水平可能比测得的更高，真实缺口更小）
Cost(c)          = cost_table[suggestion_type(c)]  （feedback 校准后）
Priority(c)      = confidence_num(c) × ExpectedGain(c) / Cost(c)
```

- 排序键：`Priority` 降序；`Low` 置信度不参与排序，转人工；
- 输出 `PriorityItem{capability_id, expected_gain, gain_lower, cost, priority, confidence, suggestion_type}`。

## 8. Stage 6：建议文案（`suggestion_writer.py`）

- **LLM 模式**：prompt 只做"实例化参数 + 写依据"，输出 schema 与 v1 相同
  （capability / root_cause_type / confidence / priority_score / suggestion_type /
  concrete_action / supporting_evidence / expected_gain）；`supporting_evidence` 引用的
  数字必须来自证据集（数字 groundedness 校验，复用 `recommendation.groundedness_check` 的数值校验思路）；
- **确定性模式**（无 LLM）：按 suggestion_type 用模板生成 concrete_action，证据逐条列出；
- 输出分组：`training`（按 priority 降序）/ `non_training` / `build_probe_first`，互不混排。

## 9. Stage 7：执行反馈回收（`feedback.py`）

- `execution_logs` 表：`capability_id, suggestion_type, predicted_gain, actual_gain,
  predicted_cost, actual_cost, created_at, note`；
- `recalibrate()`：按 suggestion_type 聚合 `ratio_cost = mean(actual_cost/predicted_cost)`、
  `ratio_gain = mean(actual_gain/predicted_gain)`，用指数平滑与基准表合并，clamp 到
  `[0.5, 3.0]`，落盘为 versioned asset `calibration`（含 `n_logs`、`updated_at`）；
- 排序时读取该 asset：`effective_cost = base_cost × ratio_cost`，
  `effective_gain = expected_gain × ratio_gain`——这就是"跑得越久排得越准"的机制落点。

## 10. 模块划分与集成

```
src/benchmark_diagnosis/intelligent_diagnosis/
  types.py                  # 各阶段共享 dataclass + 枚举
  capability_taxonomy.py    # 层级 taxonomy 加载/校验
  candidate_generation.py   # Stage 1
  probe_registry.py         # Stage 2（注册表加载 + 复测判定）
  guided_case_analyzer.py   # Stage 3
  confidence_fusion.py      # Stage 4
  priority_scorer.py        # Stage 5
  suggestion_writer.py      # Stage 6
  feedback.py               # Stage 7（execution log + recalibration）
  orchestrator.py           # 串起 Stage 1~7，产出 report 片段
  data/*.yaml               # capability_taxonomy / probe_registry / cost_table
```

集成点：`pipeline.diagnose_model(...)` 现在是统一管线，Stage 0（簇级 verdict 聚合）
之后直接跑 Stages 1-7 并把结果写入 `report["diagnosis"]`；CLI 单一 `run` 命令
（`--mode analyze` 跳过 Stage 6 建议文案）；外加 `feedback log` /
`feedback recalibrate` / `feedback list` 子命令。`diagnose` 命令、`--intelligent`
flag 与 `diagnosis.intelligent` 配置项已合并移除（旧配置仍带 `diagnosis.intelligent`
键时 `load_config` 会显式报错）。

## 11. 已知局限与演进（同 v1，+ 新增）

- probe 覆盖是最大短板：`NO_PROBE`/`PENDING_EVAL` 清单是持续产出，反哺注册表；
- `data_reweighting` 依赖训练数据配比接口，缺失时只给方向性建议；
- 组合能力缺陷目前只识别"存在"不定位"哪几个原子能力的组合"，若占比升高再建组合标签层；
- 长期：规则 + 执行反馈积累到量级后，Stage 4~6 的规则匹配可参照 DataChef
  （RL + 评测反馈做 proxy reward）演进为自动 data recipe 生成模型（后置方向）。
