# 诊断引擎设计（v3）— 评测归档 + rule base / llm agent base 双路径

> 本文描述当前实现（取代 v2 的"stages 1-7 始终自动运行"）。核心变化：
> ① 评测产物（分数 + bad case）总是归档为人类可分析的目录；② 诊断**默认关闭**，
> 由参数开启，并分为 **rule base**（默认）与 **llm agent base** 两条路径；
> ③ llm agent base 的工作流以 **skill** 形式提供，由 harness（如 DeepSeek
> Harness）执行，工具只负责 case pack 与启动/交互协议。
>
> v2 的 stages 1-7 管线保留在 `intelligent_diagnosis/` 包中，通过
> `diagnose_model(..., engine="legacy")` 显式调用，作为过渡兼容，不再默认运行。

---

## 1. 评测产物归档（需求 1）

**每次 `run`（无论是否诊断）评测后都会写入输出目录**（`run.output.dir` 或
`--output` 的父目录）：

```
<output>/
├── scores.json               # {benchmark_id: score}，与 --scores 输入同构，可直接回喂
├── eval_results.json         # 机器可读明细：metric / 样本数 / bad case 数 / 预期曲线判定
├── eval_summary.md           # 人类可读概览（分数表 + 判定基准 + bad case 统计）
└── bad_cases/
    ├── README.md             # 索引
    ├── <benchmark_id>.jsonl  # 逐条错题（question / gold / model_output / metrics / doc）
    └── <benchmark_id>.md     # 同内容的人类可读版
```

- bad case 从 harness 的 `--log_samples` 产物（`samples_*.jsonl`）提取：样本级
  primary metric < 0.5 判为失败；无 metric 时用归一化 exact-match 对 gold 兜底；
  无法判定的样本不臆断为错题（`evaluation_orchestration/artifacts.py`）。
- 实现：`write_eval_artifacts()`，在 `runner.execute_run` 里评测后无条件调用。

## 2. 诊断开关与路径（需求 2）

```yaml
diagnosis:
  enabled: false          # 默认关闭；run --diagnose 或此处 true 开启
  engine: rule            # rule | llm_agent
  llm_agent:
    enabled: false        # llm agent 开关
    harness_cmd: null     # harness 启动命令（{case_pack} 占位 / BMD_CASE_PACK 环境变量）
    interact_cmd: null    # harness 交互命令（{case_pack}/{message} 占位 / BMD_* 环境变量）
    eval_cmd: null        # 数据集验证工具（默认本 CLI 的 eval-task）
    skill_path: null      # skill 目录（null => 打包的 skills/benchmark-diagnosis/）
    max_rounds: 5         # 循环轮数上限
    timeout_seconds: 7200 # 每轮 harness 超时
```

- **开启诊断**：`run --diagnose`（或 `diagnosis.enabled: true`），且 `--mode` 为
  `analyze`/`full`。`--mode benchmark` 只评测 + 归档。
- **路径选择**：`diagnosis.engine`（或 `run --engine`）。
- **llm agent 的启用门**：`llm_agent.enabled == true` **且** `harness_cmd` **且**
  `interact_cmd` 全部配置，否则 `run --engine llm_agent` 在评测前就报错
  （`config.resolve_diagnosis_engine` 负责校验，fail-fast）。
- `--mode analyze` = 归因但不输出最终建议（rule base 省略 `dataset_suggestions`，
  llm agent 省略结论中的 `suggestions`）。

## 3. rule base 路径（需求 2.1）— `diagnosis_engine/rule_base.py`

确定性、无 LLM，四个步骤：

1. **低分数据集筛选**：每个已评分 benchmark 用 `expectation_curves.judge()` 判定——
   与**同等参数量（dense/MoE）或同等激活参数量**预期曲线的局部百分位 / z-score，
   低于阈值（`curves.percentile_threshold` 等）即为低分数据集。输出含判定基准
   （`curve_basis`：同等参数量(dense)/同等参数量(MoE)/同等激活参数量/时间前沿）。
2. **数据集-能力映射**：coverage 资产记录每个 benchmark 的能力标签
   （`design_goal_tags`）。低分 benchmark 的 shortfall（低于曲线的幅度）按
   reliability × agreement × saturation 加权，记入其标签解析后的能力
   （flat 标签经 taxonomy `aliases` 解析进层级，并 rollup 到祖先）。
3. **缺失能力过滤**：噪声下限（证据 < 最强能力 5% 丢弃）；**祖先折叠**——某能力的
   证据若被其更具体的后代能力解释（后代证据 ≥ 80%），丢弃笼统的祖先，保留叶子。
4. **能力-数据集建议**：experience 资产（能力-数据集表）为每个缺失能力匹配干预
   （精确匹配优先，祖先/后代次之），输出建议补充的数据集 + 调参 knob + 预期效果；
   probe registry 给出窄口径**验证数据集**；coverage 给出可加评测的 tagged
   benchmarks。

输出块（`report["diagnosis"]["rule_base"]`）：`low_score_benchmarks` /
`missing_capabilities` / `dataset_suggestions`。

## 4. llm agent base 路径（需求 2.2）— `diagnosis_engine/llm_agent.py`

在 rule base 之上运行，由工具 + harness 分工：

1. **工具侧**（`run_llm_agent_diagnosis`）：
   - 先跑 rule base，结论（2.1 结果）写进 case pack，agent 必须采纳；
   - 组装 **case pack**：`<output>/agent_run/`，含 `input/`（scores.json、
     rule_base_result.json、bad_cases/*.jsonl、context.md、evaluation_target.json）、
     `output/`（agent 写结论）、`skill/`（工作流副本）；
   - 用配置的 `harness_cmd` 启动 harness（`{case_pack}` 占位或 `BMD_CASE_PACK`
     环境变量），流式输出，轮询 `output/conclusion.json`；
   - 结论为 `status: "final"` 即停；否则写 `input/followup_<round>.json`（附上一轮
     draft + "继续迭代"指令）并用 `interact_cmd` 发送，最多 `max_rounds` 轮；
   - 每轮 `timeout_seconds` 超时则终止 harness，报告 partial。
2. **harness 侧（agent，按 skill 工作流）**：
   - Step 1 采纳 rule-base 结论；Step 2 bad case 分析（root cause 分类 + 能力
     标签）；Step 3 假设-验证循环：用 **eval 工具**
     （`benchmark-diagnosis eval-task --task <id> --limit <N> --base-url <url>
     --model <m> --out <dir>`，跑数据集子集并归档分数 + bad case）或 bad case
     重分析验证假设，得到新猜想继续；Step 4 写 `output/conclusion.json`
     （`status: "final"`）+ `conclusion.md`。
3. **端点**：`run --base-url/--model-id` 时端点信息进入
   `evaluation_target.json`，agent 可直接跑数据集验证；`--scores` 路径无端点，
   agent 退化为仅 bad case 验证（`available: false`）。

结论 schema 与工作流细节见 `skills/benchmark-diagnosis/SKILL.md`。

## 5. skill（需求 2.3）— `skills/benchmark-diagnosis/SKILL.md`

llm agent base 的分析工作流以 DSH skill 格式提供（frontmatter + 工作流正文），
打包副本位于 `src/benchmark_diagnosis/diagnosis_engine/skill/benchmark-diagnosis/`
（随 pip 安装分发，两份由 `tests/test_diagnosis_engines.py::test_skill_mirrors_are_in_sync`
保证一致）。安装到 DeepSeek Harness：`cp -r skills/benchmark-diagnosis
<项目根>/.dsh/skills/`（或 `$DSH_HOME/skills`）。

## 6. 集成点

- `runner.execute_run`：评测 → 归档（无条件）→ 门控 → 引擎分发 → 报告。
- `pipeline.diagnose_model(engine=...)`：`rule`（默认）/ `llm_agent` / `legacy`。
- CLI：`run --diagnose`、`run --engine`、`eval-task`（agent 的验证工具）。
- 报告：`reporting/report_generator.render_diagnosis_engine` 渲染新块；
  旧块由 `render_diagnosis` 渲染（legacy 引擎）。

## 7. 与 v2 的关系

- v2 stages 1-7（`intelligent_diagnosis/`）不再默认运行，`engine="legacy"` 保留。
- rule base 复用 v2 的离线资产：coverage（数据集-能力映射）、curves（分位判定）、
  experience（能力-数据集表）、taxonomy/probe registry；不再需要 LLM、probe 复测、
  置信度融合等阶段——诊断结论直接来自统计证据与经验库。
- 反馈回路（`feedback` 命令）仍作用于 experience 资产的 outcome 记录，供 rule
  base 的"预期效果"参考。
