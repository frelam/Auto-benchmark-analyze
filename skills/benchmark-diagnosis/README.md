# benchmark-diagnosis skill（DeepSeek Harness 工作流）

`SKILL.md` 是 llm-agent-base 诊断（设计文档 2.2/2.3）的 harness 工作流：采纳
rule-base 结论 → bad case 分析 → 假设-验证循环（可执行 `eval-task` 数据集评测）→
写回 `output/conclusion.json`。

## 安装到 DeepSeek Harness

DSH 从以下目录发现 skill（rank 高的优先）：

| rank | 位置 |
|---|---|
| 100 | `<项目根>/.dsh/skills` |
| 200 | `<项目根>/.agents/skills` |
| 400 | `$DSH_HOME/skills`（用户级） |

```bash
# 项目级安装（推荐，随仓库走）
mkdir -p .dsh/skills
cp -r skills/benchmark-diagnosis .dsh/skills/

# 或用户级安装
mkdir -p ~/.dsh/skills
cp -r skills/benchmark-diagnosis ~/.dsh/skills/
```

装好后在 DSH 会话中让 agent 加载 `benchmark-diagnosis` skill 即可按工作流执行。

> 本目录是 `src/benchmark_diagnosis/diagnosis_engine/skill/benchmark-diagnosis/`
> 的镜像（打包副本随 pip 安装分发，`agent_run/skill/` 也用它）。两者由测试保证一致，
> 修改请同步两处（`tests/test_skill_sync.py`）。
