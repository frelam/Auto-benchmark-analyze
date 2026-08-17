"""Render the full diagnosis report as Markdown / JSON.

The report carries a version trace (which coverage / portfolio / curves assets it
used) so results are reproducible (design doc sections 2.5 / 4.2.5).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def render_markdown(report: dict[str, Any]) -> str:
    """Render a structured report dict as Markdown."""
    model = report.get("model", {})
    versions = report.get("versions", {})
    lines: list[str] = []
    lines.append(f"# Benchmark Diagnosis Report — {model.get('name', model.get('model_id', 'unknown'))}")
    lines.append("")
    lines.append(f"- **model_id**: `{model.get('model_id', '?')}`")
    lines.append(f"- **arch**: {model.get('arch_type', '?')} | "
                 f"total_params: {_fmt(model.get('total_params'))} | "
                 f"active_params: {_fmt(model.get('active_params'))}")
    lines.append(f"- **generated_at**: {report.get('generated_at', '?')}")
    lines.append(f"- **mode**: {report.get('mode', 'full')} | "
                 f"**advisor**: {report.get('advisor_mode', 'rules')}")
    lines.append(f"- **asset versions**: coverage=`{versions.get('coverage_version')}` "
                 f"portfolio=`{versions.get('portfolio_version')}` "
                 f"curves=`{versions.get('curves_version')}` "
                 f"experience=`{versions.get('experience_version')}`")
    lines.append("")

    clusters = report.get("clusters", [])
    if clusters:
        lines.append(f"## Summary — {len(clusters)} capability cluster(s)")
        lines.append("")
        lines.append("| cluster | weighted score | percentile | z-score | verdict |")
        lines.append("|---|---|---|---|---|")
        for c in clusters:
            verdict = "⚠️ under-performing" if c.get("underperforming") else "✅ in range"
            lines.append(
                f"| `{c['cluster_id']}` | {_fmt(c.get('score'))} | "
                f"{_fmt(c.get('percentile'), 1)} | {_fmt(c.get('z_score'), 2)} | {verdict} |"
            )
        lines.append("")
    else:
        lines.append("_No cluster results._")
        lines.append("")

    for c in clusters:
        lines.append(f"## Cluster `{c['cluster_id']}`")
        lines.append("")
        lines.append("### Benchmarks")
        lines.append("")
        lines.append("| benchmark | weight | score |")
        lines.append("|---|---|---|")
        for b in c.get("benchmarks", []):
            lines.append(f"| `{b['benchmark_id']}` | {_fmt(b.get('weight'), 3)} | {_fmt(b.get('score'))} |")
        lines.append("")

        diagnosis = c.get("diagnosis") or {}
        if diagnosis:
            lines.append("### Diagnosis")
            lines.append("")
            lines.append(f"- sub_capability: `{diagnosis.get('sub_capability')}`")
            lines.append(f"- quantified_gap: {_fmt(diagnosis.get('quantified_gap'), 2)}")
            deficit = diagnosis.get("capability_deficit") or {}
            if deficit:
                lines.append("")
                lines.append("**Missing-capability profile** (from benchmark "
                             "correlation + bad-case analysis):")
                lines.append("")
                lines.append("| capability | deficit strength |")
                lines.append("|---|---|")
                for tag, strength in sorted(
                    deficit.items(), key=lambda kv: kv[1], reverse=True
                ):
                    lines.append(f"| `{tag}` | {_fmt(strength, 3)} |")
                narrative = diagnosis.get("deficit_narrative")
                if narrative:
                    lines.append("")
                    lines.append(f"> {narrative}")
                lines.append("")
            failure_modes = diagnosis.get("failure_modes") or {}
            if failure_modes:
                lines.append("")
                lines.append("| failure mode | fraction |")
                lines.append("|---|---|")
                for mode, frac in failure_modes.items():
                    lines.append(f"| `{mode}` | {_fmt(frac, 3)} |")
        lines.append("")

        recs = c.get("recommendations") or []
        if recs:
            lines.append("### Recommendations")
            lines.append("")
            for i, r in enumerate(recs, 1):
                lines.append(
                    f"**{i}. [{r.get('rule_id', 'external')}] {r.get('action', '')}** "
                    f"_(source: {r.get('source', '?')}, evidence: {r.get('evidence_strength', '?')})_"
                )
                if r.get("expected_effect"):
                    lines.append(f"   - *Expected effect*: {r['expected_effect']}")
                datasets = r.get("datasets") or []
                if datasets:
                    lines.append("   - *Datasets to add*:")
                    for d in datasets:
                        lines.append(
                            f"     - `{d.get('name')}` — {d.get('rationale', '')}"
                        )
                hyperparameters = r.get("hyperparameters") or []
                if hyperparameters:
                    lines.append("   - *Hyperparameter adjustments*:")
                    for h in hyperparameters:
                        lines.append(
                            f"     - `{h.get('knob')}` → {h.get('direction')} "
                            f"(typical range: {h.get('typical_range', '—')})"
                        )
                reason_chain = r.get("reason_chain") or []
                if reason_chain:
                    lines.append("   - *Reasoning*:")
                    for reason in reason_chain:
                        lines.append(f"     - {reason}")
                if r.get("validation_experiment"):
                    lines.append(f"   - *Validation*: {r['validation_experiment']}")
                lines.append("")
        lines.append("---")
        lines.append("")

    intelligent = report.get("intelligent_diagnosis")
    if intelligent:
        lines.append(render_intelligent_diagnosis(intelligent))
        lines.append("")

    figures = report.get("figures") or []
    if figures:
        lines.append("## Figures")
        lines.append("")
        for name in figures:
            lines.append(f"![{name}]({name})")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False, default=str)


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".md"
    if suffix == ".json":
        path.write_text(render_json(report), encoding="utf-8")
    else:
        path.write_text(render_markdown(report), encoding="utf-8")
    return path


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_intelligent_diagnosis(block: dict[str, Any]) -> str:
    """Render the stages 1-7 intelligent diagnosis block as Markdown."""
    lines: list[str] = []
    lines.append("## 智能诊断（Stages 1-7）")
    lines.append("")
    lines.append(
        f"- taxonomy v{block.get('taxonomy_version', '?')} · probe registry "
        f"v{block.get('probe_registry_version', '?')} · 校准 logs="
        f"{(block.get('calibration') or {}).get('n_logs', 0)}"
    )
    lines.append("")

    candidates = block.get("candidates") or []
    if not candidates:
        lines.append("_未发现低于预期的候选能力。_")
        lines.append("")
        return "\n".join(lines)

    lines.append("### 候选能力与复测")
    lines.append("")
    lines.append("| capability | screening | mode | probe 状态 | confidence | suggestion_type |")
    lines.append("|---|---|---|---|---|---|")
    verdicts = {v["capability_id"]: v for v in (block.get("verdicts") or [])}
    probes = {p["capability_id"]: p for p in (block.get("probe_results") or [])}
    for cand in candidates:
        cap = cand["capability_id"]
        verdict = verdicts.get(cap, {})
        probe = probes.get(cap, {})
        probe_state = probe.get("state", "—")
        passk = ""
        if probe.get("pass_1") is not None and probe.get("pass_k") is not None:
            passk = f" pass@1={probe['pass_1']:.2f}/pass@k={probe['pass_k']:.2f}"
        lines.append(
            f"| `{cap}` | {_fmt(cand.get('screening_score'), 2)} | "
            f"{cand.get('evidence_mode', '')} | {probe_state}{passk} | "
            f"{verdict.get('confidence', '—')} | {verdict.get('suggestion_type', '—')} |"
        )
    lines.append("")

    lines.append("### 排序后的建议（Priority = Confidence × ExpectedGain / Cost）")
    lines.append("")
    suggestions = block.get("suggestions") or {}
    training = suggestions.get("training") or []
    if training:
        lines.append("**训练类建议**（按优先级降序）")
        lines.append("")
        for s in training:
            lines.append(
                f"- **{s['capability_id']}** [{s['suggestion_type']}] "
                f"(confidence={s['confidence']}, priority={_fmt(s['priority_score'], 3)}, "
                f"gain={_fmt(s['expected_gain'], 3)})"
            )
            lines.append(f"  - {s['concrete_action']}")
            for ev in s.get("supporting_evidence") or []:
                lines.append(f"  - *evidence*: {ev}")
        lines.append("")
    for group, title in (
        ("non_training", "**非训练类（工程侧路由）**"),
        ("build_probe_first", "**先建 Probe（不直接给训练建议）**"),
        ("human_review", "**转人工复核**"),
    ):
        items = suggestions.get(group) or []
        if not items:
            continue
        lines.append(title)
        lines.append("")
        for s in items:
            lines.append(
                f"- **{s['capability_id']}** [{s['suggestion_type']}] — {s['concrete_action']}"
            )
        lines.append("")

    probe_todo = block.get("probe_todo") or {}
    build = probe_todo.get("build_probe_set") or []
    pending = probe_todo.get("pending_eval") or []
    if build or pending:
        lines.append("### Probe 待办")
        lines.append("")
        if build:
            lines.append(f"- 待建设 probe 集：{', '.join(f'`{c}`' for c in build)}")
        if pending:
            lines.append(f"- 待评测 probe：{', '.join(f'`{c}`' for c in pending)}")
        lines.append("")
    return "\n".join(lines)


def cluster_capability_labels(coverage: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Map each capability cluster to its dominant declared tags (top-3 by frequency).

    The result labels each statistically-discovered cluster with the capabilities
    its member benchmarks officially measure, which is how a cluster is read as
    "which capabilities tend to move together".
    """
    tag_counts: dict[str, Counter] = {}
    for row in coverage:
        counter = tag_counts.setdefault(row["primary_cluster"], Counter())
        for tag in row.get("design_goal_tags") or []:
            counter[tag] += 1
    return {
        cluster: [tag for tag, _ in counter.most_common(3)]
        for cluster, counter in sorted(tag_counts.items())
    }


def render_results_summary(
    *,
    coverage: list[dict[str, Any]],
    portfolios: list[dict[str, Any]],
    curves: list[dict[str, Any]],
    scores: Any,
    benchmark_names: dict[str, str],
    versions: dict[str, str],
    figure_names: list[str],
    generated_at: str,
) -> str:
    """Build a self-contained Markdown summary of the latest offline results.

    Figure paths are referenced relative to the directory that holds this file
    (i.e. ``figures/...``), so the archive can be dropped into a repo verbatim.
    """
    n_models = 0 if scores is None else int(scores.shape[0])
    n_benchmarks = 0 if scores is None else int(scores.shape[1])
    n_scores = 0 if scores is None else int(scores.notna().sum().sum())
    n_clusters = len({row["primary_cluster"] for row in coverage})
    n_curves = len(curves)

    lines: list[str] = []
    lines.append("# Benchmark Diagnosis — Offline Results")
    lines.append("")
    lines.append(f"_Generated {generated_at} from seed reference data "
                 "(approximate public leaderboard values; see `_note` in the seed)._")
    lines.append("")

    lines.append("## Overview")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| models | {n_models} |")
    lines.append(f"| benchmarks | {n_benchmarks} |")
    lines.append(f"| scores | {n_scores} |")
    lines.append(f"| capability clusters | {n_clusters} |")
    lines.append(f"| fitted expectation curves | {n_curves} |")
    lines.append("")
    lines.append("**Asset versions** (every diagnosis report traces back to these):")
    lines.append("")
    for key, value in versions.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")

    if coverage:
        lines.append("## Capability-coverage table")
        lines.append("")
        lines.append("| benchmark | cluster | breadth | reliability | saturated | "
                     "design-goal agreement | tags |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in sorted(coverage, key=lambda r: r["benchmark_id"]):
            name = benchmark_names.get(row["benchmark_id"], row["benchmark_id"])
            tags = ", ".join(row.get("design_goal_tags") or [])
            lines.append(
                f"| `{name}` | `{row['primary_cluster']}` | "
                f"{row['coverage_breadth_score']:.2f} | {row['reliability_score']:.2f} | "
                f"{'⚠️' if row.get('saturated_flag') else ''} | "
                f"{row['design_goal_agreement_score']:.2f} | {tags} |"
            )
        lines.append("")

    cluster_labels = cluster_capability_labels(coverage)
    if cluster_labels:
        lines.append("## Capability clusters")
        lines.append("")
        lines.append("_Clusters group benchmarks whose scores co-vary across models — "
                     "i.e. correlated capabilities that tend to improve together "
                     "during training._")
        lines.append("")
        for cluster, tags in cluster_labels.items():
            lines.append(f"- **{cluster}**: {', '.join(tags)}")
        lines.append("")

    if portfolios:
        lines.append("## Representative portfolios")
        lines.append("")
        for pf in sorted(portfolios, key=lambda p: p["cluster_id"]):
            combos = ", ".join(
                f"`{b['benchmark_id']}` ({b['weight']:.2f})" for b in pf["benchmarks"]
            )
            lines.append(f"- **{pf['cluster_id']}**: {combos}")
        lines.append("")

    if figure_names:
        lines.append("## Figures")
        lines.append("")
        for name in figure_names:
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"![{name}](figures/{name})")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
