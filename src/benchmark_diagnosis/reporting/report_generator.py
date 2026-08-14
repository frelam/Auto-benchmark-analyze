"""Render the full diagnosis report as Markdown / JSON.

The report carries a version trace (which coverage / portfolio / curves assets it
used) so results are reproducible (design doc sections 2.5 / 4.2.5).
"""

from __future__ import annotations

import json
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
    lines.append(f"- **asset versions**: coverage=`{versions.get('coverage_version')}` "
                 f"portfolio=`{versions.get('portfolio_version')}` "
                 f"curves=`{versions.get('curves_version')}`")
    lines.append("")

    clusters = report.get("clusters", [])
    if not clusters:
        lines.append("_No cluster results._")
        return "\n".join(lines)

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
                if r.get("validation_experiment"):
                    lines.append(f"   - *Validation*: {r['validation_experiment']}")
                lines.append("")
        lines.append("---")
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
