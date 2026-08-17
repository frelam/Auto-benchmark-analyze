"""Stage 3: guided bad-case analysis (design doc v2 section 5).

Unlike open-ended error analysis, the LLM already holds the Stage 1/2
hypothesis (candidate capability) and is asked to *verify and refine* it:

* per case: ``root_cause_type`` from a fixed vocabulary
  (content/format/grading/label_noise/ambiguous), a ``capability_tag`` drawn
  from the capability taxonomy (relabeling allowed — agreement is measured with
  ``is_within``, near misses with ``near_miss``), and one line of evidence;
* aggregation: content/format/grading/label-noise ratios + tag agreement, which
  Stage 4 fuses into confidence;
* saturation stopping: sampling stops after ``saturation_window`` consecutive
  cases with no new (cause, tag) pair, instead of a fixed budget.

Degrades gracefully (empty analysis) when the LLM is unavailable or a batch
fails to parse, matching ``diagnosis.failure_mode_analyst`` semantics.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from benchmark_diagnosis.intelligent_diagnosis.capability_taxonomy import (
    CapabilityTaxonomy,
)
from benchmark_diagnosis.intelligent_diagnosis.types import (
    CaseAnalysis,
    CaseVerdict,
    RootCauseType,
)

_BATCH_SIZE = 10

_VALID_CAUSES = {cause.value for cause in RootCauseType}


class GuidedCaseAnalyzer:
    """Guided bad-case analysis for one hypothesis capability."""

    def __init__(
        self,
        llm: Any,
        taxonomy: CapabilityTaxonomy,
        *,
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        self._llm = llm
        self._taxonomy = taxonomy
        self._batch_size = batch_size

    # ------------------------------------------------------------------ public

    def analyze(
        self,
        hypothesis: str,
        cases: list[dict],
        *,
        sample_size: int = 50,
        saturation_window: int = 20,
    ) -> CaseAnalysis:
        """Analyze failed cases against ``hypothesis``.

        Args:
            hypothesis: Candidate capability id from Stage 1/2.
            cases: Item-level failed cases (``question`` / ``model_output`` /
                ``gold``, optional ``benchmark_id`` / ``item_id``).
            sample_size: Hard cap on the number of cases to consider.
            saturation_window: Stop early when this many *consecutive classified*
                cases add no new (cause, tag) pair.

        Returns:
            Aggregated :class:`CaseAnalysis`.
        """
        sampled = cases[:sample_size]
        if not sampled or hypothesis not in self._taxonomy.nodes:
            return CaseAnalysis()

        verdicts: list[CaseVerdict] = []
        seen_pairs: set[tuple[str, str]] = set()
        consecutive_without_new = 0
        saturated = False

        for start in range(0, len(sampled), self._batch_size):
            batch = sampled[start : start + self._batch_size]
            batch_verdicts = self._classify_batch(hypothesis, batch)
            for verdict in batch_verdicts:
                verdicts.append(verdict)
                pair = (verdict.root_cause_type.value, verdict.capability_tag)
                if pair in seen_pairs:
                    consecutive_without_new += 1
                else:
                    seen_pairs.add(pair)
                    consecutive_without_new = 0
            if (
                len(verdicts) >= self._batch_size
                and consecutive_without_new >= saturation_window
            ):
                saturated = True
                break

        return self._aggregate(hypothesis, verdicts, saturated=saturated)

    # ----------------------------------------------------------------- private

    def _classify_batch(self, hypothesis: str, batch: list[dict]) -> list[CaseVerdict]:
        """Classify one batch; parse failures are skipped (not counted)."""
        try:
            data = self._llm.complete_json(
                self._build_messages(hypothesis, batch)
            )
        except Exception:  # noqa: BLE001 - analyst unavailability degrades analysis
            return []
        single = len(batch) == 1
        verdicts: list[CaseVerdict] = []
        for index in range(len(batch)):
            verdict = self._extract(hypothesis, data, index, single=single)
            if verdict is not None:
                verdicts.append(verdict)
        return verdicts

    def _build_messages(
        self, hypothesis: str, batch: list[dict]
    ) -> list[dict[str, str]]:
        node = self._taxonomy.nodes[hypothesis]
        cause_lines = "\n".join(
            f"- {cause.value}: "
            + {
                RootCauseType.CONTENT_ERROR: "内容/推理本身错误（能力假设的直接佐证）",
                RootCauseType.FORMAT_ERROR: "内容正确但输出格式不合规",
                RootCauseType.GRADING_ARTIFACT: "评测脚本/抽取逻辑误判（非模型问题）",
                RootCauseType.LABEL_NOISE: "标注/参考答案本身错误",
                RootCauseType.AMBIGUOUS: "题目有歧义或证据不足以判定",
            }[cause]
            for cause in RootCauseType
        )
        valid_ids = sorted(self._taxonomy.ids)
        case_lines = "\n".join(
            f"[{index}] question: {case.get('question', '')}\n"
            f"    model_output: {case.get('model_output', '')}\n"
            f"    gold: {case.get('gold', '')}"
            for index, case in enumerate(batch)
        )
        system = (
            "You are an LLM failure analyst. A diagnosis hypothesis says the model "
            "is deficient in capability "
            f"'{hypothesis}' ({node.name}). "
            "For each failed case: (1) pick EXACTLY ONE root_cause_type from the "
            "fixed vocabulary; (2) pick the closest capability_tag from the "
            "taxonomy — you may agree with the hypothesis or relabel it; "
            "(3) give one short evidence line. Never invent ids. Reply with JSON "
            '{"classifications": [{"case_index": 0, "root_cause_type": "<id>", '
            '"capability_tag": "<id>", "evidence": "..."}, ...]} covering every case.'
        )
        user = (
            f"HYPOTHESIS CAPABILITY: {hypothesis} ({node.name})\n\n"
            f"ROOT_CAUSE VOCABULARY:\n{cause_lines}\n\n"
            f"CAPABILITY TAXONOMY IDS:\n{', '.join(valid_ids)}\n\n"
            f"CASES:\n{case_lines}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _extract(
        self,
        hypothesis: str,
        data: Any,
        case_index: int,
        *,
        single: bool = False,
    ) -> CaseVerdict | None:
        """Extract one valid verdict; None when the entry is unusable."""
        entry: dict | None = None
        if isinstance(data, dict):
            classifications = data.get("classifications")
            if isinstance(classifications, list):
                for item in classifications:
                    if (
                        isinstance(item, dict)
                        and item.get("case_index") == case_index
                    ):
                        entry = item
                        break
            elif single and isinstance(data.get("root_cause_type"), str):
                entry = data
        if entry is None:
            return None
        cause = entry.get("root_cause_type")
        tag = entry.get("capability_tag")
        if not isinstance(cause, str) or cause not in _VALID_CAUSES:
            return None
        if not isinstance(tag, str):
            return None
        resolved = self._resolve_tag(tag)
        if resolved is None:
            return None
        return CaseVerdict(
            case_index=case_index,
            root_cause_type=RootCauseType(cause),
            capability_tag=resolved,
            evidence=str(entry.get("evidence", "")),
        )

    def _resolve_tag(self, tag: str) -> str | None:
        if tag in self._taxonomy.nodes:
            return tag
        for node in self._taxonomy.nodes.values():
            if tag in node.aliases:
                return node.id
        return None

    def _aggregate(
        self,
        hypothesis: str,
        verdicts: list[CaseVerdict],
        *,
        saturated: bool,
    ) -> CaseAnalysis:
        """Aggregate verdicts into ratios + tag agreement (see design doc v2 §5)."""
        analysis = CaseAnalysis(saturated=saturated)
        if not verdicts:
            return analysis
        analysis.cases_analyzed = len(verdicts)
        cause_counts: Counter[str] = Counter()
        tag_counts: Counter[str] = Counter()
        agreed = 0
        near_miss = 0
        for verdict in verdicts:
            cause_counts[verdict.root_cause_type.value] += 1
            tag_counts[verdict.capability_tag] += 1
            if self._taxonomy.is_within(hypothesis, verdict.capability_tag):
                agreed += 1
            elif self._taxonomy.near_miss(hypothesis, verdict.capability_tag):
                near_miss += 1
        total = len(verdicts)
        analysis.per_cause_counts = dict(cause_counts)
        analysis.per_tag_counts = dict(tag_counts)
        analysis.content_error_ratio = cause_counts[RootCauseType.CONTENT_ERROR.value] / total
        analysis.format_error_ratio = cause_counts[RootCauseType.FORMAT_ERROR.value] / total
        analysis.grading_artifact_ratio = (
            cause_counts[RootCauseType.GRADING_ARTIFACT.value] / total
        )
        analysis.label_noise_ratio = cause_counts[RootCauseType.LABEL_NOISE.value] / total
        analysis.ambiguous_ratio = cause_counts[RootCauseType.AMBIGUOUS.value] / total
        analysis.tag_agreement = agreed / total
        analysis.near_miss_ratio = near_miss / total
        return analysis


def analyze_cases(
    llm: Any,
    taxonomy: CapabilityTaxonomy,
    hypothesis: str,
    cases: list[dict],
    *,
    sample_size: int = 50,
    saturation_window: int = 20,
    batch_size: int = _BATCH_SIZE,
) -> CaseAnalysis:
    """Module-level entry point (keeps call sites symmetric with other stages)."""
    analyzer = GuidedCaseAnalyzer(llm, taxonomy, batch_size=batch_size)
    return analyzer.analyze(
        hypothesis, cases, sample_size=sample_size, saturation_window=saturation_window
    )
