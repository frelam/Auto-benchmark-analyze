"""Stage 2: single-capability probe verification (design doc v2 section 4).

Each candidate capability from Stage 1 is re-tested on a narrow probe
benchmark (CheckList-style Minimum Functionality Test):

* no probe registered                       -> ``NO_PROBE`` (build-probe todo)
* probe registered, score available         -> peer-percentile judgment:
  below threshold => ``CONFIRMED``, else ``NOT_CONFIRMED`` (source benchmark
  still low -> compositional-deficit candidate for Stage 4)
* probe registered, not evaluated this run  -> ``PENDING_EVAL`` (eval todo;
  confidence is capped exactly like ``NO_PROBE``)

Ancestor fallback: a capability without its own probe inherits its nearest
ancestor's probes (marked ``via_ancestor``), which shrinks the NO_PROBE gap
without pretending the probe is capability-exact.

When multiple samples per item exist, ``pass@1`` vs ``pass@k`` (unbiased
Chen et al. estimator) distinguishes "capability missing" from "capability
present but triggered unstably" — the rejection-sampling signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import percentileofscore

from benchmark_diagnosis.intelligent_diagnosis.capability_taxonomy import (
    CapabilityTaxonomy,
)
from benchmark_diagnosis.intelligent_diagnosis.types import (
    CandidateCapability,
    PassKStats,
    ProbeResult,
    ProbeState,
)

_PACKAGE_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_PROBE_REGISTRY_PATH = _PACKAGE_DATA / "probe_registry.yaml"


@dataclass
class ProbeEntry:
    benchmark_id: str
    note: str = ""
    via_ancestor: str | None = None  # ancestor capability whose probe this is


@dataclass
class ProbeRegistry:
    """capability_id -> probe entries, with ancestor fallback resolution."""

    version: str
    _direct: dict[str, list[ProbeEntry]]

    @classmethod
    def load(
        cls, path: str | Path | None = None, taxonomy: CapabilityTaxonomy | None = None
    ) -> ProbeRegistry:
        p = Path(path) if path is not None else DEFAULT_PROBE_REGISTRY_PATH
        with open(p, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        direct: dict[str, list[ProbeEntry]] = {}
        for cap, entries in (raw.get("probes") or {}).items():
            direct[str(cap)] = [
                ProbeEntry(
                    benchmark_id=str(e["benchmark_id"]),
                    note=str(e.get("note", "")),
                )
                for e in entries
            ]
        return cls(version=str(raw.get("version", "1")), _direct=direct, taxonomy=taxonomy)

    def __init__(
        self,
        version: str,
        _direct: dict[str, list[ProbeEntry]],
        taxonomy: CapabilityTaxonomy | None = None,
    ) -> None:
        self.version = version
        self._direct = _direct
        self._taxonomy = taxonomy

    def probes_for(self, capability_id: str) -> list[ProbeEntry]:
        """Direct probes, falling back to the nearest ancestor's probes."""
        direct = list(self._direct.get(capability_id, []))
        if direct:
            return direct
        if self._taxonomy is not None:
            for anc in self._taxonomy.ancestors(capability_id):
                inherited = self._direct.get(anc, [])
                if inherited:
                    return [
                        ProbeEntry(
                            benchmark_id=e.benchmark_id,
                            note=e.note,
                            via_ancestor=anc,
                        )
                        for e in inherited
                    ]
        return []

    def registered_ids(self) -> list[str]:
        return list(self._direct)


def estimate_pass_at_k(n_correct: int, n_total: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., Codex; k <= n_total)."""
    if n_total <= 0 or k <= 0:
        return 0.0
    k = min(k, n_total)
    n_c = min(n_correct, n_total)
    if n_c == 0:
        return 0.0
    if n_c == n_total:
        return 1.0
    return 1.0 - comb(n_total - n_c, k) / comb(n_total, k)


@dataclass
class ProbeConfig:
    """Stage 2 thresholds (defaults mirror config.diagnosis)."""

    percentile_threshold: float = 25.0
    min_peers: int = 5
    min_passk_samples: int = 8
    passk_gap_threshold: float = 0.5  # (pass_k - pass_1)/pass_1 must exceed this
    pass1_high_threshold: float = 0.5  # pass@1 must be below this for "unstable trigger"


def _passk_stats_for(
    capability_id: str,
    passk_stats: dict[str, PassKStats] | None,
    probe_benchmark_id: str | None,
) -> PassKStats | None:
    for stats in (passk_stats or {}).values():
        if stats.capability_id == capability_id:
            return stats
        if probe_benchmark_id and stats.probe_benchmark_id == probe_benchmark_id:
            return stats
    return None


def verify_candidates(
    candidates: list[CandidateCapability],
    *,
    registry: ProbeRegistry,
    probe_scores: dict[str, float] | None = None,
    peer_scores: dict[str, list[float]] | None = None,
    passk_stats: dict[str, PassKStats] | None = None,
    config: ProbeConfig | None = None,
) -> list[ProbeResult]:
    """Stage 2: verify every candidate capability against its probe(s).

    Args:
        candidates: Stage 1 output.
        registry: Probe registry with ancestor fallback.
        probe_scores: ``benchmark_id -> score`` for the target model on probe
            benchmarks evaluated this run (may be empty).
        peer_scores: ``benchmark_id -> [historical peer scores]`` used for the
            below-expectation percentile judgment.
        passk_stats: Optional ``capability_id -> PassKStats`` (or matched by
            ``probe_benchmark_id``).
        config: Thresholds.

    Returns:
        One :class:`ProbeResult` per candidate.
    """
    config = config or ProbeConfig()
    probe_scores = probe_scores or {}
    peer_scores = peer_scores or {}

    results: list[ProbeResult] = []
    for cand in candidates:
        probes = registry.probes_for(cand.capability_id)
        if not probes:
            results.append(
                ProbeResult(
                    capability_id=cand.capability_id,
                    state=ProbeState.NO_PROBE,
                    note="no probe registered for this capability (or any ancestor)",
                )
            )
            continue

        state: ProbeState | None = None
        probe_id: str | None = None
        percentile: float | None = None
        z_score: float | None = None
        note_parts: list[str] = []

        for entry in probes:
            if entry.benchmark_id not in probe_scores:
                continue
            score = float(probe_scores[entry.benchmark_id])
            peers = peer_scores.get(entry.benchmark_id) or []
            if len(peers) < config.min_peers:
                note_parts.append(
                    f"{entry.benchmark_id}: score available but only "
                    f"{len(peers)} peers (< {config.min_peers}) — cannot judge"
                )
                continue
            percentile = float(percentileofscore(peers, score, kind="rank"))
            arr = np.asarray(peers, dtype=float)
            std = float(arr.std())
            z_score = float((score - arr.mean()) / std) if std > 1e-12 else 0.0
            probe_id = entry.benchmark_id
            below = percentile < config.percentile_threshold
            state = ProbeState.CONFIRMED if below else ProbeState.NOT_CONFIRMED
            note_parts.append(
                f"{entry.benchmark_id}: percentile={percentile:.1f} "
                f"(threshold {config.percentile_threshold})"
                + ("" if entry.via_ancestor is None else f" [via {entry.via_ancestor}]")
            )
            if state == ProbeState.CONFIRMED:
                break  # one confirming probe is enough

        if state is None:
            # Probes exist but none could be judged this run.
            state = ProbeState.PENDING_EVAL
            note_parts.append(
                "probe(s) registered but not evaluated this run: "
                + ", ".join(e.benchmark_id for e in probes)
            )

        pass_1 = pass_k = gap_ratio = None
        stats = _passk_stats_for(cand.capability_id, passk_stats, probe_id)
        if stats is not None:
            pass_1, pass_k = stats.pass_1, stats.pass_k
            if pass_1 is not None and pass_k is not None and pass_1 > 0:
                gap_ratio = (pass_k - pass_1) / pass_1
            if stats.samples < config.min_passk_samples:
                note_parts.append(
                    f"pass@k has only {stats.samples} samples (< "
                    f"{config.min_passk_samples}) — treat gap cautiously"
                )

        results.append(
            ProbeResult(
                capability_id=cand.capability_id,
                state=state,
                probe_benchmark_id=probe_id,
                percentile=percentile,
                z_score=z_score,
                pass_1=pass_1,
                pass_k=pass_k,
                passk_gap_ratio=gap_ratio,
                note="; ".join(note_parts),
            )
        )
    return results


def split_todos(results: list[ProbeResult]) -> tuple[list[str], list[str]]:
    """Split results into (build_probe_list, eval_pending_list)."""
    build: list[str] = []
    pending: list[str] = []
    for r in results:
        if r.state == ProbeState.NO_PROBE:
            build.append(r.capability_id)
        elif r.state == ProbeState.PENDING_EVAL:
            pending.append(r.capability_id)
    return build, pending
