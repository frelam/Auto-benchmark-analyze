"""Hierarchical, versioned capability taxonomy (design doc v2 section 2).

The taxonomy is the shared vocabulary across stages 1-6. It is deliberately
separate from the flat failure-mode taxonomy in
``recommendation/rule_base/taxonomy.yaml``: that one classifies *why* a case
failed, this one names *which capability* is in question, with a parent/child
hierarchy so Stage 3 agreement can tolerate fine-grained relabeling
(``is_within``) and Stage 1 can roll sub-accuracy up to ancestor capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from benchmark_diagnosis.intelligent_diagnosis.types import CapabilityNode

_PACKAGE_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_TAXONOMY_PATH = _PACKAGE_DATA / "capability_taxonomy.yaml"


class TaxonomyValidationError(ValueError):
    """Raised when a taxonomy YAML violates the structural contract."""


@dataclass
class CapabilityTaxonomy:
    """Loaded + validated capability taxonomy with hierarchy helpers."""

    version: str
    nodes: dict[str, CapabilityNode]

    @classmethod
    def load(cls, path: str | Path | None = None) -> CapabilityTaxonomy:
        """Load and validate a taxonomy YAML (default: packaged seed)."""
        p = Path(path) if path is not None else DEFAULT_TAXONOMY_PATH
        with open(p, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CapabilityTaxonomy:
        version = str(raw.get("version", "1"))
        entries = raw.get("capabilities") or []
        nodes: dict[str, CapabilityNode] = {}
        for entry in entries:
            cid = str(entry["id"])
            if cid in nodes:
                raise TaxonomyValidationError(f"duplicate capability id {cid!r}")
            parent = entry.get("parent")
            nodes[cid] = CapabilityNode(
                id=cid,
                name=str(entry.get("name", cid)),
                parent=str(parent) if parent else None,
                description=str(entry.get("description", "")),
                aliases=[str(a) for a in (entry.get("aliases") or [])],
            )
        cls._validate(nodes, version)
        for node in nodes.values():
            node.level = cls._depth(node.id, nodes)
        return cls(version=version, nodes=nodes)

    @staticmethod
    def _validate(nodes: dict[str, CapabilityNode], version: str) -> None:
        if not nodes:
            raise TaxonomyValidationError("taxonomy has no capabilities")
        for cid, node in nodes.items():
            if node.parent is not None and node.parent not in nodes:
                raise TaxonomyValidationError(
                    f"capability {cid!r} references unknown parent {node.parent!r}"
                )
        # Cycle check: walk parents from every node.
        for cid in nodes:
            seen: set[str] = set()
            cur: str | None = cid
            while cur is not None:
                if cur in seen:
                    raise TaxonomyValidationError(f"parent cycle detected at {cur!r}")
                seen.add(cur)
                cur = nodes[cur].parent
        if not version.strip():
            raise TaxonomyValidationError("taxonomy version must be non-empty")

    @classmethod
    def _depth(cls, cid: str, nodes: dict[str, CapabilityNode]) -> int:
        depth = 1
        cur = nodes[cid].parent
        while cur is not None:
            depth += 1
            cur = nodes[cur].parent
        return depth

    # ------------------------------------------------------------------ lookup

    def get(self, capability_id: str) -> CapabilityNode | None:
        return self.nodes.get(capability_id)

    def resolve(self, tag: str) -> str | None:
        """Map a possibly-alias / flat tag to a canonical capability id.

        Coarse benchmark tags (e.g. ``math``, ``factuality``,
        ``agentic_tool_use``) resolve through the node ``aliases`` lists into
        the hierarchy; exact ids pass through unchanged. Returns None when the
        tag matches nothing.
        """
        if tag in self.nodes:
            return tag
        lowered = tag.strip().lower()
        for cid, node in self.nodes.items():
            if any(str(a).strip().lower() == lowered for a in node.aliases):
                return cid
        return None

    @property
    def ids(self) -> list[str]:
        return list(self.nodes)

    def ancestors(self, capability_id: str) -> list[str]:
        """Parent chain, closest first (excludes the node itself)."""
        out: list[str] = []
        cur = self.nodes[capability_id].parent
        while cur is not None:
            out.append(cur)
            cur = self.nodes[cur].parent
        return out

    def descendants(self, capability_id: str) -> list[str]:
        """All strictly-deeper nodes reachable from ``capability_id``."""
        out: list[str] = []
        for cid in self.nodes:
            if cid != capability_id and capability_id in self.ancestors(cid):
                out.append(cid)
        return out

    def is_within(self, hypothesis: str, tag: str) -> bool:
        """True when ``tag`` is ``hypothesis`` or one of its descendants.

        Used by Stage 3: an LLM relabel to a *child* of the hypothesis still
        corroborates it; a relabel to a sibling does not.
        """
        return tag == hypothesis or tag in self.descendants(hypothesis)

    def near_miss(self, hypothesis: str, tag: str) -> bool:
        """True when ``tag`` is an ancestor, a descendant's sibling, or a sibling.

        A sibling shares the same parent (e.g. ``reasoning.math`` vs
        ``reasoning.logical``): the LLM says "close but a different capability",
        which neither corroborates nor refutes the hypothesis.
        """
        if tag == hypothesis or self.is_within(hypothesis, tag):
            return False
        if hypothesis in self.ancestors(tag) or tag in self.ancestors(hypothesis):
            return True
        return self.nodes[hypothesis].parent is not None and (
            self.nodes[hypothesis].parent == self.nodes[tag].parent
        )

    def rollup(self, tag: str) -> list[str]:
        """``tag`` plus all its ancestors (for sub-accuracy roll-up)."""
        return [tag] + self.ancestors(tag)


def load_taxonomy(path: str | Path | None = None) -> CapabilityTaxonomy:
    """Convenience loader (kept for symmetric imports across stages)."""
    return CapabilityTaxonomy.load(path)
