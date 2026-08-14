"""External retrieval of papers/reports (design doc section 6.2).

Retrieval is a *supplementary* source used only when the internal rule base does
not cover a diagnosed tag — never the primary judgement. This milestone ships a
:class:`NullRetriever`; the protocol documents the extension point for wiring a
real RAG backend later.
"""

from __future__ import annotations

from typing import Any, Protocol

from benchmark_diagnosis.config import RecommendationConfig


class Retriever(Protocol):
    """Protocol for an external-paper/report retriever keyed on the taxonomy.

    A concrete implementation should index recent papers / technical report
    abstracts tagged with the same taxonomy ids used by the rule base, so that
    retrieval happens in a vocabulary the rest of the pipeline understands.
    """

    def retrieve(self, query: str, max_results: int = 5) -> list[dict]:
        """Return up to ``max_results`` snippets matching ``query``.

        Each snippet is a dict such as ``{"title": str, "snippet": str,
        "source": "external:<url>", "tags": [str, ...]}``. Implementations that
        have no matches return an empty list.
        """
        ...


class NullRetriever:
    """No-op retriever: never returns external snippets."""

    def retrieve(self, query: str, max_results: int = 5) -> list[dict]:
        """Always return an empty list (external RAG is disabled)."""
        return []


def get_retriever(config: RecommendationConfig, llm: Any = None) -> Retriever:
    """Build the retriever selected by ``config.recommendation``.

    Args:
        config: Recommendation configuration. ``retrieval_enabled`` and
            ``max_external_sources`` control whether a real backend is used.
        llm: Optional analyst LLM; reserved for embedding-based retrievers.

    Returns:
        A :class:`Retriever`. This milestone always returns a
        :class:`NullRetriever`; the external-RAG extension point is the
        ``Retriever`` protocol, so a future backend (e.g. a vector store over
        paper abstracts tagged with taxonomy ids) only needs to implement
        :meth:`Retriever.retrieve` and be selected here.
    """
    # Extension point: if config.retrieval_enabled, construct a real retriever
    # (e.g. over the taxonomy-tagged paper corpus) using config and llm.
    return NullRetriever()
