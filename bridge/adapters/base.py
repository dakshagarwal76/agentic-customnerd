"""Adapter contract shared by domain packs."""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol


def canonical_source_id(article: Any) -> str:
    """Historical article key: PMID first, then SO / URL fields.

    DietNerd and CloudNerd both keep this order so source IDs do not move.
    """
    if not isinstance(article, dict):
        return str(id(article))
    try:
        pmid = article.get("MedlineCitation", {}).get("PMID")
        if pmid:
            return f"pmid:{pmid}"
    except Exception:
        pass
    for key in ("PMID", "pmid", "id", "answer_id", "question_id", "link", "url", "title"):
        value = article.get(key)
        if value:
            return f"{key}:{value}"
    return str(id(article))


class DomainAdapter(Protocol):
    """What a domain pack must preserve: retrieval, source IDs, progress text."""

    name: str
    corpus_label: str
    default_tier: str

    def search_progress(self, body_text: str, tier_hint: Optional[str]) -> str:
        """SSE / trace line emitted before retrieval."""

    def plane_progress(self, plane: dict) -> str:
        """SSE line for one cascade plane. DietNerd does not emit planes."""

    def source_id(self, article: Any) -> str:
        """Stable id used to dedupe hits and to cite sources."""

    def retrieve(
        self,
        *,
        body_text: str,
        title: Optional[str],
        max_date: Optional[str],
        collect: Callable,
        on_plane: Callable[[dict], None],
        planes: list,
        tier_only: Optional[str] = None,
    ) -> tuple[list, dict]:
        """Run this pack's retrieval. Return articles and cascade metadata."""
