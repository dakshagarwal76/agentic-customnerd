"""DietNerd pack: PubMed Entrez, legacy retrieval, PMID source IDs."""

from __future__ import annotations

from typing import Any, Callable, Optional

from bridge.adapters.base import canonical_source_id


class DietNerdAdapter:
    name = "DietNerd"
    corpus_label = "PubMed"
    default_tier = "legacy"

    def search_progress(self, body_text: str, tier_hint: Optional[str]) -> str:
        tier = tier_hint or self.default_tier
        return f"Agent: searching {self.corpus_label} ({body_text[:80]}...) tier={tier}"

    def plane_progress(self, plane: dict) -> str:
        flag = "WIN" if plane.get("success") else (
            "EXTEND" if plane.get("extension_triggered") else "try"
        )
        return (
            f"Cascade plane {plane.get('plane')}: {plane.get('tier')}/{plane.get('step')} "
            f"[{flag}] queries={plane.get('queries_count')} articles={plane.get('articles_found')}"
        )

    def source_id(self, article: Any) -> str:
        return canonical_source_id(article)

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
        del on_plane, tier_only
        from helper_functions import query_generation

        _, _, query_list = query_generation(body_text)
        articles = collect(query_list, max_date=max_date) or []
        cascade_meta = {
            "retrieval_mode": "legacy",
            "extension_planes": planes,
            "domain": self.name,
            "corpus": self.corpus_label,
        }
        # Keep gap-fill on the PubMed path when the first pull is thin.
        try:
            from retrieval_gap_fill import gap_fill_retrieval

            articles, gap_meta = gap_fill_retrieval(
                articles,
                body_text,
                title,
                max_date,
                collect,
                cascade_meta,
            )
            if gap_meta.get("gap_fill_used"):
                cascade_meta["gap_fill"] = gap_meta
        except ImportError:
            pass
        return articles, cascade_meta
