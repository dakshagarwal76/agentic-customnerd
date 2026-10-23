"""CloudNerd pack: Stack Overflow cascade, same source-id order as before."""

from __future__ import annotations

from typing import Any, Callable, Optional

from bridge.adapters.base import canonical_source_id


class CloudNerdAdapter:
    name = "CloudNerd"
    corpus_label = "Stack Overflow"
    default_tier = "strict"

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
        articles: list = []
        cascade_meta: dict = {}
        from retrieval_cascade import cascade_collect_articles, is_cascade_mode

        use_cascade = is_cascade_mode()
        if use_cascade:
            articles, cascade_meta = cascade_collect_articles(
                body_text,
                title,
                max_date,
                collect,
                on_plane=on_plane,
                tier_only=tier_only,
            )
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

        if not use_cascade:
            from helper_functions import query_generation

            _, _, query_list = query_generation(body_text)
            articles = collect(query_list, max_date=max_date) or []
            cascade_meta = {
                "retrieval_mode": "legacy",
                "extension_planes": planes,
                "domain": self.name,
                "corpus": self.corpus_label,
            }
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
