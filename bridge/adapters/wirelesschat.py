"""WirelessChat pack stub.

The contract is the same as DietNerd. Retrieval is intentionally not wired:
selecting this class would change live behavior, so get_active_adapter never
returns it.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


class WirelessChatAdapter:
    name = "WirelessChat"
    corpus_label = "Wireless"
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
        """Wireless transcripts key off message_id, not PMID."""
        if not isinstance(article, dict):
            return str(id(article))
        for key in ("message_id", "doc_id", "id"):
            value = article.get(key)
            if value:
                return f"{key}:{value}"
        return str(id(article))

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
        del body_text, title, max_date, collect, on_plane, planes, tier_only
        raise NotImplementedError(
            "WirelessChat adapter is a contract placeholder and is not wired into retrieval."
        )
