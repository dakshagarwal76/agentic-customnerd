"""WirelessChat adapter for arXiv article dictionaries."""

from __future__ import annotations

from datetime import date
import re
from typing import Any, Callable, Optional
from urllib.parse import urlparse

_ARXIV_ID = re.compile(
    r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v[1-9]\d*)?\Z"
)
_ARXIV_VERSION = re.compile(r"v([1-9]\d*)\Z")
_ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org"}


def _article_version(ident: str) -> tuple[str, int]:
    match = _ARXIV_VERSION.search(ident)
    if not match:
        return ident, 0
    return ident[: match.start()], int(match.group(1))


def _article_id(article: Any) -> str:
    if not isinstance(article, dict):
        raise ValueError("WirelessChat requires arXiv source articles")

    ident = article.get("id")
    url = article.get("url")
    url_id = None
    if url:
        parsed = urlparse(str(url))
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in _ARXIV_HOSTS:
            raise ValueError("WirelessChat source URL is not arXiv")
        path = parsed.path
        if not path.startswith(("/abs/", "/pdf/")):
            raise ValueError("WirelessChat source URL has no article identifier")
        url_id = path[5:]
        if url_id.endswith(".pdf"):
            url_id = url_id[:-4]

    ident = ident or url_id
    if not isinstance(ident, str) or not _ARXIV_ID.fullmatch(ident):
        raise ValueError("WirelessChat source has no valid arXiv identifier")
    if url_id and url_id != ident:
        raise ValueError("WirelessChat source identifiers disagree")
    return ident


class WirelessChatAdapter:
    name = "WirelessChat"
    corpus_label = "arXiv"
    default_tier = "legacy"

    def search_progress(self, body_text: str, tier_hint: Optional[str]) -> str:
        tier = tier_hint or self.default_tier
        return f"Agent: searching arXiv ({body_text[:80]}...) tier={tier}"

    def plane_progress(self, plane: dict) -> str:
        raise ValueError("WirelessChat article retrieval does not use cascade planes")

    def source_id(self, article: Any) -> str:
        return _article_id(article)

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
        del title, on_plane, planes, tier_only
        cutoff = date.fromisoformat(max_date) if max_date else None
        found = collect([body_text], max_date=max_date)
        if not isinstance(found, list):
            raise ValueError("WirelessChat collector must return a list of source articles")

        articles, seen = [], {}
        for article in found:
            key = self.source_id(article)
            if not all(
                isinstance(article.get(field), str) and article[field].strip()
                for field in ("title", "summary")
            ):
                raise ValueError("WirelessChat source requires title and summary")
            if cutoff:
                published = article.get("published")
                if not isinstance(published, str):
                    raise ValueError("WirelessChat cutoff requires a publication date")
                if date.fromisoformat(published[:10]) > cutoff:
                    continue
            canonical, version = _article_version(key)
            previous = seen.get(canonical)
            if previous is None:
                seen[canonical] = (version, len(articles))
                articles.append(article)
            elif version > previous[0]:
                articles[previous[1]] = article
                seen[canonical] = (version, previous[1])

        return articles, {
            "retrieval_mode": "legacy",
            "extension_planes": [],
            "domain": self.name,
            "corpus": self.corpus_label,
        }
