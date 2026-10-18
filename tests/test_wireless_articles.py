import pytest

from bridge.adapters import WirelessChatAdapter, get_active_adapter


def article(ident="2401.12345v2", **changes):
    return dict(
        id=ident,
        url="https://arxiv.org/abs/" + ident,
        title="Wireless test",
        summary="Synthetic source text",
        published="2024-01-12T00:00:00Z",
        **changes,
    )


def retrieve(rows, cutoff=None):
    calls = []

    def collect(queries, **kwargs):
        calls.append((queries, kwargs))
        return rows

    result = WirelessChatAdapter().retrieve(
        body_text="MIMO latency",
        title=None,
        max_date=cutoff,
        collect=collect,
        on_plane=lambda plane: pytest.fail("cascade"),
        planes=[],
        tier_only=None,
    )
    return result, calls


def test_article_ids_use_arxiv_not_transcripts():
    adapter = WirelessChatAdapter()
    assert adapter.source_id(article()) == "2401.12345v2"
    assert adapter.source_id({"url": "https://arxiv.org/pdf/hep-th/9901001v1.pdf"}) == (
        "hep-th/9901001v1"
    )
    with pytest.raises(ValueError):
        adapter.source_id({"message_id": "m1", "PMID": "12"})


def test_duplicate_articles_and_progress():
    source = article()
    (rows, meta), calls = retrieve([source, dict(source)])
    assert rows == [source]
    assert calls == [(["MIMO latency"], {"max_date": None})]
    assert meta["corpus"] == "arXiv"
    assert meta["domain"] == "WirelessChat"
    assert WirelessChatAdapter().search_progress("MIMO", None) == (
        "Agent: searching arXiv (MIMO...) tier=legacy"
    )


def test_empty_retrieval():
    assert retrieve([])[0][0] == []


def test_matching_article_versions_keep_the_newest():
    old = article("2401.12345v1")
    new = article("2401.12345v3")
    (rows, _), _ = retrieve([old, new])
    assert rows == [new]

    (rows, _), _ = retrieve([new, old])
    assert rows == [new]


def test_cutoff_forwarded_and_enforced():
    (rows, _), calls = retrieve([article()], "2024-01-11")
    assert rows == []
    assert calls[0][1]["max_date"] == "2024-01-11"
    assert retrieve([article()], "2024-01-12")[0][0] == [article()]


@pytest.mark.parametrize(
    "row",
    [
        {"outputs": {}},
        {"message_id": "m1"},
        "answer",
        {
            "id": "2401.12345",
            "url": "https://example.invalid/a",
            "title": "x",
            "summary": "y",
        },
        {
            "id": "2401.12345",
            "url": "https://arxiv.org/abs/2401.99999",
            "title": "x",
            "summary": "y",
        },
        {"id": "2401.12345", "title": "x", "summary": ""},
    ],
)
def test_malformed_sources_rejected(row):
    with pytest.raises(ValueError):
        retrieve([row])


@pytest.mark.parametrize("published", [None, "bad", "2024-02-31"])
def test_unknown_publication_date_fails_closed(published):
    row = article()
    row["published"] = published
    with pytest.raises(ValueError):
        retrieve([row], "2024-01-12")


def test_wireless_not_silently_routed_to_cloud(monkeypatch):
    monkeypatch.setenv("AGENT_DOMAIN", "WirelessChat")
    with pytest.raises(RuntimeError, match="compatible"):
        get_active_adapter()
