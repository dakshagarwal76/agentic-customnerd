import sys
import types

import pytest

from agent.state import AgentState
import agent.tools as tools
from bridge.adapters import (
    CloudNerdAdapter,
    DietNerdAdapter,
    WirelessChatAdapter,
    get_active_adapter,
)
from bridge.adapters.base import canonical_source_id


def test_dietnerd_source_id_prefers_medline_pmid():
    article = {"MedlineCitation": {"PMID": "36518821"}, "id": "so-1", "title": "t"}
    assert DietNerdAdapter().source_id(article) == "pmid:36518821"
    assert CloudNerdAdapter().source_id(article) == canonical_source_id(article)


def test_source_id_fallbacks_and_non_dict():
    assert DietNerdAdapter().source_id({"answer_id": 9}) == "answer_id:9"
    assert DietNerdAdapter().source_id("raw").startswith("raw") is False
    assert canonical_source_id({"link": "http://example"}) == "link:http://example"


def test_progress_strings_match_historical_wording():
    diet = DietNerdAdapter()
    cloud = CloudNerdAdapter()
    body = "folic acid supplementation and memory"
    assert diet.search_progress(body, None) == (
        f"Agent: searching PubMed ({body[:80]}...) tier=legacy"
    )
    assert cloud.search_progress(body, "broad") == (
        f"Agent: searching Stack Overflow ({body[:80]}...) tier=broad"
    )
    plane = {
        "plane": 2,
        "tier": "strict",
        "step": "tagged",
        "success": False,
        "extension_triggered": True,
        "queries_count": 3,
        "articles_found": 0,
    }
    text = diet.plane_progress(plane)
    assert text == "Cascade plane 2: strict/tagged [EXTEND] queries=3 articles=0"
    win = dict(plane, success=True, extension_triggered=False)
    assert " [WIN] " in cloud.plane_progress(win)
    quiet = dict(plane, success=False, extension_triggered=False)
    assert " [try] " in cloud.plane_progress(quiet)


def test_active_adapter_follows_diet_flag(monkeypatch):
    import bridge.domain as domain

    monkeypatch.setattr(domain, "is_dietnerd", lambda: True)
    assert isinstance(get_active_adapter(), DietNerdAdapter)
    monkeypatch.setattr(domain, "is_dietnerd", lambda: False)
    monkeypatch.setenv("AGENT_DOMAIN", "WirelessChat")
    assert isinstance(get_active_adapter(), CloudNerdAdapter)


def test_wireless_source_id_and_retrieve_not_wired():
    adapter = WirelessChatAdapter()
    assert adapter.source_id({"message_id": "m1", "PMID": "9"}) == "message_id:m1"
    assert adapter.source_id({"doc_id": "d2"}) == "doc_id:d2"
    assert adapter.search_progress("wifi latency", None).startswith("Agent: searching Wireless")
    with pytest.raises(NotImplementedError):
        adapter.retrieve(
            body_text="q",
            title=None,
            max_date=None,
            collect=lambda *a, **k: [],
            on_plane=lambda plane: None,
            planes=[],
        )


def test_dietnerd_retrieve_keeps_legacy_and_gap_fill(monkeypatch):
    helper = types.ModuleType("helper_functions")
    helper.query_generation = lambda q: (None, None, [q])
    monkeypatch.setitem(sys.modules, "helper_functions", helper)

    def gap_fill_retrieval(articles, body, title, max_date, collect_fn, cascade_meta):
        del body, title, max_date, collect_fn, cascade_meta
        found = list(articles) + [{"PMID": "222"}]
        return found, {"gap_fill_used": True, "added_count": 1}

    gap = types.ModuleType("retrieval_gap_fill")
    gap.gap_fill_retrieval = gap_fill_retrieval
    monkeypatch.setitem(sys.modules, "retrieval_gap_fill", gap)

    seen = {}

    def collect(query_list, max_date=None, **kwargs):
        seen["queries"] = list(query_list)
        seen["max_date"] = max_date
        return [{"PMID": "111"}]

    articles, meta = DietNerdAdapter().retrieve(
        body_text="fiber and LDL",
        title="t",
        max_date="2020-01-01",
        collect=collect,
        on_plane=lambda plane: None,
        planes=[],
    )
    assert seen["queries"] == ["fiber and LDL"]
    assert seen["max_date"] == "2020-01-01"
    assert [a["PMID"] for a in articles] == ["111", "222"]
    assert meta["retrieval_mode"] == "legacy"
    assert meta["domain"] == "DietNerd"
    assert meta["corpus"] == "PubMed"
    assert meta["gap_fill"]["gap_fill_used"] is True


def test_dietnerd_retrieve_survives_missing_gap_fill(monkeypatch):
    helper = types.ModuleType("helper_functions")
    helper.query_generation = lambda q: (None, None, [q])
    monkeypatch.setitem(sys.modules, "helper_functions", helper)
    monkeypatch.setitem(sys.modules, "retrieval_gap_fill", None)
    # ImportError / ModuleNotFoundError when the module cannot be loaded.
    sys.modules.pop("retrieval_gap_fill", None)

    articles, meta = DietNerdAdapter().retrieve(
        body_text="q",
        title=None,
        max_date=None,
        collect=lambda query_list, max_date=None, **kwargs: [{"PMID": "1"}],
        on_plane=lambda plane: None,
        planes=["plane"],
    )
    assert articles == [{"PMID": "1"}]
    assert meta["extension_planes"] == ["plane"]
    assert "gap_fill" not in meta


def test_cloud_cascade_and_legacy(monkeypatch):
    calls = {"cascade": 0, "legacy": 0, "planes": []}

    def cascade_collect_articles(*args, **kwargs):
        calls["cascade"] += 1
        kwargs["on_plane"]({"plane": 1, "tier": "strict", "step": "s", "success": True, "queries_count": 1, "articles_found": 2})
        return [{"answer_id": 5}], {"retrieval_mode": "cascade", "extension_planes": []}

    cascade = types.ModuleType("retrieval_cascade")
    cascade.cascade_collect_articles = cascade_collect_articles
    cascade.is_cascade_mode = lambda: True
    monkeypatch.setitem(sys.modules, "retrieval_cascade", cascade)
    monkeypatch.setitem(sys.modules, "retrieval_gap_fill", None)
    sys.modules.pop("retrieval_gap_fill", None)

    articles, meta = CloudNerdAdapter().retrieve(
        body_text="s3 timeout",
        title="t",
        max_date=None,
        collect=lambda *a, **k: [],
        on_plane=lambda plane: calls["planes"].append(plane),
        planes=[],
        tier_only="strict",
    )
    assert calls["cascade"] == 1
    assert articles == [{"answer_id": 5}]
    assert meta["retrieval_mode"] == "cascade"
    assert calls["planes"][0]["success"] is True

    cascade.is_cascade_mode = lambda: False
    helper = types.ModuleType("helper_functions")

    def query_generation(q):
        calls["legacy"] += 1
        return None, None, [q]

    helper.query_generation = query_generation
    monkeypatch.setitem(sys.modules, "helper_functions", helper)
    articles, meta = CloudNerdAdapter().retrieve(
        body_text="s3 timeout",
        title=None,
        max_date=None,
        collect=lambda query_list, max_date=None, **kwargs: [{"question_id": 7}],
        on_plane=lambda plane: None,
        planes=[],
    )
    assert calls["legacy"] == 1
    assert meta["domain"] == "CloudNerd"
    assert meta["corpus"] == "Stack Overflow"
    assert meta["retrieval_mode"] == "legacy"
    assert articles[0]["question_id"] == 7


def test_tool_search_emits_dietnerd_progress(monkeypatch):
    monkeypatch.setenv("AGENT_DETERMINISTIC_LADDER", "false")
    monkeypatch.setattr(tools, "ensure_legacy_backend", lambda: None)
    monkeypatch.setattr(tools, "cascade_body_for_tier", lambda state, query, tier: (query, "title"))
    import bridge.domain as domain

    monkeypatch.setattr(domain, "is_dietnerd", lambda: True)
    helper = types.ModuleType("helper_functions")
    helper.query_generation = lambda q: (None, None, [q])
    monkeypatch.setitem(sys.modules, "helper_functions", helper)
    sys.modules.pop("retrieval_gap_fill", None)
    monkeypatch.setattr(tools, "collect_with_fallback", lambda *a, **k: [{"PMID": "42", "abstract": "omega"}])
    messages = []
    state = AgentState("s", "omega question")
    result = tools.tool_search_stackoverflow(state, on_progress=messages.append)
    assert result["corpus"] == "PubMed"
    assert result["tier_hint"] == "legacy"
    assert messages[0].startswith("Agent: searching PubMed (")
    assert messages[0].endswith("tier=legacy")
    assert state.agent_trace[-1]["detail"]["domain"] == "DietNerd"
    assert state.agent_trace[-1]["detail"]["corpus"] == "PubMed"
