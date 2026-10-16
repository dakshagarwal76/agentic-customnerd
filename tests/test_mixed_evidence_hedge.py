"""Regression: disagreement hedges survive every deterministic filter path.

QIDs 26891320 and 31631671 are the DietNerd eval questions whose answers said the
evidence was conflicting or inconclusive. Repo eval dumps are not checked in, so
the fixtures below follow the PubMed abstracts for those ids (animal-flesh iron
status; vegan diet and gut microbiota). A stored answer is used when a file under
the repo names the QID and carries an answer field.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import agent.faith_policy as fp
import agent.pipeline as pipeline
import agent.tools.claim_filter as cf
from agent.state import AgentState
from prompts.synthesis import SYNTHESIS_CONFLICT_RULE, install_synthesis_conflict_rule

HEDGE_MARKERS = (
    "evidence is mixed",
    "findings are conflicting",
    "results are mixed",
    "findings are inconclusive",
    "evidence is inconsistent",
    "evidence is heterogeneous",
    "not all studies",
)

EMPTY_CLOSERS = (
    "further research is needed",
    "more research is still needed",
    "informational purposes only",
    "qualified healthcare",
    "disclaimer",
)

# PMID 26891320: systematic review, animal flesh intake and iron status.
# "results were conflicting"; optimum intake unclear; further studies required.
QID_26891320_FIXTURE = (
    "Possibly, but the evidence is mixed: five high-quality reports linked animal-flesh "
    "intake of 85-300 g/day with better iron status [1], while other reports found no "
    "association [2]. "
    "Findings are conflicting. "
    "Results are mixed. "
    "The findings are inconclusive. "
    "The evidence is inconsistent. "
    "The evidence is heterogeneous. "
    "Not all studies agree on whether low animal-flesh diets worsen iron status. "
    "Further research is needed to confirm the relationship. "
    "Disclaimer: informational purposes only and not a substitute for professional medical advice."
)

# PMID 31631671: vegan or vegetarian diet and gut microbiota.
# No consistent association; some studies were contradictory.
QID_31631671_FIXTURE = (
    "Possibly, but the evidence is mixed: some reports described different Bacteroides "
    "and Prevotella profiles on a vegan diet [1], while others found no consistent link "
    "with microbiota composition [2]. "
    "Findings are conflicting. "
    "Results are mixed. "
    "The findings are inconclusive. "
    "The evidence is inconsistent. "
    "The evidence is heterogeneous. "
    "Not all studies agree that a vegan diet changes the gut microbiota. "
    "More research is still needed. "
    "Always consult a qualified healthcare provider."
)

FIXTURES = {
    "26891320": QID_26891320_FIXTURE,
    "31631671": QID_31631671_FIXTURE,
}


def _stored_answer(qid: str) -> str:
    """Return a checked-in eval answer for this QID, or '' when none exists."""
    root = Path(__file__).resolve().parents[1]
    skip = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}
    answer_keys = ("final_answer", "answer", "response", "output", "prediction")
    for path in root.rglob("*"):
        if not path.is_file() or any(part in skip for part in path.parts):
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".csv", ".md", ".txt"}:
            continue
        if qid not in path.name and "eval" not in path.as_posix().lower():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if qid not in text:
            continue
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            found = _answer_from_json(payload, qid, answer_keys)
            if found:
                return found
        if path.suffix.lower() == ".jsonl":
            for line in text.splitlines():
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                found = _answer_from_json(payload, qid, answer_keys)
                if found:
                    return found
    return ""


def _answer_from_json(payload, qid: str, answer_keys: tuple) -> str:
    rows = payload if isinstance(payload, list) else [payload]
    if isinstance(payload, dict):
        for key in ("results", "questions", "rows", "data"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
    for row in rows:
        if not isinstance(row, dict):
            continue
        ident = str(row.get("question_id") or row.get("qid") or row.get("id") or "")
        if ident != qid:
            continue
        for key in answer_keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def answer_for(qid: str) -> str:
    stored = _stored_answer(qid)
    return stored or FIXTURES[qid]


def _assert_hedge_survives(qid: str, raw: str, cleaned: str) -> None:
    low_raw = raw.lower()
    low = cleaned.lower()
    present = [marker for marker in HEDGE_MARKERS if marker in low_raw]
    assert present, f"{qid} input has no disagreement hedge"
    for marker in present:
        assert marker in low, f"{qid} dropped {marker!r}: {cleaned}"
    for closer in EMPTY_CLOSERS:
        if closer in low_raw:
            assert closer not in low, f"{qid} kept empty closer {closer!r}: {cleaned}"
    assert not cleaned.lower().startswith("yes.")
    assert not cleaned.lower().startswith("no.")


def test_conflict_phrases_are_not_empty_closers():
    for phrase in (
        "Findings are conflicting.",
        "Results are mixed.",
        "The findings are inconclusive.",
        "The evidence is inconsistent.",
        "The evidence is heterogeneous.",
        "Not all studies agree.",
    ):
        assert cf._EVIDENCE_CONFLICT_RE.search(phrase)
        assert not cf._NONCOMMITTAL_CLOSER_RE.search(phrase)
    assert cf._NONCOMMITTAL_CLOSER_RE.search("Further research is needed.")
    assert cf._NONCOMMITTAL_CLOSER_RE.search("The optimal quantity is undetermined.")


def test_empty_closers_still_drop():
    text = (
        "Iron status differed by diet. The optimal quantity is undetermined. "
        "Results can vary. Still undetermined. Further research is needed."
    )
    assert cf.strip_gap_talk_sentences(text) == "Iron status differed by diet."


def test_conflict_sentence_kept_with_closer_words():
    text = "Intake changed ferritin. Findings are conflicting and further research is needed."
    assert "Findings are conflicting" in cf.strip_gap_talk_sentences(text)


def _run_paths(monkeypatch, qid: str, raw: str) -> None:
    _assert_hedge_survives(qid, raw, cf.strip_gap_talk_sentences(raw))

    monkeypatch.setenv("AGENT_CLAIM_FILTER", "false")
    disabled, disabled_meta = cf.apply_claim_filter(_state(raw))
    assert disabled_meta["enabled"] is False
    _assert_hedge_survives(qid, raw, disabled)

    monkeypatch.setenv("AGENT_CLAIM_FILTER", "true")
    no_passages, _meta = cf.apply_claim_filter(_state(raw))
    _assert_hedge_survives(qid, raw, no_passages)

    seen = {}

    def _keep_all(claims, passages):
        seen["claims"] = list(claims)
        return list(claims)

    monkeypatch.setattr(cf, "_verify_claims_batch", _keep_all)
    verified, verified_meta = cf.apply_claim_filter(
        _state(raw, articles=[{"abstract": "Conflicting human evidence.", "title": "Study"}])
    )
    assert verified_meta["refuse"] is False
    assert seen["claims"], f"{qid} claim filter removed every sentence before verify"
    blob = " ".join(seen["claims"]).lower()
    assert any(marker in blob for marker in HEDGE_MARKERS if marker in raw.lower())
    _assert_hedge_survives(qid, raw, verified)

    shipped = _state(raw)
    monkeypatch.setattr(pipeline, "ensure_citations_on_refusal", lambda state: None)
    pipeline._ship_grounded_fallback(shipped, step="claim_filter_grounded_fallback")
    assert shipped.agent_trace[-1]["detail"]["shipped"] == "synthesized_partial"
    _assert_hedge_survives(qid, raw, shipped.final_answer)

    low_conf = _state(raw)
    low_conf.suppress_gap_last = True
    _assert_hedge_survives(qid, raw, fp.polish_low_confidence_answer(low_conf, raw))

    overlap = _state(raw)
    overlap.suppress_gap_last = False
    overlap.domain_overlap = 0.01
    _assert_hedge_survives(qid, raw, fp.polish_low_confidence_answer(overlap, raw))


def _state(answer: str, articles=None) -> AgentState:
    state = AgentState(session_id="s", user_query="q")
    state.final_answer = answer
    state.relevant_articles = articles or []
    return state


def test_qid_26891320_hedge_survives_every_filter(monkeypatch):
    raw = answer_for("26891320")
    assert "evidence is mixed" in raw.lower()
    _run_paths(monkeypatch, "26891320", raw)


def test_qid_31631671_hedge_survives_every_filter(monkeypatch):
    raw = answer_for("31631671")
    assert "evidence is mixed" in raw.lower()
    _run_paths(monkeypatch, "31631671", raw)


def test_synthesis_prompt_leads_with_mixed_evidence(monkeypatch):
    assert "When the cited studies disagree" in SYNTHESIS_CONFLICT_RULE
    assert 'Possibly, but the evidence is mixed' in SYNTHESIS_CONFLICT_RULE
    assert "Never give a flat Yes or No" in SYNTHESIS_CONFLICT_RULE
    assert "cite both sides" in SYNTHESIS_CONFLICT_RULE
    assert SYNTHESIS_CONFLICT_RULE in fp._FALLBACK_ANSWER_PROMPT
    for word in ("mixed", "conflicting", "inconsistent", "heterogeneous", "inconclusive", "not all studies"):
        assert word in cf.CLAIM_VERIFY_PROMPT

    prompts = types.ModuleType("openai_prompts")
    prompts.FINAL_RESPONSE_PROMPT = "Answer from the evidence."
    prompts.COMBINED_LITE_PROMPT = "Write a combined answer."
    prompts.FACTS_ONLY_PROMPT = "Use only extracted facts."
    prompts.EVIDENCE_EXTRACTION_PROMPT = "extract"
    helper = types.ModuleType("helper_functions")
    helper.FINAL_RESPONSE_PROMPT = "Answer from the evidence."
    monkeypatch.setitem(sys.modules, "openai_prompts", prompts)
    monkeypatch.setitem(sys.modules, "helper_functions", helper)

    install_synthesis_conflict_rule()
    install_synthesis_conflict_rule()

    for text in (
        prompts.FINAL_RESPONSE_PROMPT,
        prompts.COMBINED_LITE_PROMPT,
        prompts.FACTS_ONLY_PROMPT,
        helper.FINAL_RESPONSE_PROMPT,
    ):
        assert text.count("When the cited studies disagree") == 1
        assert "Possibly, but the evidence is mixed" in text
        assert "Never give a flat Yes or No" in text
    assert prompts.EVIDENCE_EXTRACTION_PROMPT == "extract"
