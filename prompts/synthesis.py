"""Synthesis instructions shared by the agent fallback and the legacy final-response prompt.

The linear backend owns ``FINAL_RESPONSE_PROMPT`` (and combined-lite / facts-only
variants) under customnerd-backend. This module does not edit that tree. At
synthesize time the same rule is appended in memory so a conflicting evidence
set cannot collapse into a flat Yes or No.
"""

from __future__ import annotations

SYNTHESIS_CONFLICT_RULE = """When the cited studies disagree, say so in the first sentence and cite both sides.
Lead with the disagreement, for example: "Possibly, but the evidence is mixed: ...".
Never give a flat Yes or No when the evidence conflicts.
Keep mixed, conflicting, inconsistent, heterogeneous, inconclusive, and "not all studies" statements in the answer, and cite the supporting side and the opposing side."""

_CONFLICT_MARKER = "When the cited studies disagree"

_SYNTHESIS_PROMPT_NAMES = {
    "FINAL_RESPONSE_PROMPT",
    "COMBINED_LITE_PROMPT",
    "COMBINED_RESPONSE_PROMPT",
    "COMBINED_LITE_RESPONSE_PROMPT",
    "FACTS_ONLY_PROMPT",
    "FACTS_ONLY_RESPONSE_PROMPT",
    "EXTRACTED_FACTS_RESPONSE_PROMPT",
    "TWO_PASS_FINAL_PROMPT",
}


def _is_synthesis_prompt_name(name: str) -> bool:
    upper = name.upper()
    if upper in _SYNTHESIS_PROMPT_NAMES:
        return True
    if not upper.endswith("PROMPT"):
        return False
    return any(token in upper for token in ("FINAL_RESPONSE", "COMBINED", "FACTS_ONLY", "SYNTHESIS"))


def install_synthesis_conflict_rule() -> str:
    """Append the mixed-evidence lead rule onto already-imported synthesis prompts.

    ``helper_functions`` binds ``FINAL_RESPONSE_PROMPT`` with ``import *``, so the
    prompts module and the helper module both need the suffix. A second call
    does not duplicate the rule.
    """
    import sys

    rule = SYNTHESIS_CONFLICT_RULE.strip()
    for mod_name in ("openai_prompts", "helper_functions", "generic_prompts"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for attr, val in list(vars(mod).items()):
            if not isinstance(val, str) or not _is_synthesis_prompt_name(attr):
                continue
            if _CONFLICT_MARKER in val:
                continue
            setattr(mod, attr, val.rstrip() + "\n\n" + rule)
    return rule
