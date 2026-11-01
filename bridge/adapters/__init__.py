"""Domain-pack adapters.

DietNerd is the live pack. WirelessChat implements the same contract but is
not selected, so today's CloudNerd / DietNerd routing stays unchanged.
"""

from __future__ import annotations

import os

from bridge.adapters.base import DomainAdapter
from bridge.adapters.cloudnerd import CloudNerdAdapter
from bridge.adapters.dietnerd import DietNerdAdapter
from bridge.adapters.wirelesschat import WirelessChatAdapter

__all__ = [
    "CloudNerdAdapter",
    "DietNerdAdapter",
    "DomainAdapter",
    "WirelessChatAdapter",
    "get_active_adapter",
]


def get_active_adapter() -> DomainAdapter:
    """Return the pack that should serve this process.

    Selection matches the historical `is_dietnerd()` branch: PubMed legacy for
    DietNerd, Stack Overflow cascade otherwise. WirelessChat is registered on
    the contract only; selecting it raises instead of falling back to CloudNerd.
    """
    from bridge.domain import is_dietnerd

    domain = (os.getenv("AGENT_DOMAIN") or os.getenv("NERD_DOMAIN") or "").strip()
    if domain == "WirelessChat":
        raise RuntimeError(
            "WirelessChat requires a compatible backend pack; live routing is not enabled"
        )

    if is_dietnerd():
        return DietNerdAdapter()
    return CloudNerdAdapter()
