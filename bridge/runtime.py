"""Serialize access to the process-global legacy backend."""
from functools import wraps
import math
import os
from threading import RLock

legacy_runtime_lock = RLock()


class LegacyRuntimeBusy(TimeoutError):
    pass


def _wait_seconds():
    seconds = float(os.getenv('AGENT_RUNTIME_LOCK_TIMEOUT_SECONDS', '30'))
    if not math.isfinite(seconds) or not 0 < seconds <= 300:
        raise ValueError('AGENT_RUNTIME_LOCK_TIMEOUT_SECONDS must be in (0, 300]')
    return seconds


def isolated_legacy_request(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not legacy_runtime_lock.acquire(timeout=_wait_seconds()):
            raise LegacyRuntimeBusy('Legacy backend busy; lock wait timed out')
        try:
            return fn(*args, **kwargs)
        finally:
            legacy_runtime_lock.release()
    return wrapper
