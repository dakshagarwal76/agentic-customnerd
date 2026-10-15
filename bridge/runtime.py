"""Serialize requests while the legacy backend uses process-global state."""
from functools import wraps
from threading import RLock

legacy_runtime_lock = RLock()


def isolated_legacy_request(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with legacy_runtime_lock:
            return fn(*args, **kwargs)
    return wrapper
