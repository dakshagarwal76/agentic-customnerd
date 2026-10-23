"""Offline two-worker telemetry replay; no provider calls."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from agent import llm_meter
try:
    from bridge.runtime import isolated_legacy_request
except ImportError:
    isolated_legacy_request = lambda fn: fn


def trial():
    entered, release, second_entered = (threading.Event() for _ in range(3))

    @isolated_legacy_request
    def first():
        llm_meter.reset()
        llm_meter.incr_chat('a', 2)
        entered.set()
        assert release.wait(2)
        return llm_meter.snapshot()['by_model']

    @isolated_legacy_request
    def second():
        second_entered.set()
        llm_meter.reset()
        llm_meter.incr_chat('b', 3)
        return llm_meter.snapshot()['by_model']

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first)
        assert entered.wait(2)
        b = pool.submit(second)
        second_entered.wait(.05)
        release.set()
        sa, sb = a.result(), b.result()
    return {'first': sa, 'second': sb, 'isolated': sa == {'a': 2} and sb == {'b': 3}}

if __name__ == '__main__':
    print(json.dumps([trial() for _ in range(20)]))
