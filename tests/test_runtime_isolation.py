import ast
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

from agent import llm_meter


def test_parallel_workers_keep_separate_meter_totals():
    from bridge.runtime import isolated_legacy_request
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()

    @isolated_legacy_request
    def first():
        llm_meter.reset()
        llm_meter.incr_chat('first', 2)
        entered.set()
        assert release.wait(2)
        return llm_meter.snapshot()

    @isolated_legacy_request
    def second():
        second_entered.set()
        llm_meter.reset()
        llm_meter.incr_chat('second', 3)
        return llm_meter.snapshot()

    with ThreadPoolExecutor(max_workers=2) as executor:
        a = executor.submit(first)
        assert entered.wait(2)
        b = executor.submit(second)
        try:
            assert not second_entered.wait(.05)
        finally:
            release.set()
        sa, sb = a.result(), b.result()
    assert sa['by_model'] == {'first': 2}
    assert sb['by_model'] == {'second': 3}


def test_guard_releases_after_exception():
    from bridge.runtime import isolated_legacy_request
    @isolated_legacy_request
    def fail():
        raise ValueError('synthetic')
    try:
        fail()
    except ValueError:
        pass
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(isolated_legacy_request(lambda: 'ok')).result(timeout=2) == 'ok'


def test_core_entry_points_and_worker_share_guard():
    root = Path(__file__).parents[1]
    for path, name in [('agent/orchestrator.py', 'run_agent'),
                       ('bridge/domain.py', 'activate_domain'),
                       ('main.py', '_run_agent_logic_locked')]:
        tree = ast.parse((root / path).read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert 'isolated_legacy_request' in [ast.unparse(n) for n in fn.decorator_list]
    tree = ast.parse((root / 'main.py').read_text())
    switch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'agent_load_domain')
    assert isinstance(switch, ast.FunctionDef)
    assert any(isinstance(n, ast.ExceptHandler) and ast.unparse(n.type) == 'LegacyRuntimeBusy' for n in ast.walk(switch))


def test_wait_timeout_does_not_enter_or_reset_state(monkeypatch):
    import pytest
    from bridge.runtime import isolated_legacy_request, legacy_runtime_lock, LegacyRuntimeBusy
    monkeypatch.setenv('AGENT_RUNTIME_LOCK_TIMEOUT_SECONDS', '0.02')
    calls = []
    @isolated_legacy_request
    def blocked():
        calls.append('entered')
    with legacy_runtime_lock:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(blocked)
            with pytest.raises(LegacyRuntimeBusy):
                future.result(timeout=1)
    assert calls == []
    blocked()
    assert calls == ['entered']


def test_guard_is_reentrant(monkeypatch):
    from bridge.runtime import isolated_legacy_request
    monkeypatch.setenv('AGENT_RUNTIME_LOCK_TIMEOUT_SECONDS', '0.02')
    inner = isolated_legacy_request(lambda: 'ok')
    outer = isolated_legacy_request(inner)
    assert outer() == 'ok'


def test_invalid_timeout_fails_before_entry(monkeypatch):
    import pytest
    from bridge.runtime import isolated_legacy_request
    calls = []
    fn = isolated_legacy_request(lambda: calls.append('entered'))
    for value in ['0', '-1', 'nan', 'inf', '301', 'bad']:
        monkeypatch.setenv('AGENT_RUNTIME_LOCK_TIMEOUT_SECONDS', value)
        with pytest.raises(ValueError):
            fn()
    assert calls == []
