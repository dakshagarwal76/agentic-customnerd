"""Exercise real public entry wrappers, with pipeline and filesystem work stubbed."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import pytest
from bridge.runtime import legacy_runtime_lock, LegacyRuntimeBusy


def test_run_agent_direct_call_times_out_before_pipeline(monkeypatch):
    from agent import orchestrator
    monkeypatch.setenv('AGENT_RUNTIME_LOCK_TIMEOUT_SECONDS', '0.02')
    calls = []
    monkeypatch.setattr(orchestrator, '_init_state_limits', lambda state: calls.append(state))
    with legacy_runtime_lock:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(LegacyRuntimeBusy):
                pool.submit(orchestrator.run_agent, SimpleNamespace()).result(timeout=1)
    assert calls == []


def test_activate_domain_direct_call_times_out_before_environment_change(monkeypatch):
    from bridge.domain import activate_domain
    monkeypatch.setenv('AGENT_RUNTIME_LOCK_TIMEOUT_SECONDS', '0.02')
    monkeypatch.setenv('AGENT_DOMAIN', 'CloudNerd')
    import os
    with legacy_runtime_lock:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(LegacyRuntimeBusy):
                pool.submit(activate_domain, 'WirelessNerd').result(timeout=1)
    assert os.environ['AGENT_DOMAIN'] == 'CloudNerd'


def test_direct_run_agent_and_domain_switch_share_lock(monkeypatch):
    from agent import orchestrator
    from bridge import domain
    monkeypatch.setenv('AGENT_RUNTIME_LOCK_TIMEOUT_SECONDS', '0.02')
    monkeypatch.setattr(orchestrator, 'AGENT_MODE', 'parity')
    entered, release = threading.Event(), threading.Event()
    def pipeline(state, **kwargs):
        entered.set()
        assert release.wait(1)
        return state
    monkeypatch.setattr(orchestrator, 'run_parity_pipeline', pipeline)
    state = SimpleNamespace()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(orchestrator.run_agent, state)
        assert entered.wait(1)
        try:
            with pytest.raises(LegacyRuntimeBusy):
                pool.submit(domain.activate_domain, 'WirelessNerd').result(timeout=1)
        finally:
            release.set()
        assert first.result(timeout=1) is state


def test_background_waiter_receives_terminal_busy_result():
    import ast
    from pathlib import Path
    from typing import Optional
    tree = ast.parse((Path(__file__).parents[1] / 'main.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_agent_logic')
    sent = []
    def busy(*args):
        raise LegacyRuntimeBusy('busy')
    namespace = dict(Optional=Optional, LegacyRuntimeBusy=LegacyRuntimeBusy,
                     _run_agent_logic_locked=busy,
                     send_update_sync=lambda session, result: sent.append((session, result)))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'main.py', 'exec'), namespace)
    namespace['_run_agent_logic']('q', 'session', None, None)
    assert sent == [('session', dict(end_output='busy', final_output='busy',
                                   citations_obj=[], agent_backend='cloudnerd-agent-backend'))]


def test_domain_http_busy_is_503(monkeypatch):
    import ast
    from pathlib import Path
    from fastapi import HTTPException, Form
    from bridge import domain
    tree = ast.parse((Path(__file__).parents[1] / 'main.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'agent_load_domain')
    fn.decorator_list=[]
    def busy(*args, **kwargs):
        raise LegacyRuntimeBusy('busy')
    monkeypatch.setattr(domain, 'activate_domain', busy)
    monkeypatch.setattr(domain, 'list_domains', lambda: ['CloudNerd'])
    namespace = dict(HTTPException=HTTPException, Form=Form, LegacyRuntimeBusy=LegacyRuntimeBusy)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'main.py', 'exec'), namespace)
    with pytest.raises(HTTPException) as result:
        namespace['agent_load_domain']('CloudNerd')
    assert result.value.status_code == 503
