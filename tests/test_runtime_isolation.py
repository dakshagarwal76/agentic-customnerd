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


def test_request_and_domain_switch_share_guard():
    tree = ast.parse((Path(__file__).parents[1] / 'main.py').read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    worker = functions['_run_agent_logic']
    assert 'isolated_legacy_request' in [ast.unparse(n) for n in worker.decorator_list]
    switch = functions['agent_load_domain']
    assert isinstance(switch, ast.FunctionDef), 'blocking domain switch must run in a worker, not the event loop'
    assert any(isinstance(n, ast.With) and any(ast.unparse(i.context_expr) == 'legacy_runtime_lock' for i in n.items) for n in ast.walk(switch))
