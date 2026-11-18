"""如果老工作线程幸存下来，启动式退休必须抑制后续的相遇。每个设备都被注入；禁止创建套接字和进程。"""
from types import SimpleNamespace
import socket
import subprocess

import pytest

from leica_tracker import physical_runtime as runtime
from leica_tracker.physical_intro_bootstrap import open_intro
from leica_tracker.physical_resident_bootstrap import open_resident
from test_physical_intro_bootstrap import setup as intro_setup
from test_physical_resident_bootstrap import setup as xyz_setup


@pytest.fixture(autouse=True)
def forbid_external_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('This test may not open a socket or launch a process')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)


def bootstrap(kind, cleanup_failure=None, *, reject_upload=True,
              reject_reader=False, interrupt=False):
    cfg, deps, objects = (intro_setup if kind == 'intro' else xyz_setup)()
    opener = open_intro if kind == 'intro' else open_resident
    calls = []
    upload_error = (KeyboardInterrupt() if interrupt else RuntimeError('Fake startup upload rejected'))

    def wrap_resource(factory, name):
        def create(*args, **kwargs):
            if name == 'rtde' and kwargs.get('receive_only') and reject_reader:
                raise upload_error
            value = factory(*args, **kwargs)
            label = ('reader' if kwargs.get('receive_only') else 'writer') if name == 'rtde' else name
            original_close = value.close
            def close():
                calls.append('close_' + label)
                if cleanup_failure == label:
                    raise RuntimeError('Fake ' + label + ' close failed')
                original_close()
            value.close = close
            if label == 'command' and reject_upload:
                def send(_):
                    raise upload_error
                value.send = send
            return value
        return create

    for key, name in (('state_factory', 'state'), ('rtde_factory', 'rtde'),
                      ('command_factory', 'command'), ('dashboard_factory', 'dashboard')):
        if key in deps:
            deps[key] = wrap_resource(deps[key], name)

    class CloseSignal:
        def set(self):
            pass
        def wait(self, timeout):
            calls.append(('wait_input', timeout))
            return cleanup_failure != 'input_timeout'

    original_gate = deps['gate_factory']
    def gate_factory(channel):
        gate = original_gate(channel)
        gate.close_finished = CloseSignal()
        gate.close_error = 'Fake input close failed' if cleanup_failure == 'input' else None
        objects['gate'] = gate
        return gate
    deps['gate_factory'] = gate_factory

    class WatchThread:
        def join(self, timeout):
            calls.append(('join_watch', timeout))
        def is_alive(self):
            return cleanup_failure == 'watch_thread'

    original_watch = deps['watch_factory']
    def watch_factory(*args, **kwargs):
        watch = original_watch(*args, **kwargs)
        watch._threads = [WatchThread()]
        original_close = watch.close
        def close():
            calls.append('close_watch')
            if cleanup_failure == 'watch':
                raise RuntimeError('Fake watchdog close failed')
            original_close()
        watch.close = close
        return watch
    deps['watch_factory'] = watch_factory

    def run():
        return opener('192.0.2.1', cfg, lambda: None, dependencies=deps)
    return run, objects, calls, upload_error


@pytest.mark.parametrize('kind', ['intro', 'xyz'])
@pytest.mark.parametrize('failure', ['input', 'input_timeout', 'reader', 'state',
                                   'command', 'watch', 'watch_thread'])
def test_failed_bootstrap_promotes_incomplete_cleanup_and_keeps_original_cause(kind, failure):
    run, objects, calls, original = bootstrap(kind, failure)
    with pytest.raises(runtime.SessionCleanupError) as caught:
        run()
    assert caught.value.__cause__ is original
    assert caught.value.cleanup_errors
    assert not objects['gate'].can_send
    assert 'close_command' in calls and 'close_reader' in calls and 'close_state' in calls
    if kind == 'intro':
        assert 'close_dashboard' in calls
    assert ('wait_input', 2.) in calls and ('join_watch', 1.) in calls


@pytest.mark.parametrize('kind', ['intro', 'xyz'])
def test_clean_failed_bootstrap_preserves_startup_error_and_retires_every_resource(kind):
    run, objects, calls, original = bootstrap(kind)
    with pytest.raises(RuntimeError) as caught:
        run()
    assert caught.value is original
    assert ('wait_input', 2.) in calls and ('join_watch', 1.) in calls
    assert not objects['gate'].can_send


@pytest.mark.parametrize('kind', ['intro', 'xyz'])
def test_unclean_bootstrap_latches_persistent_service_even_when_stationary_stop_passes(kind):
    run, _, _, _ = bootstrap(kind, 'input')
    snapshots = [dict(instance='fake', epoch=epoch, started_ns=epoch * 100,
                      active=True, control_action='start',
                      camera_connected=True, camera_fresh=True) for epoch in (1, 2, 3)]
    attempted = []
    def runner(*args, **kwargs):
        attempted.append(kwargs['before']['epoch'])
        run()
    runtime.serve_sessions(SimpleNamespace(host='offline'), 8767, 'fake', SimpleNamespace(failure=None),
        read_guest=lambda: snapshots.pop(0), session_runner=runner,
        stopped_check=lambda _: True, report=lambda *args: None,
        is_alive=lambda: bool(snapshots), sleep=lambda _: None)
    assert attempted == [1]


@pytest.mark.parametrize('kind', ['intro', 'xyz'])
@pytest.mark.parametrize('failure', ['reader', 'state', 'watch'])
def test_returned_session_close_propagates_resource_failures_after_retiring_others(kind, failure):
    run, objects, calls, _ = bootstrap(kind, failure, reject_upload=False)
    session = run()
    with pytest.raises(runtime.SessionCleanupError):
        session.close()
    assert not objects['gate'].can_send
    assert 'close_command' in calls and 'close_reader' in calls and 'close_state' in calls


@pytest.mark.parametrize('kind', ['intro', 'xyz'])
def test_partial_allocation_closes_writer_without_gate_and_reports_close_failure(kind):
    run, objects, calls, original = bootstrap(kind, 'writer', reject_reader=True)
    with pytest.raises(runtime.SessionCleanupError) as caught:
        run()
    assert caught.value.__cause__ is original
    assert 'gate' not in objects
    assert 'close_writer' in calls and 'close_state' in calls


@pytest.mark.parametrize('kind', ['intro', 'xyz'])
@pytest.mark.parametrize('cleanup_failure', [None, 'input'])
def test_bootstrap_cancellation_also_retires_resources_and_preserves_its_cause(kind, cleanup_failure):
    run, objects, calls, original = bootstrap(kind, cleanup_failure, interrupt=True)
    if cleanup_failure:
        with pytest.raises(runtime.SessionCleanupError) as caught:
            run()
        assert caught.value.__cause__ is original
    else:
        with pytest.raises(KeyboardInterrupt) as caught:
            run()
        assert caught.value is original
    assert not objects['gate'].can_send
    assert 'close_reader' in calls and 'close_state' in calls and ('wait_input', 2.) in calls
