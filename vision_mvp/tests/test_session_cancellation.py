"""验证协作式服务关闭能触及活跃会话，全部设备均使用替身。"""
import socket
import subprocess
import time
from types import SimpleNamespace

import pytest

from leica_tracker import physical_runtime as runtime
from test_physical_intro_to_xyz import fixture as intro_fixture


@pytest.fixture(autouse=True)
def forbid_external_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Cancellation tests may not open sockets or launch processes')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)


def session_fixture(monkeypatch, *, phase='active'):
    stopped = [False]
    calls = []
    control = dict(instance='fake', epoch=1, started_ns=1, active=True,
                   control_action='restart', camera_connected=True, camera_fresh=True)
    gate = SimpleNamespace(can_send=True)
    def stop_gate(reason):
        calls.append(('gate_stop', reason))
        gate.can_send = False
    gate.stop = stop_gate
    session = SimpleNamespace(gate=gate, engine=object(),
        stop=lambda: stop_gate('session_stop'), close=lambda: calls.append('session_close'))
    reference = SimpleNamespace(status=lambda _: {'blocked': False},
                                fault=lambda: calls.append('reference_fault'))
    monkeypatch.setattr(runtime, 'register_backend', lambda *a, **k: reference)
    monkeypatch.setattr(runtime, 'FeedbackPublisher', lambda *a: SimpleNamespace(
        close=lambda: calls.append('publisher_close')))
    monkeypatch.setattr(runtime, 'LeaseHeartbeat', lambda *a: SimpleNamespace(
        failure=None, close=lambda: calls.append('heartbeat_close')))
    monkeypatch.setattr(runtime, 'DashboardReceiver', lambda *a: SimpleNamespace(
        close=lambda: calls.append('dashboard_close')))
    monkeypatch.setattr(runtime.time, 'sleep', lambda _: None)

    class Source:
        failure = None
        count = 0
        def take(self):
            self.count += 1
            if phase == 'event' and 'session_open' in calls:
                stopped[0] = True
                return dict(event='person_guidance', tracking=True,
                            control_instance='fake', control_epoch=1,
                            source_monotonic_ns=time.monotonic_ns())
            if phase == 'trigger':
                if self.count > 1:
                    raise AssertionError('Trigger loop continued after service cancellation')
                stopped[0] = True
                return None
            if not stopped[0]:
                return dict(event='person_guidance', tracking=True,
                            control_instance='fake', control_epoch=1,
                            source_monotonic_ns=time.monotonic_ns())
            return None

    def open_session(*args, **options):
        assert options == {'require_camera': False}
        calls.append('session_open')
        return session
    monkeypatch.setattr(runtime, 'open_resident', open_session)

    class Runtime:
        def __init__(self, *a, **k):
            self.count = 0
        def step(self, event):
            self.count += 1
            calls.append('runtime_step')
            stopped[0] = True
            if self.count > 1:
                raise AssertionError('Runtime continued after service cancellation')
    monkeypatch.setattr(runtime, 'PhysicalRuntime', Runtime)
    profile = SimpleNamespace(host='offline', gate_timeout=.1, cfg=object(),
                              supervisor=lambda *a: object())
    return profile, control, Source(), stopped, calls, session


def test_default_persistent_runner_stops_active_session_before_another_step(monkeypatch):
    profile, control, source, stopped, calls, session = session_fixture(monkeypatch)
    runtime.serve_sessions(profile, 8767, 'fake', source, read_guest=lambda: control,
        stopped_check=lambda _: calls.append('stationary_checked') or True,
        report=lambda *a: None, is_alive=lambda: not stopped[0], sleep=lambda _: None)
    assert calls.count('runtime_step') == 1
    assert calls.count('session_open') == 1
    assert not session.gate.can_send
    assert 'session_close' in calls and 'stationary_checked' in calls
    assert 'heartbeat_close' in calls and 'publisher_close' in calls


def test_trigger_wait_cancellation_retires_lease_without_opening_robot(monkeypatch):
    profile, control, source, stopped, calls, _ = session_fixture(monkeypatch, phase='trigger')
    with pytest.raises(RuntimeError, match='service stop requested'):
        runtime.run_session(profile, 8767, 'fake', source, lambda: control,
                            before=control, should_stop=lambda: stopped[0])
    assert 'session_open' not in calls
    assert 'heartbeat_close' in calls and 'publisher_close' in calls and 'reference_fault' in calls


def test_cancellation_during_event_read_prevents_that_runtime_step(monkeypatch):
    profile, control, source, stopped, calls, session = session_fixture(monkeypatch, phase='event')
    with pytest.raises(RuntimeError, match='service stop requested'):
        runtime.run_session(profile, 8767, 'fake', source, lambda: control,
                            before=control, should_stop=lambda: stopped[0])
    assert 'runtime_step' not in calls
    assert not session.gate.can_send and 'session_close' in calls


@pytest.mark.parametrize('cancel_during_read', [False, True])
def test_bootstrap_watchdog_reader_closes_switch_on_cancellation(monkeypatch, cancel_during_read):
    profile, control, source, stopped, calls, session = session_fixture(monkeypatch)
    reads = []
    def read_control():
        reads.append('read')
        if cancel_during_read:
            stopped[0] = True
        return control
    def open_session(host, cfg, guarded_reader, **options):
        assert options == {'require_camera': False}
        if not cancel_during_read:
            stopped[0] = True
        with pytest.raises(RuntimeError, match='service stop requested'):
            guarded_reader()
        return session
    monkeypatch.setattr(runtime, 'open_resident', open_session)
    with pytest.raises(RuntimeError, match='service stop requested'):
        runtime.run_session(profile, 8767, 'fake', source, read_control,
                            before=control, should_stop=lambda: stopped[0])
    assert reads == (['read'] if cancel_during_read else [])
    assert not session.gate.can_send
    assert 'runtime_step' not in calls and 'session_close' in calls


def test_intro_cancellation_after_final_tick_never_opens_xyz():
    profile, reader, source, lease, switch, kwargs, events = intro_fixture()
    stopped = [False]
    original_opener = kwargs['intro_opener']
    def open_intro(*args, **options):
        intro = original_opener(*args, **options)
        original_tick = intro.tick
        def tick():
            original_tick()
            stopped[0] = True
        intro.tick = tick
        return intro
    kwargs['intro_opener'] = open_intro
    with pytest.raises(RuntimeError, match='service stop requested'):
        runtime.run_intro_then_open_xyz(profile, reader, source, lease, switch,
                                        should_stop=lambda: stopped[0], **kwargs)
    assert 'xyz_open' not in events and 'intro_close' in events
    assert 'stop' in events


def test_intro_cancelled_before_start_does_not_open_any_device():
    profile, reader, source, lease, switch, kwargs, events = intro_fixture()
    with pytest.raises(RuntimeError, match='service stop requested'):
        runtime.run_intro_then_open_xyz(profile, reader, source, lease, switch,
                                        should_stop=lambda: True, **kwargs)
    assert 'intro_open' not in events and 'xyz_open' not in events


def test_intro_cancellation_during_source_read_prevents_first_motion_tick():
    profile, reader, source, lease, switch, kwargs, events = intro_fixture()
    stopped = [False]
    original_take = source.take
    def take():
        event = original_take()
        stopped[0] = True
        return event
    source.take = take
    with pytest.raises(RuntimeError, match='service stop requested'):
        runtime.run_intro_then_open_xyz(profile, reader, source, lease, switch,
                                        should_stop=lambda: stopped[0], **kwargs)
    assert 'intro_tick' not in events and 'xyz_open' not in events
    assert 'intro_close' in events
