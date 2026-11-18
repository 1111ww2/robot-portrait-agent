"""通信政策和门禁测试仅使用时钟/读取器/通道。"""
import importlib
import threading
import time

import pytest

from leica_tracker.physical_interaction_supervisor import ReferenceRequestError
from leica_tracker.resident_guest_watchdog import (
    GuestControlWatchdog, GuestSnapshotRequestError, ResidentMotionGate,
)


CONTROL = dict(instance='same-guest', epoch=7, started_ns=123)


class Clock:
    def __init__(self): self.now = 10.
    def __call__(self): return self.now
    def advance(self, seconds): self.now += seconds


def snapshot(**changes):
    return {**CONTROL, 'active': True, 'control_action': 'start',
            'control_fresh': True, 'camera_connected': True,
            'camera_fresh': True, **changes}


@pytest.fixture
def policy_type():
    def construct(*args, **kwargs):
        try:
            cls = importlib.import_module('leica_tracker.network_recovery').CommunicationRecovery
        except ModuleNotFoundError:
            pytest.fail('CommunicationRecovery is not implemented')
        return cls(*args, **kwargs)
    return construct


def test_fixed_recovery_deadline_and_stable_control_require_explicit_complete(policy_type):
    clock = Clock(); policy = policy_type(CONTROL, clock=clock)
    assert policy.observe(snapshot()) == 'healthy'
    assert not policy.paused and policy.deadline is None
    assert policy.observe(snapshot(control_fresh=False)) == 'waiting'
    deadline = policy.deadline
    assert deadline == 13. and policy.paused
    clock.advance(1.)
    assert policy.observe(snapshot(camera_fresh=False)) == 'waiting'
    assert policy.deadline == deadline
    assert policy.observe(snapshot()) == 'waiting'
    clock.advance(.299)
    assert policy.observe(snapshot()) == 'waiting'
    clock.advance(.002)
    assert policy.observe(snapshot()) == 'ready'
    assert policy.paused and policy.ready and policy.deadline == deadline
    policy.complete()
    assert policy.status == 'healthy' and not policy.paused and policy.deadline is None


def test_a_new_gap_breaks_stability_without_extending_first_deadline(policy_type):
    clock = Clock(); policy = policy_type(CONTROL, clock=clock)
    policy.observe(snapshot(control_fresh=False))
    clock.advance(2.6); policy.observe(snapshot())
    clock.advance(.2); policy.observe(snapshot(camera_connected=False))
    assert policy.deadline == 13.
    clock.advance(.21)
    with pytest.raises(RuntimeError, match='expired'):
        policy.observe(snapshot())
    with pytest.raises(RuntimeError): policy.complete()
    assert policy.failure is not None


@pytest.mark.parametrize('change', [
    {'active': False}, {'control_action': 'stop'}, {'instance': 'replacement'},
    {'epoch': 8}, {'started_ns': 124}, {'control_fresh': None},
    {'camera_fresh': 1}, {'active': 1}, {'epoch': True},
    {'communication': {'status': 'waiting', 'recoverable': False, 'remaining_ms': 2000}},
    {'communication': {'status': 'healthy', 'recoverable': False, 'remaining_ms': 0}},
    {'communication': {'status': 'unknown', 'recoverable': True, 'remaining_ms': 2000}},
])
def test_stop_identity_and_schema_changes_are_terminal_even_during_recovery(policy_type, change):
    policy = policy_type(CONTROL, clock=Clock())
    policy.observe(snapshot(control_fresh=False))
    with pytest.raises(RuntimeError): policy.observe(snapshot(**change))
    original = policy.failure
    with pytest.raises(RuntimeError): policy.observe(snapshot())
    assert policy.failure == original


@pytest.mark.parametrize('kind', ['guest', 'reference'])
@pytest.mark.parametrize('category', ['timeout', 'connection'])
def test_only_bounded_known_transport_failures_begin_recovery(policy_type, kind, category):
    clock = Clock(); policy = policy_type(CONTROL, clock=clock)
    error = (GuestSnapshotRequestError(category, 'request', 100) if kind == 'guest'
             else ReferenceRequestError(category, phase='request', elapsed_ms=100))
    assert policy.transport_error(error) is True
    assert policy.paused and policy.deadline == 13.
    clock.advance(1.)
    assert policy.transport_error(error) is True and policy.deadline == 13.


@pytest.mark.parametrize('error', [
    GuestSnapshotRequestError('http_rejected', 'headers', 1, 403),
    GuestSnapshotRequestError('http_rejected', 'headers', 1, 409),
    GuestSnapshotRequestError('response', 'body', 1),
    ReferenceRequestError('protocol', phase='body', elapsed_ms=1),
    TimeoutError('unclassified'), RuntimeError('unknown'),
])
def test_auth_schema_and_unknown_transport_errors_are_not_recoverable(policy_type, error):
    policy = policy_type(CONTROL, clock=Clock())
    assert policy.transport_error(error) is False
    assert policy.failure is not None
    with pytest.raises(RuntimeError): policy.observe(snapshot())


def test_server_waiting_cannot_reset_local_deadline_or_forge_freshness(policy_type):
    clock = Clock(); policy = policy_type(CONTROL, clock=clock)
    value = snapshot(communication=dict(status='waiting', recoverable=True, remaining_ms=3000))
    assert policy.observe(value) == 'waiting'
    assert value['control_fresh'] is True
    clock.advance(2.8)
    assert policy.observe(value) == 'waiting' and policy.deadline == 13.
    clock.advance(.21)
    with pytest.raises(RuntimeError, match='expired'): policy.observe(value)


def test_first_server_ready_still_holds_for_local_stability_and_explicit_complete(policy_type):
    clock = Clock(); policy = policy_type(CONTROL, clock=clock)
    value = snapshot(communication=dict(status='ready', recoverable=True, remaining_ms=3000))
    assert policy.observe(value) == 'waiting' and policy.paused
    assert policy.deadline == 13.
    clock.advance(.299)
    assert policy.observe(value) == 'waiting'
    clock.advance(.002)
    assert policy.observe(value) == 'ready' and policy.paused
    policy.complete()
    assert policy.status == 'healthy' and not policy.paused


@pytest.mark.parametrize('initial_status', ['waiting', 'ready'])
def test_first_server_hold_can_only_shorten_local_window(initial_status, policy_type):
    clock = Clock(); policy = policy_type(CONTROL, clock=clock)
    value = snapshot(communication=dict(status=initial_status, recoverable=True, remaining_ms=500))
    assert policy.observe(value) == 'waiting' and policy.deadline == 10.5
    clock.advance(.2)
    value['communication'].update(status='ready', remaining_ms=3000)
    assert policy.observe(value) == 'waiting' and policy.deadline == 10.5
    clock.advance(.301)
    with pytest.raises(RuntimeError, match='expired'): policy.observe(value)


def test_initial_healthy_negotiated_snapshot_does_not_create_a_pause(policy_type):
    policy = policy_type(CONTROL, clock=Clock())
    assert policy.observe(snapshot(communication=dict(
        status='healthy', recoverable=True, remaining_ms=0))) == 'healthy'
    assert policy.deadline is None and not policy.paused


def test_complete_cannot_bypass_stability_or_make_terminal_policy_healthy(policy_type):
    policy = policy_type(CONTROL, clock=Clock())
    with pytest.raises(RuntimeError): policy.complete()
    policy.observe(snapshot(camera_fresh=False))
    with pytest.raises(RuntimeError): policy.complete()
    with pytest.raises(RuntimeError, match='operator stop'): policy.fail('operator stop')
    with pytest.raises(RuntimeError): policy.complete()


class Channel:
    def __init__(self): self.sent = []; self.closed = threading.Event()
    def send(self, *args): self.sent.append(args)
    def close(self): self.closed.set()


def test_pause_substitutes_native_hold_but_preserves_handshake_and_exact_identity():
    channel = Channel(); gate = ResidentMotionGate(channel); gate.arm()
    gate.pause('communication')
    endpoint = [1., 2., 3.]
    gate.send(10, 2, 99, endpoint)
    gate.send(11, 1, 99, endpoint)
    gate.send(12, 0, 99, endpoint)
    assert gate.can_send and gate.is_paused and not channel.closed.is_set()
    assert channel.sent == [(10, 3, 99, endpoint), (11, 3, 99, endpoint), (12, 0, 99, endpoint)]
    assert channel.sent[0][3] is endpoint
    gate.resume(); gate.send(13, 2, 99, endpoint)
    assert not gate.is_paused and channel.sent[-1][1] == 2


def test_stop_while_paused_is_permanent_and_resume_does_not_reopen_channel():
    channel = Channel(); gate = ResidentMotionGate(channel); gate.arm()
    gate.pause('communication'); gate.stop('operator stop')
    assert channel.closed.wait(1) and not gate.can_send
    with pytest.raises(RuntimeError): gate.resume()
    with pytest.raises(RuntimeError): gate.arm()
    with pytest.raises(RuntimeError): gate.send(1, 2, 99, [1, 2, 3])
    assert channel.sent == []


def wait_for(predicate, timeout=1.):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        if predicate(): return True
        time.sleep(.005)
    return False


def test_optional_watchdog_pauses_and_keeps_reading_until_runtime_resumes(policy_type):
    clock = Clock(); policy = policy_type(CONTROL, clock=clock)
    fresh_observed = threading.Event()
    observe = policy.observe
    def observed(value):
        result = observe(value)
        if value['camera_fresh'] and policy.paused:
            fresh_observed.set()
        return result
    policy.observe = observed
    state = snapshot(); channel = Channel(); gate = ResidentMotionGate(channel)
    watch = GuestControlWatchdog(lambda: dict(state), gate, interval=.01, timeout=.06,
                                 require_camera=False, recovery=policy)
    watch.start()
    try:
        assert watch.wait_ready(1)
        state['camera_fresh'] = False
        assert wait_for(lambda: gate.is_paused)
        assert not watch.stopped.is_set() and not channel.closed.is_set()
        state['camera_fresh'] = True
        assert fresh_observed.wait(1)
        clock.advance(.31)
        assert wait_for(lambda: policy.ready)
        assert gate.is_paused  # 新鲜控制永远不会授权原生简历本身。
        policy.complete(); gate.resume()
        assert not gate.is_paused and not watch.stopped.is_set()
        state['active'] = False
        assert watch.stopped.wait(1) and channel.closed.wait(1)
    finally: watch.close()


def test_optional_watchdog_timeout_pauses_hung_reader_then_fixed_budget_expires(policy_type):
    clock = Clock(); policy = policy_type(CONTROL, clock=clock)
    blocked = threading.Event(); calls = []
    def read():
        calls.append(1)
        if len(calls) > 1: blocked.wait(1)
        return snapshot()
    channel = Channel(); gate = ResidentMotionGate(channel)
    watch = GuestControlWatchdog(read, gate, interval=.01, timeout=.04, recovery=policy)
    watch.start()
    try:
        assert watch.wait_ready(1)
        assert wait_for(lambda: gate.is_paused)
        assert not watch.stopped.is_set()
        clock.advance(3.01)
        assert watch.stopped.wait(1) and channel.closed.wait(1)
    finally:
        blocked.set(); watch.close()


def test_new_gap_cannot_be_cleared_by_concurrent_native_resume(policy_type):
    clock = Clock(); policy = policy_type(CONTROL, clock=clock)
    channel = Channel(); gate = ResidentMotionGate(channel); gate.arm(); gate.pause('gap')
    policy.observe(snapshot(control_fresh=False)); policy.observe(snapshot())
    clock.advance(.31); assert policy.observe(snapshot()) == 'ready'
    entered = threading.Event(); release = threading.Event(); gap_done = threading.Event()
    errors = []
    def resume():
        entered.set()
        assert release.wait(1)
        gate.resume()
    def completing():
        try: policy.complete(resume=resume)
        except Exception as exc: errors.append(exc)
    def new_gap():
        with policy.synchronized():
            policy.observe(snapshot(camera_fresh=False))
            gate.pause('new gap')
        gap_done.set()
    completing_thread = threading.Thread(target=completing)
    gap_thread = threading.Thread(target=new_gap)
    completing_thread.start()
    try:
        assert entered.wait(1)
        gap_thread.start()
        assert not gap_done.wait(.03)
    finally:
        release.set()
        completing_thread.join(1)
        if gap_thread.ident is not None: gap_thread.join(1)
    assert not errors and gap_done.is_set()
    assert policy.status == 'waiting' and gate.is_paused and gate.can_send
