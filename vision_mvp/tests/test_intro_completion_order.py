"""使用仅内存传输来练习真正的引出状态机/门。"""
import copy
import socket

import pytest

from leica_tracker.physical_intro_bootstrap import open_intro
from leica_tracker.resident_guest_watchdog import ResidentMotionGate
from test_physical_intro_bootstrap import setup


@pytest.fixture
def intro(monkeypatch):
    def forbidden_socket(*args, **kwargs):
        raise AssertionError('Network is forbidden in completion-order tests')
    monkeypatch.setattr(socket, 'socket', forbidden_socket)
    profile, dependencies, objects = setup()
    now = [10.]
    dependencies['clock'] = lambda: now[0]
    dependencies['gate_factory'] = ResidentMotionGate

    class Watch:
        failure = None
        def __init__(self, read, gate, **kwargs):
            self.gate = gate
        def start(self): self.gate.arm()
        def wait_ready(self, timeout): return self.gate.can_send
        def close(self): self.gate.stop('Watchdog closed')

    dependencies['watch_factory'] = Watch
    opened = open_intro('192.0.2.1', profile, lambda: None, dependencies=dependencies)

    def publish(time, state=101, sequence=3, running=True, *, session=123,
                joint_offset=0., joint_speed=0., tcp_offset=0., tcp_speed=0.):
        now[0] = time
        for name in ('writer', 'reader', 'state'):
            objects[name].latest = (copy.deepcopy(objects[name].latest[0]), time)
        objects['writer'].latest[0].update(session=session, state=state, sequence=sequence)
        reader = objects['reader'].latest[0]
        reader['joints'] = list(profile['reference_joints'])
        reader['joints'][5] += joint_offset
        reader['joint_speed'] = [0.] * 5 + [joint_speed]
        reader['tcp_pose'] = list(profile['reference_pose'])
        reader['tcp_pose'][0] += tcp_offset
        reader['tcp_speed'] = [tcp_speed] + [0.] * 5
        objects['state'].latest[0]['mode']['running'] = running
        objects['dashboard'].checked_at = time

    for time, state, ack in ((10., 100, 0), (10.04, 100, 1), (10.08, 101, 2)):
        publish(time, state, ack)
        opened.tick()
    assert opened.progress.started_ack
    yield opened, objects, publish
    opened.close()


def test_primary_stop_before_102_observes_completion_without_another_enable(intro):
    opened, objects, publish = intro
    sent_before = list(objects['writer'].sent)
    publish(10.12, running=False)
    assert opened.tick() == 'AWAITING_COMPLETION'
    assert opened.gate.can_send  # 保持运输和客人的监督员活跃。
    assert objects['writer'].sent == sent_before
    publish(10.16, state=102, running=False)
    assert opened.tick() == 'SETTLING'
    for index in range(1, 9):
        publish(10.16 + index * .1, state=102, running=False)
        state = opened.tick()
        if state == 'COMPLETE': break
    assert state == 'COMPLETE'
    assert objects['writer'].sent == sent_before


@pytest.mark.parametrize('running', [True, False])
def test_102_first_or_simultaneous_still_requires_measured_dwell(intro, running):
    opened, objects, publish = intro
    sent_before = list(objects['writer'].sent)
    publish(10.12, state=102, running=running)
    assert opened.tick() == 'SETTLING'
    for index in range(1, 9):
        publish(10.12 + index * .1, state=102, running=False)
        state = opened.tick()
        if index < 7: assert state != 'COMPLETE'
        if state == 'COMPLETE': break
    assert state == 'COMPLETE' and objects['writer'].sent == sent_before


@pytest.mark.parametrize('change', [
    {'joint_offset': .01}, {'tcp_offset': .001},
    {'joint_speed': .004}, {'tcp_speed': .001},
])
def test_primary_stop_away_from_stationary_endpoint_fails_without_enable(intro, change):
    opened, objects, publish = intro
    sent_before = list(objects['writer'].sent)
    publish(10.12, running=False, **change)
    with pytest.raises(RuntimeError, match='no longer running'):
        opened.tick()
    assert not opened.gate.can_send and objects['writer'].sent == sent_before


def test_primary_stop_without_102_has_a_bounded_observation_window(intro):
    opened, objects, publish = intro
    sent_before = list(objects['writer'].sent)
    publish(10.12, running=False)
    assert opened.tick() == 'AWAITING_COMPLETION'
    publish(10.24, running=False)
    assert opened.tick() == 'AWAITING_COMPLETION'
    publish(10.33, running=False)
    with pytest.raises(RuntimeError, match='completion.*timed out'):
        opened.tick()
    assert not opened.gate.can_send and objects['writer'].sent == sent_before


def test_late_102_cannot_rescue_expired_completion_observation(intro):
    opened, objects, publish = intro
    sent_before = list(objects['writer'].sent)
    publish(10.12, running=False)
    assert opened.tick() == 'AWAITING_COMPLETION'
    publish(10.24, running=False)
    assert opened.tick() == 'AWAITING_COMPLETION'
    publish(10.33, state=102, running=False)
    with pytest.raises(RuntimeError, match='completion.*timed out'):
        opened.tick()
    assert not opened.gate.can_send and objects['writer'].sent == sent_before


@pytest.mark.parametrize('state,diagnostic', [(-12, 'fault -12'),
                                             (-16, 'TCP linear speed'),
                                             (-17, 'TCP angular speed')])
def test_resident_negative_code_precedes_primary_stopped_diagnostic(intro, state, diagnostic):
    opened, objects, publish = intro
    sent_before = list(objects['writer'].sent)
    publish(10.4, state=state, sequence=2, running=False)
    with pytest.raises(RuntimeError, match=diagnostic):
        opened.tick()
    assert diagnostic in opened.gate.reason
    assert not opened.gate.can_send and objects['writer'].sent == sent_before


def test_fresh_resident_fault_precedes_expired_route_timeout(intro):
    opened, _, publish = intro
    publish(41., state=-16, sequence=2, running=False)
    with pytest.raises(RuntimeError, match='TCP linear speed'):
        opened.tick()
    assert not opened.gate.can_send


def test_resident_negative_code_during_completion_observation_still_fails(intro):
    opened, _, publish = intro
    publish(10.12, running=False)
    assert opened.tick() == 'AWAITING_COMPLETION'
    publish(10.16, state=-16, running=False)
    with pytest.raises(RuntimeError, match='TCP linear speed'):
        opened.tick()
    assert not opened.gate.can_send


def test_completion_observation_does_not_survive_guest_stop_or_late_102(intro):
    opened, objects, publish = intro
    publish(10.12, running=False)
    assert opened.tick() == 'AWAITING_COMPLETION'
    sent_before = list(objects['writer'].sent)
    opened.gate.stop('Guest Stop')
    publish(10.16, state=102, running=False)
    with pytest.raises(RuntimeError, match='watchdog stopped.*Guest Stop'):
        opened.tick()
    assert objects['writer'].sent == sent_before


@pytest.mark.parametrize('producer', ['writer', 'reader'])
def test_completion_observation_keeps_independent_telemetry_freshness_limits(intro, producer):
    opened, objects, publish = intro
    publish(10.12, running=False)
    assert opened.tick() == 'AWAITING_COMPLETION'
    publish(10.28, state=102, running=False)
    objects[producer].latest = (objects[producer].latest[0], 10.12)
    with pytest.raises(RuntimeError, match='Stale or repeated'):
        opened.tick()
    assert not opened.gate.can_send


def test_completion_observation_requires_matching_session(intro):
    opened, _, publish = intro
    publish(10.12, running=False)
    assert opened.tick() == 'AWAITING_COMPLETION'
    publish(10.16, state=102, running=False, session=999)
    with pytest.raises(RuntimeError, match='session mismatch'):
        opened.tick()
    assert not opened.gate.can_send


def test_primary_restart_during_observation_cannot_reenable_motion(intro):
    opened, objects, publish = intro
    publish(10.12, running=False)
    assert opened.tick() == 'AWAITING_COMPLETION'
    sent_before = list(objects['writer'].sent)
    publish(10.16, state=101, sequence=4, running=True)
    with pytest.raises(RuntimeError, match='restarted'):
        opened.tick()
    assert objects['writer'].sent == sent_before and not opened.gate.can_send


def test_102_regression_cannot_resume_heartbeat_enables(intro):
    opened, objects, publish = intro
    publish(10.12, state=102)
    assert opened.tick() == 'SETTLING'
    sent_before = list(objects['writer'].sent)
    publish(10.16, state=101, sequence=3)
    with pytest.raises(RuntimeError, match='completion.*regressed'):
        opened.tick()
    assert objects['writer'].sent == sent_before and not opened.gate.can_send


def test_completion_settling_cannot_extend_forever_by_resetting_still_dwell(intro):
    opened, _, publish = intro
    publish(10.12, state=102)
    assert opened.tick() == 'SETTLING'
    for index in range(1, 21):
        publish(10.12 + index * .1, state=102, running=False,
                joint_speed=.004 if index % 4 == 0 else 0.)
        assert opened.tick() == 'SETTLING'
    publish(12.23, state=102, running=False)
    with pytest.raises(RuntimeError, match='stationary reference'):
        opened.tick()
    assert not opened.gate.can_send
