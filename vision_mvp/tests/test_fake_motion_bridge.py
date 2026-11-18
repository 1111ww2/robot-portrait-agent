import pytest

from leica_tracker.offline_portrait_supervisor import OfflinePortraitSupervisor
from leica_tracker.fake_motion_bridge import FakeMotionBridge, FakeMotionTransport

REFERENCE = [-.1, -.5, .2, 0., 3., -.5]
ZERO = [0.] * 6
CONTROL = {'instance': 'fake-display', 'epoch': 1}
OWNER = {'owner': 'fake-owner', 'epoch': 1, 'active': True}


def prepare():
    s = OfflinePortraitSupervisor(hold_timeout=2., motion_timeout=2.)
    s.start(REFERENCE, ZERO, sampled_at=10., control=CONTROL, active_owner=OWNER,
            now=10., epoch=1)
    return s


def test_dispatch_is_blocked_until_observation_hold_ack():
    supervisor = prepare()
    transport = FakeMotionTransport()
    bridge = FakeMotionBridge(supervisor, transport)
    with pytest.raises(ValueError):
        bridge.dispatch(now=10.)
    assert transport.calls == []
    supervisor.acknowledge_hold(now=10.)
    generation = bridge.dispatch(now=10.)
    assert transport.calls == [('start', generation, 'OPENING')]
    assert bridge.poll(now=10.) == 'pending'
    transport.complete(generation)
    assert bridge.poll(now=10.) == 'completed'
    assert supervisor.state == 'SETTLING'
    with pytest.raises(ValueError):
        supervisor.release(now=10.)


def test_stop_cancels_and_late_completion_cannot_resume():
    supervisor = prepare()
    transport = FakeMotionTransport()
    bridge = FakeMotionBridge(supervisor, transport)
    supervisor.acknowledge_hold(now=10.)
    generation = bridge.dispatch(now=10.)
    bridge.stop()
    assert transport.calls[-1] == ('stop', generation)
    assert supervisor.state == 'STOPPED'
    with pytest.raises(RuntimeError):
        transport.complete(generation)
    with pytest.raises(RuntimeError):
        bridge.poll(now=10.)


def test_heartbeat_loss_requests_stop_and_faults():
    supervisor = prepare()
    transport = FakeMotionTransport()
    bridge = FakeMotionBridge(supervisor, transport)
    supervisor.acknowledge_hold(now=10.)
    generation = bridge.dispatch(now=10.)
    with pytest.raises(RuntimeError):
        bridge.poll(now=10.4)
    assert transport.calls[-1] == ('stop', generation)
    assert supervisor.state == 'FAULT'


def test_failed_motion_latches_without_retry():
    supervisor = prepare()
    transport = FakeMotionTransport()
    bridge = FakeMotionBridge(supervisor, transport)
    supervisor.acknowledge_hold(now=10.)
    generation = bridge.dispatch(now=10.)
    transport.complete(generation, success=False)
    assert bridge.poll(now=10.) == 'failed'
    assert supervisor.state == 'FAULT'
    with pytest.raises(RuntimeError):
        bridge.dispatch(now=10.)
    assert len([c for c in transport.calls if c[0] == 'start']) == 1


def test_bridge_rejects_real_or_injected_transport():
    class NotAFake:
        def start(self, *args): raise AssertionError('must not call')
    with pytest.raises(TypeError):
        FakeMotionBridge(prepare(), NotAFake())


def test_second_direction_only_after_new_frame_and_new_hold():
    supervisor = prepare()
    transport = FakeMotionTransport()
    bridge = FakeMotionBridge(supervisor, transport)
    supervisor.acknowledge_hold(now=10.)
    first = bridge.dispatch(now=10.)
    transport.complete(first)
    assert bridge.poll(now=10.) == 'completed'
    now = 10.
    for _ in range(8):
        now = round(now + .1, 2)
        supervisor.observe_control(CONTROL, active_owner=OWNER, now=now)
        supervisor.observe_telemetry(REFERENCE, ZERO, sampled_at=now, now=now)
    release = supervisor.release(now=now)
    now += .1
    supervisor.observe_control(CONTROL, active_owner=OWNER, now=now)
    supervisor.observe_telemetry(REFERENCE, ZERO, sampled_at=now, now=now)
    assert not supervisor.observe_frame(release, now=now, tracking=True, within=False)
    stamp = round(now * 1e9)
    supervisor.observe_frame(stamp, now=now, tracking=True, within=False)
    supervisor.request_direction('MOVE_LEFT', source_ns=stamp, now=now)
    with pytest.raises(ValueError):
        bridge.dispatch(now=now)
    supervisor.acknowledge_hold(now=now)
    second = bridge.dispatch(now=now)
    assert second > first
    assert transport.calls[-1] == ('start', second, 'MOVE_LEFT')
    with pytest.raises(RuntimeError):
        transport.complete(first)
    bridge.stop()


def test_concurrent_stop_wins_over_a_delayed_fake_start():
    import threading
    supervisor = prepare()
    transport = FakeMotionTransport()
    bridge = FakeMotionBridge(supervisor, transport)
    supervisor.acknowledge_hold(now=10.)
    started = threading.Event()
    release = threading.Event()
    original_start = transport.start
    errors = []

    def delayed_start(cue, generation):
        started.set()
        assert release.wait(2.)
        return original_start(cue, generation)

    def run_dispatch():
        try:
            bridge.dispatch(now=10.)
        except RuntimeError as error:
            errors.append(str(error))

    transport.start = delayed_start
    worker = threading.Thread(target=run_dispatch)
    worker.start()
    assert started.wait(1.)
    bridge.stop()
    release.set()
    worker.join(2.)
    assert not worker.is_alive()
    assert errors
    assert supervisor.state == 'STOPPED'
    assert not [call for call in transport.calls if call[0] == 'start']


def test_stop_failure_still_latches_bridge_closed():
    supervisor = prepare()
    transport = FakeMotionTransport()
    bridge = FakeMotionBridge(supervisor, transport)
    supervisor.acknowledge_hold(now=10.)
    bridge.dispatch(now=10.)
    def fail_stop(generation):
        raise RuntimeError('fake stop submission failed')
    transport.stop = fail_stop
    with pytest.raises(RuntimeError, match='submission failed'):
        bridge.stop()
    assert supervisor.state == 'STOPPED'
    with pytest.raises(RuntimeError):
        bridge.dispatch(now=10.)
