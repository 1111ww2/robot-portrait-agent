import pytest

from leica_tracker.physical_stop_bridge import StopBridge


class FakeTransport:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def stop(self, generation):
        self.calls.append(generation)
        if self.fail:
            raise RuntimeError('link lost')


def make(transport):
    return StopBridge(transport, generation=2, max_sample_age=.2,
                      max_tcp_linear=.001, max_tcp_angular=.002,
                      max_joint_speed=.002, dwell=.5, timeout=2.)


def sample(bridge, at, now=None, tcp=None, joints=None):
    return bridge.observe(sampled_at=at, now=at if now is None else now,
                          tcp_speed=tcp or [0.] * 6,
                          joint_speed=joints or [0.] * 6)


def test_stop_is_latched_and_request_is_not_proof_of_stopping():
    transport = FakeTransport()
    bridge = make(transport)
    assert bridge.request_stop(now=1.) == 'REQUESTED'
    assert bridge.request_stop(now=1.1) == 'REQUESTED'
    assert transport.calls == [2]
    assert bridge.state == 'REQUESTED'
    sample(bridge, 1.01)
    sample(bridge, 1.15)
    sample(bridge, 1.3)
    assert bridge.state == 'REQUESTED'
    sample(bridge, 1.45)
    sample(bridge, 1.52)
    assert bridge.state == 'STATIONARY_OBSERVED'
    assert bridge.request_stop(now=1.6) == 'STATIONARY_OBSERVED'
    assert transport.calls == [2]


def test_motion_resets_dwell_even_when_tcp_speeds_zero():
    bridge = make(FakeTransport())
    bridge.request_stop(now=1.)
    sample(bridge, 1.1)
    sample(bridge, 1.2)
    sample(bridge, 1.3)
    sample(bridge, 1.4, joints=[.01, 0., 0., 0., 0., 0.])
    sample(bridge, 1.5)
    sample(bridge, 1.6)
    assert bridge.state == 'REQUESTED'
    for at in (1.7, 1.8, 1.9, 2., 2.1):
        sample(bridge, at)
    sample(bridge, 2.11)
    assert bridge.state == 'STATIONARY_OBSERVED'


def test_stale_or_pre_stop_sample_fails_closed():
    for at, now in [(1., 1.1), (1.01, 1.3), (1.2, 1.1)]:
        bridge = make(FakeTransport())
        bridge.request_stop(now=1.)
        with pytest.raises(ValueError):
            sample(bridge, at, now)
        assert bridge.state == 'FAULT'


def test_repeated_sample_does_not_confirm_stopping():
    bridge = make(FakeTransport())
    bridge.request_stop(now=1.)
    sample(bridge, 1.1)
    with pytest.raises(ValueError):
        sample(bridge, 1.1, 1.2)
    assert bridge.state == 'FAULT'


def test_stop_send_failure_is_latched_and_never_retries():
    transport = FakeTransport(fail=True)
    bridge = make(transport)
    assert bridge.request_stop(now=1.) == 'FAULT'
    assert bridge.request_stop(now=1.1) == 'FAULT'
    assert transport.calls == [2]


def test_stop_transport_failure_result_is_not_a_successful_request():
    transport = FakeTransport()
    transport.stop = lambda generation: False
    bridge = make(transport)
    assert bridge.request_stop(now=1.) == 'FAULT'


def test_stop_timeout_is_explicit_failure_not_success():
    bridge = make(FakeTransport())
    bridge.request_stop(now=1.)
    bridge.tick(now=3.01)
    assert bridge.state == 'FAULT'
    assert 'timeout' in bridge.failure.lower()


def test_disconnected_telemetry_cannot_be_reported_as_stopped():
    bridge = make(FakeTransport())
    bridge.request_stop(now=1.)
    bridge.telemetry_lost()
    assert bridge.state == 'FAULT'
    with pytest.raises(RuntimeError):
        sample(bridge, 1.1)


def test_feedback_gap_fails_instead_of_counting_time_as_stationary():
    bridge = make(FakeTransport())
    bridge.request_stop(now=1.)
    sample(bridge, 1.1)
    with pytest.raises(ValueError):
        sample(bridge, 1.7)
    assert bridge.state == 'FAULT'


def test_independent_watchdog_must_detect_telemetry_silence():
    bridge = make(FakeTransport())
    bridge.request_stop(now=1.)
    sample(bridge, 1.1)
    bridge.tick(now=1.31)
    assert bridge.state == 'FAULT'


def test_movement_after_stationary_observation_revokes_it():
    bridge = make(FakeTransport())
    bridge.request_stop(now=1.)
    for at in (1.1, 1.2, 1.3, 1.4, 1.5, 1.6):
        sample(bridge, at)
    assert bridge.state == 'STATIONARY_OBSERVED'
    sample(bridge, 1.7, tcp=[.01, 0., 0., 0., 0., 0.])
    assert bridge.state == 'FAULT'
