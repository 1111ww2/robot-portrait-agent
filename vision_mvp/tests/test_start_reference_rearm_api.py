"""New-Start后端使用真正的环回HTTP和测量过的ReferenceGate逻辑。"""
import json
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from leica_tracker import guest_display as display_module
from leica_tracker.guest_display import GuestDisplay
from leica_tracker.physical_interaction_supervisor import LocalReferenceGate, ReferenceRequestError

POSE = [-.1, -.5, .2, 0., 3., -.5]
ZERO = [0.] * 6


class Capture:
    def connected(self): return True
    def fresh(self): return True
    def stream_fresh(self): return True
    def control_owner(self): return 'owner'
    def diagnostics(self): return {}


def call(display, path, body=None):
    req = urllib.request.Request(display.origin + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'X-Guest-Token': display.token, 'X-Guest-Client': 'owner',
                 'Origin': display.origin, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1) as response:
        return json.load(response)


@pytest.fixture
def setup(monkeypatch):
    clock = SimpleNamespace(now=1000.)
    monkeypatch.setattr(display_module, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    with GuestDisplay(port=0, remote_capture=Capture(), physical_feedback=True) as display:
        # 一个旧的、公认的门在停止处生存；其之前的停留有效。
        sid = display._reference_gate.begin(POSE, 999.)
        display._reference_gate.acknowledge()
        for i in range(8):
            t = 999.3 + i * .1
            display._reference_gate.sample(sid, POSE, ZERO, t, t)
        clock.now = 1000.01
        control = call(display, '/action', {'action': 'start', 'control_protocol': 1,
            'control': call(display, '/state')['control']})['control']
        lease = call(display, '/physical-feedback', {**control, 'op': 'register', 'mode': 'intro'})['lease']
        identity = {**control, 'lease': lease}
        ref = LocalReferenceGate(display.port, display.token, timeout=.2,
            control={**control, 'started_ns': display._control_started_ns}, lease=lease)
        yield display, clock, sid, ref, identity


def advance(setup, t):
    display, clock, _, _, identity = setup
    clock.now = t
    call(display, '/control-heartbeat', {**identity,
        'sequence': display._control_heartbeat_sequence + 1})
    call(display, '/physical-feedback', {**identity, 'op': 'heartbeat'})


def test_restart_begin_retains_original_reference_but_discards_old_dwell(setup):
    display, clock, sid, ref, _ = setup
    gate = display._reference_gate
    assert gate.return_verified(sid, clock.now)
    assert ref.restart_reference is None
    assert ref.restart_begin(sid) is True
    assert ref.restart_reference == tuple(POSE)
    assert isinstance(ref.restart_reference, tuple)
    assert gate.blocked and gate.session == sid and gate.reference == tuple(POSE)
    assert gate.acknowledged and gate.steady_since is None and gate.last_sample is None
    with pytest.raises(ReferenceRequestError): ref.restart_release(sid)


def test_restart_releases_only_after_new_continuous_reference_dwell(setup):
    display, clock, sid, ref, identity = setup
    assert ref.restart_begin(sid)
    began = clock.now
    for i in range(8):
        advance(setup, began + .01 + i * .1)
        result = ref.restart_sample(sid, POSE, ZERO, clock.now)
        assert result['sampled_at'] == clock.now
        if i < 6:
            assert result['return_verified'] is False
    epoch = identity['epoch']
    release_ns = ref.restart_release(sid)
    assert release_ns == round(clock.now * 1e9)
    assert not display._reference_gate.blocked
    assert display._reference_gate.held(release_ns)
    assert not display._reference_gate.held(release_ns + 1)
    assert display._control_epoch == epoch and display._active


def test_restart_sample_requires_matching_begin_and_new_sample(setup):
    display, clock, sid, ref, _ = setup
    with pytest.raises(ReferenceRequestError): ref.restart_sample(sid, POSE, ZERO, clock.now)
    assert ref.restart_begin(sid)
    with pytest.raises(ReferenceRequestError): ref.restart_sample(sid, POSE, ZERO, clock.now - .01)
    assert display._reference_gate.last_sample is None


def test_repeated_begin_cannot_reset_verification_and_uncertain_retry_is_idempotent(setup):
    display, clock, sid, ref, identity = setup
    body = {**identity, 'op': 'restart_begin', 'session': sid, 'request_id': 'a' * 32}
    first = call(display, '/reference-gate', body)
    advance(setup, clock.now + .01)
    ref.restart_sample(sid, POSE, ZERO, clock.now)
    since = display._reference_gate.steady_since
    assert call(display, '/reference-gate', body) == first
    assert display._reference_gate.steady_since == since
    with pytest.raises(ReferenceRequestError): ref.restart_begin(sid)
    assert display._reference_gate.steady_since == since
    with pytest.raises(urllib.error.HTTPError):
        call(display, '/reference-gate', {**body, 'session': 'changed'})


@pytest.mark.parametrize('change', ['wrong_lease', 'wrong_epoch', 'guidance', 'intro_active', 'stop'])
def test_restart_requires_current_start_intro_lease_before_any_intro(setup, change):
    display, clock, sid, _, identity = setup
    body = {**identity, 'op': 'restart_begin', 'session': sid, 'request_id': 'b' * 32}
    if change == 'wrong_lease': body['lease'] = 'wrong'
    elif change == 'wrong_epoch': body['epoch'] -= 1
    elif change == 'guidance': display._physical_backend_mode = 'guidance'
    elif change == 'intro_active':
        call(display, '/physical-feedback', {**identity, 'op': 'heartbeat', 'intro_id': 2, 'intro_status': 'active'})
    else: call(display, '/action', {'action': 'stop'})
    with pytest.raises(urllib.error.HTTPError): call(display, '/reference-gate', body)
    assert display._reference_gate.blocked and display._reference_gate.reference == tuple(POSE)


@pytest.mark.parametrize('change', ['moved', 'moving', 'gapped'])
def test_restart_cannot_release_without_original_reference_and_fresh_stillness(setup, change):
    display, clock, sid, ref, _ = setup
    ref.restart_begin(sid)
    began = clock.now
    for i in range(8):
        advance(setup, began + .01 + i * (.14 if change == 'gapped' else .1))
        pose = [POSE[0] + .001, *POSE[1:]] if change == 'moved' else POSE
        speed = [.001, *ZERO[1:]] if change == 'moving' else ZERO
        if change == 'gapped' and i == 4:
            advance(setup, clock.now + .02)
        ref.restart_sample(sid, pose, speed, clock.now)
    with pytest.raises(ReferenceRequestError): ref.restart_release(sid)
    assert display._reference_gate.blocked


def test_guidance_reference_operations_still_rejected_in_intro_mode(setup):
    _, _, sid, ref, _ = setup
    with pytest.raises(ReferenceRequestError): ref.sample(sid, POSE, ZERO, 1000.01)
    with pytest.raises(ReferenceRequestError): ref.release(sid)


def test_restart_release_uncertain_retry_returns_same_result_without_clearing_another_gate(setup):
    display, clock, sid, ref, identity = setup
    ref.restart_begin(sid)
    began = clock.now
    for i in range(8):
        advance(setup, began + .01 + i * .1)
        ref.restart_sample(sid, POSE, ZERO, clock.now)
    body = {**identity, 'op': 'restart_release', 'session': sid, 'request_id': 'c' * 32}
    first = call(display, '/reference-gate', body)
    assert call(display, '/reference-gate', body) == first
    with pytest.raises(ReferenceRequestError): ref.restart_release(sid)


@pytest.mark.parametrize('op', ['restart_begin', 'restart_release'])
def test_restart_mutation_transport_retries_one_request_identity(monkeypatch, op):
    from leica_tracker import network_reference as module
    from leica_tracker.network_recovery import CommunicationRecovery
    clock = [10.]
    control = dict(instance='new-start', epoch=3, started_ns=1)
    policy = CommunicationRecovery(control, clock=lambda: clock[0])
    def wait(seconds):
        clock[0] += seconds
        policy.observe({**control, 'active': True, 'control_fresh': True,
                        'camera_connected': True, 'camera_fresh': True})
        if policy.ready: policy.complete()
    monkeypatch.setattr(module, 'time', SimpleNamespace(sleep=wait))
    calls = []
    def exchange(self, method, path, body=None):
        calls.append(dict(body))
        if len(calls) == 1:
            raise ReferenceRequestError('timeout', phase='body', elapsed_ms=100)
        return {'ok': True, 'release_ns': 11_000_000_000, 'reference': POSE}
    monkeypatch.setattr(LocalReferenceGate, '_request', exchange)
    ref = module.RecoveryReferenceGate(8777, 'token', timeout=.1,
        control=control, lease='new-lease', recovery=policy)
    getattr(ref, op)('old-blocked-session')
    assert len(calls) == 2 and calls[0] == calls[1]
    assert len(calls[0]['request_id']) == 32


def test_interrupted_restart_sample_is_never_replayed(monkeypatch):
    from leica_tracker import network_reference as module
    from leica_tracker.network_recovery import CommunicationRecovery
    control = dict(instance='new-start', epoch=3, started_ns=1)
    policy = CommunicationRecovery(control, clock=lambda: 10.)
    calls = []
    def exchange(*args, **kwargs):
        calls.append(1)
        raise ReferenceRequestError('timeout', phase='body', elapsed_ms=100)
    monkeypatch.setattr(LocalReferenceGate, '_request', exchange)
    ref = module.RecoveryReferenceGate(8777, 'token', timeout=.1,
        control=control, lease='new-lease', recovery=policy)
    with pytest.raises(module.ReferenceSampleInterrupted):
        ref.restart_sample('old-blocked-session', POSE, ZERO, 9.99)
    assert calls == [1] and policy.paused and policy.failure is None


def publish_intro_hold(display, *, source_ns, frame=True, cue='HOLD', tracking=False):
    import numpy as np
    from leica_tracker.person_guidance import GuidanceDecision, GuidanceCue
    decision = GuidanceDecision('INTRO', GuidanceCue(cue), (.5, .3, .2), None, None,
        tracking, 'INTRO_IN_PROGRESS')
    image = np.zeros((72, 128, 3), dtype=np.uint8) if frame else None
    display.publish(image, decision, source_monotonic_ns=source_ns)


def test_new_start_intro_hold_acknowledges_only_real_fresh_post_begin_image(setup):
    display, clock, sid, ref, _ = setup
    display._reference_gate.acknowledged = False
    assert ref.restart_begin(sid)
    assert not display._reference_gate.acknowledged
    advance(setup, clock.now + .01)
    publish_intro_hold(display, source_ns=round(clock.now * 1e9))
    assert display._reference_gate.acknowledged
    assert display._reference_gate.blocked
    with pytest.raises(ReferenceRequestError): ref.restart_release(sid)


@pytest.mark.parametrize('change', ['no_source', 'pre_begin', 'stale', 'future',
    'no_frame', 'moving_cue', 'tracking', 'no_restart', 'intro_active', 'stop'])
def test_invalid_intro_hold_cannot_acknowledge_old_unacknowledged_gate(setup, change):
    display, clock, sid, ref, identity = setup
    display._reference_gate.acknowledged = False
    began = clock.now
    if change != 'no_restart': assert ref.restart_begin(sid)
    advance(setup, clock.now + .01)
    source = round(clock.now * 1e9)
    if change == 'no_source': source = None
    elif change == 'pre_begin': source = round((began - .001) * 1e9)
    elif change == 'stale': source = round((clock.now - .301) * 1e9)
    elif change == 'future': source += 1_000_000
    elif change == 'intro_active':
        call(display, '/physical-feedback', {**identity, 'op': 'heartbeat',
            'intro_id': 2, 'intro_status': 'active'})
    elif change == 'stop': call(display, '/action', {'action': 'stop'})
    publish_intro_hold(display, source_ns=source, frame=change != 'no_frame',
        cue='MOVE_RIGHT' if change == 'moving_cue' else 'HOLD', tracking=change == 'tracking')
    assert not display._reference_gate.acknowledged
    assert display._reference_gate.blocked


@pytest.mark.parametrize('reference', [None, [0.] * 5, [float('nan')] * 6,
    [float('inf')] * 6, [True] * 6, 'original'])
def test_restart_begin_rejects_invalid_exported_reference(monkeypatch, reference):
    control = dict(instance='new-start', epoch=3, started_ns=1)
    ref = LocalReferenceGate(8777, 'token', timeout=.1, control=control, lease='lease')
    monkeypatch.setattr(ref, '_request', lambda *args, **kwargs:
        {'ok': True, 'reference': reference})
    with pytest.raises(RuntimeError, match='reference'):
        ref.restart_begin('original-session')
    assert ref.restart_reference is None


def measured_restart(setup):
    _, clock, sid, ref, _ = setup
    ref.restart_begin(sid)
    began = clock.now
    for i in range(8):
        advance(setup, began + .01 + i * .1)
        ref.restart_sample(sid, POSE, ZERO, clock.now)


def test_atomic_restart_release_accepts_new_fresh_sample_after_original_dwell(setup):
    display, clock, sid, ref, _ = setup
    measured_restart(setup)
    previous = display._reference_gate.last_sample
    advance(setup, clock.now + .1)
    assert clock.now > previous
    release = ref.restart_release(sid, POSE, ZERO, clock.now)
    assert release == round(clock.now * 1e9) and not display._reference_gate.blocked


def test_atomic_restart_release_does_not_create_missing_dwell_or_reapply_sample(setup):
    display, clock, sid, ref, identity = setup
    ref.restart_begin(sid)
    advance(setup, clock.now + .01)
    body = {**identity, 'op': 'restart_release', 'session': sid, 'request_id': 'e' * 32,
            'pose': POSE, 'speed': ZERO, 'sampled_at': clock.now}
    with pytest.raises(urllib.error.HTTPError): call(display, '/reference-gate', body)
    since = display._reference_gate.steady_since
    assert since == clock.now
    advance(setup, clock.now + .05)
    with pytest.raises(urllib.error.HTTPError): call(display, '/reference-gate', body)
    assert display._reference_gate.steady_since == since
    assert display._reference_gate.last_sample == body['sampled_at']
    assert display._reference_gate.blocked


@pytest.mark.parametrize('change', ['duplicate', 'pre_begin', 'wrong_pose', 'missing_speed'])
def test_atomic_restart_release_rejects_invalid_sample_without_unlocking(setup, change):
    display, clock, sid, ref, identity = setup
    began = clock.now
    measured_restart(setup)
    stamp = clock.now
    advance(setup, clock.now + .01)
    body = {**identity, 'op': 'restart_release', 'session': sid, 'request_id': 'f' * 32,
            'pose': POSE, 'speed': ZERO, 'sampled_at': clock.now}
    if change == 'duplicate': body['sampled_at'] = stamp
    elif change == 'pre_begin': body['sampled_at'] = began - .01
    elif change == 'wrong_pose': body['pose'] = [POSE[0] + .001, *POSE[1:]]
    else: del body['speed']
    with pytest.raises(urllib.error.HTTPError): call(display, '/reference-gate', body)
    assert display._reference_gate.blocked


def test_atomic_restart_release_success_retry_is_idempotent(setup):
    display, clock, sid, _, identity = setup
    measured_restart(setup)
    advance(setup, clock.now + .01)
    body = {**identity, 'op': 'restart_release', 'session': sid, 'request_id': 'd' * 32,
            'pose': POSE, 'speed': ZERO, 'sampled_at': clock.now}
    first = call(display, '/reference-gate', body)
    assert first['ok'] and not display._reference_gate.blocked
    assert call(display, '/reference-gate', body) == first
