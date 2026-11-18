"""音频录制跟随真实的原生运动；这些通道完全是假的。"""
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from test_physical_runtime import runtime
from test_runtime_status_display import Capture, request
from leica_tracker.guest_display import GuestDisplay
from leica_tracker.ipad_camera import IPadCapture


@pytest.mark.parametrize('native,sent,acked,velocity,expected', [
    (0, True, True, -.001, False),
    (1, False, True, -.001, False),
    (1, True, False, -.001, False),
    (1, True, True, .001, False),
    (1, True, True, -.001, True),
])
def test_audio_started_requires_enabled_acknowledged_outbound_motion(native, sent, acked, velocity, expected):
    run, session, sup, pub = runtime()
    run._start_light({'cue': 'MOVE_LEFT', 'source_monotonic_ns': 10_000_000_000})
    sup.state = 'MOVING'
    session.engine.motion_sent = sent
    session.engine.motion_acked = acked
    session.writer.latest = (dict(session.writer.latest[0], state=native,
        tcp_pose=[.099, -.5, .3, 0., 3.14, 0.], tcp_speed=[velocity, 0., 0., 0., 0., 0.]), 10.)
    run.step(None, now=10.)
    sample = [x for x in pub.calls if x.get('op') == 'sample'][-1]
    assert sample['motion_started'] is expected
    # 原生完成/返回不会重新触发或抹去此操作的凭证。
    session.writer.latest = (dict(session.writer.latest[0], state=0, tcp_speed=[.001, 0., 0., 0., 0., 0.]), 10.11)
    session.reader.latest = (session.reader.latest[0], 10.11)
    run.step(None, now=10.11)
    assert [x for x in pub.calls if x.get('op') == 'sample'][-1]['motion_started'] is expected
    run._start_light({'cue': 'MOVE_RIGHT', 'source_monotonic_ns': 10_110_000_000})
    assert run.action['motion_started'] is False


def test_public_audio_proof_is_boolean_latched_per_action_and_never_inferred_from_drift(monkeypatch):
    from leica_tracker import guest_display as display_module
    # 保持真正的HTTP处理，但确定性地推进测试夹具电测：Windows monotonic()在连续子时钟请求期间可能会重复。
    clock=SimpleNamespace(now=1000.)
    monkeypatch.setattr(display_module,'time',SimpleNamespace(monotonic=lambda:clock.now))
    with GuestDisplay(port=0, remote_capture=Capture(), physical_feedback=True) as d:
        control = d.control_snapshot()
        request(d, '/action', {'action': 'start', 'control_protocol': 1,
            'control': {k: control[k] for k in ('instance', 'epoch')}})
        identity = {k: d.control_snapshot()[k] for k in ('instance', 'epoch')}
        identity['lease'] = request(d, '/physical-feedback', {'op': 'register', **identity})['lease']
        clock.now+=.01
        source = round(clock.now*1e9)
        begin = {'op': 'begin', **identity, 'action_id': 1, 'source_ns': source, 'cue': 'MOVE_LEFT',
                 'reference_position': [0., 0., 0.], 'direction': [-1., 0., 0.], 'planned_displacement': .1}
        request(d, '/physical-feedback', begin)
        def sample(**changes):
            clock.now+=.01
            body = {'op': 'sample', **identity, 'action_id': 1, 'source_ns': source,
                    'cue': 'MOVE_LEFT', 'captured_at': clock.now, 'current_position': [-.001, 0., 0.]}
            body.update(changes)
            return request(d, '/physical-feedback', body)
        sample()
        with d._lock:
            assert d._current_motion()['progress'] > 0
            assert d._current_motion()['started'] is False
        for invalid in ('true', 1, None):
            with pytest.raises(HTTPError) as error:
                sample(motion_started=invalid)
            assert error.value.code == 400
        sample(motion_started=True)
        assert request(d, '/state')['motion']['started'] is True
        sample(motion_started=False, phase='returning')
        assert request(d, '/state')['motion']['started'] is True
        request(d, '/physical-feedback', {**begin, 'action_id': 2})
        sample(action_id=2)
        assert request(d, '/state')['motion']['started'] is False


def test_audio_diagnostics_are_bounded_numeric_without_transcripts():
    c = IPadCapture(read_timeout=0)
    s = c.begin('ipad')
    values = dict(direction_audio_enabled=True, direction_audio_unlocked=True,
                  direction_audio_playing=False, direction_audio_played=240,
                  direction_audio_errors=3, direction_audio_error=4)
    c.ticket('ipad', s, {**values, 'direction_audio_transcript': 'private'})
    assert all(c.diagnostics().get(k) == v for k, v in values.items())
    assert 'direction_audio_transcript' not in c.diagnostics()
    c.ticket('ipad', s, dict(direction_audio_enabled='yes', direction_audio_played=241,
                            direction_audio_error=5, direction_audio_errors=float('nan')))
    assert not any(k.startswith('direction_audio_') for k in c.diagnostics())
