"""编排的指令可以在不传输或重新播放动作的情况下过期。"""
import pytest

from leica_tracker.physical_resident_engine import Installation, ResidentEngine
from test_physical_resident_engine import Gate, feedback, reviewed


class MotionGate(Gate):
    is_paused = False

    @property
    def can_send(self):
        return not self.stopped


def staged(*, gesture=False, gate=None):
    data = reviewed()
    if gesture:
        data['gesture'] = dict(duration_s=3.3, repetitions=2, joint_speed_limit=.7)
    cfg = Installation(data)
    gate = gate or MotionGate()
    engine = ResidentEngine(cfg, gate, 123)
    engine.tick(feedback(cfg, 123), sampled_at=1., now=1.)
    engine.tick(feedback(cfg, 123, state=0, seq=1), sampled_at=1.03, now=1.03)
    (engine.gesture if gesture else engine.target)([.105, -.5, .3], now=1.04)
    return cfg, gate, engine


@pytest.mark.parametrize('gesture', [False, True])
def test_expired_staged_direction_keeps_hold_ack_and_never_sends_motion(gesture):
    cfg, gate, engine = staged(gesture=gesture)
    pending = engine.pending
    sequence, last_send, last_sample = engine.sequence, engine.last_send, engine.last_sample
    engine.cancel_before_enable()
    assert engine.state == 'READY' and engine.goal is None and engine.gesture_endpoint is None
    assert engine.steady_since is None and not engine.motion_acked and not engine.motion_sent
    assert engine.pending == pending
    assert (engine.sequence, engine.last_send, engine.last_sample) == (sequence, last_send, last_sample)
    engine.tick(feedback(cfg, 123, state=0, seq=pending[0]), sampled_at=1.06, now=1.06)
    assert not gate.stopped and engine.completed == 0
    assert all(call[1] == 0 for call in gate.calls)
    engine.target([.105, -.5, .3], now=1.07)
    assert not engine.motion_sent
    engine.tick(feedback(cfg, 123, state=0, seq=engine.pending[0]), sampled_at=1.09, now=1.09)
    assert engine.motion_sent and gate.calls[-1][1] == 1


def test_cancellation_preserves_original_hold_ack_deadline():
    cfg, gate, engine = staged()
    pending = engine.pending
    engine.cancel_before_enable()
    with pytest.raises(RuntimeError, match='acknowledgement timeout'):
        engine.tick(feedback(cfg, 123, state=0, seq=pending[0]), sampled_at=1.24, now=1.24)
    assert gate.stopped and engine.state == 'FAULT'
    assert all(call[1] == 0 for call in gate.calls)


def test_network_pause_ack_is_not_an_enabled_motion_attempt():
    cfg, gate, engine = staged(gesture=True)
    gate.is_paused = True
    engine.tick(feedback(cfg, 123, state=0, seq=2), sampled_at=1.06, now=1.06)
    engine.tick(feedback(cfg, 123, state=4, seq=3), sampled_at=1.09, now=1.09)
    assert engine.motion_acked and not engine.motion_sent
    pending = engine.pending
    assert pending[2] == 3
    engine.cancel_before_enable()
    assert engine.pending == pending
    gate.is_paused = False
    engine.tick(feedback(cfg, 123, state=4, seq=4), sampled_at=1.12, now=1.12)
    engine.tick(feedback(cfg, 123, state=0, seq=5), sampled_at=1.15, now=1.15)
    assert engine.state == 'READY' and not engine.motion_sent and not gate.stopped
    assert all(call[1] in (0, 3) for call in gate.calls)


@pytest.mark.parametrize('gesture', [False, True])
def test_cancellation_is_terminal_after_motion_packet_attempt(gesture):
    cfg, gate, engine = staged(gesture=gesture)
    engine.tick(feedback(cfg, 123, state=0, seq=2), sampled_at=1.06, now=1.06)
    assert engine.motion_sent
    before = len(gate.calls)
    with pytest.raises(RuntimeError, match='cancel'):
        engine.cancel_before_enable()
    assert engine.state == 'FAULT' and gate.stopped and len(gate.calls) == before


def test_uncertain_send_latches_motion_before_transport_exception():
    class UncertainGate(MotionGate):
        def send(self, *args):
            super().send(*args)
            if args[1] in (1, 2):
                raise RuntimeError('transport failed after write')

    cfg, gate, engine = staged(gate=UncertainGate())
    with pytest.raises(RuntimeError, match='transport failed'):
        engine.tick(feedback(cfg, 123, state=0, seq=2), sampled_at=1.06, now=1.06)
    assert engine.motion_sent and engine.state == 'FAULT'
    before = len(gate.calls)
    with pytest.raises(RuntimeError, match='closed'):
        engine.cancel_before_enable()
    assert engine.state == 'FAULT' and len(gate.calls) == before


def test_motion_history_cannot_be_hidden_by_a_later_native_pause_packet():
    cfg, gate, engine = staged(gesture=True)
    engine.tick(feedback(cfg, 123, state=0, seq=2), sampled_at=1.06, now=1.06)
    gate.is_paused = True
    engine.tick(feedback(cfg, 123, state=1, seq=3), sampled_at=1.09, now=1.09)
    assert engine.motion_sent and engine.pending[2] == 3
    with pytest.raises(RuntimeError, match='cancel'):
        engine.cancel_before_enable()
    assert engine.state == 'FAULT' and gate.stopped


def test_cancellation_cannot_reopen_explicitly_stopped_engine():
    _, gate, engine = staged()
    engine.stop()
    before = len(gate.calls)
    with pytest.raises(RuntimeError, match='closed'):
        engine.cancel_before_enable()
    assert engine.state == 'STOPPED' and gate.stopped and len(gate.calls) == before


def test_cancellation_cannot_reopen_independently_closed_gate():
    _, gate, engine = staged()
    gate.stop('explicit Stop')
    with pytest.raises(RuntimeError, match='closed'):
        engine.cancel_before_enable()
    assert engine.state == 'FAULT' and gate.stopped
