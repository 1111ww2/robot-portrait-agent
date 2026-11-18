"""离线检查：过期的指示永远不会启用运动或重新启动一次遭遇。"""
import pytest

from leica_tracker.reference_gate import ReferenceGate
from test_physical_interaction_supervisor import setup, ready, event, step, Immediate


class MeasuredReference:
    identity = {'instance': 'guest-a', 'epoch': 1}
    started_ns = 1

    def __init__(self):
        self.clock = 1.
        self.gate = ReferenceGate()
        self.ack = False
        self.samples = []
        self.releases = []
        self.faults = 0

    def begin(self, reference):
        return self.gate.begin(reference, self.clock)

    def status(self, session):
        if self.ack:
            self.gate.acknowledge()
        return {'blocked': self.gate.blocked, 'session': self.gate.session,
                'acknowledged': self.gate.acknowledged}

    def sample(self, session, pose, speed, sampled_at):
        self.gate.sample(session, pose, speed, sampled_at, self.clock)
        self.samples.append(sampled_at)
        return {'ok': True, 'sampled_at': sampled_at,
                'return_verified': self.gate.return_verified(session, self.clock)}

    def release(self, session):
        self.gate.release(session, self.clock)
        self.releases.append(self.clock)
        return self.gate.release_ns

    def fault(self):
        self.faults += 1


def encounter():
    cfg, gate, engine, _, sup = setup()
    ref = MeasuredReference()
    sup.reference = ref
    class Completed(Immediate):
        def __init__(self, fn, *args):
            super().__init__(fn, *args)
            self.elapsed = 0.
    sup.worker_factory = Completed
    ready(sup, cfg, gate)
    return cfg, gate, engine, ref, sup


def advance(ctx, t, *, xyz=None):
    cfg, gate, _, ref, sup = ctx
    ref.clock = t
    step(sup, cfg, gate, t, xyz=xyz)


def expire(ctx, staged):
    _, _, engine, ref, sup = ctx
    assert sup.accept_event(event(950_000_000), now=1.06)
    advance(ctx, 1.10)
    ref.ack = True
    advance(ctx, 1.15)
    if staged:
        advance(ctx, 1.20)
        assert engine.state == 'MOVING'
    advance(ctx, 1.30 if staged else 1.26)
    assert sup.recovering_expiry and sup.state in ('SETTLING', 'SAMPLE_PENDING')
    return 1.30 if staged else 1.26


@pytest.mark.parametrize('staged', [False, True])
def test_expiry_holds_then_reobserves_without_a_new_start(staged):
    ctx = encounter()
    cfg, gate, engine, ref, sup = ctx
    t = expire(ctx, staged)
    for _ in range(32):
        t = round(t + .05, 6)
        advance(ctx, t)
        if sup.state == 'OBSERVING':
            break
    assert sup.state == 'OBSERVING' and not sup.recovering_expiry
    assert not gate.stopped and not ref.faults
    assert not ref.gate.blocked and len(ref.releases) == 1
    assert ref.samples[-1] - ref.samples[0] >= .6
    assert all(packet[1] == 0 for packet in gate.calls)
    assert all(packet[3] == list(cfg.neutral[:3]) for packet in gate.calls)
    release_ns = ref.gate.release_ns
    assert not sup.accept_event(event(release_ns), now=t)
    assert not sup.accept_event(event(release_ns - 1), now=t)
    # 一个新的观察，而不是废弃方向的时间戳重置。
    source_ns = round((t + .01) * 1e9)
    assert sup.accept_event(event(source_ns, cue='MOVE_RIGHT'), now=t + .02)
    for delta in (.05, .10, .15):
        advance(ctx, round(t + delta, 6))
    assert any(packet[1] == 1 for packet in gate.calls)
    assert engine.session == 123 and sup._control_identity == ref.identity
    assert sup._control_started_ns == 1


def test_expiry_requires_reference_and_a_new_continuous_standstill():
    ctx = encounter()
    cfg, gate, _, ref, sup = ctx
    t = expire(ctx, True)
    for i in range(1, 15):
        advance(ctx, round(t + i * .05, 6), xyz=[cfg.neutral[0] + .0006, *cfg.neutral[1:3]])
    assert ref.gate.blocked and not ref.releases
    for i in range(15, 24):
        advance(ctx, round(t + i * .05, 6))
    assert not ref.releases  # 比参考值回差不到600毫秒
    for i in range(24, 36):
        advance(ctx, round(t + i * .05, 6))
        if sup.state == 'OBSERVING':
            break
    assert sup.state == 'OBSERVING' and ref.releases
    assert all(packet[1] == 0 for packet in gate.calls)


def test_stop_during_expiry_recovery_remains_terminal():
    ctx = encounter()
    _, gate, _, ref, sup = ctx
    t = expire(ctx, True)
    count = len(gate.calls)
    sup.stop()
    with pytest.raises(RuntimeError, match='stopped'):
        advance(ctx, t + .05)
    assert gate.stopped and len(gate.calls) == count and not ref.releases


def test_reference_failure_during_expiry_recovery_remains_terminal():
    ctx = encounter()
    _, gate, _, ref, sup = ctx
    def reject(*args):
        raise ValueError('reference sample rejected')
    ref.sample = reject
    t = expire(ctx, True)
    with pytest.raises(RuntimeError, match='Reference operation failed'):
        advance(ctx, t + .05)
    assert gate.stopped and not ref.releases
    assert all(packet[1] == 0 for packet in gate.calls)
