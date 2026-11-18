"""没有机器人I/O：练习居民配置、心跳和目标确认。"""
import pytest

from leica_tracker.physical_resident_engine import Installation, ResidentEngine


def reviewed():
    return dict(reference=[.1, -.5, .3, 0., 3.14, 0.],
                joints=[0., -1., -1., -1., 1., 0.], tcp=[0., 0., .2786, 0., 0., 0.],
                lower=[.08, -.52, .28], upper=[.12, -.48, .32],
                speed=.005, acceleration=.01, base_limit=.05, base_rate=.05,
                review=dict(reference=True, tcp=True, payload=True, path=True,
                            dynamics=True, direction=True, stop=True))


class Gate:
    def __init__(self):
        self.calls = []
        self.stopped = False

    def send(self, *args):
        if self.stopped:
            raise RuntimeError('stopped')
        self.calls.append(args)

    def stop(self, reason):
        self.stopped = True


def feedback(installation, session, *, state=2, seq=0, xyz=None):
    pose = list(installation.neutral)
    if xyz is not None:
        pose[:3] = xyz
    return dict(session=session, state=state, sequence=seq, tcp_pose=pose,
                tcp_speed=[0.] * 6, joints=list(installation.joints))


def test_draft_profiles_and_unbounded_values_are_rejected():
    p = reviewed(); p['review']['path'] = False
    with pytest.raises(ValueError, match='review'): Installation(p)
    p = reviewed(); p['lower'][0] = float('-inf')
    with pytest.raises(ValueError): Installation(p)
    p = reviewed(); p['lower'][0] = p['reference'][0]
    with pytest.raises(ValueError): Installation(p)
    for name, value in [('speed', 0.), ('acceleration', float('inf'))]:
        p = reviewed(); p[name] = value
        with pytest.raises(ValueError): Installation(p)


def test_reviewed_installation_is_immutable_after_validation():
    cfg=Installation(reviewed())
    assert cfg.robot_model=='ur10e'
    with pytest.raises(AttributeError):cfg.speed=.15
    with pytest.raises(AttributeError):cfg.acceleration=.5
    with pytest.raises(AttributeError):cfg.lower=(0.,0.,0.)
    with pytest.raises(AttributeError):del cfg.joints
    assert not hasattr(cfg,'__dict__')
    assert cfg.speed==.005 and cfg.acceleration==.01


def test_program_reuses_existing_watchdog_and_preserves_angular_guard():
    cfg = Installation(reviewed()); script = cfg.program(123)
    assert 'packet_age > 0.25' in script
    assert 'norm([actual_speed[3],actual_speed[4],actual_speed[5]]) < 0.0100000000' in script
    assert 'norm([actual_speed[0],actual_speed[1],actual_speed[2]]) < 0.0070000000' in script
    assert 'norm([relative[3],relative[4],relative[5]]) < 0.0100000000' in script
    assert 'speedl([v[0],v[1],v[2],0,0,0]' in script
    assert 'movej(' not in script and 'movel(' not in script
    assert 'set_tool' not in script and 'grip' not in script.lower()
    assert 'ever_armed = False' in script and 'ever_armed = True' in script
    assert 'if ever_armed:' in script and 'write_output_integer_register(42, 2)' in script
    assert 'global startup_age = 0.0' in script
    assert 'if not ever_armed and startup_age > 3.0:' in script


def test_hold_handshake_precedes_any_enabled_target_and_keeps_heartbeat():
    cfg = Installation(reviewed()); gate = Gate(); engine = ResidentEngine(cfg, gate, 123)
    sample = feedback(cfg, 123)
    engine.tick(sample, sampled_at=1., now=1.)
    assert gate.calls[-1] == (1, 0, 123, list(cfg.neutral[:3]))
    sample.update(state=0, sequence=1)
    engine.tick(sample, sampled_at=1.03, now=1.03)
    assert engine.state == 'READY'
    engine.target([.105, -.5, .3], now=1.03)
    sample.update(state=0, sequence=2)
    engine.tick(sample, sampled_at=1.06, now=1.06)
    assert gate.calls[-1] == (3, 1, 123, [.105, -.5, .3])
    sample.update(state=1, sequence=3)
    engine.tick(sample, sampled_at=1.09, now=1.09)
    assert gate.calls[-1][0:3] == (4, 1, 123)


def test_feedback_loss_or_guard_fault_latches_stop_no_return():
    cfg = Installation(reviewed()); gate = Gate(); engine = ResidentEngine(cfg, gate, 123)
    sample = feedback(cfg, 123); engine.tick(sample, sampled_at=1., now=1.)
    before = len(gate.calls)
    with pytest.raises(RuntimeError): engine.tick(sample, sampled_at=1., now=1.3)
    assert gate.stopped and engine.state == 'FAULT' and len(gate.calls) == before
    with pytest.raises(RuntimeError): engine.target(list(cfg.neutral[:3]), now=1.3)


def test_invalid_targets_never_send_and_stop_prevents_future_packets():
    cfg = Installation(reviewed()); gate = Gate(); engine = ResidentEngine(cfg, gate, 123)
    sample = feedback(cfg, 123); engine.tick(sample, sampled_at=1., now=1.)
    sample.update(state=0, sequence=1); engine.tick(sample, sampled_at=1.03, now=1.03)
    with pytest.raises(ValueError): engine.target([.2, -.5, .3], now=1.03)
    engine.stop(); before = len(gate.calls)
    with pytest.raises(RuntimeError): engine.tick(sample, sampled_at=1.06, now=1.06)
    assert gate.stopped and len(gate.calls) == before


def test_arrival_requires_continuous_stillness_then_held_ack():
    cfg = Installation(reviewed()); gate = Gate(); engine = ResidentEngine(cfg, gate, 123)
    sample = feedback(cfg, 123); engine.tick(sample, sampled_at=1., now=1.)
    sample.update(state=0, sequence=1); engine.tick(sample, sampled_at=1.03, now=1.03)
    engine.target([.105, -.5, .3], now=1.03)
    sample.update(state=0, sequence=2); engine.tick(sample, sampled_at=1.06, now=1.06)
    sample['tcp_pose'][:3] = [.105, -.5, .3]
    for i in range(1, 18):
        sample.update(state=1, sequence=gate.calls[-1][0])
        now = 1.06 + i * .05
        engine.tick(sample, sampled_at=now, now=now)
        if engine.state == 'HOLDING': break
    assert engine.state == 'HOLDING' and gate.calls[-1][1] == 0
    assert engine.completed == 0
    sample.update(state=0, sequence=gate.calls[-1][0])
    now += .05
    engine.tick(sample, sampled_at=now, now=now)
    assert engine.completed == 1 and engine.state == 'READY'
    # 发动机不会无声地命令返回旧位置。
    assert all(call[3] != list(cfg.neutral[:3]) for call in gate.calls[2:])


def test_unacknowledged_intermediate_movement_resets_dwell():
    cfg = Installation(reviewed()); gate = Gate(); engine = ResidentEngine(cfg, gate, 123)
    sample = feedback(cfg, 123); engine.tick(sample, sampled_at=1., now=1.)
    sample.update(state=0, sequence=1); engine.tick(sample, sampled_at=1.03, now=1.03)
    engine.target([.105, -.5, .3], now=1.03)
    sample.update(state=0, sequence=2); engine.tick(sample, sampled_at=1.06, now=1.06)
    sample.update(state=1, sequence=3); sample['tcp_pose'][:3] = [.105, -.5, .3]
    engine.tick(sample, sampled_at=1.09, now=1.09)
    now = 1.09
    for _ in range(14):
        now = round(now + .02, 4)
        sample['tcp_pose'][0] = .104; sample['tcp_speed'][0] = .004
        engine.tick(sample, sampled_at=now, now=now)
        now = round(now + .03, 4)
        sample['tcp_pose'][0] = .105; sample['tcp_speed'][0] = 0.
        sample['sequence'] = gate.calls[-1][0]
        engine.tick(sample, sampled_at=now, now=now)
    assert engine.state == 'MOVING' and engine.completed == 0


def test_gap_in_feedback_resets_continuous_stationary_observation():
    cfg = Installation(reviewed()); gate = Gate(); engine = ResidentEngine(cfg, gate, 123)
    sample = feedback(cfg, 123); engine.tick(sample, sampled_at=1., now=1.)
    sample.update(state=0, sequence=1); engine.tick(sample, sampled_at=1.03, now=1.03)
    engine.target([.105, -.5, .3], now=1.03)
    sample.update(state=0, sequence=2); engine.tick(sample, sampled_at=1.06, now=1.06)
    sample.update(state=1, sequence=3); sample['tcp_pose'][:3] = [.105, -.5, .3]
    engine.tick(sample, sampled_at=1.09, now=1.09)
    sample['sequence'] = gate.calls[-1][0]
    engine.tick(sample, sampled_at=1.25, now=1.25)
    assert engine.steady_since == 1.25
