"""离线验证审核过的引导动力学及生成的约束逻辑，不与机器人通信。"""
import math
import socket
import textwrap

import pytest

from leica_tracker.physical_resident_engine import Installation, ResidentEngine
from test_physical_resident_engine import Gate, feedback, reviewed


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Guidance configuration tests must not open a socket')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)


def installation(**fields):
    candidate = reviewed()
    candidate.update(fields)
    return Installation(candidate)


def emitted_guard_fault(cfg, *, speed=None, relative=None, joints=None, offset=None):
    """评估生成的故障谓词，其语法与Python共享。只评估直线保护链；
    这不是URScript运动模拟器，不会执行驻留或任何控制器操作。"""
    script = cfg.program(123)
    predicates = script.split('    fault = 0\n', 1)[1].split('    i = 0\n', 1)[0]
    predicates = predicates.removesuffix('    end\n')
    namespace = dict(
        actual=list(cfg.neutral), actual_speed=speed or [0.] * 6,
        relative=relative or [0.] * 6, q=joints or list(cfg.joints), qd=[0.] * 6,
        offset=offset or list(cfg.tcp), expected_tcp=list(cfg.tcp), fault=0,
        norm=lambda value: abs(value) if isinstance(value, (int, float)) else math.hypot(*value),
        in_box=lambda pose: all(cfg.lower[i] <= pose[i] <= cfg.upper[i] for i in range(3)),
    )
    exec(compile(textwrap.dedent(predicates), '<emitted guidance guards>', 'exec'), namespace)
    return namespace['fault']


@pytest.mark.parametrize('speed,acceleration', [(.006, .02), (.05, .1), (.4, .8)])
def test_explicit_reviewed_guidance_dynamics_are_not_subject_to_old_trial_caps(speed, acceleration):
    cfg = installation(speed=speed, acceleration=acceleration)
    assert cfg.speed == speed and cfg.acceleration == acceleration
    assert f'a={acceleration:.10f}, t=0.02)' in cfg.program(123)


@pytest.mark.parametrize('field', ['speed', 'acceleration'])
@pytest.mark.parametrize('value', [None, 0., -1., False, True, '0.1', math.nan, math.inf, -math.inf])
def test_guidance_dynamics_still_require_finite_positive_explicit_values(field, value):
    with pytest.raises(ValueError, match=field):
        installation(**{field: value})


@pytest.mark.parametrize('field', ['speed', 'acceleration'])
def test_guidance_dynamics_cannot_be_omitted(field):
    candidate = reviewed()
    del candidate[field]
    with pytest.raises(ValueError, match=field):
        Installation(candidate)


@pytest.mark.parametrize('fields', [
    {'speed': 1.7e308}, {'acceleration': 1e200}, {'acceleration': 1e308},
])
def test_finite_dynamics_with_unrepresentable_resident_arithmetic_are_rejected(fields):
    with pytest.raises(ValueError, match='unrepresentable'):
        installation(**fields)


@pytest.mark.parametrize('field', ['max_tcp_linear_speed', 'max_tcp_angular_speed'])
@pytest.mark.parametrize('value', [0., -1., False, True, '0.1', math.nan, math.inf, -math.inf])
def test_monitor_only_accepts_null_or_finite_positive_numeric_setting(field, value):
    with pytest.raises(ValueError, match=field):
        installation(**{field: value})


@pytest.mark.parametrize('fields,linear,angular', [
    ({}, .007, .01),
    ({'max_tcp_linear_speed': .2, 'max_tcp_angular_speed': .6}, .2, .6),
])
def test_host_and_emitted_monitors_use_the_same_legacy_or_reviewed_thresholds(fields, linear, angular):
    cfg = installation(**fields)
    for index, threshold, fault in [(0, linear, 5), (3, angular, 7)]:
        sample = feedback(cfg, 123)
        sample['tcp_speed'][index] = threshold * .99
        cfg.check_feedback(sample)
        assert emitted_guard_fault(cfg, speed=sample['tcp_speed']) == 0
        sample['tcp_speed'][index] = threshold
        with pytest.raises(ValueError, match='TCP speed'):
            cfg.check_feedback(sample)
        assert emitted_guard_fault(cfg, speed=sample['tcp_speed']) == fault


@pytest.mark.parametrize('disabled,index,retained_index,retained_threshold,fault', [
    ('max_tcp_linear_speed', 0, 3, .01, 7),
    ('max_tcp_angular_speed', 3, 0, .007, 5),
])
def test_null_disables_only_its_corresponding_host_and_robot_speed_comparison(
        disabled, index, retained_index, retained_threshold, fault):
    cfg = installation(**{disabled: None})
    sample = feedback(cfg, 123)
    sample['tcp_speed'][index] = 2.
    cfg.check_feedback(sample)
    assert emitted_guard_fault(cfg, speed=sample['tcp_speed']) == 0
    sample['tcp_speed'][retained_index] = retained_threshold
    with pytest.raises(ValueError, match='TCP speed'):
        cfg.check_feedback(sample)
    assert emitted_guard_fault(cfg, speed=sample['tcp_speed']) == fault


def test_disabled_speed_monitors_preserve_later_elif_guards_and_orientation_limit():
    cfg = installation(max_tcp_linear_speed=None, max_tcp_angular_speed=None)
    sample = feedback(cfg, 123)
    sample['tcp_speed'] = [2.] * 6
    cfg.check_feedback(sample)
    assert emitted_guard_fault(cfg, speed=sample['tcp_speed']) == 0
    assert emitted_guard_fault(cfg, speed=sample['tcp_speed'], relative=[0., 0., 0., .01, 0., 0.]) == 3
    changed_offset = list(cfg.tcp)
    changed_offset[2] += .001
    assert emitted_guard_fault(cfg, speed=sample['tcp_speed'], offset=changed_offset) == 4
    moved_base = list(cfg.joints)
    moved_base[0] += .1
    assert emitted_guard_fault(cfg, speed=sample['tcp_speed'], joints=moved_base) == 8
    sample['tcp_pose'][3:] = [0., 3.16, 0.]
    with pytest.raises(ValueError, match='orientation'):
        cfg.check_feedback(sample)


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, False])
def test_disabled_monitors_still_reject_invalid_telemetry(value):
    cfg = installation(max_tcp_linear_speed=None, max_tcp_angular_speed=None)
    sample = feedback(cfg, 123)
    sample['tcp_speed'][3] = value
    with pytest.raises(ValueError, match='TCP speed'):
        cfg.check_feedback(sample)


def test_disabled_monitors_keep_reviewed_position_bounds_and_target_braking_margin():
    cfg = installation(max_tcp_linear_speed=None, max_tcp_angular_speed=None)
    sample = feedback(cfg, 123, xyz=[.121, -.5, .3])
    with pytest.raises(ValueError, match='envelope'):
        cfg.check_feedback(sample)
    with pytest.raises(ValueError, match='braking margin'):
        cfg.check_target([.118, -.5, .3])


@pytest.mark.parametrize('index,speed', [(0, .0006), (3, .003)])
def test_disabled_monitors_still_require_stationary_arrival_before_completion(index, speed):
    cfg = installation(max_tcp_linear_speed=None, max_tcp_angular_speed=None)
    gate = Gate()
    engine = ResidentEngine(cfg, gate, 123)
    sample = feedback(cfg, 123)
    engine.tick(sample, sampled_at=1., now=1.)
    sample.update(state=0, sequence=1)
    engine.tick(sample, sampled_at=1.03, now=1.03)
    engine.target([.105, -.5, .3], now=1.03)
    sample.update(state=0, sequence=2)
    engine.tick(sample, sampled_at=1.06, now=1.06)
    sample['tcp_pose'][:3] = [.105, -.5, .3]
    sample['tcp_speed'][index] = speed
    for i in range(1, 18):
        now = 1.06 + i * .05
        sample.update(state=1, sequence=gate.calls[-1][0])
        engine.tick(sample, sampled_at=now, now=now)
    assert engine.state == 'MOVING' and engine.completed == 0
    assert engine.steady_since is None and not gate.stopped


def test_disabled_monitors_keep_stationary_reference_freshness_and_stop_latches():
    cfg = installation(max_tcp_linear_speed=None, max_tcp_angular_speed=None)
    gate = Gate()
    engine = ResidentEngine(cfg, gate, 123)
    sample = feedback(cfg, 123)
    sample['tcp_speed'][3] = .003
    with pytest.raises(RuntimeError, match='stationary reference'):
        engine.tick(sample, sampled_at=1., now=1.)
    assert engine.state == 'FAULT' and gate.stopped and gate.calls == []

    gate = Gate()
    engine = ResidentEngine(cfg, gate, 123)
    sample = feedback(cfg, 123)
    engine.tick(sample, sampled_at=1., now=1.)
    with pytest.raises(RuntimeError, match='Stale or repeated'):
        engine.tick(sample, sampled_at=1., now=1.3)
    assert gate.stopped and len(gate.calls) == 1

    gate = Gate()
    engine = ResidentEngine(cfg, gate, 123)
    engine.stop()
    with pytest.raises(RuntimeError, match='closed'):
        engine.tick(sample, sampled_at=1., now=1.)
    assert engine.state == 'STOPPED' and gate.stopped and gate.calls == []
