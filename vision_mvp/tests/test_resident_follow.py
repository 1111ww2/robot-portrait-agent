import pytest
from leica_tracker.horizontal_follow import HorizontalLimits
from leica_tracker.resident_follow import StreamPolicy, resident_program, close_session, check_tcp_speed


CFG = HorizontalLimits((-.135, -.506, .683, 0., -2.85, 1.31), -1,range_m=.1)


def test_resident_program_uses_the_reviewed_configured_range_without_ten_cm_cap():
    cfg=HorizontalLimits(CFG.neutral,-1,range_m=.5)
    script=resident_program(cfg,(0,0,.2786,0,0,0),123)
    assert 'incoming_x >= -0.635000000 and incoming_x <= 0.365000000' in script
    assert 'packet_age > 0.25' in script
    assert 'is_within_safety_limits(probe)' in script
    policy=StreamPolicy(cfg,123)
    first=policy.next(10.,cfg.neutral,(1.,10.,4,0),feedback())
    move=policy.next(10.04,cfg.neutral,(1.,10.04,4,0),feedback(first[0]))
    assert move[3]==pytest.approx(cfg.neutral[0]-.497)


def test_measured_linear_and_angular_speeds_are_not_added_across_units():
    measured = [6.58407e-5, .00129568, -.00235085, .00678856, .000777155, .000244246]
    linear, angular = check_tcp_speed(measured)
    assert linear == pytest.approx(.0026850696, abs=1e-7)
    assert angular < .01
    with pytest.raises(RuntimeError, match='linear'):
        check_tcp_speed([.008,0,0,0,0,0])
    with pytest.raises(RuntimeError, match='angular'):
        check_tcp_speed([0,0,0,.02,0,0])


def feedback(sequence=0, state=0, session=123):
    return dict(sequence=sequence, state=state, session=session)


def test_initial_hold_handshake_then_signed_target_without_program_restart():
    policy = StreamPolicy(CFG, 123)
    sample = (.8, 10., 4, 0)
    first = policy.next(10., CFG.neutral, sample, feedback())
    assert first[1] == 0
    assert policy.next(10.01, CFG.neutral, sample, feedback()) is None
    second = policy.next(10.04, CFG.neutral, sample, feedback(first[0]))
    assert second[1] == 1
    assert second[3] == pytest.approx(-.215)


def test_loss_latch_sends_disabled_packet_even_after_recovery():
    policy = StreamPolicy(CFG, 123)
    first = policy.next(10, CFG.neutral, (.8,10,4,0), feedback())
    move = policy.next(10.04, CFG.neutral, (.8,10.04,4,0), feedback(first[0]))
    held = policy.next(10.08, CFG.neutral, (.8,10.08,4,1), feedback(move[0],1))
    assert held[1] == 0
    resumed = policy.next(10.12, CFG.neutral, (.8,10.12,4,1), feedback(held[0],0))
    assert resumed[1] == 1


def test_missing_echo_fails_even_during_hold_and_wrong_session_rejected():
    policy = StreamPolicy(CFG,123)
    policy.next(10, CFG.neutral, (None,10,0,0), feedback())
    with pytest.raises(RuntimeError, match='acknowledgement'):
        policy.next(10.3, CFG.neutral, (None,10.3,0,0), feedback())
    with pytest.raises(RuntimeError, match='session'):
        StreamPolicy(CFG,123).next(10, CFG.neutral, (None,10,0,0), feedback(session=7))


def test_robot_timeout_requires_disabled_rearm_and_soft_margin():
    policy = StreamPolicy(CFG,123)
    first = policy.next(10, CFG.neutral, (1,10,4,0), feedback(state=2))
    assert first[1] == 0
    move = policy.next(10.04, CFG.neutral, (1,10.04,4,0), feedback(first[0],0))
    assert move[3] == pytest.approx(CFG.neutral[0] - .097)
    with pytest.raises(RuntimeError, match='explicit restart'):
        policy.next(10.08, CFG.neutral, (1,10.08,4,0), feedback(move[0],2))


def test_late_matching_echo_cannot_rearm_after_a_host_pause():
    policy = StreamPolicy(CFG,123)
    first = policy.next(10, CFG.neutral, (.8,10,4,0), feedback())
    with pytest.raises(RuntimeError, match='acknowledgement'):
        policy.next(10.3, CFG.neutral, (.8,10.3,4,0), feedback(first[0],0))


def test_resident_program_keeps_hold_inside_loop_and_limits_motion():
    script = resident_program(CFG, (0,0,.2786,0,0,0),123)
    assert script.count('def leica_resident_horizontal():') == 1
    assert 'speedl([vx,0,0,0,0,0], a=0.01, t=0.02)' in script
    assert 'global packet_age = packet_age + get_steptime()' in script
    assert 'packet_age > 0.25' in script
    assert 'armed = False' in script
    assert 'read_input_integer_register(44)' in script
    # 新会话不继承上一会话的序号计数器。
    assert 'previous = 0' in script
    assert 'previous = read_input_integer_register' not in script
    assert 'movel(' not in script


def test_every_resource_closes_when_both_stop_transports_fail(monkeypatch):
    closed = []
    class Resource:
        def __init__(self, name):
            self.name = name
        def send(self, program):
            raise OSError('primary failed')
        def close(self):
            closed.append(self.name)
    def failed_fallback(*args):
        raise OSError('fallback failed')
    monkeypatch.setattr('leica_tracker.resident_follow.send_script', failed_fallback)
    resources = [Resource(name) for name in ['channel','rtde','vision','states','dashboard']]
    with pytest.raises(RuntimeError, match='STOP REQUEST FAILED'):
        close_session(True, 'host', *resources)
    assert set(closed) == {'channel','rtde','vision','states','dashboard'}


def test_library_startup_cannot_upload_after_robot_changes_state(monkeypatch, tmp_path):
    import json
    import time
    from leica_tracker import resident_follow as follower

    current = dict(mode=dict(connected=True, enabled=True, power=True, mode=7,
                            emergency=False, protective=False, paused=False, running=False),
                   tcp_offset=[0.]*6, pose=list(CFG.neutral))
    class States:
        failure = notice = None
        @property
        def latest(self):
            return current, time.monotonic()
        def close(self):
            pass
    class Vision:
        latest = (None, time.monotonic(), 0, 0)
        failure = None
        def close(self):
            pass
    class Dashboard:
        def close(self):
            pass
    class Library:
        def __init__(self, host):
            current['mode']['protective'] = True
        def close(self):
            pass
    class Commands:
        def __init__(self, host):
            pass
        def send(self, script):
            pytest.fail('Program uploaded after state changed during library startup')
        def close(self):
            pass

    monkeypatch.setattr(follower, 'check_dashboard', lambda *args, **kwargs: None)
    monkeypatch.setattr(follower, 'StateReceiver', lambda host: States())
    monkeypatch.setattr(follower, 'VisionReceiver', lambda camera: Vision())
    monkeypatch.setattr(follower, 'DashboardReceiver', lambda host: Dashboard())
    monkeypatch.setattr(follower, 'RTDEChannel', Library)
    monkeypatch.setattr(follower, 'CommandChannel', Commands)
    calibration = tmp_path / 'cal.json'
    calibration.write_text(json.dumps(dict(status='HORIZONTAL_ONLY_CONFIRMED', base_axis='X',
        neutral_tcp_m_rad=CFG.neutral, camera_right_sign=-1, range_m=.1, reviewed_range_m=.1,
        robot_ip='unused.invalid', camera=1, tcp_offset_m_rad=[0.]*6)))
    with pytest.raises(RuntimeError, match='state no longer permits'):
        follower.main(['--calibration', str(calibration), '--execute'])
