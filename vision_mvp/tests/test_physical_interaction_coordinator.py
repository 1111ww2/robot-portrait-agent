"""纯粹的假输入测试；永远不要连接到机器人或生成机器人程序。"""
import pytest
import math

from leica_tracker.physical_interaction_coordinator import PhysicalInteractionCoordinator


REFERENCE = [-.1, -.5, .2, 0., 3., -.5]
ZERO = [0.] * 6
CONTROL = {'instance': 'fake-display', 'epoch': 3}
OWNER = {'owner': 'fake-owner', 'epoch': 3, 'active': True}


def start():
    c = PhysicalInteractionCoordinator(hold_timeout=2., motion_timeout=2.)
    c.start(REFERENCE, ZERO, sampled_at=10., control=CONTROL, active_owner=OWNER, now=10.)
    return c


def return_to_reference(c, start_at=10.1):
    c.motion_finished(success=True, now=start_at)
    for i in range(8):
        t = start_at + (i + 1) * .1
        c.observe_control(CONTROL, active_owner=OWNER, now=t)
        c.observe_telemetry(REFERENCE, ZERO, sampled_at=t, now=t)
    c.release(now=start_at + .81)
    return start_at + .81


def test_opening_blocks_images_until_verified_return_and_new_frame():
    c = start()
    assert not c.accept_frame(10_000_000_001, now=10.01)
    with pytest.raises(ValueError): c.intent(now=10.01)
    c.acknowledge_hold(now=10.02)
    assert c.intent(now=10.02).cue == 'OPENING'
    t = return_to_reference(c)
    assert not c.accept_frame(round((t - .01) * 1e9), now=t + .01)
    assert c.accept_frame(round((t + .02) * 1e9), now=t + .03)
    assert c.state == 'OBSERVING'


def test_direction_requires_post_return_frame_and_again_blocks_during_motion():
    c = start(); c.acknowledge_hold(now=10.02); c.intent(now=10.02)
    t = return_to_reference(c)
    with pytest.raises(ValueError): c.request_direction('MOVE_LEFT', now=t + .01)
    assert c.accept_frame(round((t + .02) * 1e9), now=t + .03)
    c.request_direction('MOVE_LEFT', now=t + .04)
    assert not c.accept_frame(round((t + .05) * 1e9), now=t + .06)
    c.acknowledge_hold(now=t + .07)
    assert c.intent(now=t + .08).cue == 'MOVE_LEFT'


def test_cannot_release_with_only_pre_completion_stationary_samples():
    c = start(); c.acknowledge_hold(now=10.01); c.intent(now=10.01)
    for i in range(8):
        t = 10.1 + i * .1
        c.observe_control(CONTROL, active_owner=OWNER, now=t)
        c.observe_telemetry(REFERENCE, ZERO, sampled_at=t, now=t)
    c.motion_finished(success=True, now=10.81)
    with pytest.raises(ValueError): c.release(now=10.82)
    assert c.state == 'SETTLING'


@pytest.mark.parametrize('bad_sample', ['stale', 'wrong_pose', 'moving'])
def test_bad_or_insufficient_return_never_unblocks_camera(bad_sample):
    c = start(); c.acknowledge_hold(now=10.01); c.intent(now=10.01)
    c.motion_finished(success=True, now=10.02)
    pose = REFERENCE.copy(); speed = ZERO.copy(); sampled_at = 10.1
    if bad_sample == 'stale': sampled_at = 9.
    if bad_sample == 'wrong_pose': pose[0] += .01
    if bad_sample == 'moving': speed[0] = .01
    if bad_sample == 'stale':
        with pytest.raises(ValueError): c.observe_telemetry(pose, speed, sampled_at=sampled_at, now=10.1)
    else:
        c.observe_telemetry(pose, speed, sampled_at=sampled_at, now=10.1)
    with pytest.raises((ValueError, RuntimeError)): c.release(now=10.11)
    assert not c.accept_frame(10_120_000_000, now=10.12)


def test_stop_failure_and_control_loss_latch_without_automatic_retries():
    for failure in ('stop', 'motion_failure', 'heartbeat'):
        c = start(); c.acknowledge_hold(now=10.01); c.intent(now=10.01)
        if failure == 'stop': c.stop()
        elif failure == 'motion_failure': c.motion_finished(success=False, now=10.02)
        else:
            with pytest.raises(RuntimeError): c.watchdog(now=10.36)
        assert not c.accept_frame(10_400_000_000, now=10.4)
        with pytest.raises((RuntimeError, ValueError)): c.intent(now=10.4)
        with pytest.raises((RuntimeError, ValueError)): c.start(REFERENCE, ZERO, sampled_at=10.4, control=CONTROL, active_owner=OWNER, now=10.4)


def test_fresh_frame_and_telemetry_are_required_before_next_action():
    c = start(); c.acknowledge_hold(now=10.01); c.intent(now=10.01)
    t = return_to_reference(c)
    assert c.accept_frame(round((t + .02) * 1e9), now=t + .03)
    with pytest.raises(RuntimeError): c.request_direction('MOVE_RIGHT', now=t + .4)
    assert c.state == 'FAULT'


def test_no_valid_reference_or_control_cannot_start():
    c = PhysicalInteractionCoordinator(hold_timeout=2., motion_timeout=2.)
    with pytest.raises(ValueError): c.start(REFERENCE, ZERO, sampled_at=9., control=CONTROL, active_owner=OWNER, now=10.)
    assert not c.accept_frame(10_100_000_000, now=10.1)


def test_unknown_motion_result_and_explicit_disconnect_latch():
    for failure in ('unknown', 'disconnect'):
        c = start(); c.acknowledge_hold(now=10.01); c.intent(now=10.01)
        if failure == 'unknown':
            with pytest.raises(ValueError): c.motion_finished(success=None, now=10.02)
        else:
            c.source_lost()
        assert c.state == 'FAULT'
        with pytest.raises(RuntimeError): c.motion_finished(success=True, now=10.03)
        assert not c.accept_frame(10_040_000_000, now=10.04)


def test_guest_restart_latches_even_if_robot_still_reports_good_data():
    c = start(); c.acknowledge_hold(now=10.01); c.intent(now=10.01)
    with pytest.raises(RuntimeError):
        c.observe_control({'instance': 'fake-display', 'epoch': 4}, active_owner=OWNER, now=10.02)
    with pytest.raises(RuntimeError):
        c.observe_telemetry(REFERENCE, ZERO, sampled_at=10.03, now=10.03)
    assert c.state == 'FAULT'


def test_stale_camera_latches_even_with_fresh_control_and_robot():
    c = start(); c.acknowledge_hold(now=10.01); c.intent(now=10.01)
    t = return_to_reference(c)
    assert c.accept_frame(round((t + .02) * 1e9), now=t + .03)
    for offset in (.1, .2, .3, .4):
        now = t + offset
        c.observe_control(CONTROL, active_owner=OWNER, now=now)
        c.observe_telemetry(REFERENCE, ZERO, sampled_at=now, now=now)
    with pytest.raises(RuntimeError): c.request_direction('MOVE_RIGHT', now=t + .4)
    assert c.state == 'FAULT'


def test_intent_can_only_be_consumed_once():
    c = start(); c.acknowledge_hold(now=10.01)
    assert c.intent(now=10.01).cue == 'OPENING'
    with pytest.raises(ValueError): c.intent(now=10.02)


def test_buffered_pre_completion_sample_fails_closed():
    c = start(); c.acknowledge_hold(now=10.01); c.intent(now=10.01)
    c.motion_finished(success=True, now=10.02)
    # 它在150毫秒的保质期窗口内，但是在完成之前被捕获的。
    with pytest.raises(ValueError):
        c.observe_telemetry(REFERENCE, ZERO, sampled_at=10.015, now=10.03)
    assert c.state == 'FAULT'
    with pytest.raises(RuntimeError): c.release(now=10.04)


@pytest.mark.parametrize('owner', [None, {}, {'owner': 'x', 'epoch': 3, 'active': False},
                                    {'owner': 'x', 'epoch': 2, 'active': True}])
def test_stopped_missing_or_wrong_owner_cannot_start(owner):
    c = PhysicalInteractionCoordinator(hold_timeout=1, motion_timeout=1)
    with pytest.raises(RuntimeError):
        c.start(REFERENCE, ZERO, sampled_at=10, control=CONTROL, active_owner=owner, now=10)
    assert c.state == 'FAULT'


def test_owner_stop_latches_even_with_valid_heartbeat_identity():
    c = start()
    with pytest.raises(RuntimeError):
        c.observe_control(CONTROL, active_owner={**OWNER, 'active': False}, now=10.1)
    assert c.state == 'FAULT'


@pytest.mark.parametrize('phase', ['hold', 'motion'])
def test_phase_timeout_latches_despite_fresh_heartbeat_and_telemetry(phase):
    c = PhysicalInteractionCoordinator(hold_timeout=.25, motion_timeout=.25)
    c.start(REFERENCE, ZERO, sampled_at=10., control=CONTROL, active_owner=OWNER, now=10.)
    if phase == 'motion':
        c.acknowledge_hold(now=10.)
        c.intent(now=10.)
    for t in (10.1, 10.2):
        c.observe_control(CONTROL, active_owner=OWNER, now=t)
        c.observe_telemetry(REFERENCE, ZERO, sampled_at=t, now=t)
    with pytest.raises(RuntimeError):
        c.observe_control(CONTROL, active_owner=OWNER, now=10.3)
    assert c.state == 'FAULT'


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf'), True])
def test_timeouts_must_be_explicit_positive_finite_numbers(timeout):
    with pytest.raises(ValueError):
        PhysicalInteractionCoordinator(hold_timeout=timeout, motion_timeout=1)


def test_observing_accepts_equivalent_rotation_vector_representation():
    c = start(); c.acknowledge_hold(now=10.01); c.intent(now=10.01)
    t = return_to_reference(c)
    c.observe_control(CONTROL, active_owner=OWNER, now=t + .05)
    pose = REFERENCE.copy()
    magnitude = math.hypot(*pose[3:])
    pose[3:] = [x * (magnitude + 2 * math.pi) / magnitude for x in pose[3:]]
    c.observe_telemetry(pose, ZERO, sampled_at=t + .05, now=t + .05)
    assert c.state == 'WAIT_FRAME'
