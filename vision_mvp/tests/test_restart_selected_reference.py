"""通过真实的本地回环事务，验证停止后为选定方案建立新参考位。"""
import pytest

from leica_tracker.physical_interaction_supervisor import ReferenceRequestError
from test_start_reference_rearm_api import (
    POSE, ZERO, advance, call, publish_intro_hold, setup,
)


def selected_begin(fixture):
    display, clock, sid, ref, _ = fixture
    assert ref.restart_begin(sid, reanchor=True)
    assert not display._reference_gate.acknowledged
    advance(fixture, clock.now + .01)
    publish_intro_hold(display, source_ns=round(clock.now * 1e9))
    return list(POSE[:1]) + [POSE[1] + .02] + POSE[2:]


def measured(fixture, pose, *, count=8):
    _, clock, sid, ref, _ = fixture
    began = clock.now
    result = None
    for index in range(count):
        advance(fixture, began + .01 + index * .1)
        result = ref.restart_sample(sid, pose, ZERO, clock.now)
    return result


def test_explicit_selected_start_commits_fresh_current_pose_without_motion(setup):
    display, clock, sid, ref, _ = setup
    old = display._reference_gate.reference
    pose = selected_begin(setup)
    assert display._reference_gate.reference == old
    assert measured(setup, pose)['return_verified']
    assert display._reference_gate.blocked and display._reference_gate.reference == old
    advance(setup, clock.now + .1)
    released = ref.restart_release(sid, pose, ZERO, clock.now)
    assert released == round(clock.now * 1e9)
    assert ref.restart_reference == tuple(pose)
    assert not display._reference_gate.blocked


def test_selected_start_cannot_reuse_old_ack_or_old_dwell(setup):
    display, clock, sid, ref, _ = setup
    assert ref.restart_begin(sid, reanchor=True)
    pose = list(POSE); pose[0] += .02
    assert not measured(setup, pose)['return_verified']
    advance(setup, clock.now + .1)
    with pytest.raises(ReferenceRequestError):
        ref.restart_release(sid, pose, ZERO, clock.now)
    assert display._reference_gate.blocked
    publish_intro_hold(display, source_ns=round(clock.now * 1e9))
    advance(setup, clock.now + .1)
    assert ref.restart_release(sid, pose, ZERO, clock.now) > 0


def test_fixed_reference_restart_still_refuses_a_displaced_stationary_pose(setup):
    display, clock, sid, ref, _ = setup
    assert ref.restart_begin(sid)
    pose = list(POSE); pose[0] += .02
    assert not measured(setup, pose)['return_verified']
    advance(setup, clock.now + .1)
    with pytest.raises(ReferenceRequestError) as caught:
        ref.restart_release(sid, pose, ZERO, clock.now)
    details = caught.value.safe_details()['reference_check']
    assert details['position_error_mm'] == pytest.approx(20.)
    assert details['status'] == 'position'
    assert display._reference_gate.blocked and display._reference_gate.reference == tuple(POSE)


@pytest.mark.parametrize('change', ['stale', 'repeated', 'moving', 'stop', 'epoch', 'lease'])
def test_invalid_selected_proof_cannot_commit_or_replace_the_old_reference(setup, change):
    display, clock, sid, ref, identity = setup
    pose = selected_begin(setup)
    measured(setup, pose)
    before = display._reference_gate.reference
    previous = clock.now
    advance(setup, clock.now + .1)
    speed = list(ZERO)
    stamp = clock.now
    if change == 'stale': stamp -= 1.
    if change == 'repeated': stamp = previous
    if change == 'moving': speed[1] = .01
    if change == 'stop': call(display, '/action', {'action': 'stop'})
    if change == 'epoch':
        call(display, '/action', {'action': 'restart', 'control_protocol': 1,
            'control': call(display, '/state')['control']})
    if change == 'lease': ref._lease = 'wrong-test-lease'
    with pytest.raises(ReferenceRequestError):
        ref.restart_release(sid, pose, speed, stamp)
    assert display._reference_gate.blocked and display._reference_gate.reference == before


def test_candidate_drift_and_sample_gap_require_a_new_full_dwell(setup):
    display, clock, sid, ref, _ = setup
    pose = selected_begin(setup)
    assert measured(setup, pose)['return_verified']
    pose[0] += .002
    advance(setup, clock.now + .2)
    result = ref.restart_sample(sid, pose, ZERO, clock.now)
    assert not result['return_verified']
    with pytest.raises(ReferenceRequestError): ref.restart_release(sid)
    assert measured(setup, pose)['return_verified']
    advance(setup, clock.now + .1)
    ref.restart_release(sid, pose, ZERO, clock.now)
    assert not display._reference_gate.blocked and ref.restart_reference == tuple(pose)


def test_repeated_start_while_active_keeps_same_control_and_pending_reference(setup):
    display, _, sid, _, _ = setup
    before = call(display, '/state')['control']
    original = display._reference_gate.reference
    result = call(display, '/action', {'action': 'start', 'control_protocol': 1, 'control': before})
    assert result['control'] == before
    assert display._active
    assert display._reference_gate.session == sid and display._reference_gate.reference == original


def test_idempotent_start_exports_sequence_without_resetting_accepted_heartbeats(setup):
    display, _, _, _, _ = setup
    control=call(display,'/state')['control']
    assert call(display,'/control-heartbeat',{**control,'sequence':42})['accepted']
    result=call(display,'/action',{'action':'start','control_protocol':1,'control':control})
    assert result['already_active'] and result['control_sequence']==42
    assert call(display,'/control-heartbeat',{**control,'sequence':43})['accepted']


def test_reference_failure_diagnostics_are_allowlisted_and_copies_are_independent():
    error=ReferenceRequestError('http_rejected',phase='response',elapsed_ms=1.,status=400,
        reference_reason='return_unverified',reference_check={
            'status':'position','position_error_mm':20.,'steady_ms':0.,
            'pose':['private'],'session':'private-session','token':'private-token',
            'angular_speed_rad_s':float('inf')})
    details=error.safe_details()
    assert details['reference_check']=={'status':'position','position_error_mm':20.,'steady_ms':0.}
    details['reference_check']['position_error_mm']=99.
    assert error.safe_details()['reference_check']['position_error_mm']==20.
    assert 'private' not in str(error)


def test_old_or_missing_hold_frame_cannot_acknowledge_a_new_selected_reference(setup):
    from leica_tracker.person_guidance import GuidanceCue, GuidanceDecision
    display, _, sid, ref, _ = setup
    assert ref.restart_begin(sid, reanchor=True)
    decision=GuidanceDecision('PHYSICAL_REFERENCE_HOLD',GuidanceCue.HOLD,None,None,None,False,'HOLD')
    display.publish(None,decision,source_monotonic_ns=None)
    assert not display._reference_gate.acknowledged


def test_completed_restart_cannot_block_the_following_guidance_observation_gate(setup):
    from leica_tracker.person_guidance import GuidanceCue, GuidanceDecision
    display, clock, sid, ref, _ = setup
    pose=selected_begin(setup)
    assert measured(setup,pose)['return_verified']
    advance(setup,clock.now+.1)
    ref.restart_release(sid,pose,ZERO,clock.now)
    with display._lock:display._physical_backend_mode='guidance'
    new_session=ref.begin(pose)
    assert new_session!=sid
    decision=GuidanceDecision('PHYSICAL_REFERENCE_HOLD',GuidanceCue.HOLD,None,None,None,False,'HOLD')
    display.publish(None,decision)
    assert display._reference_gate.acknowledged
    assert call(display,'/operator-state')['reference_gate']['reference_check']['status']=='no_sample'
