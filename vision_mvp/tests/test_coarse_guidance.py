from types import SimpleNamespace
import math


def pose(x=.61,y=.44,width=.23):
    return SimpleNamespace(center_x=x,head_y=y,shoulder_width=width,confidence=.9)


def controller():
    from leica_tracker.coarse_guidance import CoarsePortraitController
    return CoarsePortraitController()


def test_natural_pose_is_ready_without_precise_head_and_size_alignment():
    c=controller();decisions=[c.update(pose(),t) for t in range(0,1601,100)]
    assert any(d.phase=='POSITION_READY' for d in decisions)
    assert all(d.phase!='ARRIVED' for d in decisions)
    assert all(d.interaction_mode=='coarse-demo' for d in decisions)


def test_walking_through_the_zone_does_not_start_framing():
    c=controller()
    for t in range(0,2001,100):
        d=c.update(pose(x=.5+.08*math.sin(t/200)),t)
        assert d.phase not in ('POSITION_READY','ARM_FRAMING')
    still=[c.update(pose(x=.5),t) for t in range(2100,3801,100)]
    assert any(d.phase=='POSITION_READY' for d in still)


def test_staged_workflow_is_explicit_and_each_robot_gesture_is_emitted_once():
    c=controller();ds=[c.update(pose(),t) for t in range(0,13001,100)]
    phases=[d.phase for d in ds]
    assert phases.index('POSITION_READY')<phases.index('ARM_FRAMING')
    assert phases.index('ARM_FRAMING')<phases.index('SIMULATED_CAPTURE')
    assert phases.index('SIMULATED_CAPTURE')<phases.index('COMPLETING')
    assert phases.index('COMPLETING')<phases.index('DEMO_COMPLETE')
    assert sum(d.cue.value=='SIM_FRAME' for d in ds)==1
    assert sum(d.cue.value=='SIM_COMPLETE' for d in ds)==1
    assert ds[-1].capture_simulated is True
    assert all(d.framing_verified is False for d in ds)


def test_loss_and_source_gap_abort_the_framing_instead_of_finishing_it():
    for gap in [False,True]:
        c=controller()
        for t in range(0,2201,100):c.update(pose(),t)
        d=c.update(pose(),2700) if gap else c.update(None,2300)
        assert d.cue.value=='HOLD'
        stamp=2800 if gap else 2400
        resumed=c.update(pose(),stamp)
        assert resumed.phase not in ('COMPLETING','DEMO_COMPLETE')
        assert resumed.capture_simulated is False


def test_large_movement_during_framing_requires_new_positioning():
    c=controller()
    for t in range(0,2201,100):c.update(pose(),t)
    moved=c.update(pose(x=.9),2300)
    assert moved.cue.value=='HOLD'
    assert moved.phase=='REPOSITION'


def test_movement_within_the_broad_zone_can_still_abort_a_locked_framing_pose():
    c=controller()
    for t in range(0,2201,100):c.update(pose(),t)
    moved=c.update(pose(width=.32),2300)
    assert moved.cue.value=='HOLD'
    assert moved.phase=='REPOSITION'


def test_completed_demo_holds_on_loss_and_reset_starts_a_new_session():
    c=controller()
    for t in range(0,13001,100):c.update(pose(),t)
    left=c.update(None,13100)
    assert left.phase=='DEMO_COMPLETE' and left.cue.value=='HOLD'
    c.reset()
    assert c.update(pose(x=.4),13200).phase=='INVITING'
