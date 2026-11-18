from leica_tracker.person_guidance import GuidanceCue
from leica_tracker.robot_gestures import gesture_plan_for


def test_directional_cues_are_two_short_camera_relative_sweeps():
    for cue, axis, direction in [
        (GuidanceCue.MOVE_LEFT, "horizontal", -1),
        (GuidanceCue.MOVE_RIGHT, "horizontal", 1),
        (GuidanceCue.MOVE_UP, "vertical", -1),
        (GuidanceCue.MOVE_DOWN, "vertical", 1),
        (GuidanceCue.MOVE_CLOSER, "depth", 1),
        (GuidanceCue.MOVE_FARTHER, "depth", -1),
    ]:
        plan = gesture_plan_for(cue)
        assert plan.cue == cue.value
        assert plan.effector == "robot_arm"
        assert plan.axis == axis
        assert plan.direction == direction
        assert plan.pattern == "double_sweep"
        assert plan.repetitions == 2
        assert plan.coordinate_frame == "participant_relative_semantic"


def test_invite_not_yet_and_arrival_have_symbolic_gestures():
    invite = gesture_plan_for(GuidanceCue.INVITE)
    not_yet = gesture_plan_for(GuidanceCue.NOT_YET)
    arrived = gesture_plan_for(GuidanceCue.ARRIVED)

    assert invite.pattern == "open_invitation"
    assert invite.effector == "robot_arm"
    assert not_yet.pattern == "shake"
    assert not_yet.axis == "horizontal"
    assert arrived.pattern == "confirm_retract_hold"


def test_wait_and_hold_do_not_request_robot_motion():
    assert gesture_plan_for(GuidanceCue.WAIT) is None
    assert gesture_plan_for(GuidanceCue.HOLD) is None
