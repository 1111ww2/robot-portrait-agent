import pytest

from leica_tracker.person_guidance import GuidanceCue
from leica_tracker.robot_gestures import gesture_plan_for
from leica_tracker.gesture_preview import normalized_offset_for


@pytest.mark.parametrize("cue,peak,expected", [
    (GuidanceCue.MOVE_LEFT,600,(-.2,0.,0.)),
    (GuidanceCue.MOVE_UP,500,(0.,-.2,0.)),
    (GuidanceCue.MOVE_CLOSER,500,(0.,0.,.2)),
])
def test_small_directional_gesture_scales_excursion_once(cue,peak,expected):
    plan = gesture_plan_for(cue, amplitude=.2)
    assert normalized_offset_for(plan,peak) == pytest.approx(expected)
    assert normalized_offset_for(plan,plan.duration_ms) == (0.,0.,0.)


@pytest.mark.parametrize("amplitude", [0.,.1,1.01,float('nan'),float('inf'),True])
def test_directional_gesture_rejects_invalid_amplitude(amplitude):
    with pytest.raises(ValueError, match="amplitude"):
        gesture_plan_for(GuidanceCue.MOVE_LEFT, amplitude=amplitude)


def test_horizontal_double_sweep_moves_toward_cued_side_twice_and_returns_to_rest():
    plan = gesture_plan_for(GuidanceCue.MOVE_LEFT)

    first_sweep = normalized_offset_for(plan, 600)
    between_sweeps = normalized_offset_for(plan, 1200)
    second_sweep = normalized_offset_for(plan, 1800)
    finished = normalized_offset_for(plan, 2400)

    assert first_sweep == pytest.approx((-1.0, 0.0, 0.0))
    assert between_sweeps == pytest.approx((0.0, 0.0, 0.0))
    assert second_sweep == pytest.approx((-1.0, 0.0, 0.0))
    assert finished == pytest.approx((0.0, 0.0, 0.0))


def test_vertical_and_depth_sweeps_use_their_semantic_axes():
    up = normalized_offset_for(gesture_plan_for(GuidanceCue.MOVE_UP), 500)
    closer = normalized_offset_for(gesture_plan_for(GuidanceCue.MOVE_CLOSER), 500)

    assert up == pytest.approx((0.0, -1.0, 0.0))
    assert closer == pytest.approx((0.0, 0.0, 1.0))


def test_invite_shake_and_nod_have_distinct_visible_motion():
    invite = normalized_offset_for(gesture_plan_for(GuidanceCue.INVITE), 400)
    shake = normalized_offset_for(gesture_plan_for(GuidanceCue.NOT_YET), 175)
    completion = normalized_offset_for(gesture_plan_for(GuidanceCue.ARRIVED), 6000)

    assert invite != pytest.approx((0.0, 0.0, 0.0))
    assert shake[0] == pytest.approx(1.0, abs=1e-3)
    assert completion == pytest.approx((-1.,-1.,0.))


def test_missing_or_finished_plan_is_neutral():
    assert normalized_offset_for(None, 100) == pytest.approx((0.0, 0.0, 0.0))
    assert normalized_offset_for(gesture_plan_for(GuidanceCue.MOVE_UP), 2000) == pytest.approx(
        (0.0, 0.0, 0.0)
    )


def test_arrival_has_distinct_confirm_retract_plan_and_keeps_the_end_pose():
    plan=gesture_plan_for(GuidanceCue.ARRIVED)
    assert plan.pattern == 'confirm_retract_hold'
    assert plan.duration_ms == 6000
    assert normalized_offset_for(plan,6000) == pytest.approx((-1.,-1.,0.))
    assert normalized_offset_for(plan,12000) == pytest.approx((-1.,-1.,0.))
