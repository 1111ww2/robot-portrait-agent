import pytest
from leica_tracker.robot_gestures import gesture_plan_for
from leica_tracker.person_guidance import GuidanceCue


def test_right_edge_has_exactly_two_peaks_at_the_arm_sweep_peaks():
    from leica_tracker.gesture_feedback import edge_feedback_for
    plan=gesture_plan_for(GuidanceCue.MOVE_RIGHT)
    values=[edge_feedback_for(plan,t) for t in (0,600,1200,1800,2400)]
    assert [v['intensity'] for v in values]==pytest.approx([0,1,0,1,0])
    assert all(v['edge']=='right' and v['tone']=='green' for v in values[:4])


@pytest.mark.parametrize('cue,edge', [('MOVE_LEFT','left'),('MOVE_UP','top'),('MOVE_DOWN','bottom'),('MOVE_CLOSER','all'),('MOVE_FARTHER','all')])
def test_other_directions_have_unambiguous_border_mapping(cue,edge):
    from leica_tracker.gesture_feedback import edge_feedback_for
    plan=gesture_plan_for(GuidanceCue(cue),amplitude=.35)
    f=edge_feedback_for(plan,plan.duration_ms/4)
    assert f['edge']==edge and f['intensity']==pytest.approx(1.)


def test_completion_matches_both_wrist_confirmations_then_holds_green():
    from leica_tracker.gesture_feedback import edge_feedback_for
    plan=gesture_plan_for(GuidanceCue.ARRIVED)
    assert [edge_feedback_for(plan,t)['intensity'] for t in (400,1100,1800,2500,3200,6000,8000)]==pytest.approx([0,1,0,1,1,1,1])
    assert edge_feedback_for(plan,6000)['edge']=='all'


def test_hold_or_missing_plan_has_no_border_signal():
    from leica_tracker.gesture_feedback import edge_feedback_for
    assert edge_feedback_for(None,0)['intensity']==0


def test_screen_coupled_sweeps_have_smooth_six_second_double_pulse():
    from leica_tracker.gesture_feedback import edge_feedback_for
    from leica_tracker.gesture_preview import normalized_offset_for
    plan=gesture_plan_for(GuidanceCue.MOVE_RIGHT,profile='screen-coupled')
    assert plan.duration_ms==6000
    assert [edge_feedback_for(plan,t)['intensity'] for t in (0,1500,3000,4500,6000)]==pytest.approx([0,1,0,1,0])
    # 平缓起步，避免三角波在起点造成目标速度突变。
    assert normalized_offset_for(plan,30)[0]<.002
    assert gesture_plan_for(GuidanceCue.MOVE_RIGHT).duration_ms==2400


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError):
        gesture_plan_for(GuidanceCue.MOVE_RIGHT,profile='arbitrary')


def test_screen_coupled_completion_uses_slower_distinct_confirmation_beats():
    from leica_tracker.gesture_feedback import edge_feedback_for
    plan=gesture_plan_for(GuidanceCue.ARRIVED,profile='screen-coupled')
    assert plan.duration_ms==8800
    assert [edge_feedback_for(plan,t)['intensity'] for t in (400,1800,3200,4600,6000,8800)]==pytest.approx([0,1,0,1,1,1])
    assert gesture_plan_for(GuidanceCue.INVITE,profile='screen-coupled').duration_ms==3000


@pytest.mark.parametrize("cue", ["MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN", "MOVE_CLOSER", "MOVE_FARTHER"])
def test_simulator_fast_direction_has_two_smooth_peaks_without_changing_normal_profile(cue):
    from leica_tracker.gesture_feedback import edge_feedback_for
    plan=gesture_plan_for(GuidanceCue(cue),profile='sim-fast')
    assert plan.duration_ms==3300
    assert [edge_feedback_for(plan,t)['intensity'] for t in (0,825,1650,2475,3300)]==pytest.approx([0,1,0,1,0])
    assert gesture_plan_for(GuidanceCue(cue),profile='screen-coupled').duration_ms==6000

def test_simulator_fast_profile_rejects_non_directional_cues():
    with pytest.raises(ValueError):gesture_plan_for(GuidanceCue.ARRIVED,profile='sim-fast')
