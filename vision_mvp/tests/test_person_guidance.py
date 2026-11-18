from types import SimpleNamespace

import pytest

from leica_tracker.continuous import ControllerConfig
from leica_tracker.person_guidance import GuidanceCue, PersonGuidanceController


def pose(x=0.5, y=0.25, width=0.30, confidence=0.9):
    return SimpleNamespace(
        center_x=x,
        head_y=y,
        shoulder_width=width,
        confidence=confidence,
    )


@pytest.mark.parametrize("near,far", [
    (pose(x=.53), pose(x=.8)),
    (pose(y=.28), pose(y=.5)),
    (pose(width=.265), pose(width=.10)),
])
def test_directional_amplitude_grows_with_requested_axis_error(near, far):
    def decision_for(observation):
        c = PersonGuidanceController(ControllerConfig(smoothing_alpha=1.))
        c.update(pose(), 0)
        return c.update(observation, 1200)
    small, large = decision_for(near), decision_for(far)
    assert small.gesture_amplitude == pytest.approx(.35)
    assert large.gesture_amplitude == pytest.approx(1.)
    assert small.cue == large.cue


def test_first_valid_pose_captures_initial_point_and_invites():
    controller = PersonGuidanceController()

    decision = controller.update(pose(0.32, 0.41, 0.22), 100)

    assert decision.cue is GuidanceCue.INVITE
    assert decision.phase == "INVITING"
    assert decision.tracking is True
    assert decision.initial_point == pytest.approx((0.32, 0.41, 0.22))
    assert decision.target == pytest.approx((0.50, 0.25, 0.30))


@pytest.mark.parametrize(
    ("mirrored", "x", "cue"),
    [
        (True, 0.70, GuidanceCue.MOVE_LEFT),
        (True, 0.30, GuidanceCue.MOVE_RIGHT),
        (False, 0.70, GuidanceCue.MOVE_RIGHT),
        (False, 0.30, GuidanceCue.MOVE_LEFT),
    ],
)
def test_horizontal_cue_points_toward_fixed_target(mirrored, x, cue):
    controller = PersonGuidanceController(response_window_ms=100)
    controller.update(pose(), 0, mirrored=mirrored)

    decision = controller.update(pose(x=x), 1200, mirrored=mirrored)

    assert decision.cue is cue
    assert decision.target == pytest.approx((0.50, 0.25, 0.30))


@pytest.mark.parametrize(
    ("observation", "cue"),
    [
        (pose(y=0.40), GuidanceCue.MOVE_UP),
        (pose(y=0.10), GuidanceCue.MOVE_DOWN),
        (pose(width=0.15), GuidanceCue.MOVE_CLOSER),
        (pose(width=0.50), GuidanceCue.MOVE_FARTHER),
    ],
)
def test_vertical_and_apparent_distance_cues(observation, cue):
    controller = PersonGuidanceController(response_window_ms=100)
    controller.update(pose(), 0)

    decision = controller.update(observation, 1200)

    assert decision.cue is cue


def test_directional_cue_waits_for_response_then_uses_one_no_progress_shake():
    controller = PersonGuidanceController(response_window_ms=500)
    controller.update(pose(), 0)
    first = controller.update(pose(x=0.75), 1200)
    waiting = controller.update(pose(x=0.75), 1400)
    no_progress = controller.update(pose(x=0.75), 1700)

    assert first.cue is GuidanceCue.MOVE_LEFT
    assert waiting.cue is GuidanceCue.WAIT
    assert no_progress.cue is GuidanceCue.WAIT
    assert no_progress.phase == "PAUSED_FOR_PARTICIPANT"


def test_no_progress_shake_waits_for_participant_instead_of_repeating_direction():
    controller = PersonGuidanceController(
        config=ControllerConfig(smoothing_alpha=1.0),
        response_window_ms=500,
    )
    controller.update(pose(), 0)
    controller.update(pose(x=0.75), 1200)
    shake = controller.update(pose(x=0.75), 1700)
    waiting = controller.update(pose(x=0.75), 5000)
    resumed = controller.update(pose(x=0.65), 5100)

    assert shake.cue is GuidanceCue.WAIT
    assert waiting.cue is GuidanceCue.WAIT
    assert waiting.phase == "PAUSED_FOR_PARTICIPANT"
    assert resumed.cue is GuidanceCue.MOVE_LEFT


def test_tracking_loss_holds_without_cue_and_preserves_initial_point():
    controller = PersonGuidanceController()
    controller.update(pose(0.3, 0.4, 0.2), 0)

    decision = controller.update(None, 100)

    assert decision.cue is GuidanceCue.HOLD
    assert decision.tracking is False
    assert decision.current_point is None
    assert decision.initial_point == pytest.approx((0.3, 0.4, 0.2))


def test_tracking_recovery_resumes_guidance_without_recapturing_initial_point():
    controller = PersonGuidanceController(
        config=ControllerConfig(smoothing_alpha=1.0),
    )
    controller.update(pose(0.3, 0.4, 0.2), 0)
    controller.update(None, 100)

    recovered = controller.update(pose(0.7, 0.25, 0.3), 1200)

    assert recovered.cue is GuidanceCue.MOVE_LEFT
    assert recovered.tracking is True
    assert recovered.initial_point == pytest.approx((0.3, 0.4, 0.2))


def test_low_confidence_is_treated_as_tracking_loss():
    controller = PersonGuidanceController(min_confidence=0.6)

    decision = controller.update(pose(confidence=0.2), 0)

    assert decision.cue is GuidanceCue.HOLD
    assert decision.tracking is False
    assert decision.initial_point is None


def test_leaving_target_restarts_arrival_dwell():
    controller = PersonGuidanceController(
        config=ControllerConfig(smoothing_alpha=1.0),
        arrival_dwell_ms=1000,
    )
    controller.update(pose(x=0.8), 0)
    first_target_hit = controller.update(pose(), 1000)
    left_target = controller.update(pose(x=0.8), 1500)
    second_target_hit = controller.update(pose(), 2000)
    before_dwell = controller.update(pose(), 2999)
    arrived = controller.update(pose(), 3000)

    assert first_target_hit.cue is GuidanceCue.WAIT
    assert left_target.cue is GuidanceCue.MOVE_LEFT
    assert second_target_hit.cue is GuidanceCue.WAIT
    assert before_dwell.cue is GuidanceCue.WAIT
    assert arrived.cue is GuidanceCue.ARRIVED


def test_arrival_requires_one_second_in_deadband_and_nods_once():
    controller = PersonGuidanceController(
        config=ControllerConfig(smoothing_alpha=1.0),
        arrival_dwell_ms=1000,
    )
    controller.update(pose(x=0.8), 0)
    controller.update(pose(), 1200)

    early = controller.update(pose(), 1500)
    arrived = controller.update(pose(), 2200)
    terminal = controller.update(pose(), 5000)

    assert early.cue is GuidanceCue.WAIT
    assert arrived.cue is GuidanceCue.ARRIVED
    assert terminal.cue is GuidanceCue.WAIT
    assert terminal.phase == "ARRIVED"


def test_arrival_dwell_reports_progress_and_restarts_after_leaving_target():
    c=PersonGuidanceController(ControllerConfig(smoothing_alpha=1.),arrival_dwell_ms=1000)
    c.update(pose(x=.8),0)
    assert c.update(pose(),1200).arrival_progress == 0.
    assert c.update(pose(),1700).arrival_progress == pytest.approx(.5)
    c.update(pose(x=.8),1800)
    assert c.update(pose(),2000).arrival_progress == 0.
    assert c.update(pose(),3000).arrival_progress == 1.


def test_completed_session_stays_complete_when_person_leaves_but_motion_holds():
    c=PersonGuidanceController(ControllerConfig(smoothing_alpha=1.))
    c.update(pose(x=.8),0);c.update(pose(),1200);c.update(pose(),2200)
    left=c.update(None,2300)
    assert left.phase == "ARRIVED"
    assert left.cue is GuidanceCue.HOLD
    assert left.tracking is False
    assert left.arrival_progress == 1.
    assert c.update(pose(x=.8),2500).cue is GuidanceCue.WAIT
    c.reset()
    assert c.update(None,2600).phase == "TRACKING_LOST"


def test_short_measurement_outlier_does_not_restart_stable_arrival():
    c=PersonGuidanceController(ControllerConfig(smoothing_alpha=1.),median_window=5)
    c.update(pose(x=.8),0)
    for t in range(100,1201,100):
        observation=pose(x=.9,y=.6,width=.1) if t==800 else pose()
        decision=c.update(observation,t)
    assert decision.cue is GuidanceCue.ARRIVED
    assert decision.filtered_point == pytest.approx((.5,.25,.3))


def test_completion_defers_on_current_outlier_even_with_good_filtered_history():
    c=PersonGuidanceController(ControllerConfig(smoothing_alpha=1.),median_window=5)
    c.update(pose(x=.8),0)
    for t in range(100,1200,100):c.update(pose(),t)
    spike=c.update(pose(x=.9),1200)
    assert spike.phase == 'SETTLING'
    assert spike.arrival_progress < 1.
    assert c.update(pose(),1300).cue is GuidanceCue.ARRIVED


def test_a_single_target_sample_cannot_complete_a_far_person():
    c=PersonGuidanceController(ControllerConfig(smoothing_alpha=1.),median_window=5)
    for t in range(0,3001,100):
        d=c.update(pose() if t==1500 else pose(x=.8),t)
        assert d.phase not in ('ARRIVED','SETTLING')


def test_tracking_loss_clears_arrival_filter_history():
    c=PersonGuidanceController(ControllerConfig(smoothing_alpha=1.),median_window=5)
    for t in range(0,501,100):c.update(pose(),t)
    c.update(None,600)
    reacquired=c.update(pose(x=.8),700)
    assert reacquired.filtered_point == pytest.approx((.8,.25,.3))
    assert reacquired.phase != 'ARRIVED'


@pytest.mark.parametrize('width',[.001,.005,.02])
def test_tiny_subject_does_not_invite_or_guide_until_an_usable_upper_body_enters(width):
    c=PersonGuidanceController(median_window=5)
    tiny=c.update(pose(width=width),0)
    assert tiny.cue is GuidanceCue.HOLD
    assert tiny.tracking is False
    assert tiny.initial_point is None
    assert tiny.reason == 'SUBJECT_TOO_SMALL'
    assert c.update(pose(),100).cue is GuidanceCue.INVITE


def test_reset_reuses_target_but_captures_a_new_initial_point():
    config = ControllerConfig(target_x=0.62, target_head_y=0.35, target_shoulder_width=0.24)
    controller = PersonGuidanceController(config=config)
    controller.update(pose(0.2), 0)

    controller.reset()
    decision = controller.update(pose(0.8), 50)

    assert decision.cue is GuidanceCue.INVITE
    assert decision.initial_point == pytest.approx((0.8, 0.25, 0.30))
    assert decision.target == pytest.approx((0.62, 0.35, 0.24))


def test_wait_reselects_changed_axis_when_total_distance_grows():
    controller=PersonGuidanceController(
        config=ControllerConfig(smoothing_alpha=1),response_window_ms=3000)
    controller.update(pose(y=.4),0)
    assert controller.update(pose(y=.4),1200).cue is GuidanceCue.MOVE_UP
    assert controller.update(pose(y=.4),4200).cue is GuidanceCue.WAIT
    assert controller.update(pose(y=.4),8000).phase=='PAUSED_FOR_PARTICIPANT'
    changed=controller.update(pose(x=.9,y=.4),9000)
    assert changed.cue is GuidanceCue.MOVE_LEFT
    assert changed.initial_point == pytest.approx((.5,.4,.3))
    assert changed.target == pytest.approx((.5,.25,.3))
    assert controller.update(pose(x=.9,y=.4),9100).cue is GuidanceCue.WAIT
    assert controller.update(pose(x=.1,y=.4),12000).cue is GuidanceCue.MOVE_RIGHT


def test_progress_in_requested_axis_is_recognised_when_other_axis_worsens():
    controller=PersonGuidanceController(
        config=ControllerConfig(smoothing_alpha=1),response_window_ms=500)
    controller.update(pose(),0)
    assert controller.update(pose(x=.75),1200).cue is GuidanceCue.MOVE_LEFT
    # x从1.0提高到0.8，而y增长到0.75：总标准增加。
    assert controller.update(pose(x=.7,y=.4),1700).cue is GuidanceCue.MOVE_LEFT


@pytest.mark.parametrize("mirrored,first,opposite", [(True,GuidanceCue.MOVE_LEFT,GuidanceCue.MOVE_RIGHT),(False,GuidanceCue.MOVE_RIGHT,GuidanceCue.MOVE_LEFT)])
def test_sim_fast_reversal_requires_two_consecutive_overshoot_frames(mirrored, first, opposite):
    c=PersonGuidanceController(config=ControllerConfig(smoothing_alpha=1.),response_window_ms=4200,fast_direction_reversal=True)
    c.update(pose(x=.8),0,mirrored=mirrored)
    assert c.update(pose(x=.8),1200,mirrored=mirrored).cue is first
    assert c.update(pose(x=.2),1300,mirrored=mirrored).cue is GuidanceCue.WAIT
    assert c.update(pose(x=.2),1400,mirrored=mirrored).cue is opposite

def test_sim_fast_reversal_rejects_single_frame_and_near_target():
    c=PersonGuidanceController(config=ControllerConfig(smoothing_alpha=1.),response_window_ms=4200,fast_direction_reversal=True)
    c.update(pose(x=.8),0);c.update(pose(x=.8),1200)
    assert c.update(pose(x=.2),1300).cue is GuidanceCue.WAIT
    assert c.update(pose(x=.8),1400).cue is GuidanceCue.WAIT
    assert c.update(pose(x=.2),1500).cue is GuidanceCue.WAIT
    assert c.update(pose(x=.47),1600).cue is GuidanceCue.WAIT
    assert c.update(pose(x=.2),1700).cue is GuidanceCue.WAIT
    assert c.update(pose(x=.2),1800).cue is GuidanceCue.MOVE_RIGHT

def test_default_mode_still_waits_on_sustained_reversal():
    c=PersonGuidanceController(config=ControllerConfig(smoothing_alpha=1.),response_window_ms=4200)
    c.update(pose(x=.8),0);c.update(pose(x=.8),1200)
    assert c.update(pose(x=.2),1300).cue is GuidanceCue.WAIT
    assert c.update(pose(x=.2),1400).cue is GuidanceCue.WAIT

def test_sim_fast_reversal_resets_candidate_on_tracking_loss():
    c=PersonGuidanceController(config=ControllerConfig(smoothing_alpha=1.),response_window_ms=4200,fast_direction_reversal=True)
    c.update(pose(x=.8),0);c.update(pose(x=.8),1200)
    assert c.update(pose(x=.2),1300).cue is GuidanceCue.WAIT
    assert c.update(None,1400).cue is GuidanceCue.HOLD
    # 重新收购开始一个新的提示，而不是重播旧候选人。
    assert c.update(pose(x=.2),1500).cue is GuidanceCue.MOVE_RIGHT


@pytest.mark.parametrize("initial,cross,first,opposite",[
    (pose(y=.45),pose(y=.05),GuidanceCue.MOVE_UP,GuidanceCue.MOVE_DOWN),
    (pose(width=.12),pose(width=.48),GuidanceCue.MOVE_CLOSER,GuidanceCue.MOVE_FARTHER),
])
def test_sim_fast_reversal_works_for_vertical_and_depth(initial,cross,first,opposite):
    c=PersonGuidanceController(config=ControllerConfig(smoothing_alpha=1.),response_window_ms=4200,fast_direction_reversal=True)
    c.update(initial,0)
    assert c.update(initial,1200).cue is first
    assert c.update(cross,1300).cue is GuidanceCue.WAIT
    assert c.update(cross,1400).cue is opposite

def test_sim_fast_reversal_does_not_change_axis_within_response_window():
    c=PersonGuidanceController(config=ControllerConfig(smoothing_alpha=1.),response_window_ms=4200,fast_direction_reversal=True)
    c.update(pose(x=.8),0);c.update(pose(x=.8),1200)
    assert c.update(pose(x=.8,y=.6),1300).cue is GuidanceCue.WAIT
    assert c.update(pose(x=.8,y=.6),1400).cue is GuidanceCue.WAIT


def test_sim_fast_reversal_uses_fresh_raw_frames_despite_slow_median_filter():
    c=PersonGuidanceController(config=ControllerConfig(smoothing_alpha=.25),median_window=5,response_window_ms=4200,fast_direction_reversal=True)
    for t in (0,100,200,300,400):c.update(pose(x=.8),t)
    assert c.update(pose(x=.8),1200).cue is GuidanceCue.MOVE_LEFT
    assert c.update(pose(x=.2),1300).cue is GuidanceCue.WAIT
    assert c.update(pose(x=.2),1400).cue is GuidanceCue.MOVE_RIGHT


def test_arrival_two_second_dwell_is_not_triggered_early():
    c=PersonGuidanceController(ControllerConfig(smoothing_alpha=1.),arrival_dwell_ms=2000)
    c.update(pose(x=.8),0)
    assert c.update(pose(),1000).phase=='SETTLING'
    assert c.update(pose(),2999).cue is GuidanceCue.WAIT
    assert c.update(pose(),3000).cue is GuidanceCue.ARRIVED
