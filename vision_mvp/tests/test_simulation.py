import pytest

from leica_tracker.continuous import ControllerConfig
from leica_tracker.motion import MotionEnvelope, MotionIntent
from leica_tracker.simulation import VirtualFrameConfig, virtual_frame_pose


def test_neutral_intent_aligns_virtual_frame_with_composition_guide():
    target = ControllerConfig(
        target_x=0.50,
        target_head_y=0.25,
        target_shoulder_width=0.30,
    )
    intent = MotionIntent(False, 0.0, 0.0, 0.0, "TRACKING")

    pose = virtual_frame_pose(intent, target, MotionEnvelope())

    assert pose.center_x == pytest.approx(0.50)
    assert pose.head_y == pytest.approx(0.25)
    assert pose.half_width == pytest.approx(0.30)
    assert pose.top == pytest.approx(0.08)
    assert pose.bottom == pytest.approx(0.88)
    assert pose.hold is False


def test_motion_intent_moves_and_scales_the_virtual_robot_frame():
    target = ControllerConfig(
        target_x=0.50,
        target_head_y=0.25,
        target_shoulder_width=0.30,
    )
    envelope = MotionEnvelope(
        max_right_m=0.20,
        max_down_m=0.12,
        max_toward_subject_m=0.18,
    )
    intent = MotionIntent(False, 0.10, 0.06, 0.09, "TRACKING")

    pose = virtual_frame_pose(intent, target, envelope)

    assert pose.center_x == pytest.approx(0.66)
    assert pose.head_y == pytest.approx(0.36)
    assert pose.half_width == pytest.approx(0.255)
    assert pose.top == pytest.approx(0.2155)
    assert pose.bottom == pytest.approx(0.8955)


def test_lost_participant_holds_last_virtual_frame_pose():
    intent = MotionIntent(True, -0.04, 0.03, -0.06, "PERSON_LOST")

    pose = virtual_frame_pose(intent, ControllerConfig(), MotionEnvelope())

    assert pose.hold is True
    assert pose.center_x < 0.50
    assert pose.head_y > 0.25
    assert pose.half_width > 0.30


def test_virtual_frame_stays_fully_inside_preview_at_envelope_limits():
    config = VirtualFrameConfig()
    intent = MotionIntent(False, 0.20, 0.12, -0.18, "TRACKING")

    pose = virtual_frame_pose(
        intent,
        ControllerConfig(),
        MotionEnvelope(),
        config,
    )

    assert 0.0 <= pose.center_x - pose.half_width
    assert pose.center_x + pose.half_width <= 1.0
    assert 0.0 <= pose.top <= pose.bottom <= 1.0
