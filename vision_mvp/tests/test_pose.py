from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from leica_tracker import pose
from leica_tracker.pose import PoseTracker, torso_center_x

MODEL_PATH = Path(__file__).parents[1] / "models" / "pose_landmarker_lite.task"


def landmarks_with(values):
    landmarks = [
        SimpleNamespace(x=0.5, y=0.5, visibility=0.0, presence=0.0) for _ in range(33)
    ]
    for value in values:
        index, x, visibility, presence, *optional_y = value
        landmarks[index] = SimpleNamespace(
            x=x,
            y=optional_y[0] if optional_y else 0.5,
            visibility=visibility,
            presence=presence,
        )
    return landmarks


def test_torso_center_uses_median_of_visible_shoulders_and_hips():
    landmarks = landmarks_with(
        [
            (11, 0.20, 0.9, 0.9),
            (12, 0.40, 0.9, 0.9),
            (23, 0.60, 0.9, 0.9),
            (24, 0.80, 0.9, 0.9),
        ]
    )

    assert torso_center_x(landmarks, min_visibility=0.5) == pytest.approx(0.5)


def test_torso_center_accepts_two_visible_torso_points():
    landmarks = landmarks_with(
        [
            (11, 0.25, 0.8, 0.8),
            (12, 0.45, 0.8, 0.8),
            (23, 0.65, 0.2, 0.8),
            (24, 0.75, 0.8, 0.2),
        ]
    )

    assert torso_center_x(landmarks, min_visibility=0.5) == pytest.approx(0.35)


def test_torso_center_rejects_insufficient_or_out_of_frame_points():
    landmarks = landmarks_with(
        [
            (11, 0.25, 0.9, 0.9),
            (12, 1.20, 0.9, 0.9),
            (23, 0.65, 0.1, 0.9),
            (24, 0.75, 0.9, 0.1),
        ]
    )

    assert torso_center_x(landmarks, min_visibility=0.5) is None


def test_torso_center_rejects_incomplete_landmark_collection():
    assert torso_center_x([], min_visibility=0.5) is None


def test_pose_tracker_returns_no_detection_for_blank_frame():
    tracker = PoseTracker(MODEL_PATH)
    try:
        blank_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        assert tracker.detect(blank_rgb, timestamp_ms=0) is None
    finally:
        tracker.close()


def test_upper_body_observation_uses_head_position_and_shoulder_scale():
    landmarks = landmarks_with(
        [
            (0, 0.52, 0.9, 0.9, 0.24),
            (7, 0.48, 0.9, 0.9, 0.26),
            (8, 0.56, 0.9, 0.9, 0.26),
            (11, 0.40, 0.9, 0.9, 0.44),
            (12, 0.64, 0.9, 0.9, 0.44),
        ]
    )

    observation = pose.upper_body_observation(landmarks, min_visibility=0.5)

    assert observation is not None
    assert observation.center_x == pytest.approx(0.52)
    assert observation.head_y == pytest.approx(0.26)
    assert observation.shoulder_width == pytest.approx(0.24)
    assert observation.confidence == pytest.approx(0.9)


def test_upper_body_observation_requires_head_and_both_shoulders():
    shoulders_only = landmarks_with(
        [
            (11, 0.40, 0.9, 0.9, 0.44),
            (12, 0.64, 0.9, 0.9, 0.44),
        ]
    )
    one_shoulder = landmarks_with(
        [
            (0, 0.50, 0.9, 0.9, 0.24),
            (11, 0.40, 0.9, 0.9, 0.44),
        ]
    )

    assert pose.upper_body_observation(shoulders_only) is None
    assert pose.upper_body_observation(one_shoulder) is None
