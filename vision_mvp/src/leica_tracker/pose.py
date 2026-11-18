"""用于单人躯干跟踪的MediaPipe Tasks适配器。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Self

import mediapipe as mp
import numpy as np

TORSO_LANDMARK_INDEXES = (11, 12, 23, 24)
HEAD_LANDMARK_INDEXES = (0, 7, 8)
SHOULDER_LANDMARK_INDEXES = (11, 12)


@dataclass(frozen=True, slots=True)
class LandmarkPoint:
    x: float
    y: float
    visibility: float
    presence: float


@dataclass(frozen=True, slots=True)
class PoseDetection:
    center_x: float
    head_y: float
    shoulder_width: float
    confidence: float
    landmarks: tuple[LandmarkPoint, ...]


@dataclass(frozen=True, slots=True)
class UpperBodyObservation:
    """连续控制器使用的归一化图像测量。"""

    center_x: float
    head_y: float
    shoulder_width: float
    confidence: float


def _score(landmark: Any, name: str) -> float:
    value = getattr(landmark, name, None)
    return 1.0 if value is None else float(value)


def _visible_torso_x_values(
    landmarks: Sequence[Any],
    min_visibility: float,
) -> list[float]:
    if len(landmarks) <= max(TORSO_LANDMARK_INDEXES):
        return []

    values: list[float] = []
    for index in TORSO_LANDMARK_INDEXES:
        landmark = landmarks[index]
        x = getattr(landmark, "x", None)
        if x is None:
            continue
        x = float(x)
        if not 0.0 <= x <= 1.0:
            continue
        if _score(landmark, "visibility") < min_visibility:
            continue
        if _score(landmark, "presence") < min_visibility:
            continue
        values.append(x)
    return values


def torso_center_x(
    landmarks: Sequence[Any],
    min_visibility: float = 0.5,
) -> float | None:
    """返回至少两个可见肩膀/臀部的中位数x。"""
    if not 0.0 <= min_visibility <= 1.0:
        raise ValueError("min_visibility must be between 0 and 1")
    values = _visible_torso_x_values(landmarks, min_visibility)
    if len(values) < 2:
        return None
    return float(median(values))


def _visible_point(
    landmarks: Sequence[Any],
    index: int,
    min_visibility: float,
) -> Any | None:
    if index >= len(landmarks):
        return None
    point = landmarks[index]
    x = getattr(point, "x", None)
    y = getattr(point, "y", None)
    if x is None or y is None:
        return None
    if not 0.0 <= float(x) <= 1.0 or not 0.0 <= float(y) <= 1.0:
        return None
    if _score(point, "visibility") < min_visibility:
        return None
    if _score(point, "presence") < min_visibility:
        return None
    return point


def upper_body_observation(
    landmarks: Sequence[Any],
    min_visibility: float = 0.5,
) -> UpperBodyObservation | None:
    """从一个姿势中测量头部位置和明显肩宽。"""
    if not 0.0 <= min_visibility <= 1.0:
        raise ValueError("min_visibility must be between 0 and 1")

    head_points = [
        point
        for index in HEAD_LANDMARK_INDEXES
        if (point := _visible_point(landmarks, index, min_visibility)) is not None
    ]
    shoulders = [
        _visible_point(landmarks, index, min_visibility)
        for index in SHOULDER_LANDMARK_INDEXES
    ]
    if not head_points or any(point is None for point in shoulders):
        return None

    left_shoulder, right_shoulder = shoulders
    assert left_shoulder is not None and right_shoulder is not None
    shoulder_width = abs(float(right_shoulder.x) - float(left_shoulder.x))
    if shoulder_width <= 0.0:
        return None

    tracked_points = [*head_points, left_shoulder, right_shoulder]
    confidence = median(
        min(_score(point, "visibility"), _score(point, "presence"))
        for point in tracked_points
    )
    return UpperBodyObservation(
        center_x=float(median(float(point.x) for point in head_points)),
        head_y=float(median(float(point.y) for point in head_points)),
        shoulder_width=shoulder_width,
        confidence=float(confidence),
    )


class PoseTracker:
    """以视频模式运行当前的MediaPipe PoseLandmarker API。"""

    def __init__(
        self,
        model_path: str | Path,
        min_confidence: float = 0.5,
    ) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Pose model not found: {path}")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._min_confidence = min_confidence
        self._last_timestamp_ms = -1
        self._closed = False

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> PoseDetection | None:
        if self._closed:
            raise RuntimeError("PoseTracker is closed")
        if (
            rgb_frame.ndim != 3
            or rgb_frame.shape[2] != 3
            or rgb_frame.dtype != np.uint8
        ):
            raise ValueError("rgb_frame must be a uint8 HxWx3 array")
        if timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("timestamp_ms must increase for every frame")
        self._last_timestamp_ms = timestamp_ms

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None

        raw_landmarks = result.pose_landmarks[0]
        observation = upper_body_observation(raw_landmarks, self._min_confidence)
        if observation is None:
            return None
        landmarks = tuple(
            LandmarkPoint(
                x=float(point.x),
                y=float(point.y),
                visibility=_score(point, "visibility"),
                presence=_score(point, "presence"),
            )
            for point in raw_landmarks
        )
        return PoseDetection(
            center_x=observation.center_x,
            head_y=observation.head_y,
            shoulder_width=observation.shoulder_width,
            confidence=observation.confidence,
            landmarks=landmarks,
        )

    def close(self) -> None:
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
