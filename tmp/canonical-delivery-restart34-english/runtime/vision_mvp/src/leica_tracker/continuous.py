"""在相机相对坐标系中计算连续视觉伺服，不直接操作设备。"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from statistics import median
from typing import Protocol


class CompositionLike(Protocol):
    center_x: float
    head_y: float
    shoulder_width: float


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    target_x: float = 0.50
    target_head_y: float = 0.25
    target_shoulder_width: float = 0.30
    horizontal_deadband: float = 0.02
    vertical_deadband: float = 0.02
    scale_deadband: float = 0.03
    horizontal_span: float = 0.25
    vertical_span: float = 0.20
    scale_span: float = 0.15
    smoothing_alpha: float = 0.25

    def __post_init__(self) -> None:
        for name in ("target_x", "target_head_y", "target_shoulder_width"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("horizontal_deadband", "vertical_deadband", "scale_deadband"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("horizontal_span", "vertical_span", "scale_span"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class CameraRelativeCommand:
    """表示归一化运动意图，不能直接当作机器人基座坐标系的笛卡尔指令。"""

    tracking: bool
    right: float
    down: float
    toward_subject: float
    reason: str


def _bounded_response(error: float, deadband: float, span: float) -> float:
    if abs(error) <= deadband:
        return 0.0
    return max(-1.0, min(1.0, error / span))


class ContinuousController:
    """平滑姿态测量，在相机相对坐标系中输出限幅修正。"""

    def __init__(self, config: ControllerConfig | None = None, *, median_window: int = 1) -> None:
        if type(median_window) is not int or median_window < 1 or median_window % 2 == 0:
            raise ValueError('median_window must be a positive odd integer')
        self.config = config or ControllerConfig()
        self._history = deque(maxlen=median_window)
        self._smoothed: tuple[float, float, float] | None = None

    @property
    def filtered_point(self):
        return self._smoothed

    def update(self, observation: CompositionLike | None) -> CameraRelativeCommand:
        if observation is None:
            self._history.clear()
            self._smoothed = None
            return CameraRelativeCommand(False, 0.0, 0.0, 0.0, "PERSON_LOST")

        current = (
            float(observation.center_x),
            float(observation.head_y),
            float(observation.shoulder_width),
        )
        self._history.append(current)
        current=tuple(float(median(point[i] for point in self._history)) for i in range(3))
        if self._smoothed is None:
            self._smoothed = current
        else:
            alpha = self.config.smoothing_alpha
            self._smoothed = tuple(
                alpha * new + (1.0 - alpha) * old
                for new, old in zip(current, self._smoothed, strict=True)
            )

        center_x, head_y, shoulder_width = self._smoothed
        return CameraRelativeCommand(
            tracking=True,
            right=_bounded_response(
                center_x - self.config.target_x,
                self.config.horizontal_deadband,
                self.config.horizontal_span,
            ),
            down=_bounded_response(
                head_y - self.config.target_head_y,
                self.config.vertical_deadband,
                self.config.vertical_span,
            ),
            toward_subject=_bounded_response(
                self.config.target_shoulder_width - shoulder_width,
                self.config.scale_deadband,
                self.config.scale_span,
            ),
            reason="TRACKING",
        )
