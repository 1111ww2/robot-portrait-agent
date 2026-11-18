"""将连续跟踪指令限制在允许范围内，输出相机坐标系中的运动意图。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class CommandLike(Protocol):
    tracking: bool
    right: float
    down: float
    toward_subject: float
    reason: str


@dataclass(frozen=True, slots=True)
class MotionEnvelope:
    max_right_m: float = 0.20
    max_down_m: float = 0.12
    max_toward_subject_m: float = 0.18
    max_step_m: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "max_right_m",
            "max_down_m",
            "max_toward_subject_m",
            "max_step_m",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class MotionIntent:
    """相对于示教中立姿态的偏移，仍以相机坐标轴表示。"""

    hold: bool
    right_m: float
    down_m: float
    toward_subject_m: float
    reason: str


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _slew(current: float, target: float, max_step: float) -> float:
    return current + _clamp(target - current, max_step)


class MotionIntentMapper:
    """施加工作空间及单次更新的限值，不假定机器人基座的坐标轴方向。"""

    def __init__(self, envelope: MotionEnvelope | None = None) -> None:
        self.envelope = envelope or MotionEnvelope()
        self._current = (0.0, 0.0, 0.0)

    def update(self, command: CommandLike) -> MotionIntent:
        if not command.tracking:
            return MotionIntent(True, *self._current, command.reason)

        target = (
            _clamp(float(command.right), 1.0) * self.envelope.max_right_m,
            _clamp(float(command.down), 1.0) * self.envelope.max_down_m,
            _clamp(float(command.toward_subject), 1.0)
            * self.envelope.max_toward_subject_m,
        )
        self._current = tuple(
            _slew(current, wanted, self.envelope.max_step_m)
            for current, wanted in zip(self._current, target, strict=True)
        )
        return MotionIntent(False, *self._current, "TRACKING")
