"""传统水平区域备用和时间稳定。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Zone(str, Enum):
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"
    LOST = "LOST"


def classify_zone(
    x: float,
    previous: Zone = Zone.LOST,
    margin: float = 0.04,
) -> Zone:
    """对具有前区滞后的归一化x坐标进行分类。"""
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be between 0 and 1")
    if not 0.0 <= margin < 1 / 3:
        raise ValueError("margin must be between 0 and one third")

    left_boundary = 1 / 3
    right_boundary = 2 / 3

    if previous is Zone.LEFT and x < left_boundary + margin:
        return Zone.LEFT
    if previous is Zone.CENTER and left_boundary - margin <= x < right_boundary + margin:
        return Zone.CENTER
    if previous is Zone.RIGHT and x >= right_boundary - margin:
        return Zone.RIGHT

    if x < left_boundary:
        return Zone.LEFT
    if x < right_boundary:
        return Zone.CENTER
    return Zone.RIGHT


@dataclass(slots=True)
class ZoneStabilizer:
    """仅发出已确认的区域过渡和延迟参与者丢失。"""

    required_frames: int = 4
    lost_frames: int = 10
    stable_zone: Zone = field(default=Zone.LOST, init=False)
    _pending_zone: Zone | None = field(default=None, init=False)
    _pending_count: int = field(default=0, init=False)
    _missed_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.required_frames <= 0 or self.lost_frames <= 0:
            raise ValueError("frame thresholds must be positive")

    def update(self, candidate: Zone | None) -> Zone | None:
        """只有在过渡得到确认时，才返回一个新的稳定区。"""
        if candidate is None or candidate is Zone.LOST:
            self._pending_zone = None
            self._pending_count = 0
            self._missed_count += 1
            if self.stable_zone is not Zone.LOST and self._missed_count >= self.lost_frames:
                self.stable_zone = Zone.LOST
                return Zone.LOST
            return None

        self._missed_count = 0
        if candidate is self.stable_zone:
            self._pending_zone = None
            self._pending_count = 0
            return None

        if candidate is self._pending_zone:
            self._pending_count += 1
        else:
            self._pending_zone = candidate
            self._pending_count = 1

        if self._pending_count < self.required_frames:
            return None

        self.stable_zone = candidate
        self._pending_zone = None
        self._pending_count = 0
        return candidate
