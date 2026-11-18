"""将相机坐标系中的运动意图投影到虚拟机械臂搭载的画框上。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .motion import MotionEnvelope, MotionIntent


class CompositionTarget(Protocol):
    target_x: float
    target_head_y: float
    target_shoulder_width: float


@dataclass(frozen=True, slots=True)
class VirtualFrameConfig:
    """屏幕坐标响应仅用于不连接机器人的预览。"""

    horizontal_screen_span: float = 0.32
    vertical_screen_span: float = 0.22
    depth_scale_factor: float = 0.30
    min_scale: float = 0.70
    max_scale: float = 1.15
    top_from_head: float = 0.17
    bottom_from_head: float = 0.63
    edge_margin: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "horizontal_screen_span",
            "vertical_screen_span",
            "depth_scale_factor",
            "min_scale",
            "max_scale",
            "top_from_head",
            "bottom_from_head",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.edge_margin < 0.5:
            raise ValueError("edge_margin must be between 0 and 0.5")
        if self.min_scale > self.max_scale:
            raise ValueError("min_scale must not exceed max_scale")


@dataclass(frozen=True, slots=True)
class VirtualFramePose:
    center_x: float
    head_y: float
    half_width: float
    top: float
    bottom: float
    hold: bool


def _normalise(value: float, limit: float) -> float:
    return max(-1.0, min(1.0, float(value) / limit))


def _fit_interval(
    start: float, end: float, lower: float, upper: float
) -> tuple[float, float]:
    """在保持区间长度的前提下平移区间，使其落入边界内。"""

    size = end - start
    available = upper - lower
    if size >= available:
        return lower, upper
    if start < lower:
        return lower, lower + size
    if end > upper:
        return upper - size, upper
    return start, end


def virtual_frame_pose(
    intent: MotionIntent,
    target: CompositionTarget,
    envelope: MotionEnvelope,
    config: VirtualFrameConfig | None = None,
) -> VirtualFramePose:
    """将限幅后的机器人目标映射到可见的预览边框。
    此处只消费 MotionIntent，不读取人物检测结果，因此预览使用的是未来 UR5e 适配器也将接收的同一控制器输出。"""

    display = config or VirtualFrameConfig()
    right = _normalise(intent.right_m, envelope.max_right_m)
    down = _normalise(intent.down_m, envelope.max_down_m)
    toward = _normalise(
        intent.toward_subject_m,
        envelope.max_toward_subject_m,
    )

    scale = max(
        display.min_scale,
        min(display.max_scale, 1.0 - toward * display.depth_scale_factor),
    )
    half_width = float(target.target_shoulder_width) * scale
    raw_center_x = float(target.target_x) + right * display.horizontal_screen_span
    center_x = max(
        display.edge_margin + half_width,
        min(1.0 - display.edge_margin - half_width, raw_center_x),
    )
    head_y = float(target.target_head_y) + down * display.vertical_screen_span
    top, bottom = _fit_interval(
        head_y - display.top_from_head * scale,
        head_y + display.bottom_from_head * scale,
        display.edge_margin,
        1.0 - display.edge_margin,
    )

    return VirtualFramePose(
        center_x=center_x,
        head_y=head_y,
        half_width=half_width,
        top=top,
        bottom=bottom,
        hold=intent.hold,
    )
