"""为屏幕上的模拟机械臂生成可重复的归一化动画偏移。"""

from __future__ import annotations

import math

from .robot_gestures import GesturePlan
from .completion_choreography import completion_pose_at,simulated_frame_pose_at


def _stroke(progress: float, repetitions: int) -> float:
    cycle = (progress * repetitions) % 1.0
    return 2.0 * cycle if cycle <= 0.5 else 2.0 * (1.0 - cycle)


def normalized_offset_for(
    plan: GesturePlan | None,
    elapsed_ms: int,
) -> tuple[float, float, float]:
    """返回范围在 [-1, 1] 内的水平、垂直和深度偏移，用于机械臂动画。
    本地仿真桥接器将其映射为仿真关节偏移，不转换为实机坐标。"""
    if plan is None or elapsed_ms < 0:
        return (0.0, 0.0, 0.0)
    if plan.pattern in ('confirm_retract_hold','sim_frame_confirm_retract_hold','sim_frame_adjust_hold'):
        x,z,_ = (simulated_frame_pose_at(elapsed_ms) if plan.pattern=='sim_frame_adjust_hold'
                 else completion_pose_at(elapsed_ms,framed=plan.pattern=='sim_frame_confirm_retract_hold',
                                         screen_coupled=plan.profile=='screen-coupled'))
        return -x/.10, -z/.08, 0.
    if elapsed_ms >= plan.duration_ms:
        return (0.0, 0.0, 0.0)

    progress = elapsed_ms / plan.duration_ms
    if plan.profile=='agent-coupled':
        amount=math.sin(math.pi*progress*plan.repetitions)**2*plan.amplitude
        if plan.pattern=='agent_invite':return (.65*amount,-.25*amount,0.)
        if plan.pattern=='agent_ack':return (0.,.3*amount,0.)
        return tuple(amount*plan.direction if axis==plan.axis else 0. for axis in ('horizontal','vertical','depth'))
    if plan.pattern in ("double_sweep","double_sweep_smooth"):
        stroke=(math.sin(math.pi*progress*plan.repetitions)**2
                if plan.pattern=='double_sweep_smooth' else _stroke(progress,plan.repetitions))
        amount = stroke * plan.direction * plan.amplitude
        if plan.axis == "horizontal":
            return (amount, 0.0, 0.0)
        if plan.axis == "vertical":
            return (0.0, amount, 0.0)
        if plan.axis == "depth":
            return (0.0, 0.0, amount)
        return (0.0, 0.0, 0.0)

    if plan.pattern == "open_invitation":
        amount = math.sin(math.pi * progress)
        if plan.profile=='screen-coupled':amount=amount**2
        return (0.65 * amount, -0.25 * amount, 0.0)

    if plan.pattern == "shake":
        return (math.sin(2.0 * math.pi * progress), 0.0, 0.0)

    if plan.pattern == "nod":
        return (0.0, _stroke(progress, plan.repetitions) * plan.direction, 0.0)

    return (0.0, 0.0, 0.0)
