"""用于预览/模拟器输出的语义手势计划；无硬件命令。"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .person_guidance import GuidanceCue
from .completion_choreography import COMPLETE_MS,SCREEN_COMPLETE_MS


@dataclass(frozen=True, slots=True)
class GesturePlan:
    cue: str
    pattern: str
    effector: str
    axis: str | None
    direction: int
    repetitions: int
    duration_ms: int
    coordinate_frame: str = "participant_relative_semantic"
    amplitude: float = 1.0
    profile: str = 'semantic'


_DIRECTIONAL_PLANS = {
    GuidanceCue.MOVE_LEFT: ("horizontal", -1),
    GuidanceCue.MOVE_RIGHT: ("horizontal", 1),
    GuidanceCue.MOVE_UP: ("vertical", -1),
    GuidanceCue.MOVE_DOWN: ("vertical", 1),
    GuidanceCue.MOVE_CLOSER: ("depth", 1),
    GuidanceCue.MOVE_FARTHER: ("depth", -1),
}

HOLDING_PATTERNS=frozenset(('confirm_retract_hold','sim_frame_adjust_hold',
                          'sim_frame_confirm_retract_hold'))


def gesture_plan_for(cue: GuidanceCue, *, amplitude: float = 1.,
                     profile: str = 'semantic') -> GesturePlan | None:
    """返回一个象征性的计划，永远不要返回机器人姿势、角度、速度或脚本。"""
    if profile not in ('semantic','screen-coupled','sim-fast','sim-intro'):
        raise ValueError('Unknown gesture profile')
    if profile=='sim-intro':
        if cue is not GuidanceCue.INVITE or amplitude != 1.:
            raise ValueError('Simulation intro supports only the invitation')
        return GesturePlan(cue.value,'curiosity_intro','robot_arm',None,0,1,5600,profile=profile)
    paired=profile in ('screen-coupled','sim-fast')
    if profile=='sim-fast' and cue not in _DIRECTIONAL_PLANS:
        raise ValueError('Fast simulation profile only supports directional gestures')
    if (type(amplitude) not in (int, float) or not math.isfinite(amplitude)
            or not .15 <= amplitude <= 1.):
        raise ValueError("gesture amplitude must be finite and between .15 and 1")
    if cue not in _DIRECTIONAL_PLANS and amplitude != 1.:
        raise ValueError("only directional gestures support variable amplitude")
    if cue in _DIRECTIONAL_PLANS:
        axis, direction = _DIRECTIONAL_PLANS[cue]
        return GesturePlan(
            cue=cue.value,
            pattern="double_sweep_smooth" if paired else "double_sweep",
            effector="robot_arm",
            axis=axis,
            direction=direction,
            repetitions=2,
            duration_ms=3300 if profile=='sim-fast' else 6000 if paired else (2400 if axis == "horizontal" else 2000),
            amplitude=float(amplitude),
            profile=profile,
        )
    if cue is GuidanceCue.INVITE:
        return GesturePlan(cue.value, "open_invitation", "robot_arm", None, 0, 1,
                           3000 if paired else 800,profile=profile)
    if cue is GuidanceCue.NOT_YET:
        return GesturePlan(cue.value, "shake", "robot_arm", "horizontal", 0, 1, 700,profile=profile)
    if cue is GuidanceCue.ARRIVED:
        return GesturePlan(cue.value, "confirm_retract_hold", "robot_arm", None, 0, 2,
                           SCREEN_COMPLETE_MS if paired else COMPLETE_MS,profile=profile)
    if cue is GuidanceCue.SIM_FRAME:
        return GesturePlan(cue.value,"sim_frame_adjust_hold","robot_arm",None,0,1,2400,profile=profile)
    if cue is GuidanceCue.SIM_COMPLETE:
        return GesturePlan(cue.value,"sim_frame_confirm_retract_hold","robot_arm",None,0,2,
                           SCREEN_COMPLETE_MS if paired else COMPLETE_MS,profile=profile)
    return None
