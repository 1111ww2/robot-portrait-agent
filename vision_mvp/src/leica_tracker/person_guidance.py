"""在图像坐标中维护人物引导状态，为提示动作原型提供方向信息。"""

from __future__ import annotations

from dataclasses import dataclass,replace
import math
from enum import Enum

from .continuous import ContinuousController, ControllerConfig


class GuidanceCue(str, Enum):
    INVITE = "INVITE"
    ACKNOWLEDGE = "ACKNOWLEDGE"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    MOVE_CLOSER = "MOVE_CLOSER"
    MOVE_FARTHER = "MOVE_FARTHER"
    NOT_YET = "NOT_YET"
    ARRIVED = "ARRIVED"
    HOLD = "HOLD"
    WAIT = "WAIT"
    SIM_FRAME = "SIM_FRAME"
    SIM_COMPLETE = "SIM_COMPLETE"


Point = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class GuidanceDecision:
    phase: str
    cue: GuidanceCue
    target: Point
    initial_point: Point | None
    current_point: Point | None
    tracking: bool
    reason: str
    gesture_amplitude: float = 1.0
    arrival_progress: float = 0.0
    filtered_point: Point | None = None
    target_tolerances: Point = (.02,.02,.03)
    error_spans: Point = (.25,.20,.15)
    interaction_mode: str = "precise"
    stage_progress: float = 0.
    capture_simulated: bool = False
    framing_verified: bool = False
    composition_frame: tuple[float,float,float,float] | None = None
    composition_style: str = ''
    composition_note: str = ''
    model_status: str = ''
    model_name: str = ''
    model_requests: int = 0
    model_confidence: float = 0.
    model_target: Point | None = None


class PersonGuidanceController:
    """将检测到的人物相对于固定图像目标的偏移转换为提示。输出仅供原型的语义动作层使用；
    归一化坐标未经地面或机器人基座标定，不能直接驱动机器人。"""

    def __init__(
        self,
        config: ControllerConfig | None = None,
        *,
        min_confidence: float = 0.5,
        response_window_ms: int = 1500,
        invite_window_ms: int = 1000,
        arrival_dwell_ms: int = 1000,
        progress_threshold: float = 0.015,
        median_window: int = 1,
        natural_head_height: bool = False,
        repeat_cues: bool = False,
        fast_direction_reversal: bool = False,
        depth_first: bool = False,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        for name, value in (
            ("response_window_ms", response_window_ms),
            ("invite_window_ms", invite_window_ms),
            ("arrival_dwell_ms", arrival_dwell_ms),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if progress_threshold < 0:
            raise ValueError("progress_threshold must be non-negative")

        self.config = config or ControllerConfig()
        self.min_confidence = min_confidence
        self.response_window_ms = response_window_ms
        self.invite_window_ms = invite_window_ms
        self.arrival_dwell_ms = arrival_dwell_ms
        self.progress_threshold = progress_threshold
        self.median_window = median_window
        self.natural_head_height=natural_head_height
        self.repeat_cues=repeat_cues
        self.fast_direction_reversal=fast_direction_reversal
        self.depth_first=depth_first
        self._errors = ContinuousController(self.config,median_window=median_window)
        self.reset()

    def reset(self) -> None:
        """开始新的本地会话，沿用已配置目标。"""
        self._errors = ContinuousController(self.config,median_window=self.median_window)
        self._initial_point: Point | None = None
        self._invite_at_ms: int | None = None
        self._active_axis: str | None = None
        self._active_cue: GuidanceCue | None = None
        self._cue_sent_at_ms: int | None = None
        self._baseline_distance: float | None = None
        self._cue_amplitude = 1.0
        self._arrival_since_ms: int | None = None
        self._arrived = False
        self._was_tracking = False
        self._reversal_candidate = None
        self._reversal_count = 0

    @property
    def target(self) -> Point:
        return (
            self.config.target_x,
            self.config.target_head_y,
            self.config.target_shoulder_width,
        )

    def set_tolerances(self,tolerances: Point) -> None:
        """更新到位判定区域，保留跟踪历史。"""
        if len(tolerances)!=3 or any(not math.isfinite(v) or v<0 for v in tolerances):
            raise ValueError('tolerances must contain three finite nonnegative values')
        self.config=replace(self.config,horizontal_deadband=tolerances[0],
                            vertical_deadband=tolerances[1],scale_deadband=tolerances[2])
        self._errors.config=self.config

    def set_target(self,target: Point) -> None:
        """调整图像目标区域大小，不重启滤波器，也不重启结束流程。"""
        if len(target)!=3 or any(not math.isfinite(v) or not 0<v<1 for v in target):
            raise ValueError('target must contain three finite values in (0, 1)')
        self.config=replace(self.config,target_x=target[0],
                            target_head_y=target[1],target_shoulder_width=target[2])
        self._errors.config=self.config

    def update(
        self,
        observation,
        timestamp_ms: int,
        *,
        mirrored: bool = True,
        invitation_already_sent: bool = False,
    ) -> GuidanceDecision:
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")

        confidence = 0.0 if observation is None else float(
            getattr(observation, "confidence", 0.0)
        )
        too_small=observation is not None and float(observation.shoulder_width) < .03
        if observation is None or confidence < self.min_confidence or too_small:
            self._errors.update(None)
            self._arrival_since_ms = None
            self._was_tracking = False
            self._reversal_candidate = None; self._reversal_count = 0
            if self._arrived:
                return self._decision("ARRIVED", GuidanceCue.HOLD, None, False,
                                      "SESSION_COMPLETE_TRACKING_LOST")
            reason = ("PERSON_LOST" if observation is None else
                      "SUBJECT_TOO_SMALL" if too_small else "LOW_CONFIDENCE")
            return self._decision("TRACKING_LOST", GuidanceCue.HOLD, None, False, reason)

        current: Point = (
            float(observation.center_x),
            float(observation.head_y),
            float(observation.shoulder_width),
        )
        if self.natural_head_height:
            self.set_target((self.target[0],min(.995,max(.005,current[1])),self.target[2]))
        command = self._errors.update(observation)

        if self._initial_point is None:
            self._initial_point = current
            self._invite_at_ms = timestamp_ms
            self._was_tracking = True
            if not invitation_already_sent:
                return self._decision(
                    "INVITING", GuidanceCue.INVITE, current, True, "FIRST_DETECTION"
                )
            self._invite_at_ms=timestamp_ms-self.invite_window_ms

        if self._arrived:
            self._was_tracking = True
            return self._decision(
                "ARRIVED", GuidanceCue.WAIT, current, True, "ARRIVED"
            )

        errors = {
            "horizontal": float(command.right),
            "vertical": 0. if self.natural_head_height else float(command.down),
            "depth": float(command.toward_subject),
        }
        if all(value == 0.0 for value in errors.values()):
            if self._arrival_since_ms is None:
                self._arrival_since_ms = timestamp_ms
            self._active_axis = None
            self._cue_sent_at_ms = None
            self._was_tracking = True
            raw_inside=all(abs(value-target)<=tolerance for value,target,tolerance in zip(
                current,self.target,(self.config.horizontal_deadband,
                                     self.config.vertical_deadband,self.config.scale_deadband)))
            if timestamp_ms - self._arrival_since_ms >= self.arrival_dwell_ms and raw_inside:
                self._arrived = True
                return self._decision(
                    "ARRIVED", GuidanceCue.ARRIVED, current, True, "TARGET_STABLE"
                )
            return self._decision(
                "SETTLING", GuidanceCue.WAIT, current, True, "ARRIVAL_DWELL",
                arrival_progress=min(.99,(timestamp_ms-self._arrival_since_ms)/self.arrival_dwell_ms),
            )

        self._arrival_since_ms = None

        if (
            self._invite_at_ms is not None
            and timestamp_ms - self._invite_at_ms < self.invite_window_ms
        ):
            self._was_tracking = True
            return self._decision(
                "INVITING", GuidanceCue.WAIT, current, True, "WAITING_AFTER_INVITATION"
            )

        if not self._was_tracking:
            self._active_axis = None
            self._cue_sent_at_ms = None

        self._was_tracking = True
        next_axis, next_cue = self._select_cue(errors, mirrored)

        if self._active_axis is None or self._cue_sent_at_ms is None:
            self._send_cue(next_axis, next_cue, errors, timestamp_ms)
            return self._decision(
                "GUIDING", next_cue, current, True, self._reason_for(next_axis)
            )

        if timestamp_ms - self._cue_sent_at_ms < self.response_window_ms:
            if self.fast_direction_reversal:
                # 用当前运动轴的原始测量值及时识别过冲。须连续两帧超出接受区及余量，单帧噪声不能取消当前动作。
                raw = {'horizontal': current[0]-self.target[0],
                       'vertical': current[1]-self.target[1],
                       'depth': self.target[2]-current[2]}
                bands = {'horizontal': self.config.horizontal_deadband,
                         'vertical': self.config.vertical_deadband,
                         'depth': self.config.scale_deadband}
                spans = {'horizontal': self.config.horizontal_span,
                         'vertical': self.config.vertical_span,
                         'depth': self.config.scale_span}
                axis = self._active_axis
                isolated = {key: (value if key == axis else 0.) for key,value in raw.items()}
                raw_axis, raw_cue = self._select_cue(isolated, mirrored)
                if abs(raw[axis]) > bands[axis]+.015 and raw_axis == axis and raw_cue is not self._active_cue:
                    if self._reversal_candidate == raw_cue:
                        self._reversal_count += 1
                    else:
                        self._reversal_candidate = raw_cue; self._reversal_count = 1
                    if self._reversal_count >= 2:
                        reversed_errors = dict(errors)
                        reversed_errors[axis] = max(-1.,min(1.,raw[axis]/spans[axis]))
                        self._send_cue(axis,raw_cue,reversed_errors,timestamp_ms)
                        return self._decision('GUIDING',raw_cue,current,True,self._reason_for(axis))
                else:
                    self._reversal_candidate = None; self._reversal_count = 0
            return self._decision(
                "WAITING_FOR_RESPONSE",
                GuidanceCue.WAIT,
                current,
                True,
                "WAITING_FOR_RESPONSE",
            )

        if next_cue is not self._active_cue:
            self._send_cue(next_axis, next_cue, errors, timestamp_ms)
            return self._decision(
                "GUIDING", next_cue, current, True, self._reason_for(next_axis)
            )

        current_distance = abs(errors[self._active_axis])
        made_progress = (
            self._baseline_distance is not None
            and current_distance + self.progress_threshold < self._baseline_distance
        )
        if made_progress or self.repeat_cues:
            self._send_cue(next_axis, next_cue, errors, timestamp_ms)
            return self._decision(
                "GUIDING", next_cue, current, True, self._reason_for(next_axis)
            )

        return self._decision(
            "PAUSED_FOR_PARTICIPANT",
            GuidanceCue.WAIT,
            current,
            True,
            "NO_PROGRESS_WAITING_FOR_PARTICIPANT",
        )

    def _send_cue(
        self,
        axis: str,
        cue: GuidanceCue,
        errors: dict[str, float],
        timestamp_ms: int,
    ) -> None:
        self._reversal_candidate = None; self._reversal_count = 0
        self._active_axis = axis
        self._active_cue = cue
        self._cue_sent_at_ms = timestamp_ms
        self._baseline_distance = abs(errors[axis])
        self._cue_amplitude = max(.35, min(1., abs(errors[axis])))

    def _select_cue(
        self,
        errors: dict[str, float],
        mirrored: bool,
    ) -> tuple[str, GuidanceCue]:
        axis = 'depth' if self.depth_first and errors['depth'] != 0 else max(errors, key=lambda key: abs(errors[key]))
        error = errors[axis]
        if axis == "horizontal":
            if mirrored:
                cue = GuidanceCue.MOVE_LEFT if error > 0 else GuidanceCue.MOVE_RIGHT
            else:
                cue = GuidanceCue.MOVE_RIGHT if error > 0 else GuidanceCue.MOVE_LEFT
        elif axis == "vertical":
            cue = GuidanceCue.MOVE_UP if error > 0 else GuidanceCue.MOVE_DOWN
        else:
            cue = GuidanceCue.MOVE_CLOSER if error > 0 else GuidanceCue.MOVE_FARTHER
        return axis, cue

    @staticmethod
    def _reason_for(axis: str) -> str:
        return {
            "horizontal": "OFF_TARGET_HORIZONTAL",
            "vertical": "OFF_TARGET_VERTICAL",
            "depth": "OFF_TARGET_APPARENT_DISTANCE",
        }[axis]

    def _decision(
        self,
        phase: str,
        cue: GuidanceCue,
        current: Point | None,
        tracking: bool,
        reason: str,
        *,
        arrival_progress: float = 0.,
    ) -> GuidanceDecision:
        filtered=self._errors.filtered_point
        if self.natural_head_height and filtered is not None:
            filtered=(filtered[0],self.target[1],filtered[2])
        return GuidanceDecision(
            phase=phase,
            cue=cue,
            target=self.target,
            initial_point=self._initial_point,
            current_point=current,
            tracking=tracking,
            reason=reason,
            gesture_amplitude=(self._cue_amplitude if cue.value.startswith("MOVE_") else 1.),
            arrival_progress=1. if phase == "ARRIVED" else arrival_progress,
            filtered_point=filtered,
            target_tolerances=(self.config.horizontal_deadband,self.config.vertical_deadband,
                               self.config.scale_deadband),
            error_spans=(self.config.horizontal_span,self.config.vertical_span,self.config.scale_span),
        )
