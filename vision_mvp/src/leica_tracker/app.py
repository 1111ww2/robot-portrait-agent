"""Pocket 3 命令行上半身连续跟踪器。"""

from __future__ import annotations

import os

import argparse
import json
import math
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import IO, Any

import cv2
import numpy as np

from .continuous import CameraRelativeCommand, ContinuousController, ControllerConfig
from .gesture_preview import normalized_offset_for
from .completion_choreography import completion_pose_at
from .coarse_guidance import CoarsePortraitController
from .gemini_framing import GeminiFramingClient,FramingError,DEFAULT_MODEL,DEFAULT_ENV,load_api_key
from .gemini_guidance import GeminiGuidanceController
from .claude_agent import ClaudeInteractionController,ClaudeInteractionClient,FacePresence
from .claude_framing import ClaudeFramingClient, DEFAULT_MODEL as CLAUDE_MODEL, load_api_key as load_claude_key
from .guidance_feedback import draw_guidance_feedback
from .motion import MotionEnvelope, MotionIntent, MotionIntentMapper
from .person_guidance import GuidanceDecision, GuidanceCue, PersonGuidanceController
from .pose import PoseDetection, PoseTracker
from .robot_gestures import GesturePlan, gesture_plan_for, HOLDING_PATTERNS
from .simulation import virtual_frame_pose
from .zones import Zone

DEFAULT_MODEL_PATH = Path(__file__).parents[2] / "models" / "pose_landmarker_lite.task"
GUIDANCE_WINDOW = "Scene Webcam - r reset / q exit"


class CameraReadError(RuntimeError):
    """连续读取相机帧失败时抛出此异常。"""


@dataclass(slots=True)
class AppConfig:
    mode: str = "guide-person"
    guidance_profile: str = "precise"
    single_plan: bool = False
    gemini_model: str = DEFAULT_MODEL
    gemini_env_file: Path = DEFAULT_ENV
    claude_model: str = CLAUDE_MODEL
    claude_effort: str = "medium"
    claude_env_file: Path = DEFAULT_ENV
    guidance_stream: bool = False
    camera: int = 0
    camera_source: str = 'webcam'
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    source_mirrored: bool = False
    rotate: int = 0
    headless: bool = False
    horizontal_only: bool = False
    duration: float = 0.0
    max_frames: int = 0
    min_confidence: float = 0.5
    max_read_failures: int = 15
    target_x: float = 0.50
    target_head_y: float = 0.25
    target_shoulder_width: float = 0.30
    smoothing_alpha: float = 0.25
    horizontal_deadband: float = 0.02
    vertical_deadband: float = 0.02
    scale_deadband: float = 0.03
    emit_every_frames: int = 3
    max_right_m: float = 0.20
    max_down_m: float = 0.12
    max_toward_subject_m: float = 0.18
    max_step_m: float = 0.01
    model_path: Path = field(default_factory=lambda: DEFAULT_MODEL_PATH)


@dataclass(frozen=True, slots=True)
class RunStats:
    frames_processed: int
    target_events: int
    elapsed_seconds: float


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _fraction(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def _positive_fraction(value: str) -> float:
    number = _fraction(value)
    if number == 0.0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leica-tracker",
        description=(
            "Guide a participant to one fixed webcam target or preview "
            "the virtual frame-follow mode."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("guide-person", "follow-frame"),
        default="guide-person",
        help="semantic person guidance (default) or the existing virtual frame-follow mode",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="OpenCV camera index (default: 0)"
    )
    parser.add_argument('--guidance-profile',choices=('precise','coarse-demo','frame-guide','gemini-guide','claude-guide','claude-agent'),default='precise',
                        help='precise diagnostic, forgiving frame guidance, or scripted coarse demo')
    parser.add_argument('--gemini-model',default=DEFAULT_MODEL)
    parser.add_argument('--single-plan',action='store_true',help='one reference-frame Claude plan per explicit start/replan; local tracking thereafter')
    parser.add_argument('--gemini-env-file',type=Path,default=DEFAULT_ENV)
    parser.add_argument('--claude-model',default=CLAUDE_MODEL)
    parser.add_argument('--claude-effort',choices=('medium','high'),default='medium')
    parser.add_argument('--claude-env-file',type=Path,default=DEFAULT_ENV)
    parser.add_argument('--camera-source',choices=('webcam','ipad'),default='webcam')
    parser.add_argument('--guest-display',action='store_true',help='paired iPad visitor view; starts paused')
    parser.add_argument('--guest-host',choices=('127.0.0.1','0.0.0.0'),default='127.0.0.1')
    parser.add_argument('--guest-port',type=_positive_int,default=8767)
    parser.add_argument("--guidance-stream", action="store_true",
                        help="emit fresh guidance heartbeats for the local simulator")
    parser.add_argument("--width", type=_positive_int, default=1280)
    parser.add_argument("--height", type=_positive_int, default=720)
    parser.add_argument("--fps", type=_positive_int, default=30)
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use a participant-aligned mirror preview before detection",
    )
    parser.add_argument(
        "--source-mirrored",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="camera input is already mirrored; normalize it before applying --mirror",
    )
    parser.add_argument("--max-right-m", type=_positive_float, default=0.20)
    parser.add_argument("--max-down-m", type=_positive_float, default=0.12)
    parser.add_argument("--max-toward-subject-m", type=_positive_float, default=0.18)
    parser.add_argument("--max-step-m", type=_positive_float, default=0.01)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument(
        "--headless", action="store_true", help="do not open a preview window"
    )
    parser.add_argument(
        "--horizontal-only", action="store_true",
        help="follow-frame only: lock vertical and depth commands to zero",
    )
    parser.add_argument(
        "--duration", type=_nonnegative_float, default=0.0, help="stop after N seconds"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="stop after N processed frames"
    )
    parser.add_argument("--min-confidence", type=_fraction, default=0.5)
    parser.add_argument("--max-read-failures", type=_positive_int, default=15)
    parser.add_argument("--target-x", type=_fraction, default=0.50)
    parser.add_argument("--target-head-y", type=_fraction, default=0.25)
    parser.add_argument(
        "--target-shoulder-width", type=_positive_fraction, default=0.30
    )
    parser.add_argument("--smoothing-alpha", type=_positive_fraction, default=0.25)
    parser.add_argument("--horizontal-deadband", type=_fraction, default=0.02)
    parser.add_argument("--vertical-deadband", type=_fraction, default=0.02)
    parser.add_argument("--scale-deadband", type=_fraction, default=0.03)
    parser.add_argument(
        "--emit-every-frames",
        type=_positive_int,
        default=3,
        help="emit one tracking target every N frames (default: 3, about 10 Hz)",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    return parser


def serialize_zone_event(zone: Zone, x: float | None, timestamp_ms: int) -> str:
    payload = {
        "event": "zone_change",
        "timestamp_ms": timestamp_ms,
        "x": None if x is None else round(float(x), 4),
        "zone": zone.value,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def serialize_tracking_event(
    command: CameraRelativeCommand,
    intent: MotionIntent,
    timestamp_ms: int,
    *, source_monotonic_ns: int | None = None, observation=None,
) -> str:
    payload = {
        "event": "tracking_target",
        "timestamp_ms": timestamp_ms,
        "tracking": command.tracking,
        "reason": command.reason,
        "coordinate_frame": "camera_relative_normalized",
        "command": {
            "right": round(float(command.right), 4),
            "down": round(float(command.down), 4),
            "toward_subject": round(float(command.toward_subject), 4),
        },
        "motion_intent_m": {
            "right": round(float(intent.right_m), 4),
            "down": round(float(intent.down_m), 4),
            "toward_subject": round(float(intent.toward_subject_m), 4),
            "hold": intent.hold,
        },
    }
    if source_monotonic_ns is not None:
        payload["source_monotonic_ns"] = source_monotonic_ns
    if observation is not None:
        payload["image_observation"] = {
            "shoulder_width": round(float(observation.shoulder_width), 6),
            "confidence": round(float(getattr(observation, "confidence", 0.0)), 6),
        }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _point_payload(
    point: tuple[float, float, float] | None,
) -> dict[str, float] | None:
    if point is None:
        return None
    return {
        "center_x": round(float(point[0]), 6),
        "head_y": round(float(point[1]), 6),
        "shoulder_width": round(float(point[2]), 6),
    }


def serialize_guidance_event(
    decision: GuidanceDecision,
    gesture: GesturePlan | None,
    timestamp_ms: int,
    *,
    source_monotonic_ns: int | None = None,
) -> str:
    """序列化提示状态变化，不保存或输出图像。"""
    payload: dict[str, Any] = {
        "event": "person_guidance",
        "timestamp_ms": timestamp_ms,
        "coordinate_frame": "normalized_camera_image",
        "phase": decision.phase,
        "cue": decision.cue.value,
        "tracking": decision.tracking,
        "reason": decision.reason,
        "arrival_progress": getattr(decision,"arrival_progress",0.),
        "interaction_mode": getattr(decision,"interaction_mode","precise"),
        "stage_progress": getattr(decision,"stage_progress",0.),
        "capture_simulated": getattr(decision,"capture_simulated",False),
        "framing_verified": getattr(decision,"framing_verified",False),
        "composition_frame": getattr(decision,'composition_frame',None),
        "composition_style": getattr(decision,'composition_style',''),
        "model_status": getattr(decision,'model_status',''),
        "model_name": getattr(decision,'model_name',''),
        "model_requests": getattr(decision,'model_requests',0),
        "model_confidence": getattr(decision,'model_confidence',0.),
        "robot_actuated": False,
        "points": {
            "target": _point_payload(decision.target),
            "initial": _point_payload(decision.initial_point),
            "current": _point_payload(decision.current_point),
            "filtered": _point_payload(getattr(decision,"filtered_point",None)),
        },
        "gesture": None if gesture is None else asdict(gesture),
    }
    if source_monotonic_ns is not None:
        payload["source_monotonic_ns"] = source_monotonic_ns
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _rotate_frame(frame: Any, degrees: int) -> Any:
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _draw_detection(frame: Any, detection: PoseDetection | None) -> None:
    if detection is None:
        return
    height, width = frame.shape[:2]
    visible = [
        point.visibility >= 0.5
        and point.presence >= 0.5
        and 0.0 <= point.x <= 1.0
        and 0.0 <= point.y <= 1.0
        for point in detection.landmarks
    ]

    try:
        from mediapipe.tasks.python.vision import PoseLandmarksConnections

        connections = PoseLandmarksConnections.POSE_LANDMARKS
    except (ImportError, AttributeError):
        connections = ()

    for connection in connections:
        start, end = connection.start, connection.end
        if (
            start >= len(visible)
            or end >= len(visible)
            or not visible[start]
            or not visible[end]
        ):
            continue
        p1 = detection.landmarks[start]
        p2 = detection.landmarks[end]
        cv2.line(
            frame,
            (int(p1.x * width), int(p1.y * height)),
            (int(p2.x * width), int(p2.y * height)),
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
    for point, is_visible in zip(detection.landmarks, visible, strict=True):
        if is_visible:
            cv2.circle(
                frame,
                (int(point.x * width), int(point.y * height)),
                3,
                (30, 30, 30),
                -1,
            )
            cv2.circle(
                frame,
                (int(point.x * width), int(point.y * height)),
                2,
                (255, 255, 255),
                -1,
            )

    cv2.circle(
        frame,
        (int(detection.center_x * width), int(detection.head_y * height)),
        10,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )


def _draw_overlay(
    frame: Any,
    detection: PoseDetection | None,
    command: CameraRelativeCommand,
    intent: MotionIntent,
    target: ControllerConfig,
    envelope: MotionEnvelope,
    fps: float,
) -> None:
    height, width = frame.shape[:2]
    guide_half_width = int(target.target_shoulder_width * width)
    guide_center_x = int(target.target_x * width)
    guide_top = int(max(0.02, target.target_head_y - 0.17) * height)
    guide_bottom = int(min(0.95, target.target_head_y + 0.63) * height)
    guide_color = (80, 230, 150) if command.tracking else (120, 120, 120)
    cv2.rectangle(
        frame,
        (guide_center_x - guide_half_width, guide_top),
        (guide_center_x + guide_half_width, guide_bottom),
        guide_color,
        2,
        cv2.LINE_AA,
    )
    cv2.circle(
        frame,
        (guide_center_x, int(target.target_head_y * height)),
        8,
        guide_color,
        2,
        cv2.LINE_AA,
    )

    virtual = virtual_frame_pose(intent, target, envelope)
    virtual_color = (0, 190, 255) if not virtual.hold else (0, 125, 255)
    virtual_center_x = int(virtual.center_x * width)
    virtual_half_width = int(virtual.half_width * width)
    virtual_head_y = int(virtual.head_y * height)
    cv2.line(
        frame,
        (guide_center_x, int(target.target_head_y * height)),
        (virtual_center_x, virtual_head_y),
        virtual_color,
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        frame,
        (virtual_center_x - virtual_half_width, int(virtual.top * height)),
        (virtual_center_x + virtual_half_width, int(virtual.bottom * height)),
        virtual_color,
        4,
        cv2.LINE_AA,
    )
    cv2.circle(
        frame,
        (virtual_center_x, virtual_head_y),
        10,
        virtual_color,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "VIRTUAL ROBOT FRAME",
        (
            max(8, virtual_center_x - virtual_half_width),
            max(24, int(virtual.top * height) - 9),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        virtual_color,
        2,
        cv2.LINE_AA,
    )

    _draw_detection(frame, detection)
    confidence = detection.confidence if detection else 0.0
    cv2.rectangle(frame, (18, height - 144), (760, height - 18), (18, 18, 18), -1)
    cv2.putText(
        frame,
        f"{command.reason}  confidence {confidence:.2f}  {fps:.1f} fps",
        (34, height - 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        guide_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            f"right {command.right:+.2f}   down {command.down:+.2f}   "
            f"toward subject {command.toward_subject:+.2f}"
        ),
        (34, height - 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            f"VIRTUAL TCP  R {intent.right_m * 1000:+.0f} mm   "
            f"D {intent.down_m * 1000:+.0f} mm   "
            f"T {intent.toward_subject_m * 1000:+.0f} mm   "
            f"{'HOLD' if intent.hold else 'MOVING'}"
        ),
        (34, height - 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        virtual_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "GREEN = TARGET COMPOSITION   YELLOW = SIMULATED ROBOT-MOUNTED FRAME",
        (34, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def _draw_guidance_overlay(
    frame: Any,
    detection: PoseDetection | None,
    decision: GuidanceDecision,
    fps: float,
    gesture: GesturePlan | None = None,
    gesture_elapsed_ms: int = 0,
    mirrored: bool = True,
) -> None:
    """绘制固定目标、跟踪到的人物和模拟机械臂动画。"""
    height, width = frame.shape[:2]
    coarse=getattr(decision,'interaction_mode','precise')=='coarse-demo'
    ai=getattr(decision,'interaction_mode','precise') in ('gemini-guide','claude-guide','claude-agent')
    frame_guide=getattr(decision,'interaction_mode','precise') in ('frame-guide','gemini-guide','claude-guide','claude-agent')
    if ai and not decision.composition_frame:
        _draw_detection(frame,detection)
        cv2.rectangle(frame,(18,24),(width-18,76),(18,22,24),-1)
        banner=('PERSON ACQUIRED / ROBOT RESPONDS / GEMINI PLANS IN BACKGROUND'
                if decision.tracking and decision.cue in (GuidanceCue.INVITE,GuidanceCue.WAIT) else
                'GEMINI / CAMERA LIVE / GUIDANCE PAUSED')
        cv2.putText(frame,banner,
                    (30,57),cv2.FONT_HERSHEY_SIMPLEX,.57,(40,170,255),1,cv2.LINE_AA)
        draw_guidance_feedback(frame,decision,mirrored=mirrored)
        return
    target_x, target_y, target_width = decision.target
    center_x = int(target_x * width)
    head_y = int(target_y * height)
    half_width = int(target_width * width)
    top = int(max(0.02, target_y - 0.17) * height)
    bottom = int(min(0.84, target_y + 0.63) * height)
    target_color = (80, 230, 150)
    frame_left, frame_right = center_x-half_width, center_x+half_width
    if ai:
        left,upper,right,lower=decision.composition_frame
        frame_left,frame_right=int(left*width),int(right*width)
        top,bottom=int(upper*height),int(lower*height)
    tolerances=getattr(decision,'target_tolerances',(.02,.02,.03))
    tx,ty=int(tolerances[0]*width),int(tolerances[1]*height)
    if coarse:
        frame_left,frame_right=center_x-tx,center_x+tx
        top,bottom=head_y-ty,head_y+ty
    cv2.rectangle(frame,(frame_left,top),(frame_right,bottom),(15,15,15),10)
    cv2.rectangle(
        frame,
        (frame_left, top),
        (frame_right, bottom),
        target_color,
        6,
    )
    target_label = ('AI FRAME / '+decision.composition_style.upper().replace('_',' ') if ai else
                    'COARSE PHOTO AREA' if coarse else
                    'TARGET FRAME - POSITION GUIDE' if frame_guide else
                    'TARGET REACHED' if decision.phase == 'ARRIVED' else 'SCREEN TARGET ONLY')
    cv2.rectangle(frame,(frame_left-4,max(0,top-31)),(frame_left+335,top-5),(15,15,15),-1)
    cv2.putText(frame,target_label,(frame_left+8,max(20,top-12)),
                cv2.FONT_HERSHEY_SIMPLEX,.58,target_color,1,cv2.LINE_AA)
    shoulder_y=int((target_y+.19)*height)
    if coarse:
        cv2.putText(frame,'HEAD ANYWHERE IN THIS AREA',(frame_left+10,top+28),
                    cv2.FONT_HERSHEY_SIMPLEX,.43,target_color,1,cv2.LINE_AA)
    else:
        head_radius=max(10,int(width*.021))
        cv2.rectangle(frame,(center_x-tx,head_y-ty),(center_x+tx,head_y+ty),(45,90,60),-1)
        cv2.rectangle(frame,(center_x-tx,head_y-ty),(center_x+tx,head_y+ty),target_color,2)
        if frame_guide:
            cv2.putText(frame,'HEAD INSIDE THIS AREA',(center_x-tx,head_y-ty-12),
                        cv2.FONT_HERSHEY_SIMPLEX,.43,target_color,1,cv2.LINE_AA)
        else:
            cv2.circle(frame,(center_x,head_y),head_radius,(15,15,15),8,cv2.LINE_AA)
            cv2.circle(frame,(center_x,head_y),head_radius,target_color,4,cv2.LINE_AA)
            cv2.putText(frame,'HEAD HERE',(center_x+head_radius+10,max(20,head_y-head_radius-12)),
                        cv2.FONT_HERSHEY_SIMPLEX,.48,target_color,1,cv2.LINE_AA)
        shoulder_half=int(target_width*width/2)
        if frame_guide:
            # 两个范围分别约束两端肩部位置，并不指定唯一肩宽。
            inner=int((target_width-tolerances[2])*width/2)
            outer=int((target_width+tolerances[2])*width/2)
            for a,b in ((center_x-outer,center_x-inner),(center_x+inner,center_x+outer)):
                cv2.rectangle(frame,(a,shoulder_y-12),(b,shoulder_y+12),(45,90,60),-1)
        if frame_guide:
            cv2.line(frame,(center_x-inner,shoulder_y),(center_x+inner,shoulder_y),target_color,1)
        else:
            for color,thickness in [((15,15,15),8),(target_color,4)]:
                cv2.line(frame,(center_x-shoulder_half,shoulder_y),
                          (center_x+shoulder_half,shoulder_y),color,thickness,cv2.LINE_AA)
                for x in (center_x-shoulder_half,center_x+shoulder_half):
                    cv2.line(frame,(x,shoulder_y-10),(x,shoulder_y+10),color,thickness,cv2.LINE_AA)
        cv2.putText(frame,'SHOULDER WIDTH: GREEN BANDS = OK' if frame_guide else 'MATCH SHOULDER WIDTH',(center_x-shoulder_half,shoulder_y+28),
                    cv2.FONT_HERSHEY_SIMPLEX,.43,target_color,1,cv2.LINE_AA)
    _draw_detection(frame, detection)

    if decision.current_point is not None and decision.tracking:
        current_x,current_y,current_width=getattr(decision,'filtered_point',None) or decision.current_point
        current_px,current_py=int(current_x*width),int(current_y*height)
        cv2.putText(frame,"YOU",(current_px+18,max(20,current_py-12)),
                    cv2.FONT_HERSHEY_SIMPLEX,.46,(0,190,255),1,cv2.LINE_AA)
        if not coarse:
            measured_half=int(current_width*width/2)
            measured_y=shoulder_y+57
            comparison_x=center_x if frame_guide else current_px
            cv2.line(frame,(comparison_x-measured_half,measured_y),
                      (comparison_x+measured_half,measured_y),(0,190,255),3,cv2.LINE_AA)
            cv2.putText(frame,'YOUR WIDTH (SIZE ONLY)' if frame_guide else 'YOUR SHOULDER WIDTH',(comparison_x-measured_half,measured_y+23),
                        cv2.FONT_HERSHEY_SIMPLEX,.40,(0,190,255),1,cv2.LINE_AA)
        cv2.circle(
            frame,
            (int(current_x * width), int(current_y * height)),
            12,
            (0, 190, 255),
            3,
        )

    horizontal, vertical, depth = normalized_offset_for(
        gesture, gesture_elapsed_ms
    )
    shoulder = (int(width * 0.89), int(height * 0.78))
    elbow = (
        int(width * 0.86 + horizontal * width * 0.025),
        int(height * 0.70 + vertical * height * 0.025),
    )
    wrist = (
        int(width * 0.80 + horizontal * width * 0.055 + depth * width * 0.02),
        int(height * 0.62 + vertical * height * 0.06),
    )
    arm_color = (255, 160, 40)
    cv2.rectangle(
        frame,
        (int(width * 0.865), int(height * 0.80)),
        (int(width * 0.915), int(height * 0.91)),
        (90, 90, 90),
        -1,
    )
    cv2.line(frame, shoulder, elbow, arm_color, 20, cv2.LINE_AA)
    cv2.line(frame, elbow, wrist, arm_color, 15, cv2.LINE_AA)
    cv2.circle(frame, shoulder, 13, (220, 220, 220), -1, cv2.LINE_AA)
    cv2.circle(frame, elbow, 12, arm_color, -1, cv2.LINE_AA)
    cv2.circle(frame, wrist, 10, (245, 245, 245), -1, cv2.LINE_AA)
    if gesture is not None and gesture.pattern in ('confirm_retract_hold','sim_frame_confirm_retract_hold'):
        _,_,pitch=completion_pose_at(gesture_elapsed_ms,framed=gesture.pattern=='sim_frame_confirm_retract_hold')
        tool_tip=(wrist[0]+int(45*math.sin(pitch)),wrist[1]+int(45*math.cos(pitch)))
        cv2.line(frame,wrist,tool_tip,arm_color,8,cv2.LINE_AA)
    gesture_label = (
        gesture.cue
        if gesture is not None and 0 <= gesture_elapsed_ms < gesture.duration_ms
        else decision.cue.value
    )
    cv2.putText(
        frame,
        f"SIM ARM {gesture_label}" + (f" {gesture.amplitude:.0%}" if gesture else ""),
        (int(width * 0.70), int(height * 0.95)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        arm_color,
        1,
        cv2.LINE_AA,
    )

    confidence = detection.confidence if detection is not None else 0.0
    cv2.rectangle(frame, (18, height - 106), (min(830,width-18), height - 10), (18, 18, 18), -1)
    cv2.putText(frame, f"{decision.reason}", (34,height-83),
                cv2.FONT_HERSHEY_SIMPLEX, .46, (230,230,230), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"{decision.phase}  {decision.cue.value}  confidence {confidence:.2f}  {fps:.1f} fps",
        (34, height - 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        target_color if decision.tracking else (140, 140, 140),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "GREEN = TARGET   YELLOW = CONTROL POSITION   LEFT/RIGHT = YOUR DIRECTIONS",
        (34, height - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        ('GEMINI PLAN / FIXED CAMERA / PHYSICAL FRAME NOT CALIBRATED' if ai else
         'PERSON READY IS NOT FINAL FRAMING / CAPTURE IS SIMULATED' if coarse else
         'ALL 3 CHECKS OK FOR 1 s / PHYSICAL FRAME NOT CALIBRATED'),
        (34, height - 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    draw_guidance_feedback(frame,decision,mirrored=mirrored)


class SimulatorPhotoLatch:
    """到位后保留所选结果，不再发出运动指令。"""
    def __init__(self, *, enabled=False):
        self.enabled=enabled
        self.decision=None

    def reset(self):
        self.decision=None

    def apply(self,decision, *, active):
        if not self.enabled:return decision
        if not active:
            self.reset();return decision
        if self.decision is None and decision.phase=='ARRIVED' and decision.cue is GuidanceCue.ARRIVED:
            self.decision=replace(decision,cue=GuidanceCue.HOLD,tracking=False,
                                  reason='TARGET_LATCHED')
        return self.decision if self.decision is not None else decision


class PhysicalPhotoLatch:
    """到位时请求 HOLD，等待后端实际观测到停止。"""
    def __init__(self,display):
        self.display=display;self.decision=None;self.source_ns=None;self.ready=False

    def reset(self):self.decision=None;self.source_ns=None;self.ready=False

    def apply(self,decision, *, active,source_ns):
        if not active:
            self.reset();return decision
        if self.decision is None and decision.phase=='ARRIVED' and decision.cue is GuidanceCue.ARRIVED:
            try:self.display.arm_physical_photo(source_ns)
            except ValueError:
                return replace(decision,phase='SETTLING',cue=GuidanceCue.HOLD,tracking=False,
                               reason='PHYSICAL_BACKEND_UNAVAILABLE')
            self.source_ns=source_ns
            self.decision=replace(decision,phase='SETTLING',cue=GuidanceCue.HOLD,
                                  tracking=False,reason='TARGET_PENDING_PHYSICAL_STOP')
        if self.decision is None:return decision
        if not self.ready and self.display.physical_photo_ready(self.source_ns):
            self.ready=True
            self.decision=replace(self.decision,phase='ARRIVED',reason='TARGET_LATCHED')
        return self.decision


class SimulatorGestureScale:
    """根据归一化构图误差选择稳定的仿真提示幅度。"""
    def __init__(self):
        self.mode=None

    def reset(self):
        self.mode=None

    def select(self, error):
        if type(error) not in (int,float) or not math.isfinite(error) or not 0<=error<=1:
            raise ValueError('Expected a finite normalized gesture error')
        if error<=.5:self.mode='near'
        elif error>=.65:self.mode='far'
        elif self.mode is None:self.mode='far'
        return .65 if self.mode=='near' else 1.

    def current(self):
        return .65 if self.mode=='near' else 1.


class SimulatorDirectionRecovery:
    """只重发被中断且当前仍需要的本地仿真方向提示。"""
    def __init__(self, *, enabled=False):
        self.enabled=enabled
        self.pending=None
        self.retries=0
        self.missing_since=None

    def reset(self):
        self.pending=None
        self.retries=0
        self.missing_since=None

    def new_gesture(self):
        self.reset()

    def feedback_interrupted(self,available, *, source_ns,now_ns):
        if (not self.enabled or available is not False or type(source_ns) is not int
                or type(now_ns) is not int or now_ns-source_ns<=300_000_000):
            self.missing_since=None;return False
        if self.missing_since is None:
            self.missing_since=now_ns;return False
        return now_ns-self.missing_since>=200_000_000

    def interrupt(self,gesture,target, *, active):
        if not self.enabled or not active:
            self.reset();return
        if (gesture is not None and getattr(gesture,'cue','').startswith('MOVE_')
                and isinstance(target,(tuple,list)) and len(target)==3
                and all(type(v) in (int,float) and math.isfinite(v) for v in target)):
            self.pending=(gesture.cue,tuple(target))

    def apply(self,decision, *, active,source_ns,now_ns,mirrored):
        if not self.enabled or not active:
            self.reset();return decision
        if self.pending is None:return decision
        if decision.phase in ('ARRIVED','SETTLING','PHYSICAL_REFERENCE_HOLD'):
            self.reset();return decision
        if decision.cue not in (GuidanceCue.WAIT,GuidanceCue.HOLD):
            self.reset();return decision
        if (decision.cue is not GuidanceCue.WAIT or not decision.tracking
                or decision.phase not in ('WAITING_FOR_RESPONSE','PAUSED_FOR_PARTICIPANT')
                or type(source_ns) is not int or source_ns<=0 or type(now_ns) is not int
                or not 0<=now_ns-source_ns<=300_000_000):return decision
        cue,target=self.pending
        current=decision.current_point;tolerances=decision.target_tolerances
        if (not isinstance(decision.target,(tuple,list)) or tuple(decision.target)!=target):
            self.reset();return decision
        if (not all(isinstance(v,(tuple,list)) and len(v)==3 for v in (current,tolerances))
                or not all(type(x) in (int,float) and math.isfinite(x) for v in (current,tolerances) for x in v)):
            return decision
        error=(current[0]-target[0],current[1]-target[1],target[2]-current[2])
        axis=0 if cue in ('MOVE_LEFT','MOVE_RIGHT') else 1 if cue in ('MOVE_UP','MOVE_DOWN') else 2
        if tolerances[axis]<0 or abs(error[axis])<=tolerances[axis]:
            self.reset();return decision
        if axis==0:
            desired=('MOVE_LEFT' if error[axis]>0 else 'MOVE_RIGHT') if mirrored else ('MOVE_RIGHT' if error[axis]>0 else 'MOVE_LEFT')
        elif axis==1:desired='MOVE_UP' if error[axis]>0 else 'MOVE_DOWN'
        else:desired='MOVE_CLOSER' if error[axis]>0 else 'MOVE_FARTHER'
        self.pending=None;self.missing_since=None
        if desired!=cue:return decision
        if self.retries>=2:return decision
        self.retries+=1
        return replace(decision,phase='GUIDING',cue=GuidanceCue(cue),reason='SIMULATOR_DIRECTION_RECOVERY')


class SimulatorArrivalRecovery:
    """仅在收到新仿真帧时重新启用被中断的确认动作。"""
    def __init__(self, *, enabled=False):
        self.enabled=enabled
        self.pending=False

    def reset(self):
        self.pending=False

    def interrupt(self,gesture, *, active):
        if not self.enabled or not active:
            self.reset()
        elif (gesture is not None and getattr(gesture,'cue',None)=='ARRIVED'
              and getattr(gesture,'pattern',None)=='confirm_retract_hold'):
            self.pending=True

    def apply(self,decision, *, active,source_ns,now_ns):
        if not self.enabled or not active:
            self.reset();return decision
        if (not self.pending or decision.phase!='ARRIVED' or decision.cue is not GuidanceCue.WAIT
                or not decision.tracking or type(source_ns) is not int or source_ns<=0
                or type(now_ns) is not int or not 0<=now_ns-source_ns<=300_000_000):
            return decision
        current=decision.current_point;target=decision.target;tolerance=decision.target_tolerances
        if not all(isinstance(v,(list,tuple)) and len(v)==3 for v in (current,target,tolerance)):
            return decision
        if (not all(type(v) in (float,int) and math.isfinite(v)
                    for group in (current,target,tolerance) for v in group)
                or any(t<0 or abs(c-p)>t for c,p,t in zip(current,target,tolerance))):
            return decision
        self.pending=False
        return replace(decision,cue=GuidanceCue.ARRIVED,reason='SIMULATOR_REFERENCE_RECOVERY')


def guidance_controller_options(config, *, screen_coupled=False, physical_mode=False, sim_fast=False):
    """帧循环和原生验证共用感知、交互参数。
    原生动作恢复时会清除本地响应冷却状态，因此沿用既定响应窗口，避免验证参与者与正式流程采用不同标称值。"""
    options=dict(min_confidence=config.min_confidence,
        response_window_ms=3900 if sim_fast else 6700 if screen_coupled else 3000,
        median_window=5)
    if sim_fast:options['fast_direction_reversal']=True
    photo_latch=((sim_fast or physical_mode) and config.camera_source=='ipad' and config.single_plan
                 and config.guidance_profile=='claude-guide')
    if photo_latch:
        options.update(arrival_dwell_ms=2000,fixed_plan_tolerance_scale=1.5,min_subject_width=.04)
    if config.guidance_profile in ('gemini-guide','claude-guide'):
        options.update(single_plan=config.single_plan,entry_ack_ms=3000 if screen_coupled else 800,
                       interaction_mode=config.guidance_profile)
        if config.single_plan:options['portrait_quality']=True
    return options


def process_frames(
    capture: Any,
    tracker: Any,
    config: AppConfig,
    *,
    output: IO[str] = sys.stdout,
    guest_display=None,
    framing_client=None,
) -> RunStats:
    if config.mode not in {"guide-person", "follow-frame"}:
        raise ValueError("mode must be 'guide-person' or 'follow-frame'")
    controller_config = ControllerConfig(
        target_x=config.target_x,
        target_head_y=config.target_head_y,
        target_shoulder_width=config.target_shoulder_width,
        smoothing_alpha=config.smoothing_alpha,
        horizontal_deadband=config.horizontal_deadband,
        vertical_deadband=config.vertical_deadband,
        scale_deadband=config.scale_deadband,
    )
    if config.guidance_profile=='frame-guide':
        controller_config=replace(controller_config,horizontal_deadband=.08,
                                  vertical_deadband=.08,scale_deadband=.08)
    controller = ContinuousController(controller_config)
    agent_mode=config.guidance_profile=='claude-agent'
    if agent_mode and not config.mirror:
        raise ValueError('claude-agent requires a mirrored preview')
    presence=FacePresence() if agent_mode else None
    screen_coupled=False
    physical_mode=bool(guest_display is not None and
                       getattr(guest_display,'physical_feedback_enabled',False))
    physical_preview=physical_mode and os.environ.get('LEICA_PHYSICAL_PREVIEW')=='1'
    if agent_mode:
        guidance_controller=ClaudeInteractionController(client=(
            _framing_client(config) if framing_client is None else framing_client))
        guidance_controller.camera_fixed=config.camera_source!='ipad'
    else:
        guidance_class=(GeminiGuidanceController if config.guidance_profile in ('gemini-guide','claude-guide') else
                        CoarsePortraitController if config.guidance_profile=='coarse-demo' else PersonGuidanceController)
        screen_coupled=guest_display is not None and config.guidance_profile in ('gemini-guide','claude-guide')
        sim_fast=screen_coupled and os.environ.get('LEICA_SIM_FAST_GESTURES')=='1'
        guidance_options=guidance_controller_options(config,screen_coupled=screen_coupled,
                                                     physical_mode=physical_mode,sim_fast=sim_fast)
        if config.guidance_profile in ('gemini-guide','claude-guide'):
            guidance_options['client']=_framing_client(config) if framing_client is None else framing_client
        guidance_controller = guidance_class(
            config=controller_config,
            **guidance_options,
        )
    motion_envelope = MotionEnvelope(
        max_right_m=config.max_right_m,
        max_down_m=config.max_down_m,
        max_toward_subject_m=config.max_toward_subject_m,
        max_step_m=config.max_step_m,
    )
    motion_mapper = MotionIntentMapper(motion_envelope)
    started = time.monotonic()
    frames_processed = 0
    target_events = 0
    read_failures = 0
    timestamp_ms = -1
    last_tracking: bool | None = None
    last_guidance_signature: tuple[str, GuidanceCue] | None = None
    active_gesture: GesturePlan | None = None
    active_gesture_target = None
    active_gesture_source_ns = None
    active_gesture_started_ms = 0
    simulator_gesture_scale=SimulatorGestureScale()
    photo_latch=(PhysicalPhotoLatch(guest_display) if physical_mode else SimulatorPhotoLatch(enabled=(
        config.camera_source=='ipad' and config.single_plan
        and config.guidance_profile=='claude-guide' and guest_display is not None
        and os.environ.get('LEICA_SIM_FAST_GESTURES')=='1')))
    arrival_recovery=SimulatorArrivalRecovery(enabled=(
        config.camera_source=='ipad' and config.single_plan
        and config.guidance_profile=='claude-guide' and guest_display is not None
        and os.environ.get('LEICA_SIM_FAST_GESTURES')=='1' and not photo_latch.enabled))
    direction_recovery=SimulatorDirectionRecovery(enabled=(
        config.camera_source=='ipad' and config.single_plan
        and config.guidance_profile=='claude-guide' and guest_display is not None
        and os.environ.get('LEICA_SIM_FAST_GESTURES')=='1'))
    current_intent = MotionIntent(True, 0.0, 0.0, 0.0, "STARTING")
    camera_paused = False
    last_output_source_ns = 0
    # 仅参考仿真桥接器支持这个独立协议；开场阶段无须模拟人物检测或调用模型。
    intro_phase=None
    intro_id=None
    intro_identity=None
    intro_started_ns=0
    intro_last_ack_ns=0
    intro_ready_cutoff_ns=0
    intro_deadline_pause_ns=0
    communication_hold_started_ns=None
    last_intro_control_ns=0
    pending_action=None
    pending_identity=None
    action_identity=None
    last_control_identity=None
    perception_cancelled=False
    normal_read_timeout=getattr(capture,'read_timeout',None)

    def guidance_output(decision,gesture,timestamp_ms, *, source_monotonic_ns):
        result=serialize_guidance_event(decision,gesture,timestamp_ms,source_monotonic_ns=source_monotonic_ns)
        if physical_mode:
            if action_identity is None:raise RuntimeError('Missing physical guest control identity')
            payload=json.loads(result)
            payload.update(control_instance=action_identity[0],control_epoch=action_identity[1])
            result=json.dumps(payload,separators=(',',':'),sort_keys=True)
        return result

    def intro_status(now_ns):
        """使用新的仿真反馈控制开场动作，不能把它当作新的相机帧。"""
        nonlocal intro_phase,intro_last_ack_ns,intro_ready_cutoff_ns,intro_deadline_pause_ns
        prefix='physical' if physical_mode else 'simulator'
        active=getattr(guest_display,prefix+'_intro_active',lambda _id:False)(intro_id)
        ready=getattr(guest_display,prefix+'_intro_ready',lambda _id:False)(intro_id)
        if active:intro_last_ack_ns=now_ns;intro_deadline_pause_ns=0
        initial_timeout_ns=15_000_000_000 if physical_mode else 2_000_000_000
        if ((not intro_last_ack_ns and now_ns-intro_started_ns>initial_timeout_ns+intro_deadline_pause_ns) or
                (intro_last_ack_ns and now_ns-intro_last_ack_ns>1_000_000_000+intro_deadline_pause_ns)):
            intro_phase='INTRO_UNAVAILABLE'
            return False
        if physical_mode and ready and not intro_ready_cutoff_ns:
            # 以首次实测返回时刻作为相机帧的固定分界。
            # 后续控制心跳不可推进该时刻，否则上传耗时超过心跳周期的有效图像会一直被丢弃。
            intro_ready_cutoff_ns=now_ns
            capture.invalidate_pending()
        return ready

    while True:
        if config.duration and time.monotonic() - started >= config.duration:
            break

        if normal_read_timeout is not None:
            # iPad 读取通常阻塞 250 毫秒，须在桥接器 300 毫秒的新鲜度窗口内给仿真心跳留出余量。
            capture.read_timeout=min(normal_read_timeout,.08) if intro_phase=='INTRO' else normal_read_timeout
        ok, frame = (False,None) if camera_paused else capture.read()
        if (not physical_mode and intro_phase=='INTRO' and ok and config.camera_source=='ipad' and
                capture.source_monotonic_ns<=last_output_source_ns):
            # 相机凭据可能早于仿真心跳，不能把对应的图像标记为晚于控制事件。
            ok=False;frame=None
        action_identity=None
        control_changed=False
        if physical_mode:
            control_action=guest_display.take_control_action()
            action=control_action['action']
            action_identity=(control_action['instance'],control_action['epoch'])
            control_changed=last_control_identity is not None and action_identity!=last_control_identity
            last_control_identity=action_identity
        else:action=guest_display.take_action() if config.mode=='guide-person' and guest_display is not None else None
        if action is None and pending_action is not None:
            if not physical_mode or pending_identity==action_identity:action=pending_action
        pending_action=None;pending_identity=None
        if physical_mode:
            try:live_control=guest_display.control_snapshot()
            except Exception:live_control={}
            communication=live_control.get('communication',{})
            communicating_hold=bool(action!='stop' and not control_changed and live_control.get('active') is True
                and (live_control.get('instance'),live_control.get('epoch'))==action_identity
                and isinstance(communication,dict) and communication.get('recoverable') is True
                and communication.get('status') in ('waiting','ready'))
            if communicating_hold:
                now_ns=time.monotonic_ns()
                if communication_hold_started_ns is None:
                    communication_hold_started_ns=now_ns
                    # 保留选定场景，丢弃被中断的动作或拍照决策，等新画面到达后再决策。
                    photo_latch.reset();arrival_recovery.reset();direction_recovery.reset()
                    active_gesture=None;active_gesture_target=None;active_gesture_source_ns=None
                    last_guidance_signature=None
                if action in ('start','restart'):pending_action=action;pending_identity=action_identity
                timestamp_ms=max(timestamp_ms+1,int((time.monotonic()-started)*1000))
                suspend=getattr(guidance_controller,'suspend_perception',None)
                if callable(suspend):suspend(timestamp_ms)
                held=GuidanceDecision('COMMUNICATION_HOLD',GuidanceCue.HOLD,(.5,.3,.2),
                    None,None,False,'COMMUNICATION_RECOVERING',interaction_mode=config.guidance_profile)
                source_ns=time.monotonic_ns()
                while source_ns<=last_output_source_ns:
                    time.sleep(.001);source_ns=time.monotonic_ns()
                if intro_phase=='INTRO':
                    event={'event':'physical_intro','action':'heartbeat','intro_id':intro_id,
                        'source_monotonic_ns':source_ns,'control_instance':action_identity[0],
                        'control_epoch':action_identity[1]}
                    print(json.dumps(event,separators=(',',':')),file=output,flush=True)
                    last_intro_control_ns=source_ns
                else:print(guidance_output(held,None,timestamp_ms,source_monotonic_ns=source_ns),file=output,flush=True)
                guest_display.publish(frame if ok else None,held,source_monotonic_ns=source_ns)
                last_output_source_ns=source_ns;target_events+=1
                if ok and frame is not None:frames_processed+=1
                if config.max_frames and frames_processed>=config.max_frames:break
                if not ok:time.sleep(.01)
                continue
            if communication_hold_started_ns is not None:
                if not control_changed and action!='stop':
                    intro_deadline_pause_ns+=max(0,time.monotonic_ns()-communication_hold_started_ns)
                communication_hold_started_ns=None
        if (physical_mode and ok and frame is not None and
                getattr(capture,'source_monotonic_ns',0)<=control_action['control_started_ns']):
            # 在开始或选择新目标之前发放的帧凭据不能重新归入新一轮实体控制。
            if action is not None:pending_action=action;pending_identity=action_identity
            continue
        if physical_preview:
            # 由独立控制、已审核的单步预览负责动作。保持访客相机运行，不启动 AI 或开场动作。
            source_ns=capture.source_monotonic_ns if ok and frame is not None else time.monotonic_ns()
            if source_ns<=last_output_source_ns:
                time.sleep(.01);continue
            timestamp_ms=max(timestamp_ms+1,int((time.monotonic()-started)*1000))
            active=bool(guest_display.active)
            held=GuidanceDecision('PHYSICAL_PREVIEW' if active else 'GUEST_READY',GuidanceCue.HOLD,
                                  (.5,.3,.2),None,None,False,'MANUAL_POINT_PREVIEW' if active else 'VISITOR_STOPPED_OR_NOT_STARTED')
            print(guidance_output(held,None,timestamp_ms,source_monotonic_ns=source_ns),file=output,flush=True)
            guest_display.publish(frame if ok else None,held,source_monotonic_ns=source_ns)
            last_output_source_ns=source_ns;target_events+=1
            if ok and frame is not None:frames_processed+=1
            if config.max_frames and frames_processed>=config.max_frames:break
            if not ok:time.sleep(.01)
            continue
        # 开场过程中再次按开始视为重复操作，不能从中间倾斜姿态重新定位开场。即使相机读取超时也遵循此规则；
        # 停止和新目标仍然有效。
        if intro_phase=='INTRO' and action=='start':
            if physical_mode and action_identity!=intro_identity:intro_phase=None
            else:action=None
        if physical_mode and intro_phase=='INTRO' and action is None and action_identity!=intro_identity:
            intro_phase='INTRO_STOPPED'
        if not ok or frame is None:
            if intro_phase=='INTRO' and action in ('stop','start','restart'):
                # 不等待相机帧即可立即取消桥接；重新开始或选择新目标仍须等待实际画面。
                if action in ('start','restart'):pending_action=action;pending_identity=action_identity
                intro_phase='INTRO_STOPPED' if action=='stop' else None
                source_ns=time.monotonic_ns()
                while source_ns<=last_output_source_ns:
                    time.sleep(.001);source_ns=time.monotonic_ns()
                held=GuidanceDecision('GUEST_READY',GuidanceCue.HOLD,(.5,.3,.2),None,None,False,'INTRO_CANCELLED')
                timestamp_ms=max(timestamp_ms+1,int((time.monotonic()-started)*1000))
                print(guidance_output(held,None,timestamp_ms,source_monotonic_ns=source_ns),file=output,flush=True)
                last_output_source_ns=source_ns;target_events+=1
                guest_display.publish(None,held,source_monotonic_ns=source_ns)
                continue
            if intro_phase=='INTRO':
                if not guest_display.active:intro_phase='INTRO_STOPPED'
                else:intro_status(time.monotonic_ns())
                source_ns=time.monotonic_ns()
                # 较早发出的帧凭据到达时不要不断推进控制时间戳，否则在单调时钟分辨率较低的系统上会持续丢弃新帧。
                prior_intro_ns=last_intro_control_ns if physical_mode else last_output_source_ns
                if intro_phase=='INTRO' and source_ns-prior_intro_ns<50_000_000:
                    time.sleep(.01)
                    continue
                while source_ns<=prior_intro_ns:
                    time.sleep(.001);source_ns=time.monotonic_ns()
                timestamp_ms=max(timestamp_ms+1,int((time.monotonic()-started)*1000))
                if intro_phase=='INTRO':
                    # 心跳只反映仿真控制会话的状态，不能把心跳时间戳用于图像推理。
                    event={'event':'physical_intro' if physical_mode else 'simulator_intro','action':'heartbeat','intro_id':intro_id,'source_monotonic_ns':source_ns}
                    if physical_mode:event.update(control_instance=intro_identity[0],control_epoch=intro_identity[1])
                    phase='INTRO';reason='INTRO_IN_PROGRESS'
                    print(json.dumps(event,separators=(',',':')),file=output,flush=True)
                else:
                    phase='GUEST_READY' if intro_phase=='INTRO_STOPPED' else 'INTRO_UNAVAILABLE'
                    reason='VISITOR_STOPPED_OR_NOT_STARTED' if intro_phase=='INTRO_STOPPED' else 'INTRO_UNAVAILABLE'
                    held=GuidanceDecision(phase,GuidanceCue.HOLD,(.5,.3,.2),None,None,False,reason)
                    print(guidance_output(held,None,timestamp_ms,source_monotonic_ns=source_ns),file=output,flush=True)
                if physical_mode and intro_phase=='INTRO':last_intro_control_ns=source_ns
                else:last_output_source_ns=source_ns
                target_events+=1
                held=GuidanceDecision(phase,GuidanceCue.HOLD,(.5,.3,.2),None,None,False,reason)
                guest_display.publish(None,held,source_monotonic_ns=source_ns)
                continue
            if intro_phase in ('INTRO_STOPPED','INTRO_UNAVAILABLE'):
                if action=='stop':intro_phase='INTRO_STOPPED'
                elif action is not None:pending_action=action;pending_identity=action_identity
                source_ns=time.monotonic_ns()
                while source_ns<=last_output_source_ns:
                    time.sleep(.001);source_ns=time.monotonic_ns()
                timestamp_ms=max(timestamp_ms+1,int((time.monotonic()-started)*1000))
                phase='GUEST_READY' if intro_phase=='INTRO_STOPPED' else 'INTRO_UNAVAILABLE'
                held=GuidanceDecision(phase,GuidanceCue.HOLD,(.5,.3,.2),None,None,False,phase)
                print(guidance_output(held,None,timestamp_ms,source_monotonic_ns=source_ns),file=output,flush=True)
                last_output_source_ns=source_ns;target_events+=1
                guest_display.publish(None,held,source_monotonic_ns=source_ns)
                continue
            if action is not None:pending_action=action;pending_identity=action_identity
            read_failures += 1
            cancel_perception=False
            if config.camera_source=='ipad':
                if physical_mode:
                    # 感知可能中断超过界面上相机在线状态的超时阈值。根据实际所有者和控制会话身份判断交互归属。
                    try:control=guest_display.control_snapshot()
                    except Exception:control={}
                    retain_encounter=bool(control.get('active') is True and control.get('control_fresh') is True
                        and (control.get('instance'),control.get('epoch'))==action_identity)
                    cancelled=action=='stop' or control_changed or not retain_encounter
                    cancel_perception=cancelled and (not perception_cancelled or control_changed or action=='stop')
                    if cancel_perception:
                        arrival_recovery.reset();photo_latch.reset();direction_recovery.reset();simulator_gesture_scale.reset()
                        cancel=getattr(guidance_controller,'stop' if agent_mode else 'reset',None)
                        if callable(cancel):cancel()
                        if action not in ('start','restart'):pending_action=None;pending_identity=None
                    perception_cancelled=cancelled
                else:
                    retain_encounter=config.single_plan and capture.connected()
                    if not retain_encounter and (read_failures==1 or
                            (guest_display is not None and guest_display.active)):
                        capture.invalidate_pending()
                        if guest_display is not None:guest_display.pause_camera()
                        if agent_mode:guidance_controller.stop()
                # 上面的实体控制快照不能锁住随后到来的开始请求。此处不要暂停相机或使之失效；
                # 所有者或控制会话到期时，会由权威状态取消动作。
            if read_failures == 1 or cancel_perception:
                arrival_recovery.interrupt(active_gesture,active=bool(guest_display and guest_display.active))
                direction_recovery.interrupt(active_gesture,active_gesture_target,
                    active=bool(guest_display and guest_display.active))
                timestamp_ms=max(timestamp_ms+1,int((time.monotonic()-started)*1000))
                source_ns=time.monotonic_ns()
                while source_ns <= last_output_source_ns:
                    # Windows 单调时钟的刻度可能较粗，须等待时钟真正前进，不能伪造未来时间。
                    time.sleep(.001)
                    source_ns=time.monotonic_ns()
                active_gesture=None
                active_gesture_target=None;active_gesture_source_ns=None
                if config.mode=='guide-person':
                    if photo_latch.decision is not None and guest_display is not None and guest_display.active:
                        lost=photo_latch.decision
                    elif physical_mode:
                        # 缺少图像不等于画面中无人。保留缓存场景和控制器状态，避免把合成 HOLD 的时间戳用于推理。
                        lost=GuidanceDecision('GUEST_READY' if cancel_perception else 'CAMERA_PAUSED',
                            GuidanceCue.HOLD,(.5,.3,.2),None,None,False,
                            'VISITOR_STOPPED_OR_NOT_STARTED' if cancel_perception else 'CAMERA_FRAME_MISSING',
                            interaction_mode=config.guidance_profile)
                    else:
                        photo_latch.reset()
                        lost=replace(guidance_controller.update(None,timestamp_ms),
                            phase='CAMERA_PAUSED',reason='CAMERA_FRAME_MISSING')
                    if config.guidance_profile=='frame-guide':
                        lost=replace(lost,interaction_mode='frame-guide')
                    event=guidance_output(lost,None,timestamp_ms,
                                                  source_monotonic_ns=source_ns)
                    last_guidance_signature=(lost.phase,lost.cue)
                    if guest_display is not None:
                        guest_display.publish(None if config.camera_source=='ipad' else np.zeros((config.height,config.width,3),dtype=np.uint8),lost)
                else:
                    lost=replace(controller.update(None),reason='CAMERA_FRAME_MISSING')
                    current_intent=motion_mapper.update(lost)
                    event=serialize_tracking_event(lost,current_intent,timestamp_ms,
                                                   source_monotonic_ns=source_ns)
                    last_tracking=False
                print(event,file=output,flush=True)
                last_output_source_ns=source_ns
                target_events+=1
                print('CAMERA_FRAME_MISSING: perception paused; waiting for camera.',file=sys.stderr)
            if config.camera_source=='ipad':
                time.sleep(.01)
                continue
            if config.headless and read_failures >= config.max_read_failures:
                raise CameraReadError(
                    f"Camera failed to provide {read_failures} consecutive frames"
                )
            camera_paused=read_failures>=config.max_read_failures
            if config.headless:
                time.sleep(0.02)
            else:
                panel=np.zeros((config.height,config.width,3),dtype=np.uint8)
                lines=('CAMERA DISCONNECTED - ARM PAUSED',
                       'No current image. Check webcam power, mode and cable.',
                       'R: reconnect camera and restart session   Q / Esc: exit')
                for i,line in enumerate(lines):
                    cv2.putText(panel,line,(25,70+i*42),cv2.FONT_HERSHEY_SIMPLEX,
                                .65 if i==0 else .48,(40,170,255),1,cv2.LINE_AA)
                title=GUIDANCE_WINDOW if config.mode=='guide-person' else 'Leica Virtual Robot Preview — q / Esc to quit'
                cv2.imshow(title,panel)
                key=cv2.waitKey(50)&0xFF
                if key in (ord('q'),27):
                    break
                if key==ord('r'):
                    capture.release()
                    capture.open(config.camera,_camera_backend())
                    _configure_capture(capture,config)
                    camera_paused=False
                    read_failures=0
                    guidance_controller.reset()
                    last_guidance_signature=None
                    print('CAMERA_REOPEN_REQUESTED',file=sys.stderr)
            continue

        source_monotonic_ns = capture.source_monotonic_ns if config.camera_source=='ipad' else time.monotonic_ns()
        if config.camera_source=='ipad' and source_monotonic_ns <= last_output_source_ns:
            # 即使 JPEG 在模拟丢失或 HOLD 之后到达，凭据也可能更早发出。丢弃旧帧，不重写时间。
            # 按钮操作仍属本轮控制，不能因此吞掉开始或新目标事件。
            if physical_mode and action is not None:
                pending_action=action;pending_identity=action_identity
            continue
        if read_failures:
            # Gemini 已在本地处理丢失和间隔，短暂读取失败时应保留有效的固定背景模板，避免重复上传。
            if config.guidance_profile not in ('gemini-guide','claude-guide','claude-agent'):
                guidance_controller.reset()
            last_guidance_signature=None
            print('CAMERA_RECOVERED: resuming fresh camera observations.',file=sys.stderr)
        read_failures = 0
        perception_cancelled=False
        frame = _rotate_frame(frame, config.rotate)
        if config.mirror != config.source_mirrored:
            frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = max(timestamp_ms + 1, int((time.monotonic() - started) * 1000))
        if action in ('stop','start','restart'):
            arrival_recovery.reset();photo_latch.reset();direction_recovery.reset();simulator_gesture_scale.reset()
            if agent_mode and action=='stop':guidance_controller.stop()
            if action in ('start','restart'):
                if config.guidance_profile in ('gemini-guide','claude-guide','claude-agent'):guidance_controller.restart_encounter()
                else:guidance_controller.reset()
                last_guidance_signature=None;active_gesture=None
                active_gesture_target=None;active_gesture_source_ns=None
            intro_supported_flow=(config.camera_source=='ipad' and config.single_plan and
                                  config.guidance_profile=='claude-guide')
            if action=='start' and intro_supported_flow:
                available=(physical_mode or (os.environ.get('LEICA_SIM_FAST_GESTURES')=='1' and
                           bool(getattr(guest_display,'simulator_intro_available',lambda:False)())))
                intro_phase='INTRO' if available else 'INTRO_UNAVAILABLE'
                intro_id=source_monotonic_ns
                intro_identity=action_identity if physical_mode else None
                intro_started_ns=time.monotonic_ns();intro_last_ack_ns=0
                intro_ready_cutoff_ns=0
                last_intro_control_ns=0
            elif action=='stop' and intro_phase:
                intro_phase='INTRO_STOPPED'
            elif action=='restart':
                # 在新目标发出普通引导前，用正常的 HOLD 显式取消先前的开场动作。
                if intro_phase=='INTRO':
                    held=GuidanceDecision('GUEST_READY',GuidanceCue.HOLD,(.5,.3,.2),None,None,False,'INTRO_CANCELLED')
                    print(guidance_output(held,None,timestamp_ms,source_monotonic_ns=source_monotonic_ns),file=output,flush=True)
                    last_output_source_ns=source_monotonic_ns;target_events+=1
                    # 不要将同一采集时间戳重复用于引导事件。
                    intro_phase=None
                    guest_display.publish(frame,held,source_monotonic_ns=source_monotonic_ns)
                    frames_processed+=1
                    if config.max_frames and frames_processed>=config.max_frames:break
                    continue
                intro_phase=None
        if intro_phase=='INTRO' and guest_display is not None and not guest_display.active:
            intro_phase='INTRO_STOPPED'
        if intro_phase=='INTRO':
            now_ns=time.monotonic_ns()
            ready=intro_status(now_ns)
            image_after_return=(not physical_mode or
                (source_monotonic_ns>intro_ready_cutoff_ns and
                 0<=now_ns-source_monotonic_ns<=300_000_000))
            if ready and image_after_return:
                intro_phase=None
                last_guidance_signature=None
            elif not physical_mode and (not 0<=now_ns-source_monotonic_ns<=300_000_000 or
                  (source_monotonic_ns!=intro_id and last_output_source_ns and
                   source_monotonic_ns-last_output_source_ns>300_000_000)):
                intro_phase='INTRO_UNAVAILABLE'
        if intro_phase is not None:
            phase='GUEST_READY' if intro_phase=='INTRO_STOPPED' else intro_phase
            held=GuidanceDecision(phase,GuidanceCue.HOLD,(.5,.3,.2),None,None,False,
                                  'INTRO_IN_PROGRESS' if intro_phase=='INTRO' else 'INTRO_UNAVAILABLE' if intro_phase=='INTRO_UNAVAILABLE' else 'VISITOR_STOPPED_OR_NOT_STARTED')
            if intro_phase=='INTRO':
                control_ns=time.monotonic_ns() if physical_mode else source_monotonic_ns
                event={'event':'physical_intro' if physical_mode else 'simulator_intro','action':'start' if source_monotonic_ns==intro_id else 'heartbeat',
                       'intro_id':intro_id,'source_monotonic_ns':control_ns}
                if physical_mode:event.update(control_instance=intro_identity[0],control_epoch=intro_identity[1])
                print(json.dumps(event,separators=(',',':')),file=output,flush=True)
            else:
                print(guidance_output(held,None,timestamp_ms,source_monotonic_ns=source_monotonic_ns),file=output,flush=True)
            target_events+=1
            last_output_source_ns=source_monotonic_ns
            if physical_mode and intro_phase=='INTRO':last_intro_control_ns=control_ns
            if guest_display is not None:guest_display.publish(frame,held,source_monotonic_ns=source_monotonic_ns)
            frames_processed+=1
            if config.max_frames and frames_processed>=config.max_frames:break
            continue
        # 拍摄完成后不再依赖参与者位置。继续转发实时相机帧，停止姿态推理。
        locked_active=(photo_latch.decision is not None and guest_display is not None
                       and guest_display.active)
        detection = None if locked_active else tracker.detect(rgb_frame, timestamp_ms)
        if (direction_recovery.enabled and active_gesture is not None
                and getattr(active_gesture,'cue','').startswith('MOVE_')
                and timestamp_ms-active_gesture_started_ms<active_gesture.duration_ms):
            now_ns=time.monotonic_ns()
            source_age=now_ns-source_monotonic_ns
            source_gap=source_monotonic_ns-last_output_source_ns
            checker=getattr(guest_display,'simulator_motion_active',None)
            acknowledged=checker(active_gesture_source_ns,active_gesture.cue) if checker else None
            bridge_interrupted=direction_recovery.feedback_interrupted(acknowledged,
                source_ns=active_gesture_source_ns,now_ns=now_ns)
            if (source_age>300_000_000 or (last_output_source_ns and source_gap>300_000_000)
                    or bridge_interrupted):
                direction_recovery.interrupt(active_gesture,active_gesture_target,
                    active=bool(guest_display and guest_display.active))
                active_gesture=None;active_gesture_target=None;active_gesture_source_ns=None
                last_guidance_signature=None
        if (arrival_recovery.enabled and active_gesture is not None
                and getattr(active_gesture,'cue',None)=='ARRIVED'):
            # 即使 capture.read() 未报错，源帧过旧或连续源时间戳间隔超过 300 毫秒也会使桥接器中断确认动作。
            # 丢弃对应动作，等待新的正式事件，不得通过改写帧时间来伪造新鲜度。
            source_age=time.monotonic_ns()-source_monotonic_ns
            source_gap=source_monotonic_ns-last_output_source_ns
            if source_age>300_000_000 or (last_output_source_ns and source_gap>300_000_000):
                arrival_recovery.interrupt(active_gesture,active=bool(guest_display and guest_display.active))
                active_gesture=None
                # 即使先前的源事件也是 ARRIVED 且提示相同，仍须发出新的恢复事件。
                last_guidance_signature=None
        if config.mode == "guide-person":
            reference_held=False
            if guest_display is not None:
                if agent_mode and (action=='stop' or not guest_display.active):guidance_controller.stop()
            if agent_mode:
                if guest_display is not None:
                    for text in guest_display.take_messages():guidance_controller.say(text,timestamp_ms)
                active=guest_display is None or guest_display.active
                visible=presence.update(frame,detection,timestamp_ms) if active else False
                decision=guidance_controller.update(detection,timestamp_ms,frame=frame,present=visible,mirrored=config.mirror)
                if guest_display is not None:guest_display.set_dialogue(guidance_controller.dialogue)
            else:
                reference_held=(config.single_plan and guest_display is not None and
                                ((hasattr(guest_display,'reference_held') and
                                  guest_display.reference_held(source_monotonic_ns)) or
                                 (physical_mode and not guest_display.physical_backend_available())))
                if photo_latch.decision is not None and guest_display is not None and guest_display.active:
                    decision=photo_latch.decision
                elif reference_held:
                    suspend=getattr(guidance_controller,'suspend_perception',None)
                    decision=(suspend(timestamp_ms) if callable(suspend) else
                        GuidanceDecision('PHYSICAL_REFERENCE_HOLD',GuidanceCue.HOLD,(.5,.3,.2),
                            None,None,False,'AWAITING_MEASURED_REFERENCE',
                            interaction_mode=config.guidance_profile))
                else:
                    decision = guidance_controller.update(
                        detection if guest_display is None or guest_display.active else None,timestamp_ms,
                        mirrored=config.mirror,**({'frame':frame} if config.guidance_profile in ('gemini-guide','claude-guide') else {}))
                if reference_held:
                    decision=replace(decision,phase='PHYSICAL_REFERENCE_HOLD',cue=GuidanceCue.HOLD,
                                     tracking=False,current_point=None,filtered_point=None,
                                     composition_frame=None,reason='AWAITING_MEASURED_REFERENCE')
            if guest_display is not None and not guest_display.active:
                photo_latch.reset()
                simulator_gesture_scale.reset()
                decision=replace(decision,phase='PHYSICAL_REFERENCE_HOLD' if reference_held else 'GUEST_READY',cue=GuidanceCue.HOLD,
                                 tracking=False,current_point=None,filtered_point=None,
                                 composition_frame=None,reason='AWAITING_MEASURED_REFERENCE' if reference_held else 'VISITOR_STOPPED_OR_NOT_STARTED')
            if guest_display is not None and decision.phase=='SETTLING':
                decision=replace(decision,cue=GuidanceCue.HOLD)
            decision=arrival_recovery.apply(decision,active=bool(guest_display and guest_display.active),
                source_ns=source_monotonic_ns,now_ns=time.monotonic_ns())
            if physical_mode:
                decision=photo_latch.apply(decision,active=bool(guest_display and guest_display.active),source_ns=source_monotonic_ns)
            else:decision=photo_latch.apply(decision,active=bool(guest_display and guest_display.active))
            decision=direction_recovery.apply(decision,active=bool(guest_display and guest_display.active),
                source_ns=source_monotonic_ns,now_ns=time.monotonic_ns(),mirrored=config.mirror)
            if config.guidance_profile=='frame-guide':
                decision=replace(decision,interaction_mode='frame-guide')
            signature = (decision.phase, decision.cue)
            changed = signature != last_guidance_signature
            if changed or (config.guidance_stream and frames_processed % config.emit_every_frames == 0):
                if agent_mode:
                    next_gesture=guidance_controller.gesture if changed and decision.cue not in (GuidanceCue.WAIT,GuidanceCue.HOLD) else None
                else:
                    directional=sim_fast and decision.cue.value.startswith('MOVE_')
                    amplitude=getattr(decision,'gesture_amplitude',1.)
                    if changed and directional:
                        # 恢复时复用被中断动作的幅度，不能从 WAIT 推导一个人为幅度。
                        amplitude=(simulator_gesture_scale.current() if decision.reason=='SIMULATOR_DIRECTION_RECOVERY'
                                   else simulator_gesture_scale.select(amplitude))
                    next_gesture = (gesture_plan_for(decision.cue,
                        amplitude=amplitude,
                        profile='sim-fast' if directional else 'screen-coupled' if screen_coupled else 'semantic') if changed else None)
                if next_gesture is not None:
                    if (next_gesture.cue.startswith('MOVE_')
                            and decision.reason!='SIMULATOR_DIRECTION_RECOVERY'):
                        direction_recovery.new_gesture()
                    active_gesture = next_gesture
                    active_gesture_target = tuple(decision.target) if next_gesture.cue.startswith('MOVE_') else None
                    active_gesture_source_ns = source_monotonic_ns
                    active_gesture_started_ms = timestamp_ms
                    if (direction_recovery.enabled and next_gesture.cue.startswith('MOVE_')
                            and time.monotonic_ns()-source_monotonic_ns>300_000_000):
                        direction_recovery.interrupt(next_gesture,active_gesture_target,
                            active=bool(guest_display and guest_display.active))
                        active_gesture=None;active_gesture_target=None;active_gesture_source_ns=None
                elif decision.cue is GuidanceCue.HOLD:
                    active_gesture = None
                    active_gesture_target=None;active_gesture_source_ns=None
                print(
                    guidance_output(
                        decision,
                        next_gesture,
                        timestamp_ms,
                        source_monotonic_ns=source_monotonic_ns,
                    ),
                    file=output,
                    flush=True,
                )
                target_events += 1
                last_output_source_ns = source_monotonic_ns
                last_guidance_signature = signature
            if active_gesture is not None and (
                timestamp_ms - active_gesture_started_ms
                >= active_gesture.duration_ms
            ) and active_gesture.pattern not in HOLDING_PATTERNS:
                active_gesture = None
                active_gesture_target=None;active_gesture_source_ns=None
            if guest_display is not None:
                guest_display.publish(frame,decision,gesture=active_gesture,
                    gesture_elapsed_ms=timestamp_ms-active_gesture_started_ms,
                    source_monotonic_ns=source_monotonic_ns)
        else:
            command = controller.update(detection)
            if config.horizontal_only:
                command = replace(command, down=0.0, toward_subject=0.0)
            should_emit = frames_processed % config.emit_every_frames == 0
            if last_tracking is True and not command.tracking:
                should_emit = True
            if should_emit:
                current_intent = motion_mapper.update(command)
                print(
                    serialize_tracking_event(
                        command,
                        current_intent,
                        timestamp_ms,
                        source_monotonic_ns=source_monotonic_ns,
                        observation=detection,
                    ),
                    file=output,
                    flush=True,
                )
                target_events += 1
                last_output_source_ns = source_monotonic_ns
            last_tracking = command.tracking

        frames_processed += 1
        elapsed = max(time.monotonic() - started, 1e-9)
        if not config.headless:
            if config.mode == "guide-person":
                _draw_guidance_overlay(
                    frame,
                    detection,
                    decision,
                    frames_processed / elapsed,
                    gesture=active_gesture,
                    gesture_elapsed_ms=timestamp_ms - active_gesture_started_ms,
                    mirrored=config.mirror,
                )
                title = GUIDANCE_WINDOW
            else:
                _draw_overlay(
                    frame,
                    detection,
                    command,
                    current_intent,
                    controller_config,
                    motion_envelope,
                    frames_processed / elapsed,
                )
                title = "Leica Virtual Robot Preview — q / Esc to quit"
            cv2.imshow(title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if config.mode == "guide-person" and key == ord("r"):
                guidance_controller.reset()
                last_guidance_signature = None
                active_gesture = None

        if config.max_frames and frames_processed >= config.max_frames:
            break

    if config.guidance_profile in ('gemini-guide','claude-guide','claude-agent'):
        guidance_controller.close()
    if normal_read_timeout is not None:capture.read_timeout=normal_read_timeout
    return RunStats(
        frames_processed=frames_processed,
        target_events=target_events,
        elapsed_seconds=time.monotonic() - started,
    )


def _config_from_args(args: argparse.Namespace) -> AppConfig:
    if args.single_plan and args.guidance_profile!='claude-guide':
        raise ValueError('--single-plan requires claude-guide')
    if args.camera_source=='ipad' and (not args.guest_display or not args.headless or not args.mirror or args.rotate!=0 or args.mode!='guide-person'):
        raise ValueError('iPad camera requires guest display, headless, mirrored guide-person mode without rotation')
    if args.guidance_profile=='claude-agent' and not args.mirror:
        raise ValueError('claude-agent requires a mirrored preview')
    if args.max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    if args.mode == "guide-person" and args.horizontal_only:
        raise ValueError("--horizontal-only is only available with --mode follow-frame")
    if args.mode!='guide-person' and args.guidance_profile!='precise':
        raise ValueError('guidance profiles are only available with --mode guide-person')
    return AppConfig(
        mode=args.mode,
        guidance_profile=args.guidance_profile,
        single_plan=args.single_plan,
        gemini_model=args.gemini_model,
        gemini_env_file=args.gemini_env_file,
        claude_model=args.claude_model,
        claude_effort=args.claude_effort,
        claude_env_file=args.claude_env_file,
        guidance_stream=args.guidance_stream,
        camera=args.camera,
        camera_source=args.camera_source,
        width=args.width,
        height=args.height,
        fps=args.fps,
        mirror=args.mirror,
        source_mirrored=True if args.camera_source=='ipad' else args.source_mirrored,
        rotate=args.rotate,
        headless=args.headless,
        horizontal_only=args.horizontal_only,
        duration=args.duration,
        max_frames=args.max_frames,
        min_confidence=args.min_confidence,
        max_read_failures=args.max_read_failures,
        target_x=args.target_x,
        target_head_y=args.target_head_y,
        target_shoulder_width=args.target_shoulder_width,
        smoothing_alpha=args.smoothing_alpha,
        horizontal_deadband=args.horizontal_deadband,
        vertical_deadband=args.vertical_deadband,
        scale_deadband=args.scale_deadband,
        emit_every_frames=args.emit_every_frames,
        max_right_m=args.max_right_m,
        max_down_m=args.max_down_m,
        max_toward_subject_m=args.max_toward_subject_m,
        max_step_m=args.max_step_m,
        model_path=args.model,
    )


def _framing_client(config):
    if config.guidance_profile=='claude-agent':
        return ClaudeInteractionClient(model=config.claude_model,effort=config.claude_effort,
            key_provider=lambda:load_claude_key(config.claude_env_file))
    if config.guidance_profile=='claude-guide':
        return ClaudeFramingClient(model=config.claude_model,effort=config.claude_effort,
            key_provider=lambda:load_claude_key(config.claude_env_file),single_plan=config.single_plan,
            portrait_quality=config.single_plan)
    return GeminiFramingClient(model=config.gemini_model,
        key_provider=lambda:load_api_key(config.gemini_env_file),portrait_quality=config.single_plan)


def _camera_backend():
    return cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY


def _configure_capture(capture,config):
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    capture.set(cv2.CAP_PROP_FPS, config.fps)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
    except ValueError as error:
        parser.error(str(error))

    if config.camera_source!='ipad' and config.guidance_profile in ('gemini-guide','claude-guide','claude-agent'):
        try:
            _framing_client(config).check_access()
        except (FramingError,ValueError) as error:
            print(f'ERROR: {error}. {"Claude" if config.guidance_profile in ("claude-guide","claude-agent") else "Gemini"} unavailable; no camera image sent.',file=sys.stderr)
            return 4
    if config.camera_source=='ipad':
        from .ipad_camera import IPadCapture
        capture=IPadCapture()
    else:
        capture = cv2.VideoCapture(config.camera, _camera_backend())
        _configure_capture(capture,config)

    if not capture.isOpened():
        print(
            f"ERROR: Could not open camera index {config.camera}. "
            "Power on Pocket 3 and select Webcam mode.",
            file=sys.stderr,
        )
        capture.release()
        return 2

    guest=None;guest_link=None
    try:
        if args.guest_display:
            if config.mode!='guide-person':raise ValueError('Guest display requires guide-person mode')
            from .guest_display import GuestDisplay
            import os
            guest=GuestDisplay(host=args.guest_host,port=args.guest_port,
                token=os.environ.get('LEICA_GUEST_SESSION_TOKEN'),
                remote_capture=capture if config.camera_source=='ipad' else None,
                physical_feedback=os.environ.get('LEICA_PHYSICAL_FEEDBACK')=='1').__enter__()
            guest_link=Path(__file__).resolve().parents[3]/'tmp/guidance-live/guest-link.json'
            guest_link.parent.mkdir(parents=True,exist_ok=True)
            descriptor=os.open(guest_link,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
            with os.fdopen(descriptor,'w') as file:json.dump({'port':guest.port,'key':guest.token},file)
            guest_link.chmod(0o600)
            print('iPad visitor view ready; private pairing info: tmp/guidance-live/guest-link.json',file=sys.stderr)
        with PoseTracker(config.model_path, config.min_confidence) as tracker:
            stats = process_frames(capture, tracker, config,guest_display=guest)
    except (CameraReadError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 3
    finally:
        if guest is not None:guest.close()
        if guest_link is not None:guest_link.unlink(missing_ok=True)
        capture.release()
        cv2.destroyAllWindows()

    print(
        f"Processed {stats.frames_processed} frames, "
        f"{stats.target_events} tracking targets in {stats.elapsed_seconds:.1f}s.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
