"""在已认证的 Python Lab 或 Jupyter 环境中演练固定提示动作，本地教学仿真器监听回环地址。
演练不需要摄像头，不接受机器人主机地址、任意关节目标、相机帧或参与者测量数据。"""

from __future__ import annotations

import json
import math
import sys
import time


SIMULATOR_HOST = "127.0.0.1"
MOVEJ_ACCELERATION = 0.30
MOVEJ_VELOCITY = 0.22
MAX_JOINT_OFFSET_RAD = 0.20
RETURN_TOLERANCE_RAD = 0.03
SIMULATOR_CONFIRMATION = "COURSE SIMULATOR"

# 用确定性合成输入演练一次 person_guidance 流程。
# 提示之间保留 WAIT，最后用 HOLD 结束会话。
SYNTHETIC_CUES = (
    "INVITE",
    "MOVE_LEFT",
    "WAIT",
    "MOVE_RIGHT",
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_CLOSER",
    "MOVE_FARTHER",
    "NOT_YET",
    "ARRIVED",
    "HOLD",
)

ALLOWED_CUES = frozenset(
    {"INVITE", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN",
     "MOVE_CLOSER", "MOVE_FARTHER", "NOT_YET", "ARRIVED", "WAIT", "HOLD"}
)

# UR5e 的关节顺序为基座、肩、肘、腕一、腕二、腕三。方向提示只对相应关节施加小幅偏移。
_DIRECTIONAL_JOINTS = {
    "MOVE_LEFT": (0, -1),
    "MOVE_RIGHT": (0, 1),
    "MOVE_UP": (1, -1),
    "MOVE_DOWN": (1, 1),
    "MOVE_CLOSER": (2, 1),
    "MOVE_FARTHER": (2, -1),
}
_JOINT_AMPLITUDE_RAD = {0: 0.16, 1: 0.14, 2: 0.14}


def _offset_pose(start: list[float], joint: int, offset: float) -> list[float]:
    if not 0 <= joint < 6 or not math.isfinite(offset):
        raise ValueError("Invalid fixed choreography waypoint")
    if abs(offset) > MAX_JOINT_OFFSET_RAD:
        raise ValueError("Choreography exceeds the fixed joint-offset limit")
    pose = list(start)
    pose[joint] += offset
    return pose


def _waypoints(cue: str, start: list[float]) -> tuple[list[float], ...]:
    """仅根据记录的起始姿态计算固定关节目标。"""
    if cue not in ALLOWED_CUES:
        raise ValueError(f"Unsupported cue: {cue!r}")
    neutral = list(start)

    if cue in _DIRECTIONAL_JOINTS:
        joint, direction = _DIRECTIONAL_JOINTS[cue]
        sweep = _offset_pose(start, joint, direction * _JOINT_AMPLITUDE_RAD[joint])
        return sweep, neutral, sweep, neutral
    if cue == "INVITE":
        beckon = _offset_pose(start, 2, 0.18)
        return beckon, neutral, beckon, neutral
    if cue == "NOT_YET":
        left = _offset_pose(start, 0, -0.14)
        right = _offset_pose(start, 0, 0.14)
        return left, right, left, right, neutral
    if cue == "ARRIVED":
        nod_up = _offset_pose(start, 1, -0.12)
        nod_down = _offset_pose(start, 1, 0.12)
        return nod_up, nod_down, neutral
    return ()


def _move_to(robot, target: list[float], start: list[float]) -> None:
    if len(target) != 6 or not all(math.isfinite(value) for value in target):
        raise ValueError("Invalid simulator joint target")
    if max(abs(value - origin) for value, origin in zip(target, start)) > MAX_JOINT_OFFSET_RAD:
        raise ValueError("Target exceeds the fixed offset from the captured start pose")
    robot.movej(target, acc=MOVEJ_ACCELERATION, vel=MOVEJ_VELOCITY, wait=True)


def _read_six_finite_joints(robot, label: str) -> list[float]:
    joints = [float(value) for value in robot.getj()]
    if len(joints) != 6 or not all(math.isfinite(value) for value in joints):
        raise RuntimeError(f"Simulator returned an invalid {label} joint state")
    return joints


def _stop_quietly(robot) -> bool:
    try:
        robot.stopj(MOVEJ_ACCELERATION)
        return True
    except Exception as error:  # 清理失败时仍应保留最初的错误。
        print(json.dumps({"event": "stop_error", "message": str(error)}), file=sys.stderr)
        return False


def main() -> int:
    if any(cue not in ALLOWED_CUES for cue in SYNTHETIC_CUES):
        raise ValueError("Synthetic cue sequence contains an unsupported cue")

    try:
        import urx
    except ImportError as error:
        raise SystemExit(
            "This demo must run in the course Python Lab environment with its urx package installed."
        ) from error

    robot = None
    interrupted = False
    completed = False
    try:
        print(
            "Check the open Python Lab 3D page is the course simulation-only UR5e. "
            f"Type {SIMULATOR_CONFIRMATION!r} to authorize this synthetic rehearsal: ",
            end="",
            flush=True,
        )
        try:
            confirmation = input().strip()
        except EOFError as error:
            raise SystemExit("No simulator confirmation was provided; no robot connection was opened.") from error
        if confirmation != SIMULATOR_CONFIRMATION:
            raise SystemExit("Simulator confirmation did not match; no robot connection was opened.")

        # 课程快速入门指定了这个仅在回环地址上运行的仿真器。保持该地址固定，以免误连其他主机。
        robot = urx.Robot(SIMULATOR_HOST, use_rt=False)
        start = _read_six_finite_joints(robot, "starting")

        print(json.dumps({
            "event": "simulator_connected",
            "source": "synthetic",
            "simulated": True,
            "host": SIMULATOR_HOST,
            "start_joints": start,
            "cue_count": len(SYNTHETIC_CUES),
        }), flush=True)

        for sequence, cue in enumerate(SYNTHETIC_CUES, start=1):
            # 复用 leica_tracker.app 的 person_guidance JSONL 提示词汇，省略所有相机和参与者字段。
            print(json.dumps({
                "event": "person_guidance",
                "source": "synthetic",
                "sequence": sequence,
                "cue": cue,
                "simulated": True,
            }), flush=True)

            if cue == "WAIT":
                time.sleep(0.5)
                continue
            if cue == "HOLD":
                if not _stop_quietly(robot):
                    raise RuntimeError("Simulator did not accept the HOLD stop request")
                break

            for target in _waypoints(cue, start):
                _move_to(robot, target, start)
        else:
            # 序列自然结束时，再返回一次中立位。
            _move_to(robot, start, start)

        final = _read_six_finite_joints(robot, "final")
        max_error = max(abs(value - origin) for value, origin in zip(final, start))
        print(json.dumps({
            "event": "simulator_finished",
            "simulated": True,
            "final_joints": final,
            "max_return_error_rad": round(max_error, 6),
        }), flush=True)
        if max_error > RETURN_TOLERANCE_RAD:
            raise RuntimeError("Simulator did not return to the captured start joints")
        completed = True
    except KeyboardInterrupt:
        interrupted = True
        print(json.dumps({"event": "interrupted", "simulated": True}), flush=True)
    finally:
        if robot is not None:
            if interrupted or not completed:
                _stop_quietly(robot)
            robot.close()

    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
