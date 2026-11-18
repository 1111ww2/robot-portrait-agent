"""受监督、限幅的实验室水平跟随原型。只有显式传入 --execute 才能运动。"""
import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid

from .robot_state import StateReceiver


@dataclass(frozen=True)
class HorizontalLimits:
    neutral: tuple
    sign: int
    range_m: float
    segment_m: float = 0.005
    velocity: float = 0.005
    acceleration: float = 0.01
    stale_seconds: float = 0.3

    def __post_init__(self):
        if len(self.neutral) != 6 or not all(math.isfinite(x) for x in self.neutral):
            raise ValueError("A measured six-value neutral pose is required")
        if type(self.sign) is not int or self.sign not in (-1, 1):
            raise ValueError("Horizontal sign must be measured")
        if (type(self.range_m) not in (int,float) or not math.isfinite(self.range_m)
                or self.range_m <= .003 or not math.isfinite(self.neutral[0]-self.range_m)
                or not math.isfinite(self.neutral[0]+self.range_m)):
            raise ValueError("A finite, installation-specific range above 3 mm is required")
        if not 0 < self.segment_m <= .005:
            raise ValueError("Segment exceeds this lab trial's limit")
        if not 0 < self.velocity <= .005 or not 0 < self.acceleration <= .01:
            raise ValueError("Speed/acceleration exceeds the authorized slow trial")
        if not 0 < self.stale_seconds <= .5:
            raise ValueError("Watchdog must be at most 0.5 seconds")


def reviewed_horizontal_limits(cal):
    """要求为当前安装明确审核过的范围，不采用默认范围。"""
    if cal.get("status") != "HORIZONTAL_ONLY_CONFIRMED" or cal.get("base_axis") != "X":
        raise ValueError("Confirmed horizontal-only calibration is required")
    try:
        cfg=HorizontalLimits(tuple(cal["neutral_tcp_m_rad"]),cal["camera_right_sign"],range_m=cal["range_m"])
    except (KeyError,TypeError) as error:
        raise ValueError("Explicit horizontal range and calibration are required") from error
    reviewed=cal.get("reviewed_range_m")
    if type(reviewed) not in (int,float) or not math.isfinite(reviewed) or reviewed!=cfg.range_m:
        raise ValueError("Horizontal range review must match the configured range")
    return cfg


def read_target(line):
    try:
        event = json.loads(line)
        if event["event"] != "tracking_target" or event["tracking"] is not True:
            return None
        if event["motion_intent_m"]["hold"] is not False:
            return None
        command = event["command"]
        if command["down"] != 0 or command["toward_subject"] != 0:
            return None
        right = command["right"]
        if isinstance(right, bool) or not isinstance(right, (float, int)):
            return None
        return float(right) if math.isfinite(right) and -1 <= right <= 1 else None
    except (ValueError, KeyError, TypeError):
        return None


def next_target(cfg, pose, right, age):
    if len(pose) != 6 or not all(math.isfinite(x) for x in pose):
        raise ValueError("Invalid current pose")
    if abs(pose[0] - cfg.neutral[0]) > cfg.range_m + .001:
        raise ValueError("Robot left the horizontal envelope")
    if any(abs(pose[i] - cfg.neutral[i]) > .001 for i in (1, 2)):
        raise ValueError("Robot changed fixed Y/Z")
    if math.dist(pose[3:], cfg.neutral[3:]) > .01:
        raise ValueError("Robot changed fixed orientation")
    if right is None or not math.isfinite(age) or age < 0 or age > cfg.stale_seconds:
        return None
    if not math.isfinite(right) or not -1 <= right <= 1:
        return None
    wanted = cfg.neutral[0] + cfg.sign * cfg.range_m * right
    error = wanted - pose[0]
    if abs(error) < .0005:
        return None
    target = list(cfg.neutral)
    target[0] = pose[0] + max(-cfg.segment_m, min(cfg.segment_m, error))
    target[0] = max(cfg.neutral[0] - cfg.range_m, min(cfg.neutral[0] + cfg.range_m, target[0]))
    return target


class VisionSamples:
    def __init__(self):
        self.latest = (None, 0.0, 0, 0)
        self.previous = -1
        self.loss_generation = 0

    def accept(self, line, now):
        event = json.loads(line)
        stamp = event["source_monotonic_ns"]
        if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp <= self.previous:
            raise ValueError("Camera source timestamp stopped advancing")
        self.previous = stamp
        source_time = stamp / 1_000_000_000
        if source_time > now + .001:
            raise ValueError("Camera source clock is invalid")
        right = read_target(line) if now - source_time <= .3 else None
        good = self.latest[2] + 1 if right is not None else 0
        if right is None:
            self.loss_generation += 1
        self.latest = (right, source_time, good, self.loss_generation)


class VisionReceiver(VisionSamples):
    def __init__(self, camera):
        super().__init__()
        self.failure = None
        self.process = subprocess.Popen([
            sys.executable, "-u", "-m", "leica_tracker", "--camera", str(camera),
            "--horizontal-only", "--emit-every-frames", "1",
            "--max-right-m", "0.10", "--max-step-m", "0.002",
        ], stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while line := self.process.stdout.readline(8193):
                if len(line) > 8192 or not line.endswith("\n"):
                    raise ValueError("Invalid camera event length")
                self.accept(line, time.monotonic())
            self.failure = "Camera preview exited"
        except Exception as error:
            self.failure = str(error)

    def close(self):
        if self.process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                               capture_output=True, timeout=5)
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self.thread.join(timeout=1)


def send_script(host, program):
    # 限制写入可避免底层库无限排队，每条 MoveL 都是有限动作。
    with socket.create_connection((host, 30002), timeout=.5) as sock:
        sock.settimeout(.5)
        sock.sendall((program.rstrip() + "\n").encode("ascii"))


class CommandChannel:
    """使用一个有序连接，不排队发送程序，写入限时 100 毫秒。"""
    def __init__(self, host):
        self.sock = socket.create_connection((host, 30002), timeout=.5)
        self.sock.settimeout(.1)
        self.closed = False
        self.failure = None
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self):
        try:
            while not self.closed:
                try:
                    if not self.sock.recv(65536):
                        raise EOFError("Command connection closed")
                except socket.timeout:
                    continue
        except Exception as error:
            if not self.closed:
                self.failure = str(error)

    def send(self, program):
        if self.closed or self.failure:
            raise RuntimeError(self.failure or "Command connection closed")
        try:
            self.sock.sendall((program.rstrip() + "\n").encode("ascii"))
        except OSError as error:
            # 如果一段脚本只发送了一部分，就不能通过同一连接继续发送下一段程序。
            self.failure = str(error)
            self.close()
            raise

    def close(self):
        self.closed = True
        self.sock.close()
        self.thread.join(timeout=.2)


def check_dashboard(host, idle=False, *, robot_model=None):
    from .physical_robot_model import dashboard_model_names
    model_names = dashboard_model_names(robot_model) if robot_model is not None else None
    expected = [("safetystatus", "Safetystatus: NORMAL"), ("is in remote control", "true")]
    if idle:
        expected.append(("running", "Program running: false"))
    with socket.create_connection((host, 29999), timeout=.5) as sock:
        sock.settimeout(.5)
        stream = sock.makefile("rb")
        stream.readline(1024)
        for command, required in expected:
            sock.sendall((command + "\n").encode("ascii"))
            actual = stream.readline(1024).decode().strip()
            if actual != required:
                raise RuntimeError(f"{command}: {actual}")
        if model_names is not None:
            sock.sendall(b'get robot model\n')
            actual_model = stream.readline(1024).decode().strip()
            if actual_model not in model_names:
                raise RuntimeError(f'Physical robot model mismatch: {actual_model}')
            sock.sendall(b'PolyscopeVersion\n')
            software = stream.readline(1024).decode().strip()
            if not software.startswith('URSoftware 5.'):
                raise RuntimeError(f'Physical robot software is not {robot_model}/e-Series: {software}')


class DashboardReceiver:
    """管理面板请求不能阻塞动作看门狗。"""
    def __init__(self, host):
        self.host = host
        self.checked_at = time.monotonic()
        self.failure = None
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while not self.closed.is_set():
                check_dashboard(self.host)
                self.checked_at = time.monotonic()
                self.closed.wait(.5)
        except Exception as error:
            self.failure = str(error)

    def close(self):
        self.closed.set()
        self.thread.join(timeout=.1)


def movement_program(pose, target, cfg, marker):
    expected = "p[" + ",".join(f"{v:.9f}" for v in pose) + "]"
    destination = "p[" + ",".join(f"{v:.9f}" for v in target) + "]"
    return ("def leica_horizontal_segment():\n"
            f"  if point_dist(get_actual_tcp_pose(), {expected}) < 0.001:\n"
            f"    target = {destination}\n"
            "    if is_within_safety_limits(target):\n"
            f"      movel(target, a={cfg.acceleration}, v={cfg.velocity}, r=0)\n"
            "    end\n  end\n"
            f'  textmsg("{marker}")\nend\n')


def stop_program(marker):
    return f'def leica_horizontal_hold():\n  stopl(0.1)\n  textmsg("{marker}")\nend\n'


class MotionSupervisor:
    """非阻塞地切换状态，只有收到新的机器人明确确认后才恢复。"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.pending = None
        self.seen_loss = -1
        self.ready = False

    def step(self, now, current, state_time, sample, ack):
        right, vision_time, good, loss = sample
        valid = right is not None and good >= 3 and 0 <= now - vision_time <= self.cfg.stale_seconds
        pose = current["pose"]
        target = next_target(self.cfg, pose, right if valid else None, now - vision_time)
        running = current["mode"]["running"]
        lost = loss != self.seen_loss
        self.seen_loss = loss
        if self.pending:
            kind, marker, sent_at, goal = self.pending
            reversing = kind == "move" and valid and (self.cfg.neutral[0] + self.cfg.sign * self.cfg.range_m * right - pose[0]) * (goal[0] - pose[0]) < -1e-6
            if kind == "move" and (not valid or lost or reversing):
                return self._stop(now)
            if ack[0] == marker and ack[1] >= sent_at and state_time > ack[1] and not running:
                if kind == "move" and abs(pose[0] - goal[0]) > .0006:
                    raise RuntimeError("Robot did not reach the guarded segment target")
                self.pending = None
                self.ready = True
                return None
            if now - sent_at > (3 if kind == "stop" else 12):
                raise RuntimeError(f"Robot {kind} acknowledgement timed out")
            return None
        if not self.ready:
            return self._stop(now)
        if running:
            raise RuntimeError("Unexpected external robot program")
        if target is not None and not lost:
            marker = "LEICA_ACK_" + uuid.uuid4().hex
            self.pending = ("move", marker, now, target)
            return "move", movement_program(pose, target, self.cfg, marker)
        return None

    def _stop(self, now):
        marker = "LEICA_ACK_" + uuid.uuid4().hex
        self.pending = ("stop", marker, now, None)
        self.ready = False
        return "stop", stop_program(marker)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--duration", type=float, default=10, help="seconds; 0 runs until stopped")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("duration must be finite and nonnegative")
    cal = json.loads(args.calibration.read_text(encoding="utf-8-sig"))
    cfg = reviewed_horizontal_limits(cal)
    host = cal["robot_ip"]
    check_dashboard(host, idle=True)
    states = StateReceiver(host)
    dashboard = DashboardReceiver(host)
    vision = None
    channel = None
    commands = stops = 0
    supervisor = MotionSupervisor(cfg)
    started = time.monotonic()
    last_status = 0.0
    extrema = [float("inf"), float("-inf")]
    try:
        vision = VisionReceiver(cal["camera"])
        while states.latest[0] is None or vision.latest[1] == 0:
            if states.failure or vision.failure or time.monotonic() - started > 90:
                raise RuntimeError(states.failure or vision.failure or "Input startup timed out")
            time.sleep(.05)
        if args.execute:
            check_dashboard(host, idle=True)
            channel = CommandChannel(host)
        started = time.monotonic()
        while not args.duration or time.monotonic() - started < args.duration:
            now = time.monotonic()
            current, state_time = states.latest
            if states.failure or vision.failure or now - state_time > .3:
                raise RuntimeError(states.failure or vision.failure or "Robot telemetry is stale")
            if dashboard.failure or now - dashboard.checked_at > 2:
                raise RuntimeError(dashboard.failure or "Dashboard health check is stale")
            mode = current["mode"]
            if not (mode["connected"] and mode["enabled"] and mode["power"] and mode["mode"] == 7
                    and not mode["emergency"] and not mode["protective"] and not mode["paused"]):
                raise RuntimeError("Robot state no longer permits the trial")
            if math.dist(current["tcp_offset"], cal["tcp_offset_m_rad"]) > .0001:
                raise RuntimeError("TCP configuration changed")
            sample = vision.latest
            right, vision_time, good, loss_generation = sample
            age = now - vision_time
            valid = right is not None and good >= 3 and age <= cfg.stale_seconds
            target = next_target(cfg, current["pose"], right if valid else None, age)
            offset = current["pose"][0] - cfg.neutral[0]
            extrema = [min(extrema[0], offset), max(extrema[1], offset)]
            if args.execute:
                action = supervisor.step(now, current, state_time, sample, states.ack)
                if action:
                    channel.send(action[1])
                    commands += int(action[0] == "move")
                    stops += int(action[0] == "stop")
            if now - last_status > 1:
                print(json.dumps({"execute": args.execute, "tracking": valid, "right": right,
                    "x_offset_cm": round(offset * 100, 3), "commands": commands, "hold_stops": stops,
                    "phase": supervisor.pending[0] if supervisor.pending else "idle",
                    "vision_age_s": round(age, 3)}), flush=True)
                last_status = now
            time.sleep(.02)
    finally:
        if channel is not None:
            try:
                channel.send("stopl(0.1)")
            except (OSError, RuntimeError):
                channel.close()
                try:
                    send_script(host, "stopl(0.1)")
                except OSError as error:
                    print(f"STOP REQUEST FAILED: {error}", file=sys.stderr, flush=True)
            channel.close()
        if vision is not None:
            vision.close()
        states.close()
        dashboard.close()
        print(json.dumps({"finished": True, "execute": args.execute, "commands": commands,
            "hold_stops": stops, "elapsed_s": round(time.monotonic() - started, 2),
            "observed_x_offsets_m": extrema if commands or last_status else None}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
