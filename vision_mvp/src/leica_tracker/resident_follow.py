"""具有RTDE目标和程序内水平HOLD的单个居民UR计划。"""
import argparse
import json
import math
from pathlib import Path
import secrets
import time

from .horizontal_follow import (HorizontalLimits, reviewed_horizontal_limits, VisionReceiver, StateReceiver,
    DashboardReceiver, CommandChannel, check_dashboard, next_target, send_script)
from .ur_rtde_transport import (RTDEChannel, SEQ_REGISTER, ENABLE_REGISTER,
    SESSION_REGISTER, TARGET_REGISTER, STATE_REGISTER, ACK_REGISTER,
    OUTPUT_SESSION_REGISTER)

LINEAR_SPEED_LIMIT = .007  # m/s测量信封，命令保持在<= .005。
ANGULAR_SPEED_LIMIT = .01  # rad/s，命令的角速度保持为零。


def check_tcp_speed(speed):
    if len(speed) != 6 or not all(math.isfinite(v) for v in speed):
        raise RuntimeError('Invalid TCP velocity')
    linear, angular = math.hypot(*speed[:3]), math.hypot(*speed[3:])
    if linear >= LINEAR_SPEED_LIMIT:
        raise RuntimeError(f'TCP linear velocity exceeds measurement limit: {linear}')
    if angular >= ANGULAR_SPEED_LIMIT:
        raise RuntimeError(f'TCP angular velocity exceeds measurement limit: {angular}')
    return linear, angular


class StreamPolicy:
    def __init__(self, cfg, session):
        self.cfg, self.session = cfg, session
        self.sequence = 0
        self.pending = None
        self.need_hold = True
        self.seen_loss = -1
        self.last_send = -1.
        self.handshake_complete = False

    def next(self, now, pose, sample, feedback):
        if feedback['session'] != self.session:
            raise RuntimeError('Resident session changed')
        if feedback['state'] < 0:
            raise RuntimeError(f'Resident invariant fault {feedback["state"]}')
        if feedback['state'] == 2 and self.handshake_complete:
            raise RuntimeError('Resident communication timeout; explicit restart required')
        if self.pending:
            sequence, sent, enabled = self.pending
            if now - sent > .2:
                raise RuntimeError('Resident input acknowledgement timed out')
            if feedback['sequence'] == sequence:
                self.pending = None
                if enabled == 0 and feedback['state'] == 0:
                    self.need_hold = False
                    self.handshake_complete = True
        right, source, good, loss = sample
        valid = right is not None and good >= 3 and 0 <= now-source <= self.cfg.stale_seconds
        next_target(self.cfg, pose, right if valid else None, now-source)
        if loss != self.seen_loss or not valid or feedback['state'] == 2:
            self.need_hold = True
        self.seen_loss = loss
        if self.pending or now - self.last_send < .02:
            return None
        enabled = int(valid and not self.need_hold)
        target = self.cfg.neutral[0] + self.cfg.sign * self.cfg.range_m * right if enabled else pose[0]
        margin = self.cfg.range_m - .003
        target = max(self.cfg.neutral[0]-margin, min(self.cfg.neutral[0]+margin, target))
        self.sequence += 1
        self.pending = (self.sequence, now, enabled)
        self.last_send = now
        return self.sequence, enabled, self.session, target


def resident_program(cfg, tcp, session):
    if not 0 < session < 2**31 or len(tcp) != 6 or not all(math.isfinite(v) for v in tcp):
        raise ValueError('Invalid resident session or TCP')
    neutral = 'p[' + ','.join(f'{v:.9f}' for v in cfg.neutral) + ']'
    offset = 'p[' + ','.join(f'{v:.9f}' for v in tcp) + ']'
    low, high = cfg.neutral[0]-cfg.range_m, cfg.neutral[0]+cfg.range_m
    return f'''def leica_resident_horizontal():
  def bounded(value, lower, upper):
    if value < lower:
      return lower
    elif value > upper:
      return upper
    end
    return value
  end
  global packet_age = 1.0
  thread heartbeat_clock():
    while True:
      global packet_age = packet_age + get_steptime()
      sync()
    end
  end
  clock_thread = run heartbeat_clock()
  neutral = {neutral}
  expected_tcp = {offset}
  previous = 0
  consumed = 0
  armed = False
  enabled = False
  moving = False
  initial_pose = get_actual_tcp_pose()
  target_x = initial_pose[0]
  write_output_integer_register({STATE_REGISTER}, 2)
  write_output_integer_register({ACK_REGISTER}, 0)
  write_output_integer_register({OUTPUT_SESSION_REGISTER}, {session})
  while True:
    if packet_age > 0.25:
      armed = False
      enabled = False
    end
    seq = read_input_integer_register({SEQ_REGISTER})
    flag = read_input_integer_register({ENABLE_REGISTER})
    token = read_input_integer_register({SESSION_REGISTER})
    incoming_x = read_input_float_register({TARGET_REGISTER})
    seq_again = read_input_integer_register({SEQ_REGISTER})
    if seq == seq_again and seq > previous:
      if token == {session} and (flag == 0 or flag == 1) and incoming_x >= {low:.9f} and incoming_x <= {high:.9f}:
        previous = seq
        consumed = seq
        global packet_age = 0.0
        if flag == 0:
          armed = True
          enabled = False
        elif armed:
          enabled = True
          target_x = incoming_x
        end
      else:
        armed = False
        enabled = False
      end
    end
    actual = get_actual_tcp_pose()
    actual_speed = get_actual_tcp_speed()
    tcp_now = get_tcp_offset()
    fault = 0
    if not (actual[0] >= {low:.9f} and actual[0] <= {high:.9f}):
      fault = 1
    elif not (norm(actual[1]-neutral[1]) < 0.001 and norm(actual[2]-neutral[2]) < 0.001):
      fault = 2
    elif not (norm([actual[3]-neutral[3],actual[4]-neutral[4],actual[5]-neutral[5]]) < 0.01):
      fault = 3
    elif not (norm([tcp_now[0]-expected_tcp[0],tcp_now[1]-expected_tcp[1],tcp_now[2]-expected_tcp[2],tcp_now[3]-expected_tcp[3],tcp_now[4]-expected_tcp[4],tcp_now[5]-expected_tcp[5]]) < 0.0001):
      fault = 4
    elif not (norm([actual_speed[0],actual_speed[1],actual_speed[2]]) < {LINEAR_SPEED_LIMIT}):
      fault = 5
    elif not (norm([actual_speed[3],actual_speed[4],actual_speed[5]]) < {ANGULAR_SPEED_LIMIT}):
      fault = 7
    end
    if fault != 0:
      textmsg("LEICA_FAULT_CODE_SPEED_", [fault,actual_speed[0],actual_speed[1],actual_speed[2],actual_speed[3],actual_speed[4],actual_speed[5]])
      stopl(0.1)
      write_output_integer_register({STATE_REGISTER}, -fault)
      break
    end
    state = 0
    if not armed:
      state = 2
    end
    if enabled and armed:
      target_x = bounded(target_x, {low+.003:.9f}, {high-.003:.9f})
      vx = bounded((target_x-actual[0])*1.0, -{cfg.velocity}, {cfg.velocity})
      if norm(target_x-actual[0]) < 0.0003:
        vx = 0.0
      end
      probe = p[actual[0]+vx*0.4,actual[1],actual[2],actual[3],actual[4],actual[5]]
      if not is_within_safety_limits(probe):
        textmsg("LEICA_FAULT_SAFETY_", probe)
        stopl(0.1)
        write_output_integer_register({STATE_REGISTER}, -6)
        break
      end
      speedl([vx,0,0,0,0,0], a={cfg.acceleration}, t=0.02)
      moving = True
      if norm(vx) > 0.00001:
        state = 1
      end
    else:
      if moving:
        stopl(0.1)
        moving = False
      end
      sleep(0.02)
    end
    write_output_integer_register({STATE_REGISTER}, state)
    write_output_integer_register({ACK_REGISTER}, consumed)
  end
  kill clock_thread
end
'''


def check_state(current, stamp, cal, now):
    if current is None or not 0 <= now-stamp <= .3:
        raise RuntimeError('Robot telemetry is stale')
    mode = current['mode']
    if not (mode['connected'] and mode['enabled'] and mode['power'] and mode['mode'] == 7
            and not mode['emergency'] and not mode['protective'] and not mode['paused']):
        raise RuntimeError('Robot state no longer permits following')
    if math.dist(current['tcp_offset'], cal['tcp_offset_m_rad']) > .0001:
        raise RuntimeError('TCP configuration changed')


def close_session(uploaded, host, channel, rtde, vision, states, dashboard, stop_acceleration=.1):
    if not math.isfinite(stop_acceleration) or not .1 <= stop_acceleration <= .6:
        raise ValueError('Invalid stop deceleration')
    failures = []
    try:
        if uploaded:
            try:
                channel.send(f'stopl({stop_acceleration})')
            except (OSError, RuntimeError):
                channel.close()
                try:
                    send_script(host, f'stopl({stop_acceleration})')
                except Exception as error:
                    failures.append(f'STOP REQUEST FAILED: {error}')
    finally:
        for resource in (rtde, channel, vision, states, dashboard):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    failures.append(f'Resource cleanup failed: {error}')
    if failures:
        raise RuntimeError('; '.join(failures))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--duration', type=float, default=0.)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error('Invalid duration')
    cal = json.loads(args.calibration.read_text(encoding='utf-8-sig'))
    cfg = reviewed_horizontal_limits(cal)
    host = cal['robot_ip']
    check_dashboard(host, idle=True)
    states = StateReceiver(host)
    dashboard = DashboardReceiver(host)
    vision = rtde = channel = None
    uploaded = False
    sends = 0
    max_linear = max_angular = 0.
    started = time.monotonic()
    try:
        vision = VisionReceiver(cal['camera'])
        while states.latest[0] is None or vision.latest[1] == 0:
            if states.failure or vision.failure or time.monotonic()-started > 90:
                raise RuntimeError(states.failure or vision.failure or 'Camera startup timed out')
            time.sleep(.05)
        session = secrets.randbelow(2**30)+1
        policy = StreamPolicy(cfg, session)
        if args.execute:
            check_dashboard(host, idle=True)
            check_state(*states.latest, cal, time.monotonic())
            next_target(cfg, states.latest[0]['pose'], None, 0)
            rtde = RTDEChannel(host)
            channel = CommandChannel(host)
            # 本地库设置可能需要几秒钟。在设置后，在任何程序上传之前立即重新检查物理状态和所有权。
            check_dashboard(host, idle=True)
            current, stamp = states.latest
            check_state(current, stamp, cal, time.monotonic())
            next_target(cfg, current['pose'], None, 0)
            if rtde.failure:
                raise RuntimeError(rtde.failure)
            # 设置在发送前：部分上传仍然需要停止清理。
            uploaded = True
            channel.send(resident_program(cfg, cal['tcp_offset_m_rad'], session))
            deadline = time.monotonic()+3
            while True:
                fb, stamp = rtde.latest
                if rtde.failure or states.failure or time.monotonic() > deadline:
                    raise RuntimeError(rtde.failure or states.failure or 'Resident startup handshake failed')
                if fb and fb['session'] == session and states.latest[0]['mode']['running']:
                    break
                time.sleep(.02)
        started = time.monotonic()
        last_status = 0.
        while not args.duration or time.monotonic()-started < args.duration:
            # 在读取消费者时钟之前，先读取快照生产者值。并发更新不得将新样本变成负年龄。
            current, stamp = states.latest
            sample = vision.latest
            fb, feedback_time = rtde.latest if args.execute else (None, 0.)
            now = time.monotonic()
            if args.stop_file and args.stop_file.exists():
                break
            if states.failure or vision.failure or dashboard.failure or now-dashboard.checked_at > 2:
                raise RuntimeError(states.failure or vision.failure or dashboard.failure or 'Dashboard stale')
            check_state(current, stamp, cal, now)
            next_target(cfg, current['pose'], sample[0], now-sample[1])
            if args.execute:
                if rtde.failure or now-feedback_time > .15 or not current['mode']['running']:
                    raise RuntimeError(rtde.failure or 'Resident feedback stale or program ended')
                linear, angular = check_tcp_speed(fb['tcp_speed'])
                max_linear, max_angular = max(max_linear, linear), max(max_angular, angular)
                update = policy.next(now, current['pose'], sample, fb)
                if update:
                    rtde.send(*update)
                    sends += 1
            if now-last_status >= 1:
                print(json.dumps({'backend':'ur-rtde-1.6.5','execute':args.execute,'resident_running':current['mode']['running'],
                    'resident_state':fb['state'] if fb else None, 'rtde_updates':sends,
                    'tracking':sample[0] is not None and sample[2]>=3 and now-sample[1]<=.3,
                    'right':sample[0], 'x_offset_cm':round((current['pose'][0]-cfg.neutral[0])*100,3),
                    'vision_age_s':round(now-sample[1],3), 'session':session,
                    'peak_linear_mm_s':round(max_linear*1000,3),
                    'peak_angular_rad_s':round(max_angular,6)}),flush=True)
                last_status = now
            time.sleep(.01)
    finally:
        close_session(uploaded, host, channel, rtde, vision, states, dashboard)
        print(json.dumps({'finished':True,'rtde_updates':sends,'elapsed_s':round(time.monotonic()-started,2),
                          'robot_notice':states.notice}),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
