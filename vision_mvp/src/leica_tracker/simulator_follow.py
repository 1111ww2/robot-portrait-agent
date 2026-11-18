"""从合成或Pocket 3 JSONL输入驱动教师的本地模拟器。
仅限模拟：主机/端口是故意固定的，在打开URX之前需要模拟器身份。在隔离的教师模拟器环境中运行。"""
import argparse
import http.client
import json
import math
from pathlib import Path
import sys
import threading
import time

from .horizontal_follow import VisionSamples

SIM_HOST = '127.0.0.1'
SIM_HTTP_PORT = 8765


def synthetic_event(now, elapsed):
    """现有的愿景模式，包括定期的损失/恢复演习。"""
    tracked = elapsed % 24 < 20 or elapsed % 24 >= 23
    right = .8 * math.sin(2 * math.pi * elapsed / 12) if tracked else 0.
    return dict(event='tracking_target', source='synthetic',
                source_monotonic_ns=round(now*1_000_000_000),
                timestamp_ms=round(elapsed*1000), tracking=tracked,
                coordinate_frame='camera_relative_normalized',
                command=dict(right=right, down=0., toward_subject=0.),
                motion_intent_m=dict(right=right*.1, down=0., toward_subject=0., hold=not tracked),
                reason='SIMULATED_TRACKING' if tracked else 'PERSON_LOST')


def target_velocity(neutral, pose, sample, now, *, range_m=.1):
    if not math.isfinite(range_m) or not .01 <= range_m <= .25:
        raise ValueError('Simulator range must be between 1 and 25 cm')
    if (len(neutral) != 6 or len(pose) != 6
            or not all(math.isfinite(x) for x in [*neutral, *pose])):
        raise ValueError('Invalid simulator pose')
    if abs(pose[0]-neutral[0]) > range_m+.001:
        raise ValueError('Simulator left the horizontal range')
    right, stamp, good, _ = sample
    if right is None or good < 3 or not 0 <= now-stamp <= .3:
        return [0.]*6
    if not math.isfinite(right) or not -1 <= right <= 1:
        raise ValueError('Invalid horizontal input')
    # 只调整虚拟演示速度，不把这些数值传入实体控制。
    target_limit = range_m-.003
    target_x = neutral[0] + max(-target_limit, min(target_limit, -range_m*right))
    error = target_x-pose[0]
    vx = max(-.08, min(.08, error*2)) if abs(error) >= .0003 else 0.
    return [vx, 0., 0., 0., 0., 0.]


def verify_identity(manifest):
    target = manifest.get('target', {})
    if (target.get('kind') != 'simulator'
            or target.get('authority_origin') != 'simulation'
            or not target.get('instance_id')):
        raise ValueError('Teacher simulator identity required; refusing robot target')
    return target['instance_id']


def get_json(path):
    connection = http.client.HTTPConnection(SIM_HOST, SIM_HTTP_PORT, timeout=.5)
    try:
        connection.request('GET', path)
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f'Simulator HTTP status {response.status}')
        return json.loads(response.read(2_000_000))
    finally:
        connection.close()


class InputStream(VisionSamples):
    def __init__(self):
        super().__init__()
        self.failure = None
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while line := sys.stdin.readline(8193):
                if len(line) > 8192 or not line.endswith('\n'):
                    raise ValueError('Invalid JSONL input length')
                self.accept(line, time.monotonic())
            self.failure = 'Input stream closed'
        except Exception as error:
            self.failure = str(error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('synthetic', 'stdin'), default='synthetic')
    parser.add_argument('--duration', type=float, default=0.)
    parser.add_argument('--range-cm', type=float, default=10.,
                        help='Virtual horizontal half-range in cm (1–25; default 10)')
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--input-log', type=Path, help='Optional emitted synthetic JSONL evidence')
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error('Duration must be nonnegative')
    if not math.isfinite(args.range_cm) or not 1 <= args.range_cm <= 25:
        parser.error('Simulator range must be between 1 and 25 cm')
    instance = verify_identity(get_json('/api/workbench/v1/manifest'))
    import urx  # 教师补丁库的图书馆，与真实的RTDE环境隔离。

    robot = None
    log = None
    samples = VisionSamples() if args.source == 'synthetic' else InputStream()
    count = 0
    started = time.monotonic()
    try:
        if args.input_log:
            log = args.input_log.open('w', encoding='utf-8', buffering=1)
        robot = urx.Robot(SIM_HOST, use_rt=False)
        neutral = get_json('/api/sim/v1/motion')['tcp_pose']
        started = time.monotonic()
        print(json.dumps(dict(event='simulator_connected', source=args.source,
                              instance=instance, neutral_tcp=neutral, simulated=True,
                              range_cm=args.range_cm)), flush=True)
        while not args.duration or time.monotonic()-started < args.duration:
            if args.stop_file and args.stop_file.exists():
                break
            tick = time.monotonic()
            if args.source == 'synthetic':
                event = synthetic_event(tick, tick-started)
                line = json.dumps(event)
                samples.accept(line, tick)
                if log:
                    log.write(line+'\n')
            elif samples.failure:
                raise RuntimeError(samples.failure)
            state = get_json('/api/sim/v1/motion')
            if (state.get('instance_id') != instance or state.get('backend') != 'sim'
                    or state.get('service') != 'urx-browser-simulator'):
                raise RuntimeError('Simulator instance changed')
            sample = samples.latest
            velocity = target_velocity(neutral, state['tcp_pose'], sample, time.monotonic(),
                                       range_m=args.range_cm/100)
            # 如果生产者/适配器消失，有限速度命令将过期。
            robot.speedl(velocity, acc=.3, min_time=.15)
            count += 1
            print(json.dumps(dict(event='simulator_sync', sequence=count, source=args.source,
                simulated=True, source_monotonic_ns=round(sample[1]*1e9),
                tracking=sample[0] is not None and sample[2]>=3 and tick-sample[1]<=.3,
                right=sample[0], tcp_pose=state['tcp_pose'], joints=state['joints'],
                velocity=velocity, x_offset_cm=(state['tcp_pose'][0]-neutral[0])*100,
                source_age_ms=round((time.monotonic()-sample[1])*1000,2))), flush=True)
            time.sleep(max(0., .1-(time.monotonic()-tick)))
    finally:
        if robot is not None:
            try:
                robot.stopl(.3)
            finally:
                robot.close()
        if log:
            log.close()
        print(json.dumps(dict(event='simulator_finished', simulated=True,
                              updates=count, elapsed_s=time.monotonic()-started)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
