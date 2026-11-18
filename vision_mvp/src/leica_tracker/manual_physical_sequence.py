"""在现场监督下显式启动关节路径试验，独立于 iPad 相机。取消操作由本地操作员控制。
机器人侧心跳超时、速度和 TCP 检查及独立遥测均沿用开场流程。
此功能属于应用层控制，不替代安全等级控制器或经验收的急停。导入模块不会连接硬件，未审核的路径不能启用。"""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import threading
import time

from .physical_intro_bootstrap import open_intro
from .physical_intro_program import IntroProgram
from .physical_resident_bootstrap import _dependencies


class ManualControl:
    def __init__(self):
        self._lock = threading.Lock()
        self._gate = None
        self.cancelled = threading.Event()

    def bind(self, gate):
        with self._lock:
            if self._gate is not None or self.cancelled.is_set():
                gate.stop('Manual sequence cancelled or already bound')
                raise RuntimeError('Manual sequence requires a new explicit invocation')
            self._gate = gate
            gate.arm()

    def stop(self, reason='Terminal Stop'):
        with self._lock:
            self.cancelled.set()
            if self._gate is not None:
                self._gate.stop(reason)

    def snapshot(self):
        # 此处要求真实状态，不能用模拟的访客或相机快照代替。
        return {'source': 'local_manual_sequence', 'active': not self.cancelled.is_set()}


class _ManualWatch:
    def __init__(self, control, gate):
        self.control, self.gate = control, gate

    def start(self):
        self.control.bind(self.gate)

    def wait_ready(self, timeout):
        return not self.control.cancelled.is_set() and self.gate.can_send

    def close(self):
        self.control.stop('Manual sequence closed')


def open_manual_sequence(host, profile, control, *, dependencies=None, program_type=IntroProgram):
    if not isinstance(control, ManualControl):
        raise ValueError('Explicit local operator control required')
    if not isinstance(program_type, type) or not issubclass(program_type, IntroProgram):
        raise ValueError('A reviewed physical program type is required')
    program_type(profile, 1)  # 发起任何网络通信前先验证输入。
    if dependencies is None:
        deps = _dependencies()
        from .horizontal_follow import DashboardReceiver
        deps['dashboard_factory'] = DashboardReceiver
    else:
        deps = dict(dependencies)
    deps['watch_factory'] = lambda read, gate, **kwargs: _ManualWatch(control, gate)
    return open_intro(host, profile, control.snapshot, dependencies=deps, _program_type=program_type)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    document = json.loads(args.profile.read_text(encoding='utf-8-sig'))
    if document.get('mode') != 'onsite_manual_14_point_sequence':
        parser.error('Explicit onsite manual 14-point profile required')
    host, profile = document['robot_ip'], document['intro']
    program = IntroProgram(profile, 1)
    if len(program.waypoints) != 14:
        parser.error('Exactly 14 reviewed waypoints required')
    if not args.execute:
        print('14-point manual route validated offline; no robot contacted.')
        return 0
    control = ManualControl()
    def read_stop():
        try:
            # 收到任意输入（包括回车）或终端关闭均取消操作。
            sys.stdin.readline()
        finally:
            control.stop('Terminal input or disconnect')
    print('MANUAL REAL-ROBOT SEQUENCE: Enter stops. No iPad dependency. No gripper commands.', flush=True)
    threading.Thread(target=read_stop, daemon=True).start()
    opened = None
    trace = deque(maxlen=300)
    outcome = 'starting'
    error = None
    final_feedback = None
    try:
        opened = open_manual_sequence(host, profile, control)
        print('Resident ready at measured reference; executing the reviewed 14-point route.', flush=True)
        last_writer = last_reader = None
        last_report = 0.
        while not control.cancelled.is_set():
            writer, ws = opened.writer.latest
            reader, rs = opened.reader.latest
            now = time.monotonic()
            if (opened.writer.failure or opened.reader.failure or
                    not 0 <= now-ws <= .15 or not 0 <= now-rs <= .15):
                raise RuntimeError('Manual route telemetry unavailable or stale')
            if (last_writer is None or ws > last_writer) and (last_reader is None or rs > last_reader):
                trace.append({'sampled_at':now,'writer_age_s':now-ws,'reader_age_s':now-rs,
                              'resident_state':writer.get('state'),'ack':writer.get('sequence'),
                              'reader_ack':reader.get('sequence'),'writer_timestamp':writer.get('timestamp'),
                              'reader_timestamp':reader.get('timestamp'),
                              'pending':opened.progress.pending,'joints':reader.get('joints'),
                              'tcp_speed':reader.get('tcp_speed')})
                result = opened.tick()
                last_writer, last_reader = opened.progress.last_writer, opened.progress.last_reader
                if result == 'COMPLETE':
                    outcome = 'complete'
                    print(json.dumps({'event': 'manual_route_complete', 'points': 14,
                                      'joints': reader['joints'], 'tcp_pose': reader['tcp_pose'],
                                      'joint_speed': reader['joint_speed'], 'tcp_speed': reader['tcp_speed']}), flush=True)
                    return 0
            if now-last_report >= 1:
                print(json.dumps({'event': 'manual_route_feedback', 'state': writer.get('state'),
                                  'joints': reader.get('joints'), 'tcp_speed': reader.get('tcp_speed')}), flush=True)
                last_report = now
            time.sleep(.01)
        print('Stop requested; no automatic retry or return.', flush=True)
        outcome = 'cancelled'
        return 1
    except Exception as exc:
        control.stop('Manual route fault')
        outcome, error = 'fault', str(exc)
        print('Manual route stopped: '+str(exc), flush=True)
        return 2
    finally:
        control.stop('Manual route finished')
        if opened is not None:
            try:
                opened.stop()
                # 写入端关闭时，独立的接收端仍保持打开。采样到静止姿态只是观测证据，不能当作急停证明。
                deadline = time.monotonic()+3.
                while time.monotonic()<deadline:
                    sample, stamp = opened.reader.latest
                    now = time.monotonic()
                    if not opened.reader.failure and 0 <= now-stamp <= .15:
                        final_feedback = {'sampled_at':stamp,'joints':sample.get('joints'),
                            'tcp_pose':sample.get('tcp_pose'),'tcp_speed':sample.get('tcp_speed'),
                            'joint_speed':sample.get('joint_speed')}
                        if (now+2.5 < deadline and outcome != 'complete'):
                            time.sleep(.02);continue
                        if (max(abs(v) for v in sample['joint_speed']) < .003 and
                                max(abs(v) for v in sample['tcp_speed']) < .0005):break
                    time.sleep(.02)
            finally:
                opened.close()
        report = {'outcome':outcome,'error':error,'trace':list(trace),
                  'ack_diagnostic':opened.progress.ack_diagnostic if opened else None,
                  'last_independent_feedback':final_feedback}
        path = args.profile.with_suffix('.run.json')
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print('Run evidence saved: '+str(path),flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
