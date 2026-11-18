"""按明确请求为尚未完成实机调试的 XYZ 控制加载常驻程序。导入和构造对象时均不启动机器人。
open_resident 只有在确认安装已审核、实测数据正常且访客看门狗活跃时，才上传固定朝向的程序；
不发送使能目标。此模块未经硬件验证，也不包含倾斜开场、相机帧准入、轨迹或完整交互。
停止时关闭 RTDE 门控并停止心跳，常驻程序应在心跳到期时停下。
延迟上传仍处于禁用状态，停止后本会话不能再发使能。应用停止不等于实机静止，须通过返回的独立只读通道观察。
清理时不以新脚本覆盖可能属于其他任务的机器人程序。"""
import ipaddress
import math
import secrets
import time
from types import SimpleNamespace

from .physical_resident_engine import Installation, ResidentEngine
from .physical_errors import SessionCleanupError
from .reference_gate import rotation_vector_distance


def _retire_motion_resources(gate, watch, writer, resources, *, reason):
    """回收所有已取得的工作线程，保留清理未完成的证据。本地状态标记为停止时，异步输入线程未必已经关闭。
    启动失败和正常返回的会话都须通过有限时长的回收检查，才能接纳下一轮。"""
    errors = []
    def retire(action):
        try:
            action()
            return True
        except Exception as exc:
            errors.append(exc)
            return False

    gate_stopped = gate is not None and retire(lambda: gate.stop(reason))
    if watch is not None:
        retire(watch.close)
    if writer is not None and not gate_stopped:
        retire(writer.close)
    for resource in resources:
        if resource is not None:
            retire(resource.close)
    if gate is not None:
        finished = getattr(gate, 'close_finished', None)
        if finished is not None:
            def check_input_closed():
                if not finished.wait(2.) or gate.close_error:
                    raise RuntimeError('Physical input channel did not close cleanly')
            retire(check_input_closed)
    for thread in getattr(watch, '_threads', ()):
        def check_thread(thread=thread):
            thread.join(timeout=1.)
            if thread.is_alive():
                raise RuntimeError('Physical control reader did not close')
        retire(check_thread)
    return errors


class OpenResident:
    def __init__(self, engine, gate, watch, command, writer, reader, states):
        self.engine, self.gate, self.watch = engine, gate, watch
        self._command, self._writer = command, writer
        self.reader, self.states = reader, states
        self.closed = False

    @property
    def feedback(self):
        """通过写入通道的反馈推进引擎，不通过此入口写入目标。"""
        if self.closed or self._writer.failure:
            raise RuntimeError('Resident feedback channel unavailable')
        return self._writer.latest

    def stop(self):
        """停止心跳，保留独立的只读遥测通道。"""
        if not self.closed:
            self.engine.stop()
            self.watch.close()

    def close(self):
        if self.closed:
            return
        errors = []
        try: self.engine.stop()
        except Exception as exc: errors.append(exc)
        self.closed = True
        errors.extend(_retire_motion_resources(self.gate, self.watch, self._writer,
            (self._command, self.reader, self.states), reason='Resident session closed'))
        if errors:
            raise SessionCleanupError('Physical resident cleanup incomplete') from errors[0]


def _dependencies():
    from .horizontal_follow import StateReceiver, CommandChannel, check_dashboard
    from .ur_rtde_transport import RTDEChannel
    from .resident_guest_watchdog import ResidentMotionGate, GuestControlWatchdog
    return dict(state_factory=StateReceiver, rtde_factory=RTDEChannel,
                command_factory=CommandChannel, dashboard_check=check_dashboard,
                gate_factory=ResidentMotionGate, watch_factory=GuestControlWatchdog,
                clock=time.monotonic, sleep=time.sleep,
                session_factory=lambda: secrets.randbelow(2**30)+1)


def _state_snapshot(states, cfg, now):
    data, stamp = states.latest
    if (states.failure or not isinstance(data, dict) or
            type(stamp) not in (int, float) or not math.isfinite(stamp) or not 0 <= now-stamp <= .3):
        raise RuntimeError('Primary robot state unavailable or stale')
    mode = data.get('mode')
    if (not isinstance(mode, dict) or not all(mode.get(k) is True for k in ('connected', 'enabled', 'power'))
            or mode.get('mode') != 7 or any(mode.get(k) is not False for k in ('emergency','protective','paused'))):
        raise RuntimeError('Robot state does not permit resident startup')
    tcp = data.get('tcp_offset')
    if (not isinstance(tcp, (tuple,list)) or len(tcp) != 6 or
            any(type(x) not in (int,float) or not math.isfinite(x) for x in tcp) or
            math.dist(tcp, cfg.tcp) > .0001):
        raise RuntimeError('Active TCP differs from reviewed installation')
    return mode


def _feedback(source, cfg, now):
    fb, stamp = source.latest
    if (source.failure or not isinstance(fb, dict) or type(stamp) not in (int,float) or
            not math.isfinite(stamp) or not 0 <= now-stamp <= .15):
        raise RuntimeError('RTDE feedback unavailable or stale')
    pose, speed, joints = cfg.check_feedback(fb)
    if (math.dist(pose[:3], cfg.neutral[:3]) > .0005 or
            rotation_vector_distance(pose[3:], cfg.neutral[3:]) > .005 or
            max(abs(a-b) for a,b in zip(joints,cfg.joints)) > .01 or
            math.hypot(*speed[:3]) >= .0005 or math.hypot(*speed[3:]) >= .002):
        raise RuntimeError('Robot is not at the reviewed stationary reference')
    return fb


def _startup_snapshot(states, writer, reader, cfg, clock):
    # 在阅读验证时钟之前，请冻结每个收据。在启动期间推进的生产商不得使其的新样品看起来像未来日期。
    state = SimpleNamespace(latest=states.latest, failure=states.failure)
    writing = SimpleNamespace(latest=writer.latest, failure=writer.failure)
    reading = SimpleNamespace(latest=reader.latest, failure=reader.failure)
    now = clock()
    mode = _state_snapshot(state, cfg, now)
    feedback = _feedback(writing, cfg, now)
    _feedback(reading, cfg, now)
    return now, mode, feedback


def open_resident(host, cfg, read_guest, *, dependencies=None, startup_timeout=3.,
                  watch_interval=.05, watch_timeout=.3, require_camera=True):
    """明确连接并上传保持静止的常驻程序，不发送运动目标。
    本地控制看门狗必须获得有效的访客认证状态，所有目标写入均经过其门控。
    启动流程不推进返回的引擎，也不持续监测机器人，运行程序须承担这些职责才能开始实机交互。
    已审核有限路径的运行程序明确选择只监测控制，其他调用者默认仍要求相机在线。"""
    if not isinstance(cfg, Installation) or not callable(read_guest):
        raise ValueError('Reviewed installation and guest control reader required')
    if type(require_camera) is not bool:raise ValueError('Explicit camera watchdog policy required')
    try: host = str(ipaddress.ip_address(host))
    except ValueError: raise ValueError('Explicit robot IP address required') from None
    if any(type(v) not in (float,int) or not math.isfinite(v) or v <= 0
           for v in (startup_timeout,watch_interval,watch_timeout)) or watch_interval >= watch_timeout:
        raise ValueError('Invalid startup or watchdog intervals')
    d = _dependencies() if dependencies is None else dependencies
    states = writer = reader = command = gate = watch = None
    try:
        d['dashboard_check'](host, idle=True, robot_model=cfg.robot_model)
        states = d['state_factory'](host)
        writer = d['rtde_factory'](host, spatial=True)
        reader = d['rtde_factory'](host, spatial=True, receive_only=True)
        gate = d['gate_factory'](writer)
        recovery=getattr(read_guest,'communication_recovery',None)
        options={} if recovery is None else {'recovery':recovery}
        watch = d['watch_factory'](read_guest, gate, interval=watch_interval, timeout=watch_timeout,
                                  require_camera=require_camera,**options)
        watch.start()
        if not watch.wait_ready(startup_timeout) or not gate.can_send:
            raise RuntimeError('Guest watchdog not ready or stopped')
        _startup_snapshot(states, writer, reader, cfg, d['clock'])
        command = d['command_factory'](host)
        d['dashboard_check'](host, idle=True, robot_model=cfg.robot_model)
        _startup_snapshot(states, writer, reader, cfg, d['clock'])
        session = d['session_factory']()
        program = cfg.program(session)
        if not gate.can_send:
            raise RuntimeError('Guest watchdog stopped before resident upload')
        # 此前未发送使能包。若同时收到停止指令，则关闭门控。
        # 即使常驻程序延迟上传，也会以禁用状态启动，无法再收到本会话的使能指令。
        command.send(program)
        if not gate.can_send:
            raise RuntimeError('Guest watchdog stopped during resident upload')
        deadline = d['clock']()+startup_timeout
        while True:
            if not gate.can_send:
                raise RuntimeError('Guest watchdog stopped during resident startup')
            now, mode, fb = _startup_snapshot(states, writer, reader, cfg, d['clock'])
            if fb.get('session') == session and mode.get('running') is True:
                break
            if now >= deadline:
                raise RuntimeError('Resident startup acknowledgement timed out')
            d['sleep'](.02)
        engine = ResidentEngine(cfg, gate, session)
        return OpenResident(engine, gate, watch, command, writer, reader, states)
    except BaseException as exc:
        errors = _retire_motion_resources(gate, watch, writer, (command, reader, states),
                                         reason='Resident bootstrap failed')
        if errors:
            error = SessionCleanupError('Physical resident bootstrap cleanup incomplete')
            error.cleanup_errors = tuple(errors)
            raise error from exc
        raise
