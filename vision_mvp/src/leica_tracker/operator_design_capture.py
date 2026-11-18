"""收到明确开始指令后，只读采集操作员设置的参考位。此处只做准入及参考位获取，不试运行、不作实机验收。
导入模块不会打开连接。
运行函数只打开独立的 Primary 和只读 RTDE 通道，检查机型及访客身份后关闭两者。
不上传程序、不修改安装配置、不创建输入配方；动作仍由原有的显式启动流程管理。"""
import copy
import ipaddress
import math
import time

from .physical_errors import SessionCleanupError
from .physical_robot_model import SUPPORTED_MODELS
from .reference_gate import rotation_vector_distance


_FRESHNESS = .15
_DWELL = .6
_STARTUP_TIMEOUT = 8.
_PROOF_TIMEOUT = 5.
_GUEST_INTERVAL = .05


def _dependencies():
    from .horizontal_follow import StateReceiver, check_dashboard
    from .ur_rtde_transport import RTDEChannel
    return dict(dashboard_check=check_dashboard, state_factory=StateReceiver,
                rtde_factory=RTDEChannel, clock=time.monotonic, sleep=time.sleep)


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Invalid finite ' + name)
    return value


def _vector(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ValueError('Invalid six-vector ' + name)
    return tuple(_number(x, name) for x in value)


def _identity(control):
    if (not isinstance(control, dict) or not isinstance(control.get('instance'), str)
            or not 1 <= len(control['instance']) <= 128
            or type(control.get('epoch')) is not int or not 0 <= control['epoch'] < 2**53
            or type(control.get('started_ns')) is not int or control['started_ns'] <= 0):
        raise ValueError('Explicit current guest control identity required')
    return {name: control[name] for name in ('instance', 'epoch', 'started_ns')}


def _guest(value, expected):
    if (not isinstance(value, dict)
            or any(type(value.get(k)) is not type(v) or value.get(k) != v
                   for k, v in expected.items())
            or any(value.get(k) is not True for k in
                   ('active', 'control_fresh', 'camera_connected', 'camera_fresh'))
            or ('stale' in value and value['stale'] is not False)):
        # 错误信息不包含会话标识、主机地址或原始状态快照。
        raise RuntimeError('Guest control changed, inactive or camera/control not fresh')


def _retire(resources):
    errors = []
    for resource in reversed(resources):
        try:
            resource.close()
        except Exception as exc:
            errors.append(exc)
        # close 方法若只等待有限时间，返回时工作线程可能尚未退出。此时不能接纳参考位。
        for name in ('thread', 'reader', 'writer'):
            worker = getattr(resource, name, None)
            if callable(getattr(worker, 'join', None)) and callable(getattr(worker, 'is_alive', None)):
                try:
                    worker.join(timeout=1.)
                    if worker.is_alive():
                        raise RuntimeError('Selected reference read-only thread still alive')
                except Exception as exc:
                    errors.append(exc)
        process = getattr(resource, 'process', None)
        if callable(getattr(process, 'poll', None)):
            try:
                if process.poll() is None:
                    raise RuntimeError('Selected reference read-only process still alive')
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise SessionCleanupError('Selected reference read-only cleanup incomplete') from errors[0]


def capture_selected_reference(profile, *, reader, control, should_stop, dependencies=None,
                               sample_observer=None, proof_observer=None):
    """在不移动机器人的情况下，返回一份最新、静止的 (Primary, RTDE) 参考状态。
    reader 读取当前访客快照，不续租；should_stop 检查服务、数据源和租约的取消标志。
    隔离仿真和单元测试可注入替身，传入替身时不构建默认设备对象。
    原生连接构造共用八秒准入期限，RTDE 只使用剩余时间。
    之后在五秒内，须观测到至少 600 毫秒连续、新鲜且一致的静止数据。
    整个流程不重连、不运动、不更改配置，也不自动恢复。
    可选的无参 before_final_snapshot 在初次验证后、两条接收通道仍打开时检查 Docker 身份；
    回调结束后须在原五秒期限内重新累计完整的 600 毫秒。
    sample_observer 可接收每个通过校验且时钟确实前进的样本副本，包括非静止样本；
    保留原始较早接收时间，不得改变准入参考位。可能阻塞的回调前后均检查访客身份与样本时效。
    proof_observer 可在完整本地验证后使用另一个未传给 sample_observer 的新样本。
    回调返回后仍须在原截止时间内重建静止验证；显式返回 False 表示远端尚未验证，应继续采样，重新累计后再尝试。
    其他返回值保留原先只调用一次的行为。"""
    expected = _identity(control)
    if not callable(reader) or not callable(should_stop):
        raise ValueError('Guest reader and stop observer required')
    if sample_observer is not None and not callable(sample_observer):
        raise ValueError('Sample observer must be callable')
    if proof_observer is not None and not callable(proof_observer):
        raise ValueError('Proof observer must be callable')
    host = getattr(profile, 'host', None)
    try:
        address = ipaddress.ip_address(host)
    except (TypeError, ValueError):
        raise ValueError('Explicit selected controller IP required') from None
    if address.is_multicast or address.is_unspecified:
        raise ValueError('Explicit selected controller IP required')
    model = getattr(getattr(profile, 'cfg', None), 'robot_model', None)
    if model not in SUPPORTED_MODELS:
        raise ValueError('Supported selected robot model required')
    supplied = {} if dependencies is None else dependencies
    if not isinstance(supplied, dict):
        raise ValueError('Capture dependencies must be a mapping')
    before_final_snapshot = supplied.get('before_final_snapshot')
    if 'before_final_snapshot' in supplied and not callable(before_final_snapshot):
        raise ValueError('Final snapshot callback must be callable')
    # 忽略共享仿真依赖表中的无关项。采集只使用这五项，不调用创建命令或控制门控的工厂函数。
    factories = _dependencies()
    for name in tuple(factories):
        if name in supplied: factories[name] = supplied[name]
    if any(not callable(value) for value in factories.values()):
        raise ValueError('Callable capture dependencies required')
    clock, sleep = factories['clock'], factories['sleep']
    previous_clock = None

    def now():
        nonlocal previous_clock
        current = _number(clock(), 'capture clock')
        if current < 0 or (previous_clock is not None and current < previous_clock):
            raise RuntimeError('Selected reference capture clock moved backwards')
        previous_clock = current
        return current

    last_guest = None

    def guard(*, force=False):
        nonlocal last_guest
        if should_stop(): raise RuntimeError('Selected reference capture stopped')
        current = now()
        if force or last_guest is None or current - last_guest >= _GUEST_INTERVAL:
            value = reader()
            if should_stop(): raise RuntimeError('Selected reference capture stopped')
            _guest(value, expected)
            last_guest = now()
        return now()

    guard(force=True)
    startup_deadline = now() + _STARTUP_TIMEOUT

    def startup_guard():
        current = guard(force=True)
        if current >= startup_deadline:
            raise RuntimeError('Selected reference startup timed out')
        return current

    resources = []
    candidate = candidate_stamps = None
    try:
        startup_guard()
        factories['dashboard_check'](host, idle=True, robot_model=model)
        startup_guard()
        states = factories['state_factory'](host)
        resources.append(states)
        startup_guard()
        feedback = factories['rtde_factory'](host, spatial=True, receive_only=True,
            startup_timeout=startup_deadline - now())
        resources.append(feedback)
        startup_guard()
        deadline = now() + _PROOF_TIMEOUT
        previous = None
        initial_tcp = None
        since = last_accepted = baseline = None
        count = 0
        final_check_done = False
        proof_check_done = proof_pending = False
        observed_stamps = None

        def snapshot():
            nonlocal initial_tcp
            if states.failure or feedback.failure:
                raise RuntimeError('Selected reference read-only telemetry failed')
            primary, primary_at = states.latest
            native, native_at = feedback.latest
            current = now()
            if primary is None or native is None:
                return None
            if not isinstance(primary, dict) or not isinstance(native, dict):
                raise ValueError('Invalid selected reference telemetry mapping')
            for stamp in (primary_at, native_at):
                _number(stamp, 'reference receipt clock')
                if not 0 <= current - stamp <= _FRESHNESS:
                    raise RuntimeError('Selected reference telemetry receipt stale or not fresh')
            mode = primary.get('mode')
            if (not isinstance(mode, dict) or type(mode.get('mode')) is not int or mode['mode'] != 7
                    or any(mode.get(k) is not True for k in ('connected', 'enabled', 'power'))
                    or any(mode.get(k) is not False for k in
                           ('emergency', 'protective', 'paused', 'running'))):
                raise RuntimeError('Selected reference requires ready idle native mode')
            robot_primary = _number(mode.get('timestamp'), 'Primary producer clock')
            robot_native = _number(native.get('timestamp'), 'RTDE producer clock')
            if min(robot_primary, robot_native) < 0:
                raise ValueError('Invalid selected reference producer clock')
            primary_pose = _vector(primary.get('pose'), 'Primary pose')
            pose = _vector(native.get('tcp_pose'), 'RTDE pose')
            joints = _vector(native.get('joints'), 'joint vector')
            tcp = _vector(primary.get('tcp_offset'), 'active TCP')
            joint_speed = _vector(native.get('joint_speed'), 'joint speed')
            tcp_speed = _vector(native.get('tcp_speed'), 'TCP speed')
            if initial_tcp is None: initial_tcp = tcp
            elif tcp != initial_tcp:
                raise RuntimeError('Selected reference active TCP configuration changed')
            stationary = (max(map(abs, joint_speed)) < .003
                and math.hypot(*tcp_speed[:3]) < .0005 and math.hypot(*tcp_speed[3:]) < .002)
            agreeing = (math.dist(primary_pose[:3], pose[:3]) <= .0005
                and rotation_vector_distance(primary_pose[3:], pose[3:]) <= .005)
            return (primary, native, (primary_at, native_at, robot_primary, robot_native),
                    current, stationary and agreeing, (joints, pose))

        def observe(primary, native, stamps):
            nonlocal observed_stamps
            if sample_observer is None:
                return
            if observed_stamps is not None:
                if any(a < b for a, b in zip(stamps, observed_stamps)):
                    raise RuntimeError('Selected reference observer clock moved backwards')
                if not all(a > b for a, b in zip(stamps, observed_stamps)):
                    return
            current = guard(force=True)
            if any(not 0 <= current - stamp <= _FRESHNESS for stamp in stamps[:2]):
                raise RuntimeError('Selected reference sample expired before observer')
            sample_observer(copy.deepcopy(primary), copy.deepcopy(native), min(stamps[:2]))
            current = guard(force=True)
            if any(not 0 <= current - stamp <= _FRESHNESS for stamp in stamps[:2]):
                raise RuntimeError('Selected reference sample expired during observer')
            if current >= deadline:
                raise RuntimeError('Selected reference observer proof timed out')
            observed_stamps = stamps

        def same_position(left, right):
            return (max(abs(a-b) for a, b in zip(left[0], right[0])) < .003
                and math.dist(left[1][:3], right[1][:3]) < .0005
                and rotation_vector_distance(left[1][3:], right[1][3:]) < .005)

        while guard() < deadline:
            current = snapshot()
            if current is None:
                since = last_accepted = baseline = None; count = 0
                sleep(.01)
                continue
            primary, native, stamps, sampled, stationary, position = current
            if previous is not None and any(a < b for a, b in zip(stamps, previous)):
                raise RuntimeError('Selected reference producer or receipt clock stopped advancing')
            advancing = previous is None or all(a > b for a, b in zip(stamps, previous))
            if not stationary:
                since = last_accepted = baseline = None; count = 0
            if advancing:
                gap = last_accepted is not None and sampled - last_accepted > _FRESHNESS
                moved = baseline is not None and not same_position(baseline, position)
                previous = stamps
                if proof_pending:
                    proof_pending = False
                    if stationary and not gap and not moved:
                        # 为原子验证回调保留下一个新样本。若先传给 sample_observer，回调采样或放行时会重用同一时间戳。
                        guard(force=True)
                        fresh = snapshot()
                        if fresh is not None:
                            a, b, fresh_stamps, fresh_at, fresh_still, fresh_position = fresh
                            if any(x < y for x, y in zip(fresh_stamps, stamps)):
                                raise RuntimeError('Selected reference proof clock moved backwards')
                            if (fresh_still and fresh_at - last_accepted <= _FRESHNESS
                                    and same_position(baseline, fresh_position)):
                                current = now()
                                if current >= deadline:
                                    raise RuntimeError('Selected reference observer proof timed out')
                                if any(not 0 <= current - stamp <= _FRESHNESS for stamp in fresh_stamps[:2]):
                                    raise RuntimeError('Selected reference sample expired before proof observer')
                                result=proof_observer(copy.deepcopy(a), copy.deepcopy(b), min(fresh_stamps[:2]))
                                proof_check_done = result is not False
                                guard(force=True)
                                previous = fresh_stamps
                                since = last_accepted = baseline = None; count = 0
                                continue
                            observe(a, b, fresh_stamps)
                        since = last_accepted = baseline = None; count = 0
                        continue
                observe(primary, native, stamps)
                if stationary:
                    if since is None or gap or moved:
                        since = sampled; count = 0; baseline = position
                    count += 1; last_accepted = sampled
                    if count >= 3 and sampled - since >= _DWELL:
                        if before_final_snapshot is not None and not final_check_done:
                            guard(force=True)
                            # 在调用前先做标记；回调出错时终止本次采集，不重复身份检查。
                            final_check_done = True
                            before_final_snapshot()
                            guard(force=True)
                            # 保留 initial_tcp 和此前的生产端时钟。TCP 变化或时钟回退仍判为失败；
                            # 只重置静止计时，不重置总截止时间。
                            since = last_accepted = baseline = None; count = 0
                            continue
                        if proof_observer is not None and not proof_check_done:
                            proof_pending = True
                            continue
                        # 最后一次访客检查可能阻塞，结束后须重新读取原生状态，避免返回已经过时的快照。
                        guard(force=True)
                        final = snapshot()
                        if final is not None:
                            a, b, final_stamps, final_at, final_still, final_position = final
                            observe(a, b, final_stamps)
                            if (final_still and final_at - last_accepted <= _FRESHNESS
                                    and all(x >= y for x, y in zip(final_stamps, stamps))
                                    and same_position(baseline, final_position)):
                                candidate = copy.deepcopy((a, b))
                                candidate_stamps = final_stamps[:2]
                                break
                        since = last_accepted = baseline = None; count = 0
            elif last_accepted is not None and sampled - last_accepted > _FRESHNESS:
                since = last_accepted = baseline = None; count = 0
            sleep(.01)
        if candidate is None:
            raise RuntimeError('Fresh advancing stationary selected reference timed out')
    finally:
        _retire(resources)
    if should_stop(): raise RuntimeError('Selected reference capture stopped during cleanup')
    completed_at = now()
    if any(not 0 <= completed_at - stamp <= _FRESHNESS for stamp in candidate_stamps):
        raise RuntimeError('Selected reference expired during read-only cleanup')
    return candidate
