"""离线协调原型；不连接或命令机器人。
呼叫者必须提供经过验证的及时物理测距、控制心跳、每个心跳的明确主动所有者证据，以及实际观察-HOLD确认。
此类不能验证这些证据。HOLD和动作超时由呼叫者提供，并需要守护程序呼叫。
输出仅为语义标签，不是目标或机器人命令。这不是紧急停止、可执行的物理适配器或调试/安全批准。
ReferenceGate的姿势检查尚未在物理安装上验证；此组件不得用于授权运动。
所有时间，包括帧捕获时间戳，都必须使用相同的可信单调时钟。呼叫是单线程的：未来的监督员必须对其进行序列化。
此类不验证时间戳或验证帧内容。"""
from dataclasses import dataclass
import math

from .physical_control_guard import ControlGuard
from .reference_gate import ReferenceGate, rotation_vector_distance


_DIRECTIONS = frozenset({'MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP', 'MOVE_DOWN',
                         'MOVE_CLOSER', 'MOVE_FARTHER'})


def _six(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 6 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError('Six finite telemetry values required')
    return tuple(float(v) for v in value)


def _fresh(sampled_at, now, limit):
    return (type(sampled_at) in (int, float) and type(now) in (int, float)
            and math.isfinite(sampled_at) and math.isfinite(now)
            and 0 <= now - sampled_at <= limit)


@dataclass(frozen=True)
class SemanticIntent:
    """仅限描述性线索；它不包含物理轨迹或目标。"""
    cue: str


class PhysicalInteractionCoordinator:
    """用于注射式单次实机交互的原型门控。一旦停止或出现故障，此实例永远不会自动重置或返回机器人。
    此类别的输出无法单独移动机器人。"""
    def __init__(self, *, hold_timeout, motion_timeout):
        for timeout in (hold_timeout, motion_timeout):
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError('Explicit positive finite timeouts required')
        self._hold_timeout = hold_timeout
        self._motion_timeout = motion_timeout
        self._deadline = None
        self._owner = None
        self.state = 'NEW'
        self.failure = None
        self._gate = ReferenceGate()
        self._control = ControlGuard()
        self._reference = None
        self._session = None
        self._pending_cue = None
        self._last_robot_sample = None
        self._last_frame = None
        self._motion_finished_at = None

    def _fail(self, reason):
        self.state = 'FAULT'
        self.failure = reason

    def _check_deadline(self, now):
        if self._deadline is not None and now > self._deadline:
            raise RuntimeError('HOLD or action timed out; no automatic retry')

    def _guard(self, now):
        if self.state in ('NEW', 'FAULT', 'STOPPED'):
            raise RuntimeError('Encounter not active')
        try:
            if type(now) not in (int, float) or not math.isfinite(now):
                raise RuntimeError('Invalid monotonic time')
            self._control.check(now)
            self._check_deadline(now)
            if not _fresh(self._last_robot_sample, now, .15):
                raise RuntimeError('Robot telemetry unavailable or stale')
        except RuntimeError as exc:
            self._fail(str(exc))
            raise

    def _begin(self, cue, now):
        deadline = now + self._hold_timeout
        if not math.isfinite(deadline):
            raise ValueError('Invalid HOLD deadline')
        self._session = self._gate.begin(self._reference, now)
        self._pending_cue = cue
        self._last_frame = None
        self._motion_finished_at = None
        self._deadline = deadline
        self.state = 'HOLD_REQUIRED'

    @staticmethod
    def _active_owner(active_owner, control):
        if (not isinstance(active_owner, dict) or active_owner.get('active') is not True
                or not isinstance(active_owner.get('owner'), str)
                or not active_owner['owner']
                or type(active_owner.get('epoch')) is not int
                or not isinstance(control, dict)
                or type(control.get('epoch')) is not int
                or active_owner['epoch'] != control['epoch']):
            raise RuntimeError('Explicit active owner evidence required')
        return active_owner['owner'], active_owner['epoch']

    def start(self, reference, speed, *, sampled_at, control, active_owner, now):
        if self.state != 'NEW':
            raise RuntimeError('Explicit new coordinator required')
        try:
            reference, speed = _six(reference), _six(speed)
            if (not _fresh(sampled_at, now, .15) or
                    math.hypot(*speed[:3]) >= .0005 or
                    math.hypot(*speed[3:]) >= .002):
                raise ValueError('Fresh stationary reference required')
            self._control.observe(control, now)
            self._owner = self._active_owner(active_owner, control)
            self._reference = reference
            self._last_robot_sample = sampled_at
            self._begin('OPENING', now)
        except (ValueError, RuntimeError) as exc:
            self._fail(str(exc))
            raise

    def observe_control(self, control, *, active_owner, now):
        if self.state in ('NEW', 'FAULT', 'STOPPED'):
            raise RuntimeError('Encounter not active')
        try:
            if type(now) not in (int, float) or not math.isfinite(now):
                raise RuntimeError('Invalid monotonic time')
            self._check_deadline(now)
            self._control.observe(control, now)
            if self._active_owner(active_owner, control) != self._owner:
                raise RuntimeError('Active owner changed; explicit restart required')
        except RuntimeError as exc:
            self._fail(str(exc))
            raise

    def watchdog(self, *, now):
        self._guard(now)
        if self.state == 'OBSERVING' and not _fresh(self._last_frame, now, .3):
            self._fail('Camera frame unavailable or stale')
            raise RuntimeError(self.failure)

    def acknowledge_hold(self, *, now):
        self._guard(now)
        if self.state != 'HOLD_REQUIRED':
            raise ValueError('No observation HOLD is awaiting acknowledgement')
        self._gate.acknowledge()
        self.state = 'INTENT_READY'

    def intent(self, *, now):
        self._guard(now)
        if self.state != 'INTENT_READY':
            raise ValueError('No acknowledged intent is available')
        deadline = now + self._motion_timeout
        if not math.isfinite(deadline):
            self._fail('Invalid motion deadline')
            raise RuntimeError(self.failure)
        self.state = 'MOVING'
        self._deadline = deadline
        return SemanticIntent(self._pending_cue)

    def motion_finished(self, *, success, now):
        self._guard(now)
        if self.state != 'MOVING':
            raise ValueError('No action is in progress')
        if type(success) is not bool:
            self._fail('Motion result unknown; no automatic retry')
            raise ValueError(self.failure)
        if not success:
            self._fail('Motion execution failed; no automatic retry')
            return
        # 预完成静止读数不能证明回转。
        self._gate.steady_since = None
        self._gate.last_sample = None
        self._motion_finished_at = now
        self._deadline = None
        self.state = 'SETTLING'

    def observe_telemetry(self, pose, speed, *, sampled_at, now):
        if self.state in ('NEW', 'FAULT', 'STOPPED'):
            raise RuntimeError('Encounter not active')
        try:
            if type(now) not in (int, float) or not math.isfinite(now):
                raise RuntimeError('Invalid monotonic time')
            self._check_deadline(now)
            self._control.check(now)
            pose, speed = _six(pose), _six(speed)
            if (not _fresh(sampled_at, now, .15) or
                    self._last_robot_sample is not None and sampled_at <= self._last_robot_sample):
                raise ValueError('Stale, repeated or invalid robot sample')
            if self.state == 'SETTLING' and sampled_at <= self._motion_finished_at:
                raise ValueError('Robot sample predates motion completion')
            if self._gate.blocked:
                self._gate.sample(self._session, pose, speed, sampled_at, now)
            elif (math.dist(pose[:3], self._reference[:3]) > .0005 or
                    rotation_vector_distance(pose[3:], self._reference[3:]) > .005 or
                    math.hypot(*speed[:3]) >= .0005 or math.hypot(*speed[3:]) >= .002):
                raise ValueError('Robot moved away from reference while observing')
            self._last_robot_sample = sampled_at
        except (ValueError, RuntimeError) as exc:
            self._fail(str(exc))
            raise

    def release(self, *, now):
        self._guard(now)
        if self.state != 'SETTLING':
            raise ValueError('Motion has not completed')
        self._gate.release(self._session, now)
        self._session = None
        self.state = 'WAIT_FRAME'
        # 该政策必须收到闸口的自己的释放时间，永远不会由其呼叫者猜测。这不是物理返回的证明。
        return self._gate.release_ns

    def accept_frame(self, source_ns, *, now):
        if self.state in ('NEW', 'FAULT', 'STOPPED') or self._gate.held(source_ns):
            return False
        self._guard(now)
        if (self.state not in ('WAIT_FRAME', 'OBSERVING') or type(source_ns) is not int
                or not _fresh(source_ns / 1e9, now, .3)
                or self._last_frame is not None and source_ns / 1e9 <= self._last_frame):
            return False
        self._last_frame = source_ns / 1e9
        self.state = 'OBSERVING'
        return True

    def request_direction(self, cue, *, now):
        self.watchdog(now=now)
        if self.state != 'OBSERVING' or not isinstance(cue, str) or cue not in _DIRECTIONS:
            raise ValueError('A fresh observation and a supported direction are required')
        self._begin(cue, now)

    def source_lost(self):
        """锁定明确检测到的摄像头、控制或遥测断开连接。"""
        if self.state not in ('FAULT', 'STOPPED'):
            self._fail('Input source disconnected; no automatic retry')

    def stop(self):
        # 这只是一道闩锁；它不会发送停止或其他机器人命令。
        self.state = 'STOPPED'
        self.failure = 'Stopped; explicit new coordinator required'
