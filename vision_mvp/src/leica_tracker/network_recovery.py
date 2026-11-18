"""在同一次交互内限时等待通信恢复，不重连机器人。策略只批准等待；
动作程序必须独立观测到原生暂停确认和静止反馈后才能调用 complete。
机器人反馈故障、确认超时、停止指令和控制器故障均不能按通信中断处理。"""
import math
import threading
import time


RECOVERY_WINDOW_S = 3.0
RECOVERY_STABLE_S = .3


class CommunicationRecovery:
    def __init__(self, control, *, clock=time.monotonic):
        if (not isinstance(control, dict) or not isinstance(control.get('instance'), str)
                or not 1 <= len(control['instance']) <= 128
                or type(control.get('epoch')) is not int or not 0 <= control['epoch'] < 2**53
                or type(control.get('started_ns')) is not int or control['started_ns'] <= 0
                or not callable(clock)):
            raise ValueError('Exact active encounter identity and clock required')
        self.control = {key: control[key] for key in ('instance', 'epoch', 'started_ns')}
        self.clock = clock
        self._lock = threading.RLock()
        self._status = 'healthy'
        self._deadline = self._stable_since = self._last_clock = None
        self.failure = None

    def _fail(self, reason):
        if self.failure is None:
            self.failure = str(reason)
        self._status = 'failed'
        raise RuntimeError(self.failure)

    def fail(self, reason):
        with self._lock:
            self._fail(reason)

    def _now(self):
        if self.failure is not None:
            self._fail(self.failure)
        now = self.clock()
        if (type(now) not in (int, float) or not math.isfinite(now)
                or self._last_clock is not None and now < self._last_clock):
            self._fail('Communication recovery clock invalid')
        self._last_clock = now
        if self._deadline is not None and now >= self._deadline:
            self._fail('Communication recovery window expired')
        return now

    def _wait(self, now):
        if self._deadline is None:
            self._deadline = now + RECOVERY_WINDOW_S
        self._status = 'waiting'
        self._stable_since = None

    def observe(self, snapshot):
        with self._lock:
            now = self._now()
            if not isinstance(snapshot, dict):
                self._fail('Communication snapshot schema invalid')
            if (any(type(snapshot.get(key)) is not type(value) or snapshot.get(key) != value
                    for key, value in self.control.items())):
                self._fail('Communication encounter identity changed')
            if type(snapshot.get('active')) is not bool:
                self._fail('Communication active flag invalid')
            if not snapshot['active'] or snapshot.get('control_action') == 'stop':
                self._fail('Guest Stop or inactive encounter')
            if ('control_action' in snapshot
                    and snapshot['control_action'] not in ('start', 'restart', 'stop')):
                self._fail('Communication action invalid')
            fields = ('control_fresh', 'camera_connected', 'camera_fresh')
            if any(type(snapshot.get(key)) is not bool for key in fields):
                self._fail('Communication freshness schema invalid')
            if 'stale' in snapshot and type(snapshot['stale']) is not bool:
                self._fail('Communication stale flag invalid')
            server_waiting = False
            if 'communication' in snapshot:
                metadata = snapshot['communication']
                if (not isinstance(metadata, dict)
                        or metadata.get('status') not in ('healthy', 'waiting', 'ready')
                        or type(metadata.get('recoverable')) is not bool
                        or type(metadata.get('remaining_ms')) not in (int, float)
                        or not math.isfinite(metadata['remaining_ms']) or metadata['remaining_ms'] < 0):
                    self._fail('Communication recovery metadata invalid')
                if not metadata['recoverable']:
                    self._fail('Communication recovery denied')
                server_waiting = metadata['status'] == 'waiting'
                if metadata['status'] != 'healthy' and self._deadline is None:
                    # 此运行实例启动前，所有者可能已经进入恢复流程。服务端报告就绪也不能跳过本地连续正常状态的等待时间。
                    self._wait(now)
                    self._deadline = min(self._deadline, now + metadata['remaining_ms'] / 1000)
                    if now >= self._deadline:
                        self._fail('Communication recovery window expired')
            if (not all(snapshot[key] for key in fields)
                    or snapshot.get('stale') is True or server_waiting):
                self._wait(now)
            elif self._deadline is not None:
                if self._stable_since is None:
                    self._stable_since = now
                if now - self._stable_since >= RECOVERY_STABLE_S:
                    self._status = 'ready'
            return self._status

    def transport_error(self, error):
        # 延迟导入以避免和看门狗可选策略之间形成循环依赖。
        from .resident_guest_watchdog import GuestSnapshotRequestError
        from .physical_interaction_supervisor import ReferenceRequestError
        with self._lock:
            if self.failure is not None:
                return False
            try:
                now = self._now()
                if (type(error) not in (GuestSnapshotRequestError, ReferenceRequestError)
                        or error.category not in ('timeout', 'connection')
                        or error.status is not None):
                    self._fail('Communication transport failure is terminal')
                self._wait(now)
                return True
            except RuntimeError:
                return False

    def synchronized(self):
        """将看门狗暂停、动作程序恢复与策略更新串行处理。"""
        return self._lock

    def complete(self, *, resume=None):
        """仅允许运行程序在连续正常等待期结束、收到原生暂停确认且设备静止后调用。"""
        with self._lock:
            self._now()
            if self._status != 'ready':
                raise RuntimeError('Communication recovery is not ready')
            if resume is not None:
                if not callable(resume):
                    raise ValueError('Native resume callback required')
                resume()
            self._status = 'healthy'
            self._deadline = self._stable_since = None

    @property
    def status(self):
        with self._lock:
            self._now()
            return self._status

    @property
    def paused(self):
        return self.status in ('waiting', 'ready')

    @property
    def ready(self):
        return self.status == 'ready'

    @property
    def deadline(self):
        with self._lock:
            return self._deadline
