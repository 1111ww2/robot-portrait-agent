"""在相同控制租约内恢复本地参考位 RPC，不操作机器人或运动传输通道。"""
import copy
import math
import threading
import time
import uuid

from .physical_interaction_supervisor import LocalReferenceGate, ReferenceRequestError


class ReferenceSampleInterrupted(RuntimeError):
    """丢弃状态不确定的旧 TCP 样本，重新获取实测依据。"""


class ReferencePhotoInterrupted(RuntimeError):
    """取消旧的拍摄准备，必须重新到位才能发起下一次拍摄。"""


class ReferenceRecoveryAccounting:
    """精确记录单个 RPC 工作线程中已分类的失败尝试时间段。"""
    def __init__(self):
        self._lock = threading.Lock()
        self._intervals = []

    def record_failure(self, started, finished):
        if (any(type(value) not in (int, float) or not math.isfinite(value)
                for value in (started, finished)) or finished < started):
            raise RuntimeError('Reference recovery accounting clock invalid')
        with self._lock:
            self._intervals.append((started, finished))

    @property
    def intervals(self):
        with self._lock:
            return tuple(self._intervals)


class RecoveryReferenceGate(LocalReferenceGate):
    def __init__(self, port, token, *, timeout, control, lease, recovery=None):
        super().__init__(port, token, timeout=timeout, control=control, lease=lease)
        self.recovery = recovery
        self._worker_accounting = threading.local()

    def begin_worker_accounting(self):
        accounting = ReferenceRecoveryAccounting()
        self._worker_accounting.current = accounting
        return accounting

    def end_worker_accounting(self, accounting):
        if getattr(self._worker_accounting, 'current', None) is not accounting:
            raise RuntimeError('Reference recovery worker identity changed')
        self._worker_accounting.current = None

    def _recover_request(self, send, *, retry, sample=False, photo=False, defer=False):
        while True:
            if self.recovery is not None:
                self.recovery.status  # 截止时间固定，重试前同样要检查。
                if photo and self.recovery.paused:
                    raise ReferencePhotoInterrupted()
                while defer and self.recovery.paused:
                    time.sleep(.02)
            attempt_started = self.recovery.clock() if self.recovery is not None else None
            try:
                result = send()
                if self.recovery is not None:
                    self.recovery.status  # 迟到的成功响应不能恢复已过期的交互会话。
                return result
            except ReferenceRequestError as error:
                if self.recovery is None:
                    raise
                attempt_finished = self.recovery.clock()
                if (sample and error.reference_reason == 'stale_sample'
                        and self.recovery.paused):
                    raise ReferenceSampleInterrupted() from None
                if not self.recovery.transport_error(error):
                    raise
                accounting = getattr(self._worker_accounting, 'current', None)
                if accounting is not None:
                    accounting.record_failure(attempt_started, attempt_finished)
                if sample:
                    raise ReferenceSampleInterrupted() from None
                if photo:
                    raise ReferencePhotoInterrupted() from None
                if not retry:
                    raise
                time.sleep(.02)

    def _request(self, method, path, body=None):
        base = super()._request
        payload = copy.deepcopy(body)
        op = payload.get('op') if isinstance(payload, dict) else None
        deduplicated = method == 'POST' and ((path == '/reference-gate' and op in ('begin', 'release', 'restart_begin', 'restart_release'))
                        or (path == '/physical-feedback' and op == 'register'))
        retry = ((method == 'GET' and path == '/operator-state') or deduplicated
                 or (method == 'POST' and path == '/physical-feedback' and op in ('heartbeat', 'intro', 'resume')))
        sample = method == 'POST' and path == '/reference-gate' and op in ('sample', 'restart_sample')
        photo = method == 'POST' and path == '/physical-feedback' and op == 'photo_ready'
        if self.recovery is None or not (retry or sample or photo):
            return base(method, path, payload)
        if deduplicated:
            payload['request_id'] = uuid.uuid4().hex
        return self._recover_request(lambda: base(method, path, payload), retry=retry, sample=sample,
            photo=photo, defer=path == '/reference-gate' and op in ('begin', 'release', 'restart_begin', 'restart_release'))

    def photo_source(self):
        if self.recovery is None:
            return super().photo_source()
        # 准备阶段的 GET 没有副作用，网络中断后也不能根据它的响应把旧到位状态当成新的拍照许可。
        state = self._recover_request(
            lambda: LocalReferenceGate._request(self, 'GET', '/operator-state'), retry=False, photo=True)
        if (not isinstance(state.get('control'), dict)
                or state.get('control') != self._identity
                or state.get('control_started_ns') != self.started_ns or state.get('active') is not True):
            raise RuntimeError('Physical photo session unavailable')
        snapshot = {**state['control'], 'started_ns': state['control_started_ns'],
            **{key: state.get(key) for key in ('active', 'control_fresh', 'camera_connected', 'camera_fresh')}}
        if 'communication' in state:
            snapshot['communication'] = state['communication']
        self.recovery.observe(snapshot)
        if self.recovery.paused:
            raise ReferencePhotoInterrupted()
        if not isinstance(state.get('physical'), dict) or state['physical'].get('backend_available') is not True:
            raise RuntimeError('Physical photo session unavailable')
        return state['physical'].get('photo_source_ns')

    def feedback_request(self, body):
        base = super().feedback_request
        payload = copy.deepcopy(body)
        if self.recovery is None or payload.get('op') not in ('heartbeat', 'intro', 'resume'):
            return base(payload)
        return self._recover_request(lambda: base(payload), retry=True)

    def presentation_transport(self):
        # 界面离线时不建立机器人控制命令的重试队列。
        return LocalReferenceGate(self._port, self._token, timeout=self._timeout,
            control={**self._identity, 'started_ns': self.started_ns}, lease=self._lease)

    def control_transport(self, *, timeout):
        return RecoveryReferenceGate(self._port, self._token, timeout=timeout,
            control={**self._identity, 'started_ns': self.started_ns}, lease=self._lease,
            recovery=self.recovery)

    def resume_communication(self):
        return self._request('POST', '/physical-feedback', {'op': 'resume'}).get('ok') is True
