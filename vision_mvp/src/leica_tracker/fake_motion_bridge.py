"""在离线测试中连接语义意图与监管逻辑。此模块不含机器人通信、轨迹、网络输入输出或正式入口。
完成与 HOLD 均由测试方提供，不表示机器人确认或实际停止，不能据此授权或指挥实体机器人。"""
import threading

from .offline_portrait_supervisor import OfflinePortraitSupervisor


class FakeMotionTransport:
    """可重复的模拟后端，只记录指令，不产生机器人动作。"""

    def __init__(self):
        self.calls = []
        self._active = None
        self._result = None
        self._cancelled = set()
        self._lock = threading.Lock()

    def start(self, cue, generation):
        with self._lock:
            if generation in self._cancelled:
                raise RuntimeError('Fake motion was cancelled before start')
            if self._active is not None:
                raise RuntimeError('Fake motion already active')
            self.calls.append(('start', generation, cue))
            self._active = generation
            self._result = None

    def complete(self, generation, *, success=True):
        with self._lock:
            if self._active != generation or type(success) is not bool:
                raise RuntimeError('Unknown fake motion')
            self._result = 'completed' if success else 'failed'

    def poll(self, generation):
        with self._lock:
            if self._active != generation:
                raise RuntimeError('Unknown fake motion')
            if self._result is None:
                return 'pending'
            result = self._result
            self._active = self._result = None
            return result

    def stop(self, generation):
        with self._lock:
            self._cancelled.add(generation)
            self.calls.append(('stop', generation))
            if self._active == generation:
                self._active = self._result = None


class FakeMotionBridge:
    """通过模拟后端执行一个语义意图，保留已经锁定的状态。"""

    def __init__(self, supervisor, transport):
        if type(supervisor) is not OfflinePortraitSupervisor:
            raise TypeError('An offline supervisor is required')
        # 严格检查类型，避免意外接入带网络通信能力的子类。
        if type(transport) is not FakeMotionTransport:
            raise TypeError('Only the explicitly fake transport is supported')
        self._supervisor = supervisor
        self._transport = transport
        self._generation = 0
        self._active = None
        self._closed = False
        self._lock = threading.Lock()

    def _abort(self):
        with self._lock:
            self._closed = True
            active = self._active
            self._active = None
        if active is not None:
            self._transport.stop(active)

    def dispatch(self, *, now):
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError('Fake bridge stopped or faulted')
                if self._active is not None:
                    raise RuntimeError('Fake action already active')
                if self._supervisor.state != 'INTENT_READY':
                    raise ValueError('Observation HOLD has not been acknowledged')
                cue = self._supervisor.intent(now=now).cue
                self._generation += 1
                generation = self._generation
                self._active = generation
            self._transport.start(cue, generation)
            with self._lock:
                if self._closed:
                    raise RuntimeError('Fake motion stopped during start')
            return generation
        except ValueError:
            raise
        except Exception:
            with self._lock:
                self._supervisor.disconnect()
            self._abort()
            raise

    def poll(self, *, now):
        try:
            with self._lock:
                if self._closed or self._active is None:
                    raise RuntimeError('No active fake action')
                self._supervisor.watchdog(now=now)
                generation = self._active
            result = self._transport.poll(generation)
            with self._lock:
                if self._closed or self._active != generation:
                    raise RuntimeError('Fake action stopped while polling')
                if result == 'pending':
                    return result
                self._active = None
                self._supervisor.motion_finished(success=result == 'completed', now=now)
                if result == 'failed':
                    self._closed = True
                return result
        except Exception:
            with self._lock:
                self._supervisor.disconnect()
            self._abort()
            raise

    def stop(self):
        with self._lock:
            self._closed = True
            active = self._active
            self._active = None
            self._supervisor.stop()
        if active is not None:
            self._transport.stop(active)
