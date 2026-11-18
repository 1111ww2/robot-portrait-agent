"""纯图像到达策略；禁止使用机器人、摄像头、时钟或网络访问。
呼叫者必须在测量返回和静止后从*实际*参考观察门获取“release_ns”。在此处传递的时间戳不证明物理安全。
所有时间戳都必须使用与门相同的单调时钟，并且必须描述捕获，而不是服务器接收。
呼叫者提供现有跟踪器的完整到达决策作为“跟踪”和“在内”。
在每次运动之前，呼叫者必须开始_运动，在参考门释放后才释放_观察。每张新图像观察一次，并在图像缺失时过期。
一旦锁定，离开的访问者不会撤销照片。呼叫者仍然必须获取真实的摄像头图像并处理捕获失败。
停止或丢失控制会话必须呼叫stop/disconnect；该策略不命令或停止任何电机本身。"""


class PhysicalPortraitPolicy:
    """在一个明确封锁的会话中跟踪连续两秒的到达。"""

    DWELL_NS = 2_000_000_000
    MAX_IMAGE_AGE_NS = 300_000_000

    def __init__(self):
        self.active = False
        self.latched = False
        self.blocked = True
        self._epoch = None
        self._last_epoch = -1
        self._release_ns = 0
        self._max_seen_ns = 0
        self._first_ns = None
        self._last_ns = None

    @staticmethod
    def _valid_int(value):
        return type(value) is int and value > 0

    def _current(self, epoch):
        return self.active and type(epoch) is int and epoch == self._epoch

    def _reset_dwell(self):
        self._first_ns = None
        self._last_ns = None

    def start(self, epoch):
        if not self._valid_int(epoch):
            raise ValueError("epoch must be a positive integer")
        if self.active:
            if epoch == self._epoch:
                return
            raise ValueError("stop the current session before starting another")
        if epoch <= self._last_epoch:
            raise ValueError("session epoch must increase")
        self.active, self.latched, self.blocked = True, False, True
        self._epoch = self._last_epoch = epoch
        self._release_ns = 0
        self._max_seen_ns = 0
        self._reset_dwell()

    def begin_motion(self, epoch):
        if not self._current(epoch) or self.latched:
            return False
        self.blocked = True
        self._reset_dwell()
        return True

    def release_observation(self, epoch, release_ns):
        if not self._current(epoch) or self.latched or not self.blocked:
            raise ValueError("observation cannot be released in this state")
        if (not self._valid_int(release_ns) or release_ns <= self._release_ns
                or release_ns < self._max_seen_ns):
            raise ValueError("release timestamp must increase")
        self._release_ns = release_ns
        self.blocked = False
        self._reset_dwell()

    def observe(self, epoch, *, source_ns, now_ns, tracking, within):
        if not self._current(epoch):
            return False
        if self.latched:
            return True
        if self.blocked:
            return False
        if (not self._valid_int(source_ns) or not self._valid_int(now_ns)
                or type(tracking) is not bool or type(within) is not bool
                or source_ns <= self._release_ns or source_ns > now_ns
                or now_ns - source_ns > self.MAX_IMAGE_AGE_NS):
            self._reset_dwell()
            return False
        if source_ns <= self._max_seen_ns:
            if source_ns < self._max_seen_ns:
                self._reset_dwell()
            return False
        self._max_seen_ns = source_ns
        if not tracking or not within:
            self._reset_dwell()
            return False
        if self._last_ns is not None:
            if source_ns - self._last_ns > self.MAX_IMAGE_AGE_NS:
                self._reset_dwell()
        if self._first_ns is None:
            self._first_ns = source_ns
        self._last_ns = source_ns
        if source_ns - self._first_ns >= self.DWELL_NS:
            self.latched = True
        return self.latched

    def expire(self, epoch, now_ns):
        if not self._current(epoch) or self.latched:
            return
        if (not self._valid_int(now_ns) or self._last_ns is not None
                and (now_ns < self._last_ns or now_ns - self._last_ns > self.MAX_IMAGE_AGE_NS)):
            self._reset_dwell()

    def stop(self, epoch):
        if not self._current(epoch):
            return False
        self.active, self.latched, self.blocked = False, False, True
        self._reset_dwell()
        return True

    def disconnect(self, epoch):
        return self.stop(epoch)
