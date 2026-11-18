"""离线停止请求/反馈合同。这不控制机器人。注入的传输必须提供一个受限的非阻塞的“停止（生成）”操作。
它必须在提交时返回None或True，并在失败时抛出或返回False。提交不会确认机器人的停止。
该模块不提供物理传输、硬件守护程序、已认证的数据、紧急停止或安全等级停止。
未来的监督员必须独立于相机/AI工作和序列化呼叫来安排“计时”。时间值和样本时间戳必须共享一个可信的单调时钟。
阈值是明确的呼叫者输入，在实际安装中不自动得到正当性。"""
import math


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _six(value):
    return (isinstance(value, (tuple, list)) and len(value) == 6
            and all(_number(item) for item in value))


class StopBridge:
    """锁定一个停止请求，并单独报告观察到的静止样品。"""

    def __init__(self, transport, *, generation, max_sample_age, max_tcp_linear,
                 max_tcp_angular, max_joint_speed, dwell, timeout):
        if type(generation) is not int or generation <= 0:
            raise ValueError('A positive motion generation is required')
        values = (max_sample_age, max_tcp_linear, max_tcp_angular,
                  max_joint_speed, dwell, timeout)
        if any(not _number(value) or value <= 0 for value in values):
            raise ValueError('Explicit finite positive limits are required')
        if dwell >= timeout:
            raise ValueError('Dwell must be shorter than stop timeout')
        self.transport = transport
        self.generation = generation
        self.max_sample_age = max_sample_age
        self.max_tcp_linear = max_tcp_linear
        self.max_tcp_angular = max_tcp_angular
        self.max_joint_speed = max_joint_speed
        self.dwell = dwell
        self.timeout = timeout
        self.state = 'IDLE'
        self.failure = None
        self.requested_at = None
        self.last_sample = None
        self.steady_since = None

    def _fail(self, reason):
        self.state = 'FAULT'
        self.failure = reason

    def request_stop(self, *, now):
        if self.state != 'IDLE':
            return self.state
        if not _number(now):
            raise ValueError('Invalid monotonic time')
        # 提交前锁定；即使例外情况也不会重新启用或重试。
        self.state = 'REQUESTED'
        self.requested_at = now
        try:
            submitted = self.transport.stop(self.generation)
            if submitted is not None and submitted is not True:
                raise RuntimeError('Stop transport rejected the request')
        except Exception as error:
            self._fail('Stop request failed: ' + str(error))
        return self.state

    def tick(self, *, now):
        if self.state not in ('REQUESTED', 'STATIONARY_OBSERVED'):
            return self.state
        if not _number(now) or now < self.requested_at:
            self._fail('Invalid watchdog time')
        elif self.state == 'REQUESTED' and now - self.requested_at > self.timeout:
            self._fail('Stop confirmation timeout')
        elif now - (self.last_sample if self.last_sample is not None else self.requested_at) > self.max_sample_age:
            self._fail('Robot feedback missing or stale')
        return self.state

    def telemetry_lost(self):
        if self.state in ('REQUESTED', 'STATIONARY_OBSERVED'):
            self._fail('Robot telemetry disconnected')

    def observe(self, *, sampled_at, now, tcp_speed, joint_speed):
        if self.state not in ('REQUESTED', 'STATIONARY_OBSERVED'):
            raise RuntimeError('No live stop observation')
        if (not _number(sampled_at) or not _number(now) or
                sampled_at <= self.requested_at or sampled_at > now or
                now - sampled_at > self.max_sample_age or
                (self.last_sample is not None and sampled_at <= self.last_sample) or
                sampled_at - (self.last_sample if self.last_sample is not None else self.requested_at) > self.max_sample_age or
                not _six(tcp_speed) or not _six(joint_speed)):
            self._fail('Robot feedback invalid, missing or stale')
            raise ValueError(self.failure)
        if self.state == 'REQUESTED' and now - self.requested_at > self.timeout:
            self._fail('Stop confirmation timeout')
            raise ValueError(self.failure)
        self.last_sample = sampled_at
        still = (math.hypot(*tcp_speed[:3]) < self.max_tcp_linear and
                 math.hypot(*tcp_speed[3:]) < self.max_tcp_angular and
                 max(abs(item) for item in joint_speed) < self.max_joint_speed)
        if not still:
            self.steady_since = None
            if self.state == 'STATIONARY_OBSERVED':
                self._fail('Robot moved after stationary observation')
        elif self.steady_since is None:
            self.steady_since = sampled_at
        elif sampled_at - self.steady_since >= self.dwell:
            self.state = 'STATIONARY_OBSERVED'
        return self.state
