"""为已审核安装提供 XYZ 常驻程序的目标与确认引擎，不连接设备或上传程序。
通过外部持有、可独立停止的门控使用原有 RTDE 空间协议。
安装记录须对应本次实体安装及已审核路径，仿真限值或当前姿态不足以替代审核。本模块未经实机调试；
成功发送、收到确认或读到静止均不等同于安全等级的停止。复用的常驻程序固定工具朝向，不能执行倾斜开场。
监管器须串行调用所有方法；注入的门控须在停止后原子禁止写入。
访客、相机、状态、TCP 配置或管理面板失联时，由独立看门狗停止运行。"""
import math

import numpy as np

from .physical_spatial import resident_program
from .physical_robot_model import model_for_profile
from .reference_gate import rotation_vector_distance
from .native_gesture import GestureCadence, finite_gesture_program


def _vec(value, n, name):
    if (not isinstance(value, (tuple, list)) or len(value) != n or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError(f'Invalid {name}')
    return tuple(float(v) for v in value)


class Installation:
    """使用已明确审核的动力学参数。
    可选的 max_tcp_linear_speed（米/秒）和 max_tcp_angular_speed（弧度/秒）可设为有限正阈值；
    显式设为 None 仅关闭对应的额外速度比较。省略时沿用 0.007 米/秒和 0.01 弧度/秒。
    运行时将含这两个字段的完整安装配置与审核记录绑定。"""

    # 审核值只能是浮点数或不可变的浮点元组。构造后冻结对象，避免程序生成或遥测检查悄悄使用被修改的参数。
    # 冻结只能防止意外更改，不能证明来源可信。
    __slots__ = ('robot_model', 'neutral', 'joints', 'tcp', 'lower', 'upper', 'speed',
                 'acceleration', 'base_limit', 'base_rate', 'max_tcp_linear_speed',
                 'max_tcp_angular_speed', 'gesture', '_frozen')

    def __setattr__(self, name, value):
        if getattr(self, '_frozen', False):
            raise AttributeError('Reviewed installation is immutable')
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, '_frozen', False):
            raise AttributeError('Reviewed installation is immutable')
        object.__delattr__(self, name)

    def __init__(self, data):
        if not isinstance(data, dict):
            raise ValueError('Installation profile required')
        self.robot_model = model_for_profile(data)
        self.gesture = GestureCadence.from_dict(data['gesture']) if 'gesture' in data else None
        operator_selected = 'operator_selection' in data
        if operator_selected:
            from .operator_design import validate_operator_selection, INSTALLATION_SELECTION_FIELDS
            if not validate_operator_selection(data,kind='installation',fields=INSTALLATION_SELECTION_FIELDS):
                raise ValueError('Invalid operator-selected installation')
        else:
            required = ('reference', 'tcp', 'payload', 'path', 'dynamics', 'direction', 'stop')
            review = data.get('review')
            if not isinstance(review, dict) or any(review.get(k) is not True for k in required):
                raise ValueError('Installation review incomplete')
        self.neutral = _vec(data.get('reference'), 6, 'reference')
        self.joints = _vec(data.get('joints'), 6, 'joints')
        self.tcp = _vec(data.get('tcp'), 6, 'TCP')
        self.lower = _vec(data.get('lower'), 3, 'lower bound')
        self.upper = _vec(data.get('upper'), 3, 'upper bound')
        if any(not self.lower[i] + .003 < self.neutral[i] < self.upper[i] - .003 for i in range(3)):
            raise ValueError('Reference must be inside the reviewed envelope')
        for key in ('speed', 'acceleration', 'base_limit', 'base_rate'):
            value = data.get(key)
            # 只有显式选择的方案才能省略旧软件层的基座位置通道约束；
            # 基座速度、全关节速率、笛卡尔边界、控制器和租约检查仍然有效。
            if key == 'base_limit' and value is None and operator_selected:
                self.base_limit = None
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid reviewed {key}')
            if key in ('base_limit', 'base_rate') and value > math.pi:
                raise ValueError(f'Invalid reviewed {key}')
            setattr(self, key, float(value))
        # 复用的常驻程序会生成这些表达式，即使输入有限也可能因运算而溢出。
        # 此处只检查数值可表示性，不替代项目层面的速度或加速度限制。
        try:
            derived = (self.speed * 1.15 + .002,
                       (self.acceleration * .35) ** 2, 2 * self.acceleration)
        except OverflowError as exc:
            raise ValueError('Reviewed dynamics produce unrepresentable resident arithmetic') from exc
        if not all(math.isfinite(value) for value in derived):
            raise ValueError('Reviewed dynamics produce unrepresentable resident arithmetic')
        for key, legacy in (('max_tcp_linear_speed', .007), ('max_tcp_angular_speed', .01)):
            value = data.get(key, legacy)
            if value is not None:
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'Invalid reviewed {key}')
                value = float(value)
            setattr(self, key, value)
        self._frozen = True

    def bounds(self):
        return np.asarray(self.lower), np.asarray(self.upper)

    def program(self, session):
        """使用现有常驻程序，应用本次安装对应的速度监测配置。"""
        program = resident_program(self, session)
        angular_guard = ('elif False:' if self.max_tcp_angular_speed is None else
                         'elif not (norm([actual_speed[3],actual_speed[4],actual_speed[5]]) '
                         f'< {self.max_tcp_angular_speed:.10f}):')
        linear_guard = ('elif False:' if self.max_tcp_linear_speed is None else
                        'elif not (norm([actual_speed[0],actual_speed[1],actual_speed[2]]) '
                        f'< {self.max_tcp_linear_speed:.10f}):')
        replacements = (
            ('elif norm([actual_speed[3],actual_speed[4],actual_speed[5]]) > 0.02:',
             angular_guard),
            (f'elif norm([actual_speed[0],actual_speed[1],actual_speed[2]]) > {self.speed*1.15+.002:.10f}:',
             linear_guard),
            ('elif norm([relative[3],relative[4],relative[5]]) > 0.02:',
             'elif not (norm([relative[3],relative[4],relative[5]]) < 0.0100000000):'),
        )
        for previous, replacement in replacements:
            if program.count(previous) != 1:
                raise RuntimeError('Resident guard changed; review required')
            program = program.replace(previous, replacement)
        # 已结束的会话不能留下空转的常驻程序，阻止下一次明确启动。
        # 完成静止握手后如心跳丢失，执行既有 stopl 并退出，不自动返回参考位或重新使能。
        initialization='  moving = False\n  target = get_actual_tcp_pose()\n'
        clock_start='  global packet_age = 1.0\n  thread heartbeat_clock():\n'
        clock_tick='      global packet_age = packet_age + get_steptime()\n      sync()\n'
        timeout=('    if packet_age > 0.25:\n'
                 '      armed = False\n      enabled = False\n    end\n')
        hold=('        if flag == 0:\n          armed = True\n          enabled = False\n')
        for marker in (initialization,clock_start,clock_tick,timeout,hold):
            if program.count(marker)!=1:
                raise RuntimeError('Resident heartbeat structure changed; review required')
        program=program.replace(initialization,initialization+'  ever_armed = False\n')
        program=program.replace(clock_start,'  global packet_age = 1.0\n  global startup_age = 0.0\n  thread heartbeat_clock():\n')
        program=program.replace(clock_tick,'      global packet_age = packet_age + get_steptime()\n'
                                '      global startup_age = startup_age + get_steptime()\n      sync()\n')
        program=program.replace(hold,hold+'          ever_armed = True\n')
        shutdown=('    if packet_age > 0.25:\n'
                  '      armed = False\n      enabled = False\n'
                  '      if ever_armed:\n'
                  f'        if moving:\n          stopl({max(.1,self.acceleration):.10f})\n        end\n'
                  '        write_output_integer_register(42, 2)\n        break\n'
                  '      end\n    end\n'
                  '    if not ever_armed and startup_age > 3.0:\n'
                  '      armed = False\n      enabled = False\n'
                  f'      stopl({max(.1,self.acceleration):.10f})\n'
                  '      write_output_integer_register(42, 2)\n      break\n    end\n')
        program=program.replace(timeout,shutdown)
        return finite_gesture_program(program,self,session)

    def check_feedback(self, fb):
        pose = _vec(fb.get('tcp_pose'), 6, 'TCP pose')
        speed = _vec(fb.get('tcp_speed'), 6, 'TCP speed')
        joints = _vec(fb.get('joints'), 6, 'joints')
        if any(not self.lower[i] <= pose[i] <= self.upper[i] for i in range(3)):
            raise ValueError('TCP left reviewed envelope')
        if rotation_vector_distance(pose[3:], self.neutral[3:]) >= .01:
            raise ValueError('TCP orientation changed')
        if ((self.max_tcp_linear_speed is not None and
                math.hypot(*speed[:3]) >= self.max_tcp_linear_speed) or
                (self.max_tcp_angular_speed is not None and
                 math.hypot(*speed[3:]) >= self.max_tcp_angular_speed)):
            raise ValueError('TCP speed exceeded guard')
        if self.base_limit is not None and abs(joints[0] - self.joints[0]) > self.base_limit + .002:
            raise ValueError('Base joint exceeded reviewed limit')
        return pose, speed, joints

    def check_target(self, value):
        target = _vec(value, 3, 'target')
        if any(not self.lower[i] + .003 <= target[i] <= self.upper[i] - .003 for i in range(3)):
            raise ValueError('Target outside reviewed envelope and braking margin')
        return target


class ResidentEngine:
    """每个常驻会话使用静止握手、顺序 RTDE 确认及故障锁定。
    调用者须从持续前进的控制器时钟获取新反馈，并以足够频率调用 tick，满足 250 毫秒看门狗。
    停止会使门控失效，不命令机械臂返回，也不重连或重试。"""
    def __init__(self, cfg, gate, session):
        if not isinstance(cfg, Installation) or type(session) is not int or not 0 < session < 2**31:
            raise ValueError('Reviewed installation and valid session required')
        self.cfg, self.gate, self.session = cfg, gate, session
        self.state = 'HANDSHAKE'
        self.sequence = 0
        self.pending = None
        self.last_send = None
        self.last_sample = None
        self.goal = None
        self.steady_since = None
        self.motion_acked = False
        self.motion_sent = False
        self.completed = 0
        self.gesture_endpoint = None
        self._ack_receipt_key = self._ack_receipt_at = None
        self._ack_receipts_seen = False

    def _fail(self, reason):
        self.state = 'FAULT'
        self.gate.stop(reason)
        raise RuntimeError(reason)

    def stop(self):
        self.state = 'STOPPED'
        self.gate.stop('Resident engine stopped')

    def target(self, goal, *, now):
        if self.state != 'READY':
            raise RuntimeError('Resident not held and ready')
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ValueError('Invalid monotonic time')
        self.goal = self.cfg.check_target(goal)
        self.gesture_endpoint = None
        self.steady_since = None
        self.motion_acked = False
        self.motion_sent = False
        self.state = 'MOVING'

    def cancel_before_enable(self):
        """仅丢弃尚未尝试发送运动指令的暂存动作。保留未完成的原生 HOLD 确认及其原始时序。此操作不代表停止或静止证明；
        监管器放开观测前仍须核对最新参考位遥测。"""
        if self.state in ('FAULT', 'STOPPED'):
            raise RuntimeError('Resident session is closed')
        if getattr(self.gate, 'can_send', True) is not True:
            self._fail('Resident motion gate is closed')
        if (self.state != 'MOVING' or self.motion_sent
                or self.pending is not None and self.pending[2] not in (0, 3)):
            self._fail('Cannot cancel resident action after motion may have been sent')
        self.goal = self.gesture_endpoint = self.steady_since = None
        self.motion_acked = False
        self.state = 'READY'

    def gesture(self, endpoint, *, now):
        """按控制器时钟执行一次有限动作周期，最终返回参考位。"""
        if self.cfg.gesture is None:raise ValueError('Canonical gesture is not configured')
        self.target(endpoint,now=now)
        self.gesture_endpoint=self.goal
        self.goal=tuple(self.cfg.neutral[:3])

    def tick(self, fb, *, sampled_at, now):
        if self.state in ('FAULT', 'STOPPED'):
            raise RuntimeError('Resident session is closed')
        try:
            if (type(now) not in (int, float) or not math.isfinite(now) or
                    type(sampled_at) not in (int, float) or not math.isfinite(sampled_at) or
                    not 0 <= now - sampled_at <= .15 or
                    self.last_sample is not None and sampled_at <= self.last_sample):
                raise ValueError('Stale or repeated feedback')
            if not isinstance(fb, dict) or fb.get('session') != self.session:
                raise ValueError('Resident session mismatch')
            if type(fb.get('state')) is not int or type(fb.get('sequence')) is not int:
                raise ValueError('Invalid resident acknowledgement')
            if fb['state'] < 0 or fb['state'] == 2 and self.state != 'HANDSHAKE':
                raise ValueError('Resident stopped or faulted')
            has_receipt='ack_sampled_at' in fb
            if has_receipt!=('worker_sampled_at' in fb):
                raise ValueError('Incomplete resident acknowledgement receipt')
            if has_receipt:
                receipt=fb['ack_sampled_at'];received=fb.get('worker_sampled_at')
                key=(fb['session'],fb['sequence'])
                if (type(receipt) not in (int,float) or not math.isfinite(receipt)
                        or type(received) not in (int,float) or not math.isfinite(received)
                        or not 0<=now-received<=.15 or not 0<=receipt<=received):
                    raise ValueError('Invalid resident acknowledgement receipt')
                if key==self._ack_receipt_key and receipt!=self._ack_receipt_at:
                    raise ValueError('Resident acknowledgement receipt changed')
                self._ack_receipt_key,self._ack_receipt_at=key,receipt
                self._ack_receipts_seen=True
            elif self._ack_receipts_seen:
                raise ValueError('Resident acknowledgement receipt missing')
            pose, speed, joints = self.cfg.check_feedback(fb)
            if self.last_sample is None and (math.dist(pose[:3], self.cfg.neutral[:3]) > .0005 or
                    rotation_vector_distance(pose[3:], self.cfg.neutral[3:]) > .005 or
                    max(abs(a-b) for a,b in zip(joints,self.cfg.joints)) > .01 or
                    math.hypot(*speed[:3]) >= .0005 or math.hypot(*speed[3:]) >= .002):
                raise ValueError('Current pose is not the reviewed stationary reference')
            if self.last_sample is not None and sampled_at-self.last_sample > .15:
                self.steady_since = None
            self.last_sample = sampled_at
            if self.pending:
                sequence, sent, enabled = self.pending
                if fb['sequence'] == sequence:
                    # 如果工作线程按时观察到确认，即使本轮轮询较晚，也不能让更新的同一 ACK 缓存样本抹去该记录。
                    # 未收到回执时沿用原先的保守截止时间。
                    if ((has_receipt and not sent<=receipt<=sent+.2)
                            or not has_receipt and now-sent>.2):
                        raise ValueError('Resident acknowledgement timeout')
                    self.pending = None
                    if enabled:
                        self.motion_acked = True
                    if not enabled and fb['state'] == 0:
                        if self.state == 'HANDSHAKE': self.state = 'READY'
                        elif self.state == 'HOLDING':
                            self.state = 'READY'; self.completed += 1; self.goal = None
                elif now-sent>.2:
                    raise ValueError('Resident acknowledgement timeout')
            if self.state == 'MOVING' and self.goal is not None:
                near = (self.motion_acked and math.dist(pose[:3], self.goal) <= .0005 and
                        math.hypot(*speed[:3]) < .0005 and math.hypot(*speed[3:]) < .002
                        and (self.gesture_endpoint is None or fb['state']==3))
                if near:
                    if self.gesture_endpoint is not None and self.pending is None:
                        self.state = 'HOLDING'
                    elif self.steady_since is None: self.steady_since = sampled_at
                    elif sampled_at - self.steady_since >= .6 and self.pending is None:self.state='HOLDING'
                else:
                    self.steady_since = None
            if self.pending is None and (self.last_send is None or now-self.last_send >= .02):
                enabled = (2 if self.gesture_endpoint is not None else 1) if self.state == 'MOVING' else 0
                # 即使引擎已处于 READY，恢复过程中仍须明确发送原生暂停指令。flag0 会清除有限轨迹的相位。
                if getattr(self.gate, 'is_paused', False): enabled = 3
                goal = list(pose[:3] if enabled in (0,3) else (self.gesture_endpoint or self.goal))
                self.sequence += 1
                # 在传输前记录发送尝试，因为报错的写入仍可能已经到达控制器。原生暂停指令 3 不会启动本动作。
                if enabled in (1, 2):
                    self.motion_sent = True
                self.gate.send(self.sequence, enabled, self.session, goal)
                self.pending = (self.sequence, now, enabled)
                self.last_send = now
        except (ValueError, RuntimeError, TypeError, KeyError) as exc:
            self._fail(str(exc))
