"""移动机器人上的相机的故障关闭观察门。本地主管使用原始单调采样时间提供测量到的机器人姿势/速度。
该类不能独立验证传感器测量值；呼叫者必须提供实际的新机器人遥测数据。"""
import math,secrets


def _vector(value):
    if (not isinstance(value,(list,tuple)) or len(value)!=6 or
            any(type(v) not in (float,int) or not math.isfinite(v) for v in value)):
        raise ValueError('Invalid six-axis telemetry')
    return tuple(float(v) for v in value)


def rotation_vector_distance(a, b):
    """返回在[0, pi]中以弧度表示的最短相对旋转角度。旋转向量是轴角表示，并且不是唯一的。
    比较相应的旋转，而不是减去它们的向量。"""
    def quaternion(v):
        if (not isinstance(v, (list, tuple)) or len(v) != 3 or
                any(type(x) not in (float, int) or not math.isfinite(x) for x in v)):
            raise ValueError('Invalid rotation vector')
        theta = math.hypot(*v)
        if not math.isfinite(theta):
            raise ValueError('Invalid rotation magnitude')
        if theta == 0:
            return (1., 0., 0., 0.)
        scale = math.sin(theta / 2) / theta
        return (math.cos(theta / 2), *(x * scale for x in v))

    wa, ax, ay, az = quaternion(a)
    wb, bx, by, bz = quaternion(b)
    # 共轭（a）* b；q和-q编码相同的物理旋转。
    scalar = wa * wb + ax * bx + ay * by + az * bz
    x = wa * bx - wb * ax - ay * bz + az * by
    y = wa * by - wb * ay - az * bx + ax * bz
    z = wa * bz - wb * az - ax * by + ay * bx
    return 2 * math.atan2(math.hypot(x, y, z), abs(scalar))


class ReferenceGate:
    def __init__(self):
        self.session=None;self.reference=None;self.steady_since=None
        self.last_sample=None;self.release_ns=0;self.acknowledged=False
        self._last_pose=self._last_speed=None

    @property
    def blocked(self):return self.session is not None

    def begin(self,reference,now):
        if self.blocked or not math.isfinite(now):raise ValueError('Gate already blocked or invalid time')
        self.reference=_vector(reference);self.session=secrets.token_urlsafe(24)
        self.steady_since=None;self.last_sample=None;self.acknowledged=False
        self._last_pose=self._last_speed=None
        return self.session

    def acknowledge(self):
        if self.blocked:self.acknowledged=True

    def held(self,source_ns):
        return self.blocked or type(source_ns) is not int or source_ns<=self.release_ns

    def sample(self,session,pose,speed,sampled_at,now):
        if not self.session or session!=self.session:raise ValueError('Wrong reference session')
        try:
            pose=_vector(pose);speed=_vector(speed)
            if (type(sampled_at) not in (int,float) or not math.isfinite(sampled_at)
                    or not math.isfinite(now) or not 0<=now-sampled_at<=.15
                    or self.last_sample is not None and sampled_at<=self.last_sample):
                raise ValueError('Stale, repeated or invalid robot sample')
            if self.last_sample is not None and sampled_at-self.last_sample>.15:
                self.steady_since=None
            self.last_sample=sampled_at
            self._last_pose,self._last_speed=pose,speed
            inside=(math.dist(pose[:3],self.reference[:3])<=.0005 and
                    rotation_vector_distance(pose[3:],self.reference[3:])<=.005 and
                    math.hypot(*speed[:3])<.0005 and math.hypot(*speed[3:])<.002)
            if not inside:self.steady_since=None
            elif self.steady_since is None:self.steady_since=sampled_at
        except ValueError:
            self.steady_since=None
            raise

    def return_verified(self,session,now):
        return bool(self.session and session==self.session and self.acknowledged and math.isfinite(now)
                    and self.steady_since is not None and self.last_sample is not None
                    and 0<=now-self.last_sample<=.15 and self.last_sample-self.steady_since>=.6)

    def release(self,session,now):
        if not self.return_verified(session,now):
            raise ValueError('Reference return and standstill not verified')
        self.release_ns=round(now*1e9);self.session=None;self.reference=None
        self.steady_since=None;self.last_sample=None

    def check_details(self,now):
        """只有受限的诊断测量；没有姿势或会话标识符。"""
        result={'acknowledged':self.acknowledged,'steady_ms':0.}
        if self.last_sample is not None:
            result['sample_age_ms']=round((now-self.last_sample)*1000,3)
        if self.steady_since is not None and self.last_sample is not None:
            result['steady_ms']=round((self.last_sample-self.steady_since)*1000,3)
        if self.reference is not None and self._last_pose is not None:
            result.update(position_error_mm=round(math.dist(self._last_pose[:3],self.reference[:3])*1000,6),
                rotation_error_rad=round(rotation_vector_distance(self._last_pose[3:],self.reference[3:]),9),
                linear_speed_mm_s=round(math.hypot(*self._last_speed[:3])*1000,6),
                angular_speed_rad_s=round(math.hypot(*self._last_speed[3:]),9))
        if not self.blocked:status='unblocked'
        elif self.last_sample is None or self._last_pose is None or self._last_speed is None:status='no_sample'
        elif not 0<=now-self.last_sample<=.15:status='freshness'
        elif math.dist(self._last_pose[:3],self.reference[:3])>.0005:status='position'
        elif rotation_vector_distance(self._last_pose[3:],self.reference[3:])>.005:status='orientation'
        elif math.hypot(*self._last_speed[:3])>=.0005 or math.hypot(*self._last_speed[3:])>=.002:status='moving'
        elif not self.acknowledged:status='acknowledgement'
        elif self.steady_since is None or self.last_sample-self.steady_since<.6:status='dwell'
        else:status='ready'
        result['status']=status
        return result


class RestartReferenceProof:
    """在保持旧观察门时证明一个新的选定参考。仅用于在干净的预会话退休后明确的新开始。
    它不会返回机器人、重播命令或接受过期的旧证明。"""
    def __init__(self,original):
        self.original=original
        self.candidate=None
        self.last_sample=None

    @property
    def reference(self):
        return self.candidate.reference if self.candidate is not None else None

    def sample(self,session,pose,speed,sampled_at,now):
        if not self.original.blocked or session!=self.original.session:
            raise ValueError('Wrong reference session')
        try:
            pose,speed=_vector(pose),_vector(speed)
            if (type(sampled_at) not in (int,float) or not math.isfinite(sampled_at)
                    or not math.isfinite(now) or not 0<=now-sampled_at<=.15
                    or self.last_sample is not None and sampled_at<=self.last_sample):
                raise ValueError('Stale, repeated or invalid robot sample')
            gap=self.last_sample is not None and sampled_at-self.last_sample>.15
            self.last_sample=sampled_at
            moved=(self.candidate is None or math.dist(pose[:3],self.reference[:3])>.0005
                or rotation_vector_distance(pose[3:],self.reference[3:])>.005)
            if moved:
                self.candidate=ReferenceGate()
                self.candidate.begin(pose,now)
            elif gap:
                self.candidate.steady_since=None
            self.candidate.sample(self.candidate.session,pose,speed,sampled_at,now)
        except ValueError:
            if self.candidate is not None:self.candidate.steady_since=None
            raise

    def return_verified(self,session,now):
        if (not self.original.blocked or session!=self.original.session
                or not self.original.acknowledged or self.candidate is None):return False
        self.candidate.acknowledge()
        return self.candidate.return_verified(self.candidate.session,now)

    def check_details(self,now):
        if self.candidate is None:
            return {'status':'no_sample','steady_ms':0.,'acknowledged':self.original.acknowledged}
        self.candidate.acknowledged=self.original.acknowledged
        return self.candidate.check_details(now)

    def release(self,session,now):
        if not self.return_verified(session,now):
            raise ValueError('Reference return and standstill not verified')
        # 只提交已完成的新证明。旧门/会话在整个收集过程中都保持被阻止；没有中间样本取代它。
        self.original.reference=self.candidate.reference
        self.original.steady_since=self.candidate.steady_since
        self.original.last_sample=self.candidate.last_sample
        self.original._last_pose=self.candidate._last_pose
        self.original._last_speed=self.candidate._last_speed
        self.original.release(session,now)
