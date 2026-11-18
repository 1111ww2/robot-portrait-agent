"""仅供本地教学仿真器使用的相对参考位动作。映射器在记录的六关节姿态上使用标称 UR5e 模型和局部迭代逆解。
屏幕与基座方向仅是仿真约定，不是相机和实体机器人的标定变换，不能用于实机。"""
import math
import json
from dataclasses import replace
import numpy as np

from .gesture_preview import normalized_offset_for
from .gesture_feedback import edge_feedback_for
from .completion_choreography import CONFIRM_END_MS,SCREEN_CONFIRM_END_MS
from .physical_spatial import nominal_fk
from .simulator_feedback import SIM_HORIZONTAL_M,SIM_VERTICAL_M,SIM_DEPTH_M
from .simulator_guidance import GuidanceStream,MAX_AGE
from .robot_gestures import gesture_plan_for
from .person_guidance import GuidanceCue


def _vector(value):
    if (not isinstance(value,(list,tuple,np.ndarray)) or len(value)!=6 or
            any(type(x) not in (int,float,np.float64,np.float32) or not math.isfinite(float(x)) for x in value)):
        raise ValueError('Expected six finite joint or TCP values')
    return np.asarray(value,dtype=float)


def validate_reference_home(state,joints,tcp):
    """拒绝不匹配的仿真器实例、工具偏移或未经准备的姿态。"""
    q=_vector(joints);expected_tcp=_vector(tcp)
    actual=_vector(state.get('joints'));actual_tcp=_vector(state.get('active_tcp_offset'))
    if (state.get('service')!='urx-browser-simulator' or state.get('backend')!='sim'
            or np.max(np.abs(actual-q))>.002 or np.max(np.abs(actual_tcp-expected_tcp))>1e-6):
        raise ValueError('Recorded simulation reference joints and TCP are required')


class ReferenceMapper:
    """围绕一个仿真姿态求解已限幅的标称笛卡尔动作。"""
    def __init__(self,joints,tcp):
        self.joints=_vector(joints);self.tcp=_vector(tcp)
        self.origin=nominal_fk(self.joints,self.tcp)
        self.inverse=np.linalg.pinv(self._jacobian(self.joints,self.origin),rcond=1e-4)

    def _jacobian(self,joints,origin):
        columns=[];epsilon=1e-5
        for index in range(6):
            sample=joints.copy();sample[index]+=epsilon
            shifted=nominal_fk(sample,self.tcp)
            rotation=shifted[:3,:3]@origin[:3,:3].T
            omega=np.array([rotation[2,1]-rotation[1,2],rotation[0,2]-rotation[2,0],
                            rotation[1,0]-rotation[0,1]])/(2*epsilon)
            columns.append(np.r_[(shifted[:3,3]-origin[:3,3])/epsilon,omega])
        jacobian=np.stack(columns,axis=1)
        singular_values=np.linalg.svd(jacobian,compute_uv=False)
        if not np.isfinite(jacobian).all() or singular_values[-1]<1e-4:
            raise ValueError('Simulation reference is singular or invalid')
        return jacobian

    def _solve(self,translation):
        target=np.array(translation,dtype=float)
        desired=np.r_[target,[0.,0.,0.]]
        joints=self.joints+self.inverse@desired
        for _ in range(8):
            if not np.isfinite(joints).all() or np.max(np.abs(joints-self.joints))>.32:
                raise ValueError('Simulation gesture exceeds the joint envelope')
            actual=nominal_fk(joints,self.tcp)
            rotation=self.origin[:3,:3]@actual[:3,:3].T
            error=np.r_[self.origin[:3,3]+target-actual[:3,3],
                [(rotation[2,1]-rotation[1,2])/2,(rotation[0,2]-rotation[2,0])/2,
                 (rotation[1,0]-rotation[0,1])/2]]
            if np.linalg.norm(error[:3])<1e-5 and np.linalg.norm(error[3:])<5e-5:
                return (joints-self.joints).tolist()
            jacobian=self._jacobian(joints,actual)
            step=np.linalg.pinv(jacobian,rcond=1e-4)@error
            step*=min(1.,.06/max(1e-12,float(np.max(np.abs(step)))))
            joints+=step
        raise ValueError('Simulation reference gesture has no converged local solution')

    def offset(self,plan,elapsed_ms):
        if not math.isfinite(elapsed_ms):raise ValueError('Invalid gesture time')
        if plan is None or elapsed_ms<0 or elapsed_ms>=plan.duration_ms:return [0.]*6
        if plan.pattern in ('sim_frame_adjust_hold','sim_frame_confirm_retract_hold'):
            raise ValueError('Frame-mode simulation choreography is not supported at this reference')
        if plan.pattern=='confirm_retract_hold':
            # 使用实测参考位的试验会返回自己的起始位，不采用旧默认原位的回撤或保持几何。
            confirm_end=SCREEN_CONFIRM_END_MS if plan.profile=='screen-coupled' else CONFIRM_END_MS
            intensity=edge_feedback_for(plan,elapsed_ms)['intensity'] if elapsed_ms<confirm_end else 0.
            h,v,d=0.,-.25*intensity,0.
        elif plan.cue.startswith('MOVE_') and plan.pattern in ('double_sweep','double_sweep_smooth','agent_sweep'):
            # 仅专用 sim-fast 配置使用近、远方案中的幅度；其他仿真配置保留原始全幅映射。
            local_plan=plan if plan.profile=='sim-fast' else replace(plan,amplitude=1.)
            h,v,d=normalized_offset_for(local_plan,elapsed_ms)
        else:h,v,d=normalized_offset_for(plan,elapsed_ms)
        # 这些基座轴只与原仿真器正面视图对应。
        if not all(math.isfinite(x) and abs(x)<=1 for x in (h,v,d)):
            raise ValueError('Invalid simulation gesture offset')
        directional=plan.cue.startswith('MOVE_')
        # 扩大摆动幅度时，保留邀请和确认动作的原有编排。
        horizontal,vertical,depth=((SIM_HORIZONTAL_M,SIM_VERTICAL_M,SIM_DEPTH_M)
                                   if directional else (.05,.035,.025))
        # 从参与者视角观察时，左右方向提示需反转。
        horizontal_sign=1. if plan.cue in ('MOVE_LEFT','MOVE_RIGHT') else -1.
        # 在快速参考仿真中，机械臂回撤邀请参与者靠近，向其移动则提示后退。
        depth_sign=-1. if plan.profile=='sim-fast' and plan.cue in ('MOVE_CLOSER','MOVE_FARTHER') else 1.
        return self._solve([horizontal_sign*horizontal*h,depth_sign*depth*d,-vertical*v])


class ReferenceGuidanceStream(GuidanceStream):
    """沿用原有提示时效和取消语义，更新参考姿态。"""
    def __init__(self,joints,tcp):
        super().__init__();self.mapper=ReferenceMapper(joints,tcp);self.intro_anchor=None;self.intro_retract=None
        self.intro_completed=False

    def accept(self,line,now):
        event=json.loads(line)
        if event.get('event')!='simulator_intro':
            return super().accept(line,now)
        if len(line)>8192 or event.get('action') not in ('start','heartbeat'):
            raise ValueError('Invalid simulation intro event')
        ns=event.get('source_monotonic_ns');intro_id=event.get('intro_id')
        if type(ns) is not int or ns<=0 or type(intro_id) is not int or intro_id<=0 or now-ns/1e9<0:
            raise ValueError('Invalid simulation intro timestamp or id')
        with self.lock:
            if ns/1e9<=self.stamp:raise ValueError('Guidance timestamps must strictly increase')
            active_intro=(self.plan and self.plan.pattern=='curiosity_intro' and self.tracking
                          and not self.transport_interrupted and not self.intro_completed
                          and 0<=now-self.stamp<=MAX_AGE)
            # 开场中重复开始不能把倾斜姿态设为新原点。沿用原标识，只允许对应心跳续期。
            if event['action']=='start' and active_intro:return
            if event['action']=='heartbeat' and not (
                    self.plan and self.plan.pattern=='curiosity_intro' and self.plan_source_ns==intro_id
                    and self.tracking and not self.transport_interrupted):
                return
            if now-ns/1e9>MAX_AGE or (event['action']=='heartbeat' and ns/1e9-self.stamp>MAX_AGE+1e-9):
                self.plan=None;self.tracking=False;self.transport_interrupted=True;self.stamp=ns/1e9
                return
            self.stamp=ns/1e9
            if event['action']=='start':
                self.plan=gesture_plan_for(GuidanceCue.INVITE,profile='sim-intro')
                self.started=self.stamp;self.plan_source_ns=intro_id;self.intro_anchor=None;self.intro_retract=None
                self.intro_completed=False
                self.tracking=True;self.cue='INVITE';self.completion_interrupted=False;self.transport_interrupted=False

    def mark_intro_completed(self,intro_id):
        """只有实测确认已返回且静止，才允许再次开始。"""
        with self.lock:
            if (self.plan and self.plan.pattern=='curiosity_intro' and self.plan_source_ns==intro_id
                    and self.tracking and not self.transport_interrupted):
                self.intro_completed=True

    def anchor_intro(self,intro_id,joints):
        """每轮开场只读取一次仿真器实测姿态快照。"""
        with self.lock:
            if not (self.plan and self.plan.pattern=='curiosity_intro' and self.plan_source_ns==intro_id
                    and self.tracking and not self.transport_interrupted):return False
            if self.intro_anchor==intro_id:return False
            self.mapper=ReferenceMapper(joints,self.mapper.tcp)
            # 围绕倾斜姿态求解一次。在仿真正面视图中，基座 −Y 使工具远离观看者。回撤时保留倾角，不向前俯仰。
            tilted=self.mapper.joints.copy();tilted[4]-=.28
            self.intro_retract=ReferenceMapper(tilted,self.mapper.tcp)._solve([0.,-.10,0.])
            self.intro_anchor=intro_id
            return True

    def offset(self,now):
        with self.lock:
            if not self.tracking or not 0<=now-self.stamp<=MAX_AGE:
                self._cancel_completion();self.plan=None;self.transport_interrupted=True;return None
            if self.completion_interrupted or self.transport_interrupted or self.plan is None:return None
            if self.plan.pattern=='curiosity_intro':
                if self.intro_anchor!=self.plan_source_ns:return None
                return curiosity_offset((now-self.started)*1000,self.intro_retract)
            return self.mapper.offset(self.plan,(now-self.started)*1000)


def curiosity_offset(elapsed_ms,retract):
    """仅仿真：倾斜、招手两次并做出明显回程，然后静止。"""
    if not math.isfinite(elapsed_ms):raise ValueError('Invalid intro time')
    retract=_vector(retract)
    t=max(0.,min(5600.,elapsed_ms))
    def smooth(x):
        x=max(0.,min(1.,x));return x*x*(3-2*x)
    tilt=smooth(t/700.)*(1-smooth((t-4750.)/850.))
    wave=0.
    if 850.<=t<4650.:
        phase=((t-850.)/1900.)%1.
        if phase<.35:
            wave=smooth(phase/.35)
        elif phase<.70:
            # 返回时越过倾斜位置，使回程动作可见。
            wave=1.-1.25*smooth((phase-.35)/.35)
        elif phase<.85:
            wave=-.25
        else:
            wave=-.25*(1.-smooth((phase-.85)/.15))
    offset=retract*wave
    offset[4]-=.28*tilt
    return offset.tolist()
