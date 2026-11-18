"""使用标记的合成参与者输入进行自愿选择的原生URSim工作流程验证。
所有控制器执行都使用共享的原生运行时和测量到的RTDE。相机图像和构图提供商是确定性的本地测试夹具；
这不是对物理安装或真人感知的接受。"""
import argparse
import copy
import http.client
from concurrent.futures import Future
from dataclasses import replace
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
from contextlib import ExitStack, redirect_stderr
import sys
import threading
import time
from types import SimpleNamespace
import urllib.request

from .ursim_system import ROOT, verify_docker, fresh_profile, serve
from .ursim_transport import LoopbackGateway, SimRTDEChannel, SIM_HOST, dependencies

CUES=('MOVE_LEFT','MOVE_RIGHT','MOVE_UP','MOVE_DOWN','MOVE_CLOSER','MOVE_FARTHER')
POINTS=((.8,.3,.2),(.2,.3,.2),(.5,.6,.2),(.5,.1,.2),(.5,.3,.1),(.5,.3,.4))
VALIDATOR_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def participant_point(cue):
    if cue not in CUES:raise ValueError('Unsupported deterministic participant cue')
    return POINTS[CUES.index(cue)]


def cycle_proof(samples,*,require_controller_clock=False,native_hold_s=0.):
    if not samples:raise ValueError('No measured gesture samples')
    if type(native_hold_s) not in (int,float) or not math.isfinite(native_hold_s) or native_hold_s<0:
        raise ValueError('Invalid measured native hold duration')
    peaks=[];high=False;local_peak=0.
    for sample in samples:
        value=sample['fraction']
        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Invalid measured excursion')
        if value>=.9 and not high:high=True;local_peak=value
        if high:
            local_peak=max(local_peak,value)
            if value<=.35:peaks.append(local_peak);high=False
    if high:peaks.append(local_peak)
    if len(peaks)!=2:raise ValueError('Two measured excursions reaching90percent amplitude required')
    peak=max(x['joint_speed'] for x in samples)
    if not math.isfinite(peak) or peak>.735:raise ValueError('Measured joint speed exceeds command cap and tracking margin')
    error=samples[-1]['tcp_error_m']
    if not math.isfinite(error) or error>.0005:raise ValueError('Measured return is incomplete')
    result=dict(excursions=len(peaks),peak_fractions=peaks,max_joint_speed_rad_s=peak,
                max_displacement_fraction=max(x['fraction'] for x in samples),return_error_m=error,
                cadence_matched=False)
    if all('t' in sample for sample in samples):
        times=[sample['t'] for sample in samples]
        if any(type(value) not in (int,float) or not math.isfinite(value) for value in times) or any(b<=a for a,b in zip(times,times[1:])):
            raise ValueError('Invalid measured cycle timestamps')
        result['measured_cycle_s']=times[-1]-times[0]
        settling=next((sample['t'] for sample in samples if sample.get('phase')=='settling'),None)
        result['measured_moving_s']=None if settling is None else settling-times[0]
        result['cadence_clock']='wall_offline_fixture'
        result['cadence_matched']=bool(result['measured_cycle_s']<=5.3 and settling is not None and settling-times[0]<=4.1)
    if all('robot_timestamp' in sample for sample in samples):
        times=[sample['robot_timestamp'] for sample in samples]
        if (not all('t' in sample for sample in samples) or
                any(type(value) not in (int,float) or not math.isfinite(value) for value in times)
                or any(b<=a for a,b in zip(times,times[1:]))):
            raise ValueError('Invalid controller clock timestamps')
        settling=next((sample['robot_timestamp'] for sample in samples if sample.get('phase')=='settling'),None)
        result.update(controller_cycle_s=times[-1]-times[0],
            controller_moving_s=None if settling is None else settling-times[0],cadence_clock='controller')
        result['simulator_clock_ratio']=result['controller_cycle_s']/result['measured_cycle_s']
        result['cadence_matched']=bool(result['controller_cycle_s']<=5.3 and settling is not None and settling-times[0]<=4.1)
        if native_hold_s:
            if native_hold_s>=result['controller_cycle_s'] or settling is None:
                raise ValueError('Native hold exceeds measured moving cycle')
            result.update(native_hold_s=native_hold_s,
                controller_active_cycle_s=result['controller_cycle_s']-native_hold_s,
                controller_active_moving_s=settling-times[0]-native_hold_s)
            result['cadence_matched']=bool(0<=result['controller_active_cycle_s']<=5.3 and
                0<=result['controller_active_moving_s']<=4.1)
    elif require_controller_clock:raise ValueError('Actual controller clock samples required')
    return result


def _jitter_duration(value):
    if type(value) is not int or value!=0 and not 300<=value<=2000:
        raise ValueError('Communication jitter must be0 or300..2000milliseconds')
    return value


def _control_identity(control):
    names=('instance','epoch','control_started_ns')
    value={key:control.get(key) for key in names}
    if (control.get('active') is not True or not isinstance(value['instance'],str) or not value['instance']
            or type(value['epoch']) is not int or value['epoch']<0
            or type(value['control_started_ns']) is not int or value['control_started_ns']<=0):
        raise ValueError('Active control identity required for recovery proof')
    return value


class _RecoveryProbe:
    """观察实际运行时的暂停/继续，无需驱动它。"""
    def __init__(self,stage,*,feedback,control,action_id,intro_id,gap):
        if stage not in ('intro','MOVE_LEFT'):raise ValueError('Invalid recovery stage')
        self.data=dict(stage=stage,native_hold_state=103 if stage=='intro' else 4,
            native_session=feedback['session'],action_id=action_id,intro_id=intro_id,
            control=_control_identity(control),gap=gap,waiting_observed=False,same_control=True,
            stationary_hold=None,resumed_at=None,native_hold_s=None,completed=False,observations=[])
        self.last_sample=self.last_robot=self.still_since=self.first_hold_robot=None
        self._intro_return_since=self._intro_return_last=self._intro_return_robot=None
    @staticmethod
    def _at_intro_reference(feedback,pose,joints):
        from .reference_gate import rotation_vector_distance
        values=(pose,joints,feedback.get('tcp_pose'),feedback.get('joints'),
                feedback.get('joint_speed'),feedback.get('tcp_speed'))
        if any(not isinstance(v,(list,tuple)) or len(v)!=6 or
               any(type(x) not in (int,float) or not math.isfinite(x) for x in v) for v in values):return False
        return (math.dist(feedback['tcp_pose'][:3],pose[:3])<=.0005
                and rotation_vector_distance(feedback['tcp_pose'][3:],pose[3:])<=.005
                and max(abs(a-b) for a,b in zip(feedback['joints'],joints))<=.003
                and max(map(abs,feedback['joint_speed']))<.003
                and math.hypot(*feedback['tcp_speed'][:3])<.0005
                and math.hypot(*feedback['tcp_speed'][3:])<.002)
    def observe_intro_return(self,feedback,sampled_at,now,control,*,reference_pose,reference_joints):
        """仅对原始完成的介绍的全新测量返回进行密封。"""
        if self.data['stage']!='intro' or self.data['resumed_at'] is None:return
        if _control_identity(control)!=self.data['control']:raise ValueError('Intro return identity changed')
        if feedback.get('session')!=self.data['native_session']:return
        if (feedback.get('state')!=102 or not 0<=now-sampled_at<=.15 or
                not self._at_intro_reference(feedback,reference_pose,reference_joints)):
            if self.data.get('intro_return'):raise ValueError('Intro completed return regressed')
            self._intro_return_since=self._intro_return_last=self._intro_return_robot=None
            return
        robot=feedback.get('timestamp')
        if type(robot) not in (int,float) or not math.isfinite(robot):raise ValueError('Intro return clock unavailable')
        if self._intro_return_last is not None:
            if sampled_at<=self._intro_return_last or robot<=self._intro_return_robot:return
            if sampled_at-self._intro_return_last>.15:self._intro_return_since=None
        self._intro_return_last,self._intro_return_robot=sampled_at,robot
        if self._intro_return_since is None:self._intro_return_since=sampled_at
        if sampled_at-self._intro_return_since>=.6-1e-9:
            self.data['intro_return']=dict(native_session=feedback['session'],native_state=102,
                stationary_dwell_s=sampled_at-self._intro_return_since,sampled_at=sampled_at,robot_timestamp=robot,
                reference_pose=list(reference_pose),reference_joints=list(reference_joints),
                measured_pose=list(feedback['tcp_pose']),measured_joints=list(feedback['joints']))
    def observe(self,state,feedback,sampled_at,now,control):
        if _control_identity(control)!=self.data['control']:raise ValueError('Recovery control identity changed')
        robot=feedback.get('timestamp')
        if (type(sampled_at) not in (int,float) or not 0<=now-sampled_at<=.15
                or type(robot) not in (int,float) or not math.isfinite(robot)):
            raise ValueError('Recovery proof requires fresh native telemetry')
        handoff=self.data.get('intro_handoff')
        if feedback.get('session')!=self.data['native_session'] or handoff is not None:
            returned=self.data.get('intro_return')
            new_session=feedback.get('session')
            if (self.data['stage']!='intro' or returned is None
                    or type(new_session) is not int or new_session<=0 or new_session==self.data['native_session']
                    or handoff is not None and new_session!=handoff['to_session']
                    or feedback.get('state') not in (0,2)
                    or state.get('communication',{}).get('status')!='healthy'
                    or not self._at_intro_reference(feedback,returned['reference_pose'],returned['reference_joints'])):
                raise ValueError('Recovery native session changed without verified intro return/held handoff')
            if handoff is None:
                self.data['intro_handoff']=dict(from_session=self.data['native_session'],to_session=new_session,
                    native_state=feedback['state'],sampled_at=sampled_at,robot_timestamp=robot,
                    observed_at=now,completed_return_sampled_at=returned['sampled_at'])
            return
        # 被动观察者可以两次对相同的有效数据包进行轮询。它不能将缓存数据包转换为更多的停留时间或第二个时钟触发。
        if self.last_sample is not None and (sampled_at<=self.last_sample or robot<=self.last_robot):return
        if self.last_sample is not None and sampled_at-self.last_sample>.15:self.still_since=None
        self.last_sample,self.last_robot=sampled_at,robot
        communication=state.get('communication',{})
        status=communication.get('status')
        if status=='waiting' and communication.get('recoverable') is True:self.data['waiting_observed']=True
        native=feedback.get('state');motion=state.get('motion') or {}
        if (self.data['stage']=='MOVE_LEFT' and motion.get('active') and motion.get('fresh')
                and motion.get('action_id')!=self.data['action_id']):
            raise ValueError('Recovery replayed or replaced the native action')
        self.data['observations'].append(dict(t=now,sampled_at=sampled_at,robot_timestamp=robot,
            communication=status,native_state=native,joint_speed=max(map(abs,feedback['joint_speed'])),
            tcp_linear_speed=math.hypot(*feedback['tcp_speed'][:3]),tcp_angular_speed=math.hypot(*feedback['tcp_speed'][3:])))
        if native==self.data['native_hold_state']:
            if self.first_hold_robot is None:self.first_hold_robot=robot
            still=(max(map(abs,feedback['joint_speed']))<.003 and math.hypot(*feedback['tcp_speed'][:3])<.0005
                   and math.hypot(*feedback['tcp_speed'][3:])<.002)
            if status not in ('waiting','ready') or not still:self.still_since=None
            elif self.still_since is None:self.still_since=sampled_at
            elif sampled_at-self.still_since>=.3-1e-9:
                self.data['stationary_hold']=dict(dwell_s=sampled_at-self.still_since,sampled_at=sampled_at,
                    max_joint_speed=max(map(abs,feedback['joint_speed'])),
                    tcp_linear_speed=math.hypot(*feedback['tcp_speed'][:3]),
                    tcp_angular_speed=math.hypot(*feedback['tcp_speed'][3:]))
        elif self.first_hold_robot is not None and self.data['resumed_at'] is None:
            expected=(101,102) if self.data['stage']=='intro' else (0,1,3)
            if native not in expected:raise ValueError('Unexpected native state after recovery HOLD')
            if self.data['stationary_hold'] is None:raise ValueError('Recovery resumed without measured stationary HOLD')
            if status!='healthy':raise ValueError('Native motion resumed before communication recovery')
            self.data.update(resumed_at=now,resumed_native_session=feedback['session'],
                             native_hold_s=robot-self.first_hold_robot)
    def complete(self,*,control,reference_release_ns,post_return_frame_source_ns,now,return_error_m,action_id=None):
        if _control_identity(control)!=self.data['control']:raise ValueError('Recovery completed under changed identity')
        if self.data['resumed_at'] is None:raise ValueError('Recovery continuation not observed')
        if (type(reference_release_ns) is not int or type(post_return_frame_source_ns) is not int
                or not 0<reference_release_ns<post_return_frame_source_ns):raise ValueError('New post-return frame required')
        if not math.isfinite(return_error_m) or not 0<=return_error_m<=.0005:raise ValueError('Recovery return incomplete')
        if self.data['stage']=='MOVE_LEFT' and action_id is not None and action_id!=self.data['action_id']:
            raise ValueError('Recovery completed a different action')
        self.data.update(completed=True,completed_at=now,completed_control=_control_identity(control),
            reference_release_ns=reference_release_ns,post_return_frame_source_ns=post_return_frame_source_ns,
            return_error_m=return_error_m,completed_action_id=action_id,
            reference_release_scope='intro_ready_camera_cutoff' if self.data['stage']=='intro' else 'reference_gate_release')
        if self.data['stage']=='intro':self.data['intro_ready_cutoff_ns']=reference_release_ns
    def result(self):return copy.deepcopy(self.data)


def validate_communication_recovery_proof(result):
    """只有在请求注入时才需要两个原始恢复记录。"""
    duration=_jitter_duration(result.get('communication_jitter_ms',0))
    if not duration:return True
    proof=result.get('fault_injection')
    if (result.get('scenario')!='complete' or type(result.get('operator_selected_binding_count')) is not int
            or result.get('operator_selected_binding_count')!=1
            or not isinstance(proof,dict) or proof.get('kind')!='owner_heartbeat_gap'
            or type(proof.get('owner_start_count')) is not int or proof.get('owner_start_count')!=1):
        raise ValueError('Invalid recovery encounter proof')
    stages=proof.get('stages')
    if not isinstance(stages,list) or [item.get('stage') for item in stages]!=['intro','MOVE_LEFT']:
        raise ValueError('Both opening and directional recovery proofs required')
    for item,expected in zip(stages,(103,4)):
        hold=item.get('stationary_hold');gap=item.get('gap') or {}
        numeric=(item.get('resumed_at'),item.get('native_hold_s'),item.get('return_error_m'),
                 gap.get('started_at'),gap.get('resume_not_before'),gap.get('resumed_at'))
        if any(type(value) not in (int,float) or not math.isfinite(value) for value in numeric):
            raise ValueError('Invalid recovery proof clock/numerics')
        identity=item.get('control')
        if not isinstance(identity,dict):raise ValueError('Missing recovery control identity')
        _control_identity(dict(identity,active=True))
        if (item.get('native_hold_state')!=expected or item.get('waiting_observed') is not True
                or item.get('same_control') is not True or item.get('completed') is not True
                or not isinstance(hold,dict) or hold.get('dwell_s',-1)<.3-1e-9
                or any(type(hold.get(key)) not in (int,float) or not math.isfinite(hold[key])
                       for key in ('dwell_s','sampled_at','max_joint_speed','tcp_linear_speed','tcp_angular_speed'))
                or not 0<=hold['max_joint_speed']<.003 or not 0<=hold['tcp_linear_speed']<.0005
                or not 0<=hold['tcp_angular_speed']<.002
                or identity!=item.get('completed_control') or identity!=stages[0].get('control')
                or type(item.get('native_session')) is not int or item['native_session']<=0
                or item.get('resumed_native_session')!=item['native_session']
                or type(item.get('intro_id')) is not int or item['intro_id']<=0
                or item['intro_id']!=stages[0].get('intro_id')
                or expected==4 and (type(item.get('action_id')) is not int or item['action_id']<=0
                                     or item.get('completed_action_id')!=item['action_id'])
                or type(item.get('resumed_at')) not in (int,float)
                or item['resumed_at']<hold['sampled_at']
                or type(item.get('native_hold_s')) not in (int,float) or item['native_hold_s']<=0
                or gap.get('requested_ms')!=duration or type(gap.get('resumed_at')) not in (int,float)
                or abs(gap['resume_not_before']-gap['started_at']-duration/1000)>.001
                or gap['resumed_at']<gap.get('resume_not_before',float('inf'))
                or type(item.get('reference_release_ns')) is not int
                or type(item.get('post_return_frame_source_ns')) is not int
                or not 0<item['reference_release_ns']<item['post_return_frame_source_ns']
                or not 0<=item.get('return_error_m',float('inf'))<=.0005):
            raise ValueError('Incomplete native recovery/return/new-frame proof')
    return True


class _StageProof:
    """将计划阶段绑定到一个观察到的动作和真实参考会话中。"""
    def __init__(self,cue,previous_action_id=0):
        self.cue=cue;self.previous_action_id=previous_action_id
        self.action_id=None;self.gate_session=None;self.bound_at=None
        self.unexecuted_reservations=0;self.finished=False
    def observe(self,blocked,session,motion,now):
        if not blocked:return
        if not isinstance(session,str) or not session:raise ValueError('Reference session unavailable')
        if self.gate_session is None:self.gate_session=session
        elif self.gate_session!=session:
            if self.action_id is not None:raise ValueError('Reference session changed before completion')
            self.unexecuted_reservations+=1;self.gate_session=session
        if not isinstance(motion,dict) or not motion.get('active') or not motion.get('fresh'):return
        identifier=motion.get('action_id')
        if type(identifier) is not int or identifier<=0:raise ValueError('Invalid accepted action identity')
        if identifier<=self.previous_action_id:return
        if motion.get('cue')!=self.cue:raise ValueError('Observed action cue does not match participant stage')
        if self.action_id is None:self.action_id=identifier;self.bound_at=now
        elif identifier!=self.action_id:raise ValueError('Accepted action changed inside one reference session')
    def release(self,now):
        if self.action_id is None or self.gate_session is None:raise ValueError('Reference release has no bound native action')
        return dict(cue=self.cue,action_id=self.action_id,gate_session=self.gate_session,
                    bound_at=self.bound_at,released_at=now)
    def advance(self,held,blocked,session,motion,now):
        if self.finished:return None
        self.observe(blocked,session,motion,now)
        if not held and self.gate_session is not None:
            if self.action_id is None:
                self.unexecuted_reservations+=1;self.gate_session=None
                return None
            binding=self.release(now);self.finished=True;return binding
        return None


def _reference_snapshot(display,source_ns):
    with display._lock:
        gate=display._reference_gate
        return dict(held=gate.held(source_ns),blocked=gate.blocked,session=gate.session,motion=display._current_motion())


def _samples_through_release(samples,release):
    return [sample for sample in samples if sample['t']<release['t'] and
        ('robot_timestamp' not in release or sample.get('robot_timestamp',float('-inf'))<release['robot_timestamp'])]+[release]


def light_proof(rows,bindings):
    counts={cue:0 for cue in bindings}
    for state in rows:
        motion=state.get('motion') or {};feedback=state.get('feedback') or {};runtime=state.get('runtime') or {}
        cue=motion.get('cue')
        if (cue in bindings and state.get('active') is True and state.get('owner') is True
                and state.get('control_fresh') is True and state.get('tracking') is False
                and state.get('phase')=='PHYSICAL_REFERENCE_HOLD' and runtime.get('status')=='running'
                and runtime.get('reason')=='none' and motion.get('active') is True and motion.get('fresh') is True
                and motion.get('action_id')==bindings[cue] and feedback.get('gesture_id')==bindings[cue]
                and type(motion.get('sample_age_ms')) in (int,float) and 0<=motion['sample_age_ms']<=300
                and type(feedback.get('intensity')) in (int,float) and feedback['intensity']>.05):
            counts[cue]+=1
    missing=[cue for cue,count in counts.items() if not count]
    if missing:raise ValueError('No action-bound reference-hold light evidence for '+','.join(missing))
    return counts


def _cleanup_steps(steps):
    errors=[]
    for label,callback in steps:
        try:callback()
        except BaseException as exc:errors.append(label+': '+type(exc).__name__+': '+str(exc))
    return errors


class _NativeDiagnostics:
    """仅对运行时现有的传输进行不可改变的观察。"""
    def __init__(self):
        self.lock=threading.Lock();self.rows=[];self.first=None;self.errors=[];self.last={}
    def _record(self,row,*,fault=False):
        row=copy.deepcopy(row)
        with self.lock:
            key=(row['kind'],row.get('channel'))
            signature={k:v for k,v in row.items() if k!='observed_at'}
            if self.last.get(key)==signature:return
            self.last[key]=signature;self.rows.append(row)
            if fault and self.first is None:self.first=copy.deepcopy(row)
    def feedback(self,channel,snapshot,*,expected_session=None,pending_sequence=None,failure=None):
        fb,stamp=snapshot
        row=dict(kind='rtde',channel=channel,observed_at=time.monotonic(),sampled_at=stamp,
            feedback=fb,expected_session=expected_session,pending_sequence=pending_sequence,
            session_match=None if expected_session is None or not isinstance(fb,dict) else fb.get('session')==expected_session,
            sequence_match=None if pending_sequence is None or not isinstance(fb,dict) else fb.get('sequence')==pending_sequence,
            transport_failure=str(failure)[:500] if failure else None)
        fault=bool(failure or isinstance(fb,dict) and type(fb.get('state')) is int and fb['state']<0)
        self._record(row,fault=fault)
    def primary(self,channel,snapshot,*,notice=None,failure=None):
        state,stamp=snapshot
        self._record(dict(kind='primary',channel=channel,observed_at=time.monotonic(),sampled_at=stamp,
            state=state,notice=str(notice)[:1000] if notice else None,
            transport_failure=str(failure)[:500] if failure else None),fault=bool(notice or failure))
    def command(self,channel,sequence,enable,session,target):
        self._record(dict(kind='command',channel=channel,observed_at=time.monotonic(),
            sequence=sequence,enable=enable,session=session,target=target))
    def session_error(self,message):
        row=dict(kind='session_error',observed_at=time.monotonic(),message=str(message)[:1000])
        with self.lock:self.errors.append(copy.deepcopy(row))
        self._record(row,fault=True)
    def summary(self):
        with self.lock:return copy.deepcopy(dict(first_native_fault=self.first,session_errors=self.errors,
            observations=len(self.rows)))
    def trace(self):
        with self.lock:return copy.deepcopy(self.rows)


class _DiagnosticChannel:
    def __init__(self,source,diagnostics,label):
        self.source=source;self.diagnostics=diagnostics;self.label=label
        self.expected_session=None;self.pending_sequence=None
    def __getattr__(self,name):return getattr(self.source,name)
    @property
    def latest(self):
        value=self.source.latest
        self.diagnostics.feedback(self.label,value,expected_session=self.expected_session,
            pending_sequence=self.pending_sequence,failure=self.source.failure)
        return value
    def send(self,sequence,enable,session,target):
        self.expected_session=session;self.pending_sequence=sequence
        self.diagnostics.command(self.label,sequence,enable,session,target)
        return self.source.send(sequence,enable,session,target)
    def close(self):
        try:self.latest
        finally:self.source.close()


class _DiagnosticPrimary:
    def __init__(self,source,diagnostics,label):self.source,self.diagnostics,self.label=source,diagnostics,label
    def __getattr__(self,name):return getattr(self.source,name)
    @property
    def latest(self):
        value=self.source.latest
        self.diagnostics.primary(self.label,value,notice=self.source.notice,failure=self.source.failure)
        return value
    def close(self):
        try:self.latest
        finally:
            self.source.close()
            self.diagnostics.primary(self.label,self.source.latest,notice=self.source.notice,failure=self.source.failure)


class _DiagnosticStderr:
    def __init__(self,stream,diagnostics):self.stream,self.diagnostics=stream,diagnostics;self.pending='';self.lock=threading.Lock()
    def write(self,text):
        self.stream.write(text)
        with self.lock:
            self.pending+=text
            while '\n' in self.pending:
                line,self.pending=self.pending.split('\n',1)
                if line.startswith('PHYSICAL_SESSION_STOPPED: '):self.diagnostics.session_error(line.split(': ',1)[1])
            if len(self.pending)>4096:self.pending=self.pending[-4096:]
        return len(text)
    def flush(self):self.stream.flush()


def _public_state(state):
    return {key:state.get(key) for key in ('active','owner','control_fresh','phase','tracking','stale',
        'runtime','motion','feedback','direction','photo_ready','photo_latched','communication')}


def _export_identity(output):
    return dict(profile_sha256=hashlib.sha256((output/'profile.json').read_bytes()).hexdigest(),
        program_sha256={name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                        for name in ('opening.urscript','guidance.urscript')})


def record_execution_profile(profile,output,result,*,robot_model):
    """在任何运动打开器之前，记录实际绑定的数字/程序字节。"""
    from .ursim_profile import SimulationProfile
    from .physical_intro_program import IntroProgram
    if type(profile) is not SimulationProfile or profile.host!=SIM_HOST or profile.cfg.robot_model!=robot_model:
        raise ValueError('Execution record requires the exact matching isolated simulation profile')
    installation={key:getattr(profile.cfg,key) for key in ('robot_model','joints','tcp','lower','upper',
        'speed','acceleration','base_limit','base_rate','max_tcp_linear_speed','max_tcp_angular_speed')}
    installation['reference']=profile.cfg.neutral
    installation['gesture']=profile.cfg.gesture.as_dict() if profile.cfg.gesture else None
    value=dict(model=robot_model,installation=installation,intro=profile.intro,
        paths=profile.paths,simulation_only=True)
    (output/'profile.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    (output/'opening.urscript').write_text(IntroProgram(profile.intro,1).program(),encoding='utf-8')
    (output/'guidance.urscript').write_text(profile.cfg.program(1),encoding='utf-8')
    result['reference_pose']=list(profile.cfg.neutral)
    result['reference_joints']=list(profile.cfg.joints)
    result.update(_export_identity(output))
    if profile.operator_design is not None:
        for name in ('profile.json','opening.urscript','guidance.urscript'):
            with (output/('pre-start-'+name)).open('xb') as stream:
                stream.write((output/name).read_bytes())
        result['pre_start_profile_sha256']=result['profile_sha256']
        result['pre_start_program_sha256']=copy.deepcopy(result['program_sha256'])
        result['program_hash_scope']='inspection_session_1_before_explicit_start'
    if profile.simulation_metadata.get('operator_reference_bound') is True:
        result['operator_selected_binding_count']=result.get('operator_selected_binding_count',0)+1
        result['operator_selected_binding']=True
        result['reference_captured_after_explicit_start']=True
        result['program_hash_scope']='inspection_session_1_from_bound_reference_before_motion'
        previous=result.get('pre_start_program_sha256')
        result['programs_changed_on_reference_binding']=(
            previous!=result['program_sha256'] if previous is not None else None)


def _replay_trace(output):
    executable=shutil.which('node')
    if not executable:raise RuntimeError('Node.js required for production UI replay')
    command=[executable,str(ROOT/'vision_mvp/scripts/check-motion-light-trace.cjs'),str(output/'ui-state.json')]
    replay=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=60,check=False)
    try:result=json.loads(replay.stdout)
    except (ValueError,TypeError):raise RuntimeError('Production UI replay did not return a JSON result') from None
    if replay.returncode or result.get('success') is not True:
        raise RuntimeError('Production UI replay rejected the observed trace')
    return result


def negative_trigger(state,feedback,stamp,now,*,fraction,bound_action_id):
    """需要测量并接受的远足，而不是静止的停车。"""
    motion=state.get('motion') or {}
    if (motion.get('active') is not True or motion.get('fresh') is not True or motion.get('phase')!='moving'
            or motion.get('cue') not in CUES or type(bound_action_id) is not int
            or motion.get('action_id')!=bound_action_id or not 0<=now-stamp<=.15
            or type(fraction) not in (int,float) or not math.isfinite(fraction) or not .15<=fraction<=.9):return None
    qd=feedback.get('joint_speed');speed=feedback.get('tcp_speed')
    if any(not isinstance(v,(list,tuple)) or len(v)!=6 or
           any(type(x) not in (int,float) or not math.isfinite(x) for x in v) for v in (qd,speed)):return None
    joint_speed=max(map(abs,qd));linear_speed=math.hypot(*speed[:3])
    if joint_speed<=.01 or linear_speed<=.005:return None
    return dict(action_id=bound_action_id,cue=motion['cue'],fraction=fraction,
        phase='moving',observed_at=now,sampled_at=stamp,robot_timestamp=feedback['timestamp'],
        joint_speed_rad_s=joint_speed,tcp_linear_speed_m_s=linear_speed)


def prove_negative(owner,scenario,trigger,*,verify_stationary,service_alive,ui_trace,
                   timeout=5.,clock=time.monotonic,sleep=time.sleep,native_measurement=None):
    """在清理前观察实际控制撤销和独立停滞。"""
    if scenario not in ('stop','control-loss'):raise ValueError('Invalid negative scenario')
    negotiated=False;recovery=None
    if scenario=='control-loss':
        before=owner.request('/operator-state');ui_trace.append(_public_state(before))
        metadata=before.get('communication') or {}
        negotiated=metadata.get('recoverable') is True
        if negotiated:
            if metadata.get('status')!='healthy' or metadata.get('remaining_ms')!=0 or not callable(native_measurement):
                raise RuntimeError('Negotiated negative case requires healthy metadata and native measurement')
            recovery=dict(negotiated_window_s=3.,waiting_observed=False,deadline_lower_s=None,
                          deadline_upper_s=None,native_hold=None,observations=[])
    requested_at=clock()
    if scenario=='stop':owner.stop()
    else:owner.pause_heartbeat()
    expected='guest_stop' if scenario=='stop' else 'communication_timeout' if negotiated else 'control_heartbeat_expired'
    deadline=clock()+timeout
    last_native=last_robot=still_since=None
    while clock()<deadline:
        request_began=clock();state=owner.request('/operator-state');observed_at=clock()
        ui_trace.append(_public_state(state))
        revocation=(state.get('physical') or {}).get('revocation') or {}
        reason=revocation.get('reason')
        if negotiated:
            metadata=state.get('communication') or {}
            if metadata.get('status') in ('waiting','ready'):
                remaining=metadata.get('remaining_ms')
                if (metadata.get('recoverable') is not True or type(remaining) not in (int,float)
                        or not math.isfinite(remaining) or not 0<=remaining<=3000.001):
                    raise RuntimeError('Invalid negotiated communication deadline metadata')
                low=request_began+remaining/1000-.000001;high=observed_at+remaining/1000+.000001
                if recovery['deadline_lower_s'] is not None:
                    low=max(low,recovery['deadline_lower_s']);high=min(high,recovery['deadline_upper_s'])
                if low>high+.000002:raise RuntimeError('Negotiated communication deadline reset')
                recovery.update(waiting_observed=True,deadline_lower_s=low,deadline_upper_s=high)
                recovery['observations'].append(dict(request_began=request_began,observed_at=observed_at,
                    status=metadata['status'],remaining_ms=remaining))
                fb,stamp=native_measurement()
                if (not isinstance(fb,dict) or type(stamp) not in (int,float)
                        or type(fb.get('timestamp')) not in (int,float) or not math.isfinite(fb['timestamp'])
                        or not 0<=clock()-stamp<=.15):
                    raise RuntimeError('Native feedback unavailable before communication expiry')
                if last_native is None or stamp>last_native and fb['timestamp']>last_robot:
                    if last_native is not None and stamp-last_native>.15:still_since=None
                    last_native,last_robot=stamp,fb['timestamp']
                    if type(fb.get('state')) is int and fb['state']<0:
                        raise RuntimeError('Native fault before communication expiry')
                    qd=fb.get('joint_speed');speed=fb.get('tcp_speed')
                    still=(fb.get('state')==4 and isinstance(qd,(list,tuple)) and len(qd)==6
                        and type(fb.get('session')) is int and fb['session']>0
                        and isinstance(speed,(list,tuple)) and len(speed)==6
                        and all(type(x) in (int,float) and math.isfinite(x) for x in (*qd,*speed))
                        and max(map(abs,qd))<.003 and math.hypot(*speed[:3])<.0005 and math.hypot(*speed[3:])<.002)
                    previous_hold=recovery['native_hold']
                    if previous_hold is not None and (not still or fb['session']!=previous_hold['native_session']):
                        raise RuntimeError('Native recovery HOLD was not maintained until expiry')
                    if not still:still_since=None
                    elif still_since is None:still_since=stamp
                    elif stamp-still_since>=.3-1e-9:
                        recovery['native_hold']=dict(state=4,native_session=fb.get('session'),sampled_at=stamp,
                            robot_timestamp=fb['timestamp'],stationary_dwell_s=stamp-still_since)
        if reason and reason!=expected:raise RuntimeError('Unexpected first control revocation: '+str(reason))
        if reason==expected:
            if negotiated:
                if (not recovery['waiting_observed'] or recovery['deadline_lower_s'] is None
                        or observed_at<recovery['deadline_lower_s']-.000002):
                    raise RuntimeError('Communication expiry was premature or no fixed wait was observed')
                if recovery['native_hold'] is None:raise RuntimeError('Native recovery HOLD was not observed before expiry')
                recovery['no_premature_terminal']=True
            motion=state.get('motion') or {};feedback=state.get('feedback') or {}
            if motion.get('active') is True or motion.get('fresh') is True or feedback.get('intensity',0)!=0:
                raise RuntimeError('Motion feedback remained visible after control revocation')
            if not service_alive():raise RuntimeError('Backend service retired before independent Stop proof')
            stationary=verify_stationary()
            if stationary is not True:raise RuntimeError('Independent stationary proof failed')
            if not service_alive():raise RuntimeError('Backend service retired during independent Stop proof')
            result=dict(trigger=trigger,requested_at=requested_at,revocation_observed_at=observed_at,
                revocation_reason=reason,stationary=True,service_cancelled_before_proof=False,
                timing_scope='control revocation observation; not a measured robot stopping latency')
            if recovery is not None:result['communication_recovery']=recovery
            return result
        sleep(.02)
    raise TimeoutError('Expected native control revocation was not observed')


class _FixtureProvider:
    client=SimpleNamespace(model='deterministic-composition-fixture')
    def __init__(self):self.calls=0
    def submit(self,jpeg,point):
        from .gemini_framing import CompositionPlan
        self.calls+=1;future=Future()
        future.set_result(CompositionPlan((.5,.3,.2),(.1,.05,.9,.95),.99,'environmental','Deterministic validation target'))
        return future


class _Owner:
    def __init__(self,display):
        self.display=display;self.client='native-validation-owner';self.closed=threading.Event()
        self.control=None;self.failure=None;self.sequence=0;self.stop_sent=False
        self.start_count=0;self._heartbeat_lock=threading.Lock();self._heartbeat_gap=None
        self.local=threading.local()
        self.thread=threading.Thread(target=self._heartbeat,daemon=True)
    def request(self,path,body=None):
        headers={'X-Guest-Token':self.display.token,'X-Guest-Client':self.client}
        data=None
        if body is not None:
            data=json.dumps(body).encode();headers.update({'Content-Type':'application/json','Origin':self.display.origin})
        conn=getattr(self.local,'conn',None)
        if conn is None:
            port=int(self.display.origin.rsplit(':',1)[1])
            conn=http.client.HTTPConnection('127.0.0.1',port,timeout=1);self.local.conn=conn
        began=time.monotonic();phase='send';status=None
        try:
            conn.request('GET' if data is None else 'POST',path,body=data,headers=headers)
            phase='headers';response=conn.getresponse();status=response.status
            phase='body';payload=response.read()
            if status!=200:raise RuntimeError('Validation UI request rejected')
            if response.will_close:conn.close();self.local.conn=None
            phase='JSON'
            return json.loads(payload)
        except Exception as exc:
            conn.close();self.local.conn=None
            message=(f'{type(exc).__name__}: path={path} phase={phase} status={status} '
                     f'elapsed_ms={(time.monotonic()-began)*1000:.1f}')
            raise RuntimeError(message[:400]) from exc
    def start(self,action='start'):
        snapshot=self.display.control_snapshot()
        result=self.request('/action',dict(action=action,control_protocol=1,
            control={k:snapshot[k] for k in ('instance','epoch')}))
        self.control=result['control'];self.start_count+=1;self.thread.start()
    def begin_heartbeat_gap(self,duration_ms):
        _jitter_duration(duration_ms)
        if not duration_ms:raise ValueError('A real heartbeat gap is required')
        with self._heartbeat_lock:
            if self._heartbeat_gap is not None and 'resumed_at' not in self._heartbeat_gap:
                raise RuntimeError('Previous heartbeat gap is still active')
            began=time.monotonic()
            self._heartbeat_gap=dict(started_at=began,resume_not_before=began+duration_ms/1000,
                                    requested_ms=duration_ms,last_sequence=self.sequence)
            return self._heartbeat_gap
    def _heartbeat(self):
        try:
            while not self.closed.is_set():
                try:
                    with self._heartbeat_lock:
                        gap=self._heartbeat_gap;now=time.monotonic()
                        if gap is None or now>=gap['resume_not_before']:
                            self.sequence+=1
                            self.request('/control-heartbeat',{**self.control,'sequence':self.sequence})
                            if gap is not None and 'resumed_at' not in gap:
                                gap.update(resumed_at=time.monotonic(),resumed_sequence=self.sequence)
                except Exception as exc:
                    if not self.closed.is_set():self.failure=(type(exc).__name__+': '+str(exc))[:400]
                    return
                self.closed.wait(.04)
        finally:
            conn=getattr(self.local,'conn',None)
            if conn is not None:conn.close()
    def pause_heartbeat(self):
        self.closed.set()
        if self.thread.is_alive():self.thread.join(2)
        if self.thread.is_alive():raise RuntimeError('Validation heartbeat did not retire')
    def stop(self):
        self.closed.set()
        try:
            if self.control is not None and not self.stop_sent:
                self.request('/action',dict(action='stop',control_protocol=1,control=self.control,stop_origin=1))
                self.stop_sent=True
        finally:
            if self.thread.is_alive():self.thread.join(2)
            conn=getattr(self.local,'conn',None)
            if conn is not None:conn.close();self.local.conn=None
        if self.thread.is_alive():raise RuntimeError('Validation heartbeat did not retire')


class _Participant:
    def __init__(self,display,capture,owner,path,profile,*,skip_intro=False):
        import cv2
        import numpy as np
        from .gemini_guidance import GeminiGuidanceController
        from .app import PhysicalPhotoLatch
        self.display,self.capture,self.owner,self.path,self.profile=display,capture,owner,path,profile
        self.closed=threading.Event();self.failure=None;self.intro_id=None
        self.stage=0;self.was_blocked=False;self.completed=[];self.started_at=time.monotonic()
        self.stage_proof=_StageProof(CUES[0]);self.intro_ready_seen=False
        self.intro_return_proof=None;self.intro_ready_cutoff_ns=0
        self.unexecuted_reservations=[]
        self.skip_intro=skip_intro
        self.telemetry_receiver=None
        self.provider=_FixtureProvider()
        from .app import AppConfig,guidance_controller_options
        app_config=AppConfig(camera_source='ipad',guidance_profile='claude-guide',single_plan=True)
        self.controller=GeminiGuidanceController(worker=self.provider,
            **guidance_controller_options(app_config,screen_coupled=True,physical_mode=True))
        self.latch=PhysicalPhotoLatch(display)
        self.frame=np.zeros((90,160,3),dtype=np.uint8)
        self.jpeg=cv2.imencode('.jpg',self.frame)[1].tobytes()
        self.session=capture.begin(owner.client)
        self.thread=threading.Thread(target=self._run,daemon=True)
    def _intro_frame_ready(self,source_ns,now_ns):
        """镜像app.py的第一个准备就绪截止日期和截止日期后相机帧凭据。"""
        if self.intro_ready_seen:return True
        if not self.display.physical_intro_ready(self.intro_id):return False
        if not self.intro_ready_cutoff_ns:
            self.intro_ready_cutoff_ns=now_ns
            self.capture.invalidate_pending()
        if (type(source_ns) is int and source_ns>self.intro_ready_cutoff_ns
                and 0<=now_ns-source_ns<=300_000_000):
            self.intro_ready_seen=True
            # Intro没有XYZ ReferenceGate.release_ns。
            # 它的生产摄像头屏障是实际的第一个准备就绪的观察，永远不是假的门释放或重写的帧源时间戳。
            self.intro_return_proof=dict(reference_release_ns=self.intro_ready_cutoff_ns,
                                        post_return_frame_source_ns=source_ns)
            return True
        return False
    def _run(self):
        from .app import serialize_guidance_event
        from .person_guidance import GuidanceCue, GuidanceDecision
        try:
            with self.path.open('a',encoding='utf-8',buffering=1) as output:
                while not self.closed.is_set():
                    ticket=self.capture.ticket(self.owner.client,self.session)
                    self.capture.accept(self.owner.client,self.session,ticket,self.jpeg)
                    ok,frame=self.capture.read()
                    if not ok:continue
                    ns=self.capture.source_monotonic_ns
                    control=self.display.control_snapshot()
                    if not control['active']:
                        self.closed.wait(.03);continue
                    if ns<=control['control_started_ns']:
                        self.closed.wait(.005);continue
                    if self.intro_id is None:self.intro_id=ns
                    timestamp=int((time.monotonic()-self.started_at)*1000)
                    if not self.skip_intro and not self._intro_frame_ready(ns,time.monotonic_ns()):
                        event=dict(event='physical_intro',action='heartbeat',intro_id=self.intro_id,
                            source_monotonic_ns=ns,control_instance=control['instance'],control_epoch=control['epoch'])
                        decision=GuidanceDecision('INTRO',GuidanceCue.INVITE,(.5,.3,.2),None,None,False,'VALIDATION_INTRO')
                    else:
                        snapshot=_reference_snapshot(self.display,ns);blocked=snapshot['held']
                        reservations=self.stage_proof.unexecuted_reservations
                        binding=self.stage_proof.advance(**snapshot,now=time.monotonic())
                        if self.stage_proof.unexecuted_reservations!=reservations:
                            self.unexecuted_reservations.append(dict(cue=self.stage_proof.cue,observed_at=time.monotonic()))
                        if binding is not None:
                            measured,measured_at=self.telemetry_receiver()
                            if measured is None or not 0<=time.monotonic()-measured_at<=.15:
                                raise RuntimeError('Reference release observer stale')
                            self.completed.append(dict(binding,
                                reference_release_ns=self.display._reference_gate.release_ns,
                                post_return_frame_source_ns=ns,
                                return_pose=list(measured['tcp_pose']),return_sampled_at=measured_at,
                                return_robot_timestamp=measured['timestamp'],
                                return_error_m=math.dist(measured['tcp_pose'][:3],self.profile.cfg.neutral[:3]),
                                return_joint_speed=max(map(abs,measured['joint_speed']))))
                            self.stage+=1
                            if self.stage<len(CUES):self.stage_proof=_StageProof(CUES[self.stage],binding['action_id'])
                        self.was_blocked=blocked
                        if blocked:
                            decision=self.controller.suspend_perception(timestamp)
                        else:
                            point=participant_point(CUES[self.stage]) if self.stage<len(CUES) else (.5,.3,.2)
                            observation=SimpleNamespace(center_x=point[0],head_y=point[1],shoulder_width=point[2],confidence=.99)
                            decision=self.controller.update(observation,timestamp,mirrored=True,frame=frame)
                        decision=self.latch.apply(decision,active=True,source_ns=ns)
                        event=json.loads(serialize_guidance_event(decision,None,timestamp,source_monotonic_ns=ns))
                        event.update(control_instance=control['instance'],control_epoch=control['epoch'])
                    self.display.publish(None,decision,source_monotonic_ns=ns)
                    output.write(json.dumps(event,separators=(',',':'))+'\n')
                    self.closed.wait(.03)
        except Exception as exc:self.failure=type(exc).__name__+': '+str(exc)
    def close(self):
        self.closed.set()
        if self.thread.ident is not None:self.thread.join(3)
        if self.thread.is_alive():raise RuntimeError('Validation participant did not retire')


def execute_validation(source,output,*,robot_model='ur10e',timeout=180.,scenario='complete',skip_intro=False,
                       communication_jitter_ms=0):
    from .app import PhysicalPhotoLatch
    from .guest_display import GuestDisplay
    from .ipad_camera import IPadCapture
    from .motion_configuration import configure_motion_design
    from .physical_runtime import verify_stationary_after_close
    from .horizontal_follow import check_dashboard
    from .canonical_delivery import attach_simulation_exports
    output.mkdir(parents=True,exist_ok=False)
    result=dict(success=False,robot_model=robot_model,execution_backend='official_ursim_native',
        input_mode='deterministic_camera_and_composition_fixture',physical_acceptance=False,
        control_script_shared=True,joint_speed_command_rad_s=.7,nominal_gesture_duration_s=3.3,
        validator_source_sha256=VALIDATOR_SHA256,ready=False,native_workflow_success=False,
        scenario=scenario,opening_skipped=skip_intro)
    trace=[];ui_trace=[];cycles={cue:[] for cue in CUES};backend_errors=[]
    diagnostics=_NativeDiagnostics()
    recovery_probes=[]
    if communication_jitter_ms:result['communication_jitter_ms']=communication_jitter_ms
    stopped=threading.Event();participant=owner=backend=None;backend_started=False
    stack=ExitStack();inhibit=output.parent/'.canonical-validation-incomplete-cleanup.json'
    try:
        from .canonical_delivery import runtime_source_identity
        result['runtime_source_sha256']=runtime_source_identity(ROOT)['shared_runtime_source_sha256']
        if scenario not in ('complete','stop','control-loss'):raise ValueError('Invalid validation scenario')
        _jitter_duration(communication_jitter_ms)
        if communication_jitter_ms and (scenario!='complete' or skip_intro or source.get('approval_mode')!='operator_selected_design'):
            raise ValueError('Communication jitter requires the complete selected-design workflow')
        if skip_intro and scenario=='complete':raise ValueError('Complete validation cannot skip opening')
        if not math.isfinite(timeout) or not 0<timeout<=180:raise ValueError('Validation timeout must be finite and at most180seconds')
        if inhibit.exists():raise RuntimeError('Previous native validation cleanup is unverified: '+str(inhibit))
        design=configure_motion_design(source)
        identity=verify_docker(robot_model=robot_model)
        gateway=stack.enter_context(LoopbackGateway())
        profile=fresh_profile(design,robot_model=robot_model,joint_speed=.7)
        result['controller_id']=identity.controller_id
        record_execution_profile(profile,output,result,robot_model=robot_model)
        capture=IPadCapture()
        display=stack.enter_context(GuestDisplay(host='127.0.0.1',port=0,remote_capture=capture,physical_feedback=True))
        if display is not None:
            owner=_Owner(display)
            path=output/'participant.jsonl';path.touch()
            participant=_Participant(display,capture,owner,path,profile,skip_intro=skip_intro)
            if profile.operator_design is not None:
                def record_bound(actual):
                    nonlocal profile
                    if result.get('operator_selected_binding_count',0):
                        raise RuntimeError('Validation permits only one explicit selected reference binding')
                    record_execution_profile(actual,output,result,robot_model=robot_model)
                    participant.profile=actual
                    profile=actual
                profile=replace(profile,binding_observer=record_bound)
            # 观察实际的原生运行时自己的接收通道。
            # 额外的观察器连接不得增加另一个控制器的负载，也不能成为停止有效相遇的无关原因。
            runtime_factories=dependencies();channels=[];channels_lock=threading.Lock();channel_count=0;primary_count=0
            primary_factory=runtime_factories['state_factory']
            def real_primary(host,**kwargs):
                nonlocal primary_count
                primary_count+=1
                return _DiagnosticPrimary(primary_factory(host,**kwargs),diagnostics,'primary-'+str(primary_count))
            def real_channel(host,**kwargs):
                nonlocal channel_count
                channel_count+=1
                channel=_DiagnosticChannel(SimRTDEChannel(host,**kwargs),diagnostics,
                    ('reader-' if kwargs.get('receive_only') else 'writer-')+str(channel_count))
                if kwargs.get('receive_only'):
                    with channels_lock:channels.append(channel)
                return channel
            runtime_factories['rtde_factory']=real_channel
            runtime_factories['state_factory']=real_primary
            def measurement():
                with channels_lock:candidates=list(channels)
                samples=[channel.latest for channel in candidates if not channel.failure]
                samples=[item for item in samples if isinstance(item[0],dict)]
                return max(samples,key=lambda item:item[1]) if samples else (None,float('-inf'))
            participant.telemetry_receiver=measurement
            def run_backend():
                try:
                    with redirect_stderr(_DiagnosticStderr(sys.stderr,diagnostics)):
                        serve(profile,{'port':int(display.origin.rsplit(':',1)[1]),'key':display.token},path,
                            SimpleNamespace(is_alive=lambda:not stopped.is_set()),identity,
                            should_stop=stopped.is_set,gateway=gateway,factories=runtime_factories)
                except Exception as exc:backend_errors.append(type(exc).__name__+': '+str(exc))
            backend=threading.Thread(target=run_backend,daemon=True)
            participant.thread.start();backend.start();backend_started=True
            last=None;started=time.monotonic();first_motion_at=None;photo=False;recorded_returns=set();negative=None
            if True:
                deadline=time.monotonic()+3
                while not capture.fresh() and time.monotonic()<deadline:time.sleep(.01)
                if not capture.fresh():raise RuntimeError('Synthetic camera did not become fresh')
                owner.start('restart' if skip_intro else 'start')
                while time.monotonic()-started<timeout:
                    if participant.failure:raise RuntimeError('Participant: '+participant.failure)
                    if owner.failure:raise RuntimeError('Owner: '+owner.failure)
                    if backend_errors:raise RuntimeError('Backend: '+backend_errors[0])
                    state=owner.request('/state')
                    if not state['active']:raise RuntimeError('Encounter became inactive before completion')
                    if state.get('runtime',{}).get('status')=='fault':raise RuntimeError('Runtime fault: '+str(state['runtime']))
                    feedback,stamp=measurement()
                    now=time.monotonic()
                    if feedback is None or not 0<=now-stamp<=.15:
                        # 在XYZ打开之前，Bootstrap故意退出介绍频道。原生控制器监控其自身的有效性；
                        # 不要将被动监控间隙转换为停止。
                        time.sleep(.01);continue
                    row=dict(t=now,robot_timestamp=feedback['timestamp'],tcp_pose=feedback['tcp_pose'],
                        joints=feedback['joints'],joint_speed=feedback['joint_speed'],tcp_speed=feedback['tcp_speed'],
                        native_state=feedback.get('state'),native_sequence=feedback.get('sequence'),native_session=feedback.get('session'))
                    trace.append(row)
                    public=_public_state(state)
                    ui_trace.append(public)
                    motion=state.get('motion',{})
                    fraction=None
                    if motion.get('active') and motion.get('fresh') and motion.get('cue') in CUES and motion['cue'] not in recorded_returns:
                        cue=motion['cue'];origin=profile.cfg.neutral[:3];end=profile.paths[cue][-1]
                        delta=[b-a for a,b in zip(origin,end)];length=math.hypot(*delta)
                        fraction=sum((p-o)*d for p,o,d in zip(feedback['tcp_pose'],origin,delta))/(length*length)
                        if not cycles[cue] or stamp>cycles[cue][-1]['t'] and feedback['timestamp']>cycles[cue][-1]['robot_timestamp']:
                            cycles[cue].append(dict(t=stamp,robot_timestamp=feedback['timestamp'],action_id=motion.get('action_id'),phase=motion.get('phase'),
                                fraction=fraction,joint_speed=max(map(abs,feedback['joint_speed'])),
                                tcp_error_m=math.dist(feedback['tcp_pose'][:3],origin)))
                        if first_motion_at is None:first_motion_at=now
                    if communication_jitter_ms:
                        control=display.control_snapshot()
                        current=recovery_probes[-1] if recovery_probes and not recovery_probes[-1].data['completed'] else None
                        if current is not None:
                            if current.data['stage']=='intro' and participant.intro_ready_seen:
                                current.complete(control=control,now=now,
                                    return_error_m=math.dist(feedback['tcp_pose'][:3],profile.cfg.neutral[:3]),
                                    **participant.intro_return_proof)
                            elif current.data['stage']=='MOVE_LEFT' and any(item['cue']=='MOVE_LEFT' for item in participant.completed):
                                returned=next(item for item in participant.completed if item['cue']=='MOVE_LEFT')
                                current.complete(control=control,now=now,return_error_m=returned['return_error_m'],
                                    reference_release_ns=returned['reference_release_ns'],
                                    post_return_frame_source_ns=returned['post_return_frame_source_ns'],action_id=returned['action_id'])
                            else:
                                current.observe(state,feedback,stamp,now,control)
                                current.observe_intro_return(feedback,stamp,now,control,
                                    reference_pose=profile.cfg.neutral,reference_joints=profile.cfg.joints)
                        if not recovery_probes and feedback.get('state')==101 and max(map(abs,feedback['joint_speed']))>.003:
                            recovery_probes.append(_RecoveryProbe('intro',feedback=feedback,control=control,
                                action_id=None,intro_id=participant.intro_id,gap=owner.begin_heartbeat_gap(communication_jitter_ms)))
                        elif (len(recovery_probes)==1 and recovery_probes[0].data['completed']
                                and motion.get('cue')=='MOVE_LEFT' and motion.get('active') and motion.get('fresh')
                                and motion.get('phase')=='moving' and fraction is not None and .08<=fraction<=.65
                                and max(map(abs,feedback['joint_speed']))>.003):
                            recovery_probes.append(_RecoveryProbe('MOVE_LEFT',feedback=feedback,control=control,
                                action_id=motion['action_id'],intro_id=participant.intro_id,
                                gap=owner.begin_heartbeat_gap(communication_jitter_ms)))
                    if scenario!='complete' and fraction is not None:
                        trigger=negative_trigger(state,feedback,stamp,now,fraction=fraction,
                            bound_action_id=participant.stage_proof.action_id)
                        if trigger is not None:
                            negative=prove_negative(owner,scenario,trigger,
                                verify_stationary=lambda:verify_stationary_after_close(SIM_HOST,
                                    channel_factory=SimRTDEChannel,dashboard_check=check_dashboard),
                                service_alive=lambda:backend.is_alive() and not stopped.is_set(),ui_trace=ui_trace,
                                native_measurement=measurement)
                            break
                    for item in participant.completed:
                        samples=cycles[item['cue']]
                        if item['cue'] not in recorded_returns:
                            if not samples or any(s['action_id']!=item['action_id'] for s in samples):
                                raise RuntimeError('Cycle telemetry does not match accepted native action')
                            origin=profile.cfg.neutral[:3];delta=[b-a for a,b in zip(origin,profile.paths[item['cue']][-1])]
                            fraction=sum((p-o)*d for p,o,d in zip(item['return_pose'],origin,delta))/sum(d*d for d in delta)
                            release=dict(t=item['return_sampled_at'],robot_timestamp=item['return_robot_timestamp'],
                                action_id=item['action_id'],phase='reference_released',
                                fraction=fraction,joint_speed=item['return_joint_speed'],tcp_error_m=item['return_error_m'])
                            samples[:]=_samples_through_release(samples,release)
                            recorded_returns.add(item['cue'])
                    signature=(state['phase'],motion.get('cue'),motion.get('phase'),len(participant.completed))
                    if signature!=last:print('VALIDATION_PHASE '+json.dumps(dict(phase=signature[0],cue=signature[1],motion=signature[2],completed=signature[3])),flush=True);last=signature
                    if state.get('photo_ready') is True and state.get('photo_latched') is True:
                        photo=True;break
                    time.sleep(.025)
                if scenario=='complete':
                    if not photo:raise TimeoutError('Native workflow did not reach photo readiness')
                    if len(participant.completed)!=6:raise RuntimeError('All six native directions must return')
                    if not participant.intro_ready_seen:raise RuntimeError('Native opening did not become ready')
                    native_hold=next((item.data['native_hold_s'] for item in recovery_probes if item.data['stage']=='MOVE_LEFT'),0.)
                    result['directions']={cue:{**cycle_proof(cycles[cue],require_controller_clock=True,
                        native_hold_s=native_hold if cue=='MOVE_LEFT' else 0.),
                        'action_id':participant.completed[i]['action_id'],
                        'reference_session_bound':True,'accepted_at':participant.completed[i]['bound_at'],
                        'released_at':participant.completed[i]['released_at']} for i,cue in enumerate(CUES)}
                    result['cadence_matched']=all(item['cadence_matched'] for item in result['directions'].values())
                    result['composition_requests']=participant.provider.calls
                    if participant.provider.calls!=1:raise RuntimeError('Motion pause repeated composition requests')
                    result['photo_ready']=True
                    result['lit_reference_hold_samples']=light_proof(ui_trace,
                        {item['cue']:item['action_id'] for item in participant.completed})
                    if communication_jitter_ms:
                        result['fault_injection']=dict(kind='owner_heartbeat_gap',owner_start_count=owner.start_count,
                            stages=[item.result() for item in recovery_probes])
                        validate_communication_recovery_proof(result)
                    owner.stop()
                    stop_state=owner.request('/state');ui_trace.append(_public_state(stop_state))
                    stopped.set();backend.join(8)
                    if backend.is_alive():raise RuntimeError('Native workflow did not retire after explicit Stop')
                    result['stationary_after_stop']=verify_stationary_after_close(SIM_HOST,
                        channel_factory=SimRTDEChannel,dashboard_check=check_dashboard)
                    result['native_workflow_success']=True
                    (output/'ui-state.json').write_text(json.dumps(ui_trace),encoding='utf-8')
                    result['production_ui_replay']=_replay_trace(output)
                else:
                    if negative is None:raise TimeoutError('No active native excursion qualified for negative test')
                    result['negative_case']=negative;result['stationary_after_stop']=True
                    result['native_workflow_success']=True
                if source.get('approval_mode')=='operator_selected_design' and result.get('operator_selected_binding_count')!=1:
                    raise RuntimeError('Selected workflow did not execute exactly one fresh reference binding')
                result['elapsed_s']=time.monotonic()-started;result['success']=True
    except BaseException as exc:
        result['failure']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        def retire_backend():
            if backend_started:
                backend.join(8)
                if backend.is_alive():raise RuntimeError('Native backend thread is still alive')
        def prove_stationary():
            if backend_started and not result.get('stationary_after_stop'):
                result['stationary_after_cleanup']=verify_stationary_after_close(SIM_HOST,
                    channel_factory=SimRTDEChannel,dashboard_check=check_dashboard)
        steps=[]
        if owner is not None:steps.append(('owner Stop and heartbeat',owner.stop))
        steps.append(('cancel backend',stopped.set))
        if participant is not None:steps.append(('participant',participant.close))
        steps.extend([('backend thread',retire_backend),('independent stationary proof',prove_stationary),('servers and gateway',stack.close)])
        errors=_cleanup_steps(steps)
        result.update(cleanup_errors=errors,cleanup_complete=not errors,backend_errors=backend_errors,
            backend_thread_alive=bool(backend is not None and backend.is_alive()),
            participant_failure=participant.failure if participant is not None else None)
        result.update(diagnostics.summary())
        if communication_jitter_ms:
            result['fault_injection']=dict(kind='owner_heartbeat_gap',owner_start_count=owner.start_count if owner else 0,
                stages=[item.result() for item in recovery_probes])
        result['unexecuted_reservations']=participant.unexecuted_reservations if participant is not None else []
        if errors:
            result['success']=False
            result.setdefault('failure',errors[0])
            if backend_started:inhibit.write_text(json.dumps(dict(output=str(output),errors=errors),indent=2),encoding='utf-8')
        result['ready']=bool(result['success'] and result.get('cadence_matched') and result['cleanup_complete'])
        evidence_errors=_cleanup_steps([
            ('native diagnostics export',lambda:(output/'native-diagnostics.json').write_text(json.dumps(diagnostics.trace()),encoding='utf-8')),
            ('cycle sample export',lambda:(output/'cycles.json').write_text(json.dumps(cycles),encoding='utf-8')),
            ('telemetry export',lambda:(output/'telemetry.json').write_text(json.dumps(trace),encoding='utf-8')),
            ('UI state export',lambda:(output/'ui-state.json').write_text(json.dumps(ui_trace),encoding='utf-8'))])
        if evidence_errors:
            result['success']=result['ready']=False
            result['evidence_errors']=evidence_errors;result.setdefault('failure',evidence_errors[0])
        (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main(argv=None,*,root=ROOT):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--robot-model',choices=('ur5e','ur10e'),default='ur10e')
    parser.add_argument('--scenario',choices=('complete','stop','control-loss'),default='complete')
    parser.add_argument('--skip-intro',action='store_true',help='Negative cases only: issue a real new-target restart')
    parser.add_argument('--communication-jitter-ms',type=int,default=0,
                        help='Complete selected-design case only: inject owner heartbeat gaps of300..2000ms')
    parser.add_argument('--source-profile',type=Path,default=Path('tmp/robot-network/physical-ipad-ur10e-trial.json'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    try:_jitter_duration(args.communication_jitter_ms)
    except ValueError as exc:parser.error(str(exc))
    root=Path(root).resolve()
    output=(args.output or root/'tmp'/('canonical-validation-'+args.robot_model+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))).resolve()
    if not output.is_relative_to(root) or output==root:parser.error('Validation output must be inside workspace')
    if not args.execute:
        print('Native validation CLI checked offline; no Docker, server or robot interface opened.');return 0
    source_path=args.source_profile if args.source_profile.is_absolute() else root/args.source_profile
    if not source_path.resolve().is_relative_to(root):parser.error('Design source must be inside workspace')
    result=execute_validation(json.loads(source_path.read_text(encoding='utf-8-sig')),output,robot_model=args.robot_model,
        scenario=args.scenario,skip_intro=args.skip_intro,communication_jitter_ms=args.communication_jitter_ms)
    print(json.dumps(result,indent=2));return 0
