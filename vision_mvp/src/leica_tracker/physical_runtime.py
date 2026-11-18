"""原生交互运行程序要求明确开始，并通过实测反馈确认返回参考位。命令行默认离线验证。
旧安装审核记录与操作员显式选定的设计各自保留来源，选定设计不能冒充已完成的实体预览。
收到访客的明确触发后，才用新的独立静止反馈绑定参考位。
两种模式共用开场、原生运动、停止与失控处理逻辑，不自动回原位或重试，不控制夹爪、工具电源或控制器设置。"""
import argparse
import copy
from dataclasses import dataclass
import http.client
import ipaddress
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from .horizontal_follow import DashboardReceiver
from .physical_resident_bootstrap import _state_snapshot, open_resident
from .physical_resident_engine import Installation, ResidentEngine
from .physical_intro_program import IntroProgram
from .physical_intro_bootstrap import open_intro
from .physical_errors import SessionCleanupError
from .physical_robot_model import model_for_profile
from .physical_interaction_supervisor import PhysicalInteractionSupervisor, LocalReferenceGate, ReferenceRequestError
from .resident_guest_watchdog import LocalGuestControlReader


class _NoopGate:
    can_send = False
    def stop(self, _reason): pass


class SwitchingGate:
    """从已审核开场切换到 XYZ 控制连接时，持续锁定已收到的停止请求。"""
    def __init__(self):
        self._lock=threading.Lock();self._gate=None;self.failure=None;self._paused=False
    def attach(self,gate):
        with self._lock:
            if self.failure:
                gate.stop(self.failure)
                raise RuntimeError(f'Physical transition already stopped: {self.failure}')
            if self._gate is not None:raise RuntimeError('Physical gate already attached')
            if not gate.can_send:raise RuntimeError('Physical gate not armed')
            self._gate=gate
            if self._paused:gate.pause('Communication waiting across transition')
    def detach(self,gate):
        with self._lock:
            if self._gate is not gate:raise RuntimeError('Physical gate ownership changed')
            self._gate=None
    def detach_completed(self,gate,begin_retirement):
        """在异步清理之前，锁定完成授权并分离。"""
        if not callable(begin_retirement):raise ValueError('Completed retirement callback required')
        with self._lock:
            if self.failure:raise RuntimeError(self.failure)
            if self._gate is not gate:raise RuntimeError('Physical gate ownership changed')
            begin_retirement()
            self._gate=None
    def stop(self,reason):
        with self._lock:
            if self.failure:return
            self.failure=str(reason);gate=self._gate
        if gate is not None:gate.stop(reason)

    def pause(self,reason):
        with self._lock:
            if self.failure:raise RuntimeError(self.failure)
            self._paused=True;gate=self._gate
            if gate is not None:gate.pause(reason)

    def resume(self):
        with self._lock:
            if self.failure:raise RuntimeError(self.failure)
            if self._gate is not None:self._gate.resume()
            self._paused=False

    @property
    def attached(self):
        with self._lock:return self._gate is not None


@dataclass(frozen=True)
class ReviewedProfile:
    host: str
    cfg: Installation
    paths: dict
    reviewed_paths: dict
    hold_timeout: float
    action_timeout: float
    gate_timeout: float
    max_joint_speed: float
    intro: dict

    def supervisor(self, engine, gate):
        return PhysicalInteractionSupervisor(engine,gate,self.paths,reviewed_paths=self.reviewed_paths,
            hold_timeout=self.hold_timeout,action_timeout=self.action_timeout,
            gate_timeout=self.gate_timeout,max_joint_speed=self.max_joint_speed)


def _continuous_preview(preview,host,reference_joints,tcp,waypoints,reference_pose,robot_model):
    """检查完整、已接受的运行记录，不补造逐点样本。操作员将实际执行配置与整次记录绑定；本地记录未经硬件认证。
    保留的轨迹只是有界的末段数据，并非十四个端点的完整测量。
    历史试验动力学保存在 execution_profile，后续开场速度调整须单独审核，不能因此算作已测。"""
    fields=('status','robot_ip','reference_joints','reference_pose','tcp','waypoints',
            'execution_profile','execution_record','operator_decision')
    model_fields={'robot_model','reviewed_robot_model'} if 'robot_model' in preview else set()
    if (reference_pose is None or set(preview)!=set(fields)|{'reviewed'}|model_fields
            or preview.get('robot_ip')!=host or preview.get('reference_joints')!=list(reference_joints)
            or preview.get('reference_pose')!=list(reference_pose) or preview.get('tcp')!=list(tcp)
            or preview.get('waypoints')!=[list(q) for q in waypoints]
            or preview.get('operator_decision')!='accept'
            or not isinstance(preview.get('reviewed'),dict) or set(preview['reviewed'])!=set(fields)
            or any(preview['reviewed'][key]!=preview.get(key) for key in fields)):
        raise ValueError('Continuous intro preview does not match operator acceptance')
    def number(value):
        return type(value) in (int,float) and math.isfinite(value)
    def vector(value):
        return isinstance(value,(list,tuple)) and len(value)==6 and all(number(v) for v in value)
    execution=preview['execution_profile'];record=preview['execution_record']
    if (not isinstance(execution,dict) or execution.get('mode')!='onsite_manual_14_point_sequence'
            or execution.get('robot_ip')!=host or not isinstance(execution.get('intro'),dict)):
        raise ValueError('Continuous intro preview execution profile missing or mismatched')
    actual=IntroProgram(execution['intro'],1)
    if model_for_profile(execution)!=robot_model or actual.robot_model!=robot_model:
        raise ValueError('Continuous intro preview execution robot model does not match profile')
    if (actual.reference_joints!=tuple(reference_joints) or actual.reference_pose!=tuple(reference_pose)
            or actual.tcp!=tuple(tcp) or actual.waypoints!=tuple(tuple(q) for q in waypoints)):
        raise ValueError('Continuous intro preview executed a different route or installation')
    record_fields={'outcome','error','trace','ack_diagnostic','last_independent_feedback'}
    if (not isinstance(record,dict) or set(record)!=record_fields or record.get('outcome')!='complete'
            or record.get('error') is not None or record.get('ack_diagnostic') is not None):
        raise ValueError('Continuous intro preview run did not complete successfully')
    trace=record['trace']
    if not isinstance(trace,list) or not 3<=len(trace)<=300:
        raise ValueError('Continuous intro preview trace missing')
    prior_local=prior_writer=prior_reader=0.;prior_ack=prior_reader_ack=0
    completed_at=None;had_active=False;had_motion=False
    for row in trace:
        if not isinstance(row,dict):raise ValueError('Invalid continuous intro preview trace')
        local,writer,reader=(row.get(k) for k in ('sampled_at','writer_timestamp','reader_timestamp'))
        wa,ra=row.get('writer_age_s'),row.get('reader_age_s')
        ack,rack,state=row.get('ack'),row.get('reader_ack'),row.get('resident_state')
        q,speed=row.get('joints'),row.get('tcp_speed')
        if (not all(number(v) for v in (local,writer,reader,wa,ra))
                or local<prior_local or local<=0 or writer<=prior_writer or reader<=prior_reader
                or not 0<=wa<=.15 or not 0<=ra<=.15
                or type(ack) is not int or type(rack) is not int or not 0<ack<2**31 or not 0<rack<2**31
                or ack<prior_ack or rack<prior_reader_ack or type(state) is not int or state not in (101,102)
                or not vector(q) or not vector(speed)
                or (actual.max_tcp_linear_speed is not None and
                    math.hypot(*speed[:3])>actual.max_tcp_linear_speed)
                or (actual.max_tcp_angular_speed is not None and
                    math.hypot(*speed[3:])>actual.max_tcp_angular_speed)):
            raise ValueError('Invalid, stale or mismatched continuous intro preview telemetry')
        if state==101:
            if completed_at is not None:raise ValueError('Continuous intro preview restarted after completion')
            had_active=True
            had_motion=had_motion or max(abs(a-b) for a,b in zip(q,reference_joints))>.003
        else:
            if (not had_active or not had_motion or ack!=rack
                    or max(abs(a-b) for a,b in zip(q,reference_joints))>.003
                    or math.hypot(*speed[:3])>=.0005 or math.hypot(*speed[3:])>=.002):
                raise ValueError('Continuous intro preview has no stationary native completion')
            if completed_at is None:completed_at=local
        prior_local,prior_writer,prior_reader=local,writer,reader
        prior_ack,prior_reader_ack=ack,rack
    if completed_at is None or prior_local-completed_at<.6-1e-9:
        raise ValueError('Continuous intro preview completion dwell missing')
    final=record['last_independent_feedback']
    if not isinstance(final,dict):raise ValueError('Continuous intro preview final receiver missing')
    q,pose,speed,qd=(final.get(k) for k in ('joints','tcp_pose','tcp_speed','joint_speed'))
    stamp=final.get('sampled_at');last_reader_time=trace[-1]['sampled_at']-trace[-1]['reader_age_s']
    from .reference_gate import rotation_vector_distance
    if (not all(vector(v) for v in (q,pose,speed,qd)) or not number(stamp)
            or stamp+1e-9<last_reader_time or stamp>prior_local+.15
            or max(abs(a-b) for a,b in zip(q,reference_joints))>.003
            or math.dist(pose[:3],reference_pose[:3])>.0005
            or rotation_vector_distance(pose[3:],reference_pose[3:])>.005
            or max(abs(v) for v in qd)>=.003 or math.hypot(*speed[:3])>=.0005
            or math.hypot(*speed[3:])>=.002):
        raise ValueError('Continuous intro preview final independent state is not stationary at reference')


def validate_preview(preview,host,reference_joints,tcp,waypoints, *, reference_pose=None,robot_model='ur10e'):
    """校验操作员提供的十四点记录是否匹配，不据此证明记录真实。"""
    if isinstance(preview,dict) and model_for_profile(preview)!=robot_model:
        raise ValueError('Intro preview robot model does not match reviewed physical profile')
    if isinstance(preview,dict) and preview.get('status')=='OPERATOR_ACCEPTED_CONTINUOUS':
        if not isinstance(waypoints,(tuple,list)) or len(waypoints)!=14:
            raise ValueError('Exactly 14 continuous intro preview targets required')
        return _continuous_preview(preview,host,reference_joints,tcp,waypoints,reference_pose,robot_model)
    preview_fields=('status','robot_ip','reference_joints','tcp','waypoints',
                    'observed_indices','observations','operator_decision')
    if (not isinstance(preview,dict) or not isinstance(waypoints,(tuple,list)) or len(waypoints)!=14
            or preview.get('status')!='OPERATOR_ACCEPTED' or preview.get('robot_ip')!=host
            or preview.get('reference_joints')!=list(reference_joints)
            or preview.get('tcp')!=list(tcp)
            or preview.get('waypoints')!=[list(q) for q in waypoints]
            or preview.get('observed_indices')!=list(range(1,15))
            or preview.get('operator_decision')!='accept'
            or not isinstance(preview.get('reviewed'),dict)
            or set(preview['reviewed'])!=set(preview_fields)
            or any(preview['reviewed'][key]!=preview.get(key) for key in preview_fields)):
        raise ValueError('Explicit completed intro point preview and operator review required')
    observations=preview.get('observations')
    if not isinstance(observations,list) or len(observations)!=14:
        raise ValueError('Intro preview observations missing')
    previous_time=0.;previous_joints=reference_joints
    for index,(observation,target) in enumerate(zip(observations,waypoints),1):
        if not isinstance(observation,dict) or observation.get('index')!=index:
            raise ValueError('Intro preview observation index mismatch')
        q=observation.get('joints');pose=observation.get('tcp_pose');stamp=observation.get('sampled_at')
        tolerance=observation.get('joint_tolerance')
        segment=max(abs(a-b) for a,b in zip(target,previous_joints))
        if (not isinstance(q,(list,tuple)) or len(q)!=6 or not isinstance(pose,(list,tuple)) or len(pose)!=6
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in [*q,*pose])
                or type(stamp) not in (int,float) or not math.isfinite(stamp) or stamp<=previous_time
                or type(tolerance) not in (int,float) or not math.isfinite(tolerance)
                or not 0<tolerance<=.003 or tolerance>=segment
                or max(abs(a-b) for a,b in zip(q,target))>tolerance):
            raise ValueError('Invalid or mismatched intro preview observation')
        previous_time=stamp;previous_joints=target


def reviewed_profile(data):
    """在打开任何套接字前，检查显式批准的类型及所有路径。"""
    if isinstance(data,dict) and 'approval_mode' in data:
        if data['approval_mode']!='operator_selected_design':
            raise ValueError('Unsupported physical profile approval mode')
        from .operator_design import selected_profile
        return selected_profile(data)
    if not isinstance(data,dict) or data.get('status') != 'PHYSICAL_INTERACTION_REVIEWED':
        raise ValueError('Physical interaction profile review required')
    robot_model=model_for_profile(data)
    try:
        host = str(ipaddress.ip_address(data['robot_ip']))
        if (data.get('reviewed_robot_ip') != host or
                data.get('reviewed_installation') != data.get('installation')):
            raise ValueError('Robot IP and installation must match their review')
        runtime_keys=('hold_timeout','action_timeout','gate_timeout','max_joint_speed')
        approved_runtime=data.get('reviewed_runtime')
        if (not isinstance(approved_runtime,dict) or set(approved_runtime)!=set(runtime_keys)
                or any(approved_runtime[key]!=data.get(key) for key in runtime_keys)):
            raise ValueError('Runtime stop and timeout settings must match their review')
        # IP 不是唯一的硬件身份；实际控制器仍需要由现场操作员识别和审查。
        cfg = Installation(copy.deepcopy(data['installation']))
        intro=copy.deepcopy(data['intro'])
        compiled=IntroProgram(intro,1)
        if cfg.robot_model!=robot_model or compiled.robot_model!=robot_model:
            raise ValueError('Reviewed physical, intro and installation robot models differ')
        if (compiled.reference_joints!=cfg.joints or compiled.reference_pose!=cfg.neutral or compiled.tcp!=cfg.tcp):
            raise ValueError('Reviewed intro and XYZ references differ')
        validate_preview(data.get('intro_preview'),host,compiled.reference_joints,compiled.tcp,compiled.waypoints,
                         reference_pose=compiled.reference_pose,robot_model=robot_model)
        profile = ReviewedProfile(host,cfg,copy.deepcopy(data['paths']),copy.deepcopy(data['reviewed_paths']),
            data['hold_timeout'],data['action_timeout'],data['gate_timeout'],data['max_joint_speed'],intro)
        # 构造函数验证完整的六向匹配路径；此门是惰性的，不会启动本地/网络进程。
        profile.supervisor(ResidentEngine(cfg,_NoopGate(),1),None)
        # 一个投影被发送到 iPad 上，用于整个方向手势。拒绝会导致错误表示的弯曲/回溯。
        origin=cfg.neutral[:3]
        for waypoints in profile.paths.values():
            delta=[v-o for v,o in zip(waypoints[-1],origin)]
            length=math.hypot(*delta)
            if length<=0:raise ValueError('Invalid directional light path')
            axis=[x/length for x in delta];previous=0.
            for point in waypoints:
                offset=[v-o for v,o in zip(point,origin)]
                projection=sum(a*b for a,b in zip(offset,axis))
                residual=math.hypot(*(v-projection*a for v,a in zip(offset,axis)))
                if projection<=previous or residual>1e-6:
                    raise ValueError('Directional light requires a straight outward path')
                previous=projection
        return profile
    except (KeyError,TypeError) as exc:
        raise ValueError('Incomplete reviewed physical profile') from exc


class FeedbackPublisher:
    """关键租约发送器加上单独的有限最佳努力显示器发送器。"""
    def __init__(self, reference, gate, *, clock=time.monotonic, max_latency=.25,recovery=None):
        if type(max_latency) not in (int,float) or not math.isfinite(max_latency) or not 0<max_latency<=.3:
            raise ValueError('Physical feedback deadline must be at most 300 ms')
        self.reference,self.gate,self.clock=reference,gate,clock
        self.max_latency=max_latency
        self.recovery=recovery
        make_control=getattr(reference,'control_transport',None)
        self._control_reference=(make_control(timeout=max_latency) if callable(make_control) else reference)
        self.queue=queue.Queue(maxsize=12)
        self.failed=None
        self._ready={};self._intro_ack={}
        self._pending={};self._sequence=0
        self._lock=threading.Lock()
        self._presentation=[]
        self._presentation_lock=threading.Lock()
        self._presentation_wake=threading.Event()
        self._presentation_thread=None
        self.presentation_failures=0
        self._closed=threading.Event()
        self._thread=threading.Thread(target=self._run,name='physical-feedback-sender',daemon=True)
        self._watch=threading.Thread(target=self._watchdog,name='physical-feedback-deadline',daemon=True)
        self._thread.start()
        self._watch.start()

    def _fail(self,reason):
        with self._lock:
            if self.failed:return
            self.failed=reason
        self.gate.stop(reason)
        self._closed.set()

    def _watchdog(self):
        while not self._closed.wait(min(.025,self.max_latency/4)):
            with self._lock:
                # 匹配提交的锁定：在获得此锁定之前，不得将较新的待定时间戳与在获得此锁定之前采样的时钟进行比较。
                now=self.clock()
                expired=any(not 0<=now-t<=self.max_latency for t in self._pending.values())
            if expired:
                if self.recovery is not None:
                    try:
                        with self.recovery.synchronized():
                            if not self.recovery.paused:
                                error=ReferenceRequestError('timeout',phase='request',elapsed_ms=self.max_latency*1000)
                                if not self.recovery.transport_error(error):raise RuntimeError('Feedback recovery denied')
                            self.gate.pause('Control HTTP waiting')
                        continue
                    except Exception:
                        self._fail('Physical feedback recovery expired')
                        return
                self._fail('Physical feedback deadline exceeded')
                return

    def _run(self):
        try:
            while not self._closed.is_set():
                try: sequence,body=self.queue.get(timeout=.05)
                except queue.Empty: continue
                stage='send'
                try:
                    send=getattr(self._control_reference,'feedback_request',None)
                    result=(send(body) if callable(send) else
                            self._control_reference._request('POST','/physical-feedback',body))
                    stage='response'
                    if result.get('ok') is not True:raise RuntimeError('Physical feedback rejected')
                    stage='acknowledge'
                    with self._lock:
                        self._pending.pop(sequence,None)
                        if body.get('op')=='begin':self._ready[body['action_id']]=self.clock()
                        if body.get('op')=='intro':self._intro_ack[(body['intro_id'],body['status'])]=self.clock()
                        if body.get('op')=='heartbeat' and 'intro_status' in body:
                            self._intro_ack[(body['intro_id'],body['intro_status'])]=self.clock()
                except ReferenceRequestError as exc:
                    self._fail('Physical feedback unavailable: '+str(exc))
                except Exception as exc:
                    allowed=(RuntimeError,OSError,ValueError,TypeError,KeyError,AttributeError,TimeoutError)
                    details={'stage':stage,
                        'op':body.get('op') if body.get('op') in ('heartbeat','intro','resume','photo_ready','begin') else 'other',
                        'exception_kind':type(exc).__name__ if type(exc) in allowed else 'Other',
                        'recovery_failed':getattr(self.recovery,'failure',None) is not None}
                    self._fail('Physical feedback unavailable [unclassified] '+
                               json.dumps(details,sort_keys=True,separators=(',',':')))
        finally:
            close=getattr(self._control_reference,'close_feedback_transport',None)
            if callable(close):
                try:close()
                except Exception:self._fail('Physical feedback transport cleanup failed')

    def submit(self,body):
        if self.failed or self._closed.is_set():raise RuntimeError(self.failed or 'Physical feedback unavailable')
        if body.get('op') in ('begin','sample','clear'):
            self._submit_presentation(body)
            return
        full=False
        with self._lock:
            self._sequence+=1;sequence=self._sequence
            submitted=self.clock()
            # 只能更换队列中的相同实时/状态脉冲。在途请求保留其原始提交截止日期。
            if body.get('op') in ('heartbeat','intro'):
                with self.queue.mutex:
                    for index,(old_sequence,old_body) in enumerate(self.queue.queue):
                        if old_body==body:
                            self._pending.pop(old_sequence,None)
                            self._pending[sequence]=submitted
                            self.queue.queue[index]=(sequence,dict(body))
                            return
            self._pending[sequence]=submitted
            try:self.queue.put_nowait((sequence,dict(body)))
            except queue.Full:
                self._pending.pop(sequence,None);full=True
        if full:
            self._fail('Physical feedback backlog')
            raise RuntimeError('Physical feedback backlog') from None

    def _submit_presentation(self,body):
        with self._presentation_lock:
            if self._presentation_thread is None:
                self._presentation_thread=threading.Thread(target=self._run_presentation,
                    name='physical-display-sender',daemon=True)
                self._presentation_thread.start()
            # 队列样本只描述该动作的最新实际姿势。切勿在缓慢的UI I/O后队列图像/遥测积压。
            if (body.get('op')=='sample' and self._presentation
                    and self._presentation[-1].get('op')=='sample'
                    and self._presentation[-1].get('action_id')==body.get('action_id')):
                self._presentation[-1]=dict(body)
            else:
                if len(self._presentation)>=4:
                    self._presentation.pop(0)
                    self.presentation_failures=min(2**31-1,self.presentation_failures+1)
                self._presentation.append(dict(body))
        self._presentation_wake.set()

    def _run_presentation(self):
        make=getattr(self.reference,'presentation_transport',None)
        transport=make() if callable(make) else self.reference
        try:
            while not self._closed.is_set():
                with self._presentation_lock:
                    body=self._presentation.pop(0) if self._presentation else None
                    if body is None:self._presentation_wake.clear()
                if body is None:
                    self._presentation_wake.wait(.05);continue
                try:
                    send=getattr(transport,'feedback_request',None) if callable(make) else None
                    result=(send(body) if callable(send) else transport._request('POST','/physical-feedback',body))
                    if result.get('ok') is not True:raise RuntimeError('Display feedback rejected')
                    if body.get('op')=='begin':
                        with self._lock:self._ready[body['action_id']]=self.clock()
                except Exception:
                    # 丢失呈现并不意味着丢失独立维护的拥有者/原生控制流。请勿重播。
                    with self._presentation_lock:
                        self.presentation_failures=min(2**31-1,self.presentation_failures+1)
        finally:
            close=getattr(transport,'close_feedback_transport',None) if callable(make) else None
            if callable(close):
                try:close()
                except Exception:
                    with self._presentation_lock:self.presentation_failures=min(2**31-1,self.presentation_failures+1)

    def ready_after(self,action_id):
        with self._lock:return self._ready.get(action_id)

    def intro_acknowledged(self,intro_id,status):
        with self._lock:return (intro_id,status) in self._intro_ack

    def close(self):
        self._closed.set()
        self._presentation_wake.set()
        for thread in (self._thread,self._watch,self._presentation_thread):
            if thread is None:continue
            thread.join(timeout=1.)
            if thread.is_alive():raise SessionCleanupError('Physical feedback worker did not close')


class LeaseHeartbeat:
    """在过渡过程中，独立地保持相同的后端租约有效。"""
    def __init__(self,publisher, *, interval=.08):
        if type(interval) not in (int,float) or not math.isfinite(interval) or not 0<interval<=.1:
            raise ValueError('Lease heartbeat interval must be at most 100 ms')
        self.publisher=publisher;self.interval=interval;self._intro=None
        self._lock=threading.Lock();self._closed=threading.Event();self.failure=None
        self._thread=threading.Thread(target=self._run,name='physical-lease-heartbeat',daemon=True)
        self._thread.start()
    def set_intro(self,intro_id,status):
        if type(intro_id) is not int or intro_id<=0 or status not in ('active','ready'):
            raise ValueError('Invalid intro status')
        with self._lock:
            if status=='ready' and not self.publisher.intro_acknowledged(intro_id,'active'):
                raise RuntimeError('Intro active status not acknowledged')
            self._intro=(intro_id,status)
    def _run(self):
        while not self._closed.is_set():
            try:
                with self._lock:intro=self._intro
                body={'op':'heartbeat'}
                if intro:body.update(intro_id=intro[0],intro_status=intro[1])
                self.publisher.submit(body)
            except Exception:
                self.failure='Physical lease heartbeat failed';self._closed.set();return
            self._closed.wait(self.interval)
    def close(self):
        self._closed.set()
        self._thread.join(timeout=1.)
        if self._thread.is_alive():raise SessionCleanupError('Physical heartbeat did not close')


class PhysicalRuntime:
    """序列化监控；源和HTTP发送器独立运行。"""
    def __init__(self,session,supervisor,profile,dashboard,publisher, *, control,external_heartbeat=False,recovery=None):
        if (not isinstance(control,dict) or not isinstance(control.get('instance'),str)
                or type(control.get('epoch')) is not int or type(control.get('started_ns')) is not int
                or control['started_ns']<=0):
            raise ValueError('Active guest identity required')
        self.session,self.supervisor,self.profile=session,supervisor,profile
        self.dashboard,self.publisher,self.control=dashboard,publisher,dict(control)
        self.last_writer=self.last_reader=None
        self.last_heartbeat=None
        self.external_heartbeat=external_heartbeat
        self.last_light=None
        self.action=None
        self.action_id=0
        self.unsupported_intro=False
        self.recovery=recovery
        self._network_started=self._network_still_since=None
        self._network_last_stamps=None
        self._network_resume_job=None
        self._network_frame_floor=0
        self.network_paused_seconds=0.
        self.clock=time.monotonic
        self.supervisor.bind_control_identity(self.control)
        self.supervisor.set_move_readiness(self._light_ready)

    def _light_ready(self,cue,source_ns):
        action=self.action
        return bool(action and action['cue']==cue and action['source_ns']==source_ns
                    and not self.publisher.failed)

    def _fail(self,reason):
        self.session.gate.stop(str(reason))
        self.supervisor.stop()
        raise RuntimeError(str(reason))

    def _sample(self,snapshot,now):
        value,stamp=snapshot
        if (not isinstance(value,dict) or type(stamp) not in (int,float) or not math.isfinite(stamp)
                or not 0<=now-stamp<=.15):raise RuntimeError('Robot telemetry stale')
        return value,stamp

    def _start_light(self,event):
        cue=event['cue'];point=self.profile.paths[cue][-1]
        origin=self.profile.cfg.neutral[:3]
        delta=[a-b for a,b in zip(point,origin)]
        distance=math.hypot(*delta)
        if not math.isfinite(distance) or distance<=0:raise ValueError('Invalid light reference')
        self.action_id+=1
        self.action={'action_id':self.action_id,'source_ns':event['source_monotonic_ns'],
                     'cue':cue,'last_sample':None,'motion_started':False,'direction':[x/distance for x in delta]}
        self.publisher.submit({'op':'begin',**self.control,'action_id':self.action_id,
            'source_ns':event['source_monotonic_ns'],'cue':cue,'reference_position':list(origin),
            'direction':[x/distance for x in delta],'planned_displacement':distance})

    def _network_hold(self,writer,ws,reader,rs,now,advancing):
        """在语义工作暂停的同时，泵出真正的本地HOLD ACKs。"""
        if self.recovery is None or not self.recovery.paused:
            return False
        if self._network_started is None:
            self._network_started=now
            pause=getattr(self.supervisor,'pause_for_network',None)
            if callable(pause):pause(now)
        self.session.gate.pause('Waiting for same-encounter communication')
        if advancing:
            self.session.engine.tick(writer,sampled_at=ws,now=now)
            self.last_writer,self.last_reader=ws,rs
            if (self._network_last_stamps is not None and
                    any(new-old>.15 for new,old in zip((ws,rs),self._network_last_stamps))):
                self._network_still_since=None
            self._network_last_stamps=(ws,rs)
            qd=reader.get('joint_speed');speed=reader.get('tcp_speed')
            still=(writer.get('state')==4 and isinstance(qd,(list,tuple)) and len(qd)==6
                   and all(type(x) in (int,float) and math.isfinite(x) for x in qd)
                   and max(map(abs,qd))<.003 and isinstance(speed,(list,tuple)) and len(speed)==6
                   and all(type(x) in (int,float) and math.isfinite(x) for x in speed)
                   and math.hypot(*speed[:3])<.0005 and math.hypot(*speed[3:])<.002
                   and math.hypot(*writer['tcp_speed'][:3])<.0005
                   and math.hypot(*writer['tcp_speed'][3:])<.002)
            if not still:self._network_still_since=None
            elif self._network_still_since is None:self._network_still_since=min(ws,rs)
        if not self.recovery.ready:
            return True
        if (self._network_still_since is None or min(ws,rs)-self._network_still_since<.3):
            return True
        if self._network_resume_job is None:
            # HTTP确认可能需要等待；它绝不能阻止原生ACK。
            job=SimpleNamespace(done=threading.Event(),value=None,error=None)
            def resume_server():
                try:job.value=self.supervisor.reference.resume_communication()
                except Exception as exc:job.error=exc
                finally:job.done.set()
            self._network_resume_job=job
            threading.Thread(target=resume_server,name='physical-network-resume',daemon=True).start()
        job=self._network_resume_job
        if not job.done.is_set():return True
        if job.error is not None:raise RuntimeError('Communication resume rejected') from job.error
        if job.value is not True:raise RuntimeError('Communication resume not acknowledged')
        # 在监控程序中的新丢失/停止情况下完成序列化。
        self.recovery.complete(resume=getattr(self.recovery,'resume_gate',self.session.gate.resume))
        self._network_frame_floor=round(now*1e9)
        resume=getattr(self.supervisor,'resume_after_network',None)
        if callable(resume):resume(now,self._network_frame_floor)
        self.network_paused_seconds+=now-self._network_started
        self._network_started=self._network_still_since=None
        self._network_last_stamps=None
        self._network_resume_job=None
        return True

    def close_recovery(self):
        job=self._network_resume_job
        if job is not None and not job.done.wait(1.):
            raise SessionCleanupError('Communication resume worker did not close')

    def step(self,event, *, now=None):
        admitted=None
        try:
            # 在读取本地时钟之前，对每个生产者进行快照。
            # 否则，同时更新的、完全新的样本可能会看起来来自未来，并虚假地故障运行中的居民。
            state_snapshot=self.session.states.latest
            state_failure=self.session.states.failure
            writer,ws=self.session.feedback
            reader_snapshot=self.session.reader.latest
            reader_failure=self.session.reader.failure
            dashboard_failure=self.dashboard.failure
            dashboard_checked=self.dashboard.checked_at
            if now is None:now=self.clock()
            if type(now) not in (int,float) or not math.isfinite(now):raise RuntimeError('Invalid clock')
            if not self.session.gate.can_send:
                details=[]
                watchdog_failure=getattr(getattr(self.session,'watch',None),'failure',None)
                gate_reason=getattr(self.session.gate,'reason',None)
                if watchdog_failure:details.append('watchdog='+str(watchdog_failure))
                if gate_reason:details.append('gate='+str(gate_reason))
                raise RuntimeError('Motion or guest watchdog stopped'+(': '+'; '.join(details) if details else ''))
            if self.publisher.failed:raise RuntimeError('Physical feedback unavailable')
            if (dashboard_failure or type(dashboard_checked) not in (int,float)
                    or not 0<=now-dashboard_checked<=2):
                raise RuntimeError('Dashboard status stale or unsafe')
            mode=_state_snapshot(SimpleNamespace(latest=state_snapshot,failure=state_failure),self.profile.cfg,now)
            if mode.get('running') is not True:raise RuntimeError('Resident program no longer running')
            if (not isinstance(writer,dict) or type(ws) not in (int,float) or not math.isfinite(ws)
                    or not 0<=now-ws<=.15):raise RuntimeError('Writer feedback stale')
            if reader_failure:raise RuntimeError('Robot telemetry connection failed')
            reader,rs=self._sample(reader_snapshot,now)
            if not self.external_heartbeat and (self.last_heartbeat is None or now-self.last_heartbeat>=.1):
                self.publisher.submit({'op':'heartbeat',**self.control})
                self.last_heartbeat=now
            advancing=(self.last_writer is None or ws>self.last_writer) and (self.last_reader is None or rs>self.last_reader)
            if self._network_hold(writer,ws,reader,rs,now,advancing):return None
            if advancing:
                self.supervisor.tick(writer,sampled_at=ws,joint_feedback=reader,joint_sampled_at=rs,now=now)
                self.last_writer,self.last_reader=ws,rs
            if self.action and (self.supervisor.state in ('OBSERVING','PHOTO_PENDING','PHOTO_READY')
                                or getattr(self.supervisor,'recovering_expiry',False)):
                # 当原始参考仍在验证时，过期的、从未启用的方向将被取消。不要在HOLD期间将被取消的提示作为移动/稳定动作呈现。
                self.publisher.submit({'op':'clear','action_id':self.action['action_id']})
                self.action=None
            if self.action and self.supervisor.state in ('MOVING','SETTLING','SAMPLE_PENDING','RELEASE_PENDING'):
                if ((self.last_light is None or now-self.last_light>=.1)
                        and (self.action['last_sample'] is None or ws>self.action['last_sample'])):
                    velocity=sum(a*b for a,b in zip(writer['tcp_speed'][:3],self.action['direction']))
                    # 仅限呈现证据：位置漂移或分阶段方向绝不能宣布运动。这个目标必须被发送、确认并观察其向外移动。
                    engine=self.session.engine
                    if (writer.get('state')==1 and getattr(engine,'motion_sent',False) is True
                            and getattr(engine,'motion_acked',False) is True and velocity>.0001):
                        self.action['motion_started']=True
                    phase=('settling' if self.supervisor.state!='MOVING' else
                           'returning' if velocity<-.0001 else 'moving')
                    self.publisher.submit({'op':'sample',**self.action,'captured_at':ws,
                                           'current_position':writer['tcp_pose'][:3],'phase':phase})
                    self.action['last_sample']=ws;self.last_light=now
            if (isinstance(event,dict) and
                    (not self._network_frame_floor or
                     type(event.get('source_monotonic_ns')) is int and
                     event['source_monotonic_ns']>self._network_frame_floor)):
                if event.get('event')=='physical_intro':
                    # 现有的居民不会轮换。永远不要宣传没有发生过的完成或活跃的倾斜。
                    self.unsupported_intro=True
                elif (event.get('event')=='person_guidance'
                      and event.get('control_instance')==self.control['instance']
                      and event.get('control_epoch')==self.control['epoch']
                      and type(event.get('source_monotonic_ns')) is int
                      and event['source_monotonic_ns']>self.control['started_ns']):
                    accepted=self.supervisor.accept_event(event,now=now)
                    if accepted and event.get('cue','').startswith('MOVE_'):
                        self._start_light(event)
                        admitted=(self.control['instance'],self.control['epoch'],self.control['started_ns'],
                                  event['source_monotonic_ns'])
        except Exception as exc:
            self._fail(str(exc))
        return admitted


def _step_source_event(runtime,source,event):
    admitted=runtime.step(event)
    acknowledge=getattr(source,'acknowledge',None)
    if isinstance(admitted,tuple) and len(admitted)==4 and callable(acknowledge):acknowledge(admitted)
    return admitted


class LatestEventSource:
    """最新的状态加上一个新的、明确未承认的方向意图。"""
    _MOVES=frozenset(('MOVE_LEFT','MOVE_RIGHT','MOVE_UP','MOVE_DOWN','MOVE_CLOSER','MOVE_FARTHER'))
    def __init__(self,stream,*,clock=time.monotonic):
        self.stream=stream;self.lock=threading.Lock();self.latest=None;self.failure=None
        self.clock=clock;self._control=None;self._pending=None;self._floor_ns=0;self._scope_invalid=False
        self.thread=threading.Thread(target=self._read,name='physical-source',daemon=True);self.thread.start()
    def bind_control(self,control):
        with self.lock:
            self.latest=self._pending=None;self._control=None;self._floor_ns=0;self._scope_invalid=False
            if (not isinstance(control,dict) or not isinstance(control.get('instance'),str)
                    or not 1<=len(control['instance'])<=128 or type(control.get('epoch')) is not int
                    or not 0<=control['epoch']<2**53 or type(control.get('started_ns')) is not int
                    or control['started_ns']<=0):raise ValueError('Explicit active source control scope required')
            self._control=(control['instance'],control['epoch'],control['started_ns'])
    def unbind_control(self):
        with self.lock:
            self.latest=self._pending=None;self._control=None;self._floor_ns=0;self._scope_invalid=False
    def _scope_matches(self,value):
        return (self._control is not None and value.get('control_instance')==self._control[0]
                and type(value.get('control_epoch')) is int and value['control_epoch']==self._control[1])
    def _fresh(self,value,now):
        ns=value.get('source_monotonic_ns')
        return (self._control is not None and type(ns) is int and ns>self._control[2]
                and type(now) in (int,float) and math.isfinite(now) and 0<=now-ns/1e9<=.3)
    def _direction(self,value):
        return (isinstance(value,dict) and value.get('event')=='person_guidance'
                and isinstance(value.get('cue'),str) and value['cue'] in self._MOVES)
    def _eligible(self,value,now):
        return (self._direction(value) and not self._scope_invalid and self._scope_matches(value)
                and value.get('coordinate_frame')=='normalized_camera_image' and value.get('robot_actuated') is False
                and value.get('tracking') is True and self._fresh(value,now)
                and value['source_monotonic_ns']>self._floor_ns)
    def _clear_pending(self,barrier=None):
        if self._pending is not None:self._floor_ns=max(self._floor_ns,self._pending['source_monotonic_ns'])
        if type(barrier) is int:self._floor_ns=max(self._floor_ns,barrier)
        self._pending=None
    def _ingest(self,value):
        with self.lock:
            if self._control is None:self.latest=value;return
            now=self.clock();ns=value.get('source_monotonic_ns')
            if value.get('event') in ('person_guidance','physical_intro') and not self._scope_matches(value):
                newest=self._floor_ns
                if self._pending is not None:newest=max(newest,self._pending['source_monotonic_ns'])
                if isinstance(self.latest,dict) and self._scope_matches(self.latest) and type(self.latest.get('source_monotonic_ns')) is int:
                    newest=max(newest,self.latest['source_monotonic_ns'])
                # 序列化的旧轮次积压不是当前遇到的变化的证据。只有一个新的、更新的范围才能使此绑定无效；
                # 它永远不会重新绑定到该范围。
                if not self._fresh(value,now) or ns<=newest:return
                self.latest=value
                self._clear_pending();self._scope_invalid=True;return
            self.latest=value
            if self._eligible(value,now):
                if self._pending is None or ns>self._pending['source_monotonic_ns']:
                    self._pending=copy.deepcopy(value)
                return
            if self._pending is not None and type(ns) is int and ns<self._pending['source_monotonic_ns']:return
            # 只有真正的后提示响应窗口状态才能保留意图。
            # HOLD/loss、target/photo、AI-error和Stop记录取代了它。
            preserve=(self._scope_matches(value) and self._fresh(value,now)
                and value.get('event')=='person_guidance' and value.get('tracking') is True
                and value.get('coordinate_frame')=='normalized_camera_image' and value.get('robot_actuated') is False
                and value.get('cue')=='WAIT' and value.get('phase')=='WAITING_FOR_RESPONSE'
                and value.get('reason')=='WAITING_FOR_RESPONSE')
            if not preserve:self._clear_pending(ns if self._fresh(value,now) else None)
    def acknowledge(self,identity):
        with self.lock:
            if (not isinstance(identity,tuple) or len(identity)!=4 or identity[:3]!=self._control
                    or type(identity[3]) is not int):return
            ns=identity[3];self._floor_ns=max(self._floor_ns,ns)
            if self._pending is not None and self._pending['source_monotonic_ns']==ns:self._pending=None
            if self._direction(self.latest) and type(self.latest.get('source_monotonic_ns')) is int and self.latest['source_monotonic_ns']<=ns:
                self.latest=None
    def _read(self):
        try:
            while True:
                line=self.stream.readline(8193)
                if not line:raise RuntimeError('Guidance source ended')
                if len(line)>8192 or not line.endswith('\n'):raise ValueError('Invalid guidance input')
                value=json.loads(line)
                if not isinstance(value,dict):raise ValueError('Invalid guidance event')
                self._ingest(value)
        except Exception:
            with self.lock:self.latest=self._pending=None;self.failure='Guidance source unavailable'
    def take(self):
        with self.lock:
            if self.failure:return None
            now=self.clock()
            if self._pending is not None:
                if self._eligible(self._pending,now):return copy.deepcopy(self._pending)
                self._clear_pending()
            value=self.latest;self.latest=None
            if self._control is not None and self._direction(value):return None
            return value


# 对一次性CLI和现有呼叫者的兼容性。启动程序在相机管道生命周期内，在所有遇到的情况下，拥有一个确切的源。
_Source=LatestEventSource


def wait_for_guest(read_guest,source, *, timeout,clock=time.monotonic,sleep=time.sleep):
    """在未连接到机器人之前等待，直到访客启动 iPad。"""
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Positive guest wait timeout required')
    deadline=clock()+timeout;identity=None;seen_active=False
    while clock()<deadline:
        if source.failure:raise RuntimeError(source.failure)
        snapshot=read_guest()
        if (not isinstance(snapshot,dict) or not isinstance(snapshot.get('instance'),str)
                or type(snapshot.get('epoch')) is not int):
            raise RuntimeError('Guest control unavailable')
        current=(snapshot['instance'],snapshot['epoch'])
        if identity is not None and current!=identity:
            # 准备/重新连接相机会增加未激活的周期。还没有连接任何机器人。只有在开始后才能绑定；
            # 稍后的停止或不同的服务器实例仍然会中止此启动。
            if seen_active or current[0]!=identity[0]:
                raise RuntimeError('Guest session changed while waiting')
        identity=current
        if snapshot.get('active') is True:
            seen_active=True
            if snapshot.get('camera_connected') is True and snapshot.get('camera_fresh') is True:
                return snapshot
        sleep(.05)
    raise RuntimeError('Guest startup timed out')


def _check_service_stop(should_stop,switch):
    try:requested=should_stop()
    except Exception:
        switch.stop('Physical service cancellation check failed')
        raise
    if requested:
        switch.stop('Physical service stop requested')
        raise RuntimeError('Physical service stop requested')


def wait_for_trigger(source,switch, *, observed_at,timeout,expected_action,control,clock=time.monotonic,sleep=time.sleep,
                     should_stop=lambda:False):
    """需要一个新的激活后事件；介绍心跳可能会取代“开始”。"""
    if (any(type(x) not in (int,float) or not math.isfinite(x) for x in (observed_at,timeout))
            or timeout<=0 or expected_action not in ('start','restart') or not isinstance(control,dict)
            or not isinstance(control.get('instance'),str) or type(control.get('epoch')) is not int
            or type(control.get('started_ns')) is not int or control['started_ns']<=0):
        raise ValueError('Valid activation time and trigger timeout required')
    deadline=clock()+timeout
    while clock()<deadline:
        _check_service_stop(should_stop,switch)
        if switch.failure:raise RuntimeError(switch.failure)
        if source.failure:raise RuntimeError(source.failure)
        event=source.take();now=clock()
        _check_service_stop(should_stop,switch)
        if isinstance(event,dict):
            ns=event.get('source_monotonic_ns')
            fresh=type(ns) is int and ns>control['started_ns'] and ns/1e9>observed_at and 0<=now-ns/1e9<=.3
            intro=(type(ns) is int and event.get('event')=='physical_intro' and event.get('action') in ('start','heartbeat')
                   and type(event.get('intro_id')) is int and control['started_ns']<event['intro_id']<=ns
                   and event.get('control_instance')==control.get('instance')
                   and event.get('control_epoch')==control.get('epoch'))
            target=(event.get('event')=='person_guidance' and event.get('tracking') is True
                    and event.get('control_instance')==control.get('instance')
                    and event.get('control_epoch')==control.get('epoch'))
            if fresh and ((expected_action=='start' and intro) or (expected_action=='restart' and target)):
                return event
        sleep(.005)
    raise RuntimeError('Guest start trigger timed out')


def register_backend(port,key,control, *, timeout,factory=LocalReferenceGate,mode='guidance',recovery_window_s=None):
    """获取单个访客纪元租约，而不在输出中暴露它。"""
    provisional=factory(port,key,timeout=timeout,control=control,lease='registration')
    if mode not in ('intro','guidance'):raise ValueError('Invalid physical backend mode')
    body={'op':'register'}
    if mode=='intro':body['mode']='intro'
    if recovery_window_s is not None:body['recovery_window_s']=recovery_window_s
    result=provisional._request('POST','/physical-feedback',body)
    lease=result.get('lease') if isinstance(result,dict) and result.get('ok') is True else None
    if not isinstance(lease,str) or not 1<=len(lease)<=512:
        raise RuntimeError('Physical backend registration failed')
    return factory(port,key,timeout=timeout,control=control,lease=lease)


def wait_for_intro_camera(reader,control,check_source, *, timeout=None,clock=time.monotonic,sleep=time.sleep):
    """经过验证的回拨后，在控制有效期间，等待实际图像。可选的呼叫者超时仍然可用。
    在没有本地程序运行的情况下，协商的会面使用其现有的固定通信截止日期。
    准备状态仍然暂停：下一个居民必须证明本地HOLD才能恢复。"""
    if timeout is not None and (type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0):
        raise ValueError('Positive camera wait timeout required')
    deadline=None if timeout is None else clock()+timeout
    recovery=getattr(reader,'communication_recovery',None)
    while deadline is None or clock()<deadline:
        if recovery is None:check_source()
        else:recovery.status
        try:value=reader()
        except Exception as exc:
            if recovery is None or not recovery.transport_error(exc):raise
            check_source()
            sleep(.02)
            continue
        if recovery is not None:
            # 先进入共享暂停状态，再检查数据源时效；在这一间隔内，旧开场流程可能已经正常停止出帧。
            recovery.observe(value)
            check_source()
        if (not isinstance(value,dict) or value.get('active') is not True or
                recovery is None and value.get('control_fresh') is not True or
                any(value.get(k)!=control[k] for k in ('instance','epoch','started_ns'))):
            raise RuntimeError('Intro control lost while waiting for camera at reference')
        if (value.get('control_fresh') is True and value.get('camera_connected') is True
                and value.get('camera_fresh') is True
                and (recovery is None or recovery.status in ('healthy','ready'))):return
        sleep(.02)
    raise RuntimeError('Fresh camera unavailable after intro return')


def run_intro_then_open_xyz(profile,reader,source,lease_heartbeat,switch, *, control,
                            intro_event,reference,publisher,clock=time.monotonic,sleep=time.sleep,
                            intro_opener=open_intro,xyz_opener=open_resident,
                            dashboard_factory=DashboardReceiver,runtime_factory=PhysicalRuntime,
                            should_stop=lambda:False):
    """运行已审查的介绍，验证退货/关闭，并在准备好之前暂停XYZ。所有提供的设备都可能是假的。这尚未在实际手臂上调试；
    成功的软件退货并不能证明碰撞或停止。"""
    intro_id=intro_event.get('intro_id') if isinstance(intro_event,dict) else None
    now=clock()
    if (not isinstance(control,dict) or not isinstance(control.get('instance'),str)
            or type(control.get('epoch')) is not int or type(control.get('started_ns')) is not int
            or type(now) not in (int,float) or not math.isfinite(now)
            or not isinstance(intro_event,dict) or intro_event.get('event')!='physical_intro'
            or intro_event.get('action') not in ('start','heartbeat')
            or intro_event.get('control_instance')!=control['instance']
            or intro_event.get('control_epoch')!=control['epoch']
            or type(intro_id) is not int or intro_id<=control['started_ns']
            or type(intro_event.get('source_monotonic_ns')) is not int
            or intro_id>intro_event['source_monotonic_ns']
            or not 0<=now-intro_event['source_monotonic_ns']/1e9<=.3):
        raise ValueError('Fresh physical intro identity required')
    intro=session=dashboard=None
    last_intro=None;last_ws=last_rs=None;active_set=False
    recovery=getattr(reader,'communication_recovery',None)
    network_started=network_still=None;network_job=None;network_stamps=None

    def hold_intro_if_needed():
        nonlocal network_started,network_still,network_job,network_stamps
        if recovery is None or not recovery.paused:return
        now=clock()
        if network_started is None:network_started=now
        intro.gate.pause('Opening communication waiting')
        fb,ws=intro.writer.latest;independent,rs=intro.reader.latest
        if (not isinstance(fb,dict) or not isinstance(independent,dict)
                or not all(0<=now-stamp<=.15 for stamp in (ws,rs))):
            raise RuntimeError('Opening recovery telemetry stale')
        if (network_stamps is not None and
                any(new-old>.15 for new,old in zip((ws,rs),network_stamps))):network_still=None
        network_stamps=(ws,rs)
        speed=independent.get('tcp_speed');qd=independent.get('joint_speed')
        still=(fb.get('state') in (103,102) and isinstance(speed,(list,tuple)) and len(speed)==6
               and isinstance(qd,(list,tuple)) and len(qd)==6
               and all(type(x) in (int,float) and math.isfinite(x) for x in (*speed,*qd))
               and max(map(abs,qd))<.003 and math.hypot(*speed[:3])<.0005
               and math.hypot(*speed[3:])<.002)
        if not still:network_still=None
        elif network_still is None:network_still=min(ws,rs)
        if (not recovery.ready or network_still is None or min(ws,rs)-network_still<.3
                or last_intro is None or not 0<=now-last_intro<=.3):return
        if network_job is None:
            job=SimpleNamespace(done=threading.Event(),value=None,error=None)
            def request_resume():
                try:job.value=reference.resume_communication()
                except Exception as exc:job.error=exc
                finally:job.done.set()
            network_job=job
            threading.Thread(target=request_resume,name='intro-network-resume',daemon=True).start()
        if not network_job.done.is_set():return
        if network_job.error is not None:raise RuntimeError('Opening communication resume failed') from network_job.error
        if network_job.value is not True:raise RuntimeError('Opening resume not acknowledged')
        recovery.complete(resume=getattr(recovery,'resume_gate',intro.gate.resume))
        network_started=network_still=network_job=None
        network_stamps=None

    def check_source(*,allow_guidance=False,allow_initial=False):
        nonlocal last_intro
        _check_service_stop(should_stop,switch)
        for channel,failure in (('source',source.failure),('control',switch.failure),
                                ('lease',lease_heartbeat.failure),('feedback',publisher.failed)):
            if failure:raise RuntimeError(f'Intro {channel} stopped: {failure}')
        event=source.take();now=clock()
        _check_service_stop(should_stop,switch)
        if event is not None:
            valid_intro=(isinstance(event,dict) and event.get('event')=='physical_intro'
                         and event.get('intro_id')==intro_id and event.get('action') in ('start','heartbeat')
                         and event.get('control_instance')==control.get('instance')
                         and event.get('control_epoch')==control.get('epoch'))
            waiting=recovery is not None and recovery.paused
            valid_guidance=((allow_guidance or waiting and isinstance(event,dict) and event.get('cue')=='HOLD')
                            and isinstance(event,dict) and event.get('event')=='person_guidance'
                            and event.get('control_instance')==control.get('instance')
                            and event.get('control_epoch')==control.get('epoch'))
            if not valid_intro and not valid_guidance:
                raise RuntimeError('Intro was interrupted')
            ns=event.get('source_monotonic_ns')
            if type(ns) is not int or ns<=control['started_ns'] or now<ns/1e9:
                raise RuntimeError('Intro heartbeat stale')
            if now-ns/1e9>.3:
                if waiting:return None,now
                raise RuntimeError('Intro heartbeat stale')
            if valid_intro or allow_guidance:last_intro=ns/1e9
        if last_intro is None and allow_initial:
            return event,now
        if last_intro is None or not 0<=now-last_intro<=.3:
            if recovery is not None and recovery.paused:return event,now
            raise RuntimeError('Intro heartbeat unavailable or stale')
        return event,now

    try:
        _check_service_stop(should_stop,switch)
        intro=intro_opener(profile.host,profile.intro,reader,require_camera=False)
        _check_service_stop(should_stop,switch)
        switch.attach(intro.gate)
        print('PHYSICAL_INTRO_CONNECTED: held at reference; awaiting fresh iPad control event.',flush=True)
        # 打开真实 RTDE 接口可能耗时数秒。初始化完毕后，须收到新的源心跳才能首次使能。
        deadline=clock()+.3
        while last_intro is None:
            check_source(allow_initial=True)
            if recovery is not None and recovery.paused:
                intro.gate.pause('Opening startup communication waiting')
                ws=intro.writer.latest[1];rs=intro.reader.latest[1]
                if ((last_ws is None or ws>last_ws) and (last_rs is None or rs>last_rs)):
                    intro.tick()
                    last_ws,last_rs=intro.progress.last_writer,intro.progress.last_reader
                hold_intro_if_needed()
            elif clock()>deadline:raise RuntimeError('Fresh intro heartbeat unavailable after startup')
            if last_intro is None:sleep(.005)
        intro_tick_at=clock()
        while intro.progress.state!='COMPLETE' or recovery is not None and recovery.paused:
            check_source()
            writer,ws=intro.writer.latest;independent,rs=intro.reader.latest
            now=clock()
            if (intro.writer.failure or intro.reader.failure or not isinstance(writer,dict)
                    or not isinstance(independent,dict) or any(type(x) not in (int,float)
                    or not math.isfinite(x) or not 0<=now-x<=.15 for x in (ws,rs))):
                raise RuntimeError('Intro independent telemetry stale')
            if (last_ws is None or ws>last_ws) and (last_rs is None or rs>last_rs):
                _check_service_stop(should_stop,switch)
                tick_now=clock()
                if recovery is not None and recovery.paused and intro.progress.started is not None:
                    intro.progress.started+=max(0.,tick_now-intro_tick_at)
                intro_tick_at=tick_now
                if intro.progress.state!='COMPLETE':intro.tick()
                # Intro.tick 自行读取快照。若调用期间生产端推进，以它实际消费的时间戳为准。
                last_ws,last_rs=intro.progress.last_writer,intro.progress.last_reader
                if intro.progress.started_ack and not active_set:
                    lease_heartbeat.set_intro(intro_id,'active');active_set=True
                    print('PHYSICAL_INTRO_ACTIVE: robot acknowledged opening motion.',flush=True)
            hold_intro_if_needed()
            sleep(.005)
        _check_service_stop(should_stop,switch)
        if not active_set:raise RuntimeError('Intro completed without an active acknowledgement')
        print('PHYSICAL_INTRO_COMPLETE: stationary return independently confirmed.',flush=True)
        deadline=clock()+.3
        while not publisher.intro_acknowledged(intro_id,'active'):
            check_source()
            if clock()>deadline:raise RuntimeError('Intro active status not acknowledged')
            sleep(.005)
        # 开场程序按审核路径返回后会自行退出。等待写入端关闭，间隔期间保留同一访客租约。
        switch.detach_completed(intro.gate,intro.begin_completed_retirement)
        _close_motion_session(intro);intro=None
        if switch.failure:raise RuntimeError(switch.failure)
        wait_for_intro_camera(reader,control,check_source,clock=clock,sleep=sleep)
        session=xyz_opener(profile.host,profile.cfg,reader,require_camera=False)
        _check_service_stop(should_stop,switch)
        switch.attach(session.gate)
        dashboard=dashboard_factory(profile.host)
        supervisor=profile.supervisor(session.engine,reference)
        options={} if recovery is None else {'recovery':recovery}
        runtime=runtime_factory(session,supervisor,profile,dashboard,publisher,
                                control=control,external_heartbeat=True,**options)
        deadline=clock()+3.
        while session.engine.state!='READY':
            event,now=check_source()
            _step_source_event(runtime,source,event)
            if (now>deadline+getattr(runtime,'network_paused_seconds',0.)
                    and not (recovery is not None and recovery.paused)):
                raise RuntimeError('XYZ resident hold acknowledgement timed out')
            sleep(.005)
        _check_service_stop(should_stop,switch)
        lease_heartbeat.set_intro(intro_id,'ready')
        deadline=clock()+.3
        while not publisher.intro_acknowledged(intro_id,'ready'):
            event,now=check_source(allow_guidance=True)
            # 收到就绪确认后应用可能立即发出视觉引导；处理这些引导前，先完成就绪确认的交接。
            _step_source_event(runtime,source,None if event and event.get('event')=='person_guidance' else event)
            if (now>deadline+getattr(runtime,'network_paused_seconds',0.)
                    and not (recovery is not None and recovery.paused)):
                raise RuntimeError('Intro ready status not acknowledged')
            sleep(.005)
        _check_service_stop(should_stop,switch)
        print('PHYSICAL_GUIDANCE_READY: opening complete; accepting visual guidance.',flush=True)
        return session,runtime,dashboard
    except Exception as exc:
        switch.stop(str(exc))
        if recovery is not None:
            try:recovery.fail(str(exc))
            except RuntimeError:pass
        cleanup_errors=[]
        if network_job is not None and not network_job.done.wait(1.):
            cleanup_errors.append(RuntimeError('Opening resume worker did not close'))
        if intro is not None:
            try:_close_motion_session(intro)
            except Exception as error:cleanup_errors.append(error)
        if session is not None:
            try:_close_motion_session(session)
            except Exception as error:cleanup_errors.append(error)
        if dashboard is not None:
            try:dashboard.close()
            except Exception:pass
        if cleanup_errors:raise SessionCleanupError('Physical transition cleanup incomplete') from cleanup_errors[0]
        raise


def _close_motion_session(session, *, completed=False):
    if completed:session.close_completed()
    else:session.close()
    gate=session.gate
    finished=getattr(gate,'close_finished',None)
    if finished is not None and (not finished.wait(2.) or gate.close_error):
        raise SessionCleanupError('Physical input channel did not close cleanly')
    for thread in getattr(getattr(session,'watch',None),'_threads',()):
        thread.join(timeout=1.)
        if thread.is_alive():raise SessionCleanupError('Physical control reader did not close')


def wait_initial_communication(reader,reference,recovery,switch, *, sleep=time.sleep,check=lambda:None):
    """在打开任何原生程序之前，完成有限的拥有者等待。"""
    if switch.attached:raise RuntimeError('Initial communication wait is only before native upload')
    while True:
        check()
        try:reader()
        except Exception as exc:
            if not recovery.transport_error(exc):raise
        if not recovery.paused:return
        if recovery.ready:
            if reference.resume_communication() is not True:raise RuntimeError('Initial communication resume rejected')
            recovery.complete(resume=switch.resume)
            return
        sleep(.02)


def run_session(profile,port,key,source,reader, *, before,report=None,should_stop=lambda:False,
                 dependencies=None,allow_reference_rearm=False,lifecycle=None):
    """运行一个明确的客机纪元，并同步退出其工作者。调用者拥有持久化的相机/源。如果发生故障，不会发出返回、重置或重试。
    新会话仍然使用不变的参考检查。"""
    factories={} if dependencies is None else dependencies
    intro_runner=factories.get('intro_runner',run_intro_then_open_xyz)
    xyz_opener=factories.get('xyz_opener',open_resident)
    dashboard_factory=factories.get('dashboard_factory',DashboardReceiver)
    observed_at=time.monotonic()
    control={'instance':before['instance'],'epoch':before['epoch'],'started_ns':before['started_ns']}
    session=dashboard=publisher=reference=heartbeat=runtime=None
    switch=SwitchingGate()
    recovery=None
    if getattr(getattr(profile,'cfg',None),'gesture',None) is not None:
        from .network_recovery import CommunicationRecovery
        recovery=CommunicationRecovery(control)
        recovery.resume_gate=switch.resume
    def session_reader():
        # 初始化和运行阶段的独立看门狗也监测退出，即使主线程阻塞在打开连接或读取遥测的步骤。
        _check_service_stop(should_stop,switch)
        value=reader()
        if recovery is not None:
            with recovery.synchronized():
                recovery.observe(value)
                if recovery.paused:switch.pause('Waiting for communication')
        _check_service_stop(should_stop,switch)
        return value
    close_reader_thread=getattr(reader,'close_current_thread',None)
    if callable(close_reader_thread):session_reader.close_current_thread=close_reader_thread
    if recovery is not None:session_reader.communication_recovery=recovery
    try:
        bind_source=getattr(source,'bind_control',None)
        if callable(bind_source):bind_source(control)
        _check_service_stop(should_stop,switch)
        registration={}
        if recovery is not None:
            from .network_reference import RecoveryReferenceGate
            registration={'factory':lambda *a,**kw:RecoveryReferenceGate(*a,recovery=recovery,**kw),
                          'recovery_window_s':3.0}
        reference=register_backend(port,key,control,
        timeout=min(profile.gate_timeout,.25 if getattr(getattr(profile,'cfg',None),'gesture',None) is not None else .1),
                                   mode='intro' if before.get('control_action')=='start' else 'guidance',**registration)
        publisher=FeedbackPublisher(reference,switch,**({} if recovery is None else {'recovery':recovery}))
        heartbeat=LeaseHeartbeat(publisher)
        gate_state=reference.status(None)
        restart_session=None
        if gate_state.get('blocked') is not False:
            if not allow_reference_rearm or before.get('control_action')!='start':
                raise RuntimeError('Existing reference gate remains blocked; no automatic reset')
            restart_session=gate_state.get('session')
            if not isinstance(restart_session,str) or not restart_session:
                raise RuntimeError('Previous reference session unavailable for explicit Start')
        elif lifecycle is not None:
            lifecycle['reference_owned']=True
        if recovery is not None:
            wait_initial_communication(session_reader,reference,recovery,switch,
                check=lambda:_check_service_stop(should_stop,switch))
        trigger=wait_for_trigger(source,switch,observed_at=observed_at,timeout=15.,
                                 expected_action=before.get('control_action'),control=control,
                                 should_stop=should_stop)
        _check_service_stop(should_stop,switch)
        bind_reference=getattr(profile,'bind_before_motion',None)
        if callable(bind_reference) or restart_session is not None:
            def capture_stopped():
                _check_service_stop(should_stop,switch)
                for name,failure in (('source',source.failure),('lease',heartbeat.failure),
                                     ('feedback',getattr(publisher,'failed',None))):
                    if failure:raise RuntimeError(f'Reference capture {name} stopped: {failure}')
                # 此时尚未上传程序。控制短暂中断时等待有效的所有者状态，不伪造新的开始请求。
                while recovery is not None and recovery.paused and not switch.attached:
                    _check_service_stop(should_stop,switch)
                    try:session_reader()
                    except Exception as exc:
                        if not recovery.transport_error(exc):raise
                    if recovery.ready:
                        if reference.resume_communication() is not True:raise RuntimeError('Startup resume rejected')
                        recovery.complete(resume=switch.resume)
                        break
                    time.sleep(.02)
                return False
            def capture_reader():
                while True:
                    capture_stopped()
                    try:value=session_reader()
                    except Exception as exc:
                        if recovery is None or not recovery.transport_error(exc):raise
                        continue
                    if recovery is None or not recovery.paused:return value
            if callable(close_reader_thread):capture_reader.close_current_thread=close_reader_thread
            def restart_request(fn,*args):
                # 此处没有运行中的原生程序。可恢复的本地 HTTP 请求等待同一所有者时，继续监测控制状态。
                from .physical_interaction_supervisor import _Worker
                worker_factory=factories.get('startup_reference_worker',_Worker)
                deadline=time.monotonic()+min(profile.gate_timeout,.25)+(3. if recovery is not None else 0.)
                worker=worker_factory(fn,*args)
                try:
                    while True:
                        if worker.done and worker.error is not None:raise worker.error
                        capture_stopped()
                        if time.monotonic()>deadline:
                            raise RuntimeError('Startup reference operation timed out')
                        if worker.done:
                            if worker.error is not None:raise worker.error
                            return worker.value
                        current=capture_reader()
                        if (not isinstance(current,dict) or current.get('active') is not True
                                or any(current.get(k)!=v for k,v in control.items())):
                            raise RuntimeError('Guest control changed during reference restart')
                        time.sleep(.01)
                finally:
                    close=getattr(worker,'close',None)
                    try:
                        if callable(close):close(.5)
                        elif getattr(worker,'done',None) is not True:
                            raise RuntimeError('Startup reference worker still pending')
                    except Exception as exc:
                        raise SessionCleanupError('Startup reference worker cleanup incomplete') from exc
                    if type(worker.error) is ReferenceRequestError:
                        print('PHYSICAL_RESTART_RPC_FAILED: '+json.dumps(worker.error.safe_details(),sort_keys=True),
                              file=sys.stderr,flush=True)
                        if sys.exc_info()[1] is not worker.error:
                            raise worker.error
            restart_receipt=None
            restart_release_ns=None
            sample_observer=None
            proof_observer=None
            if restart_session is not None:
                reanchor=(callable(bind_reference) and getattr(profile,'reference_restart_policy',None)
                          =='capture_stationary_on_explicit_start')
                begin=(lambda sid:reference.restart_begin(sid,reanchor=True)) if reanchor else reference.restart_begin
                if restart_request(begin,restart_session) is not True:
                    raise RuntimeError('Previous reference restart verification rejected')
                restart_sample_floor=time.monotonic()
                restart_origin=getattr(reference,'restart_reference',None)
                if (not isinstance(restart_origin,(tuple,list)) or len(restart_origin)!=6
                        or any(type(x) not in (int,float) or not math.isfinite(x) for x in restart_origin)):
                    raise RuntimeError('Original reference missing from restart verification')
                if lifecycle is not None:lifecycle['reference_owned']=True
                def still_at_restart_origin(native):
                    from .reference_gate import rotation_vector_distance
                    pose=native['tcp_pose'];speed=native['tcp_speed'];qd=native['joint_speed']
                    return (math.dist(pose[:3],restart_origin[:3])<=.0005
                        and rotation_vector_distance(pose[3:],restart_origin[3:])<=.005
                        and math.hypot(*speed[:3])<.0005 and math.hypot(*speed[3:])<.002
                        and max(map(abs,qd))<.003)
                def sample_observer(primary,native,sampled_at):
                    nonlocal restart_receipt
                    capture_stopped()
                    if restart_release_ns is not None:
                        if not still_at_restart_origin(native):
                            raise RuntimeError('Robot left the original reference during restart capture')
                        return
                    if sampled_at<restart_sample_floor:
                        return  # 开始操作前缓存的回执不能作为新的重启依据。
                    try:
                        receipt=reference.restart_sample(restart_session,native['tcp_pose'],
                                                         native['tcp_speed'],sampled_at)
                    except ReferenceRequestError as exc:
                        print('PHYSICAL_RESTART_RPC_FAILED: '+json.dumps(exc.safe_details(),sort_keys=True),
                              file=sys.stderr,flush=True)
                        raise
                    if (not isinstance(receipt,dict) or receipt.get('ok') is not True
                            or type(receipt.get('return_verified')) is not bool
                            or receipt.get('sampled_at')!=sampled_at):
                        raise RuntimeError('Previous reference restart sample receipt invalid')
                    restart_receipt=receipt
                    capture_stopped()
                def proof_observer(primary,native,sampled_at):
                    nonlocal restart_release_ns,restart_origin
                    capture_stopped()
                    if restart_receipt is None:
                        raise RuntimeError('Restart original reference proof unavailable')
                    if reanchor and restart_receipt['return_verified'] is not True:
                        print('PHYSICAL_RESTART_WAITING: '+json.dumps({
                            'reference_policy':'current_stationary' if reanchor else 'original_reference',
                            'reference_check':restart_receipt.get('reference_check',{})},sort_keys=True),
                            file=sys.stderr,flush=True)
                        return False  # 在同一个采集截止时间内，继续读取验证所需的数据。
                    # 保持两个独立接收端打开，原子提交新样本并放行。SDK 清理不能耗尽最后一个样本的 150 毫秒有效时间。
                    try:
                        restart_release_ns=restart_request(reference.restart_release,restart_session,
                            native['tcp_pose'],native['tcp_speed'],sampled_at)
                    except ReferenceRequestError as exc:
                        if reanchor and exc.reference_reason=='return_unverified':return False
                        raise
                    if (type(restart_release_ns) is not int
                            or not 0<restart_release_ns<=round(time.monotonic()*1e9)):
                        raise RuntimeError('Previous reference restart release timestamp invalid')
                    if reanchor:
                        restart_origin=reference.restart_reference
                        if (not isinstance(restart_origin,(tuple,list)) or len(restart_origin)!=6
                                or any(type(x) not in (int,float) or not math.isfinite(x) for x in restart_origin)
                                or not still_at_restart_origin(native)):
                            raise RuntimeError('Committed selected reference differs from measured restart pose')
                    capture_stopped()
            if callable(bind_reference):
                capture_options={'reader':capture_reader,'control':control,'should_stop':capture_stopped}
                if sample_observer is not None:
                    capture_options.update(sample_observer=sample_observer,proof_observer=proof_observer)
                bound=bind_reference(**capture_options)
            else:
                from .operator_design_capture import capture_selected_reference
                capture_selected_reference(profile,reader=capture_reader,control=control,
                    should_stop=capture_stopped,sample_observer=sample_observer,proof_observer=proof_observer)
                bound=profile
            capture_stopped()
            if (bound is None or bound.host!=profile.host or
                    getattr(bound.cfg,'robot_model',None)!=getattr(profile.cfg,'robot_model',None)):
                raise RuntimeError('Selected bound reference identity changed')
            rebound=bound is not profile
            profile=bound
            release_ns=None
            if restart_session is not None:
                if restart_release_ns is None:
                    raise RuntimeError('Restart requires standstill at the previous reference; no automatic return')
                release_ns=restart_release_ns
                capture_stopped()
                print('PHYSICAL_RESTART_REFERENCE_VERIFIED: '+
                      ('new selected reference stationary; ' if reanchor else 'previous reference stationary; ')+
                      'awaiting fresh control for the new explicit Start.',flush=True)
            if rebound or release_ns is not None:
                # 采集或身份验证可能比原开始事件的有效期更长。改用本次交互的新心跳，不重放旧开始事件，也不放宽原时间限制。
                fresh_trigger=wait_for_trigger(source,switch,
                    observed_at=max(time.monotonic(),(release_ns or 0)/1e9),timeout=.3,
                    expected_action=before.get('control_action'),control=control,should_stop=capture_stopped)
                capture_stopped()
                if (trigger.get('event')=='physical_intro' and
                        fresh_trigger.get('intro_id')!=trigger.get('intro_id')):
                    raise RuntimeError('Selected reference intro identity changed during capture')
                trigger=fresh_trigger
        if trigger['event']=='physical_intro':
            session,runtime,dashboard=intro_runner(profile,session_reader,source,heartbeat,switch,
                control=control,intro_event=trigger,reference=reference,publisher=publisher,
                should_stop=should_stop)
        else:
            # 选择新目标会跳过开场，但仍须从当前有效、经审核的参考位开始，并建立新的静止待命程序。
            session=xyz_opener(profile.host,profile.cfg,session_reader,require_camera=False)
            _check_service_stop(should_stop,switch)
            switch.attach(session.gate)
            after=session_reader()
            if any(after.get(k)!=control[k] for k in ('instance','epoch','started_ns')):
                raise RuntimeError('Guest identity changed during startup')
            supervisor=profile.supervisor(session.engine,reference)
            dashboard=dashboard_factory(profile.host)
            runtime=PhysicalRuntime(session,supervisor,profile,dashboard,publisher,
                                    control=control,external_heartbeat=True,recovery=recovery)
        if report is not None:report(before,'running','none')
        while True:
            _check_service_stop(should_stop,switch)
            if source.failure:raise RuntimeError(source.failure)
            if switch.failure or heartbeat.failure:
                raise RuntimeError('Physical lease or control stopped: '+str(switch.failure or heartbeat.failure))
            event=source.take()
            _check_service_stop(should_stop,switch)
            _step_source_event(runtime,source,event)
            time.sleep(.005)
    finally:
        errors=[]
        if recovery is not None:
            try:recovery.fail('Physical session ended')
            except RuntimeError:pass
        def retire(action):
            try:action()
            except Exception as exc:errors.append(exc)
        unbind_source=getattr(source,'unbind_control',None)
        if callable(unbind_source):retire(unbind_source)
        retire(lambda:switch.stop('Physical runtime closed'))
        close_recovery=getattr(runtime,'close_recovery',None)
        if callable(close_recovery):retire(close_recovery)
        if session is not None:retire(session.stop)
        if heartbeat is not None:retire(heartbeat.close)
        close_reference_workers=getattr(getattr(runtime,'supervisor',None),'close_workers',None)
        if callable(close_reference_workers):retire(close_reference_workers)
        if publisher is not None:retire(publisher.close)
        if reference is not None:
            try: reference.fault()
            except Exception: pass
        if dashboard is not None:retire(dashboard.close)
        if session is not None:retire(lambda:_close_motion_session(session))
        if errors:raise SessionCleanupError('Physical session cleanup incomplete') from errors[0]


def verify_stationary_after_close(host, *, timeout=3.,clock=time.monotonic,sleep=time.sleep,
                                  channel_factory=None,dashboard_check=None):
    """新鲜的仅限接收的停止证明；不需要或命令回家。"""
    if channel_factory is None:
        from .ur_rtde_transport import RTDEChannel
        channel_factory=RTDEChannel
    if dashboard_check is None:
        from .horizontal_follow import check_dashboard
        dashboard_check=check_dashboard
    channel=channel_factory(host,spatial=True,receive_only=True)
    try:
        deadline=clock()+timeout;last=None;stationary_since=None
        while clock()<deadline:
            fb,stamp=channel.latest;failure=channel.failure;now=clock()
            if failure:raise RuntimeError('Independent stop telemetry unavailable')
            if (not isinstance(fb,dict) or type(stamp) not in (int,float)
                    or not math.isfinite(stamp) or not 0<=now-stamp<=.15):
                raise RuntimeError('Independent stop telemetry stale')
            robot_stamp=fb.get('timestamp');speed=fb.get('tcp_speed');qd=fb.get('joint_speed')
            if (type(robot_stamp) not in (int,float) or not math.isfinite(robot_stamp)
                    or any(not isinstance(v,(list,tuple)) or len(v)!=6 or
                           any(type(x) not in (int,float) or not math.isfinite(x) for x in v)
                           for v in (speed,qd))):
                raise RuntimeError('Independent stop telemetry invalid')
            if last is not None and robot_stamp<last:raise RuntimeError('Robot stop clock reversed')
            if last is None or robot_stamp>last:
                last=robot_stamp
                stationary=(math.hypot(*speed[:3])<.0005 and math.hypot(*speed[3:])<.002
                            and max(abs(v) for v in qd)<.003)
                if not stationary:stationary_since=None
                elif stationary_since is None:stationary_since=robot_stamp
                elif robot_stamp-stationary_since>=.25:
                    # 独立读取当前运行状态，确认原生程序确已结束，而非仅暂停在某个路径点。
                    dashboard_check(host,idle=True)
                    return True
            sleep(.02)
        raise RuntimeError('Independent stationary stop was not confirmed')
    finally:channel.close()


def publish_runtime_status(port,key,snapshot,status,reason):
    """仅显示回环报告；无租约续签或机器人命令。"""
    conn=http.client.HTTPConnection('127.0.0.1',port,timeout=.1)
    try:
        body=json.dumps({'instance':snapshot['instance'],'epoch':snapshot['epoch'],
                         'status':status,'reason':reason}).encode()
        conn.request('POST','/runtime-status',body=body,headers={
            'X-Guest-Token':key,'X-Guest-Client':'physical-supervisor',
            'Origin':f'http://127.0.0.1:{port}','Content-Type':'application/json',
            'Connection':'close'})
        response=conn.getresponse();response.read(4096)
        return response.status==200
    except (OSError,http.client.HTTPException):return False
    finally:conn.close()


def _fault_reason(error):
    text=str(error).lower()
    if 'speed guard' in text or 'fault -16' in text or 'fault -17' in text:return 'speed'
    if 'control' in text or 'guest' in text or 'heartbeat' in text:return 'control'
    if 'reference' in text:return 'reference'
    if 'camera' in text or 'guidance source' in text:return 'camera'
    return 'runtime'


def serve_sessions(profile,port,key,source, *, read_guest=None,session_runner=None,
                   stopped_check=None,report=None,is_alive=lambda:True,sleep=time.sleep):
    """持续的摄像头消费者；每个明确的启动最多尝试一次。在失败后，只有另一个用户创建的周期才能尝试启动。
    如果停止证明失败，后续的启动首先重复只读观察；不完整的资源清理会永久抑制此启动器实例。"""
    reader=read_guest or LocalGuestControlReader(port,key,
        timeout=.2 if getattr(profile.cfg,'gesture',None) is not None else .1)
    runner=session_runner or run_session
    stopped_check=stopped_check or verify_stationary_after_close
    report=report or (lambda snapshot,status,reason:publish_runtime_status(port,key,snapshot,status,reason))
    attempted=set();waiting_identity=None;pending_stop=False;cleanup_failed=False
    reference_lineage=None
    while is_alive():
        if source.failure:raise RuntimeError(source.failure)
        try:snapshot=reader()
        except Exception:
            sleep(.1);continue
        if (not isinstance(snapshot,dict) or not isinstance(snapshot.get('instance'),str)
                or type(snapshot.get('epoch')) is not int):
            sleep(.05);continue
        identity=(snapshot['instance'],snapshot['epoch'])
        if snapshot.get('active') is not True:
            if identity!=waiting_identity:
                report(snapshot,'waiting_start','none');waiting_identity=identity
            sleep(.05);continue
        if identity in attempted:
            sleep(.05);continue
        if (snapshot.get('control_action') not in ('start','restart')
                or type(snapshot.get('started_ns')) is not int or snapshot['started_ns']<=0
                or snapshot.get('camera_connected') is not True or snapshot.get('camera_fresh') is not True):
            sleep(.05);continue
        if (getattr(getattr(profile,'cfg',None),'gesture',None) is not None
                and snapshot.get('control_fresh') is not True):
            sleep(.05);continue
        # 建立任何连接前先标记尝试；参考位检查即使失败也算一次尝试。
        attempted.add(identity)
        if cleanup_failed:
            report(snapshot,'fault','runtime');continue
        if pending_stop:
            try:pending_stop=stopped_check(profile.host) is not True
            except Exception:pending_stop=True
            if pending_stop:
                report(snapshot,'fault','runtime');continue
        report(snapshot,'starting','none')
        error=None
        lifecycle={}
        try:
            options={'before':snapshot,'report':report}
            if session_runner is None:
                options['should_stop']=lambda:not is_alive()
            if session_runner is None or getattr(runner,'supports_reference_restart',False) is True:
                options.update(lifecycle=lifecycle,allow_reference_rearm=reference_lineage==snapshot['instance'])
            runner(profile,port,key,source,reader,**options)
        except SessionCleanupError as exc:cleanup_failed=True;error=exc
        except Exception as exc:error=exc
        finally:
            try:pending_stop=stopped_check(profile.host) is not True
            except Exception as exc:
                pending_stop=True
                if error is None:error=exc
        if cleanup_failed:
            reference_lineage=None
        elif lifecycle.get('reference_owned') is True:
            reference_lineage=snapshot['instance']
        if error is not None:
            print('PHYSICAL_SESSION_STOPPED: '+str(error),file=sys.stderr,flush=True)
            report(snapshot,'fault',_fault_reason(error))
        elif pending_stop:report(snapshot,'fault','runtime')
        else:report(snapshot,'stopped','none')
    return 0


def main(argv=None, *, source=None,read_guest=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--guest-session',type=Path,help='local session file containing port and key')
    parser.add_argument('--execute',action='store_true',help='explicitly start reviewed physical resident')
    parser.add_argument('--guest-wait',type=float,default=120.,help='seconds to wait for active iPad')
    args=parser.parse_args(argv)
    profile=reviewed_profile(json.loads(args.profile.read_text(encoding='utf-8-sig')))
    if not args.execute:
        print(json.dumps({'profile_reviewed':True,'robot_connected':False,'resident_uploaded':False}))
        return 0
    if args.guest_session is None:parser.error('--guest-session is required for physical execution')
    pairing=json.loads(args.guest_session.read_text(encoding='utf-8-sig'))
    port,key=pairing['port'],pairing['key']
    reader=read_guest or LocalGuestControlReader(port,key,
        timeout=.2 if getattr(profile.cfg,'gesture',None) is not None else .1)
    source=source or LatestEventSource(sys.stdin)
    before=wait_for_guest(reader,source,timeout=args.guest_wait)
    return run_session(profile,port,key,source,reader,before=before)


if __name__=='__main__':
    raise SystemExit(main())
