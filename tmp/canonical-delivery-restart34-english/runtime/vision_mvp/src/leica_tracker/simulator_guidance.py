"""在LOCAL教师模拟器上的语义网络摄像头手势；没有硬件路径。HTTP固定为127.0.0.1:8766；
URX固定为127.0.0.1:30002。下面的联合偏移量是一个未验证的模拟编舞，而不是物理姿势。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import http.client
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time

from .gesture_preview import normalized_offset_for
from .completion_choreography import (completion_pose_at,simulated_frame_pose_at,
                                      CONFIRM_END_MS,SCREEN_CONFIRM_END_MS)
from .person_guidance import GuidanceCue
from .robot_gestures import gesture_plan_for,HOLDING_PATTERNS
from .agent_actions import agent_gesture
from .simulator_follow import verify_identity
from .simulator_fixed_base import HOME_X
from .simulator_planar import HOME_Z
from .simulator_spatial import HOME_Y, pose_at_xyz

HTTP_PORT = 8766
MAX_AGE = .3
HOME_JOINTS = pose_at_xyz(HOME_X, HOME_Y, HOME_Z)


def completion_joint_offset(elapsed_ms, *, framed=False,screen_coupled=False):
    """补偿原始RG6手腕几何形状，以获得固定TCP间距。"""
    inward,up,pitch=completion_pose_at(elapsed_ms,framed=framed,screen_coupled=screen_coupled)
    # 对于这个受保护的模型，d5=.0997，d6+活性RG6 TCP=.0996+.2786。
    wrist_x=.0997*(1-math.cos(pitch))+.3782*math.sin(pitch)
    wrist_z=-.0997*math.sin(pitch)+.3782*(1-math.cos(pitch))
    target=pose_at_xyz(HOME_X+inward-wrist_x,HOME_Y,HOME_Z+up-wrist_z)
    target[3]+=pitch
    return [q-home for q,home in zip(target,HOME_JOINTS)]


def validate_home(state):
    """此模拟器特定的IK需要其原始主机和RG6 TCP。"""
    joints = state.get('joints', [])
    if (len(joints) != 6 or not all(math.isfinite(q) for q in joints)
            or max(abs(a-b) for a,b in zip(joints,HOME_JOINTS)) > .002
            or state.get('active_tcp_offset') != [0.,0.,.2786,0.,0.,0.]):
        raise ValueError('Reset teacher simulator to original home and RG6 TCP before guidance')


def get_state(path):
    connection = http.client.HTTPConnection('127.0.0.1', HTTP_PORT, timeout=.5)
    try:
        connection.request('GET', path)
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f'Simulator HTTP {response.status}')
        return json.loads(response.read(2_000_000))
    finally:
        connection.close()


class GuidanceStream:
    """验证源的最新性，在WAIT心跳之间保留有限的手势。"""
    def __init__(self):
        self.stamp = -math.inf
        self.plan = None
        self.started = 0.
        self.plan_source_ns = None
        self.tracking = False
        self.cue = 'HOLD'
        self.failure = None
        self.camera_status = {}
        self.completion_interrupted = False
        self.transport_interrupted = False
        self.lock = threading.Lock()

    def accept(self, line, now):
        if len(line) > 8192:
            raise ValueError('Guidance JSONL exceeds 8192 characters')
        event = json.loads(line)
        if (event.get('event') != 'person_guidance'
                or event.get('coordinate_frame') != 'normalized_camera_image'
                or event.get('robot_actuated') is not False
                or type(event.get('tracking')) is not bool):
            raise ValueError('Expected semantic guidance preview event')
        ns = event.get('source_monotonic_ns')
        if type(ns) is not int or ns <= 0 or now-ns/1e9 < 0:
            raise ValueError('Stale/future guidance source timestamp')
        cue = GuidanceCue(event.get('cue'))
        raw = event.get('gesture')
        if raw is not None and not isinstance(raw,dict):
            raise ValueError('Gesture must be an object or null')
        amplitude=raw.get('amplitude',1.) if raw is not None else 1.
        profile=raw.get('profile','semantic') if raw is not None else 'semantic'
        if profile=='sim-intro':
            raise ValueError('Simulation intro requires its dedicated event')
        if profile=='agent-coupled':
            repetitions=raw.get('repetitions')
            duration=raw.get('duration_ms')
            if type(repetitions) is not int or repetitions<1 or type(duration) is not int or duration%repetitions:
                raise ValueError('Invalid agent timing')
            canonical=agent_gesture(cue.value,amplitude,repetitions,duration//repetitions)
        else:
            canonical = gesture_plan_for(cue,amplitude=amplitude,profile=profile)
        if raw is not None and (canonical is None or raw != asdict(canonical)):
            raise ValueError('Gesture must match the canonical semantic cue')
        if not event['tracking'] and cue is not GuidanceCue.HOLD:
            raise ValueError('Lost tracking requires HOLD')
        with self.lock:
            if ns/1e9 <= self.stamp:
                raise ValueError('Guidance timestamps must strictly increase')
            if now-ns/1e9 > MAX_AGE:
                # 摄像头推理/操作系统调度有时可能会超时。立即按住并丢弃提示，不要杀死输入。
                # 保留原始源时间，并取消任何正在进行的手势。
                self._cancel_completion()
                self.plan = None
                self.tracking = False
                self.stamp = ns/1e9
                self.cue = 'HOLD'
                self.camera_status = {'phase':'SOURCE_STALE',
                    'reason':'GUIDANCE_SOURCE_EXPIRED','source_monotonic_ns':ns}
                return
            if ns/1e9-self.stamp > MAX_AGE + 1e-9:
                self._cancel_completion()
                self.plan = None
            self.stamp = ns/1e9
            self.tracking = event['tracking'] and cue is not GuidanceCue.HOLD
            self.cue = cue.value
            self.camera_status = {'phase':event.get('phase'), 'reason':event.get('reason'),
                                  'interaction_mode':event.get('interaction_mode','precise'),
                                  'capture_simulated':event.get('capture_simulated',False),
                                  'framing_verified':event.get('framing_verified',False),
                                  'composition_frame':event.get('composition_frame'),
                                  'composition_style':event.get('composition_style',''),
                                  'model_status':event.get('model_status',''),
                                  'model_name':event.get('model_name',''),
                                  'model_confidence':event.get('model_confidence',0.),
                                  'model_requests':event.get('model_requests',0),
                                  'stage_progress':event.get('stage_progress',0.),
                                  'arrival_progress':event.get('arrival_progress',0.),
                                  'points':event.get('points'), 'source_monotonic_ns':ns}
            if not self.tracking:
                self._cancel_completion()
                self.plan = None
            elif raw is not None:
                self.plan = canonical
                self.started = self.stamp
                self.plan_source_ns = ns
                self.completion_interrupted = False
                self.transport_interrupted = False

    def interrupt_transport(self):
        """反馈丢失时取消运动；WAIT 不得重放旧方案。"""
        with self.lock:
            self._cancel_completion()
            self.plan = None
            self.transport_interrupted = True

    def _cancel_completion(self):
        if self.plan is not None and self.plan.pattern in HOLDING_PATTERNS:
            self.completion_interrupted = True

    def offset(self, now):
        with self.lock:
            if not self.tracking or not 0 <= now-self.stamp <= MAX_AGE:
                self._cancel_completion()
                return None
            if self.completion_interrupted or self.transport_interrupted:
                return None
            if self.plan is not None and self.plan.pattern == 'confirm_retract_hold':
                return completion_joint_offset((now-self.started)*1000,screen_coupled=self.plan.profile=='screen-coupled')
            if self.plan is not None and self.plan.pattern == 'sim_frame_confirm_retract_hold':
                return completion_joint_offset((now-self.started)*1000,framed=True,screen_coupled=self.plan.profile=='screen-coupled')
            if self.plan is not None and self.plan.pattern == 'sim_frame_adjust_hold':
                x,z,_=simulated_frame_pose_at((now-self.started)*1000)
                target=pose_at_xyz(HOME_X+x,HOME_Y,HOME_Z+z)
                return [q-home for q,home in zip(target,HOME_JOINTS)]
            h, v, d = normalized_offset_for(self.plan, (now-self.started)*1000)
        # 正面视图：屏幕左侧是基座+X，朝向观众是+Y。肩膀/肘部在基座固定的情况下产生侧向运动。
        # 深度使用基座旋转加上手臂/手腕补偿，因此X/Z和工具方向保持不变。
        target = pose_at_xyz(HOME_X-.12*h, HOME_Y+.025*d, HOME_Z-.08*v)
        return [q-home for q,home in zip(target,HOME_JOINTS)]


def joint_velocity(neutral, offset, joints, previous, dt):
    if (any(len(vector) != 6 for vector in (neutral, offset, joints, previous))
            or not all(math.isfinite(x) for vec in (neutral, offset, joints, previous) for x in vec)
            or not math.isfinite(dt) or dt <= 0 or dt > .3
            or max(map(abs, offset)) > .35
            or max(abs(a-b) for a,b in zip(joints,neutral)) > .45):
        raise ValueError('Invalid/out-of-envelope simulated joint target')
    requested = [6*(n+o-q) for n,o,q in zip(neutral,offset,joints)]
    scale = min(1., .6/max(1e-12,max(map(abs,requested))))
    requested = [r*scale for r in requested]
    change = [r-p for r,p in zip(requested,previous)]
    acceleration_scale = min(1.,1.2*dt/max(1e-12,max(map(abs,change))))
    # 一个完整的矢量尺度可以保持肩部/肘部/手腕耦合。
    return [p+delta*acceleration_scale for p,delta in zip(previous,change)]


def _wait_simulation_tick(stop, deadline, *, stop_file=None, clock=None, sleep=None):
    """仅用于本地参考模拟器，请使用精确的有限等待。Event.wait可能会因Windows计时器触发而超出预期；
    短时间sleep切片可以保持绝对触发截止日期的准确性，同时最多每5毫秒检查一次停止信号。不会进行忙等待。"""
    # 在Windows Python 3.12上，monotonic()可以具有15.6毫秒的分辨率；
    # perf_counter()提供高分辨率的独立截止日期时钟。
    clock=clock or time.perf_counter
    sleep=sleep or time.sleep
    while not stop.is_set() and not (stop_file is not None and stop_file.exists()):
        remaining=deadline-clock()
        if remaining<=0:return
        sleep(min(.005,remaining))


class LocalJointStream:
    def __init__(self):
        self.socket = socket.create_connection(('127.0.0.1',30002),timeout=.5)
        self.socket.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        self.closed = threading.Event()
        self.reader = threading.Thread(target=self._drain,daemon=True)
        self.reader.start()

    def _drain(self):
        while not self.closed.is_set():
            try:
                if not self.socket.recv(65536):
                    break
            except socket.timeout:
                continue
            except OSError:
                break
        self.closed.set()

    def send(self,rates):
        if self.closed.is_set():
            raise RuntimeError('Local simulator disconnected')
        values=','.join(f'{q:.9f}' for q in rates)
        self.socket.sendall(f'speedj([{values}], 1.2, 0.10)\n'.encode('ascii'))

    def close(self):
        try:
            self.socket.sendall(b'stopj(1.2)\n')
        except OSError:
            pass
        self.closed.set()
        self.socket.close()
        self.reader.join(timeout=1.)


def read_input(samples):
    try:
        while line := sys.stdin.readline(8193):
            if not line.endswith('\n'):
                raise ValueError('Incomplete guidance JSONL')
            samples.accept(line,time.monotonic())
        samples.failure = 'Input stream closed'
    except Exception as error:
        samples.failure = str(error)


def synthetic_event(elapsed, now):
    cues = ['INVITE','MOVE_LEFT','MOVE_RIGHT','MOVE_UP','MOVE_DOWN',
            'MOVE_CLOSER','MOVE_FARTHER','NOT_YET','ARRIVED','HOLD']
    durations=[4.]*len(cues)
    durations[-2]=8.  # 让确认、撤回和保留的结束完成。
    local=elapsed%sum(durations)
    index=0
    while local >= durations[index]:
        local-=durations[index]
        index+=1
    cue = GuidanceCue(cues[index])
    plan = gesture_plan_for(cue)
    return dict(event='person_guidance',cue=cue.value,tracking=cue is not GuidanceCue.HOLD,
                coordinate_frame='normalized_camera_image',robot_actuated=False,
                source_monotonic_ns=round(now*1e9),
                gesture=asdict(plan) if plan and local < .05 else None)


def build_guidance_stream(state,reference_path=None):
    """仅在具有自身映射的情况下，才使用明确选择的录制姿势。"""
    if reference_path is None:
        validate_home(state)
        return GuidanceStream()
    root=Path(__file__).resolve().parents[3]
    path=Path(reference_path).resolve()
    if not path.is_relative_to(root):raise ValueError('Simulation reference must be inside the workspace')
    ref=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(ref,dict) or ref.get('scope')!='historical_simulator_reference_only':
        raise ValueError('Expected a simulation-only reference')
    urx=state.get('urx',{});velocity=state.get('joint_velocity')
    if (urx.get('clients')!=0 or urx.get('program_running') is not False or urx.get('control_owner')!='idle'
            or not isinstance(velocity,list) or len(velocity)!=6
            or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>.001 for v in velocity)):
        raise RuntimeError('Recorded-reference simulator must be idle and stationary')
    from .simulator_reference_guidance import ReferenceGuidanceStream,validate_reference_home
    joints,tcp=ref.get('joints_rad'),ref.get('tcp_offset_m_rad')
    validate_reference_home(state,joints,tcp)
    return ReferenceGuidanceStream(joints,tcp)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=['stdin','synthetic'],default='stdin')
    parser.add_argument('--duration',type=float,default=0.)
    parser.add_argument('--stop-file',type=Path)
    parser.add_argument('--reference',type=Path,help='simulation-only recorded joint/TCP reference')
    args=parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error('duration must be finite and nonnegative')
    instance=verify_identity(get_state('/api/workbench/v1/manifest'))
    state=get_state('/api/sim/v1/state')
    if (state.get('instance_id')!=instance or state.get('backend')!='sim'
            or state.get('service')!='urx-browser-simulator'):
        raise RuntimeError('Simulator identity changed before connecting')
    if state['urx']['clients'] or state['urx']['program_running']:
        raise RuntimeError('Simulator must be idle with no other controller')
    samples=build_guidance_stream(state,args.reference)
    neutral=state['joints']
    neutral_tcp=state.get('tcp_pose')
    if len(neutral)!=6 or not all(math.isfinite(x) for x in neutral):
        raise ValueError('Invalid simulator neutral joints')
    stop=threading.Event()
    old_handlers={sig:signal.signal(sig,lambda *_:stop.set()) for sig in (signal.SIGINT,signal.SIGTERM)}
    robot=None
    feedback=None
    photo_gate=None
    intro_gate=None
    feedback_config=os.environ.get('LEICA_SIM_FEEDBACK') if args.reference else None
    if feedback_config:
        from .simulator_feedback import MotionFeedbackSender,SimulatorPhotoGate,SimulatorIntroGate
        feedback=MotionFeedbackSender(feedback_config)
        photo_gate=SimulatorPhotoGate()
        intro_gate=SimulatorIntroGate()
    started=last=time.monotonic()
    previous=[0.]*6
    count=0
    synthetic_index=None
    feedback_failures=0
    try:
        robot=LocalJointStream()
        if args.source=='stdin':
            threading.Thread(target=read_input,args=(samples,),daemon=True).start()
        print(json.dumps(dict(event='guidance_simulator_connected',instance=instance,
                              simulated=True,neutral_joints=neutral)),flush=True)
        while not stop.is_set() and (not args.duration or time.monotonic()-started<args.duration):
            precise_tick=time.perf_counter()
            tick=time.monotonic()
            if args.stop_file and args.stop_file.exists():
                break
            if samples.failure:
                if samples.failure=='Input stream closed':
                    break
                raise ValueError(samples.failure)
            if args.source=='synthetic':
                elapsed=tick-started
                event=synthetic_event(elapsed,tick)
                index=event['cue']
                if index != synthetic_index:
                    plan=gesture_plan_for(GuidanceCue(event['cue']))
                    event['gesture']=asdict(plan) if plan else None
                    synthetic_index=index
                else:
                    event['gesture']=None
                samples.accept(json.dumps(event),tick)
            try:
                state=get_state('/api/sim/v1/motion')
            except (OSError,http.client.HTTPException):
                # 在反馈超时期间，保持摄像头摄入状态。停止本地模拟器并取消提示，而不是在反馈返回时使用缓存姿势或重播其手势。
                samples.interrupt_transport()
                if photo_gate is not None:photo_gate.reset()
                if intro_gate is not None:intro_gate.reset()
                robot.send([0.]*6)
                previous=[0.]*6
                last=time.monotonic()
                feedback_failures+=1
                print(json.dumps(dict(event='guidance_simulator_hold',simulated=True,
                    hold=True,reason='SIMULATOR_FEEDBACK_UNAVAILABLE',
                    failures=feedback_failures)),flush=True)
                stop.wait(.25)
                continue
            if (state.get('instance_id')!=instance or state.get('backend')!='sim'
                    or state.get('service')!='urx-browser-simulator'):
                raise RuntimeError('Simulator identity changed')
            if feedback_failures:
                print(json.dumps(dict(event='guidance_simulator_feedback_restored',
                    simulated=True,failures=feedback_failures)),flush=True)
                feedback_failures=0
            if args.reference is not None:
                with samples.lock:
                    intro_source=(samples.plan_source_ns if samples.plan and samples.plan.pattern=='curiosity_intro' else None)
                if intro_source is not None and samples.anchor_intro(intro_source,state['joints']):
                    # 介绍和后来的方向提示都共享了开始时的测量姿势，而不是过程的旧参考。
                    neutral=list(state['joints']);neutral_tcp=list(state['tcp_pose'])
            offset=samples.offset(time.monotonic())
            dt=min(.1,max(.001,tick-last))
            rates=([0.]*6 if offset is None else joint_velocity(neutral,offset,state['joints'],previous,dt))
            robot.send(rates)
            if feedback is not None:
                with samples.lock:
                    plan=samples.plan
                    source_ns=samples.plan_source_ns
                    plan_started=samples.started
                ready=photo_gate.update(
                    source_ns=source_ns,cue=plan.cue if plan else None,
                    pattern=plan.pattern if plan else None,hold=offset is None,
                    elapsed_ms=(time.monotonic()-plan_started)*1000,
                    # 参考映射器确认在最后一个脉冲后已经处于中性目标；不要将保持结束持续时间添加到访客的等待时间中。
                    duration_ms=(SCREEN_CONFIRM_END_MS if plan and plan.profile=='screen-coupled'
                                 else CONFIRM_END_MS if plan else None),
                    joints=state.get('joints'),neutral=neutral,
                    velocity=state.get('joint_velocity'),now=time.monotonic())
                intro_ready=intro_gate.update(source_ns=source_ns,cue=plan.cue if plan else None,
                    pattern=plan.pattern if plan else None,hold=offset is None,
                    elapsed_ms=(time.monotonic()-plan_started)*1000,
                    duration_ms=plan.duration_ms if plan else None,
                    joints=state.get('joints'),neutral=neutral,velocity=state.get('joint_velocity'),now=time.monotonic())
                if intro_ready:
                    samples.mark_intro_completed(source_ns)
                try:
                    feedback.send(source_ns=source_ns,cue=plan.cue if plan else None,
                        hold=offset is None,delta=[a-b for a,b in zip(state['tcp_pose'][:3],neutral_tcp[:3])],
                        amplitude=plan.amplitude if plan else 1.,photo_ready=ready,
                        profile=plan.profile if plan else 'semantic',intro_ready=intro_ready)
                except OSError:
                    pass  # 缺少电测空白时，访客灯会亮起；请勿中断控制。
            previous=rates
            last=tick
            count+=1
            print(json.dumps(dict(event='guidance_simulator_sync',simulated=True,
                sequence=count,control_tick_ns=round(precise_tick*1_000_000_000),
                cue=samples.cue,hold=offset is None,joints=state['joints'],
                tcp_pose=state['tcp_pose'],joint_rates=rates,
                gesture_amplitude=samples.plan.amplitude if samples.plan else 0.,
                gesture_pattern=samples.plan.pattern if samples.plan else None,
                gesture_elapsed_ms=round((tick-samples.started)*1000) if samples.plan else None,
                camera_status=samples.camera_status)),flush=True)
            if args.reference is not None:
                _wait_simulation_tick(stop,precise_tick+.05,stop_file=args.stop_file)
            else:
                stop.wait(max(0.,.05-(time.monotonic()-tick)))
    finally:
        if feedback is not None:feedback.close()
        if robot is not None:
            robot.close()
        for sig,handler in old_handlers.items():
            signal.signal(sig,handler)
        print(json.dumps(dict(event='guidance_simulator_finished',simulated=True,updates=count)),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
