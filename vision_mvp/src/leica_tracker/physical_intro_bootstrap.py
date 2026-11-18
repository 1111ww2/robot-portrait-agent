"""审查过的引言的明确引导，永远不会自动启动硬件。
一次性程序使用一个标量RTDE写入通道、一个独立的仅接收空间通道和一个单独的客机控制守护程序。
仅上传不会发送运动启用。所有停止/故障路径都会关闭输入门，从不发送返回。未为物理安装授权。"""
import copy
import ipaddress
import math
from types import SimpleNamespace

from .physical_intro_program import IntroProgram, IntroProgress, _vector
from .physical_errors import SessionCleanupError
from .physical_resident_bootstrap import _dependencies, _state_snapshot, _retire_motion_resources
from .reference_gate import rotation_vector_distance


def _watchdog_detail(message,watch,gate):
    details=[]
    failure=getattr(watch,'failure',None);reason=getattr(gate,'reason',None)
    if failure:details.append('watchdog='+str(failure))
    if reason:details.append('gate='+str(reason))
    return message+(': '+'; '.join(details) if details else '')


def _sample(source,now):
    value,stamp=source.latest
    if (source.failure or not isinstance(value,dict) or type(stamp) not in (int,float)
            or not math.isfinite(stamp) or not 0<=now-stamp<=.15):
        raise RuntimeError('Intro telemetry unavailable or stale')
    return value,stamp


def _stationary_reference(source,program,now):
    fb,_=_sample(source,now)
    try:
        pose=_vector(fb.get('tcp_pose'),'TCP pose');speed=_vector(fb.get('tcp_speed'),'TCP speed')
        q=_vector(fb.get('joints'),'joints');qd=_vector(fb.get('joint_speed'),'joint speed')
    except ValueError as exc:raise RuntimeError(str(exc)) from exc
    if (math.dist(pose[:3],program.reference_pose[:3])>.0005 or
            rotation_vector_distance(pose[3:],program.reference_pose[3:])>.005 or
            max(abs(a-b) for a,b in zip(q,program.reference_joints))>.003 or
            math.hypot(*speed[:3])>=.0005 or math.hypot(*speed[3:])>=.002 or
            max(abs(v) for v in qd)>=.003):
        raise RuntimeError('Robot is not at the reviewed stationary intro reference')
    final=getattr(program,'final_joints',None);tolerance=getattr(program,'joint_tolerance',None)
    if final is not None and max(abs(a-b) for a,b in zip(q,final))<=tolerance:
        raise RuntimeError('Robot is already within this step target tolerance')
    return fb


def _startup_snapshot(states,writer,reader,program,clock):
    # 在采样验证时钟之前，冻结所有收据时间戳，就像OpenIntro.tick那样。
    # 并发生产者更新属于下一个周期，不得与更早的时钟瞬间进行比较。
    state=SimpleNamespace(latest=states.latest,failure=states.failure)
    writing=SimpleNamespace(latest=writer.latest,failure=writer.failure)
    reading=SimpleNamespace(latest=reader.latest,failure=reader.failure)
    now=clock()
    mode=_state_snapshot(state,program,now)
    writer_fb,_=_sample(writing,now)
    _stationary_reference(reading,program,now)
    return now,mode,writer_fb


class OpenIntro:
    def __init__(self,progress,gate,watch,command,writer,reader,states,dashboard,clock):
        self.progress,self.gate,self.watch=progress,gate,watch
        self.command,self.writer,self.reader,self.states=command,writer,reader,states
        self.dashboard=dashboard
        self.clock=clock;self.closed=False

    def tick(self):
        if self.closed:raise RuntimeError('Intro is closed')
        try:
            # 在阅读时钟之前，先对两个生产者进行快照；两者都不能代替对方的原始收据时间戳。
            ws=self.writer.latest;writer_failure=self.writer.failure
            rs=self.reader.latest;reader_failure=self.reader.failure
            state_snapshot=self.states.latest;state_failure=self.states.failure
            checked=self.dashboard.checked_at;dashboard_failure=self.dashboard.failure
            now=self.clock()
            if not self.gate.can_send:
                raise RuntimeError('Intro watchdog stopped: '+str(getattr(self.watch,'failure',None) or self.gate.reason))
            if (dashboard_failure or type(checked) not in (int,float) or not math.isfinite(checked)
                    or not 0<=now-checked<=2):raise RuntimeError('Intro dashboard status stale or unsafe')
            mode=_state_snapshot(SimpleNamespace(latest=state_snapshot,failure=state_failure),self.progress.program,now)
            writer,wt=ws;reader,rt=rs
            if writer_failure or reader_failure:raise RuntimeError('Intro telemetry failed')
            # Primary和RTDE可以按任意顺序报告程序完成情况。
            # 在决定是否允许停止的程序进入其受限的严格不发送观察阶段之前，进度会验证两个流。
            return self.progress.tick(writer,writer_sampled_at=wt,reader=reader,reader_sampled_at=rt,
                                      now=now,primary_running=mode.get('running'))
        except Exception as exc:
            self.gate.stop(str(exc))
            raise

    def stop(self):
        if not self.closed:
            self.progress.stop();self.watch.close()

    def close(self):
        if self.closed:return
        errors=[]
        try:self.progress.stop()
        except Exception as exc:errors.append(exc)
        self.closed=True
        errors.extend(_retire_motion_resources(self.gate,self.watch,self.writer,
            (self.command,self.reader,self.states,self.dashboard),reason='Intro session closed'))
        if errors:raise SessionCleanupError('Physical intro cleanup incomplete') from errors[0]

    def begin_completed_retirement(self):
        if self.closed or self.progress.state!='COMPLETE':
            raise SessionCleanupError('Intro measured completion required before normal retirement')
        try:
            self.watch.retire_completed()
            self.gate.retire_completed('Physical intro completed')
        except Exception as error:
            raise SessionCleanupError('Intro stopped before normal retirement') from error

    def close_completed(self):
        self.begin_completed_retirement()
        self.close()


def open_intro(host,profile,read_guest, *, dependencies=None,startup_timeout=3.,
               watch_interval=.05,watch_timeout=.3,_program_type=IntroProgram,require_camera=True):
    """上传一个保留的介绍居民；调用者必须明确勾选以启用它。"""
    if not callable(read_guest):raise ValueError('Guest control reader required')
    try:host=str(ipaddress.ip_address(host))
    except ValueError:raise ValueError('Explicit robot IP address required') from None
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0
           for v in (startup_timeout,watch_interval,watch_timeout)) or watch_interval>=watch_timeout:
        raise ValueError('Invalid startup or watchdog intervals')
    # 在打开任何套接字之前验证；保留一个私密的不可变副本。
    if not isinstance(_program_type,type) or not issubclass(_program_type,IntroProgram):
        raise ValueError('Explicit reviewed program type required')
    profile=copy.deepcopy(profile);validated_program=_program_type(profile,1)
    if dependencies is None:
        d=_dependencies()
        from .horizontal_follow import DashboardReceiver
        d['dashboard_factory']=DashboardReceiver
    else:d=dependencies
    states=writer=reader=command=gate=watch=dashboard=None
    try:
        d['dashboard_check'](host,idle=True,robot_model=validated_program.robot_model)
        states=d['state_factory'](host)
        writer=d['rtde_factory'](host,spatial=False)
        reader=d['rtde_factory'](host,spatial=True,receive_only=True)
        gate=d['gate_factory'](writer)
        recovery=getattr(read_guest,'communication_recovery',None)
        options={} if recovery is None else {'recovery':recovery}
        watch=d['watch_factory'](read_guest,gate,interval=watch_interval,timeout=watch_timeout,require_camera=require_camera,**options)
        watch.start()
        if not watch.wait_ready(startup_timeout) or not gate.can_send:
            raise RuntimeError(_watchdog_detail('Guest watchdog not ready',watch,gate))
        session=d['session_factory']();program=_program_type(profile,session)
        _startup_snapshot(states,writer,reader,program,d['clock'])
        command=d['command_factory'](host);d['dashboard_check'](host,idle=True,robot_model=program.robot_model)
        dashboard=d['dashboard_factory'](host)
        _startup_snapshot(states,writer,reader,program,d['clock'])
        if not gate.can_send:raise RuntimeError(_watchdog_detail('Guest watchdog stopped before intro upload',watch,gate))
        command.send(program.program())
        if not gate.can_send:raise RuntimeError(_watchdog_detail('Guest watchdog stopped during intro upload',watch,gate))
        deadline=d['clock']()+startup_timeout
        while True:
            if not gate.can_send:raise RuntimeError(_watchdog_detail('Guest watchdog stopped during intro startup',watch,gate))
            now,mode,writer_fb=_startup_snapshot(states,writer,reader,program,d['clock'])
            if writer_fb.get('session')==session and writer_fb.get('state')==100 and mode.get('running') is True:break
            if now>=deadline:raise RuntimeError('Intro resident startup acknowledgement timed out')
            d['sleep'](.02)
        return OpenIntro(IntroProgress(program,gate),gate,watch,command,writer,reader,states,dashboard,d['clock'])
    except BaseException as exc:
        errors=_retire_motion_resources(gate,watch,writer,(command,reader,states,dashboard),
                                       reason='Intro bootstrap failed')
        if errors:
            error=SessionCleanupError('Physical intro bootstrap cleanup incomplete')
            error.cleanup_errors=tuple(errors)
            raise error from exc
        raise
