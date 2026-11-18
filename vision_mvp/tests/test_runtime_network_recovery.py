"""测试同一次交互内的恢复，不使用套接字、子进程或实机接口。"""
from types import SimpleNamespace

import pytest

from leica_tracker.physical_runtime import PhysicalRuntime
from leica_tracker.physical_resident_engine import Installation, ResidentEngine
from test_physical_resident_engine import reviewed
from test_physical_runtime import runtime


class HoldGate:
    can_send = True
    def __init__(self):
        self.is_paused = True
        self.calls = []
    def send(self, *values): self.calls.append(values)
    def stop(self, reason): self.can_send = False


def test_ready_resident_pumps_explicit_hold_without_inventing_a_target():
    cfg = Installation(reviewed());gate = HoldGate();engine = ResidentEngine(cfg,gate,123)
    engine.state = 'READY';engine.last_sample = 9.98
    fb = dict(session=123,state=4,sequence=0,tcp_pose=list(cfg.neutral),
              tcp_speed=[0.]*6,joints=list(cfg.joints))
    engine.tick(fb,sampled_at=10.,now=10.)
    assert gate.calls[-1][1] == 3
    assert gate.calls[-1][3] == list(cfg.neutral[:3])
    assert engine.state == 'READY' and engine.goal is None


class Recovery:
    paused = True
    ready = False
    def __init__(self): self.completed = 0
    def complete(self, *, resume=None):
        assert self.ready
        if resume: resume()
        self.paused=False;self.completed+=1


def setup():
    run,session,sup,pub=runtime();recovery=Recovery()
    gate=session.gate;gate.is_paused=False
    gate.pause=lambda reason:setattr(gate,'is_paused',True)
    gate.resume=lambda:setattr(gate,'is_paused',False)
    session.engine.ticks=[]
    session.engine.tick=lambda *a,**k:session.engine.ticks.append((a,k))
    session.reader.latest[0].update(tcp_speed=[0.]*6)
    session.writer.latest[0]['state']=4
    sup.reference=SimpleNamespace(resume_communication=lambda:True)
    run=PhysicalRuntime(session,sup,run.profile,run.dashboard,pub,
        control=run.control,recovery=recovery)
    return run,session,sup,pub,recovery


def stamp(run,session,t):
    run.dashboard.checked_at=t
    for channel in (session.writer,session.reader,session.states):
        channel.latest=(channel.latest[0],t)


def test_waiting_pumps_native_ack_but_admits_no_old_direction_or_supervisor_step():
    run,session,sup,pub,recovery=setup()
    run.step({'event':'person_guidance','cue':'MOVE_LEFT'},now=10.)
    assert session.gate.is_paused and len(session.engine.ticks)==1
    assert sup.ticks==[] and sup.events==[] and recovery.completed==0


def test_healthy_control_alone_cannot_resume_before_pause_ack_and_stationary_dwell():
    run,session,sup,pub,recovery=setup();recovery.ready=True
    session.writer.latest[0]['state']=1
    run.step(None,now=10.)
    stamp(run,session,10.1);session.writer.latest[0]['state']=4
    session.reader.latest[0]['joint_speed'][0]=.02
    run.step(None,now=10.1)
    assert recovery.completed==0 and session.gate.is_paused
    session.reader.latest[0]['joint_speed'][0]=0.
    for t in (10.2,10.3,10.4):
        stamp(run,session,t);run.step(None,now=t)
    assert recovery.completed==0
    stamp(run,session,10.51);run.step(None,now=10.51)
    assert recovery.completed==1 and not session.gate.is_paused
    assert sup.events==[]


def test_robot_feedback_age_is_not_relaxed_while_guest_network_waits():
    run,session,sup,pub,recovery=setup()
    with pytest.raises(RuntimeError):run.step(None,now=10.3)
    assert not session.gate.can_send and recovery.completed==0


def test_gap_in_native_samples_restarts_recovery_stationary_dwell():
    run,session,sup,pub,recovery=setup();recovery.ready=True
    for t in (10.,10.16,10.26,10.31):
        stamp(run,session,t);run.step(None,now=t)
    assert recovery.completed==0 and session.gate.is_paused


def test_opening_startup_gap_pumps_hold_past_old_300ms_without_reusing_start():
    from leica_tracker.physical_runtime import run_intro_then_open_xyz, SwitchingGate
    clock=[10.];ticks=[];recovery=Recovery()
    recovery.fail=lambda reason:None
    gate=HoldGate();gate.pause=lambda reason:None
    progress=SimpleNamespace(state='STARTING',started=None,last_writer=None,last_reader=None,started_ack=False)
    class Fresh:
        def __init__(self,value):self.value=value
        @property
        def latest(self):return self.value,clock[0]
    def tick():
        ticks.append(clock[0]);progress.started=progress.started or clock[0]
        progress.last_writer=progress.last_reader=clock[0]
    intro=SimpleNamespace(gate=gate,progress=progress,tick=tick,close=lambda:None,
        writer=Fresh(dict(state=103)),reader=Fresh(dict(tcp_speed=[0.]*6,joint_speed=[0.]*6)))
    class Source:
        failure=None
        def take(self):
            if clock[0]>=10.8:raise RuntimeError('test warm wait done')
            return None
    def reader():return {}
    reader.communication_recovery=recovery
    control=dict(instance='x',epoch=1,started_ns=1)
    event=dict(event='physical_intro',action='start',intro_id=12,
        source_monotonic_ns=10_000_000_000,control_instance='x',control_epoch=1)
    with pytest.raises(RuntimeError,match='test warm wait done'):
        run_intro_then_open_xyz(SimpleNamespace(host='offline',intro={}),reader,Source(),
            SimpleNamespace(failure=None),SwitchingGate(),control=control,intro_event=event,
            reference=None,publisher=SimpleNamespace(failed=None),clock=lambda:clock[0],
            sleep=lambda dt:clock.__setitem__(0,clock[0]+dt),intro_opener=lambda *a,**k:intro)
    assert ticks and ticks[-1]>=10.79


def test_pending_start_waits_for_real_owner_freshness_before_any_runner_attempt():
    from leica_tracker.physical_runtime import serve_sessions
    live=[True];attempts=[]
    base=dict(instance='same',epoch=1,started_ns=1,active=True,control_action='start',
              camera_connected=True,camera_fresh=True)
    snapshots=iter([{**base,'control_fresh':False},{**base,'control_fresh':True}])
    def runner(*a,before,**kw):attempts.append(before['control_fresh']);live[0]=False
    serve_sessions(SimpleNamespace(host='offline',cfg=SimpleNamespace(gesture=object())),
        8777,'fake',SimpleNamespace(failure=None),read_guest=lambda:next(snapshots),
        session_runner=runner,stopped_check=lambda _:True,report=lambda *a:None,
        is_alive=lambda:live[0],sleep=lambda _:None)
    assert attempts==[True]


def test_pre_native_wait_clears_held_start_before_requesting_intro_event():
    from leica_tracker.physical_runtime import wait_initial_communication,SwitchingGate
    recovery=Recovery();reads=[];resumes=[]
    def reader():
        reads.append(True)
        if len(reads)==3:recovery.ready=True
    wait_initial_communication(reader,SimpleNamespace(resume_communication=lambda:resumes.append(True) or True),
        recovery,SwitchingGate(),sleep=lambda _:None)
    assert len(reads)==3 and resumes==[True] and recovery.completed==1
