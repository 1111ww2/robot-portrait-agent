"""运行时测试使用假渠道，不会打开物理连接。"""
import pytest

from leica_tracker.physical_runtime import (FeedbackPublisher, LeaseHeartbeat, PhysicalRuntime, SwitchingGate,
                                           reviewed_profile, wait_for_guest, wait_for_trigger, register_backend,
                                           run_intro_then_open_xyz)
from leica_tracker.physical_resident_engine import Installation, ResidentEngine
from test_physical_resident_engine import reviewed


def data():
    cfg = reviewed(); ref = cfg['reference'][:3]
    cues = ('MOVE_LEFT','MOVE_RIGHT','MOVE_UP','MOVE_DOWN','MOVE_CLOSER','MOVE_FARTHER')
    offsets = ((-.005,0,0),(.005,0,0),(0,0,.005),(0,0,-.005),(0,-.005,0),(0,.005,0))
    paths = {cue:[[ref[i]+offset[i] for i in range(3)]] for cue,offset in zip(cues,offsets)}
    q=cfg['joints'];waypoints=[]
    for i in range(1,14):
        point=list(q);point[-1]+=.02*i;waypoints.append(point)
    waypoints.append(list(q))
    intro={'reference_joints':q,'reference_pose':cfg['reference'],'tcp':cfg['tcp'],
           'waypoints':waypoints, 'joint_speed':.02,'joint_acceleration':.02,
           'max_tcp_angular_speed':.04,'max_tcp_linear_speed':.01,'timeout':40.,
           'review':{k:True for k in ('robot','mount','tcp','payload','path','dynamics','stop','gesture')}}
    import copy
    intro['reviewed']={k:copy.deepcopy(intro[k]) for k in
        ('reference_joints','reference_pose','tcp','waypoints','joint_speed','joint_acceleration',
         'max_tcp_angular_speed','max_tcp_linear_speed','timeout')}
    preview={'status':'OPERATOR_ACCEPTED','robot_ip':'192.0.2.1','reference_joints':list(q),
             'tcp':list(cfg['tcp']),'waypoints':copy.deepcopy(waypoints),'observed_indices':list(range(1,15)),
             'observations':[{'index':i,'joints':list(point),'tcp_pose':list(cfg['reference']),
                              'sampled_at':100.+i,'joint_tolerance':.001}
                             for i,point in enumerate(waypoints,1)],'operator_decision':'accept'}
    preview['reviewed']=copy.deepcopy(preview)
    runtime={'hold_timeout':1.,'action_timeout':30.,'gate_timeout':.1,'max_joint_speed':.002}
    return {'status':'PHYSICAL_INTERACTION_REVIEWED','robot_ip':'192.0.2.1','reviewed_robot_ip':'192.0.2.1','intro':intro,'intro_preview':preview,
            'installation':cfg,'reviewed_installation':copy.deepcopy(cfg),'paths':paths,'reviewed_paths':paths,
            **runtime,'reviewed_runtime':copy.deepcopy(runtime)}


class Gate:
    def __init__(self): self.can_send=True; self.reason=None
    def stop(self, reason): self.can_send=False; self.reason=reason


def test_guest_preparation_may_change_inactive_epoch_before_start():
    from types import SimpleNamespace
    values=iter([
        {'instance':'same','epoch':0,'active':False},
        {'instance':'same','epoch':1,'active':False},
        {'instance':'same','epoch':2,'active':True,'camera_connected':True,'camera_fresh':True},
    ])
    now=[0.]
    def sleep(dt):now[0]+=dt
    result=wait_for_guest(lambda:next(values),SimpleNamespace(failure=None),timeout=1.,clock=lambda:now[0],sleep=sleep)
    assert result['epoch']==2


class Channel:
    def __init__(self, value, stamp=10.): self.latest=(value,stamp); self.failure=None


class Engine:
    def __init__(self,cfg): self.cfg=cfg; self.gate=Gate(); self.state='READY'
    def stop(self): self.state='STOPPED'; self.gate.stop('stop')


class Session:
    def __init__(self, cfg):
        self.engine=Engine(cfg); self.gate=self.engine.gate
        self.writer=Channel({'tcp_pose':list(cfg.neutral),'tcp_speed':[0.]*6,
                             'joints':list(cfg.joints),'session':123,'state':0,'sequence':1})
        self.reader=Channel({'joint_speed':[0.]*6})
        self.states=Channel({'mode':{'connected':True,'enabled':True,'power':True,'mode':7,
                                      'emergency':False,'protective':False,'paused':False,'running':True},
                             'tcp_offset':list(cfg.tcp)})
    @property
    def feedback(self): return self.writer.latest


class Supervisor:
    def __init__(self,engine): self.engine=engine; self.state='OBSERVING'; self.ticks=[]; self.events=[]
    def set_move_readiness(self,callback): self.move_ready=callback
    def bind_control_identity(self,control): self.identity=control
    def tick(self,*args,**kwargs): self.ticks.append((args,kwargs))
    def accept_event(self,event,*,now): self.events.append(event); return False
    def stop(self): self.state='STOPPED'; self.engine.stop()


class Dashboard:
    def __init__(self): self.failure=None; self.checked_at=10.


class Publisher:
    def __init__(self): self.failed=None; self.calls=[]
    def submit(self, body): self.calls.append(body)
    def ready_after(self, action_id): return 10.


def runtime():
    profile=reviewed_profile(data()); session=Session(profile.cfg)
    sup=Supervisor(session.engine); pub=Publisher()
    return PhysicalRuntime(session,sup,profile,Dashboard(),pub,control={'instance':'x','epoch':1,'started_ns':9_000_000_000}),session,sup,pub


def test_stopped_gate_reports_latched_watchdog_and_gate_reasons():
    from types import SimpleNamespace
    run,session,_sup,_pub=runtime()
    session.watch=SimpleNamespace(failure='Guest snapshot failed: camera_fresh_not_true')
    session.gate.reason='first gate closure'
    session.gate.can_send=False
    with pytest.raises(RuntimeError) as error:
        run.step(None,now=10.)
    message=str(error.value)
    assert message.startswith('Motion or guest watchdog stopped')
    assert 'watchdog=Guest snapshot failed: camera_fresh_not_true' in message
    assert 'gate=first gate closure' in message


def test_stopped_gate_reports_reason_without_optional_watchdog():
    run,session,_sup,_pub=runtime()
    session.gate.reason='RTDE input failed'
    session.gate.can_send=False
    with pytest.raises(RuntimeError,match='gate=RTDE input failed'):
        run.step(None,now=10.)


def test_unreviewed_profile_fails_before_any_connections():
    candidate=data(); candidate['status']='DRAFT'
    with pytest.raises(ValueError,match='review'):reviewed_profile(candidate)
    candidate=data(); candidate['reviewed_paths']=dict(candidate['paths']); candidate['reviewed_paths']['MOVE_LEFT']=[[0,0,0]]
    with pytest.raises(ValueError):reviewed_profile(candidate)
    candidate=data(); del candidate['intro']
    with pytest.raises(ValueError):reviewed_profile(candidate)
    candidate=data(); del candidate['intro_preview']
    with pytest.raises(ValueError):reviewed_profile(candidate)
    candidate=data(); candidate['intro_preview']['observed_indices'].pop()
    with pytest.raises(ValueError,match='preview'):reviewed_profile(candidate)
    candidate=data(); candidate['intro_preview']['observations'][2]['joints'][0]+=.1
    candidate['intro_preview']['reviewed']['observations']=__import__('copy').deepcopy(candidate['intro_preview']['observations'])
    with pytest.raises(ValueError,match='preview'):reviewed_profile(candidate)
    candidate=data(); candidate['robot_ip']='192.0.2.2'
    with pytest.raises(ValueError,match='review'):reviewed_profile(candidate)
    candidate=data(); candidate['installation']['lower'][0]-=.01
    with pytest.raises(ValueError,match='review'):reviewed_profile(candidate)
    candidate=data();candidate['max_joint_speed']=.1
    with pytest.raises(ValueError,match='review'):reviewed_profile(candidate)
    candidate=data(); candidate['intro']['tcp']=list(candidate['intro']['tcp']);candidate['intro']['tcp'][2]+=.01
    candidate['intro']['reviewed']['tcp']=list(candidate['intro']['tcp'])
    with pytest.raises(ValueError,match='intro'):reviewed_profile(candidate)


def test_noncollinear_or_backtracking_outbound_path_cannot_claim_directional_light():
    candidate=data(); path=[[.098,-.5,.3],[.095,-.499,.3]]
    candidate['paths']['MOVE_LEFT']=path
    candidate['reviewed_paths']['MOVE_LEFT']=path
    with pytest.raises(ValueError,match='light'):reviewed_profile(candidate)
    candidate=data(); path=[[.096,-.5,.3],[.098,-.5,.3],[.095,-.5,.3]]
    candidate['paths']['MOVE_LEFT']=path
    candidate['reviewed_paths']['MOVE_LEFT']=path
    with pytest.raises(ValueError,match='light'):reviewed_profile(candidate)


def test_independent_samples_advance_before_each_robot_tick_and_stale_stops():
    run,session,sup,pub=runtime()
    run.step(None,now=10.)
    assert len(sup.ticks)==1
    session.writer.latest=(session.writer.latest[0],10.02)
    run.step(None,now=10.02)
    assert len(sup.ticks)==1
    session.reader.latest=(session.reader.latest[0],10.03)
    run.step(None,now=10.03)
    assert len(sup.ticks)==2
    with pytest.raises(RuntimeError):run.step(None,now=10.3)
    assert not session.gate.can_send


def test_feedback_failure_and_tcp_change_close_gate():
    run,session,sup,pub=runtime(); pub.failed='network error'
    with pytest.raises(RuntimeError):run.step(None,now=10.)
    assert not session.gate.can_send
    run,session,sup,pub=runtime(); session.states.latest[0]['tcp_offset'][2]+=.01
    with pytest.raises(RuntimeError):run.step(None,now=10.)
    assert not session.gate.can_send


def test_intro_is_never_claimed_ready_without_supported_rotation():
    run,session,sup,pub=runtime()
    run.step({'event':'physical_intro','action':'start','intro_id':1,'source_monotonic_ns':10_000_000_000},now=10.)
    assert not any(x.get('op')=='intro' for x in pub.calls)
    assert sup.events==[]


def test_measured_light_feedback_is_bound_to_accepted_cue_and_fresh_position():
    run,session,sup,pub=runtime()
    def accept(event, *, now): sup.state='BEGIN_PENDING'; return True
    sup.accept_event=accept
    event={'event':'person_guidance','cue':'MOVE_LEFT','source_monotonic_ns':10_000_000_000,
           'control_instance':'x','control_epoch':1}
    run.step(event,now=10.)
    begin=[x for x in pub.calls if x.get('op')=='begin']
    assert len(begin)==1 and begin[0]['cue']=='MOVE_LEFT' and begin[0]['direction']==pytest.approx([-1,0,0])
    assert sup.move_ready('MOVE_LEFT',event['source_monotonic_ns']) is True
    assert sup.move_ready('MOVE_RIGHT',event['source_monotonic_ns']) is False
    sup.state='MOVING'
    session.writer.latest=(dict(session.writer.latest[0],tcp_pose=[.099,-.5,.3,0.,3.14,0.]),10.11)
    session.reader.latest=(session.reader.latest[0],10.11)
    session.states.latest=(session.states.latest[0],10.11); run.dashboard.checked_at=10.11
    run.step(None,now=10.11)
    sample=[x for x in pub.calls if x.get('op')=='sample'][-1]
    assert sample['action_id']==begin[0]['action_id'] and sample['source_ns']==event['source_monotonic_ns']
    assert sample['current_position']==pytest.approx([.099,-.5,.3])
    sup.state='OBSERVING'; session.writer.latest=(session.writer.latest[0],10.14)
    session.reader.latest=(session.reader.latest[0],10.14)
    run.step(None,now=10.14)
    assert pub.calls[-1]['op']=='clear'


def test_presentation_ack_is_not_a_continuous_motor_lease():
    run,session,sup,pub=runtime()
    pub.ready_after=lambda action_id:None
    def accept(event, *, now):sup.state='BEGIN_PENDING';return True
    sup.accept_event=accept
    event={'event':'person_guidance','cue':'MOVE_LEFT','source_monotonic_ns':10_000_000_000,
           'control_instance':'x','control_epoch':1}
    run.step(event,now=10.)
    assert sup.move_ready('MOVE_LEFT',event['source_monotonic_ns']) is True


def test_wait_for_active_camera_before_touching_robot():
    state = {'t':0.}; source=type('Source',(),{'failure':None})()
    snapshots=iter([{'instance':'a','epoch':0,'active':False},
                    {'instance':'a','epoch':1,'active':True,'camera_connected':True,'camera_fresh':True}])
    def clock():return state['t']
    def sleep(seconds):state['t']+=seconds
    result=wait_for_guest(lambda:next(snapshots),source,timeout=2.,clock=clock,sleep=sleep)
    assert result['epoch']==1 and state['t']>0


def test_user_stop_while_waiting_or_timeout_aborts_without_robot():
    state = {'t':0.}; source=type('Source',(),{'failure':None})()
    def clock():return state['t']
    def sleep(seconds):state['t']+=seconds
    snapshots=iter([{'instance':'a','epoch':0,'active':True,'camera_fresh':False},
                    {'instance':'a','epoch':1,'active':False}])
    with pytest.raises(RuntimeError,match='changed'):
        wait_for_guest(lambda:next(snapshots),source,timeout=1.,clock=clock,sleep=sleep)
    with pytest.raises(RuntimeError,match='timed out'):
        wait_for_guest(lambda:{'instance':'a','epoch':0,'active':False},source,timeout=.1,clock=clock,sleep=sleep)


def test_guest_server_change_before_start_still_aborts():
    from types import SimpleNamespace
    values=iter([{'instance':'a','epoch':0,'active':False},
                 {'instance':'b','epoch':1,'active':True,'camera_connected':True,'camera_fresh':True}])
    now=[0.]
    def sleep(dt):now[0]+=dt
    with pytest.raises(RuntimeError,match='changed'):
        wait_for_guest(lambda:next(values),SimpleNamespace(failure=None),timeout=1.,clock=lambda:now[0],sleep=sleep)


def test_guest_can_start_before_camera_fresh_without_false_abort():
    state={'t':0.}; source=type('Source',(),{'failure':None})()
    values=iter([{'instance':'a','epoch':0,'active':False},
                 {'instance':'a','epoch':1,'active':True,'camera_connected':True,'camera_fresh':False},
                 {'instance':'a','epoch':1,'active':True,'camera_connected':True,'camera_fresh':True}])
    def sleep(seconds):state['t']+=seconds
    assert wait_for_guest(lambda:next(values),source,timeout=1.,clock=lambda:state['t'],sleep=sleep)['epoch']==1


def test_blocked_feedback_request_closes_gate_without_waiting_for_http_return():
    import threading
    blocked=threading.Event(); gate=Gate()
    class Reference:
        def _request(self,*args): blocked.wait(2.); return {'ok':True}
    publisher=FeedbackPublisher(Reference(),gate,max_latency=.05)
    try:
        publisher.submit({'op':'heartbeat'})
        deadline=__import__('time').monotonic()+1.
        while gate.can_send and __import__('time').monotonic()<deadline:
            __import__('time').sleep(.005)
        assert not gate.can_send and publisher.failed
    finally:
        blocked.set(); publisher.close()


def test_clock_is_sampled_after_concurrent_feedback_snapshots():
    run,session,sup,pub=runtime(); current={'now':10.}
    original=type(session).feedback
    class UpdatedSession(Session):
        @property
        def feedback(self):
            current['now']=10.01
            self.writer.latest=(self.writer.latest[0],10.01)
            self.reader.latest=(self.reader.latest[0],10.01)
            self.states.latest=(self.states.latest[0],10.01)
            return self.writer.latest
    updated=UpdatedSession(run.profile.cfg); run.session=updated
    run.clock=lambda:current['now']
    run.step(None)
    assert len(sup.ticks)==1 and sup.ticks[0][1]['now']==10.01


def test_register_uses_returned_epoch_bound_lease_for_feedback():
    created=[]
    class Reference:
        def __init__(self,port,key,*,timeout,control,lease):
            self.lease=lease; self.control=control; created.append(self)
        def _request(self,method,path,body):
            assert body=={'op':'register'}
            return {'ok':True,'lease':'server-generated-lease'}
    ref=register_backend(8767,'secret',{'instance':'a','epoch':7},timeout=.1,factory=Reference)
    assert ref.lease=='server-generated-lease' and ref.control=={'instance':'a','epoch':7}


def test_stop_during_stage_transition_blocks_late_gate_attach():
    switch=SwitchingGate();first=Gate();switch.attach(first);switch.detach(first)
    switch.stop('visitor stopped')
    second=Gate()
    with pytest.raises(RuntimeError):switch.attach(second)
    assert not second.can_send


def test_fresh_intro_heartbeat_is_a_valid_start_when_first_event_was_overwritten():
    class Source:
        failure=None
        def take(self):return {'event':'physical_intro','action':'heartbeat','intro_id':10_050_000_000,
                               'source_monotonic_ns':10_100_000_000,'control_instance':'a','control_epoch':2}
    trigger=wait_for_trigger(Source(),SwitchingGate(),observed_at=10.,timeout=1.,expected_action='start',control={'instance':'a','epoch':2,'started_ns':9_000_000_000},clock=lambda:10.1,sleep=lambda _:None)
    assert trigger['intro_id']==10_050_000_000


def test_start_wait_rejects_pre_activation_or_stale_events():
    class Source:
        failure=None
        def take(self):return {'event':'physical_intro','action':'heartbeat','intro_id':9_950_000_000,
                               'source_monotonic_ns':9_900_000_000,'control_instance':'a','control_epoch':2}
    t={'v':10.}
    def sleep(s):t['v']+=s
    with pytest.raises(RuntimeError,match='timed out'):
        wait_for_trigger(Source(),SwitchingGate(),observed_at=10.,timeout=.2,expected_action='start',control={'instance':'a','epoch':2,'started_ns':9_000_000_000},clock=lambda:t['v'],sleep=sleep)


def test_intro_entry_rejects_wrong_identity_before_opening_any_device():
    profile=reviewed_profile(data());opened=[];switch=SwitchingGate()
    event={'event':'physical_intro','action':'start','intro_id':10_050_000_000,
           'source_monotonic_ns':10_100_000_000,'control_instance':'other','control_epoch':2}
    with pytest.raises(ValueError):
        run_intro_then_open_xyz(profile,None,None,None,switch,control={'instance':'a','epoch':2,'started_ns':9_000_000_000},
            intro_event=event,reference=None,publisher=None,clock=lambda:10.1,intro_opener=lambda *a:opened.append(True))
    assert opened==[]


def test_start_never_interprets_old_tracking_as_new_target():
    events=iter([{'event':'person_guidance','tracking':True,'source_monotonic_ns':10_100_000_000},
                 {'event':'physical_intro','action':'heartbeat','intro_id':10_050_000_000,'source_monotonic_ns':10_104_000_000,
                  'control_instance':'a','control_epoch':2}])
    class Source:
        failure=None
        def take(self):return next(events)
    t={'v':10.1}
    def sleep(s):t['v']+=s
    value=wait_for_trigger(Source(),SwitchingGate(),observed_at=10.,timeout=1.,expected_action='start',control={'instance':'a','epoch':2,'started_ns':9_000_000_000},clock=lambda:t['v'],sleep=sleep)
    assert value['event']=='physical_intro'


def test_old_epoch_intro_is_not_accepted_even_when_timestamp_is_fresh():
    class Source:
        failure=None
        def take(self):return {'event':'physical_intro','action':'heartbeat','intro_id':10_050_000_000,
                               'source_monotonic_ns':10_100_000_000,'control_instance':'a','control_epoch':1}
    t={'v':10.1}
    def sleep(s):t['v']+=s
    with pytest.raises(RuntimeError,match='timed out'):
        wait_for_trigger(Source(),SwitchingGate(),observed_at=10.,timeout=.2,expected_action='start',control={'instance':'a','epoch':2,'started_ns':9_000_000_000},clock=lambda:t['v'],sleep=sleep)


def test_person_frame_before_epoch_start_is_ignored():
    run,session,sup,pub=runtime()
    run.control['started_ns']=10_000_000_000
    run.step({'event':'person_guidance','tracking':True,'source_monotonic_ns':10_000_000_000,
              'control_instance':'x','control_epoch':1,'cue':'MOVE_LEFT'},now=10.)
    assert sup.events==[]


def test_intro_ready_cannot_be_published_until_active_status_is_acknowledged():
    class P:
        def __init__(self):self.calls=[];self.ready=False
        def submit(self,body):self.calls.append(body)
        def intro_acknowledged(self,intro_id,status):return self.ready and status=='active'
    p=P();heart=LeaseHeartbeat(p,interval=.01)
    try:
        heart.set_intro(12,'active')
        with pytest.raises(RuntimeError):heart.set_intro(12,'ready')
        p.ready=True;heart.set_intro(12,'ready')
        __import__('time').sleep(.03)
        assert any(x.get('op')=='heartbeat' and x.get('intro_status')=='ready' for x in p.calls)
        assert all(x.get('op')=='heartbeat' for x in p.calls)
    finally:heart.close()


def test_actual_guest_feedback_and_fake_rtde_direction_loop():
    import threading, time
    from urllib.parse import urlsplit
    from leica_tracker.guest_display import GuestDisplay
    from leica_tracker.person_guidance import GuidanceCue, GuidanceDecision
    from leica_tracker.physical_interaction_supervisor import PhysicalInteractionSupervisor
    from leica_tracker.resident_guest_watchdog import GuestControlWatchdog, LocalGuestControlReader, ResidentMotionGate
    from test_guest_display import request
    class Capture:
        def connected(self):return True
        def fresh(self):return True
        def stream_fresh(self):return True
        def diagnostics(self):return {}
    class Channel:
        def __init__(self):self.calls=[];self.closed=threading.Event()
        def send(self,*args):self.calls.append(args)
        def close(self):self.closed.set()
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        with request(display,'/action','POST',{'action':'start'}):pass
        identity=display.control_snapshot(); control={'instance':identity['instance'],'epoch':identity['epoch'],'started_ns':identity['control_started_ns']}
        channel=Channel();gate=ResidentMotionGate(channel)
        reader=LocalGuestControlReader(urlsplit(display.origin).port,display.token,timeout=.2)
        watch=GuestControlWatchdog(reader,gate,interval=.02,timeout=.3);watch.start()
        assert watch.wait_ready(1)
        ref=register_backend(urlsplit(display.origin).port,display.token,control,timeout=.2)
        publisher=FeedbackPublisher(ref,gate)
        profile=reviewed_profile(data());cfg=profile.cfg;engine=ResidentEngine(cfg,gate,123)
        sup=profile.supervisor(engine,ref); session=Session(cfg);session.engine=engine;session.gate=gate
        dashboard=Dashboard();run=PhysicalRuntime(session,sup,profile,dashboard,publisher,control=control)
        hold=GuidanceDecision('PHYSICAL_REFERENCE_HOLD',GuidanceCue.HOLD,(.5,.5,.2),None,None,False,'PHYSICAL_REFERENCE_HOLD')
        stop=threading.Event()
        def renew_visitor():
            sequence=0
            while not stop.is_set():
                try:
                    sequence+=1
                    with request(display,'/control-heartbeat','POST',{
                        'instance':control['instance'],'epoch':control['epoch'],'sequence':sequence}):pass
                except Exception:return
                stop.wait(.05)
        thread=threading.Thread(target=renew_visitor,daemon=True);thread.start()
        def tick(event=None):
            now=time.monotonic();sent=channel.calls[-1] if channel.calls else None
            xyz=engine.goal if engine.state=='MOVING' else cfg.neutral[:3]
            sample={'session':123,'state':sent[1] if sent else 2,'sequence':sent[0] if sent else 0,
                    'tcp_pose':list(xyz)+list(cfg.neutral[3:]),'tcp_speed':[0.]*6,'joints':list(cfg.joints)}
            session.writer.latest=(sample,now);session.reader.latest=({'joint_speed':[0.]*6,'tcp_speed':[0.]*6},now)
            session.states.latest=(session.states.latest[0],now);dashboard.checked_at=now
            run.step(event,now=now)
        try:
            tick();time.sleep(.03);tick();assert engine.state=='READY'
            event={'event':'person_guidance','coordinate_frame':'normalized_camera_image','robot_actuated':False,
                   'tracking':True,'cue':'MOVE_LEFT','source_monotonic_ns':time.monotonic_ns(),
                   'control_instance':control['instance'],'control_epoch':control['epoch']}
            tick(event);deadline=time.monotonic()+5;saw_motion=False;saw_light=False
            while time.monotonic()<deadline:
                with display._lock:blocked=display._reference_gate.blocked
                if blocked:display.publish(None,hold)
                tick()
                with display._lock:
                    fb=display._current_feedback()
                    saw_light |= fb.get('intensity',0)>0
                saw_motion |= any(c[1]==1 for c in channel.calls)
                if saw_motion and sup.state=='OBSERVING':break
                time.sleep(.03)
            assert saw_motion and saw_light and sup.state=='OBSERVING'
        finally:
            stop.set();thread.join(1);publisher.close();watch.close()
