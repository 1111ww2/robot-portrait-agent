"""跨居民过渡测试仅使用假设备。"""
from types import SimpleNamespace
import pytest

from leica_tracker.physical_runtime import run_intro_then_open_xyz


def fixture(*,stop_at=None,close_failure=False):
    events=[];clock=[10.]
    class Switch:
        failure=None
        def attach(self,gate):
            if self.failure:raise RuntimeError(self.failure)
            events.append('attach_'+gate.name)
        def detach(self,gate):events.append('detach_'+gate.name)
        def detach_completed(self,gate,begin_retirement):
            if self.failure:raise RuntimeError(self.failure)
            begin_retirement();self.detach(gate)
        def stop(self,reason):self.failure=reason;events.append('stop')
    switch=Switch()
    class Source:
        failure=None
        def take(self):return {'event':'physical_intro','action':'heartbeat','intro_id':8,
                               'control_instance':'test','control_epoch':1,
                               'source_monotonic_ns':round(clock[0]*1e9)}
    class Lease:
        failure=None
        def set_intro(self,intro,status):events.append(status)
    class Publisher:
        failed=None
        def intro_acknowledged(self,intro,status):return status in events
    class Data:
        failure=None
        @property
        def latest(self):return {},clock[0]
    class Intro:
        def __init__(self):
            self.gate=SimpleNamespace(name='intro');self.writer=Data();self.reader=Data()
            self.progress=SimpleNamespace(state='STARTING',started_ack=False,last_writer=None,last_reader=None)
        def tick(self):
            events.append('intro_tick');self.progress.started_ack=True
            self.progress.last_writer=clock[0];self.progress.last_reader=clock[0]
            self.progress.state='COMPLETE'
            if stop_at=='intro':switch.stop('operator stopped')
        def close(self):
            events.append('intro_close')
            if stop_at=='close':switch.stop('operator stopped')
            if close_failure:raise RuntimeError('close failed')
        def begin_completed_retirement(self):events.append('intro_begin_retirement')
        def close_completed(self):self.begin_completed_retirement();self.close()
    class Session:
        def __init__(self):self.gate=SimpleNamespace(name='xyz');self.engine=SimpleNamespace(state='HANDSHAKE')
        def close(self):events.append('xyz_close')
    class Runtime:
        def __init__(self,session,*args,**kwargs):self.session=session
        def step(self,event):events.append('xyz_hold');self.session.engine.state='READY'
    class Dashboard:
        def close(self):events.append('dashboard_close')
    class Profile:
        host='192.0.2.1';intro={};cfg={}
        def supervisor(self,engine,reference):return None
    def intro_open(*args,**kwargs):
        assert kwargs=={'require_camera':False}
        events.append('intro_open');return Intro()
    def xyz_open(*args,**kwargs):
        assert kwargs=={'require_camera':False}
        events.append('xyz_open');return Session()
    def sleep(delta):clock[0]+=delta
    kwargs={'control':{'instance':'test','epoch':1,'started_ns':1},
            'intro_event':{'event':'physical_intro','action':'heartbeat','control_instance':'test','control_epoch':1,
                           'intro_id':8,'source_monotonic_ns':round(clock[0]*1e9)},'reference':object(),
            'publisher':Publisher(),'clock':lambda:clock[0],'sleep':sleep,'intro_opener':intro_open,
            'xyz_opener':xyz_open,'dashboard_factory':lambda host:Dashboard(),'runtime_factory':Runtime}
    def read_control():
        return {**kwargs['control'],'active':True,'control_fresh':True,'camera_connected':True,'camera_fresh':True}
    return Profile(),read_control,Source(),Lease(),switch,kwargs,events


def test_intro_finishes_and_writer_closes_before_xyz_then_ready():
    profile,reader,source,lease,switch,kwargs,events=fixture()
    run_intro_then_open_xyz(profile,reader,source,lease,switch,**kwargs)
    assert events.index('intro_tick')<events.index('active')<events.index('intro_close')<events.index('xyz_open')
    assert events.index('xyz_hold')<events.index('ready')


def test_failed_completed_retirement_never_opens_xyz():
    from leica_tracker.physical_errors import SessionCleanupError
    profile,reader,source,lease,switch,kwargs,events=fixture()
    opener=kwargs['intro_opener']
    def intro_open(*args,**options):
        intro=opener(*args,**options)
        def fail_completed():
            events.append('completed_close_failed')
            raise SessionCleanupError('SDK disconnect did not complete')
        intro.close=fail_completed
        return intro
    kwargs['intro_opener']=intro_open
    with pytest.raises(SessionCleanupError):
        run_intro_then_open_xyz(profile,reader,source,lease,switch,**kwargs)
    assert 'completed_close_failed' in events and 'xyz_open' not in events


@pytest.mark.parametrize('stop_at,close_failure',[('intro',False),('close',False),(None,True)])
def test_stop_or_close_failure_never_starts_xyz(stop_at,close_failure):
    profile,reader,source,lease,switch,kwargs,events=fixture(stop_at=stop_at,close_failure=close_failure)
    with pytest.raises(RuntimeError):run_intro_then_open_xyz(profile,reader,source,lease,switch,**kwargs)
    assert 'xyz_open' not in events and 'ready' not in events


def test_return_handoff_waits_through_owner_gap_before_opening_xyz_without_replaying_intro():
    from leica_tracker.network_recovery import CommunicationRecovery
    profile,original_reader,source,lease,switch,kwargs,events=fixture()
    clock=kwargs['clock'];control=kwargs['control']
    recovery=CommunicationRecovery(control,clock=clock)
    take=source.take
    def fresh_event():
        event=take();event['source_monotonic_ns']=int(clock()*1e9)-1000
        return event
    source.take=fresh_event
    def reader():
        state=original_reader()
        state['control_fresh']=clock()>=10.2
        return state
    reader.communication_recovery=recovery
    xyz_open=kwargs['xyz_opener']
    def held_xyz_open(*args,**options):
        assert clock()>=10.5 and recovery.status=='ready' and recovery.paused
        return xyz_open(*args,**options)
    kwargs['xyz_opener']=held_xyz_open
    run_intro_then_open_xyz(profile,reader,source,lease,switch,**kwargs)
    assert events.count('intro_open')==1 and events.count('intro_tick')==1
    assert events.index('intro_close')<events.index('xyz_open')<events.index('ready')
    assert recovery.paused and recovery.deadline==pytest.approx(13.005)
