"""引入了bootstrap，只使用注入的对象，并且从不连接到硬件。"""
import threading
import pytest

from leica_tracker.physical_intro_bootstrap import open_intro
from test_physical_intro_program import profile


def setup(*,stop_during_upload=False,wrong_reference=False):
    p=profile();objects={};now=[10.]
    class State:
        failure=None
        latest=({'mode':{'connected':True,'enabled':True,'power':True,'mode':7,
                         'emergency':False,'protective':False,'paused':False,'running':False},
                 'tcp_offset':p['tcp']},10.)
        def close(self):pass
    class Device:
        failure=None
        def __init__(self,reader):
            self.reader=reader;self.sent=[]
            self.latest=({'session':0,'state':0,'sequence':0} if not reader else
                         {'joints':list(p['reference_joints']),'joint_speed':[0.]*6,
                          'tcp_pose':list(p['reference_pose']),'tcp_speed':[0.]*6},10.)
            if reader and wrong_reference:self.latest[0]['tcp_pose'][0]+=.01
        def send(self,*args):self.sent.append(args)
        def close(self):pass
    class Gate:
        def __init__(self,device):self.device=device;self.can_send=False;self.close_finished=threading.Event();self.close_error=None
        def send(self,*args):
            if not self.can_send:raise RuntimeError('stopped')
            self.device.send(*args)
        def stop(self,reason):self.can_send=False;self.close_finished.set()
    class Watch:
        def __init__(self,read,gate,**kwargs):self.gate=gate;objects['watch']=self
        def start(self):self.gate.can_send=True
        def wait_ready(self,timeout):return self.gate.can_send
        def close(self):self.gate.stop('closed')
    class Command:
        def __init__(self):self.sent=[]
        def send(self,script):
            self.sent.append(script);objects['writer'].latest[0].update(session=123,state=100)
            objects['state'].latest[0]['mode']['running']=True
            if stop_during_upload:objects['watch'].close()
        def close(self):pass
    class Dashboard:
        failure=None;checked_at=10.
        def close(self):pass
    def state(host):objects['state']=State();return objects['state']
    def rtde(host,**kwargs):
        assert kwargs.get('spatial') is bool(kwargs.get('receive_only'))
        device=Device(bool(kwargs.get('receive_only')))
        objects['reader' if device.reader else 'writer']=device
        return device
    def command(host):objects['command']=Command();return objects['command']
    def dashboard(host):objects['dashboard']=Dashboard();return objects['dashboard']
    def dashboard_check(*args,**kwargs):objects.setdefault('dashboard_checks',[]).append((args,kwargs))
    deps={'state_factory':state,'rtde_factory':rtde,'command_factory':command,
          'dashboard_check':dashboard_check,'gate_factory':Gate,'watch_factory':Watch,
          'dashboard_factory':dashboard,
          'clock':lambda:now[0],'sleep':lambda delta:now.__setitem__(0,now[0]+delta),
          'session_factory':lambda:123}
    return p,deps,objects


def test_intro_bootstrap_uses_scalar_writer_and_does_not_enable_before_return():
    p,deps,objects=setup();result=open_intro('192.0.2.1',p,lambda:None,dependencies=deps)
    assert len(objects['command'].sent)==1 and objects['writer'].sent==[]
    assert result.progress.state=='STARTING'
    assert objects['dashboard_checks'] and all(call[1].get('robot_model')=='ur10e'
                                                for call in objects['dashboard_checks'])
    result.close()
    assert not objects['watch'].gate.can_send


def test_intro_bootstrap_rejects_wrong_start_before_upload():
    p,deps,objects=setup(wrong_reference=True)
    with pytest.raises(RuntimeError):open_intro('192.0.2.1',p,lambda:None,dependencies=deps)
    assert 'command' not in objects


def test_intro_stop_during_upload_cannot_enable_motion():
    p,deps,objects=setup(stop_during_upload=True)
    with pytest.raises(RuntimeError):open_intro('192.0.2.1',p,lambda:None,dependencies=deps)
    assert objects['writer'].sent==[] and not objects['watch'].gate.can_send


def test_intro_startup_stop_preserves_exact_watchdog_and_gate_cause():
    p,deps,objects=setup();calls=[0]
    def clock():
        calls[0]+=1
        if calls[0]==3:
            objects['watch'].failure='Guest control epoch or instance changed'
            objects['watch'].gate.reason='First epoch closure'
            objects['watch'].gate.can_send=False
        return 10.
    deps['clock']=clock
    with pytest.raises(RuntimeError) as error:
        open_intro('192.0.2.1',p,lambda:None,dependencies=deps)
    message=str(error.value)
    assert message.startswith('Guest watchdog stopped during intro startup')
    assert 'watchdog=Guest control epoch or instance changed' in message
    assert 'gate=First epoch closure' in message
    assert objects['writer'].sent==[]


def test_primary_stop_before_start_ack_cannot_be_treated_as_completion():
    p,deps,objects=setup();result=open_intro('192.0.2.1',p,lambda:None,dependencies=deps)
    try:
        objects['state'].latest[0]['mode']['running']=False
        with pytest.raises(RuntimeError,match='no longer running'):
            result.tick()
        assert objects['writer'].sent==[] and not result.gate.can_send
    finally:result.close()


def test_tick_snapshots_all_sources_before_reading_clock():
    p,deps,objects=setup();result=open_intro('192.0.2.1',p,lambda:None,dependencies=deps)
    def clock_advances_producers():
        data,_=objects['state'].latest
        objects['state'].latest=(data,10.001)
        objects['dashboard'].checked_at=10.001
        return 10.
    result.clock=clock_advances_producers
    assert result.tick()=='STARTING'
    assert objects['writer'].sent==[(1,0,123,0.)]
    result.close()


@pytest.mark.parametrize('check_clock_call',[1,2,4],ids=['first-check','pre-upload','startup-loop'])
@pytest.mark.parametrize('producer',['state','writer','reader'])
def test_bootstrap_snapshots_producers_before_each_check_clock(check_clock_call,producer):
    p,deps,objects=setup();calls=[0]
    def clock_with_concurrent_producer_update():
        calls[0]+=1
        now=10.+(calls[0]-1)*.001
        if calls[0]==check_clock_call:
            data,_=objects[producer].latest
            # 模型一个生产者在返回时钟实例后立即发布。这个新样本属于下一个验证周期。
            objects[producer].latest=(data,now+.0001)
        return now
    deps['clock']=clock_with_concurrent_producer_update
    result=open_intro('192.0.2.1',p,lambda:None,dependencies=deps)
    try:
        assert len(objects['command'].sent)==1
        assert objects['writer'].sent==[]
    finally:result.close()


@pytest.mark.parametrize('producer',['writer','reader'])
@pytest.mark.parametrize('stamp',[9.849,10.001],ids=['older-than-150ms','already-future'])
def test_bootstrap_still_rejects_stale_or_already_future_snapshots(producer,stamp):
    p,deps,objects=setup();factory=deps['rtde_factory']
    def device(host,**kwargs):
        result=factory(host,**kwargs)
        name='reader' if kwargs.get('receive_only') else 'writer'
        if name==producer:result.latest=(result.latest[0],stamp)
        return result
    deps['rtde_factory']=device
    with pytest.raises(RuntimeError,match='unavailable or stale'):
        open_intro('192.0.2.1',p,lambda:None,dependencies=deps)
    assert 'command' not in objects and objects['writer'].sent==[]
