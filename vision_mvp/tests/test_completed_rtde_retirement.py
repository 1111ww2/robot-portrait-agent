"""完成的退休开放使用假SDK资源和真实的当地儿童。"""
import io
import json
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from leica_tracker import atomic_rtde_transport as atomic
from leica_tracker import ur_rtde_transport as transport
from leica_tracker.physical_errors import SessionCleanupError
from leica_tracker.physical_intro_bootstrap import OpenIntro
from leica_tracker.resident_guest_watchdog import ResidentMotionGate


SDK_PROGRAM='''
import sys,time
from types import SimpleNamespace
from leica_tracker.ur_rtde_transport import _sdk_worker
class Receive:
    def __init__(self,*args):self.stamp=0.
    def isConnected(self):self.stamp+=.02;return True
    def getTimestamp(self):return self.stamp
    def getOutputIntRegister(self,index):return {42:102,43:1,40:123}[index]
    def getActualTCPSpeed(self):return [0.]*6
    def disconnect(self):print('SDK_RECEIVE_DISCONNECT',file=sys.stderr,flush=True)
class IO:
    def __init__(self,*args):pass
    def isConnected(self):return True
    def disconnect(self):print('SDK_IO_DISCONNECT',file=sys.stderr,flush=True)
sys.modules['rtde_receive']=SimpleNamespace(RTDEReceiveInterface=Receive)
sys.modules['rtde_io']=SimpleNamespace(RTDEIOInterface=IO)
raise SystemExit(_sdk_worker('unused.invalid'))
'''


def sdk_channel(monkeypatch,program=SDK_PROGRAM):
    real_popen=subprocess.Popen;children=[]
    def create(command,**kwargs):
        child=real_popen([sys.executable,'-B','-u','-c',program],stderr=subprocess.PIPE,**kwargs)
        children.append(child);return child
    monkeypatch.setattr(transport.subprocess,'Popen',create)
    return transport.RTDEChannel('unused.invalid'),children


def test_completed_retirement_runs_real_sdk_finally_before_returning(monkeypatch):
    channel,children=sdk_channel(monkeypatch)
    try:
        getattr(channel,'close_completed',channel.close)()
        assert children[0].returncode==0
        diagnostics=children[0].stderr.read()
        assert 'SDK_IO_DISCONNECT' in diagnostics and 'SDK_RECEIVE_DISCONNECT' in diagnostics
        assert not channel.reader.is_alive() and not channel.writer.is_alive()
        with pytest.raises(RuntimeError):channel.send(1,1,123,0.)
    finally:
        channel.close();children[0].stderr.close()


def test_fault_close_remains_hard_and_does_not_wait_for_sdk_finally(monkeypatch):
    channel,children=sdk_channel(monkeypatch)
    started=time.monotonic();channel.close()
    try:
        assert time.monotonic()-started<1.5 and children[0].returncode!=0
        assert 'SDK_IO_DISCONNECT' not in children[0].stderr.read()
    finally:children[0].stderr.close()


def test_uncooperative_completed_worker_is_killed_and_never_reported_as_graceful(monkeypatch):
    program=('import json,time;print(json.dumps({"feedback":{"timestamp":1.,"state":102,'
             '"sequence":1,"session":123,"tcp_speed":[0.]*6},"sampled_at":time.monotonic()}),flush=True);time.sleep(30)')
    channel,children=sdk_channel(monkeypatch,program)
    started=time.monotonic()
    try:
        with pytest.raises(SessionCleanupError):
            getattr(channel,'close_completed',channel.close)()
        assert time.monotonic()-started<2. and children[0].poll() is not None
    finally:channel.close();children[0].stderr.close()


def test_failed_sdk_disconnect_cannot_be_reported_as_completed_retirement(monkeypatch):
    program=SDK_PROGRAM.replace("print('SDK_IO_DISCONNECT',file=sys.stderr,flush=True)",
                                "raise RuntimeError('offline disconnect failure')")
    channel,children=sdk_channel(monkeypatch,program)
    try:
        with pytest.raises(SessionCleanupError):
            getattr(channel,'close_completed',channel.close)()
        assert children[0].poll() is not None
        assert 'SDK_RECEIVE_DISCONNECT' in children[0].stderr.read()
    finally:channel.close();children[0].stderr.close()


def completed_intro(channel,state='COMPLETE'):
    gate=ResidentMotionGate(channel);gate.arm()
    progress=SimpleNamespace(state=state)
    def stop():progress.state='STOPPED';gate.stop('intro stopped')
    progress.stop=stop
    resource=SimpleNamespace(close=lambda:None)
    watch=SimpleNamespace(close=lambda:gate.stop('watch closed'),retire_completed=lambda:None,_threads=[])
    return OpenIntro(progress,gate,watch,resource,channel,resource,resource,resource,time.monotonic)


def test_actual_completed_intro_closes_gate_before_successful_sdk_retirement(monkeypatch):
    channel,children=sdk_channel(monkeypatch);intro=completed_intro(channel)
    try:
        intro.close_completed()
        assert intro.closed and not intro.gate.can_send and intro.gate.close_error is None
        assert intro.gate.close_finished.is_set() and children[0].returncode==0
        assert 'SDK_IO_DISCONNECT' in children[0].stderr.read()
        with pytest.raises(RuntimeError):intro.gate.send(1,1,123,0.)
    finally:intro.close();children[0].stderr.close()


@pytest.mark.parametrize('state',['ACTIVE','FAULT','STOPPED'])
def test_incomplete_intro_cannot_request_completed_retirement(monkeypatch,state):
    channel,children=sdk_channel(monkeypatch);intro=completed_intro(channel,state)
    try:
        with pytest.raises(SessionCleanupError):intro.close_completed()
        assert not intro.closed
    finally:intro.close();children[0].stderr.close()


def test_prior_stop_cannot_be_upgraded_to_graceful_completed_retirement(monkeypatch):
    channel,children=sdk_channel(monkeypatch);intro=completed_intro(channel)
    intro.gate.stop('operator stop')
    try:
        with pytest.raises(SessionCleanupError):intro.close_completed()
        assert intro.gate.close_finished.wait(2.) and children[0].returncode!=0
        assert 'SDK_IO_DISCONNECT' not in children[0].stderr.read()
    finally:intro.close();children[0].stderr.close()


def test_malformed_sdk_input_cannot_exit_as_successful_completed_retirement(monkeypatch):
    channel,children=sdk_channel(monkeypatch)
    try:
        channel.process.stdin.write('not-json\n');channel.process.stdin.flush()
        assert children[0].wait(timeout=1.)!=0
        with pytest.raises(SessionCleanupError):channel.close_completed()
        diagnostics=children[0].stderr.read()
        assert 'SDK_IO_DISCONNECT' in diagnostics and 'SDK_RECEIVE_DISCONNECT' in diagnostics
    finally:channel.close();children[0].stderr.close()


def test_completed_retirement_hard_falls_back_when_parent_writer_is_blocked(monkeypatch):
    channel,children=sdk_channel(monkeypatch);original=channel.process.stdin
    entered=threading.Event();release=threading.Event();closed_while_alive=[]
    class BlockedPipe:
        def write(self,value):entered.set();release.wait(3.)
        def flush(self):pass
        def close(self):
            closed_while_alive.append(channel.process.poll() is None)
            release.set();original.close()
    channel.process.stdin=BlockedPipe()
    channel.pending.put_nowait(dict(update=[1,0,123,0.],deadline=time.monotonic()+10.))
    assert entered.wait(1.)
    started=time.monotonic()
    try:
        with pytest.raises(SessionCleanupError):
            getattr(channel,'close_completed',channel.close)()
        assert time.monotonic()-started<2. and children[0].poll() is not None
        assert closed_while_alive==[False]
    finally:
        release.set();channel.close();children[0].stderr.close()


def test_completed_retirement_kills_worker_if_closing_stdin_itself_blocks(monkeypatch):
    channel,children=sdk_channel(monkeypatch);original=channel.process.stdin
    entered=threading.Event();release=threading.Event();done=threading.Event();errors=[]
    class BlockedEOF:
        def close(self):entered.set();release.wait(3.);original.close()
    channel.process.stdin=BlockedEOF()
    def retire():
        try:channel.close_completed()
        except Exception as error:errors.append(error)
        finally:done.set()
    thread=threading.Thread(target=retire,daemon=True);thread.start()
    try:
        assert entered.wait(1.)
        assert done.wait(1.5),'A blocking stdin close must trigger bounded hard cleanup'
        assert len(errors)==1 and isinstance(errors[0],SessionCleanupError)
        assert children[0].poll() is not None
    finally:
        release.set();thread.join(timeout=2.)
        channel.process.stdin=original;channel.close();children[0].stderr.close()


@pytest.mark.parametrize('kind',['sdk','atomic'])
def test_worker_discards_queued_enable_when_eof_arrives_during_feedback(monkeypatch,kind):
    ended=[False];writes=[];closed=[]
    class Event:
        def is_set(self):return ended[0]
        def set(self):ended[0]=True
        def wait(self,delay):pass
    class Thread:
        def __init__(self,**kwargs):pass
        def start(self):pass
    incoming=queue.Queue(maxsize=1)
    incoming.put_nowait(dict(update=[1,1,123,[0,0,0] if kind=='atomic' else 0.],deadline=time.monotonic()+60.))
    adapter=atomic if kind=='atomic' else transport
    monkeypatch.setattr(adapter,'threading',SimpleNamespace(Event=Event,Thread=Thread))
    monkeypatch.setattr(adapter,'queue',SimpleNamespace(Queue=lambda **kwargs:incoming,Empty=queue.Empty))
    if kind=='atomic':
        class Channel:
            def __init__(self,*args,**kwargs):pass
            def poll(self,**kwargs):ended[0]=True;return []
            def send(self,*args):writes.append(args)
            def close(self):closed.append(True)
        result=atomic.worker('unused.invalid',ack_clock_factory=transport.AckReceiptClock,
            validate_update=transport.validate_spatial_target,channel_factory=Channel,
            input_stream=iter(()),output_stream=io.StringIO(),clock=lambda:10.)
    else:
        class Resource:
            def __init__(self,*args):pass
            def isConnected(self):return True
            def disconnect(self):closed.append(True)
            def setInputIntRegister(self,*args):writes.append(args);return True
            def setInputDoubleRegister(self,*args):writes.append(args);return True
        monkeypatch.setitem(sys.modules,'rtde_receive',SimpleNamespace(RTDEReceiveInterface=Resource))
        monkeypatch.setitem(sys.modules,'rtde_io',SimpleNamespace(RTDEIOInterface=Resource))
        def feedback(*args,**kwargs):
            ended[0]=True
            return dict(timestamp=1.,state=102,sequence=1,session=123,tcp_speed=[0.]*6)
        monkeypatch.setattr(transport,'read_feedback',feedback)
        monkeypatch.setattr(transport.time,'monotonic',lambda:10.)
        result=transport._sdk_worker('unused.invalid')
    assert result==0 and closed and writes==[]


def timed_retirement_channel(monkeypatch,*,writer=.01,eof=.01,process=1.06,reader=.01,exit_code=0):
    """每次假阻塞操作都会消耗其自身的实际时钟间隔。"""
    now=[10.];budgets=[]
    class Thread:
        def __init__(self,delay):self.delay=delay;self.alive=True
        def join(self,timeout):
            if self.alive:
                budgets.append(('join',timeout));now[0]+=min(timeout,self.delay)
                self.alive=self.delay>timeout
        def is_alive(self):return self.alive
    class Pipe:
        def __init__(self,delay=0.):self.delay=delay
        def close(self):now[0]+=self.delay
    class Process:
        def __init__(self):self.stdin=Pipe(eof);self.stdout=Pipe();self.returncode=None
        def poll(self):return self.returncode
        def wait(self,timeout):
            if self.returncode is not None:return self.returncode
            budgets.append(('process',timeout));now[0]+=min(timeout,process)
            if process>timeout:raise subprocess.TimeoutExpired('unused',timeout)
            self.returncode=exit_code;return exit_code
        def terminate(self):self.returncode=1
        def kill(self):self.returncode=1
    channel=object.__new__(transport.RTDEChannel)
    channel.closed=False;channel.failure=None
    channel.writer=Thread(writer);channel.reader=Thread(reader);channel.process=Process()
    monkeypatch.setattr(transport,'time',SimpleNamespace(monotonic=lambda:now[0]))
    return channel,now,budgets


def test_measured_1060ms_sdk_cleanup_fits_completed_retirement(monkeypatch):
    channel,now,budgets=timed_retirement_channel(monkeypatch)
    channel.close_completed()
    assert channel.process.returncode==0 and now[0]==pytest.approx(11.09)
    assert any(kind=='process' and timeout>.5 for kind,timeout in budgets)


@pytest.mark.parametrize('process,reader,phase',[(1.44,.01,'process'),(1.40,.20,'reader')])
def test_normal_retirement_stages_share_one_absolute_1800ms_deadline(monkeypatch,process,reader,phase):
    channel,now,budgets=timed_retirement_channel(monkeypatch,writer=.19,eof=.19,process=process,reader=reader)
    with pytest.raises(SessionCleanupError) as failure:channel.close_completed()
    details=failure.value.retirement_details
    assert details['phase']==phase and details['category']=='timeout'
    assert details['duration_s']==pytest.approx(1.8)
    assert channel.process.poll() is not None
    assert all(timeout<=1.5 for kind,timeout in budgets if kind=='process')


def test_nonzero_cleanup_reports_bounded_diagnostic_before_hard_fallback(monkeypatch,capsys):
    channel,now,budgets=timed_retirement_channel(monkeypatch,process=.1,exit_code=3)
    gate=ResidentMotionGate(channel);gate.arm();gate.retire_completed('normal completion')
    assert gate.close_finished.wait(2.)
    assert gate.close_details==dict(phase='process',category='nonzero_exit',duration_s=.12,exit_code=3)
    line=capsys.readouterr().err.strip()
    assert line.startswith('RTDE_COMPLETED_RETIREMENT_FAILED ')
    assert json.loads(line.split(' ',1)[1])==gate.close_details


@pytest.mark.parametrize('change',[
    {'phase':'private-controller-text'}, {'category':'private-controller-text'},
    {'duration_s':float('nan')}, {'exit_code':'private-controller-text'},
    {'unexpected':'private-controller-text'},
])
def test_retirement_failure_log_rejects_untrusted_diagnostic_fields(capsys,change):
    details=dict(phase='process',category='timeout',duration_s=.5,exit_code=None)
    details.update(change)
    class Channel:
        def close_completed(self):
            error=SessionCleanupError('private-controller-text')
            error.retirement_details=details
            raise error
    gate=ResidentMotionGate(Channel());gate.arm();gate.retire_completed('normal completion')
    assert gate.close_finished.wait(1.) and gate.close_error
    assert gate.close_details is None and capsys.readouterr().err==''
