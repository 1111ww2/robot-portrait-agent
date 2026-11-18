"""状态覆盖后仅保留一条尚未接纳且仍新鲜的 MOVE，不重试运动。"""
from contextlib import contextmanager
import json
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from leica_tracker.physical_runtime import LatestEventSource
from leica_tracker import physical_runtime as runtime_module
from test_physical_runtime import runtime as fake_runtime
from test_physical_interaction_supervisor import setup,ready,step


CONTROL=dict(instance='x',epoch=1,started_ns=9_000_000_000)


def direction(ns=10_000_000_000,cue='MOVE_RIGHT',**updates):
    return dict(dict(event='person_guidance',coordinate_frame='normalized_camera_image',
        robot_actuated=False,tracking=True,phase='GUIDING',cue=cue,
        control_instance='x',control_epoch=1,source_monotonic_ns=ns),**updates)


def waiting(ns=10_010_000_000):
    return direction(ns,cue='WAIT',phase='WAITING_FOR_RESPONSE',reason='WAITING_FOR_RESPONSE')


class Stream:
    def __init__(self):self.inbox=queue.Queue();self.condition=threading.Condition();self.reads=0;self.sent=0
    def readline(self,_limit):
        with self.condition:self.reads+=1;self.condition.notify_all()
        return self.inbox.get(timeout=2.)
    def send(self,*values):
        self.sent+=len(values)
        for value in values:self.inbox.put(json.dumps(value)+'\n')
        with self.condition:assert self.condition.wait_for(lambda:self.reads>=self.sent+1,timeout=1.)


@contextmanager
def source_at(now=10.02,control=CONTROL):
    clock=[now];stream=Stream();source=LatestEventSource(stream,clock=lambda:clock[0])
    source.bind_control(control)
    try:yield source,stream,clock
    finally:
        stream.inbox.put('');source.thread.join(1.)
        assert not source.thread.is_alive()


def test_serialized_move_then_wait_before_take_retains_original_event_until_ack():
    with source_at() as (source,stream,clock):
        move=direction();stream.send(move,waiting())
        assert source.take()==move and source.take()==move
        offered=source.take();offered['cue']='MOVE_LEFT'
        assert source.take()==move  # 消费方修改自己的数据不能改写队列中已有的意图。
        source.acknowledge(('x',1,9_000_000_000,10_000_000_000))
        assert source.take()==waiting() and source.take() is None
        stream.send(move)
        assert source.take() is None  # 同一源标识一经接纳不能再次引入。


def test_original_serialized_slot_drops_move_when_wait_arrives_before_first_take():
    stream=Stream();source=LatestEventSource(stream);ns=time.monotonic_ns()-20_000_000
    bind=getattr(source,'bind_control',None)
    if callable(bind):bind(dict(CONTROL,started_ns=ns-1_000_000_000))
    move=direction(ns);wait=waiting(ns+1)
    try:
        stream.send(move,wait)
        assert source.take()==move
    finally:
        stream.inbox.put('');source.thread.join(1.)


def test_pending_release_ack_does_not_consume_intent_and_ready_admits_once_with_original_time():
    cfg,gate,_engine,_ref,sup=setup();ready(sup,cfg,gate)
    control=dict(instance='guest-a',epoch=1,started_ns=1)
    with source_at(1.11,control) as (source,stream,clock):
        move=direction(1_106_000_000,control_instance='guest-a')
        wait=dict(move,source_monotonic_ns=1_108_000_000,cue='WAIT',phase='WAITING_FOR_RESPONSE',reason='WAITING_FOR_RESPONSE')
        stream.send(move,wait)
        sup.state='RELEASE_PENDING';sup._job_kind='release';sup._job_started=1.10
        sup._job=SimpleNamespace(done=False,error=None,value=None,elapsed=None)
        assert sup.accept_event(source.take(),now=clock[0]) is False
        sup._job=SimpleNamespace(done=True,error=None,value=1_100_000_000,elapsed=.02)
        clock[0]=1.14;step(sup,cfg,gate,clock[0])
        assert sup.state=='OBSERVING'
        offered=source.take();assert offered==move
        assert sup.accept_event(offered,now=clock[0]) is True
        source.acknowledge(('guest-a',1,1,move['source_monotonic_ns']))
        assert source.take()==wait and source.take() is None
        assert sup.state=='BEGIN_PENDING'


def test_newer_move_replaces_one_intent_and_old_ack_cannot_erase_it():
    with source_at(10.05) as (source,stream,_):
        first=direction(9_990_000_000);newer=direction(10_020_000_000,cue='MOVE_LEFT')
        stream.send(first,newer,direction(10_000_000_000,cue='MOVE_UP'))
        source.acknowledge(('x',1,9_000_000_000,first['source_monotonic_ns']))
        assert source.take()==newer
        source.acknowledge(('x',1,8_000_000_000,newer['source_monotonic_ns']))
        assert source.take()==newer
        source.acknowledge(('x',1,9_000_000_000,newer['source_monotonic_ns']))
        assert source.take() is None


@pytest.mark.parametrize('old_ns',[8_999_999_999,9_500_000_000,11_000_000_000])
def test_old_scope_backlog_after_bind_cannot_poison_fresh_current_intent(old_ns):
    with source_at() as (source,stream,_):
        move=direction()
        stream.send(direction(old_ns,control_epoch=0),move,waiting())
        assert source.take()==move


def test_older_foreign_scope_cannot_clear_newer_pending_but_fresh_newer_scope_invalidates():
    with source_at() as (source,stream,_):
        move=direction()
        stream.send(move,direction(9_990_000_000,control_epoch=0),waiting())
        assert source.take()==move
        stream.send(direction(10_015_000_000,control_epoch=2),direction(10_016_000_000))
        assert source.take() is None


@pytest.mark.parametrize('newer',[
    direction(10_010_000_000,tracking=False,cue='HOLD',phase='NO_PERSON'),
    direction(10_010_000_000,cue='ARRIVED',phase='ARRIVED'),
    direction(10_010_000_000,tracking=False,cue='HOLD',phase='SETTLING',reason='TARGET_PENDING_PHYSICAL_STOP'),
    direction(10_010_000_000,tracking=False,cue='HOLD',phase='AI_API_ERROR'),
    direction(10_010_000_000,event='physical_intro',action='stop'),
    direction(10_010_000_000,control_epoch=2),
    direction(11_000_000_000),
])
def test_newer_loss_photo_error_stop_control_or_future_input_clears_pending(newer):
    with source_at() as (source,stream,_):
        stream.send(direction(),newer)
        offered=source.take()
        assert offered is None or offered.get('event')!='person_guidance' or offered.get('cue') not in ('MOVE_LEFT','MOVE_RIGHT','MOVE_UP','MOVE_DOWN','MOVE_CLOSER','MOVE_FARTHER')


def test_expiry_and_bind_unbind_and_source_failure_cannot_replay_old_intent():
    with source_at() as (source,stream,clock):
        move=direction();stream.send(move)
        clock[0]=10.301;assert source.take() is None
        clock[0]=10.02;stream.send(dict(move,source_monotonic_ns=10_005_000_000))
        source.bind_control(CONTROL);assert source.take() is None
        stream.send(dict(move,source_monotonic_ns=10_006_000_000))
        source.unbind_control();source.bind_control(CONTROL);assert source.take() is None
        stream.send(dict(move,source_monotonic_ns=10_007_000_000))
        stream.inbox.put('invalid-json\n');source.thread.join(1.)
        assert source.failure and source.take() is None


def test_runtime_ack_identity_exists_only_after_native_admission_and_light_success():
    run,_session,sup,publisher=fake_runtime()
    move=direction()
    sup.state='RELEASE_PENDING'
    sup.accept_event=lambda event,now:False
    assert run.step(move,now=10.) is None
    sup.accept_event=lambda event,now:True
    assert run.step(move,now=10.)==('x',1,9_000_000_000,10_000_000_000)
    assert len([x for x in publisher.calls if x['op']=='begin'])==1
    run2,_session2,sup2,_pub2=fake_runtime();sup2.accept_event=lambda event,now:True
    run2._start_light=lambda event:(_ for _ in ()).throw(RuntimeError('light failed'))
    with pytest.raises(RuntimeError,match='light failed'):run2.step(move,now=10.)


@pytest.mark.parametrize('admitted',[False,True])
def test_run_session_binds_before_restart_consumption_acks_only_admission_and_unbinds(monkeypatch,admitted):
    from test_session_cancellation import session_fixture
    profile,control,source,stopped,calls,_session=session_fixture(monkeypatch)
    source.bind_control=lambda value:calls.append(('bind',dict(value)))
    source.unbind_control=lambda:calls.append('unbind')
    source.acknowledge=lambda identity:calls.append(('ack',identity))
    class Runtime:
        def __init__(self,*args,**kwargs):pass
        def step(self,event):
            calls.append('runtime_step');stopped[0]=True
            return (control['instance'],control['epoch'],control['started_ns'],event['source_monotonic_ns']) if admitted else None
    monkeypatch.setattr(runtime_module,'PhysicalRuntime',Runtime)
    with pytest.raises(RuntimeError,match='service stop requested'):
        runtime_module.run_session(profile,8767,'fake',source,lambda:control,
            before=control,should_stop=lambda:stopped[0])
    bind_index=next(i for i,item in enumerate(calls) if isinstance(item,tuple) and item[0]=='bind')
    assert bind_index<calls.index('session_open') and calls.count('unbind')==1
    acks=[item for item in calls if isinstance(item,tuple) and item[0]=='ack']
    assert len(acks)==int(admitted)
    if admitted:assert calls.index('runtime_step')<calls.index(acks[0])<calls.index('unbind')
