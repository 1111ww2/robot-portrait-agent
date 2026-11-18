"""持续性访问使用一个源和明确的时段；完全离线。"""
import threading
from types import SimpleNamespace

import pytest

from leica_tracker import physical_runtime as runtime


def control(epoch=1, active=True, action='start'):
    return dict(instance='display', epoch=epoch, started_ns=epoch*100,
                active=active, control_action=action,
                camera_connected=True, camera_fresh=True)


class Source:
    failure=None
    def take(self):return None


def drive(snapshots, runner, checker, reports=None):
    remaining=list(snapshots)
    reports=[] if reports is None else reports
    def read():return remaining.pop(0)
    runtime.serve_sessions(SimpleNamespace(host='offline'),8767,'local-test',Source(),
        read_guest=read, session_runner=runner, stopped_check=checker,
        report=lambda snapshot,status,reason:reports.append((snapshot['epoch'],status,reason)),
        is_alive=lambda:bool(remaining), sleep=lambda _:None)
    return reports


def test_each_explicit_epoch_runs_once_after_previous_stop_is_measured():
    calls=[]
    def run(*args,**kwargs):calls.append(('run',kwargs['before']['epoch']));raise RuntimeError('Stopped')
    def stopped(host):calls.append(('stopped',host));return True
    drive([control(0,False,'stop'),control(1),control(1),control(2,False,'stop'),
           control(3,action='restart'),control(3)],run,stopped)
    assert calls==[('run',1),('stopped','offline'),('run',3),('stopped','offline')]


def test_fault_does_not_retry_same_epoch_and_reference_error_is_visible():
    calls=[]
    def run(*args,**kwargs):
        calls.append(kwargs['before']['epoch'])
        raise RuntimeError('Robot is not at the reviewed stationary intro reference')
    reports=drive([control(1),control(1),control(1),control(2)],run,lambda _:True)
    assert calls==[1,2]
    assert (1,'fault','reference') in reports


def test_unconfirmed_stop_blocks_next_session_until_later_explicit_start():
    calls=[];checks=iter([False,False,True,True])
    def run(*args,**kwargs):calls.append(kwargs['before']['epoch'])
    drive([control(1),control(2),control(2),control(3)],run,lambda _:next(checks))
    assert calls==[1,3]


def test_cleanup_failure_latches_launcher_without_new_robot_connection():
    calls=[]
    def run(*args,**kwargs):
        calls.append('run')
        raise runtime.SessionCleanupError('Old sender still alive')
    def stopped(_):calls.append('verify');return True
    drive([control(1),control(2),control(3)],run,stopped)
    assert calls==['run','verify']


def test_inactive_stale_camera_and_non_start_actions_never_open_session():
    stale=control(2);stale['camera_fresh']=False
    drive([control(1,False),stale,control(3,action='stop')],
          lambda *a,**k:pytest.fail('No motion session expected'),
          lambda *a:pytest.fail('No robot connection expected'))


def test_single_source_reader_survives_consumers_and_replaces_latest_event():
    incoming=__import__('queue').Queue()
    class Stream:
        def readline(self,_):return incoming.get(timeout=1)
    source=runtime.LatestEventSource(Stream())
    reader=source.thread
    incoming.put('{"round":1}\n')
    assert wait_event(source)=={'round':1}
    incoming.put('{"round":2}\n')
    assert wait_event(source)=={'round':2}
    assert source.thread is reader and reader.is_alive()
    incoming.put('');reader.join(timeout=1)
    assert not reader.is_alive() and source.failure


def wait_event(source):
    for _ in range(100):
        value=source.take()
        if value is not None:return value
        threading.Event().wait(.002)
    pytest.fail('reader did not publish event')


def test_publisher_and_heartbeat_close_join_their_threads():
    class Reference:
        def _request(self,*_):return {'ok':True}
    gate=SimpleNamespace(stop=lambda _:None)
    publisher=runtime.FeedbackPublisher(Reference(),gate)
    heartbeat=runtime.LeaseHeartbeat(publisher)
    heartbeat.close();publisher.close()
    assert not heartbeat._thread.is_alive()
    assert not publisher._thread.is_alive()
    assert not publisher._watch.is_alive()


def stop_probe(*, moving=False, stale=False, frozen=False):
    now=[10.];calls=[]
    class Channel:
        failure=None
        @property
        def latest(self):
            return dict(timestamp=10. if frozen else now[0],
                        tcp_speed=[.1 if moving else 0.]*6,joint_speed=[0.]*6),now[0]-(1. if stale else 0.)
        def close(self):calls.append('close-reader')
    def factory(host,**kwargs):
        assert kwargs=={'spatial':True,'receive_only':True}
        calls.append('receive-only');return Channel()
    def dashboard(host,**kwargs):
        assert kwargs=={'idle':True};calls.append('idle-check')
    def sleep(dt):now[0]+=dt
    return calls,dict(timeout=.5,clock=lambda:now[0],sleep=sleep,
                      channel_factory=factory,dashboard_check=dashboard)


def test_stop_confirmation_uses_advancing_independent_read_only_samples_and_idle():
    calls,options=stop_probe()
    assert runtime.verify_stationary_after_close('offline',**options) is True
    assert calls==['receive-only','idle-check','close-reader']


@pytest.mark.parametrize('case',['moving','stale','frozen'])
def test_stop_confirmation_rejects_moving_stale_or_repeated_samples(case):
    calls,options=stop_probe(**{case:True})
    with pytest.raises(RuntimeError):runtime.verify_stationary_after_close('offline',**options)
    assert calls==['receive-only','close-reader']


def test_run_session_closes_every_resource_even_if_heartbeat_close_fails(monkeypatch):
    calls=[]
    def heartbeat_close():calls.append('heartbeat-close');raise RuntimeError('close failed')
    heartbeat=SimpleNamespace(close=heartbeat_close)
    publisher=SimpleNamespace(close=lambda:calls.append('publisher-close'))
    reference=SimpleNamespace(status=lambda _:{'blocked':False},fault=lambda:calls.append('fault'))
    monkeypatch.setattr(runtime,'register_backend',lambda *a,**k:reference)
    monkeypatch.setattr(runtime,'FeedbackPublisher',lambda *a,**k:publisher)
    monkeypatch.setattr(runtime,'LeaseHeartbeat',lambda *a,**k:heartbeat)
    def trigger(*a,**k):raise RuntimeError('Trigger failed')
    monkeypatch.setattr(runtime,'wait_for_trigger',trigger)
    with pytest.raises(runtime.SessionCleanupError):
        runtime.run_session(SimpleNamespace(gate_timeout=.1),8767,'local-test',Source(),lambda:control(),before=control())
    assert calls==['heartbeat-close','publisher-close','fault']
