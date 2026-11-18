"""一个已完成的旧开放监控器不能毒害下一个居民的政策。"""
import threading
import json
from types import SimpleNamespace

import pytest

from leica_tracker.network_recovery import CommunicationRecovery
from leica_tracker.physical_intro_bootstrap import OpenIntro
from leica_tracker.physical_runtime import FeedbackPublisher
from leica_tracker.resident_guest_watchdog import GuestControlWatchdog,GuestSnapshotRequestError,ResidentMotionGate


CONTROL=dict(instance='same-encounter',epoch=1,started_ns=1)


def snapshot(**change):
    return {**CONTROL,'active':True,'control_fresh':True,'camera_connected':True,'camera_fresh':True,**change}


class Channel:
    def __init__(self):self.normal=threading.Event();self.hard=threading.Event()
    def close_completed(self):self.normal.set()
    def close(self):self.hard.set()


def monitoring(read_result):
    entered=threading.Event();release=threading.Event()
    policy=CommunicationRecovery(CONTROL)
    channel=Channel();gate=ResidentMotionGate(channel);gate.arm()
    def reader():
        entered.set()
        assert release.wait(1.)
        if isinstance(read_result,Exception):raise read_result
        return read_result
    watch=GuestControlWatchdog(reader,gate,interval=.01,timeout=.5,recovery=policy)
    watch.start();assert entered.wait(.5)
    return policy,channel,gate,watch,release


def test_completed_intro_stops_old_monitor_admission_before_retiring_its_gate():
    policy,channel,gate,watch,release=monitoring(snapshot())
    retire=gate.retire_completed
    def release_old_snapshot(reason):
        retire(reason)
        release.set()
        watch._threads[0].join(timeout=.5)
    gate.retire_completed=release_old_snapshot
    progress=SimpleNamespace(state='COMPLETE',stop=lambda:gate.stop('progress stopped'))
    resource=SimpleNamespace(close=lambda:None)
    intro=OpenIntro(progress,gate,watch,resource,channel,resource,resource,resource,lambda:10.)
    try:
        intro.close_completed()
        assert channel.normal.is_set() and not channel.hard.is_set()
        assert policy.failure is None and policy.status=='healthy'
    finally:release.set();watch.close();intro.close()


@pytest.mark.parametrize('result',[
    {**snapshot(),'active':False},
    GuestSnapshotRequestError('timeout','request',100.),
    RuntimeError('late old monitor error'),
])
def test_inflight_retired_monitor_never_mutates_shared_policy(result):
    policy,channel,gate,watch,release=monitoring(result)
    try:
        watch.close()
        release.set();watch._threads[0].join(timeout=.5)
        assert not watch._threads[0].is_alive()
        assert policy.failure is None and policy.status=='healthy' and policy.deadline is None
    finally:release.set();watch.close()


def test_already_stopped_monitor_cannot_request_normal_completed_retirement():
    policy,channel,gate,watch,release=monitoring(snapshot())
    try:
        watch.close()
        with pytest.raises(RuntimeError):watch.retire_completed()
        assert channel.hard.wait(.5) and not channel.normal.is_set()
    finally:
        release.set();watch.close();watch._threads[0].join(timeout=.5)


def test_new_resident_monitor_still_terminates_on_real_stop_after_old_monitor_retired():
    policy,channel,gate,watch,release=monitoring(snapshot())
    active=[True];new_channel=Channel();new_gate=ResidentMotionGate(new_channel)
    new_watch=GuestControlWatchdog(lambda:snapshot(active=active[0]),new_gate,
        interval=.01,timeout=.2,recovery=policy)
    try:
        watch.retire_completed();gate.retire_completed('completed')
        release.set();watch._threads[0].join(timeout=.5)
        assert policy.status=='healthy'
        new_watch.start();assert new_watch.wait_ready(.5)
        active[0]=False
        assert new_watch.stopped.wait(.5) and new_channel.hard.wait(.5)
        assert not new_gate.can_send and policy.failure is not None
    finally:
        release.set();watch.close();new_watch.close()


@pytest.mark.parametrize('error,kind',[(RuntimeError('private-controller-text'),'RuntimeError'),
                                    (OSError('private-controller-text'),'OSError')])
def test_feedback_unclassified_failure_keeps_only_safe_stage_and_policy_flags(error,kind):
    policy=CommunicationRecovery(CONTROL);stopped=threading.Event()
    class Reference:
        def feedback_request(self,body):
            try:policy.fail('private-controller-text')
            except RuntimeError:pass
            raise error
    publisher=FeedbackPublisher(Reference(),SimpleNamespace(stop=lambda reason:stopped.set()),recovery=policy)
    try:
        publisher.submit({'op':'heartbeat'})
        assert stopped.wait(1.)
        prefix='Physical feedback unavailable [unclassified] '
        assert publisher.failed.startswith(prefix+'{')
        details=json.loads(publisher.failed[len(prefix):])
        assert details==dict(stage='send',op='heartbeat',exception_kind=kind,recovery_failed=True)
        assert 'private' not in publisher.failed
    finally:publisher.close()
