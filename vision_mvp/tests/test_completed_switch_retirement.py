"""完成手交授权比赛；仅限内存通道，无协议。"""
import threading

import pytest

from leica_tracker.network_recovery import CommunicationRecovery
from leica_tracker.physical_runtime import FeedbackPublisher, SwitchingGate
from leica_tracker.resident_guest_watchdog import ResidentMotionGate


class Channel:
    def __init__(self, *, wait=False):
        self.entered=threading.Event();self.release=threading.Event();self.closed=False
        if not wait:self.release.set()
    def close_completed(self):
        self.entered.set()
        assert self.release.wait(2.),'test retirement was not released'
        self.closed=True
    def close(self):self.closed=True;self.release.set()


def armed(channel=None, cls=ResidentMotionGate):
    channel=channel or Channel();gate=cls(channel);gate.arm();return gate,channel


def detach(switch,gate,begin):
    method=getattr(switch,'detach_completed',None)
    assert callable(method),'Completed gate latch and detach must be one switch transaction'
    return method(gate,begin)


def launch(callback):
    errors=[]
    def run():
        try:callback()
        except BaseException as error:errors.append(error)
    thread=threading.Thread(target=run,daemon=True);thread.start();return thread,errors


def test_publisher_pause_during_slow_sdk_retirement_has_no_retired_gate_authority():
    gate,channel=armed(Channel(wait=True));paused=threading.Event()
    class Switch(SwitchingGate):
        def pause(self,reason):super().pause(reason);paused.set()
    switch=Switch();switch.attach(gate)
    publisher=None;watch=None
    try:
        detach(switch,gate,lambda:gate.retire_completed('normal complete'))
        assert channel.entered.wait(.5) and not switch.attached and not gate.can_send
        recovery=CommunicationRecovery(dict(instance='offline',epoch=1,started_ns=1),clock=lambda:10.)
        publisher=object.__new__(FeedbackPublisher)
        publisher._lock=threading.Lock();publisher._pending={1:9.};publisher.clock=lambda:10.
        publisher.max_latency=.25;publisher._closed=threading.Event()
        publisher.recovery=recovery;publisher.gate=switch;publisher.failed=None
        watch,errors=launch(publisher._watchdog)
        assert paused.wait(.5),'The real publisher watchdog did not request recovery HOLD'
        assert publisher.failed is None and switch.failure is None and recovery.status=='waiting'
        # 暂停属于过渡，并由新武装的本土门继承。这里不能有旧门回调作为动作权限。
        next_gate,next_channel=armed()
        switch.attach(next_gate)
        assert next_gate.is_paused and next_gate.can_send
        switch.stop('test complete');assert next_gate.close_finished.wait(.5)
        assert next_channel.closed and not errors
    finally:
        if publisher is not None:publisher._closed.set()
        if watch is not None:watch.join(.5)
        channel.release.set()
        if not gate.close_finished.is_set():gate.stop('test cleanup')
        assert gate.close_finished.wait(.5)


def test_inflight_pause_finishes_before_completed_gate_latch_and_detach():
    pause_entered=threading.Event();release_pause=threading.Event();begin_entered=threading.Event()
    class BlockingPauseGate(ResidentMotionGate):
        def pause(self,reason):
            pause_entered.set();assert release_pause.wait(1.)
            return super().pause(reason)
    gate,channel=armed(cls=BlockingPauseGate);switch=SwitchingGate();switch.attach(gate)
    assert callable(getattr(switch,'detach_completed',None)),'Atomic completed detach is missing'
    pause_thread,pause_errors=launch(lambda:switch.pause('brief gap'))
    retire_thread=None
    try:
        assert pause_entered.wait(.5)
        def begin():begin_entered.set();gate.retire_completed('normal complete')
        retire_thread,retire_errors=launch(lambda:detach(switch,gate,begin))
        assert not begin_entered.wait(.05),'Retirement raced an already selected pause delegate'
        release_pause.set();pause_thread.join(.5);retire_thread.join(.5)
        assert not pause_thread.is_alive() and not retire_thread.is_alive()
        assert not pause_errors and not retire_errors and not switch.attached
        assert gate.close_finished.wait(.5)
    finally:
        release_pause.set();pause_thread.join(.5)
        if retire_thread is not None:retire_thread.join(.5)
        gate.stop('test cleanup');channel.release.set()


def test_stop_before_completed_detach_wins_and_never_calls_normal_retirement():
    gate,channel=armed();switch=SwitchingGate();switch.attach(gate);calls=[]
    switch.stop('operator stop')
    try:
        with pytest.raises(RuntimeError):detach(switch,gate,lambda:calls.append(True))
        assert calls==[] and switch.failure=='operator stop' and not gate.can_send
    finally:assert gate.close_finished.wait(.5)


def test_stop_during_atomic_completed_detach_remains_latched_for_next_attach():
    gate,channel=armed();switch=SwitchingGate();switch.attach(gate)
    assert callable(getattr(switch,'detach_completed',None)),'Atomic completed detach is missing'
    entered=threading.Event();release=threading.Event()
    def begin():
        entered.set();assert release.wait(1.)
        gate.retire_completed('normal complete')
    retire_thread,retire_errors=launch(lambda:detach(switch,gate,begin));stop_thread=None
    try:
        assert entered.wait(.5)
        stop_thread,stop_errors=launch(lambda:switch.stop('operator stop during transition'))
        assert stop_thread.is_alive()
        release.set();retire_thread.join(.5);stop_thread.join(.5)
        assert not retire_errors and not stop_errors and not switch.attached
        assert switch.failure=='operator stop during transition'
        next_gate,_=armed()
        with pytest.raises(RuntimeError):switch.attach(next_gate)
        assert not next_gate.can_send and next_gate.close_finished.wait(.5)
    finally:
        release.set();retire_thread.join(.5)
        if stop_thread is not None:stop_thread.join(.5)
        gate.stop('test cleanup');assert gate.close_finished.wait(.5)


def test_stop_after_completed_detach_still_blocks_all_new_motion_authority():
    gate,channel=armed();switch=SwitchingGate();switch.attach(gate)
    try:
        detach(switch,gate,lambda:gate.retire_completed('normal complete'))
        switch.stop('operator stop in detached gap')
        next_gate,_=armed()
        with pytest.raises(RuntimeError):switch.attach(next_gate)
        assert not next_gate.can_send and next_gate.close_finished.wait(.5)
        assert switch.failure=='operator stop in detached gap'
    finally:gate.stop('test cleanup');assert gate.close_finished.wait(.5)


def test_failed_begin_retirement_keeps_old_ownership_and_can_only_fail_closed():
    gate,channel=armed();switch=SwitchingGate();switch.attach(gate)
    def fail():raise RuntimeError('watch quiescence failed')
    try:
        with pytest.raises(RuntimeError,match='watch quiescence failed'):detach(switch,gate,fail)
        assert switch.attached and gate.can_send
        switch.stop('retirement admission failed')
        assert switch.failure and not gate.can_send
    finally:gate.stop('test cleanup');assert gate.close_finished.wait(.5)


def test_completed_detach_rejects_wrong_gate_without_touching_either_owner():
    gate,channel=armed();other,other_channel=armed();switch=SwitchingGate();switch.attach(gate);calls=[]
    try:
        with pytest.raises(RuntimeError):detach(switch,other,lambda:calls.append(True))
        assert switch.attached and gate.can_send and other.can_send and calls==[]
    finally:
        switch.stop('test cleanup');other.stop('test cleanup')
        assert gate.close_finished.wait(.5) and other.close_finished.wait(.5)
