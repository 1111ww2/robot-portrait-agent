"""使用模拟通道，以确定顺序重现策略检查和原生暂停之间的竞争。"""
import threading
from types import SimpleNamespace

import pytest

from leica_tracker import network_recovery
from leica_tracker.network_recovery import CommunicationRecovery
from leica_tracker import physical_runtime as runtime_module
from leica_tracker.physical_runtime import FeedbackPublisher, SwitchingGate, run_session
from leica_tracker.resident_guest_watchdog import ResidentMotionGate


CONTROL = dict(instance='guest-a', epoch=1, started_ns=1)


def snapshot(**changes):
    return {**CONTROL, 'active': True, 'control_fresh': True,
            'camera_connected': True, 'camera_fresh': True, **changes}


class Channel:
    def __init__(self): self.closed = threading.Event()
    def send(self, *args): pass
    def close(self): self.closed.set()


class OneTick:
    def __init__(self): self.calls = 0
    def wait(self, _timeout):
        self.calls += 1
        return self.calls > 1
    def set(self): self.calls = 2


class RacingRecovery(CommunicationRecovery):
    def __init__(self, control, *, clock):
        super().__init__(control, clock=clock)
        self.interpose = False
        self.finish_started = threading.Event(); self.finish_done = threading.Event()
        self.finish = None; self.finish_error = None; self.finish_thread = None

    @property
    def paused(self):
        result = super().paused
        if self.interpose:
            self.interpose = False
            def finishing():
                self.finish_started.set()
                try: self.finish()
                except Exception as exc: self.finish_error = exc
                finally: self.finish_done.set()
            self.finish_thread = threading.Thread(target=finishing)
            self.finish_thread.start()
            assert self.finish_started.wait(1)
            # 旧代码在此处已经释放策略锁，导致 complete 可能早于迟到的暂停执行。
            # 修正后继续持锁，complete 必须等待。
            self.finish_done.wait(.03)
        return result


def ready_policy(policy, clock):
    policy.observe(snapshot(control_fresh=False))
    policy.observe(snapshot())
    clock.now += .31
    assert policy.observe(snapshot()) == 'ready'


@pytest.mark.parametrize('finish_kind', ['complete', 'stop'])
def test_feedback_watchdog_stale_pause_cannot_undo_completion_or_revive_stop(finish_kind):
    clock = SimpleNamespace(now=10.)
    policy = RacingRecovery(CONTROL, clock=lambda: clock.now)
    channel = Channel(); gate = ResidentMotionGate(channel); gate.arm(); gate.pause('waiting')
    ready_policy(policy, clock)
    publisher = FeedbackPublisher.__new__(FeedbackPublisher)
    publisher._closed = OneTick(); publisher._lock = threading.Lock()
    publisher._pending = {1: 10.}; publisher.max_latency = .1
    publisher.clock = lambda: clock.now; publisher.recovery = policy
    publisher.gate = gate; publisher.failed = None
    if finish_kind == 'complete':
        policy.finish = lambda: policy.complete(resume=gate.resume)
    else:
        def stop():
            gate.stop('operator stop')
            policy.fail('operator stop')
        policy.finish = stop
    policy.interpose = True
    publisher._watchdog()
    assert policy.finish_done.wait(1)
    policy.finish_thread.join(1)
    if finish_kind == 'complete':
        assert policy.finish_error is None and policy.status == 'healthy'
        assert gate.can_send and not gate.is_paused
    else:
        assert channel.closed.wait(1) and not gate.can_send
        assert gate.reason == 'operator stop'
        with pytest.raises(RuntimeError): gate.resume()


def test_run_session_reader_cannot_apply_a_stale_pause_after_atomic_completion(monkeypatch):
    clock = SimpleNamespace(now=10.)
    policy = RacingRecovery(CONTROL, clock=lambda: clock.now)
    switch = SwitchingGate(); channel = Channel(); gate = ResidentMotionGate(channel); gate.arm()
    observations = []
    class Finished(Exception): pass
    reference = SimpleNamespace(status=lambda _: {'blocked': False}, fault=lambda: True)
    monkeypatch.setattr(runtime_module, 'SwitchingGate', lambda: switch)
    monkeypatch.setattr(network_recovery, 'CommunicationRecovery', lambda _: policy)
    monkeypatch.setattr(runtime_module, 'register_backend', lambda *args, **kwargs: reference)
    monkeypatch.setattr(runtime_module, 'FeedbackPublisher', lambda *args, **kwargs:
                        SimpleNamespace(failed=None, close=lambda: None))
    monkeypatch.setattr(runtime_module, 'LeaseHeartbeat', lambda *args, **kwargs:
                        SimpleNamespace(failure=None, close=lambda: None))
    monkeypatch.setattr(runtime_module, 'wait_initial_communication', lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_module, 'wait_for_trigger', lambda *args, **kwargs:
                        {'event': 'person_guidance'})
    def opener(host, cfg, reader, *, require_camera):
        switch.attach(gate); ready_policy(policy, clock)
        switch.pause('waiting')
        policy.finish = lambda: policy.complete(resume=switch.resume)
        policy.interpose = True
        reader()
        assert policy.finish_done.wait(1)
        policy.finish_thread.join(1)
        observations.append((policy.status, gate.is_paused, gate.can_send, policy.finish_error))
        raise Finished()
    profile = SimpleNamespace(host='127.0.0.2', cfg=SimpleNamespace(gesture=True), gate_timeout=.25)
    with pytest.raises(Finished):
        run_session(profile, 8777, 'fake-token', SimpleNamespace(failure=None), snapshot,
                    before={**CONTROL, 'control_action': 'restart'}, dependencies={'xyz_opener': opener})
    assert observations == [('healthy', False, True, None)]
    assert not gate.can_send  # 会话正常结束时，finally 仍须永久停用该对象。
