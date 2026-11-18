"""网络保留会计使用现有的假居民/参考设置。"""
import importlib
import threading
from types import SimpleNamespace

import pytest

from test_physical_interaction_supervisor import setup, ready, event, step
from test_physical_resident_engine import feedback
from leica_tracker.network_recovery import CommunicationRecovery
from leica_tracker.physical_interaction_supervisor import LocalReferenceGate, ReferenceRequestError
from leica_tracker import physical_interaction_supervisor as supervisor_module


def pump_engine(sup, cfg, gate, times):
    for now in times:
        sup.engine.tick(feedback(cfg, 123, state=0, seq=gate.calls[-1][0]),
                        sampled_at=now, now=now)


def test_network_pause_shifts_action_deadline_once_and_resets_measured_dwell():
    cfg, gate, engine, ref, sup = setup(); ready(sup, cfg, gate)
    sup._deadline = 10.; sup.state = 'SETTLING'
    sup._settle_since = 1.; sup._last_settle_sample = 1.1
    sup._sample_acknowledged_at = 1.1; sup._reference_return_verified = True
    sup.pause_for_network(1.2); sup.pause_for_network(1.3)
    sup.resume_after_network(2.2, 2_200_000_000)
    assert sup._deadline == 11. and sup.state == 'SETTLING'
    assert sup._settle_since is None and sup._sample_acknowledged_at is None
    assert sup._reference_return_verified is False
    assert sup._release_ns >= 2_200_000_000 and sup._last_frame_ns >= 2_200_000_000
    assert not any(call[1] for call in gate.calls)


def test_job_timeout_subtracts_only_job_execution_overlapping_explicit_pause():
    cfg, gate, engine, ref, sup = setup()
    sup._job = SimpleNamespace(done=True, elapsed=2.05, error=None, value='same')
    sup._job_kind = 'begin'; sup._job_started = 1.1
    sup.pause_for_network(1.12); sup.resume_after_network(3.12, 3_120_000_000)
    assert sup._finish_job(3.2) == ('begin', 'same')
    assert not gate.stopped


def test_pause_after_slow_completed_rpc_never_hides_its_original_timeout():
    cfg, gate, engine, ref, sup = setup()
    sup._job = SimpleNamespace(done=True, elapsed=.12, error=None, value='same')
    sup._job_kind = 'begin'; sup._job_started = 1.
    sup.pause_for_network(1.2); sup.resume_after_network(2.2, 2_200_000_000)
    with pytest.raises(RuntimeError, match='Reference operation timed out'):
        sup._finish_job(2.3)
    assert gate.stopped


@pytest.mark.parametrize('state', ['PHOTO_LOOKUP', 'PHOTO_SETTLING', 'PHOTO_PENDING', 'PHOTO_READY'])
def test_photo_network_resume_requires_new_arrival_instead_of_old_result(state):
    cfg, gate, engine, ref, sup = setup(); ready(sup, cfg, gate)
    sup.state = state; sup._photo_source_ns = 1_000_000_000
    sup._job = SimpleNamespace(done=True, elapsed=.01, error=None, value=True)
    sup._job_kind = 'photo_ready'; sup._job_started = 1.
    sup.pause_for_network(1.1); sup.resume_after_network(1.5, 1_500_000_000)
    assert sup.state == 'OBSERVING' and sup._job is None and sup._photo_source_ns is None
    assert not sup.accept_event(event(1_400_000_000), now=1.5)
    assert ref.calls == [] and not any(call[1] for call in gate.calls)


def test_lost_sample_result_resets_proof_without_closing_resident():
    try: error_type = importlib.import_module('leica_tracker.network_reference').ReferenceSampleInterrupted
    except ModuleNotFoundError: pytest.fail('ReferenceSampleInterrupted is not implemented')
    cfg, gate, engine, ref, sup = setup()
    sup.state = 'SAMPLE_PENDING'; sup._reference_return_verified = True
    sup._job = SimpleNamespace(done=True, elapsed=.02, error=error_type(), value=None)
    sup._job_kind = 'sample'; sup._job_started = 1.
    assert sup._finish_job(1.03) == (None, None)
    assert sup.state == 'SETTLING' and sup._job is None
    assert not sup._reference_return_verified and not gate.stopped


def test_even_successful_pre_pause_sample_receipt_is_discarded_on_resume():
    cfg, gate, engine, ref, sup = setup()
    sup.state = 'SAMPLE_PENDING'
    sup._job = SimpleNamespace(done=True, elapsed=.01, error=None,
        value={'ok': True, 'sampled_at': 1., 'return_verified': True})
    sup._job_kind = 'sample'; sup._job_started = 1.
    sup._sample_payload_at = 1.; sup._reference_return_verified = True
    sup.pause_for_network(1.02); sup.resume_after_network(1.5, 1_500_000_000)
    assert sup._job is None and sup.state == 'SETTLING'
    assert sup._sample_payload_at is None and not sup._reference_return_verified
    assert not gate.stopped


def test_late_release_ack_cannot_lower_post_recovery_frame_floor():
    cfg, gate, engine, ref, sup = setup(); ready(sup, cfg, gate)
    sup.state = 'RELEASE_PENDING'; sup._session = 'same'
    sup._job = SimpleNamespace(done=True, elapsed=.01, error=None, value=1_100_000_000)
    sup._job_kind = 'release'; sup._job_started = 1.1
    sup.pause_for_network(1.12)
    pump_engine(sup, cfg, gate, (1.1, 1.2, 1.3, 1.4, 1.5))
    sup.resume_after_network(1.5, 1_500_000_000)
    # 跳过新的结算工作：这是已经完成的发布工作。
    sup._observe_reference_stillness = lambda *args: True
    step(sup, cfg, gate, 1.51)
    assert sup.state == 'OBSERVING' and sup._release_ns == 1_500_000_000


def test_already_accepted_direction_can_enable_after_hold_without_forging_frame_time():
    cfg, gate, engine, ref, sup = setup(); ready(sup, cfg, gate)
    assert sup.accept_event(event(1_060_000_000), now=1.06)
    sup.pause_for_network(1.07)
    pump_engine(sup, cfg, gate, (1.1, 1.2, 1.3, 1.4, 1.5))
    sup.resume_after_network(1.5, 1_500_000_000)
    ref.ack = True
    step(sup, cfg, gate, 1.51)
    step(sup, cfg, gate, 1.55)
    step(sup, cfg, gate, 1.6)
    assert sup.state == 'MOVING' and not gate.stopped
    assert sup._decision_ns == 1_060_000_000
    assert any(call[1] == 1 for call in gate.calls)


@pytest.mark.parametrize('budget', [.1, .25])
@pytest.mark.parametrize('pending_poll', [False, True])
def test_classified_full_timeout_then_recovery_and_fast_retry_preserves_rpc_budget(monkeypatch, budget, pending_poll):
    from leica_tracker import network_reference
    clock = SimpleNamespace(now=10.)
    control = dict(instance='guest-a', epoch=1, started_ns=1)
    policy = CommunicationRecovery(control, clock=lambda: clock.now)
    cfg, gate, engine, unused_ref, sup = setup()
    sup.gate_timeout = budget
    reference = network_reference.RecoveryReferenceGate(8777, 'test-token', timeout=budget,
        control=control, lease='same-lease', recovery=policy)
    calls = []
    retry_entered = threading.Event(); release_retry = threading.Event()
    def exchange(self, method, path, body):
        calls.append(dict(body))
        if len(calls) == 1:
            clock.now += budget  # 超时真的消耗了整个旧预算。
            raise ReferenceRequestError('timeout', phase='body', elapsed_ms=budget * 1000)
        clock.now += .02
        if pending_poll:
            retry_entered.set()
            assert release_retry.wait(1)
        return {'session': 'same-session'}
    def wait(seconds):
        if sup._network_pause_started is None:
            sup.pause_for_network(clock.now)
        clock.now += seconds
        policy.observe({**control, 'active': True, 'control_fresh': True,
                        'camera_connected': True, 'camera_fresh': True})
        if policy.ready:
            sup.resume_after_network(clock.now, round(clock.now * 1e9))
            policy.complete()
    monkeypatch.setattr(LocalReferenceGate, '_request', exchange)
    monkeypatch.setattr(supervisor_module, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(network_reference, 'time', SimpleNamespace(sleep=wait))
    sup.worker_factory = supervisor_module._Worker
    sup._submit('begin', reference.begin, clock.now, [0.] * 6)
    worker = sup._job
    if pending_poll:
        try:
            assert retry_entered.wait(1)
            assert sup._finish_job(clock.now) == (None, None)
            assert not gate.stopped
        finally:
            release_retry.set()
    assert worker._done.wait(1)
    assert worker.elapsed > budget + .3
    assert len(calls) == 2 and calls[0] == calls[1]
    assert sup._finish_job(clock.now) == ('begin', 'same-session')
    assert not gate.stopped


@pytest.mark.parametrize('budget', [.1, .25])
def test_unclassified_slow_success_still_exceeds_original_rpc_budget(monkeypatch, budget):
    from leica_tracker import network_reference
    clock = SimpleNamespace(now=10.)
    control = dict(instance='guest-a', epoch=1, started_ns=1)
    cfg, gate, engine, unused_ref, sup = setup(); sup.gate_timeout = budget
    policy = CommunicationRecovery(control, clock=lambda: clock.now)
    reference = network_reference.RecoveryReferenceGate(8777, 'test-token', timeout=budget,
        control=control, lease='same-lease', recovery=policy)
    def slow(*args, **kwargs):
        clock.now += budget + .02
        return {'session': 'same-session'}
    monkeypatch.setattr(LocalReferenceGate, '_request', slow)
    monkeypatch.setattr(supervisor_module, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    sup.worker_factory = supervisor_module._Worker
    sup._submit('begin', reference.begin, clock.now, [0.] * 6)
    assert sup._job._done.wait(1)
    with pytest.raises(RuntimeError, match='Reference operation timed out'):
        sup._finish_job(clock.now)
    assert gate.stopped and policy.deadline is None


def test_overlapping_failed_attempt_and_pause_are_not_counted_twice():
    from leica_tracker.network_reference import ReferenceRecoveryAccounting
    cfg, gate, engine, ref, sup = setup()
    accounting = ReferenceRecoveryAccounting(); accounting.record_failure(10., 10.1)
    sup._job = SimpleNamespace(done=True, elapsed=.52, error=None,
        value='same-session', recovery_accounting=accounting)
    sup._job_kind = 'begin'; sup._job_started = 10.
    sup.pause_for_network(10.05); sup.resume_after_network(10.4, 10_400_000_000)
    # 联合信用是0.4s，因此剩余的0.12s必须不及原始的0.1s。
    # 将重叠部分加起来两次会错误地通过，剩余0.07s。
    with pytest.raises(RuntimeError, match='Reference operation timed out'):
        sup._finish_job(10.52)
    assert gate.stopped


def test_classified_timeout_keeps_already_admitted_direction_eligible_for_first_enable(monkeypatch):
    from leica_tracker import network_reference
    cfg, gate, engine, unused_ref, sup = setup(); ready(sup, cfg, gate)
    clock = SimpleNamespace(now=1.06)
    control = dict(instance='guest-a', epoch=1, started_ns=1)
    policy = CommunicationRecovery(control, clock=lambda: clock.now)
    reference = network_reference.RecoveryReferenceGate(8777, 'test-token', timeout=.25,
        control=control, lease='same-lease', recovery=policy)
    sup.reference = reference; sup.gate_timeout = .25
    sup.worker_factory = supervisor_module._Worker
    begin_calls = []
    def advance(seconds):
        clock.now += seconds
        pump_engine(sup, cfg, gate, (clock.now,))
    def exchange(self, method, path, body=None):
        if method == 'GET':
            return {'active': True, 'control': self.identity, 'control_started_ns': 1,
                    'reference_gate': {'blocked': True, 'acknowledged': True, 'session': 'same'}}
        begin_calls.append(dict(body))
        if len(begin_calls) == 1:
            # 在整个HTTP中断期间，原生ACK泵持续运行。
            for _ in range(5): advance(.05)
            raise ReferenceRequestError('timeout', phase='body', elapsed_ms=250)
        advance(.02)
        return {'session': 'same'}
    def wait(seconds):
        if sup._network_pause_started is None:
            sup.pause_for_network(clock.now)
        advance(seconds)
        policy.observe({**control, 'active': True, 'control_fresh': True,
                        'camera_connected': True, 'camera_fresh': True})
        if policy.ready:
            sup.resume_after_network(clock.now, round(clock.now * 1e9))
            policy.complete()
    monkeypatch.setattr(LocalReferenceGate, '_request', exchange)
    monkeypatch.setattr(supervisor_module, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(network_reference, 'time', SimpleNamespace(sleep=wait))
    source_ns = 1_010_000_000  # 在中断之前，50岁时就已经被接受了。
    assert sup.accept_event(event(source_ns), now=clock.now)
    assert sup._job._done.wait(1)
    clock.now += .01; step(sup, cfg, gate, clock.now)
    assert sup._job._done.wait(1)  # 匹配开始/状态ACK，仍然是相同的操作。
    clock.now += .01; step(sup, cfg, gate, clock.now)
    clock.now += .01; step(sup, cfg, gate, clock.now)
    assert sup.state == 'MOVING' and not gate.stopped
    assert sup._decision_ns == source_ns and len(begin_calls) == 2


def test_admitted_decision_failure_credit_unions_pause_and_is_applied_once_after_completion():
    from leica_tracker.network_reference import ReferenceRecoveryAccounting
    cfg, gate, engine, ref, sup = setup(); sup.gate_timeout = .25
    source_ns = 10_000_000_000
    sup._decision_ns = source_ns; sup.state = 'WAIT_ACK'
    accounting = ReferenceRecoveryAccounting(); accounting.record_failure(10.15, 10.4)
    worker = SimpleNamespace(done=False, elapsed=None, error=None, value={'acknowledged': True},
        recovery_accounting=accounting)
    sup._job = worker; sup._job_kind = 'status'; sup._job_started = 10.15
    sup.pause_for_network(10.3); sup.resume_after_network(10.7, 10_700_000_000)
    assert sup._network_decision_pause_s == pytest.approx(.4)
    assert sup._finish_job(10.71) == (None, None)
    assert sup._network_decision_pause_s == pytest.approx(.4)  # 待处理的不会被记入。
    worker.done = True; worker.elapsed = .57
    assert sup._finish_job(10.72)[0] == 'status'
    # 失败的间隔.25与暂停重叠了.10；总合并为.55，而不是.65。
    assert sup._network_decision_pause_s == pytest.approx(.55)
    assert sup._finish_job(10.73) == (None, None)
    assert sup._network_decision_pause_s == pytest.approx(.55)
    assert sup._decision_ns == source_ns and not gate.stopped
