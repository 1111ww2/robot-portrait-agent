"""参考恢复练习模拟交换，从不打开界面。"""
import importlib
import re
from types import SimpleNamespace

import pytest

from leica_tracker.network_recovery import CommunicationRecovery
from leica_tracker.physical_interaction_supervisor import LocalReferenceGate, ReferenceRequestError


CONTROL = dict(instance='guest-a', epoch=1, started_ns=1)


class Clock:
    def __init__(self): self.now = 10.
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


@pytest.fixture
def module():
    def load():
        try: return importlib.import_module('leica_tracker.network_reference')
        except ModuleNotFoundError: pytest.fail('RecoveryReferenceGate is not implemented')
    return load


def gate(module, monkeypatch):
    module = module(); clock = Clock(); policy = CommunicationRecovery(CONTROL, clock=clock)
    monkeypatch.setattr(module, 'time', SimpleNamespace(sleep=clock.sleep))
    return module, clock, policy, module.RecoveryReferenceGate(
        8777, 'temporary-token', timeout=.1, control=CONTROL, lease='lease', recovery=policy)


@pytest.mark.parametrize('op', ['begin', 'release', 'register'])
def test_ambiguous_mutation_retries_same_request_id_and_body_only(module, monkeypatch, op):
    module, clock, policy, ref = gate(module, monkeypatch); calls = []
    def finish_hold(seconds):
        clock.sleep(seconds)
        policy.observe({**CONTROL, 'active': True, 'control_fresh': True,
                        'camera_connected': True, 'camera_fresh': True})
        if policy.ready: policy.complete()
    if op != 'register': monkeypatch.setattr(module, 'time', SimpleNamespace(sleep=finish_hold))
    def exchange(self, method, path, body=None):
        calls.append((method, path, dict(body)))
        if len(calls) in (1, 3):
            raise ReferenceRequestError('timeout', phase='body', elapsed_ms=100)
        return {'ok': True, 'session': 'original', 'release_ns': 123, 'lease': 'original'}
    monkeypatch.setattr(LocalReferenceGate, '_request', exchange)
    body = {'op': op, 'session': 'same'}
    ref._request('POST', '/reference-gate' if op != 'register' else '/physical-feedback', body)
    ref._request('POST', '/reference-gate' if op != 'register' else '/physical-feedback', body)
    assert len(calls) == 4
    assert calls[0] == calls[1] and calls[2] == calls[3]
    assert re.fullmatch('[0-9a-f]{32}', calls[0][2]['request_id'])
    assert calls[0][2]['request_id'] != calls[2][2]['request_id']
    assert 'request_id' not in body
    assert policy.paused is (op == 'register')


@pytest.mark.parametrize('status', [403, 409])
def test_auth_or_non_sample_rejection_is_terminal_without_retry(module, monkeypatch, status):
    module, clock, policy, ref = gate(module, monkeypatch); calls = []
    def rejected(*args, **kwargs):
        calls.append(1)
        raise ReferenceRequestError('http_rejected', phase='response', elapsed_ms=1, status=status)
    monkeypatch.setattr(LocalReferenceGate, '_request', rejected)
    with pytest.raises(ReferenceRequestError): ref.begin([0.] * 6)
    assert calls == [1] and policy.failure is not None


def test_fixed_policy_deadline_bounds_retries_and_late_success(module, monkeypatch):
    module, clock, policy, ref = gate(module, monkeypatch); calls = []
    def failed(*args, **kwargs):
        calls.append(clock.now); clock.now += 1.
        raise ReferenceRequestError('connection', phase='request', elapsed_ms=100)
    monkeypatch.setattr(LocalReferenceGate, '_request', failed)
    with pytest.raises(ReferenceRequestError): ref.status(None)
    assert policy.deadline == 14. and len(calls) == 4
    assert policy.failure and all(stamp < policy.deadline for stamp in calls)


def test_interrupted_sample_is_not_replayed_and_tells_supervisor_to_refresh(module, monkeypatch):
    module, clock, policy, ref = gate(module, monkeypatch); calls = []
    def failed(*args, **kwargs):
        calls.append(1)
        raise ReferenceRequestError('timeout', phase='body', elapsed_ms=100)
    monkeypatch.setattr(LocalReferenceGate, '_request', failed)
    with pytest.raises(module.ReferenceSampleInterrupted):
        ref.sample('same', [0.] * 6, [0.] * 6, 9.99)
    assert calls == [1] and policy.paused and policy.failure is None


@pytest.mark.parametrize('op', ['heartbeat', 'intro'])
def test_critical_feedback_retries_same_body_on_same_owned_transport(module, monkeypatch, op):
    module, clock, policy, ref = gate(module, monkeypatch); calls = []
    def feedback(self, body):
        calls.append(dict(body))
        if len(calls) == 1:
            raise ReferenceRequestError('connection', phase='request', elapsed_ms=100)
        return {'ok': True}
    monkeypatch.setattr(LocalReferenceGate, 'feedback_request', feedback)
    assert ref.feedback_request({'op': op, 'intro_id': 4}) == {'ok': True}
    assert calls == [{'op': op, 'intro_id': 4}] * 2


def test_presentation_is_legacy_best_effort_and_control_clone_shares_policy(module, monkeypatch):
    module, clock, policy, ref = gate(module, monkeypatch)
    assert type(ref.presentation_transport()) is LocalReferenceGate
    clone = ref.control_transport(timeout=.2)
    assert type(clone) is module.RecoveryReferenceGate and clone.recovery is policy
    assert clone.identity == ref.identity and clone.started_ns == ref.started_ns


def test_resume_communication_uses_explicit_idempotent_protocol(module, monkeypatch):
    module, clock, policy, ref = gate(module, monkeypatch); calls = []
    monkeypatch.setattr(LocalReferenceGate, '_request', lambda self, method, path, body:
                        calls.append((method, path, body)) or {'ok': True})
    assert ref.resume_communication() is True
    assert calls == [('POST', '/physical-feedback', {'op': 'resume'})]


def test_begin_waits_for_runtime_completion_instead_of_mutating_during_ready_hold(module, monkeypatch):
    module, clock, policy, ref = gate(module, monkeypatch); calls = []; waits = []
    policy.observe({**CONTROL, 'active': True, 'control_fresh': False,
                    'camera_connected': True, 'camera_fresh': True})
    def wait(seconds):
        waits.append(policy.status); clock.sleep(seconds)
        policy.observe({**CONTROL, 'active': True, 'control_fresh': True,
                        'camera_connected': True, 'camera_fresh': True})
        if policy.ready:
            assert calls == []
            waits.append('runtime complete'); policy.complete()
    monkeypatch.setattr(module, 'time', SimpleNamespace(sleep=wait))
    monkeypatch.setattr(LocalReferenceGate, '_request', lambda *args:
                        calls.append(policy.status) or {'session': 'same'})
    assert ref.begin([0.] * 6) == 'same'
    assert calls == ['healthy'] and 'runtime complete' in waits


@pytest.mark.parametrize('operation', ['photo_source', 'photo_ready'])
def test_photo_transport_gap_cancels_old_photo_result_without_retry(module, monkeypatch, operation):
    module, clock, policy, ref = gate(module, monkeypatch); calls = []
    def failed(*args, **kwargs):
        calls.append(1)
        raise ReferenceRequestError('timeout', phase='body', elapsed_ms=100)
    monkeypatch.setattr(LocalReferenceGate, '_request', failed)
    with pytest.raises(module.ReferencePhotoInterrupted):
        ref.photo_source() if operation == 'photo_source' else ref.photo_ready(123)
    assert calls == [1] and policy.paused and policy.failure is None
