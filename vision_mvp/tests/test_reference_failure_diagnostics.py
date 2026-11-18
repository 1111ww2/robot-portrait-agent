"""本地 HTTP 出现真实故障时，保留固定原因分类，不记录私有文本。"""
import json
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from leica_tracker.guest_display import GuestDisplay
from leica_tracker.physical_interaction_supervisor import LocalReferenceGate,ReferenceRequestError
from test_physical_interaction_supervisor import setup


@pytest.mark.parametrize('case,expected,op',[
    ('blocked','gate_blocked','begin'),('session','wrong_session','sample'),
    ('stale','stale_sample','sample'),('return','return_unverified','release'),
    ('invalid','invalid_reference','begin'),
])
def test_actual_reference_http400_preserves_only_known_gate_cause(case,expected,op):
    with GuestDisplay(port=0) as display:
        ref=LocalReferenceGate(urlsplit(display.origin).port,display.token,timeout=1.,
            control=display.control_snapshot(),lease='private-lease-not-for-logs')
        session=ref.begin([0.]*6)
        with pytest.raises(ReferenceRequestError) as caught:
            if case=='blocked':ref.begin([0.]*6)
            elif case=='session':ref.sample('private-wrong-session',[0.]*6,[0.]*6,time.monotonic())
            elif case=='stale':ref.sample(session,[0.]*6,[0.]*6,time.monotonic()-1.)
            elif case=='return':ref.release(session)
            else:
                # 先打开新的门控，使测试实际到达六轴验证，而非提前被已关闭的门控拒绝。
                with display._lock:display._reference_gate.session=None
                ref.begin([0.]*5)
        error=caught.value
        assert error.status==400 and error.category=='http_rejected'
        assert error.reference_reason==expected and error.reference_op==op
        details=error.safe_details()
        assert details['reference_reason']==expected and details['reference_op']==op
        assert all(secret not in str(error) for secret in (display.token,session,'private-lease','private-wrong-session'))


def test_actual_http_unknown_gate_exception_text_is_redacted():
    with GuestDisplay(port=0) as display:
        ref=LocalReferenceGate(urlsplit(display.origin).port,display.token,timeout=1.,
            control=display.control_snapshot(),lease='private-test-lease')
        session=ref.begin([0.]*6)
        def reject(*args):raise ValueError('sentinel-private-token-and-camera-body')
        with display._lock:display._reference_gate.sample=reject
        with pytest.raises(ReferenceRequestError) as caught:ref.sample(session,[0.]*6,[0.]*6,time.monotonic())
        assert caught.value.status==400 and caught.value.reference_reason is None
        assert caught.value.reference_op=='sample'
        assert 'sentinel-private' not in str(caught.value)


@pytest.mark.parametrize('category,phase,status',[
    ('http_rejected','response',400),('timeout','headers',None),
])
def test_supervisor_retains_failed_operation_and_safe_request_error_then_stops(category,phase,status):
    _cfg,gate,_engine,ref,sup=setup()
    error=ReferenceRequestError(category,phase=phase,elapsed_ms=20.,status=status,
        reference_reason='stale_sample',reference_op='sample')
    sup._job=SimpleNamespace(done=True,elapsed=.02,error=error,value=None)
    sup._job_kind='sample';sup._job_started=1.
    with pytest.raises(RuntimeError,match='Reference operation failed') as caught:sup._finish_job(1.03)
    details=json.loads(str(caught.value).split('Reference operation failed ',1)[1])
    assert details['operation']=='sample' and details['category']==category and details['phase']==phase
    assert details['elapsed_ms']==20.
    if status is not None:assert details['status']==status and details['reference_reason']=='stale_sample'
    assert gate.stopped and sup.state=='FAULT' and ('fault',) in ref.calls


def test_unknown_worker_exception_and_unsafe_kind_never_log_private_message():
    _cfg,gate,_engine,_ref,sup=setup()
    sup._job=SimpleNamespace(done=True,elapsed=.02,error=RuntimeError('sentinel-private-token'),value=None)
    sup._job_kind='sentinel-private-kind';sup._job_started=1.
    with pytest.raises(RuntimeError) as caught:sup._finish_job(1.03)
    details=json.loads(str(caught.value).split('Reference operation failed ',1)[1])
    assert details=={'operation':'unknown','error_type':'RuntimeError'}
    assert 'sentinel-private' not in str(caught.value) and gate.stopped


def test_request_error_rejects_unknown_reason_and_operation_enum_text():
    error=ReferenceRequestError('http_rejected',phase='response',elapsed_ms=1.,status=400,
        reference_reason='sentinel-private-reason',reference_op='sentinel-private-op')
    assert error.reference_reason is None and error.reference_op is None
    assert 'sentinel-private' not in str(error)


def test_completed_overdue_error_keeps_deadline_fault_and_safe_http_cause():
    _cfg,gate,_engine,_ref,sup=setup()
    error=ReferenceRequestError('http_rejected',phase='response',elapsed_ms=200.,status=400,
        reference_reason='stale_sample',reference_op='sample')
    sup._job=SimpleNamespace(done=True,elapsed=.2,error=error,value=None)
    sup._job_kind='sample';sup._job_started=1.
    with pytest.raises(RuntimeError,match='Reference operation timed out') as caught:sup._finish_job(1.2)
    details=json.loads(str(caught.value).split(' worker_error ',1)[1])
    assert details['operation']=='sample' and details['status']==400 and details['reference_reason']=='stale_sample'
    assert gate.stopped and sup.state=='FAULT'
