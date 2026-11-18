"""修复了本地控制故障的非机密诊断程序；禁止机器人访问。"""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from leica_tracker import physical_interaction_supervisor as transport
from leica_tracker.physical_runtime import FeedbackPublisher
from test_intro_control_freshness import display,register,update


def reference():
    return transport.LocalReferenceGate(8767,'sentinel-secret',timeout=.1,
        control={'instance':'local','epoch':1,'started_ns':1},lease='sentinel-lease')


@pytest.mark.parametrize('failure,category',[(TimeoutError('sentinel-secret'),'timeout'),
                                           (ConnectionResetError('sentinel-secret'),'connection')])
def test_local_request_classifies_network_errors_without_leaking_text(monkeypatch,failure,category):
    class Connection:
        def __init__(self,*a,**k):pass
        def request(self,*a,**k):raise failure
        def close(self):pass
    monkeypatch.setattr(transport.http.client,'HTTPConnection',Connection)
    with pytest.raises(RuntimeError) as caught:reference()._request('POST','/physical-feedback',{'op':'heartbeat'})
    assert caught.value.category==category
    assert caught.value.phase=='request'
    assert 'sentinel' not in str(caught.value)


def test_http_rejection_preserves_only_allowed_reason_and_numeric_ages(monkeypatch):
    body={'reason':'control_heartbeat_expired','error':'sentinel-secret',
          'physical_revocation':{'reason':'control_heartbeat_expired','mode':'intro',
             'control_age_s':.312,'backend_age_s':.03,'camera_age_s':.04,'lease':'sentinel-lease'}}
    class Connection:
        def __init__(self,*a,**k):pass
        def request(self,*a,**k):pass
        def getresponse(self):return SimpleNamespace(status=400,read=lambda _:json.dumps(body).encode())
        def close(self):pass
    monkeypatch.setattr(transport.http.client,'HTTPConnection',Connection)
    with pytest.raises(RuntimeError) as caught:reference()._request('POST','/physical-feedback',{'op':'heartbeat'})
    error=caught.value
    assert error.category=='http_rejected' and error.status==400
    assert error.reason=='control_heartbeat_expired'
    assert error.diagnostics['control_age_s']==.312
    assert 'sentinel' not in str(error) and 'lease' not in error.diagnostics
    body['reason']='sentinel-secret';body['physical_revocation']['mode']='sentinel-lease'
    with pytest.raises(RuntimeError) as caught:reference()._request('GET','/operator-state')
    assert 'sentinel' not in str(caught.value)


def test_first_revocation_freezes_reason_and_ages_despite_later_healthy_inputs(display):
    d,capture=display;identity=register(d)
    capture._last_received=time.monotonic()-.02
    with d._lock:d._seen-=.31
    with pytest.raises(ValueError):update(d,identity,'heartbeat')
    first=dict(d._physical_revocation)
    assert first['reason']=='control_heartbeat_expired' and first['mode']=='intro'
    assert first['control_age_s']>=.3 and first['backend_age_s']<.3
    assert first['camera_age_s']>=.02
    with d._lock:d._seen=time.monotonic()
    update(d,identity,'fault')
    assert d._physical_revocation==first
    assert not any('token' in k or 'lease' in k for k in first)


def test_backend_timeout_is_distinct_from_control_timeout(display):
    d,_=display;identity=register(d)
    with d._lock:d._physical_backend_at-=.31
    with pytest.raises(ValueError):update(d,identity,'heartbeat')
    assert d._physical_revocation['reason']=='backend_heartbeat_expired'


def test_publisher_preserves_fixed_transport_classification():
    stopped=threading.Event();reasons=[]
    class Reference:
        def _request(self,*args):raise transport.ReferenceRequestError('timeout',phase='headers',elapsed_ms=100.)
    gate=SimpleNamespace(stop=lambda reason:(reasons.append(reason),stopped.set()))
    publisher=FeedbackPublisher(Reference(),gate)
    try:
        publisher.submit({'op':'heartbeat'})
        assert stopped.wait(1)
        assert 'timeout' in publisher.failed and 'headers' in publisher.failed
        assert reasons==[publisher.failed]
    finally:publisher.close()


def test_local_http_rejection_and_operator_snapshot_expose_same_frozen_cause(display):
    d,_=display;identity=register(d)
    ref=transport.LocalReferenceGate(d.port,d.token,timeout=.2,
        control={**identity,'started_ns':d._control_started_ns},lease=identity['lease'])
    with d._lock:d._seen-=.31
    with pytest.raises(transport.ReferenceRequestError) as caught:
        ref._request('POST','/physical-feedback',{'op':'heartbeat'})
    error=caught.value
    assert error.category=='http_rejected' and error.reason=='control_heartbeat_expired'
    state=ref._request('GET','/operator-state')
    assert state['physical']['revocation']==d._physical_revocation
    assert state['physical']['revocation']['control_age_s']==error.diagnostics['control_age_s']
    assert state['physical']['mode']=='intro'
    assert d.token not in str(error) and identity['lease'] not in str(error)
