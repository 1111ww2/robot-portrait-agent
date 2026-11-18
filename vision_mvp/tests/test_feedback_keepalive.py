"""通过真实的本地 HTTP 请求进行回归测试，反馈使用替身，不访问机器人。"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from leica_tracker.physical_interaction_supervisor import LocalReferenceGate
from leica_tracker.physical_runtime import FeedbackPublisher


@pytest.fixture
def endpoint():
    state = SimpleNamespace(calls=[], connections=0, finished=[], behavior='ok', lock=threading.Lock())
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        disable_nagle_algorithm=True
        def setup(self):
            super().setup()
            with state.lock:
                state.connections+=1
                self.identity=state.connections
        def log_message(self,*args):pass
        def handle(self):
            try:super().handle()
            except (BrokenPipeError,ConnectionResetError):pass
        def finish(self):
            try:super().finish()
            finally:state.finished.append(self.identity)
        def do_POST(self):
            data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state.calls.append((self.identity, data, dict(self.headers)))
            if state.behavior=='timeout':time.sleep(.35)
            if state.behavior=='slow':time.sleep(.15)
            body=(b'{"ok":true,"padding":"'+b'x'*262144+b'"}' if state.behavior=='oversize' else b'{"ok":true}')
            status=400 if state.behavior=='reject' else 200
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            if state.behavior!='eof':
                self.send_header('Content-Length',str(len(body)+(5 if state.behavior=='truncate' else 0)))
            if state.behavior in ('close','truncate','eof'):
                self.send_header('Connection','close');self.close_connection=True
            self.end_headers()
            try:self.wfile.write(body);self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            state.calls.append((self.identity,None,dict(self.headers)))
            body=b'{"ok":true}'
            self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);thread.start()
    state.port=server.server_port
    try:yield state
    finally:server.shutdown();server.server_close();thread.join(1)


def reference(endpoint, *, timeout=.1):
    return LocalReferenceGate(endpoint.port,'test-token',timeout=timeout,
        control={'instance':'test-instance','epoch':1,'started_ns':1},lease='test-lease')


def wait_for(predicate):
    deadline=time.monotonic()+2
    while not predicate():
        if time.monotonic()>deadline:pytest.fail('Feedback worker did not reach expected state')
        time.sleep(.002)


def publisher(endpoint, **kwargs):
    reasons=[]
    gate=SimpleNamespace(stop=reasons.append)
    return FeedbackPublisher(reference(endpoint,**kwargs),gate),reasons


def test_sender_reuses_connection_and_preserves_each_authenticated_body(endpoint):
    pub,reasons=publisher(endpoint)
    try:
        pub.submit({'op':'heartbeat'})
        wait_for(lambda:len(endpoint.calls)==1 and not pub._pending)
        pub.submit({'op':'intro','intro_id':2,'status':'active'})
        wait_for(lambda:len(endpoint.calls)==2 and not pub._pending)
        assert endpoint.calls[0][0]==endpoint.calls[1][0]
        for _,body,headers in endpoint.calls:
            assert body['instance']=='test-instance' and body['epoch']==1 and body['lease']=='test-lease'
            assert headers['X-Guest-Token']=='test-token'
            assert headers['Origin']==f'http://127.0.0.1:{endpoint.port}'
        assert reasons==[]
    finally:pub.close()
    wait_for(lambda:endpoint.calls[0][0] in endpoint.finished)


def test_other_reference_calls_use_independent_connection(endpoint):
    pub,reasons=publisher(endpoint)
    try:
        pub.submit({'op':'heartbeat'});wait_for(lambda:len(endpoint.calls)==1 and not pub._pending)
        assert pub.reference._request('GET','/operator-state')=={'ok':True}
        pub.submit({'op':'heartbeat'});wait_for(lambda:len(endpoint.calls)==3 and not pub._pending)
        assert endpoint.calls[0][0]==endpoint.calls[2][0]!=endpoint.calls[1][0]
        assert reasons==[]
    finally:pub.close()


@pytest.mark.parametrize('behavior,category',[('truncate','response'),('oversize','response'),('timeout','timeout'),('reject','http_rejected')])
def test_bad_feedback_response_closes_sender_without_replaying_post(endpoint,behavior,category):
    endpoint.behavior=behavior
    pub,reasons=publisher(endpoint,timeout=.05)
    try:
        pub.submit({'op':'heartbeat'})
        wait_for(lambda:pub.failed is not None)
        assert category in pub.failed or behavior=='timeout' and 'deadline' in pub.failed
        assert len(endpoint.calls)==1
        assert reasons==[pub.failed]
    finally:pub.close()
    wait_for(lambda:endpoint.calls[0][0] in endpoint.finished)


@pytest.mark.parametrize('behavior',['close','eof'])
def test_server_connection_close_is_honored_for_next_distinct_request(endpoint,behavior):
    endpoint.behavior=behavior;pub,reasons=publisher(endpoint)
    try:
        pub.submit({'op':'heartbeat'});wait_for(lambda:len(endpoint.calls)==1 and not pub._pending)
        pub.submit({'op':'heartbeat'});wait_for(lambda:len(endpoint.calls)==2 and not pub._pending)
        assert endpoint.calls[0][0]!=endpoint.calls[1][0]
        assert reasons==[]
    finally:pub.close()


def test_sender_closes_transport_in_own_thread_even_when_idle():
    seen=[];received=threading.Event()
    class Reference:
        def feedback_request(self,body):seen.append(('request',threading.get_ident()));received.set();return {'ok':True}
        def close_feedback_transport(self):seen.append(('close',threading.get_ident()))
        def _request(self,*args):pytest.fail('Dedicated feedback transport was not selected')
    pub=FeedbackPublisher(Reference(),SimpleNamespace(stop=lambda reason:None))
    try:pub.submit({'op':'heartbeat'});assert received.wait(1)
    finally:pub.close()
    assert [item[0] for item in seen]==['request','close']
    assert seen[0][1]==seen[1][1]!=threading.get_ident()


def test_control_heartbeat_response_150ms_uses_existing_250ms_budget(endpoint):
    endpoint.behavior='slow'
    pub,reasons=publisher(endpoint,timeout=.05)
    try:
        pub.submit({'op':'heartbeat'})
        wait_for(lambda:len(endpoint.calls)==1 and not pub._pending)
        assert reasons==[] and pub.failed is None
    finally:pub.close()


def test_repeated_150ms_control_responses_do_not_queue_duplicate_intro_rpcs(endpoint):
    from leica_tracker.physical_runtime import LeaseHeartbeat
    endpoint.behavior='slow'
    pub,reasons=publisher(endpoint,timeout=.05)
    heart=LeaseHeartbeat(pub)
    try:
        heart.set_intro(7,'active')
        wait_for(lambda:len(endpoint.calls)>=5)
        assert pub.failed is None and reasons==[] and heart.failure is None
        assert all(body['op']=='heartbeat' for _,body,_ in endpoint.calls)
        assert any(body.get('intro_id')==7 and body.get('intro_status')=='active' for _,body,_ in endpoint.calls)
        assert pub.intro_acknowledged(7,'active')
    finally:heart.close();pub.close()
