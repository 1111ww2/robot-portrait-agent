"""每个 TLS 处理程序独占一个上游连接，失败的 POST 不得重放。"""
from email.message import Message
import io
from types import SimpleNamespace

import pytest

from leica_tracker import guest_https


def fixture(monkeypatch):
    clock=[10.];connections=[];statuses=[];plans=[]
    class Connection:
        def __init__(self,host,port,*,timeout):
            self.host=host;self.port=port;self.timeout=timeout;self.timeouts=[];self.sent=[];self.closed=0
            self.sock=SimpleNamespace(settimeout=self.timeouts.append)
            connections.append(self)
        def request(self,method,path,body,headers):
            self.sent.append((method,path,body,dict(headers)))
            self.plan=plans.pop(0) if plans else {}
            if self.plan.get('send_error'):raise OSError('write failed after being attempted')
        def getresponse(self):
            if self.plan.get('response_error'):raise guest_https.http.client.RemoteDisconnected('response lost after accepted write')
            if 'response' in self.plan:return self.plan['response']
            body=self.plan.get('body',b'{}')
            return SimpleNamespace(status=self.plan.get('status',200),
                will_close=self.plan.get('will_close',False),
                length=0,isclosed=lambda:True,read=lambda limit:body[:limit],getheader=lambda *a:None)
        def close(self):self.closed+=1;self.sock=None
    monkeypatch.setattr(guest_https.http.client,'HTTPConnection',Connection)
    proxy=object.__new__(guest_https.GuestHTTPS)
    proxy.files={};proxy.upstream_port=8777;proxy.port=8768;proxy.setup_port=8769;proxy.addresses={'localhost'}
    proxy.timing_callback=None;proxy.timing_clock=lambda:clock[0];proxy.upstream_clock=lambda:clock[0]
    handler=object.__new__(proxy._handler(False));handler.client_address=('127.0.0.1',1);handler.close_connection=False
    handler.connection=SimpleNamespace(settimeout=lambda _:None)
    handler.send_response=statuses.append;handler.send_header=lambda *a:None;handler.end_headers=lambda:None
    handler.wfile=io.BytesIO()
    handler.rfile=io.BytesIO()
    def call(path='/camera/ticket',body=b'{}',*,origin='https://localhost:8768',token='pair-A',length=None):
        handler.command='POST';handler.path=path;handler.rfile=io.BytesIO(body)
        headers=Message()
        kind='image/jpeg' if path=='/camera/frame' else 'application/octet-stream' if path=='/transcribe' else 'application/json'
        for name,value in {'Host':'localhost:8768','Origin':origin,'Content-Type':kind,
            'Content-Length':str(len(body) if length is None else length),'X-Guest-Token':token,
            'X-Guest-Client':'private-client','X-Camera-Session':'private-session','X-Camera-Ticket':'private-ticket'}.items():headers[name]=value
        handler.headers=headers;handler.handle_request()
    return SimpleNamespace(clock=clock,connections=connections,statuses=statuses,plans=plans,handler=handler,call=call,proxy=proxy)


def test_sequential_requests_on_one_tls_handler_connect_once_and_refresh_headers(monkeypatch):
    f=fixture(monkeypatch)
    f.call(token='pair-A');f.clock[0]+=.08;f.call('/camera/frame',b'fresh JPEG',token='pair-B')
    assert f.statuses==[200,200] and len(f.connections)==1
    conn=f.connections[0]
    assert conn.closed==0 and [row[1] for row in conn.sent]==['/camera/ticket','/camera/frame']
    assert [row[3]['X-Guest-Token'] for row in conn.sent]==['pair-A','pair-B']
    assert conn.sent[1][2]==b'fresh JPEG'
    f.handler.finish();assert conn.closed==1


def test_cache_does_not_cross_tls_handlers(monkeypatch):
    f=fixture(monkeypatch);f.call()
    other=object.__new__(type(f.handler))
    assert getattr(other,'_upstream',None) is None
    f.handler.finish()


def test_idle_one_second_expires_before_any_new_write(monkeypatch):
    f=fixture(monkeypatch);f.call();first=f.connections[0]
    assert first.closed==0
    f.clock[0]+=1.;f.call()
    assert first.closed==1 and len(f.connections)==2
    assert len(first.sent)==1 and len(f.connections[1].sent)==1
    f.handler.finish()


def test_reused_connection_and_socket_timeout_follow_each_route(monkeypatch):
    f=fixture(monkeypatch);f.call();f.call('/transcribe',b'audio');f.call('/camera/frame',b'JPEG')
    assert len(f.connections)==1
    assert f.connections[0].timeouts==[3,45,3]
    assert f.connections[0].timeout==3
    f.handler.finish()


@pytest.mark.parametrize('method',['GET','POST'])
def test_accepted_write_then_lost_response_returns_once_without_replaying(monkeypatch,method):
    f=fixture(monkeypatch);f.call();f.plans.append({'response_error':True})
    if method=='GET':
        f.handler.command='GET';f.handler.path='/state'
        headers=Message();headers['Host']='localhost:8768';headers['X-Guest-Token']='pair-A'
        f.handler.headers=headers;f.handler.handle_request()
    else:f.call('/action',b'{"action":"start"}')
    assert f.statuses==[200,503] and len(f.connections)==1
    assert len(f.connections[0].sent)==2 and f.connections[0].sent[1][0]==method
    assert f.connections[0].closed==1
    assert getattr(f.handler,'_upstream',None) is None
    f.handler.finish()


@pytest.mark.parametrize('plan,status',[({'will_close':True},200),({'body':b'x'*(2*1024*1024+1)},502),({'send_error':True},503)])
def test_nonreusable_response_or_error_discards_connection(monkeypatch,plan,status):
    f=fixture(monkeypatch);f.plans.append(plan);f.call()
    assert f.statuses==[status] and f.connections[0].closed==1
    assert getattr(f.handler,'_upstream',None) is None
    f.handler.finish()


def test_bad_origin_and_camera_body_limit_reject_before_upstream(monkeypatch):
    f=fixture(monkeypatch);f.call(origin='https://wrong.invalid')
    assert f.statuses==[403] and not f.connections
    f=fixture(monkeypatch);f.call('/camera/frame',b'{}',length=524289)
    assert f.statuses==[413] and not f.connections
    f.handler.finish()


def test_client_disconnect_or_handler_finish_closes_only_its_owned_upstream(monkeypatch):
    f=fixture(monkeypatch);f.call();conn=f.connections[0]
    assert conn.closed==0
    f.handler.finish();f.handler.finish()
    assert conn.closed==1 and f.handler.wfile.closed and f.handler.rfile.closed


def raw_response(data):
    sock=SimpleNamespace(makefile=lambda *a:io.BytesIO(data))
    response=guest_https.http.client.HTTPResponse(sock);response.begin()
    return response


def test_truncated_content_length_response_is_not_forwarded_or_cached_or_replayed(monkeypatch):
    f=fixture(monkeypatch)
    response=raw_response(b'HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n{}')
    f.plans.append({'response':response})
    f.call('/action',b'{"action":"start"}')
    assert f.statuses==[503] and len(f.connections)==1
    assert len(f.connections[0].sent)==1 and f.connections[0].closed==1
    assert getattr(f.handler,'_upstream',None) is None
    f.handler.finish()


def test_close_delimited_response_requires_eof_and_is_forwarded_without_reuse(monkeypatch):
    f=fixture(monkeypatch)
    response=raw_response(b'HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{}')
    f.plans.append({'response':response});f.call()
    assert f.statuses==[200] and response.isclosed()
    assert f.connections[0].closed==1 and getattr(f.handler,'_upstream',None) is None
    f.handler.finish()
