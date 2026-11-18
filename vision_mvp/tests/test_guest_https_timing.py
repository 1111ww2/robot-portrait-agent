"""按需生成的耗时记录只含数值和固定路由，不记录请求数据。"""
from email.message import Message
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from leica_tracker import guest_https
from test_physical_system import workspace


def capture(root, clock, duration=1.):
    return guest_https.HTTPSRequestTimings(root/'timing.json',duration=duration,clock=lambda:clock[0])


def test_capture_is_bounded_private_numeric_and_written_once_after_duration(workspace):
    clock=[10.];value=capture(workspace,clock)
    for i in range(4100):
        value.record({'route':'/camera/frame','status':200,'request_body_bytes':123,
                      'body_read_ms':12,'headers':'private-token','jpeg':b'private-image'})
    assert not value.finish_if_due() and not value.output.exists()
    clock[0]=11.
    assert value.finish_if_due()
    before=value.output.read_bytes();result=json.loads(before)
    assert len(result['events'])==4096
    assert all(set(row)=={'route','status','request_body_bytes','body_read_ms','elapsed_ms'} for row in result['events'])
    assert b'private' not in before and b'headers' not in before and b'jpeg' not in before
    assert value.finish_if_due() and value.output.read_bytes()==before


def test_invalid_capture_route_and_numbers_are_ignored_and_early_finish_is_allowed(workspace):
    clock=[10.];value=capture(workspace,clock)
    value.record({'route':'/camera/frame?key=private','status':200})
    value.record({'route':'/message','status':200})
    value.record({'route':'/camera/ticket','body_read_ms':float('nan'),'status':200})
    assert value.finish()
    records=json.loads(value.output.read_text())['events']
    assert records==[{'route':'/camera/ticket','status':200,'elapsed_ms':0.}]


def test_write_failure_is_numeric_and_never_escapes_into_server_lifecycle(workspace,monkeypatch):
    clock=[10.];value=capture(workspace,clock)
    def reject(*args,**kwargs):raise PermissionError('private filesystem detail')
    monkeypatch.setattr(guest_https.os,'open',reject)
    assert value.finish() is True and value.write_error==1
    assert value.finish() is True


def run_request(monkeypatch, *, body_delay=.12, request_delay=.03, response_delay=.07, fail=None,
                route='/camera/frame?key=private-token', timing=True):
    clock=[10.];records=[]
    proxy=object.__new__(guest_https.GuestHTTPS)
    proxy.files={};proxy.upstream_port=8777;proxy.port=8768;proxy.setup_port=8769;proxy.addresses={'localhost'}
    proxy.timing_callback=records.append if timing else None
    proxy.timing_clock=lambda:clock[0]
    handler=object.__new__(proxy._handler(False));handler.command='POST';handler.path=route
    handler.client_address=('127.0.0.1',1);handler.close_connection=False
    headers=Message()
    for key,value in {'Host':'localhost:8768','Origin':'https://localhost:8768',
        'Content-Type':'image/jpeg' if route.startswith('/camera/frame') else 'application/json',
        'Content-Length':'18','X-Guest-Token':'private-token','X-Guest-Client':'private-client'}.items():headers[key]=value
    handler.headers=headers
    def read(size):
        clock[0]+=body_delay
        if fail=='body':raise TimeoutError('private error text')
        return b'private-image-body'
    handler.rfile=SimpleNamespace(read=read)
    handler.connection=SimpleNamespace(settimeout=lambda _:None)
    handler.send_response=lambda status:None;handler.send_header=lambda *a:None;handler.end_headers=lambda:None
    handler.wfile=SimpleNamespace(write=lambda _:None)
    class Connection:
        def __init__(self,*args,**kwargs):pass
        def request(self,*args,**kwargs):
            clock[0]+=request_delay
            if fail=='upstream':raise TimeoutError('private upstream address')
        def getresponse(self):
            clock[0]+=response_delay
            return SimpleNamespace(status=200,length=0,isclosed=lambda:True,
                                   read=lambda _:b'private response',getheader=lambda *a:None)
        def close(self):pass
    monkeypatch.setattr(guest_https.http.client,'HTTPConnection',Connection)
    handler.handle_request()
    return records


@pytest.mark.parametrize('body,upstream,response',[(.4,.01,.02),(.01,.1,.4)])
def test_handler_splits_client_body_wait_from_upstream_time_without_secrets(monkeypatch,body,upstream,response):
    records=run_request(monkeypatch,body_delay=body,request_delay=upstream,response_delay=response)
    assert len(records)==1
    row=records[0]
    assert row['route']=='/camera/frame' and row['status']==200
    assert row['body_read_ms']==pytest.approx(body*1000)
    assert row['upstream_request_ms']==pytest.approx(upstream*1000)
    assert row['upstream_response_ms']==pytest.approx(response*1000)
    assert row['server_total_ms']==pytest.approx((body+upstream+response)*1000)
    assert row['request_body_bytes']==18 and row['error']==0
    encoded=json.dumps(row)
    assert 'private' not in encoded and '?' not in encoded and 'localhost' not in encoded


@pytest.mark.parametrize('failure,status,error',[('body',400,1),('upstream',503,3)])
def test_timing_errors_are_numeric_and_keep_the_same_http_result(monkeypatch,failure,status,error):
    row=run_request(monkeypatch,fail=failure)[0]
    assert row['status']==status and row['error']==error
    assert 'private' not in json.dumps(row)


def test_default_off_and_unobserved_routes_do_not_emit_events(monkeypatch):
    assert run_request(monkeypatch,timing=False)==[]
    assert run_request(monkeypatch,route='/action')==[]


def script():
    path=Path(__file__).resolve().parents[1]/'scripts/run-ipad-https.py'
    spec=importlib.util.spec_from_file_location('https_timing_cli',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('duration',['0','121','nan'])
def test_cli_rejects_invalid_timing_duration_before_pairing_io(workspace,monkeypatch,duration):
    entry=script();monkeypatch.setattr(entry,'ROOT',workspace)
    monkeypatch.setattr(entry,'load_pairing',lambda *a:pytest.fail('invalid timing reached pairing'))
    with pytest.raises(SystemExit):entry.main(['--timing-output','timing.json','--timing-duration',duration])


def test_cli_rejects_outside_and_existing_timing_output_before_pairing_io(workspace,monkeypatch):
    entry=script();monkeypatch.setattr(entry,'ROOT',workspace)
    monkeypatch.setattr(entry,'load_pairing',lambda *a:pytest.fail('invalid timing reached pairing'))
    existing=workspace/'keep.json';existing.write_text('preserve')
    for path in (existing,workspace.parent/'outside.json'):
        with pytest.raises(SystemExit):entry.main(['--timing-output',str(path)])
    assert existing.read_text()=='preserve'


def test_cli_keeps_tls_alive_after_capture_completes(workspace,monkeypatch,capsys):
    import qrcode
    entry=script();output=workspace/'tmp/guidance-live';output.mkdir(parents=True,exist_ok=True)
    der=workspace/'root.cer';der.write_bytes(b'public fixture certificate')
    stop=workspace/'stop.request';timing=workspace/'timing.json'
    monkeypatch.setattr(entry,'ROOT',workspace);monkeypatch.setattr(entry,'OUTPUT',output)
    monkeypatch.setattr(entry,'load_pairing',lambda *a:{'port':8777,'key':'private-fixture-token'})
    monkeypatch.setattr(entry,'create_certificates',lambda *a:{'der':der})
    monkeypatch.setattr(qrcode,'make',lambda *a:SimpleNamespace(save=lambda path:None))
    monkeypatch.setattr(entry.signal,'signal',lambda *a:None)
    clock=[0.];live=[False];observed=[]
    original_capture=guest_https.HTTPSRequestTimings
    monkeypatch.setattr(entry,'HTTPSRequestTimings',lambda path,duration:original_capture(path,duration=duration,clock=lambda:clock[0]))
    class Proxy:
        def __init__(self,*a,**kw):assert callable(kw['timing_callback'])
        def __enter__(self):live[0]=True
        def __exit__(self,*a):live[0]=False
    monkeypatch.setattr(entry,'GuestHTTPS',Proxy)
    def sleep(_):
        assert live[0]
        observed.append(timing.exists());clock[0]+=1.
        if len(observed)==2:stop.write_text('test owner stop')
    monkeypatch.setattr(entry,'time',SimpleNamespace(sleep=sleep))
    entry.main(['--address','127.0.0.1','--stop-file',str(stop),
                '--timing-output',str(timing),'--timing-duration','1'])
    assert observed==[False,True] and live[0] is False
    assert 'private-fixture-token' not in capsys.readouterr().out
