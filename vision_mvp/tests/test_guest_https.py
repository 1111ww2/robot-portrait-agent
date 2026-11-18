import json
import ssl
import urllib.request
import urllib.error

import pytest

pytest.importorskip('cryptography',reason='Install the handoff extra for local HTTPS')


def test_certificate_server_pairing_survives_camera_shutdown(tmp_path):
    from leica_tracker.guest_https import load_pairing
    live=tmp_path/'live.json';cached=tmp_path/'https'/'pair.json'
    pair={'port':8767,'key':'test-token-abcdefghijklmnopqrstuvwxyz'}
    live.write_text(json.dumps(pair))
    assert load_pairing(live,cached)==pair
    live.unlink()
    assert load_pairing(live,cached)==pair
    live.write_text(json.dumps({'port':8767,'key':'bad'}))
    with pytest.raises(ValueError):load_pairing(live,cached)

def test_local_certificate_and_https_proxy_preserve_pairing_and_origin(tmp_path):
    from leica_tracker.guest_https import create_certificates, GuestHTTPS
    from leica_tracker.guest_display import GuestDisplay
    files=create_certificates(tmp_path,'127.0.0.1')
    context=ssl.create_default_context(cafile=str(files['ca']))
    with GuestDisplay(port=0) as guest, GuestHTTPS(files,upstream_port=guest.port,host='127.0.0.1',port=0,setup_port=0,addresses=['127.0.0.1']) as proxy:
        origin=f'https://127.0.0.1:{proxy.port}'
        headers={'X-Guest-Token':guest.token,'X-Guest-Client':'tls-test','Origin':origin,'Content-Type':'application/json'}
        def call(path,body=None,extra=None):
            req=urllib.request.Request(origin+path,headers={**headers,**(extra or {})},data=json.dumps(body).encode() if body else None)
            return urllib.request.urlopen(req,context=context,timeout=3)
        with call('/?key='+guest.token) as r:assert r.status==200
        with call('/state') as r:assert not json.load(r)['active']
        with call('/action',{'action':'start'}) as r:assert json.load(r)['active']
        with call('/message',{'text':'Half-length portrait'}) as r:assert r.status==200
        assert guest.take_messages()==['Half-length portrait']
        with pytest.raises(urllib.error.HTTPError) as error:call('/action',{'action':'stop'},{'Origin':'https://evil.invalid'})
        assert error.value.code==403 and guest.active
        with pytest.raises(urllib.error.HTTPError) as error:call('/state',extra={'X-Guest-Token':'wrong'})
        assert error.value.code==403
        with pytest.raises(urllib.error.HTTPError) as error:call('/state',extra={'Host':'evil.invalid'})
        assert error.value.code==403
        with call('/action',{'action':'stop'}) as r:assert not json.load(r)['active']
        for secret_path in ('/server-key.pem','/ca-key.pem','/../ca-key.pem'):
            with pytest.raises(urllib.error.HTTPError):call(secret_path)
        setup=f'http://127.0.0.1:{proxy.setup_port}'
        with urllib.request.urlopen(setup+'/leica-root.mobileconfig',timeout=3) as r:
            import plistlib
            profile=plistlib.loads(r.read())
            assert profile['PayloadContent'][0]['PayloadType']=='com.apple.security.root'
        with pytest.raises(urllib.error.HTTPError):urllib.request.urlopen(setup+'/ca-key.pem',timeout=3)
        with pytest.raises(urllib.error.URLError):urllib.request.urlopen(origin+'/state',timeout=3)


def test_certificates_reused_and_new_address_needs_explicit_configuration(tmp_path):
    from leica_tracker.guest_https import create_certificates
    first=create_certificates(tmp_path,'192.168.1.20')
    before=first['ca'].read_bytes()
    create_certificates(tmp_path,'192.168.1.20')
    assert first['ca'].read_bytes()==before
    create_certificates(tmp_path,'192.168.1.21')
    assert first['ca'].read_bytes()==before
    from cryptography import x509
    import ipaddress
    cert=x509.load_pem_x509_certificate(first['cert'].read_bytes())
    assert ipaddress.ip_address('192.168.1.21') in cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(x509.IPAddress)


def test_https_forwards_paired_camera_jpeg_with_separate_body_limit(tmp_path):
    from leica_tracker.guest_https import create_certificates,GuestHTTPS
    from leica_tracker.guest_display import GuestDisplay
    from leica_tracker.ipad_camera import IPadCapture
    import cv2,numpy as np
    files=create_certificates(tmp_path,'127.0.0.1');c=IPadCapture(read_timeout=0)
    context=ssl.create_default_context(cafile=str(files['ca']))
    with GuestDisplay(port=0,remote_capture=c) as guest, GuestHTTPS(files,upstream_port=guest.port,host='127.0.0.1',port=0,setup_port=0,addresses=[]) as proxy:
        origin=f'https://127.0.0.1:{proxy.port}'
        headers={'X-Guest-Token':guest.token,'X-Guest-Client':'ipad','Origin':origin,'Content-Type':'application/json'}
        def post(path,data,extra=None):
            req=urllib.request.Request(origin+path,data=data,headers={**headers,**(extra or {})})
            return urllib.request.urlopen(req,context=context,timeout=3)
        with post('/camera/start',b'{}') as r:s=json.load(r)['session']
        with post('/camera/ticket',json.dumps({'session':s}).encode()) as r:t=json.load(r)['ticket']
        data=cv2.imencode('.jpg',np.random.default_rng(4).integers(0,255,(120,180,3),dtype=np.uint8))[1].tobytes()
        assert len(data)>2048
        with post('/camera/frame',data,{'Content-Type':'image/jpeg','X-Camera-Session':s,'X-Camera-Ticket':t}) as r:assert r.status==200
        assert c.read()[0]
        with pytest.raises(urllib.error.HTTPError):post('/message',b'x'*2049)


def test_https_reuses_socket_and_rejected_body_does_not_desync(tmp_path):
    import http.client
    from leica_tracker.guest_https import create_certificates,GuestHTTPS
    from leica_tracker.guest_display import GuestDisplay
    files=create_certificates(tmp_path,'127.0.0.1')
    with GuestDisplay(port=0) as guest, GuestHTTPS(files,upstream_port=guest.port,host='127.0.0.1',port=0,setup_port=0,addresses=[]) as proxy:
        context=ssl.create_default_context(cafile=str(files['ca']))
        c=http.client.HTTPSConnection('127.0.0.1',proxy.port,context=context,timeout=2)
        try:
            c.request('GET','/state',headers={'X-Guest-Token':guest.token});r=c.getresponse();r.read()
            assert r.version==11 and c.sock is not None
            sock=c.sock
            c.request('GET','/state',headers={'X-Guest-Token':guest.token});r=c.getresponse();r.read()
            assert r.status==200 and c.sock is sock
            c.request('POST','/action',b'{}',headers={'Content-Type':'application/json','Origin':'https://wrong'})
            r=c.getresponse();r.read();assert r.status==403 and c.sock is None
        finally:c.close()


def test_small_rejected_post_body_is_drained_but_large_or_streamed_bodies_are_not():
    import io
    from leica_tracker.guest_https import drain_rejected_body
    class Connection:
        def __init__(self):self.timeout=None
        def settimeout(self,value):self.timeout=value
    for headers,expected in [({'Content-Length':'2'},2),({'Content-Length':'2049'},0),
                             ({'Content-Length':'2','Transfer-Encoding':'chunked'},0),
                             ({'Content-Length':'bad'},0)]:
        stream=io.BytesIO(b'{}');connection=Connection()
        drain_rejected_body(headers,stream,connection)
        assert stream.tell()==expected
        if expected:assert connection.timeout==.25


def test_https_audio_upload_forwards_language_with_its_own_body_limit(tmp_path):
    from leica_tracker.guest_https import create_certificates,GuestHTTPS
    from leica_tracker.guest_display import GuestDisplay
    files=create_certificates(tmp_path,'127.0.0.1')
    with GuestDisplay(port=0) as guest, GuestHTTPS(files,upstream_port=guest.port,host='127.0.0.1',port=0,setup_port=0,addresses=[]) as proxy:
        class Speech:
            ready=True
            def transcribe(self,data,language):
                assert len(data)==4096 and language=='zh'
                return 'Test response'
        guest.speech=Speech()
        context=ssl.create_default_context(cafile=str(files['ca']));origin=f'https://127.0.0.1:{proxy.port}'
        headers={'X-Guest-Token':guest.token,'X-Guest-Client':'test','Origin':origin,'Content-Type':'application/json'}
        with urllib.request.urlopen(urllib.request.Request(origin+'/action',data=b'{"action":"start"}',headers=headers),context=context,timeout=3):pass
        headers.update({'Content-Type':'application/octet-stream','X-Speech-Language':'zh'})
        with urllib.request.urlopen(urllib.request.Request(origin+'/transcribe',data=b'x'*4096,headers=headers),context=context,timeout=3) as r:assert json.load(r)['text']=='Test response'


def test_https_rejected_origin_drains_body_sent_after_headers(tmp_path):
    """TLS 可能分别写入标头和正文；请求被拒绝时客户端仍应收到 403。"""
    import http.client
    import time
    from leica_tracker.guest_https import create_certificates,GuestHTTPS
    from leica_tracker.guest_display import GuestDisplay

    class SplitBodyConnection(http.client.HTTPSConnection):
        def send(self,data):
            if data==b'{}':time.sleep(.06)
            return super().send(data)

    files=create_certificates(tmp_path,'127.0.0.1')
    with GuestDisplay(port=0) as guest, GuestHTTPS(files,upstream_port=guest.port,host='127.0.0.1',port=0,setup_port=0,addresses=[]) as proxy:
        context=ssl.create_default_context(cafile=str(files['ca']))
        connection=SplitBodyConnection('127.0.0.1',proxy.port,context=context,timeout=2)
        try:
            for _ in range(2):
                connection.request('GET','/state',headers={'X-Guest-Token':guest.token})
                response=connection.getresponse();response.read();assert response.status==200
            connection.request('POST','/action',b'{}',headers={'Content-Type':'application/json','Origin':'https://wrong'})
            response=connection.getresponse();response.read()
            assert response.status==403 and connection.sock is None
        finally:connection.close()

def test_https_photo_upload_requires_owner_and_passes_to_one_photo_share(tmp_path):
    import json,ssl,time,urllib.request,urllib.error
    from leica_tracker.guest_https import create_certificates,GuestHTTPS
    from leica_tracker.guest_display import GuestDisplay
    class Capture:
        def fresh(self):return True
    class Feedback:
        def photo_ready(self,source):return source==42
        def close(self):pass
    files=create_certificates(tmp_path,'127.0.0.1');ctx=ssl.create_default_context(cafile=str(files['ca']))
    with GuestDisplay(port=0,remote_capture=Capture()) as guest, GuestHTTPS(files,upstream_port=guest.port,host='127.0.0.1',port=0,setup_port=0,addresses=[]) as proxy:
        origin=f'https://127.0.0.1:{proxy.port}'
        headers={'X-Guest-Token':guest.token,'X-Guest-Client':'photo-test','Origin':origin,'Content-Type':'application/json'}
        with urllib.request.urlopen(urllib.request.Request(origin+'/action',data=b'{"action":"start"}',headers=headers),context=ctx,timeout=3):pass
        with guest._lock:
            guest._sim_feedback=Feedback();guest._gesture=object();guest._gesture_source_ns=42;guest._updated=time.monotonic();guest._photo_latched=True
            guest._state={'phase':'ARRIVED','tracking':True};guest._operator={'photo_current':(.5,.4,.3),'target':(.5,.4,.3),'tolerances':(.05,.05,.05)}
        headers['Content-Type']='image/jpeg'
        with urllib.request.urlopen(urllib.request.Request(origin+'/photo',data=b'\xff\xd8\xfftest\xff\xd9',headers=headers),context=ctx,timeout=3) as response:
            result=json.load(response)
        assert result['qr_data_url'].startswith('data:image/png;base64,') and '_url' not in result
        token=guest._photo_share.item[0]
        with urllib.request.urlopen(f'http://127.0.0.1:{guest._photo_share.port}/image/{token}',timeout=3) as response:
            assert response.headers['Cache-Control']=='no-store' and response.read()==b'\xff\xd8\xfftest\xff\xd9'
        headers['Origin']='https://wrong.example'
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(origin+'/photo',data=b'\xff\xd8\xfftest\xff\xd9',headers=headers),context=ctx,timeout=3)
        assert error.value.code==403
