"""仅循环回显假显示；不会创建机器人控制器。"""
import json
import time
import urllib.error
import urllib.request

import pytest

from leica_tracker.guest_display import GuestDisplay


class Capture:
    def connected(self): return True
    def fresh(self): return True
    def stream_fresh(self): return True
    def diagnostics(self): return {}


def call(d, path, body=None, client='owner'):
    request=urllib.request.Request(d.origin+path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'X-Guest-Token':d.token,'X-Guest-Client':client,
                 'Content-Type':'application/json','Origin':d.origin})
    with urllib.request.urlopen(request, timeout=1) as response:
        return json.load(response)


def start(d):
    control=call(d,'/state')['control']
    return call(d,'/action',{'action':'start','control_protocol':1,'control':control})['control']


def beat(d, identity, sequence=1, **kwargs):
    return call(d,'/control-heartbeat',{**identity,'sequence':sequence},**kwargs)


def test_physical_start_rejects_legacy_page_before_activating():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as d:
        with pytest.raises(urllib.error.HTTPError) as error:
            call(d,'/action',{'action':'start'})
        assert error.value.code==409
        assert not d.active
        assert start(d)['epoch']>0


def test_start_sent_before_stop_cannot_be_applied_after_stop():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as d:
        before=call(d,'/state')['control']
        call(d,'/action',{'action':'stop'})
        with pytest.raises(urllib.error.HTTPError) as error:
            call(d,'/action',{'action':'start','control_protocol':1,'control':before})
        assert error.value.code==409
        assert not d.active


def test_heartbeat_is_independent_of_state_snapshot_and_does_not_start():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as d:
        identity=start(d)
        with d._lock:d._seen-=.1
        before=d._seen
        assert beat(d,identity)['accepted'] is True
        assert d._seen>before
        call(d,'/action',{'action':'stop'})
        with pytest.raises(urllib.error.HTTPError):beat(d,identity,2)
        assert not d.active


def test_new_protocol_state_reads_do_not_renew_owner_control():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as d:
        start(d)
        with d._lock:d._seen-=.1
        before=d._seen
        call(d,'/state')
        assert d._seen==before


def test_https_proxy_forwards_lightweight_owner_heartbeat(tmp_path):
    import ssl
    from leica_tracker.guest_https import GuestHTTPS,create_certificates
    files=create_certificates(tmp_path,'127.0.0.1')
    context=ssl.create_default_context(cafile=str(files['ca']))
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as d:
        with GuestHTTPS(files,upstream_port=d.port,host='127.0.0.1',port=0,setup_port=0,addresses=[]) as proxy:
            origin=f'https://127.0.0.1:{proxy.port}'
            def post(path,body,client='owner'):
                r=urllib.request.Request(origin+path,data=json.dumps(body).encode(),headers={
                    'X-Guest-Token':d.token,'X-Guest-Client':client,
                    'Origin':origin,'Content-Type':'application/json'})
                with urllib.request.urlopen(r,context=context,timeout=1) as response:return json.load(response)
            control=call(d,'/state')['control']
            identity=post('/action',{'action':'start','control_protocol':1,'control':control})['control']
            assert post('/control-heartbeat',{**identity,'sequence':1})['accepted']
            with pytest.raises(urllib.error.HTTPError):
                post('/control-heartbeat',{**identity,'sequence':2},client='other')
            post('/action',{'action':'stop'})
            with pytest.raises(urllib.error.HTTPError):
                post('/control-heartbeat',{**identity,'sequence':3})


def test_wrong_owner_old_epoch_and_reordered_packets_cannot_refresh():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as d:
        identity=start(d)
        assert beat(d,identity,2)['accepted']
        before=d._seen
        assert beat(d,identity,1)['accepted'] is False
        assert d._seen==before
        with pytest.raises(urllib.error.HTTPError):beat(d,identity,3,client='stranger')
        assert d._seen==before
        call(d,'/action',{'action':'stop'})
        new=start(d)
        assert new['epoch']!=identity['epoch']
        before=d._seen
        with pytest.raises(urllib.error.HTTPError):beat(d,identity,4)
        assert d._seen==before


def test_expired_intro_cannot_be_revived_by_heartbeat_or_state_read():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as d:
        identity=start(d)
        with d._lock:
            d._physical_feedback_update({'op':'register','mode':'intro',**identity})
            d._seen-=.312
        with pytest.raises(urllib.error.HTTPError):beat(d,identity)
        before=d._seen
        call(d,'/state')
        assert d._seen==before
        assert not d.physical_backend_available()
        assert d._physical_revocation['reason']=='control_heartbeat_expired'


@pytest.mark.parametrize('sequence',[True,0,-1,1.5,'1',2**31])
def test_invalid_sequence_rejected_without_freshness_change(sequence):
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as d:
        identity=start(d);before=d._seen
        with pytest.raises(urllib.error.HTTPError):beat(d,identity,sequence)
        assert d._seen==before
