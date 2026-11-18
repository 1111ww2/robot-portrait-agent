"""接受的有限路径可能会丢失图像，永远不会失去访问者控制或后端租约。"""
import json
import time
import urllib.request

import pytest

from leica_tracker.guest_display import GuestDisplay
from leica_tracker.resident_guest_watchdog import LocalGuestControlReader


class Capture:
    current=True
    def connected(self):return self.current
    def fresh(self):return self.current
    def stream_fresh(self):return self.current
    def diagnostics(self):return {}


@pytest.fixture
def display():
    capture=Capture()
    with GuestDisplay(port=0,remote_capture=capture,physical_feedback=True) as d:
        with d._lock:
            d._active=True;d._owner='ipad-owner';d._seen=time.monotonic()
            d._control_epoch=1;d._control_action='start'
        yield d,capture


def register(d,mode='intro'):
    identity={'instance':d._control_instance,'epoch':d._control_epoch}
    with d._lock:
        result=d._physical_feedback_update({'op':'register','mode':mode,**identity})
    return {**identity,**result}


def update(d,identity,op,**kwargs):
    with d._lock:return d._physical_feedback_update({'op':op,**identity,**kwargs})


def test_registered_intro_survives_image_loss_with_real_control_heartbeat(display):
    d,capture=display;identity=register(d)
    update(d,identity,'intro',status='active',intro_id=8)
    capture.current=False
    update(d,identity,'heartbeat')
    update(d,identity,'intro',status='active',intro_id=8)
    assert d.physical_intro_active(8)
    state=d.control_snapshot()
    assert state['control_fresh'] is True
    assert state['camera_fresh'] is False and state['camera_connected'] is False


@pytest.mark.parametrize('expired',['control','backend'])
def test_intro_stops_if_either_live_control_or_backend_lease_expires(display,expired):
    d,capture=display;identity=register(d);capture.current=False
    with d._lock:
        if expired=='control':d._seen-=.31
        else:d._physical_backend_at-=.31
    with pytest.raises(ValueError):update(d,identity,'heartbeat')
    assert not d.physical_backend_available()
    with d._lock:d._seen=time.monotonic();d._physical_backend_at=time.monotonic()
    with pytest.raises(ValueError):update(d,identity,'heartbeat')


def test_intro_ready_metadata_switches_to_guidance_without_rechecking_image_age(display):
    d,capture=display;identity=register(d)
    update(d,identity,'intro',status='active',intro_id=8)
    capture.current=False
    update(d,identity,'intro',status='ready',intro_id=8)
    assert d._physical_backend_mode=='guidance' and d.physical_intro_ready(8)
    update(d,identity,'intro',status='ready',intro_id=8)
    update(d,identity,'heartbeat')
    assert d.physical_backend_available() and not d.control_snapshot()['camera_fresh']


@pytest.mark.parametrize('mode',['intro','guidance'])
def test_registration_always_requires_fresh_camera(display,mode):
    d,capture=display;capture.current=False
    with pytest.raises(ValueError):register(d,mode)


@pytest.mark.parametrize('op',['begin','sample','photo_ready'])
def test_intro_mode_cannot_be_used_for_guidance_actions(display,op):
    d,_=display;identity=register(d)
    with pytest.raises(ValueError,match='guidance'):update(d,identity,op)


def test_heartbeat_body_cannot_change_registered_mode_during_image_gap(display):
    d,capture=display;identity=register(d,'guidance');capture.current=False
    update(d,identity,'heartbeat',mode='intro')
    assert d._physical_backend_mode=='guidance' and d.physical_backend_available()


def test_only_owner_state_poll_refreshes_control_not_operator_or_backend(display):
    d,_=display
    def request(path,client):
        req=urllib.request.Request(d.origin+path,headers={'X-Guest-Token':d.token,'X-Guest-Client':client})
        with urllib.request.urlopen(req,timeout=1) as response:return json.load(response)
    with d._lock:d._seen-=.31
    assert request('/operator-state','ipad-owner')['control_fresh'] is False
    assert request('/state','physical-supervisor')['control_fresh'] is False
    assert request('/state','ipad-owner')['control_fresh'] is True
    reader=LocalGuestControlReader(d.port,d.token,timeout=.2)
    assert reader()['control_fresh'] is True
    with d._lock:d._seen-=.31
    assert reader()['control_fresh'] is False


def test_stop_and_epoch_changes_still_revoke_intro(display):
    d,capture=display;identity=register(d);capture.current=False
    with d._lock:d._control_epoch+=1;d._active=False
    with pytest.raises(ValueError):update(d,identity,'heartbeat')
    assert not d.physical_backend_available()


def test_late_owner_heartbeat_does_not_restore_expired_intro_control(display):
    d,_=display;identity=register(d)
    with d._lock:d._seen-=.31
    req=urllib.request.Request(d.origin+'/state',headers={
        'X-Guest-Token':d.token,'X-Guest-Client':'ipad-owner'})
    with urllib.request.urlopen(req,timeout=1) as response:
        assert json.load(response)['control_fresh'] is True
    with pytest.raises(ValueError):update(d,identity,'heartbeat')
