import json
import urllib.error
import urllib.request
import pytest
from leica_tracker.guest_display import GuestDisplay


class Capture:
    available=True
    def connected(self):return self.available
    def fresh(self):return self.available
    def stream_fresh(self):return self.available
    def diagnostics(self):return {}
    def control_owner(self):return 'actual-ipad' if self.available else None


def request(display,path,body=None,client='local-operator'):
    headers={'X-Guest-Token':display.token,'X-Guest-Client':client,'Origin':display.origin,'Content-Type':'application/json'}
    req=urllib.request.Request(display.origin+path,headers=headers,data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(req,timeout=1) as response:return json.load(response)


def action(display,name):
    control=display.control_snapshot()
    return {'instance':control['instance'],'epoch':control['epoch'],'action':name}


def test_operator_cannot_start_without_the_ipad_heartbeat_handshake():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        body=action(display,'start')
        with pytest.raises(urllib.error.HTTPError) as error:
            request(display,'/operator-action',body)
        assert error.value.code==409
        assert display.take_action() is None
        assert not display.active
        assert display._physical_lease is None
        with pytest.raises(urllib.error.HTTPError):request(display,'/operator-action',body)


def test_operator_cannot_start_without_current_camera_but_can_stop():
    camera=Capture()
    with GuestDisplay(port=0,remote_capture=camera,physical_feedback=True) as display:
        request(display,'/action',{'action':'start','control_protocol':1,
            'control':{k:display.control_snapshot()[k] for k in ('instance','epoch')}},client='actual-ipad')
        camera.available=False
        with pytest.raises(urllib.error.HTTPError):request(display,'/operator-action',action(display,'restart'))
        assert request(display,'/operator-action',action(display,'stop'))['ok']
        assert display.active is False


def test_operator_route_is_not_available_in_simulator_or_unpaired_mode():
    with GuestDisplay(port=0,remote_capture=Capture()) as display:
        with pytest.raises(urllib.error.HTTPError):request(display,'/operator-action',action(display,'start'))
