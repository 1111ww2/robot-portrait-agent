import json
import time
import urllib.error
import urllib.request

import pytest
from leica_tracker.guest_display import GuestDisplay


class Capture:
    def connected(self):return True
    def fresh(self):return True
    def stream_fresh(self):return True
    def diagnostics(self):return {}


def request(display,path,body=None):
    headers={'X-Guest-Token':display.token,'X-Guest-Client':'test-supervisor','Origin':display.origin,'Content-Type':'application/json'}
    req=urllib.request.Request(display.origin+path,headers=headers,data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(req,timeout=1) as response:return json.load(response)


def test_status_only_never_activates_or_registers_a_motion_backend():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        before=display.control_snapshot();seen=display._seen
        body={k:before[k] for k in ('instance','epoch')};body.update(status='fault',reason='camera')
        assert request(display,'/runtime-status',body)['ok']
        state=request(display,'/state')
        assert state['runtime']['status']=='fault' and state['runtime']['reason']=='camera'
        assert not state['active'] and display._physical_lease is None and display._seen==seen
        request(display,'/action',{'action':'start','control_protocol':1,'control':{k:before[k] for k in ('instance','epoch')}})
        with pytest.raises(urllib.error.HTTPError):request(display,'/runtime-status',body)
        assert request(display,'/state')['runtime']['status']=='starting'


@pytest.mark.parametrize('changes',[{'status':'go'},{'reason':'untrusted detail'},{'epoch':True}])
def test_status_rejects_invalid_or_unbound_content(changes):
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        control=display.control_snapshot()
        body={k:control[k] for k in ('instance','epoch')};body.update(status='waiting_start',reason='none');body.update(changes)
        with pytest.raises(urllib.error.HTTPError):request(display,'/runtime-status',body)


def test_inactive_actual_reference_hold_remains_visible_over_waiting_status_without_mutation():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        control=display.control_snapshot()
        body={k:control[k] for k in ('instance','epoch')}
        body.update(status='waiting_start',reason='none')
        request(display,'/runtime-status',body)
        with display._lock:
            display._reference_gate.begin([0.]*6,time.monotonic())
            display._reference_gate.acknowledge()
            gate_before=vars(display._reference_gate).copy()
            reported_before=display._runtime_status.copy()
            seen_before=display._seen
        for path in ('/state','/operator-state'):
            state=request(display,path)
            assert state['runtime']['status']=='fault'
            assert state['runtime']['reason']=='reference'
            assert not state['active']
        with display._lock:
            assert vars(display._reference_gate)==gate_before
            assert display._runtime_status==reported_before
            assert display._seen==seen_before
            assert display._physical_lease is None
        assert display.control_snapshot()==control


def test_active_running_reference_hold_is_not_reported_as_a_fault():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        control=display.control_snapshot()
        request(display,'/action',{'action':'start','control_protocol':1,
            'control':{k:control[k] for k in ('instance','epoch')}})
        control=display.control_snapshot()
        body={k:control[k] for k in ('instance','epoch')}
        body.update(status='running',reason='none')
        request(display,'/runtime-status',body)
        with display._lock:display._reference_gate.begin([0.]*6,time.monotonic())
        state=request(display,'/state')
        assert state['active']
        assert state['runtime']['status']=='running'
        assert state['runtime']['reason']=='none'


@pytest.mark.parametrize('reason',['speed','control','camera','runtime'])
def test_current_explicit_fault_has_priority_over_inactive_reference_hold(reason):
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        control=display.control_snapshot()
        body={k:control[k] for k in ('instance','epoch')}
        body.update(status='fault',reason=reason)
        request(display,'/runtime-status',body)
        with display._lock:display._reference_gate.begin([0.]*6,time.monotonic())
        state=request(display,'/state')
        assert state['runtime']['status']=='fault'
        assert state['runtime']['reason']==reason


def test_old_reference_phase_without_an_actual_gate_does_not_override_waiting_status():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        control=display.control_snapshot()
        body={k:control[k] for k in ('instance','epoch')}
        body.update(status='waiting_start',reason='none')
        request(display,'/runtime-status',body)
        with display._lock:display._state['phase']='PHYSICAL_REFERENCE_HOLD'
        state=request(display,'/state')
        assert state['runtime']['status']=='waiting_start'
        assert state['runtime']['reason']=='none'
