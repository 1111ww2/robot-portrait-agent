"""验证实际感知和回环地址上显示接口的约定，不调用机器人或模型 API。"""
from concurrent.futures import Future
import json
from types import SimpleNamespace
import urllib.error
import urllib.request

import numpy as np
import pytest

from leica_tracker.gemini_framing import CompositionPlan
from leica_tracker.gemini_guidance import GeminiGuidanceController
from leica_tracker.guest_display import GuestDisplay
from leica_tracker import guest_display as display_module
from leica_tracker.person_guidance import GuidanceDecision, GuidanceCue

FRAME=np.zeros((72,128,3),np.uint8)
PLAN=CompositionPlan((.62,.30,.24),(.12,.04,.85,.94),.9,'environmental','fixture')


@pytest.fixture(autouse=True)
def display_clock(monkeypatch):
    # 保留实际 HTTP 解析和线程，控制协议时间，避免机器繁忙导致显示测试意外触发租约过期。
    clock=SimpleNamespace(now=1000.)
    monkeypatch.setattr(display_module,'time',SimpleNamespace(monotonic=lambda:clock.now,
        step=lambda:setattr(clock,'now',clock.now+.001)))
    return clock


class Worker:
    def __init__(self):self.calls=[]
    def submit(self,jpeg,point):
        future=Future();self.calls.append(future);return future


def tick(controller,timestamp,x=.5,width=.2):
    return controller.update(SimpleNamespace(center_x=x,head_y=.3,shoulder_width=width,confidence=.9),
                             timestamp,frame=FRAME)


def planned():
    worker=Worker();controller=GeminiGuidanceController(worker=worker,single_plan=True,entry_ack_ms=1)
    tick(controller,0);worker.calls[0].set_result(PLAN);tick(controller,100)
    return controller,worker


def test_known_motion_suspends_observation_without_fake_loss_or_new_invitation():
    controller,worker=planned();inner=controller._inner
    for timestamp in (200,1000,50000):
        hold=controller.suspend_perception(timestamp)
        assert hold.cue is GuidanceCue.HOLD and not hold.tracking
        assert hold.reason=='AWAITING_MEASURED_REFERENCE'
        assert controller._plan is PLAN and controller._inner is inner
        assert controller._lost_since is None and controller._entry_seen
    decision=tick(controller,50100,x=.8,width=.24)
    assert decision.model_status=='AI_PLAN_LOCKED' and decision.target==PLAN.target
    assert decision.cue is not GuidanceCue.INVITE and decision.phase!='AI_ENTRY_ACK'
    assert len(worker.calls)==1 and controller._entry_seen
    assert controller._inner._arrival_since_ms is None
    assert controller._last_point==(.8,.3,.24)


def test_pending_model_result_can_cache_while_suspended_but_never_emits_a_move():
    worker=Worker();controller=GeminiGuidanceController(worker=worker,single_plan=True,entry_ack_ms=1)
    tick(controller,0);worker.calls[0].set_result(PLAN)
    hold=controller.suspend_perception(100)
    assert hold.cue is GuidanceCue.HOLD and not hold.tracking
    assert controller._scene_template is PLAN and controller._pending is None
    for timestamp in range(200,1200,100):controller.suspend_perception(timestamp)
    decision=tick(controller,1300)
    assert decision.target==PLAN.target and len(worker.calls)==1


def test_explicit_new_target_during_suspend_still_discards_the_old_scene():
    controller,worker=planned();controller.suspend_perception(200)
    controller.restart_encounter();tick(controller,300)
    assert len(worker.calls)==1 and controller._scene_template is None
    for timestamp in range(400,5500,100):tick(controller,timestamp)
    assert len(worker.calls)==2 and controller._scene_template is None


def test_return_observation_starts_a_new_arrival_dwell_instead_of_reusing_old_measurements():
    controller,worker=planned()
    for timestamp in range(200,1300,100):decision=tick(controller,timestamp,x=.62,width=.24)
    assert decision.phase=='SETTLING' and controller._inner._arrival_since_ms is not None
    controller.suspend_perception(1400);controller.suspend_perception(5000)
    resumed=tick(controller,5100,x=.62,width=.24)
    assert resumed.phase=='SETTLING' and resumed.arrival_progress==0
    assert resumed.cue is not GuidanceCue.INVITE and len(worker.calls)==1


class Capture:
    def connected(self):return True
    def fresh(self):return True
    def stream_fresh(self):return True
    def diagnostics(self):return {}
    def end(self,client,session):
        if client!='owner' or session!='camera-session':raise ValueError('Wrong camera session')


def call(display,path,body=None,client='owner'):
    display_module.time.step()
    request=urllib.request.Request(display.origin+path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'X-Guest-Token':display.token,'X-Guest-Client':client,
                 'Content-Type':'application/json','Origin':display.origin})
    with urllib.request.urlopen(request,timeout=1) as response:return json.load(response)


def start(display):
    return call(display,'/action',{'action':'start','control_protocol':1,
                                'control':call(display,'/state')['control']})['control']


def begin(display,control):
    lease=call(display,'/physical-feedback',{**control,'op':'register'})['lease']
    source=round(display_module.time.monotonic()*1e9)
    identity={**control,'lease':lease,'source_ns':source,'action_id':7,'cue':'MOVE_RIGHT'}
    call(display,'/physical-feedback',{**identity,'op':'begin','reference_position':[0.,0.,0.],
        'direction':[1.,0.,0.],'planned_displacement':.12})
    return identity


def test_combined_intro_heartbeat_atomically_keeps_lease_and_acknowledges_actual_stage():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        control=start(display)
        lease=call(display,'/physical-feedback',{**control,'op':'register','mode':'intro'})['lease']
        identity={**control,'lease':lease};before=display._physical_backend_at
        call(display,'/physical-feedback',{**identity,'op':'heartbeat','intro_id':2,'intro_status':'active'})
        assert display._physical_backend_at>before and display.physical_intro_active(2)
        before=display._physical_backend_at
        call(display,'/physical-feedback',{**identity,'op':'heartbeat','intro_id':2,'intro_status':'ready'})
        assert display._physical_backend_at>before and display.physical_intro_ready(2)
        assert display._physical_backend_mode=='guidance'


@pytest.mark.parametrize('metadata',[{'intro_id':2,'intro_status':'ready'},
    {'intro_id':2,'intro_status':'untrusted'}, {'intro_id':True,'intro_status':'active'},
    {'intro_id':2}, {'intro_status':'active'}])
def test_invalid_combined_intro_metadata_cannot_renew_control_lease(metadata):
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        control=start(display)
        lease=call(display,'/physical-feedback',{**control,'op':'register','mode':'intro'})['lease']
        before=display._physical_backend_at
        with pytest.raises(urllib.error.HTTPError):
            call(display,'/physical-feedback',{**control,'lease':lease,'op':'heartbeat',**metadata})
        assert display._physical_backend_at==before and display._physical_intro is None
        assert display._physical_backend_mode=='intro'


def test_fresh_native_motion_cue_survives_reference_observation_hold_without_secret_identity():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        control=start(display);identity=begin(display,control)
        display.publish(None,GuidanceDecision('PHYSICAL_REFERENCE_HOLD',GuidanceCue.HOLD,
            (.5,.3,.2),None,None,False,'AWAITING_MEASURED_REFERENCE'))
        assert call(display,'/state')['motion']=={'active':False,'fresh':False}
        call(display,'/physical-feedback',{**identity,'op':'sample','captured_at':display_module.time.monotonic(),
             'current_position':[.048,0.,0.],'phase':'returning'})
        state=call(display,'/state');motion=state['motion']
        assert state['tracking'] is False and state['direction'] is None
        assert motion['active'] and motion['fresh'] and motion['cue']=='MOVE_RIGHT'
        assert motion['action_id']==state['feedback']['gesture_id']==7
        assert motion['phase']=='returning' and motion['progress']==pytest.approx(.4)
        assert 0<=motion['sample_age_ms']<=300
        assert state['feedback']['edge']=='right' and state['feedback']['intensity']==pytest.approx(.4)
        assert not any(key in motion for key in ('lease','instance','source_ns','epoch'))
        with display._lock:display._physical_action['captured_at']-=.31
        stale=call(display,'/state')
        assert stale['motion']=={'active':False,'fresh':False}
        assert stale['feedback']['gesture_id']==0
        call(display,'/physical-feedback',{**identity,'op':'sample','captured_at':display_module.time.monotonic(),
            'current_position':[.06,0.,0.],'phase':'returning'})
        recovered=call(display,'/state')
        assert recovered['motion']['action_id']==7 and recovered['motion']['fresh']
        assert recovered['feedback']['intensity']==pytest.approx(.5)


def test_motion_phase_is_a_bounded_enum_not_free_text():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        identity=begin(display,start(display))
        with pytest.raises(urllib.error.HTTPError):
            call(display,'/physical-feedback',{**identity,'op':'sample','captured_at':display_module.time.monotonic(),
                'current_position':[.048,0.,0.],'phase':'arbitrary-private-text'})
        assert call(display,'/state')['motion']=={'active':False,'fresh':False}


def test_late_presentation_clear_cannot_erase_a_newer_measured_motion():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        identity=begin(display,start(display));new={**identity,'action_id':8}
        call(display,'/physical-feedback',{**new,'op':'begin','reference_position':[0.,0.,0.],
            'direction':[1.,0.,0.],'planned_displacement':.12})
        call(display,'/physical-feedback',{**new,'op':'sample','captured_at':display_module.time.monotonic(),
            'current_position':[.024,0.,0.]})
        assert call(display,'/state')['motion']['action_id']==8
        result=call(display,'/physical-feedback',{**identity,'op':'clear'})
        assert call(display,'/state')['motion'].get('action_id')==8
        assert result['cleared'] is False
        assert call(display,'/state')['motion']['action_id']==8
        assert call(display,'/physical-feedback',{**new,'op':'clear'})['cleared'] is True
        assert call(display,'/state')['motion']=={'active':False,'fresh':False}


def test_late_sample_from_an_old_action_cannot_blank_the_current_motion():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        old=begin(display,start(display));new={**old,'action_id':8}
        call(display,'/physical-feedback',{**new,'op':'begin','reference_position':[0.,0.,0.],
            'direction':[1.,0.,0.],'planned_displacement':.12})
        call(display,'/physical-feedback',{**new,'op':'sample','captured_at':display_module.time.monotonic(),
            'current_position':[.024,0.,0.]})
        with pytest.raises(urllib.error.HTTPError):
            call(display,'/physical-feedback',{**old,'op':'sample','captured_at':display_module.time.monotonic(),
                'current_position':[.09,0.,0.]})
        state=call(display,'/state')
        assert state['motion']['action_id']==8 and state['feedback']['intensity']==pytest.approx(.2)


def test_presentation_feedback_does_not_renew_the_control_backend_heartbeat():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        identity=begin(display,start(display));stamp=display._physical_backend_at
        call(display,'/physical-feedback',{**identity,'op':'sample','captured_at':display_module.time.monotonic(),
            'current_position':[.024,0.,0.]})
        call(display,'/physical-feedback',{**identity,'op':'clear'})
        assert display._physical_backend_at==stamp


def test_actual_fresh_motion_promotes_only_display_status_when_running_report_was_missed():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        control=start(display);identity=begin(display,control)
        call(display,'/runtime-status',{**control,'status':'starting','reason':'none'})
        before=dict(display._runtime_status)
        call(display,'/physical-feedback',{**identity,'op':'sample','captured_at':display_module.time.monotonic(),
            'current_position':[.024,0.,0.]})
        state=call(display,'/state')
        assert state['motion']['active'] and state['runtime']=={'epoch':control['epoch'],'status':'running','reason':'none'}
        assert display._runtime_status==before
        assert display._active and display._physical_backend_at is not None
        with display._lock:display._physical_action['captured_at']-=.31
        assert call(display,'/state')['runtime']['status']=='starting'


def test_actual_motion_never_hides_an_explicit_current_runtime_fault():
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        control=start(display);identity=begin(display,control)
        call(display,'/physical-feedback',{**identity,'op':'sample','captured_at':display_module.time.monotonic(),
            'current_position':[.024,0.,0.]})
        call(display,'/runtime-status',{**control,'status':'fault','reason':'speed'})
        state=call(display,'/state')
        assert state['motion']['active'] and state['runtime']['status']=='fault'
        assert state['runtime']['reason']=='speed'


def latched_photo(display,control,identity):
    source=int(display_module.time.monotonic()*1e9)
    display.arm_physical_photo(source)
    call(display,'/physical-feedback',{**identity,'op':'photo_ready','source_ns':source})
    call(display,'/physical-feedback',{**identity,'op':'clear'})
    display.publish(FRAME,GuidanceDecision('ARRIVED',GuidanceCue.HOLD,(.62,.3,.24),None,
        (.62,.3,.24),True,'TARGET_LATCHED'))


def test_latched_native_photo_promotes_display_status_without_a_motion_or_status_post(display_clock):
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        control=start(display);identity=begin(display,control)
        call(display,'/runtime-status',{**control,'status':'starting','reason':'none'})
        before=dict(display._runtime_status)
        latched_photo(display,control,identity)
        # 锁定后首次停止确认仍然有效，但独立的后端租约和所有者租约都必须保持有效。
        display_clock.now+=.4
        display._seen=display_clock.now
        display._physical_backend_at=display_clock.now
        state=call(display,'/state')
        assert not state['motion']['active'] and state['photo_ready'] and state['photo_latched']
        assert state['runtime']=={'epoch':control['epoch'],'status':'running','reason':'none'}
        assert display._runtime_status==before


@pytest.mark.parametrize('invalid',['missing','old_epoch','unconfirmed','future','unlatched','backend_expired','fault'])
def test_photo_display_status_requires_current_native_confirmation_and_live_backend(invalid):
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        control=start(display);identity=begin(display,control)
        call(display,'/runtime-status',{**control,'status':'starting','reason':'none'})
        latched_photo(display,control,identity)
        if invalid=='missing':display._physical_photo=None
        elif invalid=='old_epoch':display._physical_photo['epoch']-=1
        elif invalid=='unconfirmed':display._physical_photo['ready_at']=None
        elif invalid=='future':display._physical_photo['ready_at']=display_module.time.monotonic()+1
        elif invalid=='unlatched':display._photo_latched=False
        elif invalid=='backend_expired':display._physical_backend_at-=.31
        elif invalid=='fault':call(display,'/runtime-status',{**control,'status':'fault','reason':'speed'})
        state=call(display,'/state')
        assert state['runtime']['status']==('fault' if invalid=='fault' else 'starting')


@pytest.mark.parametrize('stage',['headers','body'])
def test_disconnected_http_reply_is_not_retried_as_a_second_response(stage):
    with GuestDisplay(port=0) as display:
        handler_class=display._handler();handler=handler_class.__new__(handler_class)
        responses=[]
        handler.send_response=lambda code:responses.append(code)
        handler.send_header=lambda *args:None
        def aborted(*args):raise ConnectionAbortedError('fixture client closed')
        handler.end_headers=aborted if stage=='headers' else lambda:None
        handler.wfile=SimpleNamespace(write=aborted)
        handler.reply(200,{'ok':True})
        assert responses==[200] and handler.close_connection


def test_current_owner_page_hidden_stop_still_cancels_motion(capsys):
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        current=start(display);begin(display,current)
        call(display,'/action',{'action':'stop','control_protocol':1,'stop_origin':5,'control':current})
        assert not display.active
        line=next(line for line in capsys.readouterr().err.splitlines() if line.startswith('CONTROL_STOP_ACCEPTED '))
        data=json.loads(line.split(' ',1)[1])
        assert data['stop_origin']==5 and data['owner_matches'] and data['control_matches']


def test_automatic_stop_from_old_page_cannot_cancel_the_current_owner(capsys):
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        current=start(display)
        with pytest.raises(urllib.error.HTTPError) as caught:
            call(display,'/action',{'action':'stop','control_protocol':1,'stop_origin':5,
                'control':{**current,'epoch':current['epoch']-1}},client='old-page')
        assert caught.value.code==409 and display.active
        assert 'CONTROL_STOP_ACCEPTED' not in capsys.readouterr().err


def test_explicit_stop_remains_available_and_records_only_numeric_bounded_origin(capsys):
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        current=start(display);begin(display,current)
        call(display,'/action',{'action':'stop','control_protocol':1,'stop_origin':1,
            'control':{**current,'epoch':current['epoch']-1}},client='other-display')
        assert not display.active
        lines=[line for line in capsys.readouterr().err.splitlines() if line.startswith('CONTROL_STOP_ACCEPTED ')]
        assert len(lines)==1
        diagnostic=json.loads(lines[0].split(' ',1)[1])
        assert diagnostic['stop_origin']==1 and diagnostic['route']==1
        assert diagnostic['owner_matches'] is False and diagnostic['control_matches'] is False
        assert diagnostic['epoch']==current['epoch']
        assert display.token not in lines[0] and 'other-display' not in lines[0]
        assert call(display,'/operator-state')['last_stop']==diagnostic


def test_camera_end_records_its_own_origin_without_reclassifying_it_as_button_stop(capsys):
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        current=start(display);begin(display,current)
        call(display,'/camera/stop',{'session':'camera-session','stop_origin':7,'control':current})
        assert not display.active
        lines=[line for line in capsys.readouterr().err.splitlines() if line.startswith('CONTROL_STOP_ACCEPTED ')]
        assert len(lines)==1
        diagnostic=json.loads(lines[0].split(' ',1)[1])
        assert diagnostic['stop_origin']==7 and diagnostic['route']==2
        assert diagnostic['owner_matches'] and diagnostic['control_matches']
