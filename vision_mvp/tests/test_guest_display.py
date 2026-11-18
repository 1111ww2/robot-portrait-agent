import json,time
import urllib.request,urllib.error
from dataclasses import replace
import numpy as np
import pytest
from leica_tracker.person_guidance import GuidanceDecision,GuidanceCue


def test_control_snapshot_reports_stop_even_after_action_consumed():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as display:
        before=display.control_snapshot()
        assert before['active'] is False
        with request(display,'/action','POST',{'action':'start'}):pass
        active=display.control_snapshot()
        assert active['instance']==before['instance'] and active['epoch']==before['epoch']+1
        assert active['active'] is True and active['control_action']=='start'
        assert active['control_started_ns']>=before['control_started_ns']
        assert active['camera_connected'] is False and active['camera_fresh'] is False
        assert display.take_action()=='start'
        with request(display,'/action','POST',{'action':'stop'}):pass
        assert display.take_action()=='stop'
        stopped=display.control_snapshot()
        assert stopped['instance']==before['instance'] and stopped['epoch']==active['epoch']+1
        assert stopped['active'] is False and stopped['control_action']=='stop'
        assert stopped['control_started_ns']>=active['control_started_ns']
        assert stopped['camera_connected'] is False and stopped['camera_fresh'] is False


def test_control_snapshot_does_not_renew_visitor_lease():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as display:
        with request(display,'/action','POST',{'action':'start'}):pass
        first=display.control_snapshot()
        with display._lock:display._seen-=display._lease+1
        expired=display.control_snapshot()
        assert expired['active'] is False and expired['epoch']==first['epoch']+1
        assert display.take_action()=='stop'


def test_operator_action_remains_bound_to_epoch_after_consumption():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        with request(d,'/action','POST',{'action':'start'}):pass
        first=d.control_snapshot();assert first['control_action']=='start'
        assert d.take_action()=='start'
        with request(d,'/operator-state') as r:assert json.load(r)['control_action']=='start'
        with request(d,'/action','POST',{'action':'restart'}):pass
        second=d.control_snapshot();assert second['epoch']==first['epoch']+1 and second['control_action']=='restart'
        with request(d,'/action','POST',{'action':'stop'}):pass
        assert d.control_snapshot()['control_action']=='stop'


def test_control_snapshot_disables_control_on_display_close():
    from leica_tracker.guest_display import GuestDisplay
    display=GuestDisplay(port=0)
    with display:
        with request(display,'/action','POST',{'action':'start'}):pass
        assert display.control_snapshot()['active'] is True
    assert display.control_snapshot()['active'] is False


def test_control_snapshot_camera_health_uses_upload_not_published_frame():
    from leica_tracker.guest_display import GuestDisplay
    class Capture:
        def __init__(self):self.is_connected=True;self.is_fresh=True
        def connected(self):return self.is_connected
        def fresh(self):return self.is_fresh
        def diagnostics(self):return {}
    capture=Capture()
    with GuestDisplay(port=0,remote_capture=capture) as display:
        with request(display,'/action','POST',{'action':'start'}):pass
        snap=display.control_snapshot()
        assert snap['camera_fresh'] is True and snap['camera_connected'] is True
        with request(display,'/operator-state') as response:state=json.load(response)
        assert state['camera_fresh'] is True and state['stale'] is True
        capture.is_fresh=False
        assert display.control_snapshot()['camera_fresh'] is False
        capture.is_connected=False
        assert display.control_snapshot()['camera_connected'] is False


def test_watchdog_polls_actual_control_snapshot_and_latches_on_http_stop():
    from leica_tracker.guest_display import GuestDisplay
    from leica_tracker.resident_guest_watchdog import GuestControlWatchdog,ResidentMotionGate
    class Channel:
        def __init__(self):self.sent=[];self.closed=False
        def send(self,*args):self.sent.append(args)
        def close(self):self.closed=True
    class Capture:
        def connected(self):return True
        def fresh(self):return True
    with GuestDisplay(port=0,remote_capture=Capture()) as display:
        with request(display,'/action','POST',{'action':'start'}):pass
        channel=Channel();gate=ResidentMotionGate(channel)
        watchdog=GuestControlWatchdog(display.control_snapshot,gate,interval=.01,timeout=.2)
        try:
            watchdog.start()
            assert watchdog.wait_ready(1)
            gate.send(1,1,1,0.)
            with request(display,'/action','POST',{'action':'stop'}):pass
            assert watchdog.stopped.wait(1)
            assert gate.close_finished.wait(1) and channel.closed
            with pytest.raises(RuntimeError):gate.send(2,1,1,0.)
            assert len(channel.sent)==1
        finally:watchdog.close()


def test_physical_feedback_is_measured_fresh_and_latched_to_action_and_epoch():
    from leica_tracker.guest_display import GuestDisplay
    class Capture:
        def connected(self):return True
        def fresh(self):return True
        def stream_fresh(self):return True
        def diagnostics(self):return {}
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        with request(display,'/action','POST',{'action':'start'}):pass
        control=display.control_snapshot()
        identity={'instance':control['instance'],'epoch':control['epoch']}
        with request(display,'/physical-feedback','POST',{'op':'register',**identity}) as r:identity['lease']=json.load(r)['lease']
        with request(display,'/physical-feedback','POST',{'op':'heartbeat',**identity}):pass
        source=time.monotonic_ns()
        begin={'op':'begin',**identity,
               'source_ns':source,'action_id':1,'cue':'MOVE_LEFT',
               'reference_position':[0.,0.,0.],'direction':[-1.,0.,0.], 'planned_displacement':.1}
        with request(display,'/physical-feedback','POST',begin):pass
        sample={'op':'sample',**identity,'action_id':1,'source_ns':source,'cue':'MOVE_LEFT',
                'captured_at':time.monotonic(),'current_position':[-.05,0.,0.]}
        with request(display,'/physical-feedback','POST',sample):pass
        with request(display,'/state') as response:feedback=json.load(response)['feedback']
        assert feedback['edge']=='left' and feedback['intensity']==.5
        bad={**sample,'action_id':2}
        with pytest.raises(urllib.error.HTTPError):request(display,'/physical-feedback','POST',bad)
        with request(display,'/state') as response:assert json.load(response)['feedback']['edge'] is None
        with pytest.raises(urllib.error.HTTPError):request(display,'/physical-feedback','POST',begin)
        begin={**begin,'action_id':2,'source_ns':time.monotonic_ns()}
        with request(display,'/physical-feedback','POST',begin):pass
        sample={**sample,'action_id':2,'source_ns':begin['source_ns'],'captured_at':time.monotonic()}
        with request(display,'/physical-feedback','POST',sample):pass
        with request(display,'/action','POST',{'action':'stop'}):pass
        with request(display,'/state') as response:assert json.load(response)['feedback']['edge'] is None
        with pytest.raises(urllib.error.HTTPError):request(display,'/physical-feedback','POST',sample)


def test_physical_feedback_defaults_off_and_rejects_stale_sample():
    from leica_tracker.guest_display import GuestDisplay
    class Capture:
        def connected(self):return True
        def fresh(self):return True
        def stream_fresh(self):return True
        def diagnostics(self):return {}
    with GuestDisplay(port=0) as display:
        with pytest.raises(urllib.error.HTTPError):request(display,'/physical-feedback','POST',{'op':'begin'})
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as display:
        with request(display,'/action','POST',{'action':'start'}):pass
        control=display.control_snapshot();source=time.monotonic_ns()
        identity={'instance':control['instance'],'epoch':control['epoch']}
        with request(display,'/physical-feedback','POST',{'op':'register',**identity}) as r:identity['lease']=json.load(r)['lease']
        with request(display,'/physical-feedback','POST',{'op':'heartbeat',**identity}):pass
        begin={'op':'begin',**identity,
               'source_ns':source,'action_id':1,'cue':'MOVE_RIGHT',
               'reference_position':[0.,0.,0.],'direction':[1.,0.,0.],'planned_displacement':.1}
        with request(display,'/physical-feedback','POST',begin):pass
        sample={'op':'sample',**identity,'action_id':1,'source_ns':source,'cue':'MOVE_RIGHT',
                'captured_at':time.monotonic()-1.,'current_position':[.05,0.,0.]}
        with pytest.raises(urllib.error.HTTPError):request(display,'/physical-feedback','POST',sample)
        with request(display,'/state') as response:assert json.load(response)['feedback']['edge'] is None


def test_physical_backend_intro_and_photo_require_current_live_identity():
    from leica_tracker.guest_display import GuestDisplay
    class Capture:
        def connected(self):return True
        def fresh(self):return True
        def stream_fresh(self):return True
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as d:
        with request(d,'/action','POST',{'action':'start'}):pass
        state=d.control_snapshot();identity={'instance':state['instance'],'epoch':state['epoch']}
        intro=time.monotonic_ns()
        assert not d.physical_backend_available()
        with request(d,'/physical-feedback','POST',{'op':'register',**identity}) as r:identity['lease']=json.load(r)['lease']
        with request(d,'/physical-feedback','POST',{'op':'heartbeat',**identity}):pass
        assert d.physical_backend_available()
        with pytest.raises(urllib.error.HTTPError):
            request(d,'/physical-feedback','POST',{'op':'intro','status':'ready','intro_id':intro,**identity})
        with request(d,'/physical-feedback','POST',{'op':'intro','status':'active','intro_id':intro,**identity}):pass
        assert d.physical_intro_active(intro) and not d.physical_intro_ready(intro)
        with request(d,'/physical-feedback','POST',{'op':'intro','status':'ready','intro_id':intro,**identity}):pass
        assert d.physical_intro_ready(intro)
        frame=time.monotonic_ns()
        d.arm_physical_photo(frame)
        with pytest.raises(urllib.error.HTTPError):
            request(d,'/physical-feedback','POST',{'op':'photo_ready','source_ns':frame+1,**identity})
        assert not d.physical_photo_ready(frame)
        with request(d,'/physical-feedback','POST',{'op':'photo_ready','source_ns':frame,**identity}):pass
        assert d.physical_photo_ready(frame)
        with request(d,'/action','POST',{'action':'stop'}):pass
        assert not d.physical_intro_ready(intro) and not d.physical_photo_ready(frame)
        with pytest.raises(urllib.error.HTTPError):request(d,'/physical-feedback','POST',{'op':'heartbeat',**identity})


def test_physical_arrival_waits_for_matching_backend_stop_before_photo_latch():
    from leica_tracker.app import PhysicalPhotoLatch
    from leica_tracker.guest_display import GuestDisplay
    class Capture:
        def connected(self):return True
        def fresh(self):return True
        def stream_fresh(self):return True
        def diagnostics(self):return {}
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as d:
        with request(d,'/action','POST',{'action':'start'}):pass
        state=d.control_snapshot();identity={'instance':state['instance'],'epoch':state['epoch']}
        with request(d,'/physical-feedback','POST',{'op':'register',**identity}) as r:identity['lease']=json.load(r)['lease']
        with request(d,'/physical-feedback','POST',{'op':'heartbeat',**identity}):pass
        frame=time.monotonic_ns();latch=PhysicalPhotoLatch(d)
        arrived=replace(decision(),phase='ARRIVED',cue=GuidanceCue.ARRIVED)
        pending=latch.apply(arrived,active=True,source_ns=frame)
        assert (pending.phase,pending.cue,pending.reason)==('SETTLING',GuidanceCue.HOLD,'TARGET_PENDING_PHYSICAL_STOP')
        with request(d,'/operator-state') as r:assert json.load(r)['physical']['photo_source_ns']==frame
        assert latch.apply(decision(),active=True,source_ns=time.monotonic_ns()).phase=='SETTLING'
        with request(d,'/physical-feedback','POST',{'op':'photo_ready','source_ns':frame,**identity}):pass
        result=latch.apply(decision(),active=True,source_ns=time.monotonic_ns())
        assert (result.phase,result.cue,result.reason)==('ARRIVED',GuidanceCue.HOLD,'TARGET_LATCHED')


def test_operator_control_epoch_survives_action_consumption_and_changes_on_cancel():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        def control():
            with request(d,'/operator-state') as r:return json.load(r)['control']
        first=control()
        with request(d,'/action','POST',{'action':'stop'}):pass
        assert d.take_action()=='stop'
        stopped=control()
        assert stopped['instance']==first['instance'] and stopped['epoch']==first['epoch']+1
        d.pause_camera()
        assert control()['epoch']==stopped['epoch']+1
        with request(d,'/state') as r:assert json.load(r)['control']==control()


def test_operator_reference_status_exposes_current_session_only_to_local_operator():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        with request(d,'/reference-gate','POST',{'op':'begin','reference':[0.,0.,0.,0.,0.,0.]}) as r:
            session=json.load(r)['session']
        with request(d,'/operator-state') as r:gate=json.load(r)['reference_gate']
        assert {key:gate[key] for key in ('blocked','acknowledged','session')}=={
            'blocked':True,'acknowledged':False,'session':session}
        assert gate['reference_check']=={'acknowledged':False,'steady_ms':0.,'status':'no_sample'}
        with request(d,'/state') as r:assert 'reference_gate' not in json.load(r)
        with pytest.raises(urllib.error.HTTPError):request(d,'/operator-state',token=False)


def test_physical_reference_operations_reject_cancelled_control_epoch():
    from leica_tracker.guest_display import GuestDisplay
    class Capture:
        def connected(self):return True
        def fresh(self):return True
        def stream_fresh(self):return True
        def diagnostics(self):return {}
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as d:
        with request(d,'/action','POST',{'action':'start'}):pass
        control=d.control_snapshot()
        identity={'instance':control['instance'],'epoch':control['epoch']}
        with request(d,'/physical-feedback','POST',{'op':'register',**identity}) as r:identity['lease']=json.load(r)['lease']
        with pytest.raises(urllib.error.HTTPError):
            request(d,'/reference-gate','POST',{'op':'begin','reference':[0.]*6})
        with request(d,'/reference-gate','POST',{'op':'begin','reference':[0.]*6,**identity}) as r:
            session=json.load(r)['session']
        with request(d,'/action','POST',{'action':'stop'}):pass
        with pytest.raises(urllib.error.HTTPError):
            request(d,'/reference-gate','POST',{'op':'release','session':session,**identity})
        with request(d,'/operator-state') as r:assert json.load(r)['reference_gate']['blocked']


def test_revoked_or_expired_physical_lease_cannot_release_or_rearm():
    from leica_tracker.guest_display import GuestDisplay
    class Capture:
        def connected(self):return True
        def fresh(self):return True
        def stream_fresh(self):return True
        def diagnostics(self):return {}
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as d:
        with request(d,'/action','POST',{'action':'start'}):pass
        control=d.control_snapshot();identity={'instance':control['instance'],'epoch':control['epoch']}
        with request(d,'/physical-feedback','POST',{'op':'register',**identity}) as r:lease=json.load(r)['lease']
        identity['lease']=lease
        with request(d,'/reference-gate','POST',{'op':'begin','reference':[0.]*6,**identity}) as r:
            session=json.load(r)['session']
        frame=time.monotonic_ns();d.arm_physical_photo(frame)
        with request(d,'/physical-feedback','POST',{'op':'fault',**identity}):pass
        for path,body in [('/reference-gate',{'op':'release','session':session,**identity}),
                          ('/physical-feedback',{'op':'photo_ready','source_ns':frame,**identity}),
                          ('/physical-feedback',{'op':'heartbeat',**identity}),
                          ('/physical-feedback',{'op':'register','instance':identity['instance'],'epoch':identity['epoch']})]:
            with pytest.raises(urllib.error.HTTPError):request(d,path,'POST',body)
        assert not d.physical_photo_ready(frame)
        with request(d,'/operator-state') as r:assert json.load(r)['reference_gate']['blocked']


def test_expired_physical_lease_is_not_renewed_by_late_heartbeat():
    from leica_tracker.guest_display import GuestDisplay
    class Capture:
        def connected(self):return True
        def fresh(self):return True
        def stream_fresh(self):return True
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as d:
        with request(d,'/action','POST',{'action':'start'}):pass
        state=d.control_snapshot();identity={'instance':state['instance'],'epoch':state['epoch']}
        with request(d,'/physical-feedback','POST',{'op':'register',**identity}) as r:identity['lease']=json.load(r)['lease']
        with d._lock:d._physical_backend_at-=1.
        with pytest.raises(urllib.error.HTTPError):request(d,'/physical-feedback','POST',{'op':'heartbeat',**identity})
        with pytest.raises(urllib.error.HTTPError):
            request(d,'/physical-feedback','POST',{'op':'register','instance':state['instance'],'epoch':state['epoch']})
        assert not d.physical_backend_available()


def test_fault_revokes_lease_even_during_camera_drop_and_stays_revoked_after_reconnect():
    from leica_tracker.guest_display import GuestDisplay
    class Capture:
        connected_now=True;fresh_now=True
        def connected(self):return self.connected_now
        def fresh(self):return self.fresh_now
        def stream_fresh(self):return self.fresh_now
    capture=Capture()
    with GuestDisplay(port=0,remote_capture=capture,physical_feedback=True) as d:
        with request(d,'/action','POST',{'action':'start'}):pass
        state=d.control_snapshot();identity={'instance':state['instance'],'epoch':state['epoch']}
        with request(d,'/physical-feedback','POST',{'op':'register',**identity}) as r:identity['lease']=json.load(r)['lease']
        capture.fresh_now=False
        with request(d,'/physical-feedback','POST',{'op':'fault',**identity}):pass
        capture.fresh_now=True
        assert not d.physical_backend_available()
        with pytest.raises(urllib.error.HTTPError):request(d,'/physical-feedback','POST',{'op':'heartbeat',**identity})


def test_expired_visitor_lease_changes_operator_control_epoch():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        with request(d,'/action','POST',{'action':'start'}):pass
        with request(d,'/operator-state') as r:first=json.load(r)['control']
        with d._lock:d._seen-=d._lease+1
        with request(d,'/operator-state') as r:last=json.load(r)['control']
        assert last['epoch']==first['epoch']+1
def decision(cue=GuidanceCue.MOVE_RIGHT):
    return GuidanceDecision('GUIDING',cue,(.62,.3,.24),None,(.48,.3,.24),True,'OFF_TARGET_HORIZONTAL')


def test_participant_messages_are_paired_bounded_and_cleared_by_stop():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        with request(d,'/action','POST',{'action':'start'}):pass
        for kwargs in ({'token':False},{'origin':False}):
            with pytest.raises(urllib.error.HTTPError):request(d,'/message','POST',{'text':'Half-length portrait'},**kwargs)
        with request(d,'/message','POST',{'text':'Half-length portrait'}):pass
        assert d.take_messages()==['Half-length portrait']
        with pytest.raises(urllib.error.HTTPError):request(d,'/message','POST',{'text':'x'*301})
        with request(d,'/message','POST',{'text':'I do not want to move'}):pass
        d.set_dialogue({'speech':'Stay in your current position.','id':1})
        with request(d,'/action','POST',{'action':'stop'}):pass
        assert d.take_messages()==[]
        with request(d,'/state') as r:assert not json.load(r)['dialogue'].get('speech')
        with pytest.raises(urllib.error.HTTPError):request(d,'/message','POST',{'text':'Again'})


def request(display,path,method='GET',body=None,token=True,origin=True):
    if path=='/action' and isinstance(body,dict) and display.physical_feedback_enabled:
        control=display.control_snapshot()
        body={'control_protocol':1,'control':{k:control[k] for k in ('instance','epoch')},**body}
    headers={'X-Guest-Client':'test-ipad'}
    if token:headers['X-Guest-Token']=display.token
    if method=='POST':
        headers['Content-Type']='application/json'
        if origin:headers['Origin']=display.origin
    r=urllib.request.Request(display.origin+path,data=None if body is None else json.dumps(body).encode(),headers=headers,method=method)
    return urllib.request.urlopen(r,timeout=2)


def test_stream_and_controls_require_pairing_and_same_origin():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        with pytest.raises(urllib.error.HTTPError) as e:request(d,'/state',token=False)
        assert e.value.code==403
        with pytest.raises(urllib.error.HTTPError) as e:request(d,'/action','POST',{'action':'start'},origin=False)
        assert e.value.code==403
        with request(d,'/action','POST',{'action':'start'}) as response:assert json.load(response)['ok']
        assert d.active and d.take_action()=='start'
        with request(d,'/action','POST',{'action':'stop'}) as response:assert json.load(response)['ok']
        assert not d.active and d.take_action()=='stop'


def test_guest_stream_publishes_jpeg_without_exposing_provider_configuration():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        d.publish(np.zeros((72,128,3),dtype=np.uint8),decision())
        with request(d,'/frame.jpg') as response:
            assert response.headers['Content-Type']=='image/jpeg'
            assert response.read().startswith(b'\xff\xd8')
        with request(d,'/state') as response:state=json.load(response)
        assert state['direction']=='MOVE_RIGHT' and not state['active']
        assert not any(k in json.dumps(state).lower() for k in ('api_key','gemini_env','explanation'))


def test_disconnected_ipad_expires_the_motion_lease():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0,lease_seconds=.05) as d:
        with request(d,'/action','POST',{'action':'start'}):pass
        assert d.active
        time.sleep(.08)
        assert not d.active and d.take_action()=='stop'


def test_unrecognized_guest_actions_cannot_become_robot_commands():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        with pytest.raises(urllib.error.HTTPError) as e:request(d,'/action','POST',{'action':'movej'})
        assert e.value.code==400 and not d.active


def test_wait_keeps_the_last_direction_visible_but_tracking_loss_hides_it():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        frame=np.zeros((72,128,3),dtype=np.uint8)
        d.publish(frame,decision())
        d.publish(frame,replace(decision(),phase='PAUSED_FOR_PARTICIPANT',cue=GuidanceCue.WAIT))
        with request(d,'/state') as r:assert json.load(r)['direction']=='MOVE_RIGHT'
        d.publish(frame,replace(decision(),phase='TRACKING_LOST',cue=GuidanceCue.HOLD,tracking=False))
        with request(d,'/state') as r:assert json.load(r)['direction'] is None


def test_latest_frame_request_waits_for_a_new_sequence_instead_of_replaying_old_jpeg():
    from concurrent.futures import ThreadPoolExecutor
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        d.publish(np.zeros((72,128,3),dtype=np.uint8),decision())
        with request(d,'/frame.jpg') as r:
            seq=r.headers.get('X-Frame-Sequence');old=r.read()
        assert seq=='1'
        def next_frame():
            with request(d,'/frame.jpg?after=1') as r:return r.headers['X-Frame-Sequence'],r.read()
        with ThreadPoolExecutor(1) as pool:
            result=pool.submit(next_frame)
            time.sleep(.06)
            assert not result.done()
            d.publish(np.full((72,128,3),100,dtype=np.uint8),decision())
            seq,new=result.result(timeout=2)
        assert seq=='2' and new!=old


def test_video_publisher_can_exceed_ten_frames_a_second():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        for _ in range(5):
            d.publish(np.zeros((72,128,3),dtype=np.uint8),decision());time.sleep(.04)
        with request(d,'/state') as r:s=json.load(r)
        assert s['frame_sequence']>=4
        assert s['server_fps']>15


def test_thirty_fps_camera_is_not_accidentally_halved_by_the_encode_interval(monkeypatch):
    from types import SimpleNamespace
    import leica_tracker.guest_display as module
    clock=[100.]
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    with module.GuestDisplay(port=0) as display:
        for _ in range(7):
            display.publish(np.zeros((72,128,3),dtype=np.uint8),decision())
            clock[0]+=.03
        with request(display,'/state') as r:assert json.load(r)['frame_sequence']==7


def test_restarted_video_server_resets_an_old_browser_frame_cursor():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        d.publish(np.zeros((72,128,3),dtype=np.uint8),decision())
        with request(d,'/frame.jpg?after=999999') as r:
            assert r.status==200 and r.headers['X-Frame-Sequence']=='1'


def test_guest_picture_excludes_the_operator_head_point_and_target_region():
    import cv2
    from leica_tracker.guest_display import GuestDisplay
    d0=replace(decision(),composition_frame=(.1,.1,.9,.9))
    with GuestDisplay(port=0) as d:
        d.publish(np.zeros((72,128,3),dtype=np.uint8),d0)
        with request(d,'/frame.jpg') as r:image=cv2.imdecode(np.frombuffer(r.read(),np.uint8),cv2.IMREAD_COLOR)
        assert int(image.max())<=3


def test_wait_heartbeats_keep_the_original_screen_sweep_phase(monkeypatch):
    from types import SimpleNamespace
    import leica_tracker.guest_display as module
    from leica_tracker.robot_gestures import gesture_plan_for
    clock=[100.];monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    plan=gesture_plan_for(GuidanceCue.MOVE_RIGHT)
    with module.GuestDisplay(port=0) as d:
        values=[];identities=[]
        for elapsed in (600,1200,1800):
            clock[0]+=.1
            d.publish(np.zeros((72,128,3),np.uint8),replace(decision(),cue=GuidanceCue.WAIT),gesture=plan,gesture_elapsed_ms=elapsed)
            with request(d,'/frame.jpg') as r:
                feedback=json.loads(r.headers['X-Frame-Feedback']);r.read()
            values.append(feedback['intensity']);identities.append(feedback['gesture_id'])
        assert values==pytest.approx([1,0,1]) and len(set(identities))==1
        d.publish(np.zeros((72,128,3),np.uint8),replace(decision(),cue=GuidanceCue.HOLD,tracking=False),gesture=plan,gesture_elapsed_ms=1900)
        with request(d,'/state') as r:assert json.load(r)['feedback']['intensity']==0


def test_screen_phase_accounts_for_camera_processing_time(monkeypatch):
    from types import SimpleNamespace
    import leica_tracker.guest_display as module
    from leica_tracker.robot_gestures import gesture_plan_for
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:101.))
    with module.GuestDisplay(port=0) as d:
        d.publish(np.zeros((72,128,3),np.uint8),decision(),
            gesture=gesture_plan_for(GuidanceCue.MOVE_RIGHT),gesture_elapsed_ms=100,
            source_monotonic_ns=round(100.5*1e9))
        with request(d,'/state') as r:signal=json.load(r)['feedback']
        assert signal['edge']=='right' and signal['intensity']==pytest.approx(1.)


def test_isolated_voice_check_uses_pairing_and_excludes_camera_and_model_ui():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        with pytest.raises(urllib.error.HTTPError):urllib.request.urlopen(d.origin+'/?voicecheck=1')
        with urllib.request.urlopen(d.origin+'/?voicecheck=1&key='+d.token) as r:page=r.read().decode()
        assert 'Isolated microphone check' in page
        assert 'speechSynthesis' not in page
        assert 'GuestDialogue' not in page
        assert 'video:' not in page
        assert not d.active


def test_audio_transcription_is_paired_owner_only_and_does_not_submit_robot_commands():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        class Speech:
            ready=True
            def transcribe(self,data,language):
                assert data==b'audio';assert language=='en'
                return 'Hello, can you hear me?'
        d.speech=Speech()
        def audio(client='test-ipad',token=True):
            headers={'Origin':d.origin,'X-Guest-Client':client,'Content-Type':'application/octet-stream'}
            if token:headers['X-Guest-Token']=d.token
            return urllib.request.urlopen(urllib.request.Request(d.origin+'/transcribe',data=b'audio',headers=headers),timeout=2)
        with pytest.raises(urllib.error.HTTPError):audio()
        with request(d,'/action','POST',{'action':'start'}):pass
        with pytest.raises(urllib.error.HTTPError):audio(token=False)
        with pytest.raises(urllib.error.HTTPError):audio(client='different-ipad')
        with audio() as r:assert json.load(r)['text']=='Hello, can you hear me?'
        assert d.take_messages()==[]


def test_operator_view_is_paired_read_only_and_keeps_overlays_out_of_guest_state():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        d.publish(np.zeros((72,128,3),np.uint8),decision())
        with pytest.raises(urllib.error.HTTPError):urllib.request.urlopen(d.origin+'/operator')
        with urllib.request.urlopen(d.origin+'/operator?key='+d.token) as r:
            assert 'Operator view' in r.read().decode()
        with request(d,'/operator-state') as r:s=json.load(r)
        assert s['operator']['target']==list(decision().target)
        assert s['operator']['cue']==decision().cue.value
        assert s['operator']['coordinate_frame']=='mirrored_camera_image'
        assert s['operator']['physical_motion_verified'] is False
        with request(d,'/state') as r:assert 'operator' not in json.load(r)
        assert not d.active and d.take_action() is None


def test_operator_drops_old_geometry_when_camera_frame_is_missing():
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        d.publish(np.zeros((72,128,3),np.uint8),decision())
        d.publish(None,replace(decision(),phase='CAMERA_PAUSED',cue=GuidanceCue.HOLD,tracking=False))
        with request(d,'/operator-state') as r:s=json.load(r)['operator']
        assert s['current'] is None and s['target'] is None and s['cue']=='HOLD'


def test_simulated_motion_light_uses_fresh_matching_actual_pose(monkeypatch):
    import socket
    from leica_tracker.guest_display import GuestDisplay
    from leica_tracker.robot_gestures import gesture_plan_for
    from leica_tracker.simulator_feedback import MotionFeedbackSender
    s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind(('127.0.0.1',0));port=s.getsockname()[1];s.close()
    config=f'{port}:'+'a'*32
    monkeypatch.setenv('LEICA_SIM_FEEDBACK',config)
    with GuestDisplay(port=0) as d:
        with request(d,'/action','POST',{'action':'start'}):pass
        plan=gesture_plan_for(GuidanceCue.MOVE_RIGHT,profile='screen-coupled')
        source=time.monotonic_ns()
        d.publish(np.zeros((40,40,3),dtype=np.uint8),decision(),gesture=plan,source_monotonic_ns=source)
        with request(d,'/state') as r: assert json.load(r)['feedback']['intensity']==0
        sender=MotionFeedbackSender(config)
        try:
            sender.send(source_ns=source,cue='MOVE_RIGHT',hold=False,delta=[.06,0.,0.]);time.sleep(.05)
            with request(d,'/state') as r: assert json.load(r)['feedback']['intensity']==.5
            sender.send(source_ns=source,cue='MOVE_RIGHT',hold=True,delta=[.06,0.,0.]);time.sleep(.05)
            with request(d,'/state') as r: assert json.load(r)['feedback']['intensity']==0
        finally:sender.close()


def test_simulator_motion_active_uses_fresh_matching_bridge_feedback():
    from leica_tracker.guest_display import GuestDisplay
    class Feedback:
        def feedback(self,source,cue):
            return {'edge':'left','intensity':0.} if source==123 and cue=='MOVE_LEFT' else None
    d=GuestDisplay(port=0)
    try:
        assert d.simulator_motion_active(123,'MOVE_LEFT') is None
        d._sim_feedback=Feedback()
        assert d.simulator_motion_active(123,'MOVE_LEFT') is True
        assert d.simulator_motion_active(124,'MOVE_LEFT') is False
        assert d.simulator_motion_active(123,'MOVE_RIGHT') is False
    finally:
        d._sim_feedback=None;d._server.server_close()
