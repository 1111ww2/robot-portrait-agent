import base64
from concurrent.futures import Future
from types import SimpleNamespace
import json

import numpy as np
import pytest


def proposal():
    return dict(subject_visible=True,confidence=.92,style='environmental',
                frame=[.12,.04,.85,.94],target=[.62,.30,.24],
                explanation='Keep the window as background with space on the left.')


def pose(x=.62,y=.30,width=.24):
    return SimpleNamespace(center_x=x,head_y=y,shoulder_width=width,confidence=.9)


def test_gemini_request_is_stateless_schema_constrained_and_image_bearing():
    from leica_tracker.gemini_framing import GeminiFramingClient
    requests=[]
    def transport(request,timeout):
        requests.append(request)
        return dict(status='completed',steps=[dict(type='model_output',content=[
            dict(type='text',text=json.dumps(proposal()))])])
    client=GeminiFramingClient(model='gemini-3.1-flash-lite',key_provider=lambda:'test-secret',transport=transport)
    result=client.analyze(b'jpeg-bytes',(.48,.3,.24))
    body=json.loads(requests[0].data)
    assert body['store'] is False and body['model']=='gemini-3.1-flash-lite'
    assert body['response_format']['mime_type']=='application/json'
    assert body['input'][1]['data']==base64.b64encode(b'jpeg-bytes').decode()
    assert result.target==(.62,.3,.24)


@pytest.mark.parametrize('change',[
    {'target':[float('nan'),.3,.24]}, {'target':[1.2,.3,.24]},
    {'target':[True,.3,.24]}, {'frame':[.9,.1,.1,.9]},
    {'frame':[.4,.1,.5,.6]}, {'confidence':.2}, {'subject_visible':False},
    {'joint_angles':[1,2,3]}, {'style':'run_python'},
])
def test_untrusted_plan_cannot_create_motion(change):
    from leica_tracker.gemini_framing import validate_plan,FramingError
    value=proposal();value.update(change)
    with pytest.raises(FramingError):validate_plan(value)


def test_rest_auth_error_never_echoes_provider_message_or_key():
    import io,urllib.error
    from leica_tracker.gemini_framing import GeminiFramingClient,FramingError
    def transport(request,timeout):
        raise urllib.error.HTTPError(request.full_url,400,'Bad Request',{},io.BytesIO(
            b'{"error":{"message":"test-secret invalid","details":[{"reason":"API_KEY_INVALID"}]}}'))
    client=GeminiFramingClient(model='gemini-3.1-flash-lite',key_provider=lambda:'test-secret',transport=transport)
    with pytest.raises(FramingError) as error:client.analyze(b'jpeg',(.62,.3,.24))
    assert error.value.code=='AI_AUTH_ERROR'
    assert 'test-secret' not in str(error.value)


def test_model_refusal_or_incomplete_output_is_not_a_plan():
    from leica_tracker.gemini_framing import GeminiFramingClient,FramingError
    for response in ({'status':'in_progress'}, {'status':'completed','steps':[]}):
        c=GeminiFramingClient(key_provider=lambda:'test-secret',transport=lambda *a:response)
        with pytest.raises(FramingError):c.analyze(b'jpeg',(.62,.3,.24))


class DeferredWorker:
    def __init__(self):self.calls=[]
    def submit(self,jpeg,point):
        future=Future();self.calls.append((jpeg,point,future));return future


def controller():
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    worker=DeferredWorker()
    return GeminiGuidanceController(worker=worker,stability_ms=0),worker


def tick(c,t,observation=None):
    return c.update(observation or pose(),t,frame=np.zeros((72,128,3),dtype=np.uint8))


@pytest.mark.parametrize('width,confidence,accepted', [
    (.0399,.9,False),(.04,.9,True),(.05,.9,True),(.06,.9,True),(.05,.49,False),
])
def test_far_view_threshold_requires_both_width_and_confidence(width,confidence,accepted):
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    w=DeferredWorker()
    c=GeminiGuidanceController(worker=w,single_plan=True,interaction_mode='claude-guide',
                               min_subject_width=.04,min_confidence=.5)
    obs=pose(width=width);obs.confidence=confidence
    decision=tick(c,0,obs)
    assert (decision.phase!='TRACKING_LOST') is accepted
    assert len(w.calls)==int(accepted)
    if not accepted:assert decision.cue.value=='HOLD'


def test_far_view_is_opt_in_and_does_not_relax_default():
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    w=DeferredWorker();c=GeminiGuidanceController(worker=w,single_plan=True,interaction_mode='claude-guide')
    d=tick(c,0,pose(width=.05))
    assert d.phase=='TRACKING_LOST' and d.reason=='SUBJECT_TOO_SMALL' and not w.calls


def resolve(worker):
    from leica_tracker.gemini_framing import validate_plan
    worker.calls[-1][2].set_result(validate_plan(proposal()))


def test_waits_without_blocking_then_uses_dynamic_target_and_completes_locally():
    c,w=controller()
    d=tick(c,0)
    assert d.phase=='AI_ENTRY_ACK' and d.cue.value=='INVITE'
    assert len(w.calls)==1
    for t in range(100,1000,100):assert tick(c,t).cue.value=='WAIT'
    assert len(w.calls)==1
    resolve(w)
    decisions=[tick(c,t) for t in range(1000,4000,100)]
    assert all(d.target==(.62,.3,.24) for d in decisions)
    assert sum(d.cue.value=='ARRIVED' for d in decisions)==1
    assert all(d.interaction_mode=='gemini-guide' and not d.framing_verified for d in decisions)
    assert len(w.calls)==1


@pytest.mark.parametrize('interrupt',['reset','format'])
def test_late_model_reply_cannot_override_an_explicitly_changed_scene(interrupt):
    c,w=controller();tick(c,0)
    if interrupt=='reset':c.reset()
    else:c.update(pose(),100,frame=np.zeros((128,72,3),dtype=np.uint8))
    resolve(w)
    d=(tick(c,200) if interrupt=='reset' else
       c.update(pose(),200,frame=np.zeros((128,72,3),dtype=np.uint8)))
    assert d.cue.value in ('HOLD','INVITE','WAIT') and not d.composition_frame
    assert d.phase!='ARRIVED'


def test_auth_error_latches_without_request_storm_until_reset():
    from leica_tracker.gemini_framing import FramingError
    c,w=controller();tick(c,0)
    w.calls[0][2].set_exception(FramingError('AI_AUTH_ERROR'))
    assert tick(c,100).phase=='AI_AUTH_ERROR'
    for t in range(200,10000,100):assert tick(c,t).cue.value=='HOLD'
    assert len(w.calls)==1
    c.reset();tick(c,10000)
    assert len(w.calls)==2


def test_fixed_scene_does_not_expire_while_person_is_still_finding_position():
    c,w=controller();tick(c,0);resolve(w)
    for t in range(100,30500,100):d=tick(c,t,pose(x=.5))
    assert d.target==(.62,.3,.24) and d.composition_frame==pytest.approx((.12,.04,.85,.94))
    assert d.phase!='AI_THINKING' and len(w.calls)==1


def test_key_loader_only_reads_explicit_local_config_and_never_executes_it(tmp_path,monkeypatch):
    from leica_tracker.gemini_framing import load_api_key
    monkeypatch.setenv('GEMINI_API_KEY','old-invalid')
    p=tmp_path/'.env.local';p.write_text('GEMINI_API_KEY="local-test-secret"\n')
    assert load_api_key(p)=='local-test-secret'
    assert load_api_key(tmp_path/'absent')=='old-invalid'


def test_actual_http_client_contract_with_local_fixture():
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    import threading
    from leica_tracker.gemini_framing import GeminiFramingClient,_http_json
    received=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            received.append((self.headers.get('x-goog-api-key'),body))
            result={'status':'completed','steps':[{'type':'model_output','content':[
                {'type':'text','text':json.dumps(proposal())}]}]}
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
            self.wfile.write(json.dumps(result).encode())
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def transport(request,timeout):
        from urllib.request import Request
        local=Request(f'http://127.0.0.1:{server.server_port}/',data=request.data,headers=dict(request.header_items()))
        return _http_json(local,timeout)
    try:
        result=GeminiFramingClient(key_provider=lambda:'fixture-key',transport=transport).analyze(b'jpeg',(.48,.3,.24))
        assert result.frame==(.12,.04,.85,.94)
        assert received[0][0]=='fixture-key'
        assert received[0][1]['store'] is False
    finally:server.shutdown();server.server_close();thread.join(timeout=2)


def test_loss_of_confidence_never_sends_a_new_camera_image():
    c,w=controller();p=pose();p.confidence=.1
    assert tick(c,0,p).cue.value=='HOLD'
    assert len(w.calls)==0


def test_closed_controller_cannot_restart_a_request_or_accept_old_reply():
    c,w=controller();tick(c,0);c.close();resolve(w)
    d=tick(c,100)
    assert d.phase=='AI_STOPPED' and d.cue.value=='HOLD'
    assert len(w.calls)==1


def test_overloaded_provider_is_reported_as_retryable_busy_state():
    import io,urllib.error
    from leica_tracker.gemini_framing import GeminiFramingClient,FramingError
    def transport(request,timeout):
        raise urllib.error.HTTPError(request.full_url,503,'Unavailable',{},io.BytesIO(b'{}'))
    with pytest.raises(FramingError) as error:
        GeminiFramingClient(key_provider=lambda:'fixture',transport=transport).analyze(b'jpeg',(.62,.3,.24))
    assert error.value.code=='AI_SERVICE_BUSY'


def test_replanning_cannot_bypass_the_process_request_cap():
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    w=DeferredWorker();c=GeminiGuidanceController(worker=w,max_requests=1,stability_ms=0)
    tick(c,0);c.reset();resolve(w)
    assert tick(c,100).phase=='AI_BUDGET_REACHED'
    assert len(w.calls)==1


def test_model_input_is_a_small_jpeg_not_full_resolution_video():
    import cv2
    c,w=controller()
    c.update(pose(),0,frame=np.zeros((1080,1920,3),dtype=np.uint8))
    image=cv2.imdecode(np.frombuffer(w.calls[0][0],dtype=np.uint8),cv2.IMREAD_COLOR)
    assert image.shape[:2]==(432,768)


def test_director_plan_cannot_accept_same_standing_position_without_an_invitation():
    from leica_tracker.gemini_framing import GeminiFramingClient,FramingError
    response=dict(status='completed',steps=[dict(type='model_output',content=[
        dict(type='text',text=json.dumps(proposal()))])])
    client=GeminiFramingClient(key_provider=lambda:'fixture',transport=lambda *args:response)
    with pytest.raises(FramingError) as error:client.analyze(b'jpeg',(.62,.3,.24))
    assert error.value.code=='AI_NO_INTERACTION_TARGET'


def test_director_guides_a_visible_move_before_local_completion():
    c,w=controller();tick(c,0,pose(x=.5));resolve(w)
    before=[tick(c,t,pose(x=.5 if t<1400 else .54)) for t in range(100,3100,100)]
    assert any(d.cue.value=='MOVE_RIGHT' for d in before)
    assert not any(d.phase=='ARRIVED' for d in before)
    after=[tick(c,t,pose(x=.60)) for t in range(3100,5100,100)]
    assert sum(d.cue.value=='ARRIVED' for d in after)==1


def test_first_capture_is_immediate_but_unstable_retries_do_not_upload():
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    from leica_tracker.gemini_framing import FramingError
    w=DeferredWorker();c=GeminiGuidanceController(worker=w)
    assert tick(c,0).cue.value=='INVITE'
    assert len(w.calls)==1
    w.calls[0][2].set_exception(FramingError('AI_NETWORK_ERROR'))
    assert tick(c,100).phase=='AI_NETWORK_ERROR'
    for t in range(200,16001,100):tick(c,t,pose(x=.48 if t%200==0 else .60))
    assert len(w.calls)==1
    for t in range(16100,17201,100):tick(c,t,pose(x=.5))
    assert len(w.calls)==2


def test_flickering_detection_does_not_repeat_the_first_capture():
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    w=DeferredWorker();c=GeminiGuidanceController(worker=w)
    for t in range(0,4000,100):
        if t%300==200:c.update(None,t,frame=None)
        else:tick(c,t)
    assert len(w.calls)==1


def test_one_missed_pose_holds_motion_but_keeps_pending_model_reply():
    c,w=controller();tick(c,0)
    assert c.update(None,100,frame=None).cue.value=='HOLD'
    resolve(w)
    d=tick(c,200)
    assert d.composition_frame and d.cue.value=='WAIT'
    for t in range(300,901,100):d=tick(c,t)
    assert d.model_status=='AI_PLAN_LOCKED'
    assert len(w.calls)==1


def test_sustained_loss_keeps_the_fixed_scene_reply_without_another_upload():
    c,w=controller();tick(c,0)
    for t in range(100,1001,100):c.update(None,t,frame=None)
    resolve(w)
    d=tick(c,1100)
    assert d.cue.value=='INVITE' and d.composition_frame==pytest.approx((.12,.04,.85,.94))
    assert len(w.calls)==1


def test_last_allowed_request_is_not_reported_exhausted_while_in_flight():
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    w=DeferredWorker();c=GeminiGuidanceController(worker=w,max_requests=1)
    for t in range(0,900,100):tick(c,t)
    assert len(w.calls)==1
    assert tick(c,900).phase=='AI_THINKING'
    resolve(w)
    assert tick(c,1000).model_status=='AI_PLAN_LOCKED'


def test_tiny_partial_person_does_not_trigger_a_paid_model_request():
    c,w=controller()
    for t in range(0,2000,100):assert tick(c,t,pose(width=.04)).cue.value=='HOLD'
    assert len(w.calls)==0


def test_manual_reset_cannot_create_a_burst_of_model_requests():
    c,w=controller();tick(c,0);resolve(w);c.reset()
    for t in range(100,4901,100):tick(c,t)
    assert len(w.calls)==1
    tick(c,5000)
    assert len(w.calls)==2


def test_brief_tracking_loss_after_completion_does_not_replay_the_ending():
    c,w=controller();tick(c,0);resolve(w)
    before=[tick(c,t) for t in range(100,2200,100)]
    assert sum(d.cue.value=='ARRIVED' for d in before)==1
    assert c.update(None,2200,frame=None).cue.value=='HOLD'
    after=[tick(c,t) for t in range(2300,3000,100)]
    assert all(d.cue.value=='WAIT' for d in after)
    assert len(w.calls)==1


def test_verbose_explanation_is_shortened_without_discarding_valid_geometry():
    from leica_tracker.gemini_framing import validate_plan
    value=proposal();value['explanation']='Use the window and negative space. '*35
    result=validate_plan(value)
    assert result.target==(.62,.30,.24)
    assert result.frame==(.12,.04,.85,.94)
    assert 0<len(result.explanation)<=240
    assert result.explanation.startswith('Use the window')


def test_long_explanation_still_cannot_hide_invalid_motion_coordinates():
    from leica_tracker.gemini_framing import validate_plan,FramingError
    value=proposal();value['explanation']='A detailed rationale. '*100
    value['target']=[float('nan'),.3,.24]
    with pytest.raises(FramingError,match='AI_INVALID_COORDINATES'):validate_plan(value)


def test_long_model_note_reaches_local_guidance_without_a_second_api_request():
    from leica_tracker.gemini_framing import GeminiFramingClient
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    requests=[]
    def transport(request,timeout):
        requests.append(request)
        value=proposal();value['explanation']='Window background gives the portrait intentional negative space. '*20
        return dict(status='completed',steps=[dict(type='model_output',content=[
            dict(type='text',text=json.dumps(value))])])
    client=GeminiFramingClient(key_provider=lambda:'fixture',transport=transport)
    class Worker:
        def submit(self,jpeg,point):
            f=Future();f.set_result(client.analyze(jpeg,point));return f
    c=GeminiGuidanceController(worker=Worker(),stability_ms=0)
    assert tick(c,0,pose(x=.48)).cue.value=='INVITE'
    d=tick(c,100,pose(x=.48))
    assert d.cue.value=='WAIT' and d.composition_frame
    for t in range(200,1000,100):d=tick(c,t,pose(x=.48))
    assert d.model_status=='AI_PLAN_LOCKED' and len(d.composition_note)<=240
    assert len(requests)==1


def test_entry_immediately_acknowledges_and_captures_without_standing_wait():
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    w=DeferredWorker();c=GeminiGuidanceController(worker=w)
    d=tick(c,0,pose(x=.48))
    assert d.cue.value=='INVITE'
    assert len(w.calls)==1 and w.calls[0][1]==(.48,.3,.24)
    for t in range(100,1600,100):assert tick(c,t,pose(x=.48)).cue.value=='WAIT'
    assert len(w.calls)==1


def test_model_result_does_not_repeat_the_entry_invitation():
    c,w=controller();decisions=[tick(c,0,pose(x=.48))];resolve(w)
    decisions.extend(tick(c,t,pose(x=.48)) for t in range(100,2000,100))
    assert sum(d.cue.value=='INVITE' for d in decisions)==1
    assert any(d.cue.value=='MOVE_RIGHT' for d in decisions)


def test_screen_coupled_invitation_finishes_before_the_direction_gesture():
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    w=DeferredWorker();c=GeminiGuidanceController(worker=w,entry_ack_ms=3000)
    assert tick(c,0,pose(x=.48)).cue.value=='INVITE'
    resolve(w)
    for t in range(100,3000,100):
        assert tick(c,t,pose(x=.48)).cue.value=='WAIT'
    assert tick(c,3000,pose(x=.48)).cue.value=='MOVE_RIGHT'


def test_head_region_grows_near_and_shrinks_far_in_pixel_space():
    from leica_tracker.gemini_framing import validate_plan
    p=validate_plan(proposal())
    far=p.tolerances_for(.12,image_aspect=16/9)
    near=p.tolerances_for(.36,image_aspect=16/9)
    assert near[0]>far[0]*2 and near[1]>far[1]*2
    assert far[0]*1280==pytest.approx(far[1]*720)


def test_live_head_area_and_local_acceptance_use_current_person_scale():
    c,w=controller();tick(c,0);resolve(w)
    decisions=[tick(c,t) for t in range(100,1000,100)]
    for t in range(1000,2200,100):far=tick(c,t,pose(x=.67,width=.22))
    near=tick(c,2200,pose(x=.67,width=.28))
    assert near.target_tolerances[0]>far.target_tolerances[0]
    assert near.target_tolerances[1]>far.target_tolerances[1]
    assert far.phase=='GUIDING' or far.phase=='WAITING_FOR_RESPONSE'
    assert near.phase=='SETTLING'
    assert len(w.calls)==1


def test_large_distance_change_stops_old_directions_then_resumes_locally():
    c,w=controller();tick(c,0,pose(x=.48));resolve(w)
    for t in range(100,5001,100):tick(c,t,pose(x=.48))
    first=tick(c,5100,pose(x=.48,width=.36))
    assert first.cue.value=='HOLD' and first.phase=='AI_REFRAMING'
    for t in range(5200,6501,100):d=tick(c,t,pose(x=.48,width=.36))
    assert len(w.calls)==1
    assert d.target[:2]==(.62,.3)
    assert d.composition_frame is not None and d.phase!='AI_THINKING'
    assert d.cue.value!='HOLD'


def test_person_size_scales_the_picture_without_moving_the_chosen_head_location():
    from leica_tracker.gemini_framing import validate_plan
    template=validate_plan(proposal())
    far=template.for_shoulder_width(.12,reference_width=.24)
    assert far.frame==pytest.approx((.37,.17,.735,.62))
    assert far.target==pytest.approx((.62,.3,.12))
    assert template.frame==(.12,.04,.85,.94)
    near=template.for_shoulder_width(.36,reference_width=.24)
    assert near.frame==pytest.approx((.073125,.015625,.8715625,1.))
    assert near.target==pytest.approx((.62,.3,.2625))


@pytest.mark.parametrize('width,reference',[(float('nan'),.24),(.2,0),(-.2,.24),(float('inf'),.24)])
def test_invalid_person_size_cannot_generate_crop_geometry(width,reference):
    from leica_tracker.gemini_framing import validate_plan
    with pytest.raises(ValueError):
        validate_plan(proposal()).for_shoulder_width(width,reference_width=reference)


def test_reentering_the_fixed_scene_reuses_template_with_new_person_size():
    c,w=controller();tick(c,0,pose(x=.48));resolve(w)
    for t in range(100,1100,100):tick(c,t,pose(x=.48))
    for t in range(1100,2100,100):c.update(None,t)
    decisions=[tick(c,t,pose(x=.4,width=.12)) for t in range(2100,4200,100)]
    assert decisions[0].cue.value=='INVITE'
    assert decisions[0].initial_point==(.4,.3,.12)
    assert sum(d.cue.value=='INVITE' for d in decisions)==1
    assert any(d.cue.value=='MOVE_RIGHT' for d in decisions)
    assert decisions[-1].target==pytest.approx((.62,.3,.12))
    assert decisions[-1].composition_frame==pytest.approx((.37,.17,.735,.62))
    assert len(w.calls)==1


def test_camera_gap_holds_once_then_reuses_cached_background():
    c,w=controller();tick(c,0,pose(x=.48));resolve(w)
    for t in range(100,1100,100):tick(c,t,pose(x=.48))
    assert tick(c,1500,pose(x=.48)).cue.value=='HOLD'
    d=tick(c,1600,pose(x=.48))
    assert d.composition_frame is not None and d.target[:2]==(.62,.3)
    assert d.phase!='AI_THINKING' and len(w.calls)==1


def test_manual_reset_clears_cached_scene_but_retains_cloud_request_spacing():
    c,w=controller();tick(c,0,pose(x=.48));resolve(w)
    for t in range(100,1100,100):tick(c,t,pose(x=.48))
    c.reset()
    assert tick(c,1200,pose(x=.48)).composition_frame is None
    assert len(w.calls)==1
    for t in range(1300,5101,100):d=tick(c,t,pose(x=.48))
    assert d.composition_frame is None and len(w.calls)==2


def test_changed_camera_aspect_discards_cached_image_geometry():
    c,w=controller();tick(c,0,pose(x=.48));resolve(w)
    for t in range(100,1100,100):tick(c,t,pose(x=.48))
    d=c.update(pose(x=.48),1100,frame=np.zeros((128,72,3),dtype=np.uint8))
    assert d.cue.value=='HOLD' and d.composition_frame is None
    assert d.reason=='CAMERA_FORMAT_CHANGED'


def test_unfittable_near_person_gets_local_step_back_without_cloud_replanning():
    c,w=controller();tick(c,0);resolve(w)
    for t in range(100,1000,100):tick(c,t)
    decisions=[tick(c,t,pose(width=.80)) for t in range(1000,5000,100)]
    assert any(d.cue.value=='MOVE_FARTHER' for d in decisions)
    assert not any(d.cue.value=='ARRIVED' for d in decisions)
    assert all(0<=v<=1 for d in decisions for v in d.composition_frame)
    assert all(d.target[:2]==(.62,.3) for d in decisions)
    assert len(w.calls)==1


def test_local_scale_noise_does_not_reverse_the_lateral_request():
    c,w=controller();tick(c,0,pose(x=.48));resolve(w)
    decisions=[tick(c,t,pose(x=.48,width=(.239,.241)[(t//100)%2]))
               for t in range(100,10000,100)]
    assert any(d.cue.value=='MOVE_RIGHT' for d in decisions)
    assert not any(d.cue.value in ('MOVE_LEFT','MOVE_CLOSER','MOVE_FARTHER') for d in decisions)
    assert all(d.target[0]==.62 for d in decisions)
    assert len(w.calls)==1


def test_cached_scene_remains_usable_after_the_only_allowed_model_request():
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    w=DeferredWorker();c=GeminiGuidanceController(worker=w,max_requests=1)
    tick(c,0,pose(x=.48));resolve(w)
    for t in range(100,1100,100):tick(c,t,pose(x=.48))
    for t in range(1100,2100,100):c.update(None,t)
    decisions=[tick(c,t,pose(x=.48)) for t in range(2100,4200,100)]
    assert any(d.cue.value=='MOVE_RIGHT' for d in decisions)
    assert all(d.phase!='AI_BUDGET_REACHED' for d in decisions)
    assert len(w.calls)==1


@pytest.mark.parametrize('width',[.06,.09,.18,.24,.36,.7])
def test_scaled_crop_stays_inside_image_at_close_and_far_limits(width):
    from leica_tracker.gemini_framing import validate_plan
    value=proposal();value.update(frame=[.03,.01,.94,.98],target=[.63,.34,.28])
    plan=validate_plan(value).for_shoulder_width(width,reference_width=.28)
    l,t,r,b=plan.frame;x,y,shoulders=plan.target
    assert 0<=l<r<=1 and 0<=t<b<=1
    assert x-shoulders/2>=l and x+shoulders/2<=r
    assert .04<=shoulders<=.7


def test_model_result_is_cached_while_person_is_absent_and_only_moves_on_reentry():
    c,w=controller();tick(c,0,pose(x=.48))
    for t in range(100,1001,100):c.update(None,t)
    resolve(w)
    for t in range(1100,6001,100):
        d=c.update(None,t)
        assert d.cue.value=='HOLD' and not d.tracking
        assert d.composition_frame is None and d.model_status=='AI_SCENE_CACHED'
    decisions=[tick(c,t,pose(x=.48)) for t in range(6100,8501,100)]
    assert decisions[0].cue.value=='INVITE'
    assert any(d.cue.value=='MOVE_RIGHT' for d in decisions)
    assert len(w.calls)==1


@pytest.mark.parametrize('interrupt',['gap','movement'])
def test_person_motion_during_cloud_planning_does_not_invalidate_the_fixed_scene(interrupt):
    c,w=controller();tick(c,0,pose(x=.48))
    if interrupt=='gap':assert tick(c,600,pose(x=.85)).cue.value=='HOLD'
    resolve(w)
    decisions=[tick(c,t,pose(x=.85)) for t in range(700,2501,100)]
    assert any(d.cue.value=='MOVE_LEFT' for d in decisions)
    assert all(d.target[:2]==(.62,.3) for d in decisions if d.composition_frame)
    assert len(w.calls)==1


def test_model_error_is_visible_in_telemetry_even_while_tracking_is_lost():
    from leica_tracker.gemini_framing import FramingError
    c,w=controller();tick(c,0)
    w.calls[0][2].set_exception(FramingError('AI_RATE_LIMIT'))
    d=c.update(None,100)
    assert d.cue.value=='HOLD' and not d.tracking
    assert d.model_status=='AI_RATE_LIMIT'
    assert tick(c,200).phase=='AI_RATE_LIMIT'
    assert len(w.calls)==1


def test_a_reply_older_than_the_cloud_deadline_is_not_cached_as_a_scene():
    c,w=controller();tick(c,0)
    for t in range(100,16001,100):c.update(None,t)
    resolve(w)
    d=c.update(None,16100)
    assert d.cue.value=='HOLD' and d.model_status=='AI_STALE_RESULT'
    d=tick(c,16200)
    assert d.composition_frame is None and d.phase=='AI_STALE_RESULT'
    assert len(w.calls)==1


def test_visitor_restart_invites_again_without_throwing_away_the_scene():
    c,w=controller();tick(c,0,pose(x=.48));resolve(w)
    for t in range(100,1501,100):tick(c,t,pose(x=.48))
    c.restart_encounter()
    d=tick(c,1600,pose(x=.48))
    assert d.cue.value=='INVITE' and d.composition_frame is not None
    assert d.model_requests==1 and len(w.calls)==1


def test_walking_farther_does_not_require_crouching_to_the_old_head_height():
    c,w=controller();tick(c,0,pose(x=.48,y=.5,width=.3));resolve(w)
    decisions=[tick(c,t,pose(x=.48,y=.20,width=.12)) for t in range(100,4000,100)]
    assert any(d.cue.value=='MOVE_RIGHT' for d in decisions)
    assert not any(d.cue.value in ('MOVE_UP','MOVE_DOWN') for d in decisions)
    assert all(d.target[1]==pytest.approx(.20) for d in decisions if d.composition_frame)
    assert len(w.calls)==1


def test_full_image_template_does_not_force_a_person_back_to_the_initial_tiny_size():
    from leica_tracker.gemini_framing import validate_plan
    c,w=controller();tick(c,0,pose(x=.48,y=.5,width=.075))
    value=proposal();value.update(frame=[0,0,1,1],target=[.62,.5,.075])
    w.calls[0][2].set_result(validate_plan(value))
    for t in range(100,1000,100):tick(c,t,pose(x=.48,y=.5,width=.075))
    decisions=[tick(c,t,pose(x=.48,y=.25,width=.26)) for t in range(1000,6000,100)]
    assert any(d.cue.value=='MOVE_RIGHT' for d in decisions)
    assert not any(d.cue.value in ('MOVE_UP','MOVE_DOWN','MOVE_FARTHER','MOVE_CLOSER') for d in decisions)
    assert decisions[-1].target==pytest.approx((.62,.25,.26))
    assert all(0<=v<=1 for d in decisions for v in d.composition_frame)
    assert len(w.calls)==1


def test_off_target_gemini_guidance_repeats_without_silently_pausing_forever():
    c,w=controller();tick(c,0,pose(x=.48));resolve(w)
    decisions=[tick(c,t,pose(x=.48)) for t in range(100,10000,100)]
    assert sum(d.cue.value=='MOVE_RIGHT' for d in decisions)>=3
    assert not any(d.phase=='PAUSED_FOR_PARTICIPANT' for d in decisions)
    assert len(w.calls)==1


@pytest.mark.parametrize('head_y',[.08,.25,.75,.95])
@pytest.mark.parametrize('width',[.06,.26,.65,.8])
def test_local_portrait_fit_follows_head_height_and_stays_in_the_image(head_y,width):
    from leica_tracker.gemini_framing import validate_plan
    plan=validate_plan(proposal()).for_person((.48,head_y,width))
    l,t,r,b=plan.frame;x,y,w=plan.target
    assert 0<=l<r<=1 and 0<=t<b<=1
    assert x==.62 and y==pytest.approx(head_y)
    assert l-1e-9<=x-w/2 and x+w/2<=r+1e-9
    assert t<=y<=b
