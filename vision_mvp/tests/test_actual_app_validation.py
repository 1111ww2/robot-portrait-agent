"""用于真实应用场景重播的离线合同；没有提供商或机器人I/O。"""
import io
import json

import numpy as np
import pytest


def test_full_scene_replay_preserves_the_visible_background_without_blank_canvas(tmp_path):
    import cv2
    from leica_tracker.actual_app_validation import load_scene
    frame=np.full((120,180,3),(10,80,150),dtype=np.uint8)
    frame[:,0:20]=(200,50,30)
    source=tmp_path/'scene.png'
    assert cv2.imwrite(str(source),frame)
    scene=load_scene(source,full_scene=True)
    assert scene.shape==frame.shape and np.array_equal(scene,frame)
    assert np.all(scene[0,0]==(200,50,30))
    assert np.all(scene[-1,-1]==(10,80,150))


def test_already_good_scene_can_finish_without_forcing_a_native_direction():
    from leica_tracker.actual_app_validation import photo_admission
    summary=dict(max_model_requests=1,phases={'ARRIVED':2},direction_events=0)
    assert photo_admission({},summary)=='already_good_without_direction'
    for bad in (dict(summary,max_model_requests=0),dict(summary,phases={}),
                dict(summary,direction_events=1)):
        with pytest.raises(RuntimeError):photo_admission({},bad)
    with pytest.raises(RuntimeError):photo_admission({1:{'completed':False}},summary)
    assert photo_admission({1:{'completed':True}},summary)=='native_direction_completed'


def test_journal_counts_direction_events_separately_from_arrival():
    from leica_tracker.actual_app_validation import AppEventJournal
    journal=AppEventJournal(io.StringIO())
    for cue,phase in (('MOVE_CLOSER','GUIDING'),('WAIT','WAITING_FOR_RESPONSE'),('HOLD','ARRIVED')):
        journal.write(json.dumps(dict(event='person_guidance',cue=cue,phase=phase,model_requests=1))+'\n')
    assert journal.summary()['direction_events']==1


def test_live_scene_validation_uses_production_single_plan_portrait_client():
    from leica_tracker.actual_app_validation import validation_client
    def transport(*_):pytest.fail('construction must not call the provider')
    key=lambda:'offline-test-key'
    client=validation_client(transport,key)
    assert client.model=='claude-opus-5-5'
    assert client.single_plan is True and client.portrait_quality is True
    assert client.supports_portrait_context is True
    assert client.transport is transport and client.key_provider is key


def test_scene_cannot_react_until_an_actual_native_action_is_observed():
    from leica_tracker.actual_app_validation import SceneReplay
    scene=SceneReplay(np.zeros((640,960,3),np.uint8))
    observation=dict(source_ns=1,point=[.3,.3,.2],target=[.45,.32,.22])
    scene.advance(observation)
    assert scene.matrix==pytest.approx([1,0,0])
    scene.observe_native_action(dict(active=True,fresh=False,cue='MOVE_RIGHT',action_id=1))
    scene.advance(observation)
    assert scene.matrix==pytest.approx([1,0,0])
    scene.observe_native_action(dict(active=True,fresh=True,phase='moving',cue='MOVE_RIGHT',action_id=1,
        started=False,progress=.1))
    scene.advance(observation)
    assert scene.matrix==pytest.approx([1,0,0])
    scene.observe_native_action(dict(active=True,fresh=True,phase='moving',cue='MOVE_RIGHT',action_id=1,
        started=True,progress=.1))
    scene.advance(observation)
    assert scene.matrix!=[1,0,0]


def test_scene_solves_real_observation_target_affine_without_injecting_measurements():
    from leica_tracker.actual_app_validation import SceneReplay
    scene=SceneReplay(np.zeros((640,960,3),np.uint8))
    observation=dict(source_ns=1,point=[.3,.3,.2],target=[.45,.32,.22])
    scene.observe_native_action(dict(active=True,fresh=True,phase='moving',cue='MOVE_RIGHT',action_id=1,
        started=True,progress=.1))
    for _ in range(150):scene.advance(observation)
    scale,tx,ty=scene.matrix
    assert (scale*.3*960+tx)/960==pytest.approx(.45,abs=1e-6)
    assert (scale*.3*640+ty)/640==pytest.approx(.32,abs=1e-6)
    assert scale*.2==pytest.approx(.22,abs=1e-6)
    assert scene.frame().shape==(640,960,3)


def test_event_journal_accepts_only_actual_fresh_tracking_geometry():
    from leica_tracker.actual_app_validation import AppEventJournal
    sink=io.StringIO();journal=AppEventJournal(sink)
    event=dict(event='person_guidance',tracking=True,source_monotonic_ns=10,
        phase='GUIDING',cue='MOVE_RIGHT',model_requests=1,
        points=dict(current=dict(center_x=.3,head_y=.3,shoulder_width=.2),target=dict(center_x=.45,head_y=.32,shoulder_width=.22)))
    journal.write(json.dumps(event)+'\n')
    assert journal.observation()['source_ns']==10
    held=dict(event,tracking=False,source_monotonic_ns=20,points={'current':None,'target':None})
    journal.write(json.dumps(held)+'\n')
    assert journal.observation()['source_ns']==10
    journal.write(json.dumps(dict(event,source_monotonic_ns=5))+'\n')
    assert journal.observation()['source_ns']==10
    assert journal.summary()['max_model_requests']==1
    assert len(sink.getvalue().splitlines())==3


def test_default_cli_opens_no_validation_or_image_interfaces(monkeypatch,tmp_path,capsys):
    from leica_tracker import actual_app_validation as module
    monkeypatch.setattr(module,'execute_validation',lambda *a,**k:pytest.fail('offline native boundary'))
    monkeypatch.setattr(module,'check_input',lambda *a,**k:pytest.fail('offline perception boundary'))
    assert module.main([],root=tmp_path)==0
    assert 'offline' in capsys.readouterr().out.lower()


def test_photo_receipt_requires_decoded_download_matching_actual_share_bytes():
    import hashlib
    from leica_tracker.actual_app_validation import verify_photo_receipt
    image=b'actual submitted photo bytes';digest=hashlib.sha256(image).hexdigest()
    receipt=dict(success=True,qr_decoded=True,download_verified=True,
        capture_sha256=digest,download_sha256=digest)
    assert verify_photo_receipt(receipt,image)['sha256']==digest
    for key in ('qr_decoded','download_verified','capture_sha256','download_sha256'):
        bad=dict(receipt);bad.pop(key)
        with pytest.raises(ValueError):verify_photo_receipt(bad,image)
    with pytest.raises(ValueError):verify_photo_receipt(receipt,b'different server bytes')
    with pytest.raises(ValueError):verify_photo_receipt({'success':True},image)


@pytest.mark.parametrize('statuses',[(200,),(400,200)])
def test_camera_feeder_generates_each_new_image_only_after_its_ticket(monkeypatch,statuses):
    from types import SimpleNamespace
    import cv2
    from leica_tracker import actual_app_validation as module
    events=[];bodies=[];tickets=[];counter={'ticket':0,'image':0}
    def request(path,body):
        if path=='/camera/start':return {'session':'replay-session'}
        assert path=='/camera/ticket'
        counter['ticket']+=1;events.append(('ticket',counter['ticket']))
        return {'ticket':'ticket-'+str(counter['ticket'])}
    owner=SimpleNamespace(request=request,client='owned-replay',local=SimpleNamespace(conn=None),
        display=SimpleNamespace(origin='http://127.0.0.1:12345',token='unused-test-token'))
    class Scene:
        def advance(self,_):events.append(('advance',counter['image']+1))
        def frame(self):
            counter['image']+=1;events.append(('frame',counter['image']))
            return counter['image']
    def encode(_kind,frame,_settings):
        events.append(('encode',frame))
        return True,SimpleNamespace(tobytes=lambda:('fresh-image-'+str(frame)).encode())
    monkeypatch.setattr(cv2,'imencode',encode)
    feeder=module._CameraFeeder(owner,Scene(),SimpleNamespace(observation=lambda:None))
    class Connection:
        def __init__(self,*_,**__):self.index=0
        def request(self,method,path,body,headers):
            assert (method,path)==('POST','/camera/frame')
            bodies.append(body);tickets.append(headers['X-Camera-Ticket']);events.append(('post',len(bodies)))
        def getresponse(self):
            status=statuses[self.index];self.index+=1
            if self.index==len(statuses):feeder.closed.set()
            return SimpleNamespace(status=status,read=lambda _:b'{}')
        def close(self):pass
    monkeypatch.setattr(module.http.client,'HTTPConnection',Connection)
    feeder._run()
    assert feeder.failure is None and feeder.frames==1 and feeder.rejected==len(statuses)-1
    for i in range(1,len(statuses)+1):
        assert events.index(('ticket',i))<events.index(('advance',i))<events.index(('frame',i))
        assert events.index(('frame',i))<events.index(('encode',i))<events.index(('post',i))
    assert len(set(tickets))==len(statuses) and len(set(bodies))==len(statuses)


def test_provider_transport_reports_pending_request_before_recording_completion(monkeypatch):
    import threading
    from types import SimpleNamespace
    from leica_tracker import actual_app_validation as module,gemini_framing
    entered=threading.Event();release=threading.Event()
    def blocked_request(_request,_timeout):
        entered.set();assert release.wait(2)
        return {'stop_reason':'end_turn'}
    monkeypatch.setattr(gemini_framing,'_http_json',blocked_request)
    transport=module._RecordingTransport()
    request=SimpleNamespace(full_url='https://provider.invalid/messages',get_method=lambda:'POST')
    worker=threading.Thread(target=lambda:transport(request,1),daemon=True)
    try:
        worker.start();assert entered.wait(1)
        assert transport.pending_request_count()==1
        assert transport.snapshot()==[]
        with pytest.raises(RuntimeError,match='provider request'):transport.assert_retired()
    finally:
        release.set();worker.join(2)
    assert not worker.is_alive()
    assert transport.pending_request_count()==0
    transport.assert_retired()
    assert transport.snapshot()[0]['success'] is True


def test_application_close_reports_daemon_provider_worker_until_it_retires():
    import threading
    from types import SimpleNamespace
    from leica_tracker.actual_app_validation import _Application
    app=_Application(None,None,None,None,1)
    # 此离线生命周期测试不会启动任何应用程序、摄像头或提供商。
    app.thread=SimpleNamespace(is_alive=lambda:False)
    release=threading.Event()
    worker=threading.Thread(target=lambda:release.wait(2),daemon=True,name='ai-framing')
    try:
        worker.start()
        assert app.pending_provider_worker_count()==1
        with pytest.raises(RuntimeError,match='provider worker'):app.close()
    finally:
        release.set();worker.join(2)
    assert app.pending_provider_worker_count()==0
    app.close()
