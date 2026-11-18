import time,urllib.request,urllib.error
from pathlib import Path
import pytest
from leica_tracker.photo_share import PhotoShare

JPEG=b'\xff\xd8\xfftest-jpeg\xff\xd9'


def test_photo_download_displays_official_local_leica_mark_and_concept():
    with PhotoShare(host='127.0.0.1',address='127.0.0.1',ttl_seconds=3) as share:
        photo=share.publish(JPEG)
        page=photo['_url']
        with urllib.request.urlopen(page) as response:
            html=response.read().decode('utf-8')
            assert response.headers['Content-Security-Policy']=="default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; base-uri 'none'"
        assert 'The camera chooses where you stand. Who is framing whom?' in html
        token=page.rsplit('/',1)[-1]
        assert f'/logo/{token}' in html
        assert f'/image/{token}' in html
        with urllib.request.urlopen(page.replace('/p/','/logo/')) as response:
            assert response.headers['Content-Type']=='image/svg+xml'
            assert response.read()==Path('vision_mvp/src/leica_tracker/leica-logo.svg').read_bytes()
        with urllib.request.urlopen(page.replace('/p/','/image/')) as response:
            assert response.read()==JPEG
        share.clear()
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(page.replace('/p/','/logo/'))
        assert error.value.code==404

def test_photo_share_has_unique_short_lived_readonly_tokens_and_no_cache():
    with PhotoShare(host='127.0.0.1', address='127.0.0.1', ttl_seconds=.1) as share:
        first=share.publish(JPEG)
        assert first['qr_data_url'].startswith('data:image/png;base64,')
        page=first['_url']
        with urllib.request.urlopen(page) as r:
            assert r.status==200 and r.headers['Cache-Control']=='no-store'
            assert b'<img' in r.read()
        with urllib.request.urlopen(page.replace('/p/','/image/')) as r:assert r.read()==JPEG
        second=share.publish(JPEG+b'2')
        with pytest.raises(urllib.error.HTTPError) as e:urllib.request.urlopen(page)
        assert e.value.code==404
        time.sleep(.15)
        with pytest.raises(urllib.error.HTTPError) as e:urllib.request.urlopen(second['_url'])
        assert e.value.code==404
        assert share.empty

def test_photo_share_rejects_invalid_images_and_clear_removes_access():
    with PhotoShare(host='127.0.0.1',address='127.0.0.1') as share:
        with pytest.raises(ValueError):share.publish(b'not-jpeg')
        item=share.publish(JPEG);share.clear()
        with pytest.raises(urllib.error.HTTPError):urllib.request.urlopen(item['_url'])

def test_paired_owner_photo_upload_requires_ready_and_stop_invalidates(monkeypatch):
    import json,urllib.request
    from leica_tracker.guest_display import GuestDisplay
    class Capture:
        def fresh(self):return True
    class Feedback:
        def photo_ready(self,source):return source==42
        def close(self):pass
    with GuestDisplay(port=0,remote_capture=Capture()) as d:
        def post(path,body,content,client='test-ipad',token=True):
            h={'X-Guest-Client':client,'Origin':d.origin,'Content-Type':content}
            if token:h['X-Guest-Token']=d.token
            return urllib.request.urlopen(urllib.request.Request(d.origin+path,data=body,headers=h,method='POST'),timeout=3)
        with post('/action',b'{"action":"start"}','application/json'):pass
        with pytest.raises(urllib.error.HTTPError) as e:post('/photo',JPEG,'image/jpeg')
        assert e.value.code==409
        with d._lock:
            d._sim_feedback=Feedback();d._gesture=object();d._gesture_source_ns=42;d._updated=time.monotonic();d._photo_latched=True
            d._state={'phase':'ARRIVED','tracking':True};d._operator={'photo_current':(.5,.4,.3),'target':(.5,.4,.3),'tolerances':(.05,.05,.05)}
        with pytest.raises(urllib.error.HTTPError):post('/photo',JPEG,'image/jpeg',client='stranger')
        with post('/photo',JPEG,'image/jpeg') as response:
            result=json.load(response)
        assert result['qr_data_url'].startswith('data:image/png;base64,') and '_url' not in result
        assert d._photo_share and not d._photo_share.empty
        with post('/action',b'{"action":"stop"}','application/json'):pass
        assert d._photo_share.empty

def test_delayed_photo_upload_cannot_survive_stop(monkeypatch):
    import json,urllib.request,threading
    from leica_tracker.guest_display import GuestDisplay
    from leica_tracker.photo_share import PhotoShare
    class Capture:
        def fresh(self):return True
    class Feedback:
        def photo_ready(self,source):return source==42
        def close(self):pass
    entered=threading.Event();release=threading.Event();original=PhotoShare.publish
    def delayed(self,jpeg):
        entered.set();assert release.wait(3);return original(self,jpeg)
    monkeypatch.setattr(PhotoShare,'publish',delayed)
    with GuestDisplay(port=0,remote_capture=Capture()) as d:
        def post(path,body,content):
            h={'X-Guest-Client':'test-ipad','X-Guest-Token':d.token,'Origin':d.origin,'Content-Type':content}
            return urllib.request.urlopen(urllib.request.Request(d.origin+path,data=body,headers=h,method='POST'),timeout=5)
        with post('/action',b'{"action":"start"}','application/json'):pass
        with d._lock:
            d._sim_feedback=Feedback();d._gesture=object();d._gesture_source_ns=42;d._updated=time.monotonic();d._photo_latched=True
            d._state={'phase':'ARRIVED','tracking':True};d._operator={'photo_current':(.5,.4,.3),'target':(.5,.4,.3),'tolerances':(.05,.05,.05)}
        result=[]
        def upload():
            try:
                with post('/photo',JPEG,'image/jpeg') as r:result.append(r.status)
            except urllib.error.HTTPError as e:result.append(e.code)
        def stop():
            with post('/action',b'{"action":"stop"}','application/json'):pass
        t=threading.Thread(target=upload);t.start();assert entered.wait(2)
        u=threading.Thread(target=stop);u.start()
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            with d._lock:
                if not d._active:break
            time.sleep(.01)
        assert not d._active
        release.set();t.join(5);u.join(5)
        assert not t.is_alive() and not u.is_alive() and result==[409]
        assert d._photo_share.empty

def test_old_discard_never_removes_newer_photo():
    with PhotoShare(host='127.0.0.1',address='127.0.0.1') as share:
        old=share.publish(JPEG);new=share.publish(JPEG)
        share.discard(old['_url'].rsplit('/',1)[-1])
        assert not share.empty
        with urllib.request.urlopen(new['_url']) as response:assert response.status==200

def test_photo_latch_ignores_later_position_and_tracking_changes():
    import time
    from leica_tracker.guest_display import GuestDisplay
    class Capture:
        fresh_value=True
        def fresh(self):return self.fresh_value
    class Feedback:
        def photo_ready(self,source):return source==42
        def close(self):pass
    c=Capture()
    with GuestDisplay(port=0,remote_capture=c) as d:
        with d._lock:
            d._active=True;d._seen=time.monotonic();d._sim_feedback=Feedback();d._gesture=object();d._gesture_source_ns=42;d._updated=time.monotonic();d._photo_latched=True
            d._state={'phase':'ARRIVED','tracking':True};d._operator={'photo_current':(.5,.4,.3),'target':(.5,.4,.3),'tolerances':(.05,.05,.05)}
            assert d._photo_ready_locked()
            d._operator['photo_current']=(.9,.4,.3);assert d._photo_ready_locked()
            d._operator['photo_current']=(.5,.4,.3);c.fresh_value=False;assert d._photo_ready_locked()
            c.fresh_value=True;d._state['tracking']=False;assert d._photo_ready_locked()
            d._state['tracking']=True;d._gesture_source_ns=43;assert d._photo_ready_locked()
            d._active=False;assert not d._photo_ready_locked()


def test_target_latch_survives_tracking_loss_and_uploads_current_photo():
    import json,urllib.request
    from leica_tracker.guest_display import GuestDisplay
    from leica_tracker.person_guidance import GuidanceDecision,GuidanceCue
    class Capture:
        fresh_value=True
        def fresh(self):return self.fresh_value
        def connected(self):return True
        def diagnostics(self):return {}
    c=Capture()
    with GuestDisplay(port=0,remote_capture=c) as d:
        def post(path,body,content,client='test-ipad'):
            h={'X-Guest-Client':client,'X-Guest-Token':d.token,'Origin':d.origin,'Content-Type':content}
            return urllib.request.urlopen(urllib.request.Request(d.origin+path,data=body,headers=h,method='POST'),timeout=3)
        def state():
            req=urllib.request.Request(d.origin+'/state',headers={'X-Guest-Client':'test-ipad','X-Guest-Token':d.token})
            with urllib.request.urlopen(req,timeout=3) as r:return json.load(r)
        with post('/action',b'{"action":"start"}','application/json'):pass
        old=GuidanceDecision('ARRIVED',GuidanceCue.ARRIVED,(.5,.4,.3),None,(.5,.4,.3),True,'TARGET_STABLE')
        d.publish(None,old)
        assert state().get('photo_latched') is False
        target=GuidanceDecision('ARRIVED',GuidanceCue.HOLD,(.5,.4,.3),None,None,False,'TARGET_LATCHED')
        d.publish(None,target)
        assert state()['photo_latched'] is True
        c.fresh_value=False
        gone=GuidanceDecision('CAMERA_PAUSED',GuidanceCue.HOLD,(.5,.4,.3),None,None,False,'CAMERA_FRAME_MISSING')
        d.publish(None,gone)
        assert state()['photo_latched'] is True
        with post('/photo',JPEG,'image/jpeg') as r:assert json.load(r)['qr_data_url'].startswith('data:image/png;base64,')
        with pytest.raises(urllib.error.HTTPError):post('/photo',JPEG,'image/jpeg',client='stranger')
        c.fresh_value=True
        with post('/action',b'{"action":"restart"}','application/json'):pass
        assert state()['photo_latched'] is False
        assert d._photo_share.empty
        d.publish(None,target)
        assert state()['photo_latched'] is True
        with post('/action',b'{"action":"stop"}','application/json'):pass
        assert state()['photo_latched'] is False
        with pytest.raises(urllib.error.HTTPError):post('/photo',JPEG,'image/jpeg')


def test_target_latch_clears_on_restart_and_lease_expiry():
    import time
    from leica_tracker.guest_display import GuestDisplay
    from leica_tracker.person_guidance import GuidanceDecision,GuidanceCue
    d=GuestDisplay(port=0,lease_seconds=.01)
    try:
        target=GuidanceDecision('ARRIVED',GuidanceCue.HOLD,(.5,.4,.3),None,None,False,'TARGET_LATCHED')
        with d._lock:d._active=True;d._seen=time.monotonic()
        d.publish(None,target)
        with d._lock:
            assert d._photo_latched
            d._seen=time.monotonic()-1;d._expire();assert not d._photo_latched
        with d._lock:d._active=True;d._seen=time.monotonic()
        d.publish(None,target)
        d.pause_camera()
        with d._lock:assert not d._photo_latched
    finally:
        d._server.server_close()
