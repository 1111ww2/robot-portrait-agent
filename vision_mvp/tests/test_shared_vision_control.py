"""在内存中测试共享运动控制权策略，不连接机器人或网络。"""
from types import SimpleNamespace

import pytest

from leica_tracker import guest_display
from leica_tracker.ipad_camera import IPadCapture
from leica_tracker.physical_resident_bootstrap import open_resident
from test_guest_camera_atomic import post
from test_ipad_camera import jpeg
from test_physical_resident_bootstrap import setup as resident_setup


@pytest.fixture
def encounter(monkeypatch):
    now=[10.]
    monkeypatch.setattr(guest_display,'time',SimpleNamespace(monotonic=lambda:now[0]))
    monkeypatch.setattr(guest_display,'ThreadingHTTPServer',lambda *a:SimpleNamespace(
        server_address=('127.0.0.1',8767),serve_forever=lambda:None))
    class Camera:
        available=True
        def fresh(self):return self.available
        def stream_fresh(self):return self.available
        def connected(self):return self.available
    camera=Camera();display=guest_display.GuestDisplay(remote_capture=camera,physical_feedback=True)
    display._set_action_locked('start','owner',1)
    identity={'instance':display._control_instance,'epoch':display._control_epoch}
    registered=display._physical_feedback_update({'op':'register','mode':'intro',**identity})
    return display,camera,now,identity,{**identity,**registered}


@pytest.mark.parametrize('mode',['intro','guidance'])
def test_camera_gap_does_not_revoke_shared_motor_ownership_or_backend_heartbeat(encounter,mode):
    display,camera,now,identity,backend=encounter
    if mode=='guidance':
        display._physical_feedback_update({'op':'intro','status':'active','intro_id':10_001,**backend})
        display._physical_feedback_update({'op':'intro','status':'ready','intro_id':10_001,**backend})
    camera.available=False
    for sequence in range(1,11):
        now[0]+=.1
        assert display._control_heartbeat_locked({**identity,'sequence':sequence},'owner',now[0])['accepted']
        display._physical_feedback_update({'op':'heartbeat',**backend})
        assert display.physical_backend_available()
        assert display.control_snapshot()['camera_fresh'] is False
        assert not display._physical_revoked and display._physical_revocation is None


def test_intro_ready_metadata_does_not_recheck_camera_after_a_fresh_observation(encounter):
    display,camera,_now,_identity,backend=encounter
    display._physical_feedback_update({'op':'intro','status':'active','intro_id':10_001,**backend})
    camera.available=False
    display._physical_feedback_update({'op':'intro','status':'ready','intro_id':10_001,**backend})
    assert display.physical_intro_ready(10_001) and display.physical_backend_available()


def light_begin(backend,source_ns):
    return {'op':'begin',**backend,'action_id':1,'source_ns':source_ns,'cue':'MOVE_RIGHT',
            'reference_position':[0.,0.,0.],'direction':[1.,0.,0.],'planned_displacement':.1}


def test_delayed_light_metadata_keeps_original_admitted_source_without_rechecking_age(encounter):
    display,camera,now,identity,backend=encounter
    display._physical_feedback_update({'op':'intro','status':'active','intro_id':10_001,**backend})
    display._physical_feedback_update({'op':'intro','status':'ready','intro_id':10_001,**backend})
    source_ns=round(now[0]*1e9);camera.available=False
    # 监管器已接纳这个来源。提示灯的 HTTP 元数据可能晚到；期间真实的所有者和后端心跳可维持已接纳路径。
    for sequence in range(1,6):
        now[0]+=.1
        display._control_heartbeat_locked({**identity,'sequence':sequence},'owner',now[0])
        display._physical_feedback_update({'op':'heartbeat',**backend})
    display._physical_feedback_update(light_begin(backend,source_ns))
    assert display._physical_action['source_ns']==source_ns
    assert display.physical_backend_available() and not display._physical_revoked


@pytest.mark.parametrize('source_ns',[True,0,-1,10_000_000_001])
def test_light_metadata_still_requires_positive_nonfuture_original_source(encounter,source_ns):
    display,_camera,_now,_identity,backend=encounter
    display._physical_feedback_update({'op':'intro','status':'active','intro_id':10_001,**backend})
    display._physical_feedback_update({'op':'intro','status':'ready','intro_id':10_001,**backend})
    with pytest.raises(ValueError,match='identity'):
        display._physical_feedback_update(light_begin(backend,source_ns))


@pytest.mark.parametrize('loss',['owner','backend','stop'])
def test_control_loss_or_stop_still_revokes_during_a_camera_gap(encounter,loss):
    display,camera,now,_identity,_backend=encounter
    camera.available=False
    if loss=='owner':
        now[0]+=.312;display._physical_backend_at=now[0]-.01
    elif loss=='backend':
        now[0]+=.312;display._seen=now[0]-.01
    else:display._set_action_locked('stop','owner',1)
    assert not display.physical_backend_available()
    assert display._physical_revoked or loss=='stop'


def test_initial_backend_registration_still_requires_a_fresh_camera(encounter):
    display,camera,_now,_identity,_backend=encounter
    display._set_action_locked('restart','owner',1);camera.available=False
    with pytest.raises(ValueError,match='camera'):
        display._physical_feedback_update({'op':'register','instance':display._control_instance,
                                          'epoch':display._control_epoch})


@pytest.mark.parametrize('same_owner,control_fresh',[(True,True),(True,False),(False,True)])
def test_camera_renewal_preserves_only_same_active_owner_with_fresh_control(monkeypatch,same_owner,control_fresh):
    now=[10.]
    monkeypatch.setattr(guest_display,'time',SimpleNamespace(monotonic=lambda:now[0]))
    monkeypatch.setattr(guest_display,'ThreadingHTTPServer',lambda *a:SimpleNamespace(
        server_address=('127.0.0.1',8767),serve_forever=lambda:None))
    camera=IPadCapture(clock=lambda:now[0],read_timeout=0)
    old=camera.begin('owner');ticket=camera.ticket('owner',old);camera.accept('owner',old,ticket,jpeg())
    display=guest_display.GuestDisplay(remote_capture=camera,physical_feedback=True)
    display._set_action_locked('start','owner',1);before=display._control_epoch
    now[0]+=3.
    display._seen=now[0]-(.01 if control_fresh else .312)
    client='owner' if same_owner else 'new-page'
    status,result=post(display,'/camera/start',client,{})
    assert status==200 and result['session']!=old
    if same_owner and control_fresh:
        assert display._active and display._control_epoch==before
        assert display._owner=='owner' and display._control_action=='start'
    else:
        assert not display._active and display._control_epoch==before+1
    assert not camera.stream_fresh() and camera._last_received is None
    camera.release()


def test_explicit_camera_stop_cancels_and_same_owner_camera_restart_cannot_revive(encounter):
    display,_capture,now,_identity,_backend=encounter
    camera=IPadCapture(clock=lambda:now[0],read_timeout=0)
    display.remote_capture=camera;session=camera.begin('owner')
    before=display._control_epoch
    assert post(display,'/camera/stop','owner',{'session':session})[0]==200
    assert not display._active and display._control_epoch==before+1
    assert post(display,'/camera/start','owner',{})[0]==200
    assert not display._active and display._control_action=='stop'
    assert display._control_epoch==before+2 and not camera.stream_fresh()
    camera.release()


@pytest.mark.parametrize('required',[True,False])
def test_resident_watchdog_camera_policy_is_explicit_and_defaults_to_required(required):
    cfg,deps,_objects=resident_setup();calls=[];factory=deps['watch_factory']
    def watch(*args,**kwargs):calls.append(kwargs);return factory(*args,**kwargs)
    deps['watch_factory']=watch
    options={} if required else {'require_camera':False}
    session=open_resident('192.0.2.1',cfg,lambda:None,dependencies=deps,**options)
    try:assert calls[0]['require_camera'] is required
    finally:session.close()


def test_invalid_resident_camera_policy_is_rejected_before_devices():
    cfg,deps,objects=resident_setup()
    with pytest.raises(ValueError):
        open_resident('192.0.2.1',cfg,lambda:None,dependencies=deps,require_camera='false')
    assert objects=={}
