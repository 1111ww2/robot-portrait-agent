"""使用真实处理函数和模拟 HTTP 套接字测试相机与控制请求交错到达的情况。"""
from collections import deque
from email.message import Message
import io
import json
import threading
from types import SimpleNamespace

import pytest

from leica_tracker.guest_display import GuestDisplay
from leica_tracker.ipad_camera import IPadCapture


def display_for(camera):
    display=object.__new__(GuestDisplay)
    display.remote_capture=camera;display._lock=threading.Lock()
    display._physical_feedback_enabled=False;display._closed=False
    display._control_epoch=0;display._physical_registered_epoch=None
    display._state={};display._actions=deque();display._messages=deque()
    display._dialogue={};display._active=False
    return display


def post(display,route,client,body):
    request=object.__new__(display._handler())
    raw=json.dumps(body).encode();headers=Message()
    for key,value in {'Host':'127.0.0.1:8777','Origin':'http://127.0.0.1:8777',
            'Content-Type':'application/json','Content-Length':str(len(raw)),
            'X-Guest-Client':client}.items():headers[key]=value
    request.path=route;request.headers=headers;request.rfile=io.BytesIO(raw)
    request.connection=SimpleNamespace(settimeout=lambda _:None)
    request.authorized=lambda:True;replies=[]
    request.reply=lambda status,data,*a,**kw:replies.append((status,data))
    request.do_POST()
    return replies[-1]


def test_old_stop_cannot_pause_a_new_encounter_after_session_validation():
    validated=threading.Event();release=threading.Event();new_finished=threading.Event()
    errors=[]
    class Capture(IPadCapture):
        def end(self,client,session):
            super().end(client,session)
            validated.set()
            if not release.wait(2):raise RuntimeError('test barrier timed out')
    camera=Capture();old_session=camera.begin('old-page');display=display_for(camera)
    def old_stop():
        try:assert post(display,'/camera/stop','old-page',{'session':old_session})[0]==200
        except Exception as exc:errors.append(exc)
    def new_start():
        try:
            assert post(display,'/camera/start','new-page',{})[0]==200
            with display._lock:display._set_action_locked('start','new-page',1)
            new_finished.set()
        except Exception as exc:errors.append(exc)
    old=threading.Thread(target=old_stop);new=threading.Thread(target=new_start)
    old.start()
    try:
        assert validated.wait(1)
        new.start()
        assert not new_finished.wait(.05), 'new start must wait for the complete old camera/control transaction'
    finally:
        release.set();old.join(2)
        if new.ident is not None:new.join(2)
        camera.release()
    assert not old.is_alive() and not new.is_alive() and errors==[]
    assert display._active is True and display._owner=='new-page'
    assert display._control_action=='start'


def test_camera_begin_and_pause_complete_before_a_concurrent_explicit_start():
    validated=threading.Event();release=threading.Event();started=threading.Event()
    armed=[False];errors=[]
    class Capture(IPadCapture):
        def begin(self,client):
            session=super().begin(client)
            if armed[0]:
                validated.set()
                if not release.wait(2):raise RuntimeError('test barrier timed out')
            return session
    camera=Capture();camera.begin('owner');display=display_for(camera);armed[0]=True
    def begin():
        try:assert post(display,'/camera/start','owner',{})[0]==200
        except Exception as exc:errors.append(exc)
    def start():
        with display._lock:display._set_action_locked('start','owner',1)
        started.set()
    opening=threading.Thread(target=begin);control=threading.Thread(target=start)
    opening.start()
    try:
        assert validated.wait(1)
        control.start()
        assert not started.wait(.05)
    finally:
        release.set();opening.join(2)
        if control.ident is not None:control.join(2)
        camera.release()
    assert errors==[] and not opening.is_alive() and not control.is_alive()
    assert display._active and display._control_action=='start'


@pytest.mark.parametrize('route', ['/camera/start','/camera/stop'])
def test_camera_transitions_hold_display_lock_before_capture_and_cancel_valid_encounter(route):
    calls=[];display=None
    class Capture(IPadCapture):
        def begin(self,client):
            if display is not None:
                assert display._lock.locked()
                calls.append('begin')
            return super().begin(client)
        def end(self,client,session):
            assert display._lock.locked()
            calls.append('end')
            return super().end(client,session)
    camera=Capture();session=camera.begin('owner');display=display_for(camera)
    with display._lock:display._set_action_locked('start','owner',1)
    before=display._control_epoch
    display.pause_camera=lambda:pytest.fail('locked route must not call the lock-taking public callback')
    assert post(display,route,'owner',{'session':session})[0]==200
    assert calls==['begin' if route.endswith('start') else 'end']
    assert not display._active and display._control_action=='stop'
    assert display._control_epoch==before+1
    camera.release()


def test_already_stale_old_stop_is_rejected_without_cancelling_current_encounter():
    camera=IPadCapture();old=camera.begin('old-page');camera.end('old-page',old)
    current=camera.begin('new-page');display=display_for(camera)
    with display._lock:display._set_action_locked('start','new-page',1)
    before=display._control_epoch
    assert post(display,'/camera/stop','old-page',{'session':old})[0]==400
    assert display._active and display._control_epoch==before and camera._session==current
    camera.release()
