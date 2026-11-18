"""选定的参考绑定使用假通道，并排在任何开头者之前。"""
from types import SimpleNamespace

import pytest

from leica_tracker import physical_runtime as runtime
from test_session_cancellation import session_fixture, forbid_external_io


def test_selected_reference_binds_after_trigger_before_xyz(monkeypatch):
    profile, control, source, stopped, calls, session = session_fixture(monkeypatch)
    profile.cfg = SimpleNamespace(robot_model='ur5e')
    captured = SimpleNamespace(host=profile.host, cfg=SimpleNamespace(robot_model='ur5e'),
                               supervisor=profile.supervisor)
    def bind(*, reader, control, should_stop):
        calls.append('bind_reference')
        assert reader()['active'] is True
        assert control['epoch'] == 1 and should_stop() is False
        return captured
    profile.bind_before_motion = bind
    def opening(host, cfg, reader, **kwargs):
        assert host == captured.host and cfg is captured.cfg
        calls.append('bound_xyz_open')
        return session
    with pytest.raises(RuntimeError, match='service stop requested'):
        runtime.run_session(profile, 8777, 'fake', source, lambda: control,
            before=control, should_stop=lambda: stopped[0],
            dependencies={'xyz_opener': opening})
    assert calls.index('bind_reference') < calls.index('bound_xyz_open')
    assert 'session_close' in calls


def test_selected_binding_runs_before_opening_route(monkeypatch):
    profile, control, source, stopped, calls, session = session_fixture(monkeypatch)
    control['control_action'] = 'start'
    profile.cfg = SimpleNamespace(robot_model='ur10e')
    captured = SimpleNamespace(host=profile.host, cfg=SimpleNamespace(robot_model='ur10e'))
    monkeypatch.setattr(runtime, 'wait_for_trigger', lambda *a, **k: {'event':'physical_intro'})
    profile.bind_before_motion = lambda **kwargs: calls.append('bind_reference') or captured
    def opening(actual, *a, **kwargs):
        assert actual is captured
        calls.append('bound_intro_open')
        return session, SimpleNamespace(step=lambda event: stopped.__setitem__(0, True)), SimpleNamespace(close=lambda:None)
    with pytest.raises(RuntimeError, match='service stop requested'):
        runtime.run_session(profile, 8777, 'fake', source, lambda: control,
            before=control, should_stop=lambda: stopped[0], dependencies={'intro_runner':opening})
    assert calls.index('bind_reference') < calls.index('bound_intro_open')


def test_cancel_during_selected_capture_never_opens_motion(monkeypatch):
    profile, control, source, stopped, calls, _ = session_fixture(monkeypatch)
    profile.cfg = SimpleNamespace(robot_model='ur5e')
    def bind(*, reader, control, should_stop):
        calls.append('bind_reference')
        stopped[0] = True
        should_stop()
        pytest.fail('Capture must observe the service cancellation')
    profile.bind_before_motion = bind
    with pytest.raises(RuntimeError, match='service stop requested'):
        runtime.run_session(profile, 8777, 'fake', source, lambda:control,
            before=control, should_stop=lambda:stopped[0])
    assert 'bind_reference' in calls and 'session_open' not in calls
    assert 'heartbeat_close' in calls and 'publisher_close' in calls


@pytest.mark.parametrize('change', ['host','model'])
def test_capture_cannot_retarget_another_controller_or_model(monkeypatch, change):
    profile, control, source, stopped, calls, _ = session_fixture(monkeypatch)
    profile.cfg = SimpleNamespace(robot_model='ur5e')
    bound = SimpleNamespace(host='other' if change=='host' else profile.host,
        cfg=SimpleNamespace(robot_model='ur10e' if change=='model' else 'ur5e'))
    profile.bind_before_motion = lambda **kwargs:bound
    with pytest.raises(RuntimeError, match='bound reference identity'):
        runtime.run_session(profile, 8777, 'fake', source, lambda:control,
            before=control, should_stop=lambda:stopped[0])
    assert 'session_open' not in calls


def test_blocked_observation_gate_prevents_capture(monkeypatch):
    profile, control, source, stopped, calls, _ = session_fixture(monkeypatch)
    profile.bind_before_motion = lambda **kwargs:pytest.fail('Blocked gate cannot capture a new reference')
    monkeypatch.setattr(runtime,'register_backend',lambda *a,**k:SimpleNamespace(
        status=lambda _:dict(blocked=True),fault=lambda:None))
    with pytest.raises(RuntimeError, match='remains blocked'):
        runtime.run_session(profile,8777,'fake',source,lambda:control,before=control)
    assert 'session_open' not in calls


@pytest.mark.parametrize('renewal',['same','different_intro','stopped'])
def test_capture_renews_fresh_same_encounter_trigger_before_motion(monkeypatch,renewal):
    profile,control,source,stopped,calls,session=session_fixture(monkeypatch)
    control['control_action']='start';profile.cfg=SimpleNamespace(robot_model='ur10e')
    bound=SimpleNamespace(host=profile.host,cfg=SimpleNamespace(robot_model='ur10e'))
    old=dict(event='physical_intro',intro_id=123,source_monotonic_ns=2)
    fresh=dict(event='physical_intro',intro_id=456 if renewal=='different_intro' else 123,
               source_monotonic_ns=3)
    waits=[]
    def wait(*a,**kwargs):
        waits.append(kwargs);calls.append('trigger'+str(len(waits)))
        if len(waits)==2 and renewal=='stopped':stopped[0]=True
        return old if len(waits)==1 else fresh
    monkeypatch.setattr(runtime,'wait_for_trigger',wait)
    profile.bind_before_motion=lambda **kwargs:calls.append('capture') or bound
    def opening(actual,*a,**kwargs):
        assert actual is bound and kwargs['intro_event'] is fresh
        calls.append('motion_open')
        return session,SimpleNamespace(step=lambda event:stopped.__setitem__(0,True)),SimpleNamespace(close=lambda:None)
    expected='intro identity changed' if renewal=='different_intro' else 'service stop requested'
    with pytest.raises(RuntimeError,match=expected):
        runtime.run_session(profile,8777,'fake',source,lambda:control,before=control,
            should_stop=lambda:stopped[0],dependencies=dict(intro_runner=opening))
    assert len(waits)==2 and calls.index('trigger1')<calls.index('capture')<calls.index('trigger2')
    assert waits[1]['control']==waits[0]['control'] and waits[1]['expected_action']=='start'
    if renewal!='same':assert 'motion_open' not in calls
