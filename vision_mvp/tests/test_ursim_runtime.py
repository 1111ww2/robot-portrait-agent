"""注入仿真依赖不能改变原生调用者的行为，也不能打开外部设备。"""
from types import SimpleNamespace

import pytest

from leica_tracker import physical_runtime as runtime
from test_session_cancellation import session_fixture, forbid_external_io


def test_session_uses_injected_simulator_xyz_and_dashboard(monkeypatch):
    profile, control, source, stopped, calls, session = session_fixture(monkeypatch)
    def sim_open(*args, **kwargs):
        assert kwargs == {'require_camera': False}
        calls.append('sim_xyz')
        return session
    def sim_dashboard(*args):
        calls.append('sim_dashboard')
        return SimpleNamespace(close=lambda: calls.append('sim_dashboard_closed'))
    with pytest.raises(RuntimeError, match='service stop requested'):
        runtime.run_session(profile, 8777, 'fake', source, lambda: control,
            before=control, should_stop=lambda: stopped[0],
            dependencies={'xyz_opener': sim_open, 'dashboard_factory': sim_dashboard})
    assert 'sim_xyz' in calls and 'sim_dashboard' in calls
    assert 'session_open' not in calls
    assert 'session_close' in calls and 'sim_dashboard_closed' in calls


def test_session_uses_injected_simulator_opening(monkeypatch):
    profile, control, source, stopped, calls, session = session_fixture(monkeypatch)
    control['control_action'] = 'start'
    monkeypatch.setattr(runtime, 'wait_for_trigger', lambda *a, **k: {'event': 'physical_intro'})
    def opening(*args, **kwargs):
        calls.append('sim_opening')
        class Active:
            def step(self, event):
                stopped[0] = True
        return session, Active(), SimpleNamespace(close=lambda: calls.append('sim_dashboard_closed'))
    with pytest.raises(RuntimeError, match='service stop requested'):
        runtime.run_session(profile, 8777, 'fake', source, lambda: control,
            before=control, should_stop=lambda: stopped[0],
            dependencies={'intro_runner': opening})
    assert 'sim_opening' in calls and 'session_close' in calls
    assert 'session_open' not in calls
