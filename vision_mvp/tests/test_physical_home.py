"""初始化是在离线的情况下进行的测试；不使用物理端点。"""
import importlib
import json
import math
import pytest
from test_physical_spatial import profile


def module():
    assert importlib.util.find_spec('leica_tracker.physical_home'), 'Missing initialization workflow'
    return importlib.import_module('leica_tracker.physical_home')


def configuration():
    p=profile()
    p['initialization']=dict(on_connect=True,path_reviewed=True,
        start_joints_rad=[math.radians(v) for v in [-90,-60,-100,-60,90,0]],
        waypoints_rad=[p['neutral_joints_rad']],joint_speed_rad_s=.05,
        joint_acceleration_rad_s2=.05,timeout_s=180.)
    return p


def test_already_at_home_skips_movement_even_without_route_review():
    m=module();p=configuration();p['initialization']['path_reviewed']=False
    assert m.prepare_route(p,p['neutral_joints_rad']) is None


def test_reviewed_route_ends_at_home_and_preserves_known_start():
    m=module();p=configuration();route=m.prepare_route(p,p['initialization']['start_joints_rad'])
    assert route['waypoints_rad'][-1]==p['neutral_joints_rad']
    assert route['joint_speed_rad_s']==.05


def test_unknown_start_and_unreviewed_transition_cannot_move():
    m=module();p=configuration();q=list(p['initialization']['start_joints_rad']);q[0]+=.1
    with pytest.raises(ValueError,match='start'):m.prepare_route(p,q)
    p['initialization']['path_reviewed']=False
    with pytest.raises(ValueError,match='path'):m.prepare_route(p,p['initialization']['start_joints_rad'])


@pytest.mark.parametrize('change',[dict(joint_speed_rad_s=.5),dict(joint_acceleration_rad_s2=float('nan')),
    dict(timeout_s=-1),dict(waypoints_rad=[]),dict(waypoints_rad=[[0.]*6])])
def test_invalid_or_non_home_route_is_rejected(change):
    p=configuration();p['initialization'].update(change)
    with pytest.raises(ValueError):module().prepare_route(p,p['initialization']['start_joints_rad'])


def test_generated_program_waits_for_live_session_and_stops_on_timeout():
    m=module();p=configuration();route=m.prepare_route(p,p['initialization']['start_joints_rad'])
    script=m.home_program(route,p['tcp_offset_m_rad'],123)
    assert 'movej(' in script and 'r=0' in script
    assert 'read_input_integer_register(44)' in script
    assert 'write_output_integer_register(40, 123)' in script
    assert 'kill motion_thread' in script and 'stopj(0.2)' in script
    assert 'packet_age > 0.25' in script
    assert 'is_within_safety_limits(' in script


def test_arrival_requires_matching_joints_and_standstill():
    m=module();q=configuration()['neutral_joints_rad']
    assert m.at_rest_at_home(q,[0.]*6,q)
    assert not m.at_rest_at_home(q,[.02,0,0,0,0,0],q)
    moved=list(q);moved[1]+=.02
    assert not m.at_rest_at_home(moved,[0.]*6,q)


def test_speed_percent_is_a_downward_software_override(tmp_path,capsys):
    from leica_tracker.physical_spatial import main
    p=tmp_path/'profile.json';p.write_text(json.dumps(profile()),encoding='utf-8')
    assert main(['--profile',str(p),'--speed-percent','50'])==0
    assert json.loads(capsys.readouterr().out)['speed_m_s']==pytest.approx(.09)
    with pytest.raises(SystemExit):main(['--profile',str(p),'--speed-percent','101'])


@pytest.mark.parametrize('interrupt',[None,'stop_file','connection'])
def test_initialization_supervisor_waits_for_arrival_or_requests_joint_stop(monkeypatch,interrupt):
    from types import SimpleNamespace
    import re
    from leica_tracker.physical_spatial import SpatialConfig
    m=module();cfg=SpatialConfig.from_dict(configuration())
    class Plant:
        now=10.;phase=0;token=0;sequence=0;move_at=None;failure=None
        sent=[]
        def advance(self,dt):
            self.now+=dt
            if self.move_at is not None and self.now-self.move_at>.12:self.phase=102
            if interrupt=='connection' and self.move_at is not None and self.now-self.move_at>.04:
                self.failure='connection lost'
        def state(self):
            return dict(mode=dict(connected=True,enabled=True,power=True,mode=7,
                emergency=False,protective=False,paused=False,running=self.phase in (100,101)),
                tcp_offset=cfg.tcp,pose=cfg.neutral)
        def feedback(self):
            return dict(session=self.token,state=self.phase,sequence=self.sequence,
                tcp_pose=cfg.neutral,tcp_speed=[0.]*6,
                joints=cfg.joints if self.phase==102 else cfg.data['initialization']['start_joints_rad'])
    plant=Plant()
    class States:
        failure=None
        @property
        def latest(self):return plant.state(),plant.now
    class RTDE:
        @property
        def failure(self):return plant.failure
        @property
        def latest(self):return plant.feedback(),plant.now
        def send(self,seq,enable,session,target):
            assert enable==1 and session==plant.token
            plant.sequence=seq
            if plant.move_at is None:plant.move_at=plant.now;plant.phase=101
    class Dashboard:
        failure=None
        @property
        def checked_at(self):return plant.now
    class StopFile:
        def exists(self):
            return interrupt=='stop_file' and plant.move_at is not None and plant.now-plant.move_at>.04
    class Channel:
        def __init__(self,host):pass
        def send(self,script):
            plant.sent.append(script)
            if script.startswith('def'):
                plant.token=int(re.search(r'write_output_integer_register\(40, (\d+)\)',script).group(1))
                plant.phase=100
        def close(self):pass
    monkeypatch.setattr(m,'CommandChannel',Channel)
    monkeypatch.setattr(m,'check_dashboard',lambda *a,**k:None)
    call=lambda:m.initialize_home(cfg,States(),Dashboard(),RTDE(),StopFile(),
        clock=lambda:plant.now,sleep=plant.advance)
    if interrupt:
        with pytest.raises(RuntimeError):call()
        assert plant.sent[-1]=='stopj(0.2)'
    else:
        assert call()==cfg.neutral
        assert plant.now-plant.move_at>=.27
        assert not any(s.startswith('stop') for s in plant.sent)
