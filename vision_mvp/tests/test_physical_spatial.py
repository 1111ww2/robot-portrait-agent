"""没有机器人连接：共享映射、运动限制和激活门。"""
import importlib
import itertools
import math

import pytest


def module():
    assert importlib.util.find_spec('leica_tracker.physical_spatial'), 'Missing physical XYZ follower'
    return importlib.import_module('leica_tracker.physical_spatial')


def profile():
    return dict(status='DRAFT', robot_ip='192.168.1.100', camera=1,
                neutral_tcp_m_rad=[-.4919, -.1333, .2093, 2.221441469, 2.221441469, 0.],
                neutral_joints_rad=[0., -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0.],
                tcp_offset_m_rad=[0., 0., .2786, 0., 0., 0.],
                rotation=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
                range_scale=1., speed_m_s=.18, acceleration_m_s2=.6,
                base_angle_deg=3., base_rate_deg_s=3.,
                review=dict(home=False, path=False, direction=False, tool_payload=False, dynamics=False))


def test_same_home_maps_every_target_exactly_like_simulator():
    from leica_tracker.simulator_spatial import input_target_3d
    cfg=module().SpatialConfig.from_dict(profile())
    for sample in itertools.product([-1.,-.3,0.,.7,1.], repeat=3):
        assert cfg.target(sample)==pytest.approx(input_target_3d(*sample),abs=1e-12)


def test_translation_reanchors_without_changing_direction_or_distance():
    m=module();p=profile();p['neutral_tcp_m_rad'][:3]=[.1,-.5,.5]
    cfg=m.SpatialConfig.from_dict(p)
    original=m.SpatialConfig.from_dict(profile())
    for command in [(1,0,0),(0,-1,0),(0,0,1)]:
        a=cfg.target(command);b=original.target(command)
        assert [a[i]-cfg.neutral[i] for i in range(3)]==pytest.approx([b[i]-original.neutral[i] for i in range(3)])


@pytest.mark.parametrize('change', [dict(speed_m_s=math.nan),dict(range_scale=2),
    dict(rotation=[[1,0,0],[0,1,0],[0,0,-1]]),dict(base_angle_deg=20)])
def test_invalid_or_relaxed_profile_rejected(change):
    p=profile();p.update(change)
    with pytest.raises(ValueError):module().SpatialConfig.from_dict(p)


def test_draft_cannot_upload_even_when_execute_requested():
    cfg=module().SpatialConfig.from_dict(profile())
    with pytest.raises(ValueError,match='review'):cfg.require_execution_review()
    p=profile();p['status']='PHYSICAL_XYZ_REVIEWED';p['review']={k:True for k in p['review']}
    module().SpatialConfig.from_dict(p).require_execution_review()


def test_pose_change_never_automatically_moves_robot_back_to_home():
    cfg=module().SpatialConfig.from_dict(profile())
    pose=list(cfg.neutral);pose[0]+=.02
    with pytest.raises(ValueError,match='home'):cfg.check_start(pose,cfg.joints,cfg.tcp)
    cfg.check_start(cfg.neutral,cfg.joints,cfg.tcp)


def test_loss_ack_rearm_and_stale_data_cannot_enable_motion():
    m=module();cfg=m.SpatialConfig.from_dict(profile());p=m.SpatialPolicy(cfg,123)
    pose=cfg.neutral;fb=dict(session=123,state=2,sequence=0)
    packet=p.next(10.,pose,((1,0,0),10.,3,0),fb)
    assert packet[1]==0
    fb.update(state=0,sequence=packet[0])
    packet=p.next(10.03,pose,((1,0,0),10.03,4,0),fb)
    assert packet[1]==1
    fb.update(state=1,sequence=packet[0])
    packet=p.next(10.06,pose,(None,10.06,0,1),fb)
    assert packet[1]==0 and packet[3]==pytest.approx(pose[:3])
    fb.update(state=0,sequence=packet[0])
    assert p.next(10.09,pose,((1,0,0),9.5,3,1),fb)[1]==0


def test_communication_timeout_is_latched_after_handshake():
    m=module();cfg=m.SpatialConfig.from_dict(profile());p=m.SpatialPolicy(cfg,123)
    p.handshake_complete=True
    with pytest.raises(RuntimeError,match='timeout'):
        p.next(1.,cfg.neutral,((0,0,0),1.,3,0),dict(session=123,state=2,sequence=0))


def test_nominal_fk_independently_matches_teacher_home():
    m=module();cfg=m.SpatialConfig.from_dict(profile())
    assert m.nominal_fk(cfg.joints,cfg.tcp)[:3,3]==pytest.approx(cfg.neutral[:3],abs=1e-8)


def test_new_program_has_vector_watchdog_and_guarded_joint_prediction():
    m=module();cfg=m.SpatialConfig.from_dict(profile());s=m.resident_program(cfg,123)
    assert 'read_input_float_register(44)' in s
    assert 'speedl([v[0],v[1],v[2],0,0,0]' in s
    assert 'packet_age > 0.25' in s
    assert 'get_inverse_kin_has_solution(probe, q, tcp=offset)' in s
    assert 'is_within_safety_limits(probe)' in s
    assert 'is_within_safety_limits(probe,' not in s
    assert 'base_scale' in s and 'get_actual_joint_speeds()' in s
    assert 'movej(' not in s and 'movel(' not in s


def test_snapshot_mismatch_and_overspeed_are_rejected():
    m=module();cfg=m.SpatialConfig.from_dict(profile())
    fb=dict(tcp_pose=cfg.neutral,joints=cfg.joints,tcp_speed=[0.]*6)
    cfg.check_feedback(fb)
    fb['tcp_speed']=[.18,.18,0,0,0,0]
    with pytest.raises(ValueError,match='speed'):cfg.check_feedback(fb)


def test_velocity_leaves_room_for_watchdog_delay_and_braking():
    m=module();cfg=m.SpatialConfig.from_dict(profile())
    assert hasattr(m,'cartesian_velocity'), 'Missing braking-aware XYZ controller'
    low,high=cfg.bounds()
    p=list(cfg.neutral[:3]);p[0]=low[0]+.01
    v=m.cartesian_velocity(cfg,p,[low[0],p[1],p[2]])
    speed=abs(v[0]);distance=speed*.35+speed*speed/(2*cfg.acceleration)
    assert distance<=.007+1e-9
    assert v[0]<0 and math.hypot(*v)<=cfg.speed


def test_physical_cleanup_uses_the_profile_deceleration(monkeypatch):
    from leica_tracker.resident_follow import close_session
    sent=[]
    class Channel:
        def send(self,script):sent.append(script)
        def close(self):pass
    import inspect
    assert 'stop_acceleration' in inspect.signature(close_session).parameters
    close_session(True,'unused',Channel(),None,None,None,None,stop_acceleration=.6)
    assert sent==['stopl(0.6)']


def test_cli_draft_execute_and_offline_preview_never_open_a_socket(tmp_path,monkeypatch,capsys):
    import json
    import socket
    path=tmp_path/'draft.json';path.write_text(json.dumps(profile()),encoding='utf-8')
    def forbid(*args,**kwargs):raise AssertionError('Unexpected network access')
    monkeypatch.setattr(socket,'create_connection',forbid)
    assert module().main(['--profile',str(path)])==0
    assert json.loads(capsys.readouterr().out)['robot_connected'] is False
    with pytest.raises(ValueError,match='review'):
        module().main(['--profile',str(path),'--execute','--stop-file',str(tmp_path/'stop')])


def test_existing_stop_file_prevents_network_connection(tmp_path,monkeypatch):
    import json
    m=module();data=profile();data['status']='PHYSICAL_XYZ_REVIEWED'
    data['review']={k:True for k in data['review']}
    path=tmp_path/'approved.json';path.write_text(json.dumps(data),encoding='utf-8')
    stop=tmp_path/'stop';stop.touch()
    monkeypatch.setattr(m,'check_dashboard',lambda *a,**kw:pytest.fail('Unexpected connection'))
    with pytest.raises(ValueError,match='Stop file'):
        m.main(['--profile',str(path),'--execute','--stop-file',str(stop)])


def test_axis_angle_sign_at_pi_does_not_invent_a_tool_rotation():
    assert module().orientation_error([0,math.pi,0],[0,-math.pi,0])==pytest.approx(0,abs=1e-7)


def test_diagonal_targets_share_one_speed_budget_and_loss_does_not_move_home():
    m=module();cfg=m.SpatialConfig.from_dict(profile())
    for command in itertools.product([-1,0,1],repeat=3):
        v=m.cartesian_velocity(cfg,cfg.neutral[:3],cfg.target(command))
        assert math.hypot(*v)<=.18+1e-10
    p=m.SpatialPolicy(cfg,123)
    pose=list(cfg.neutral);pose[0]+=.03
    update=p.next(1.,pose,(None,1.,0,1),dict(session=123,state=0,sequence=0))
    assert update[1]==0 and update[3]==pose[:3]


def test_commissioning_scale_still_allows_small_depth_motion():
    m=module();data=profile();data.update(range_scale=.05,speed_m_s=.005,acceleration_m_s2=.01)
    cfg=m.SpatialConfig.from_dict(data)
    v=m.cartesian_velocity(cfg,cfg.neutral[:3],cfg.target((0,0,1)))
    assert v[1]>0, 'Fixed 3 mm braking margin must not swallow the small depth envelope'
