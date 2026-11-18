"""仅使用模拟状态快照，不连接仿真器、实体机器人、套接字或 API。"""
import copy
from dataclasses import FrozenInstanceError

import pytest

from leica_tracker.physical_intro_program import IntroProgram
from leica_tracker.physical_runtime import ReviewedProfile
from test_physical_runtime import data


def snapshots():
    pose = [.4, -.2, .6, 0., 0., 0.]
    primary = {'pose': list(pose), 'tcp_offset': [0.] * 6,
               'mode': {'connected': True, 'enabled': True, 'power': True, 'mode': 7,
                        'emergency': False, 'protective': False, 'paused': False, 'running': False}}
    feedback = {'tcp_pose': list(pose), 'joints': [.3, -1.2, -1.4, -.8, 1.4, .2],
                'tcp_speed': [0.] * 6, 'joint_speed': [0.] * 6}
    return primary, feedback


def test_native_runtime_profile_reanchors_motion_without_physical_acceptance():
    from leica_tracker.ursim_profile import SimulationProfile, build_profile
    source = data()
    source['installation'].update(payload_kg=.7, center_of_gravity_m=[0., 0., .12])
    primary, feedback = snapshots()
    profile = build_profile(source, primary, feedback)
    assert isinstance(profile, SimulationProfile)
    assert isinstance(profile, ReviewedProfile)
    assert profile.simulation_only is True and profile.provenance == 'SIMULATION_ONLY'
    assert profile.host == '127.0.0.2' and profile.cfg.robot_model == 'ur10e'
    assert profile.cfg.neutral == tuple(feedback['tcp_pose'])
    assert profile.cfg.joints == tuple(feedback['joints'])
    assert profile.cfg.tcp == (0.,) * 6  # 使用新的仿真 TCP 替换实机测试夹具中 0.2786 米的工具参数。
    assert profile.cfg.lower == pytest.approx([.38, -.22, .58])
    assert profile.cfg.upper == pytest.approx([.42, -.18, .62])
    assert profile.cfg.speed == .005 and profile.cfg.acceleration == .01
    assert profile.simulation_metadata['payload_kg'] == .7
    assert profile.simulation_metadata['center_of_gravity_m'] == (0., 0., .12)
    assert not hasattr(profile, 'intro_preview')
    assert not hasattr(profile, 'status')
    assert profile.intro['provenance'] == 'SIMULATION_ONLY'
    assert 'intro_preview' not in profile.intro and 'operator_decision' not in profile.intro
    opening = IntroProgram(profile.intro, 123)
    assert opening.joint_speed == .6 and opening.joint_acceleration == 1.2
    assert opening.max_tcp_linear_speed is None and opening.max_tcp_angular_speed is None
    assert len(opening.waypoints) == 14
    assert opening.waypoints[0] == pytest.approx([.3, -1.2, -1.4, -.8, 1.4, .22])
    assert opening.waypoints[12] == pytest.approx([.3, -1.2, -1.4, -.8, 1.4, .46])
    assert opening.waypoints[-1] == tuple(feedback['joints'])
    assert opening.reference_pose == profile.cfg.neutral and opening.tcp == profile.cfg.tcp
    # 两个原生程序编译器都必须接受该内存仿真方案。
    for program in (opening.program(), profile.cfg.program(123)):
        assert program.startswith('def ')
        assert 'set_tcp(' not in program and 'set_payload(' not in program


@pytest.mark.parametrize('cue, expected', [
    ('MOVE_LEFT', [.395, -.2, .6]), ('MOVE_RIGHT', [.405, -.2, .6]),
    ('MOVE_UP', [.4, -.2, .605]), ('MOVE_DOWN', [.4, -.2, .595]),
    ('MOVE_CLOSER', [.4, -.205, .6]), ('MOVE_FARTHER', [.4, -.195, .6]),
])
def test_all_six_paths_translate_design_offsets_to_fresh_simulated_reference(cue, expected):
    from leica_tracker.ursim_profile import build_profile
    profile = build_profile(data(), *snapshots())
    assert profile.paths[cue][0] == pytest.approx(expected)
    assert profile.reviewed_paths[cue][0] == pytest.approx(expected)
    assert profile.cfg.check_target(profile.paths[cue][0]) == pytest.approx(expected)


def test_source_and_snapshots_never_alias_generated_profile():
    from leica_tracker.ursim_profile import build_profile
    source = data()
    primary, feedback = snapshots()
    originals = copy.deepcopy((source, primary, feedback))
    profile = build_profile(source, primary, feedback)
    assert (source, primary, feedback) == originals
    source['intro']['waypoints'][0][-1] = 9.
    source['paths']['MOVE_LEFT'][0][0] = 9.
    source['installation']['reference'][0] = 9.
    primary['tcp_offset'][0] = 9.
    feedback['tcp_pose'][0] = 9.
    feedback['joints'][0] = 9.
    assert profile.cfg.neutral == (.4, -.2, .6, 0., 0., 0.)
    assert profile.intro['waypoints'][0][-1] == pytest.approx(.22)
    assert profile.paths['MOVE_LEFT'][0][0] == pytest.approx(.395)
    profile.paths['MOVE_LEFT'][0][0] = .396
    assert profile.reviewed_paths['MOVE_LEFT'][0][0] == pytest.approx(.395)
    for key, value in [('simulation_only', False), ('provenance', 'PHYSICAL'), ('host', '192.0.2.1')]:
        with pytest.raises((AttributeError, FrozenInstanceError)):
            setattr(profile, key, value)


@pytest.mark.parametrize('host', ['127.0.0.1', 'localhost', '192.168.1.100', '::1', None])
def test_only_dedicated_simulator_loopback_is_accepted(host):
    from leica_tracker.ursim_profile import build_profile
    with pytest.raises(ValueError, match='127.0.0.2'):
        build_profile(data(), *snapshots(), host=host)


@pytest.mark.parametrize('field,value', [('power', False), ('enabled', False), ('mode', 5),
                                        ('protective', True), ('running', True)])
def test_unready_primary_cannot_become_simulator_reference(field, value):
    from leica_tracker.ursim_profile import build_profile
    primary, feedback = snapshots()
    primary['mode'][field] = value
    with pytest.raises(ValueError, match='ready'):
        build_profile(data(), primary, feedback)


@pytest.mark.parametrize('bad_sample', ['linear_motion', 'angular_motion', 'joint_motion', 'pose_disagreement', 'nan'])
def test_moving_disagreeing_or_nonfinite_samples_are_rejected(bad_sample):
    from leica_tracker.ursim_profile import build_profile
    primary, feedback = snapshots()
    if bad_sample == 'linear_motion': feedback['tcp_speed'][0] = .001
    if bad_sample == 'angular_motion': feedback['tcp_speed'][4] = .003
    if bad_sample == 'joint_motion': feedback['joint_speed'][1] = .004
    if bad_sample == 'pose_disagreement': primary['pose'][0] += .001
    if bad_sample == 'nan': feedback['joints'][0] = float('nan')
    with pytest.raises(ValueError):
        build_profile(data(), primary, feedback)


@pytest.mark.parametrize('bad_design', ['missing_cue', 'bent_path', 'backtracking', 'outside_bounds',
                                      'thirteen_points', 'no_return', 'unsupported_model', 'bad_speed'])
def test_invalid_motion_design_cannot_generate_runtime_plan(bad_design):
    from leica_tracker.ursim_profile import build_profile
    source = data()
    kwargs = {}
    if bad_design == 'missing_cue': del source['paths']['MOVE_LEFT']
    if bad_design == 'bent_path': source['paths']['MOVE_LEFT'] = [[.098, -.5, .3], [.095, -.499, .3]]
    if bad_design == 'backtracking': source['paths']['MOVE_LEFT'] = [[.096, -.5, .3], [.098, -.5, .3]]
    if bad_design == 'outside_bounds': source['paths']['MOVE_LEFT'][0][0] = .05
    if bad_design == 'thirteen_points': source['intro']['waypoints'].pop(0)
    if bad_design == 'no_return': source['intro']['waypoints'][-1][-1] += .1
    if bad_design == 'unsupported_model': kwargs['robot_model'] = 'ur99'
    if bad_design == 'bad_speed': kwargs['joint_speed'] = True
    with pytest.raises(ValueError):
        build_profile(source, *snapshots(), **kwargs)


def test_unreviewed_design_is_only_motion_data_and_never_reuses_accepted_preview():
    from leica_tracker.ursim_profile import build_profile
    source = data()
    source['status'] = 'DRAFT'
    source['intro_preview'] = {'status': 'NOT_ACCEPTED', 'operator_decision': 'reject'}
    source['intro']['review'] = {}
    source['intro']['reviewed'] = {}
    source['installation']['review'] = {}
    profile = build_profile(source, *snapshots())
    assert profile.provenance == 'SIMULATION_ONLY'
    assert profile.intro['reviewed']['reference_pose'] == [.4, -.2, .6, 0., 0., 0.]
    assert profile.intro['reviewed']['tcp'] == [0.] * 6
    assert profile.intro['waypoints'][-1] == [.3, -1.2, -1.4, -.8, 1.4, .2]
