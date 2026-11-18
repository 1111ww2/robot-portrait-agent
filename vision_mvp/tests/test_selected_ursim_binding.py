"""在限定的模拟 URSim 传输通道上验证选定方案的绑定。"""
import copy
from dataclasses import replace
import json
import socket
import subprocess
from types import SimpleNamespace

import pytest

from leica_tracker.operator_design import prepare_selected_design
from leica_tracker.ursim_profile import SimulationProfile, build_profile
from test_physical_runtime import data
from test_ursim_profile import snapshots


@pytest.fixture(autouse=True)
def no_external_interfaces(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Selected URSim unit tests must not open a live interface')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)


def design(model='ur10e'):
    source = data()
    return prepare_selected_design(source, model, start_joints=source['installation']['joints'])


def test_original_reference_negative_fixture_preserves_the_exact_fenced_profile_type():
    profile=build_profile(design(),*snapshots(),robot_model='ur10e',joint_speed=.7)
    assert profile.reference_restart_policy=='capture_stationary_on_explicit_start'
    fixed=replace(profile,preserve_restart_reference=True)
    assert type(fixed) is SimulationProfile and fixed.host==profile.host
    assert fixed.reference_restart_policy is None and fixed.operator_design==profile.operator_design
    assert fixed.cfg is profile.cfg and fixed.paths is profile.paths


@pytest.mark.parametrize('model', ['ur5e', 'ur10e'])
def test_selected_capture_returns_exact_simulation_profile_and_current_geometry(model):
    initial = build_profile(design(model), *snapshots(), robot_model=model, joint_speed=.7)
    primary, feedback = snapshots()
    primary['pose'][0] += .025
    feedback['tcp_pose'][0] += .025
    feedback['joints'][0] += .1
    primary['tcp_offset'][2] = .2786
    calls = []
    def capture(target, **kwargs):
        assert target.host == '127.0.0.2' and target.cfg.robot_model == model
        calls.append('capture')
        return primary, feedback
    seen = []
    profile = replace(initial, capture_provider=capture, binding_observer=seen.append)
    bound = profile.bind_before_motion(reader=lambda:None,control={},should_stop=lambda:False)
    assert type(bound) is SimulationProfile and bound.simulation_only is True
    assert bound.host == '127.0.0.2' and bound.cfg.robot_model == model
    assert bound.cfg.joints == tuple(feedback['joints'])
    assert bound.cfg.neutral == tuple(feedback['tcp_pose'])
    assert bound.cfg.tcp == tuple(primary['tcp_offset'])
    assert bound.intro['provenance'] == 'SIMULATION_ONLY' and bound.intro['simulation_only'] is True
    assert bound.simulation_metadata['operator_reference_bound'] is True
    assert calls == ['capture'] and seen == [bound]


def test_legacy_simulation_profile_does_not_start_capture():
    profile = build_profile(data(), *snapshots())
    assert profile.bind_before_motion(reader=lambda:None,control={},should_stop=lambda:False) is profile


def test_selected_simulation_profile_has_no_physical_capture_fallback():
    profile = build_profile(design(), *snapshots(), joint_speed=.7)
    with pytest.raises(RuntimeError,match='fenced.*capture'):
        profile.bind_before_motion(reader=lambda:None,control={},should_stop=lambda:False)


def test_selected_source_model_must_match_simulator_identity():
    with pytest.raises(ValueError,match='model'):
        build_profile(design('ur5e'), *snapshots(), robot_model='ur10e', joint_speed=.7)


@pytest.mark.parametrize('model', ['ur5e','ur10e'])
def test_selected_nullable_base_excursion_is_preserved_in_simulation_and_fresh_binding(model):
    from leica_tracker.operator_design import selected_profile
    source=data()
    source['installation']['base_limit']=None
    selected=prepare_selected_design(source,model,start_joints=source['installation']['joints'])
    # 这些数值仅表示明确选定的策略，不构成模拟生成的审核结果。
    assert selected_profile(selected).cfg.base_limit is None
    profile=build_profile(selected,*snapshots(),robot_model=model,joint_speed=.7)
    assert type(profile) is SimulationProfile and profile.cfg.base_limit is None
    assert 'review' not in profile.intro and 'reviewed' not in profile.intro
    assert profile.simulation_metadata.get('operator_reference_bound') is not True
    seen=[]
    bound=replace(profile,capture_provider=lambda *a,**k:snapshots(),binding_observer=seen.append).bind_before_motion(
        reader=lambda:None,control={},should_stop=lambda:False)
    assert type(bound) is SimulationProfile and bound.host=='127.0.0.2'
    assert bound.cfg.base_limit is None and bound.cfg.base_rate==.7
    assert bound.simulation_metadata['operator_reference_bound'] is True and seen==[bound]
    from leica_tracker.physical_intro_program import IntroProgram
    assert IntroProgram(bound.intro,1).joint_speed==.7
    program=bound.cfg.program(1)
    assert program.startswith('def ') and 'None' not in program


def test_legacy_simulator_review_does_not_authorize_null_base_excursion():
    source=data();source['installation']['base_limit']=None
    with pytest.raises(ValueError,match='base_limit'):
        build_profile(source,*snapshots())


def test_binding_refuses_a_retargeted_host_before_provider():
    profile = build_profile(design(), *snapshots(), joint_speed=.7)
    profile = replace(profile,host='192.0.2.1',capture_provider=lambda *a,**k:pytest.fail('retargeted host'))
    with pytest.raises(ValueError,match='isolated'):
        profile.bind_before_motion(reader=lambda:None,control={},should_stop=lambda:False)


def _fake_serve(monkeypatch, tmp_path, *, changed_after_capture=False):
    from leica_tracker import ursim_system as system
    from leica_tracker import operator_design_capture as capture_module
    initial = build_profile(design(), *snapshots(), joint_speed=.7)
    identity = system.DockerIdentity('c','g','n')
    calls = []
    checks = [0]
    def verify(*a, **kw):
        checks[0] += 1
        if changed_after_capture and checks[0] >= 3:
            return replace(identity, controller_id='changed')
        return identity
    monkeypatch.setattr(system,'verify_docker',verify)
    monkeypatch.setattr(system,'TailEventStream',lambda *a,**k:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(system.native,'LatestEventSource',lambda *a:SimpleNamespace(
        thread=SimpleNamespace(join=lambda **kw:None,is_alive=lambda:False)))
    sentinels = {name:(lambda *a,**k:None) for name in
        ['dashboard_check','state_factory','rtde_factory','dashboard_factory']}
    def capture(target, *, dependencies, **kwargs):
        assert target.host == '127.0.0.2'
        assert all(dependencies[name] is sentinels[name] for name in
            ['dashboard_check','state_factory','rtde_factory'])
        calls.append('capture')
        dependencies['before_final_snapshot']()
        return snapshots()
    monkeypatch.setattr(capture_module,'capture_selected_reference',capture)
    def runner(actual, *a, **kwargs):
        bound = actual.bind_before_motion(reader=lambda:None,control={},should_stop=lambda:False)
        calls.append('after_binding')
        assert type(bound) is SimulationProfile and bound.simulation_metadata['operator_reference_bound'] is True
    monkeypatch.setattr(system.native,'run_session',runner)
    monkeypatch.setattr(system.native,'serve_sessions',lambda actual,*a,**kw:
        kw['session_runner'](actual,8777,'fake',None,None,before={}))
    result = system.serve(initial,dict(port=8777,key='fake'),tmp_path/'camera.jsonl',
        SimpleNamespace(is_alive=lambda:True),identity,factories=sentinels)
    return result,calls,checks[0]


def test_serve_injects_only_sim_capture_factories_and_checks_identity_around_capture(monkeypatch,tmp_path):
    _,calls,checks = _fake_serve(monkeypatch,tmp_path)
    assert calls == ['capture','after_binding'] and checks >= 3


def test_changed_docker_after_capture_never_returns_a_bound_motion_profile(monkeypatch,tmp_path):
    with pytest.raises(RuntimeError,match='identity changed'):
        _fake_serve(monkeypatch,tmp_path,changed_after_capture=True)


def test_docker_identity_check_runs_with_open_receivers_then_reference_is_refreshed(monkeypatch,tmp_path):
    from leica_tracker import ursim_system as system
    from test_operator_design_capture import setup, CONTROL
    capture_target, options, clock, objects = setup()
    capture_target.host = '127.0.0.2'
    initial = build_profile(design('ur5e'),*snapshots(),robot_model='ur5e',joint_speed=.7)
    identity = system.DockerIdentity('c','g','n','ur5e')
    events = []
    after_docker = []
    def verify(*args,**kwargs):
        assert not any(channel.closed for channel in objects['channels']), 'Docker check ran after capture receivers closed'
        events.append('verify')
        if len(events)==3:
            assert len(objects['channels'])==2
            clock.sleep(.4)  # 真实 Docker 检查的阻塞时间可能超过 150 毫秒。
            after_docker.append(clock.now)
        return identity
    monkeypatch.setattr(system,'verify_docker',verify)
    monkeypatch.setattr(system,'TailEventStream',lambda *a,**k:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(system.native,'LatestEventSource',lambda *a:SimpleNamespace(
        thread=SimpleNamespace(join=lambda **kw:None,is_alive=lambda:False)))
    def run(actual,*args,**kwargs):
        bound=actual.bind_before_motion(reader=options['reader'],control=CONTROL,should_stop=lambda:False)
        assert bound.simulation_metadata['operator_reference_bound'] is True
        assert after_docker and clock.now-after_docker[0]>=.6
        assert all(channel.closed for channel in objects['channels'])
        events.append('bound_before_motion')
    monkeypatch.setattr(system.native,'run_session',run)
    monkeypatch.setattr(system.native,'serve_sessions',lambda actual,*a,**kw:
        kw['session_runner'](actual,8777,'fake',None,None,before={}))
    system.serve(initial,dict(port=8777,key='fake'),tmp_path/'camera.jsonl',
        SimpleNamespace(is_alive=lambda:True),identity,
        factories={**options['dependencies'],'dashboard_factory':lambda *a:None})
    assert events==['verify','verify','verify','bound_before_motion']


def test_actual_bound_profile_and_programs_are_recorded_before_motion(tmp_path):
    from leica_tracker.ursim_validation import record_execution_profile
    from leica_tracker.physical_intro_program import IntroProgram
    initial = build_profile(design(), *snapshots(), joint_speed=.7)
    result = {}
    record_execution_profile(initial,tmp_path,result,robot_model='ur10e')
    original_hash = result['profile_sha256']
    primary,feedback = snapshots()
    primary['pose'][1] += .01;feedback['tcp_pose'][1] += .01
    bound = replace(initial,capture_provider=lambda *a,**k:(primary,feedback)).bind_before_motion(
        reader=lambda:None,control={},should_stop=lambda:False)
    record_execution_profile(bound,tmp_path,result,robot_model='ur10e')
    assert result['reference_pose'] == list(bound.cfg.neutral)
    assert result['profile_sha256'] != original_hash
    record = json.loads((tmp_path/'profile.json').read_text(encoding='utf-8'))
    assert record['installation']['reference'] == list(bound.cfg.neutral)
    assert (tmp_path/'opening.urscript').read_text(encoding='utf-8') == IntroProgram(bound.intro,1).program()
    assert (tmp_path/'guidance.urscript').read_text(encoding='utf-8') == bound.cfg.program(1)
    assert result['programs_changed_on_reference_binding'] is True
    assert result['program_hash_scope'] == 'inspection_session_1_from_bound_reference_before_motion'
    import hashlib
    assert hashlib.sha256((tmp_path/'pre-start-profile.json').read_bytes()).hexdigest() == original_hash
