"""重启采样回调沿用限定在官方仿真器内的采集路径。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from leica_tracker.ursim_profile import SimulationProfile, build_profile
from test_physical_runtime import data
from test_selected_ursim_binding import design, no_external_interfaces
from test_ursim_profile import snapshots


@pytest.mark.parametrize('model', ['ur5e', 'ur10e'])
def test_selected_simulator_forwards_sample_observer_to_fenced_provider(model):
    initial = build_profile(design(model), *snapshots(), robot_model=model, joint_speed=.7)
    observer = lambda *_: None
    calls = []
    def capture(target, **kwargs):
        assert target.host == '127.0.0.2' and target.cfg.robot_model == model
        calls.append(kwargs)
        return snapshots()
    bound = replace(initial, capture_provider=capture).bind_before_motion(
        reader=lambda: None, control={}, should_stop=lambda: False, sample_observer=observer)
    assert type(bound) is SimulationProfile and bound.cfg.robot_model == model
    assert calls[0]['sample_observer'] is observer


def test_legacy_simulator_refuses_to_silently_ignore_restart_observation():
    profile = build_profile(data(), *snapshots())
    with pytest.raises(RuntimeError, match='[Ll]egacy.*observer|[Ll]egacy.*observation'):
        profile.bind_before_motion(reader=lambda: None, control={}, should_stop=lambda: False,
            sample_observer=lambda *_: None)


def test_invalid_observer_cannot_reach_simulator_capture_provider():
    profile = replace(build_profile(design(), *snapshots(), joint_speed=.7),
        capture_provider=lambda *a, **k: pytest.fail('Invalid hook reached provider'))
    with pytest.raises(ValueError, match='observer'):
        profile.bind_before_motion(reader=lambda: None, control={}, should_stop=lambda: False,
            sample_observer=True)


def test_system_capture_hook_preserves_docker_and_transport_fences(monkeypatch, tmp_path):
    from leica_tracker import operator_design_capture, ursim_system as system
    profile = build_profile(design(), *snapshots(), joint_speed=.7)
    identity = system.DockerIdentity('controller', 'gateway', 'network')
    checks = []
    def verify(*a, **k):
        checks.append(True)
        return identity
    monkeypatch.setattr(system, 'verify_docker', verify)
    monkeypatch.setattr(system, 'TailEventStream', lambda *a, **k: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(system.native, 'LatestEventSource', lambda *a: SimpleNamespace(
        thread=SimpleNamespace(join=lambda **k: None, is_alive=lambda: False)))
    factories = {key: (lambda *a, **k: None) for key in
        ('dashboard_check', 'state_factory', 'rtde_factory', 'dashboard_factory')}
    observed = []
    proofs = []
    observer = lambda *args: observed.append(args)
    proof_observer = lambda *args: proofs.append(args)
    def capture(target, *, dependencies, sample_observer=None, proof_observer=None, **kwargs):
        assert target.host == '127.0.0.2'
        for key in ('dashboard_check', 'state_factory', 'rtde_factory'):
            assert dependencies[key] is factories[key]
        assert sample_observer is observer
        assert proof_observer is not None
        dependencies['before_final_snapshot']()
        primary, native = snapshots()
        sample_observer(primary, native, 10.)
        proof_observer(primary, native, 10.1)
        return primary, native
    monkeypatch.setattr(operator_design_capture, 'capture_selected_reference', capture)
    lifecycle = {}
    def runner(actual, *args, **kwargs):
        assert kwargs['lifecycle'] is lifecycle and kwargs['allow_reference_rearm'] is True
        bound = actual.bind_before_motion(reader=lambda: None, control={},
            should_stop=lambda: False, sample_observer=observer, proof_observer=proof_observer)
        assert type(bound) is SimulationProfile
    monkeypatch.setattr(system.native, 'run_session', runner)
    def serve(actual, *args, **kwargs):
        callback = kwargs['session_runner']
        assert getattr(callback, 'supports_reference_restart', False) is True
        callback(actual, 8777, 'test', None, None, before={},
            lifecycle=lifecycle, allow_reference_rearm=True)
    monkeypatch.setattr(system.native, 'serve_sessions', serve)
    system.serve(profile, dict(port=8777, key='test'), tmp_path / 'unused.jsonl',
        SimpleNamespace(is_alive=lambda: True), identity, factories=factories)
    assert len(checks) == 3 and len(observed) == len(proofs) == 1
