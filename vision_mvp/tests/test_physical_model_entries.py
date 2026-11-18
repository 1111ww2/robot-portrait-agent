"""模型路由仅使用合成记录，每个运行时边界都关闭。"""
import copy
import json
from pathlib import Path
import runpy
import socket

import pytest

from test_physical_delivery import data
from test_physical_launcher import launcher
from test_physical_system import module, workspace


def ur5e_data():
    value = data()
    value['robot_model'] = value['reviewed_robot_model'] = 'ur5e'
    for key in ('intro', 'installation'):
        value[key]['robot_model'] = value[key]['reviewed_robot_model'] = 'ur5e'
    value['reviewed_installation'] = copy.deepcopy(value['installation'])
    preview = value['intro_preview']
    preview['robot_model'] = preview['reviewed_robot_model'] = 'ur5e'
    return value


def write_ur5e_profile(root):
    path = root / 'ur5e-synthetic-reviewed.json'
    path.write_text(json.dumps(ur5e_data()), encoding='utf-8')
    return path


def test_ur5e_plan_binds_raw_model_even_when_injected_validator_returns_none(workspace):
    system = module()
    path = write_ur5e_profile(workspace)
    seen = []
    plan = system.prepare_plan(root=workspace, profile=path, robot_model='ur5e',
                               profile_validator=lambda value: seen.append(value))
    assert seen and plan.robot_model == 'ur5e'
    command = system.child_command(plan, workspace / 'stop.request')
    assert command[command.index('--robot-model') + 1] == 'ur5e'


@pytest.mark.parametrize('expected,explicit', [('ur5e', False), ('ur10e', True)])
def test_system_mismatched_model_rejects_before_review_or_runtime(workspace, expected, explicit):
    system = module()
    profile = write_ur5e_profile(workspace) if explicit else None
    with pytest.raises(ValueError, match='model'):
        system.prepare_plan(root=workspace, profile=profile, robot_model=expected,
            profile_validator=lambda _: pytest.fail('wrong model reached review validation'))


def test_legacy_system_plan_and_child_keep_ur10e(workspace):
    system = module()
    plan = system.prepare_plan(root=workspace, profile_validator=lambda _: None)
    assert plan.robot_model == 'ur10e'
    command = system.child_command(plan, workspace / 'stop.request')
    assert command[command.index('--robot-model') + 1] == 'ur10e'


@pytest.mark.parametrize('dedicated,override', [('ur5e', 'ur10e'), ('ur10e', 'ur5e')])
def test_dedicated_system_model_flag_cannot_change_identity(workspace, dedicated, override):
    with pytest.raises(SystemExit):
        module().main(['--robot-model', override], root=workspace, default_model=dedicated,
            profile_validator=lambda _: pytest.fail('flag override reached profile'))


def test_ur5e_system_accepts_only_a_synthetic_reviewed_profile_offline(workspace, monkeypatch, capsys):
    system = module()
    path = write_ur5e_profile(workspace)
    monkeypatch.setattr(socket, 'socket', lambda *a, **k: pytest.fail('offline opened socket'))
    monkeypatch.setattr(system.subprocess, 'Popen', lambda *a, **k: pytest.fail('offline opened process'))
    assert system.main(['--profile', str(path)], root=workspace, default_model='ur5e') == 0
    assert 'UR5e' in capsys.readouterr().out


def test_ur5e_default_draft_rejects_offline_before_pairing_or_launch(workspace, monkeypatch):
    system = module()
    relative = Path('vision_mvp/config/physical-ur5e.template.json')
    draft = workspace / relative
    draft.parent.mkdir(parents=True)
    draft.write_text(json.dumps({'status': 'DRAFT_UR5E_NOT_EXECUTABLE',
                                 'robot_model': 'ur5e', 'reviewed_robot_model': None}))
    monkeypatch.setattr(socket, 'socket', lambda *a, **k: pytest.fail('draft opened socket'))
    monkeypatch.setattr(system.subprocess, 'Popen', lambda *a, **k: pytest.fail('draft opened process'))
    monkeypatch.setattr(system, 'validated_pair', lambda *a: pytest.fail('draft reached pairing'))
    with pytest.raises(ValueError, match='review'):
        system.main([], root=workspace, default_model='ur5e', default_profile=relative)


def test_ur5e_entry_cannot_use_a_legacy_ur10e_profile_path(workspace):
    with pytest.raises(ValueError, match='model'):
        module().main(['--profile', str(workspace / 'tmp/robot-network/physical-ipad-ur10e-trial.json'),
                       '--robot-model', 'ur5e'], root=workspace, default_model='ur5e',
                      profile_validator=lambda _: pytest.fail('cross-model profile reached review'))


@pytest.mark.parametrize('script,model', [('run-ur5e-ipad-system.py', 'ur5e'),
                                         ('run-ur10e-ipad-system.py', 'ur10e')])
def test_dedicated_system_wrappers_bind_expected_identity(monkeypatch, script, model):
    system = module()
    calls = []
    monkeypatch.setattr(system, 'main', lambda **kwargs: calls.append(kwargs) or 0)
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(scripts / script), run_name='__main__')
    assert result.value.code == 0
    assert calls[0]['default_model'] == model
    assert calls[0]['default_profile'] == Path(f'vision_mvp/config/selected-{model}-0.7.json')


@pytest.mark.parametrize('expected,model', [('ur5e', 'ur10e'), ('ur10e', 'ur5e')])
def test_live_launcher_rejects_model_mismatch_before_any_port_or_process(workspace, monkeypatch, expected, model):
    launch = launcher()
    profile = workspace / 'synthetic.json'
    value = ur5e_data() if model == 'ur5e' else data()
    profile.write_text(json.dumps(value), encoding='utf-8')
    monkeypatch.setattr(launch, 'ROOT', workspace)
    monkeypatch.setattr(launch, '_port_open', lambda *a: pytest.fail('mismatch probed port'))
    monkeypatch.setattr(launch, 'ensure_guest_port_available', lambda *a: pytest.fail('mismatch probed port'))
    monkeypatch.setattr(launch.subprocess, 'Popen', lambda *a, **k: pytest.fail('mismatch opened process'))
    with pytest.raises(ValueError, match='model'):
        launch.main(['--profile', str(profile), '--robot-model', expected, '--execute',
                     '--guest-session', str(workspace / 'tmp/guidance-live/https/pair.json')])


def test_ur5e_delivery_uses_its_actual_model_and_start_helper(workspace, monkeypatch):
    from leica_tracker.physical_delivery import prepare_delivery
    monkeypatch.setattr(socket, 'socket', lambda *a, **k: pytest.fail('offline opened socket'))
    original = ur5e_data()
    before = copy.deepcopy(original)
    output = workspace / 'delivery'
    manifest = prepare_delivery(original, output, workspace=workspace)
    assert original == before
    assert manifest['robot_model'] == 'ur5e'
    assert '# Complete UR5e + iPad' in (output / 'README.md').read_text(encoding='utf-8')
    startup = (output / 'start-reviewed-ipad.ps1').read_text(encoding='utf-8')
    assert 'start-ur5e-ipad-system.ps1' in startup
    assert 'start-ur10e-ipad-system.ps1' not in startup
    assert json.loads((output / 'baseline-profile.json').read_text())['robot_model'] == 'ur5e'


def test_delivery_expected_model_mismatch_creates_no_output(workspace):
    from leica_tracker.physical_delivery import prepare_delivery
    output = workspace / 'rejected'
    with pytest.raises(ValueError, match='model'):
        prepare_delivery(data(), output, workspace=workspace, robot_model='ur5e')
    assert not output.exists()


def test_ur5e_delivery_default_draft_creates_no_bundle(workspace, monkeypatch):
    from leica_tracker import physical_delivery
    draft = workspace / 'ur5e-draft.json'
    draft.write_text(json.dumps({'status': 'DRAFT_UR5E_NOT_EXECUTABLE',
                                 'robot_model': 'ur5e', 'reviewed_robot_model': None}))
    output = workspace / 'rejected'
    monkeypatch.setattr(physical_delivery, 'ROOT', workspace)
    with pytest.raises(SystemExit):
        physical_delivery.main(['--output', str(output)], default_model='ur5e', default_profile=draft)
    assert not output.exists()


@pytest.mark.parametrize('dedicated,override', [('ur5e', 'ur10e'), ('ur10e', 'ur5e')])
def test_dedicated_delivery_model_flag_cannot_change_identity(dedicated, override):
    from leica_tracker.physical_delivery import main
    with pytest.raises(SystemExit):
        main(['--robot-model', override], default_model=dedicated)


@pytest.mark.parametrize('script,model', [('prepare-ur5e-ipad.py', 'ur5e'),
                                         ('prepare-ur10e-ipad.py', 'ur10e')])
def test_dedicated_delivery_wrappers_bind_expected_identity(monkeypatch, script, model):
    from leica_tracker import physical_delivery
    calls = []
    monkeypatch.setattr(physical_delivery, 'main', lambda **kwargs: calls.append(kwargs) or 0)
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(scripts / script), run_name='__main__')
    assert result.value.code == 0
    assert calls[0]['default_model'] == model
