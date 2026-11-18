"""测试用实际机型的 PowerShell 入口调用只操作文件的 Python 替身，不访问设备。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Windows PowerShell model entry')


@pytest.fixture
def entry_workspace(tmp_path):
    root = tmp_path / "model entry '\u56fe'"
    scripts = root / 'vision_mvp/scripts'
    scripts.mkdir(parents=True)
    interpreter = root / 'vision_mvp/.venv/Scripts/python.exe'
    interpreter.parent.mkdir(parents=True)
    os.link(sys.executable, interpreter)
    (interpreter.parent.parent / 'pyvenv.cfg').write_text(
        'home = ' + str(Path(sys._base_executable).parent) + '\ninclude-system-site-packages = false\n',
        encoding='utf-8')
    for model in ('ur5e', 'ur10e'):
        name = f'start-{model}-ipad-system.ps1'
        shutil.copyfile(ROOT / 'vision_mvp/scripts' / name, scripts / name)
        (scripts / f'run-{model}-ipad-system.py').write_text(
            "import json,sys\nfrom pathlib import Path\n"
            "root=Path(__file__).resolve().parents[2]\n"
            "(root/'entry-argv.json').write_text(json.dumps(sys.argv[1:]),encoding='utf-8')\n",
            encoding='utf-8')
    return root


@pytest.mark.parametrize('model', ['ur5e', 'ur10e'])
@pytest.mark.parametrize('explicit_pair', [False, True])
def test_model_entry_uses_current_pair_by_default_and_preserves_explicit_override(
        entry_workspace, model, explicit_pair):
    root = entry_workspace
    profile = root / 'selected profile.json'
    pair = root / 'explicit paired session \u56fe.json'
    command = ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
               '-File', str(root / 'vision_mvp/scripts' / f'start-{model}-ipad-system.ps1'),
               '-Profile', str(profile)]
    if explicit_pair:
        command += ['-GuestSession', str(pair), '-Execute']
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                            errors='replace', timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    arguments = json.loads((root / 'entry-argv.json').read_text(encoding='utf-8'))
    assert arguments[:2] == ['--profile', str(profile)]
    if explicit_pair:
        index = arguments.index('--guest-session')
        assert arguments[index + 1] == str(pair)
        assert arguments[-1] == '--execute'
    else:
        assert '--guest-session' not in arguments
        assert '--execute' not in arguments
    assert 'tmp/robot-network/ipad-trial-pairing.json' not in arguments


def test_automatic_pair_resolution_reuses_current_pair_instead_of_old_trial(tmp_path):
    from leica_tracker.physical_system import prepare_plan
    root = tmp_path / 'current pairing'
    scripts = root / 'vision_mvp/scripts'
    scripts.mkdir(parents=True)
    (scripts / 'run-physical-live.py').write_text('# file-only fixture\n', encoding='utf-8')
    profile = root / 'profile.json'
    profile.write_text('{"robot_model":"ur10e","reviewed_robot_model":"ur10e"}', encoding='utf-8')
    live = root / 'tmp/guidance-live'
    (live / 'https').mkdir(parents=True)
    current = {'port': 8777, 'key': 'current-fixture-pairing-key-123456'}
    for path in (live / 'guest-link.json', live / 'https/pair.json'):
        path.write_text(json.dumps(current), encoding='utf-8')
    (live / 'https-public-status.json').write_text(
        '{"address":"192.168.1.24","port":8768,"setup_port":8769}', encoding='utf-8')
    legacy = root / 'tmp/robot-network/ipad-trial-pairing.json'
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({'port': 8767, 'key': 'legacy-fixture-pairing-key-1234567'}), encoding='utf-8')
    plan = prepare_plan(root=root, profile=profile, profile_validator=lambda value:value)
    assert plan.guest_session == live / 'guest-link.json'
    assert plan.pairing == current


@pytest.mark.parametrize('model', ['ur5e', 'ur10e'])
def test_model_entry_defaults_to_the_complete_selected_design(entry_workspace, model):
    root = entry_workspace
    result = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
        '-File',str(root/'vision_mvp/scripts'/f'start-{model}-ipad-system.ps1')],
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    arguments = json.loads((root/'entry-argv.json').read_text(encoding='utf-8'))
    assert arguments == ['--profile',f'vision_mvp/config/selected-{model}-0.7.json']
