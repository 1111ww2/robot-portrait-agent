"""物理启动器在执行明确审查过的配置文件之前是惰性的。"""
import importlib.util
import io
import json
import os
import subprocess
from pathlib import Path

import pytest


def launcher():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'run-physical-live.py'
    spec = importlib.util.spec_from_file_location('physical_launcher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_camera_command_uses_ipad_and_claude_without_simulator(tmp_path):
    launch = launcher()
    command = launch.camera_command('python', 8767, tmp_path / '.env.local')
    assert '--camera-source' in command and command[command.index('--camera-source') + 1] == 'ipad'
    assert '--profile' not in command and 'claude-guide' in command and '--single-plan' in command
    assert '--guidance-stream' in command and '--headless' in command
    assert '--guest-port' in command and command[command.index('--guest-port') + 1] == '8767'
    assert not any('simulator' in part for part in command)


def test_camera_environment_isolates_feedback_and_keeps_existing_pair():
    launch = launcher()
    env = launch.camera_environment({'LEICA_SIM_FEEDBACK': 'old',
                                     'LEICA_SIM_FAST_GESTURES': '1', 'SOME_VAR': 'yes'}, 'secret-token')
    assert env['LEICA_PHYSICAL_FEEDBACK'] == '1'
    assert env['LEICA_GUEST_SESSION_TOKEN'] == 'secret-token'
    assert env['SOME_VAR'] == 'yes'
    assert 'LEICA_SIM_FEEDBACK' not in env and 'LEICA_SIM_FAST_GESTURES' not in env


@pytest.mark.parametrize('pair', [{}, {'port': True, 'key': 'a' * 32},
                                 {'port': 8767, 'key': 'short'},
                                 {'port': 0, 'key': 'a' * 32}])
def test_pairing_must_be_local_valid_record(pair):
    with pytest.raises(ValueError): launcher().validate_pairing(pair)


def test_validate_only_does_not_launch_any_process(monkeypatch):
    launch = launcher()
    monkeypatch.setattr(launch, 'validate_profile', lambda _: None)
    monkeypatch.setattr(launch.subprocess, 'Popen', lambda *a, **k: pytest.fail('must remain offline'))
    assert launch.main(['--profile', str(launch.ROOT / 'profile.json')]) == 0


def test_launcher_owns_one_camera_and_source_for_persistent_sessions(monkeypatch,capsys):
    from leica_tracker import physical_runtime
    launch=launcher();calls=[]
    class Camera:
        stdout=io.StringIO('')
        done=False
        def poll(self):return 0 if self.done else None
        def wait(self,**kwargs):calls.append('wait-camera');self.done=True
        def terminate(self):calls.append('terminate-camera');self.done=True
    camera=Camera()
    sources=[]
    class Source:
        def __init__(self,stream):
            assert stream is camera.stdout
            sources.append(self)
            self.thread=type('Reader',(),{'join':lambda self,**kw:calls.append('join-reader')})()
    def serve(profile,port,key,source,**kwargs):
        calls.append('serve')
        assert source is sources[0] and kwargs['is_alive']() is True
        assert camera.done is False
        return 0
    monkeypatch.setattr(launch,'validate_profile',lambda _:None)
    monkeypatch.setattr(launch,'ensure_guest_port_available',lambda _:None)
    monkeypatch.setattr(launch,'_wait_for_guest',lambda *args:None)
    monkeypatch.setattr(Path,'read_text',lambda *args,**kwargs:json.dumps({'port':8767,'key':'k'*32}))
    monkeypatch.setattr(launch.subprocess,'Popen',lambda *args,**kwargs:camera)
    monkeypatch.setattr(physical_runtime,'LatestEventSource',Source)
    monkeypatch.setattr(physical_runtime,'serve_sessions',serve)
    assert launch.main(['--profile',str(launch.ROOT/'fake.json'),'--execute','--keep-display-on-fault'])==0
    assert len(sources)==1
    assert calls==['serve','terminate-camera','wait-camera','join-reader']
    assert 'waiting for explicit iPad Start' in capsys.readouterr().err


def test_profile_outside_workspace_is_rejected(monkeypatch, tmp_path):
    launch = launcher()
    # 将所有测试文件都保存在实际工作区内；使用虚拟项目根目录，这样它仍然是一个外部项目输入，带有本地基临时文件。
    monkeypatch.setattr(launch, 'ROOT', tmp_path / 'workspace')
    monkeypatch.setattr(launch, 'validate_profile', lambda _: pytest.fail('must reject outside path first'))
    with pytest.raises(ValueError, match='workspace'):
        launch.main(['--profile', str(tmp_path / 'profile.json')])


def test_switch_only_requests_verified_managed_simulator_stop(monkeypatch):
    launch = launcher()
    opened = iter([True, False])
    monkeypatch.setattr(launch, '_port_open', lambda _: next(opened))
    monkeypatch.setattr(launch.sys, 'platform', 'win32')
    calls = []
    monkeypatch.setattr(launch.subprocess, 'run', lambda args, **kw: calls.append((args, kw)))
    launch.ensure_guest_port_available(8767)
    assert len(calls) == 1
    assert calls[0][0][-2:] == ['-Port', '8767']
    assert calls[0][1]['check'] is True


def test_other_platform_will_not_kill_an_unknown_port_owner(monkeypatch):
    launch = launcher()
    monkeypatch.setattr(launch, '_port_open', lambda _: True)
    monkeypatch.setattr(launch.sys, 'platform', 'darwin')
    with pytest.raises(RuntimeError, match='occupied'):
        launch.ensure_guest_port_available(8767)


def test_require_free_port_rejects_existing_listener_without_stopping_it(monkeypatch):
    launch = launcher()
    monkeypatch.setattr(launch, 'validate_profile', lambda _: None)
    monkeypatch.setattr(Path, 'read_text', lambda *args, **kwargs: json.dumps({'port': 8767, 'key': 'k' * 32}))
    monkeypatch.setattr(launch, '_port_open', lambda _: True)
    monkeypatch.setattr(launch, 'ensure_guest_port_available', lambda _: pytest.fail('must not auto-stop existing services'))
    monkeypatch.setattr(launch.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('occupied port must reject startup'))
    with pytest.raises(RuntimeError, match='occupied'):
        launch.main(['--profile', str(launch.ROOT / 'fake.json'), '--execute', '--require-free-port'])


def test_stop_file_cannot_escape_workspace(monkeypatch, tmp_path):
    launch = launcher()
    monkeypatch.setattr(launch, 'ROOT', tmp_path / 'workspace')
    monkeypatch.setattr(launch, 'validate_profile', lambda _: pytest.fail('reject path first'))
    with pytest.raises(ValueError, match='workspace'):
        launch.main(['--profile', str(launch.ROOT / 'fake.json'), '--stop-file', str(tmp_path / 'stop.request')])


def test_existing_stop_file_rejects_startup(monkeypatch):
    launch = launcher()
    monkeypatch.setattr(launch, 'validate_profile', lambda _: pytest.fail('reject stop file first'))
    with pytest.raises(ValueError, match='already exists'):
        launch.main(['--profile', str(launch.ROOT / 'fake.json'), '--stop-file', str(Path(__file__).resolve())])


def test_cooperative_stop_file_retires_persistent_service(monkeypatch):
    from leica_tracker import physical_runtime
    launch = launcher()
    stop_file = launch.ROOT / 'tmp/test-child-cooperative-stop.request'
    # 存在被注入：没有停止文件、子进程、网络或机器人。
    written = {'stop': False}
    original_exists = Path.exists
    monkeypatch.setattr(Path, 'exists', lambda path: written['stop'] if path == stop_file else original_exists(path))
    class Camera:
        stdout = io.StringIO('')
        done = False
        def poll(self): return 0 if self.done else None
        def terminate(self): self.done = True
        def wait(self, **kwargs): self.done = True
    camera = Camera()
    class Source:
        def __init__(self, stream):
            self.thread = type('Reader', (), {'join': lambda *args, **kwargs: None})()
    def serve(*args, is_alive):
        assert is_alive() is True
        written['stop'] = True
        assert is_alive() is False
        return 0
    monkeypatch.setattr(launch, 'validate_profile', lambda _: None)
    monkeypatch.setattr(launch, '_port_open', lambda _: False)
    monkeypatch.setattr(launch, 'ensure_guest_port_available', lambda _: pytest.fail('must not auto-stop'))
    monkeypatch.setattr(launch, '_wait_for_guest', lambda *args: None)
    monkeypatch.setattr(Path, 'read_text', lambda *args, **kwargs: json.dumps({'port': 8767, 'key': 'k' * 32}))
    monkeypatch.setattr(launch.subprocess, 'Popen', lambda *args, **kwargs: camera)
    monkeypatch.setattr(physical_runtime, 'LatestEventSource', Source)
    monkeypatch.setattr(physical_runtime, 'serve_sessions', serve)
    assert launch.main(['--profile', str(launch.ROOT / 'fake.json'), '--execute',
                        '--require-free-port', '--stop-file', str(stop_file)]) == 0
    assert camera.done is True


@pytest.mark.parametrize('case', ['valid_duplicate', 'unknown_script', 'port_changed', 'root_changed'])
def test_windows_stop_script_uses_only_verified_mock_listener(case):
    if os.name != 'nt':
        pytest.skip('Windows PowerShell process APIs')
    launch = launcher()
    program = r'''
$ErrorActionPreference='Stop'
$scriptFile=$env:LEICA_TEST_STOP_SCRIPT
$workspace=[IO.Path]::GetFullPath((Join-Path (Split-Path $scriptFile -Parent) '../..'))
$exe=Join-Path $workspace 'vision_mvp/.venv/Scripts/python.exe'
$script=Join-Path $workspace 'vision_mvp/scripts/run-guidance-live.py'
$global:fakeCreated=Get-Date
$global:stopCount=0; $global:netCalls=0; $global:rootCalls=0
$global:fakeRoot=[pscustomobject]@{ProcessId=111;ParentProcessId=1;ExecutablePath=$exe;CreationDate=$global:fakeCreated;CommandLine=('"'+$exe+'" -u "'+$script+'"')}
$global:fakeListener=[pscustomobject]@{ProcessId=222;ParentProcessId=111;ExecutablePath='C:\Python\python.exe';CreationDate=$global:fakeCreated;CommandLine='python.exe -m leica_tracker --guest-display'}
if ($env:LEICA_TEST_CASE -eq 'unknown_script') { $global:fakeRoot.CommandLine='python.exe C:\elsewhere\run-guidance-live.py' }
function Get-NetTCPConnection { param($State,$LocalPort,$ErrorAction) $global:netCalls++; if ($env:LEICA_TEST_CASE -eq 'port_changed' -and $global:netCalls -gt 1) { return [pscustomobject]@{OwningProcess=333} }; return @([pscustomobject]@{OwningProcess=222},[pscustomobject]@{OwningProcess=222}) }
function Get-CimInstance { param($ClassName,$Filter) if ($Filter -eq 'ProcessId=111') { $global:rootCalls++; if ($env:LEICA_TEST_CASE -eq 'root_changed' -and $global:rootCalls -gt 1) { return [pscustomobject]@{CreationDate=$global:fakeCreated.AddSeconds(1);CommandLine=$global:fakeRoot.CommandLine;ExecutablePath=$global:fakeRoot.ExecutablePath} }; return $global:fakeRoot }; if ($Filter -eq 'ProcessId=222') {return $global:fakeListener}; return $null }
function Test-Path { param($LiteralPath) return $true }
function Get-Content { param($LiteralPath,[switch]$Raw) return '111' }
function Get-Item { param($LiteralPath) return [pscustomobject]@{LastWriteTime=$global:fakeCreated} }
function Get-Process { param($Id,$ErrorAction) $handle=[pscustomobject]@{StartTime=$global:fakeCreated;Handle=1}; $handle | Add-Member ScriptMethod Kill { $global:stopCount++ }; $handle | Add-Member ScriptMethod WaitForExit { param($ms) return $true }; return $handle }
$failed=$false
try { & $scriptFile -Port 8767 } catch { $failed=$true; Write-Output $_.Exception.Message }
if ($env:LEICA_TEST_CASE -eq 'valid_duplicate') { if ($failed -or $global:stopCount -ne 1) {exit 1} }
else { if (-not $failed -or $global:stopCount -ne 0) {exit 1} }
'''
    env = dict(os.environ, LEICA_TEST_CASE=case,
               LEICA_TEST_STOP_SCRIPT=str(launch.VISION / 'scripts' / 'stop-managed-simulator.ps1'))
    result = subprocess.run(['powershell', '-NoProfile', '-Command', program],
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
