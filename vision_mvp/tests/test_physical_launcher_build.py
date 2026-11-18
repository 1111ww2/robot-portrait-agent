"""使用假进程API和工作区本地固定件重用练习启动器。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "vision_mvp/scripts/start-physical-live.ps1"


@pytest.fixture
def launcher_workspace():
    if os.name != "nt":
        pytest.skip("Windows PowerShell process APIs")
    base = ROOT / "tmp/test-launcher-build"
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="case-", dir=base) as directory:
        root = Path(directory)
        scripts = root / "vision_mvp/scripts"
        sources = root / "vision_mvp/src/leica_tracker"
        scripts.mkdir(parents=True)
        (sources / "nested").mkdir(parents=True)
        shutil.copyfile(SCRIPT, scripts / SCRIPT.name)
        (scripts / "run-physical-live.py").write_text("raise RuntimeError('never run')\n")
        (sources / "runtime.py").write_text("BUILD = 1\n")
        (sources / "nested/worker.py").write_text("BUILD = 1\n")
        (sources / "guest.html").write_text("<html>fixture</html>\n")
        (root / "profile.json").write_text('{"speed": 0.1}\n')
        (root / "pairing.json").write_text('{"port": 8767, "key": "fixture-private-pairing-key"}\n')
        yield root


PROGRAM = r'''
$ErrorActionPreference='Stop'
class LauncherTestTcpClient {
    static [int]$Count=0
    LauncherTestTcpClient() {
        [LauncherTestTcpClient]::Count++
        throw 'Forbidden offline port probe'
    }
}
$accelerators=[psobject].Assembly.GetType('System.Management.Automation.TypeAccelerators')
$accelerators::Add('Net.Sockets.TcpClient',[LauncherTestTcpClient])
$scriptFile=$env:LEICA_TEST_BUILD_SCRIPT
$root=Split-Path (Split-Path (Split-Path $scriptFile -Parent) -Parent) -Parent
$profilePath=Join-Path $root 'profile.json'
$pairPath=Join-Path $root 'pairing.json'
$logs=Join-Path $root 'tmp/physical-standalone'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$recordFile=Join-Path $logs 'process.json'
$runner=Join-Path $root 'vision_mvp/scripts/run-physical-live.py'
# Export only the pure helper. Dot-sourcing the launcher would start its body.
$tokens=$null; $parseErrors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($scriptFile,[ref]$tokens,[ref]$parseErrors)
if ($parseErrors.Count) { throw 'Launcher parse failed' }
$helper=$ast.Find({param($node)
    $node -is [Management.Automation.Language.FunctionDefinitionAst] -and
    $node.Name -eq 'Get-LauncherBuildFingerprint'
},$false)
$fingerprint='missing-build-helper'
if ($helper) {
    Invoke-Expression $helper.Extent.Text
    $fingerprint=Get-LauncherBuildFingerprint $root $profilePath $pairPath
}
$record=@{process_id=111;profile=$profilePath;build_fingerprint=$fingerprint}
if ($env:LEICA_TEST_BUILD_CASE -eq 'legacy') { $record.Remove('build_fingerprint') }
$record | ConvertTo-Json | Set-Content -LiteralPath $recordFile -Encoding utf8
$before=[IO.File]::ReadAllText($recordFile)
switch ($env:LEICA_TEST_BUILD_CASE) {
    'source' { Add-Content -LiteralPath (Join-Path $root 'vision_mvp/src/leica_tracker/nested/worker.py') -Value '# changed' }
    'guest' { Add-Content -LiteralPath (Join-Path $root 'vision_mvp/src/leica_tracker/guest.html') -Value '<!-- changed -->' }
    'launcher_ps1' { Add-Content -LiteralPath $scriptFile -Value '# changed' }
    'launcher_py' { Add-Content -LiteralPath $runner -Value '# changed' }
    'profile' { Add-Content -LiteralPath $profilePath -Value ' ' }
    'pairing' { [IO.File]::WriteAllText($pairPath,'{"port":8767,"key":"changed-private-pairing-key"}') }
    'added_source' { [IO.File]::WriteAllText((Join-Path $root 'vision_mvp/src/leica_tracker/new.py'),'NEW = True') }
    'removed_source' { Remove-Item -LiteralPath (Join-Path $root 'vision_mvp/src/leica_tracker/nested/worker.py') }
}
$global:cimCalls=0; $global:startCalls=0; $global:stopCalls=0; $global:portCalls=0; $global:validationCalls=0
function Get-CimInstance {
    param($ClassName,$Filter)
    $global:cimCalls++
    if ($Filter -ne 'ProcessId=111') { throw 'Unexpected process lookup' }
    $command='"'+$runner+'" --profile "'+$profilePath+'"'
    if ($env:LEICA_TEST_BUILD_CASE -eq 'unknown') { $command='unrelated.exe' }
    return [pscustomobject]@{ProcessId=111;CommandLine=$command}
}
function Start-Process {
    $global:startCalls++
    if ($env:LEICA_TEST_BUILD_CASE -eq 'persist') { return [pscustomobject]@{Id=333} }
    throw 'Must not start a process'
}
function Stop-Process { $global:stopCalls++; throw 'Must not stop a process' }
function Get-NetTCPConnection { $global:portCalls++; throw 'Must not probe ports' }
function Invoke-LauncherTestPython { $global:validationCalls++; $global:LASTEXITCODE=0 }
$failed=$false; $message=''
try {
    if ($env:LEICA_TEST_BUILD_CASE -eq 'persist') {
        # Execute only the real launch/persistence block with fake validation and Start-Process.
        # In particular, do not run the port probe or launcher body to export helpers.
        $workspaceRoot=$root; $python='Invoke-LauncherTestPython'; $buildFingerprint=$fingerprint
        Push-Location $root
        & ([ScriptBlock]::Create($ast.EndBlock.Statements[-1].Extent.Text))
    } else { & $scriptFile -Profile $profilePath -GuestSession $pairPath }
}
catch { $failed=$true; $message=$_.Exception.Message }
$stored=Get-Content -LiteralPath $recordFile -Raw | ConvertFrom-Json
@{failed=$failed;message=$message;cim_calls=$global:cimCalls;start_calls=$global:startCalls;
  stop_calls=$global:stopCalls;port_calls=($global:portCalls+[LauncherTestTcpClient]::Count);
  validation_calls=$global:validationCalls;stored_pid=$stored.process_id;
  stored_fingerprint=$stored.build_fingerprint;expected_fingerprint=$fingerprint;
  record_unchanged=([IO.File]::ReadAllText($recordFile) -eq $before)} | ConvertTo-Json -Compress
'''


def run_launcher(root, case):
    env = dict(os.environ, LEICA_TEST_BUILD_CASE=case,
               LEICA_TEST_BUILD_SCRIPT=str(root / "vision_mvp/scripts" / SCRIPT.name))
    result = subprocess.run(["powershell", "-NoProfile", "-Command", PROGRAM],
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fixture-private-pairing-key" not in result.stdout + result.stderr
    assert "changed-private-pairing-key" not in result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_matching_live_build_is_reused_without_start_or_stop(launcher_workspace):
    result = run_launcher(launcher_workspace, "unchanged")
    assert not result["failed"], result["message"]
    assert result["cim_calls"] == 1
    assert result["start_calls"] == result["stop_calls"] == result["port_calls"] == 0
    assert result["record_unchanged"]


@pytest.mark.parametrize("case", ["source", "guest", "launcher_ps1", "launcher_py",
                                 "profile", "pairing", "added_source", "removed_source", "legacy"])
def test_changed_or_legacy_live_build_requires_explicit_onsite_restart(launcher_workspace, case):
    result = run_launcher(launcher_workspace, case)
    assert result["failed"], f"{case}: reused a stale live build"
    assert "onsite" in result["message"].lower()
    assert "stop" in result["message"].lower() and "restart" in result["message"].lower()
    assert result["cim_calls"] == 1
    assert result["start_calls"] == result["stop_calls"] == result["port_calls"] == 0
    assert result["record_unchanged"]


def test_unknown_process_is_rejected_without_killing_it(launcher_workspace):
    result = run_launcher(launcher_workspace, "unknown")
    assert result["failed"]
    assert "identity differs" in result["message"]
    assert result["start_calls"] == result["stop_calls"] == result["port_calls"] == 0
    assert result["record_unchanged"]


def test_new_launch_record_persists_fingerprint_without_real_process_or_probe(launcher_workspace):
    result = run_launcher(launcher_workspace, "persist")
    assert not result["failed"], result["message"]
    assert result["validation_calls"] == result["start_calls"] == 1
    assert result["cim_calls"] == result["stop_calls"] == result["port_calls"] == 0
    assert result["stored_pid"] == 333
    assert len(result["stored_fingerprint"]) == 64
    assert result["stored_fingerprint"] == result["expected_fingerprint"]
