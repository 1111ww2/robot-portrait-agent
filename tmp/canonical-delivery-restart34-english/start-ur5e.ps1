# 按机型启动固定版本的运行包，复用当前项目环境。
param([string]$RuntimeRoot = 'C:\Users\13715\OneDrive\Desktop\deco2021',
      [string]$Profile = (Join-Path $PSScriptRoot 'profiles/ur5e.json'),
      [string]$GuestSession = '')
$ErrorActionPreference='Stop'
$python=Join-Path $RuntimeRoot 'vision_mvp/.venv/Scripts/python.exe'
$verifier=Join-Path $PSScriptRoot 'runtime/vision_mvp/scripts/verify-canonical-runtime.py'
$previousPythonPath=$env:PYTHONPATH
$env:PYTHONPATH=Join-Path $RuntimeRoot 'vision_mvp/src'
try {
    & $python $verifier --bundle $PSScriptRoot --runtime-root $RuntimeRoot
    if ($LASTEXITCODE -ne 0) { throw 'Use the exact validated runtime source before connecting a robot.' }
    $entry=Join-Path $RuntimeRoot 'vision_mvp/scripts/start-ur5e-ipad-system.ps1'
    if ($GuestSession) { & $entry -Execute -Profile $Profile -GuestSession $GuestSession }
    else { & $entry -Execute -Profile $Profile }
    if ($LASTEXITCODE -ne 0) { throw 'The model-bound runtime entry failed; inspect its preflight output.' }
} finally { $env:PYTHONPATH=$previousPythonPath }
