# 使用选定的 UR5e 方案。默认离线；
# -Execute 启动完整服务并复用现有配对，动作仍需等待 iPad 上的开始指令。
param(
    [switch]$Execute,
    [string]$Profile='vision_mvp/config/selected-ur5e-0.7.json',
    [string]$GuestSession='',
    [string]$Address
)
$ErrorActionPreference='Stop'
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$python=Join-Path $workspaceRoot 'vision_mvp/.venv/Scripts/python.exe'
$entry=Join-Path $workspaceRoot 'vision_mvp/scripts/run-ur5e-ipad-system.py'
$launchArguments=@('-u',$entry,'--profile',$Profile)
if ($GuestSession) { $launchArguments+=@('--guest-session',$GuestSession) }
if ($Address) { $launchArguments+=@('--address',$Address) }
if ($Execute) { $launchArguments+='--execute' }
Push-Location $workspaceRoot
try {
    & $python @launchArguments
    $result=$LASTEXITCODE
} finally { Pop-Location }
exit $result
