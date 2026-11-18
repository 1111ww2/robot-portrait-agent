# 默认离线检查；加上 -Execute 才在现场启动 HTTPS、相机和机器人服务。
# 动作仍需等待 iPad 上的开始指令。
param(
    [switch]$Execute,
    [string]$Profile='vision_mvp/config/selected-ur10e-0.7.json',
    [string]$GuestSession='',
    [string]$Address
)
$ErrorActionPreference='Stop'
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$python=Join-Path $workspaceRoot 'vision_mvp/.venv/Scripts/python.exe'
$entry=Join-Path $workspaceRoot 'vision_mvp/scripts/run-ur10e-ipad-system.py'
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
