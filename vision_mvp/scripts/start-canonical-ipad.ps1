# 选择已发布的固定版本，默认先做离线验证。
param(
    [ValidateSet('ur5e','ur10e')][string]$RobotModel,
    [string]$Profile='',
    [string]$GuestSession='',
    [string]$RuntimeRoot='',
    [switch]$Execute
)
$ErrorActionPreference='Stop'
if (-not $RobotModel) { throw 'Choose -RobotModel ur5e or ur10e.' }
$RobotModel=$RobotModel.ToLowerInvariant()
if (-not $RuntimeRoot) { $RuntimeRoot=Join-Path $PSScriptRoot '../..' }
$workspaceRoot=(Resolve-Path -LiteralPath $RuntimeRoot).ProviderPath
$python=Join-Path $workspaceRoot 'vision_mvp/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'The existing project Python environment is required.' }

# 使用 Python 解析 Windows 目录连接，仅比较路径字符串前缀无法识别这些连接。
$pathGuard=@'
from pathlib import Path
import json,sys
parent=Path(sys.argv[1]).resolve()
candidate=Path(sys.argv[2])
target=(candidate if candidate.is_absolute() else parent/candidate).resolve()
if target==parent or not target.is_relative_to(parent):
    raise ValueError('Canonical path must remain inside its selected project or bundle')
print(json.dumps(str(target),ensure_ascii=True))
'@
function Resolve-ContainedPath([string]$Parent,[string]$Candidate,[switch]$File) {
    if ([string]::IsNullOrWhiteSpace($Candidate)) { throw 'A canonical path is missing.' }
    $resolvedOutput=@(& $python -c $pathGuard $Parent $Candidate)
    if ($LASTEXITCODE -ne 0 -or $resolvedOutput.Count -ne 1) { throw 'Canonical path is outside the selected project or bundle.' }
    $resolved=[string]($resolvedOutput[0] | ConvertFrom-Json)
    $pathType=if ($File) { 'Leaf' } else { 'Container' }
    if (-not (Test-Path -LiteralPath $resolved -PathType $pathType)) { throw 'A published canonical file or directory is missing.' }
    return $resolved
}

$pointerPath=Resolve-ContainedPath $workspaceRoot 'tmp/canonical-delivery-current.json' -File
$pointer=Get-Content -LiteralPath $pointerPath -Raw -Encoding UTF8 | ConvertFrom-Json
$bundleRelative=[string]$pointer.bundle_path
$expectedHash=[string]$pointer.shared_runtime_source_sha256
$expectedManifestHash=[string]$pointer.manifest_sha256
if ([IO.Path]::IsPathRooted($bundleRelative) -or $expectedHash -cnotmatch '^[0-9a-f]{64}$' -or
    $expectedManifestHash -cnotmatch '^[0-9a-f]{64}$') {
    throw 'The canonical pointer requires a workspace-relative bundle, source hash and manifest hash.'
}
$bundle=Resolve-ContainedPath $workspaceRoot $bundleRelative
$manifestPath=Resolve-ContainedPath $bundle 'manifest.json' -File
if (Get-Command Get-FileHash -ErrorAction SilentlyContinue) {
    $actualManifestHash=(Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
} else {
    # 若当前终端的模块路径受限，可能找不到 Get-FileHash。
    # 此时在本机使用同一 SHA256 算法，无须安装模块。
    $manifestHasher=[System.Security.Cryptography.SHA256]::Create()
    try {
        $actualManifestHash=([BitConverter]::ToString($manifestHasher.ComputeHash(
            [System.IO.File]::ReadAllBytes($manifestPath)))).Replace('-','').ToLowerInvariant()
    } finally { $manifestHasher.Dispose() }
}
if ($actualManifestHash -cne $expectedManifestHash) {
    throw 'The canonical manifest differs from the accepted published build.'
}
$manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.shared_runtime_source_sha256 -cne $expectedHash -or $manifest.joint_speed_rad_s -ne 0.7 -or
    $manifest.workflow -ne 'ONE_SHARED_NATIVE_RUNTIME') {
    throw 'The canonical pointer, source identity or 0.7 rad/s design does not match.'
}
$modelProperty=$manifest.models.PSObject.Properties[$RobotModel]
if ($null -eq $modelProperty) { throw 'The published bundle does not contain the requested robot model.' }
$model=$modelProperty.Value
if ($model.entry -cne ('start-'+$RobotModel+'.ps1')) { throw 'The model-bound entry does not match the requested robot model.' }
$modelEntry=Resolve-ContainedPath $bundle ([string]$model.entry) -File
$verifier=Resolve-ContainedPath $bundle 'runtime/vision_mvp/scripts/verify-canonical-runtime.py' -File
if ($Profile) { $profilePath=Resolve-ContainedPath $workspaceRoot $Profile -File }
else {
    if ([IO.Path]::IsPathRooted([string]$model.profile)) { throw 'The bundled model profile must be relative.' }
    $profilePath=Resolve-ContainedPath $bundle ([string]$model.profile) -File
}
$design=Get-Content -LiteralPath $profilePath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($design.robot_model -cne $RobotModel -or $design.intro.robot_model -cne $RobotModel -or
    $design.installation.robot_model -cne $RobotModel -or $design.intro.joint_speed -ne 0.7 -or
    $design.installation.base_rate -ne 0.7 -or $design.installation.gesture.joint_speed_limit -ne 0.7) {
    throw 'The selected installation profile must match this robot model and canonical 0.7 rad/s design.'
}
$entryArguments=@{RuntimeRoot=$workspaceRoot;Profile=$profilePath}
if ($GuestSession) {
    $guestPath=Resolve-ContainedPath $workspaceRoot $GuestSession -File
    $entryArguments['GuestSession']=$guestPath
}
$previousPythonPath=$env:PYTHONPATH
$env:PYTHONPATH=Join-Path $workspaceRoot 'vision_mvp/src'
$entryExitCode=0
try {
    & $python $verifier --bundle $bundle --runtime-root $workspaceRoot
    if ($LASTEXITCODE -ne 0) { throw 'Canonical source verification failed; no device entry was started.' }
    Write-Output ('CANONICAL_ENTRY_VERIFIED model='+$RobotModel+' joint_speed_rad_s=0.7')
    if ($Execute) {
        & $modelEntry @entryArguments
        $entryExitCode=$LASTEXITCODE
    } else { Write-Output 'CANONICAL_OFFLINE_VERIFIED: no device or service was started.' }
} finally { $env:PYTHONPATH=$previousPythonPath }
exit $entryExitCode
