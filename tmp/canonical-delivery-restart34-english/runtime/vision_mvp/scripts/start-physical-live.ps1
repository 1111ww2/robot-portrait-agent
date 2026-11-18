param(
    [Parameter(Mandatory=$true)][string]$Profile,
    [string]$GuestSession = 'tmp/robot-network/ipad-trial-pairing.json'
)
$ErrorActionPreference='Stop'
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
function WorkspaceFile([string]$value) {
    $candidate=if ([IO.Path]::IsPathRooted($value)) { $value } else { Join-Path $workspaceRoot $value }
    $resolved=(Resolve-Path -LiteralPath $candidate).Path
    if (-not $resolved.StartsWith($workspaceRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) {
        throw 'The profile and pairing must be inside this workspace.'
    }
    return $resolved
}
function Get-LauncherBuildFingerprint([string]$root,[string]$profileFile,[string]$pairFile) {
    $sourceRoot=Join-Path $root 'vision_mvp/src/leica_tracker'
    $files=@(Get-ChildItem -LiteralPath $sourceRoot -Filter '*.py' -File -Recurse |
        ForEach-Object { $_.FullName })
    $files+=Join-Path $sourceRoot 'guest.html'
    $files+=Join-Path $root 'vision_mvp/scripts/start-physical-live.ps1'
    $files+=Join-Path $root 'vision_mvp/scripts/run-physical-live.py'
    $sha=[Security.Cryptography.SHA256]::Create()
    try {
        $hashFile={param($file)
            [BitConverter]::ToString($sha.ComputeHash([IO.File]::ReadAllBytes($file))).Replace('-','')
        }
        $entries=@(foreach ($file in ($files | Sort-Object -Unique)) {
            $name=$file.Substring($root.Length).Replace('\','/')
            $name+':'+(& $hashFile $file)
        })
        # 计算配对内容的哈希值，不显示或保存配对密钥本身。
        $entries+='profile:'+(& $hashFile $profileFile)
        $entries+='pairing:'+(& $hashFile $pairFile)
        $manifest="leica-physical-build-v1`n"+($entries -join "`n")
        return [BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($manifest))).Replace('-','').ToLowerInvariant()
    } finally { $sha.Dispose() }
}
$profilePath=WorkspaceFile $Profile
$pairPath=WorkspaceFile $GuestSession
$buildFingerprint=Get-LauncherBuildFingerprint $workspaceRoot $profilePath $pairPath
$python=Join-Path $workspaceRoot 'vision_mvp/.venv/Scripts/python.exe'
$runner=Join-Path $workspaceRoot 'vision_mvp/scripts/run-physical-live.py'
$logs=Join-Path $workspaceRoot 'tmp/physical-standalone'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$recordFile=Join-Path $logs 'process.json'
if (Test-Path -LiteralPath $recordFile) {
    $record=Get-Content -LiteralPath $recordFile -Raw | ConvertFrom-Json
    $existing=Get-CimInstance Win32_Process -Filter ('ProcessId='+[int]$record.process_id)
    if ($existing) {
        if ($existing.CommandLine.Contains($runner) -and $record.profile -eq $profilePath) {
            if ($record.build_fingerprint -ne $buildFingerprint) {
                throw 'The running physical service has an old or unknown build. During an explicitly authorized onsite session, stop it and restart this launcher to load the current source, profile and pairing. Nothing was stopped or restarted.'
            }
            Write-Output ('Physical iPad service is already running. PID '+$existing.ProcessId)
            exit 0
        }
        throw 'Saved process identity differs; inspect it before starting another controller.'
    }
}
$pair=Get-Content -LiteralPath $pairPath -Raw | ConvertFrom-Json
$probe=[Net.Sockets.TcpClient]::new()
$portBusy=$false
try {
    $connection=$probe.ConnectAsync('127.0.0.1',[int]$pair.port)
    $portBusy=$connection.Wait(250) -and $probe.Connected
} catch { $portBusy=$false } finally { $probe.Dispose() }
if ($portBusy) {
    throw 'The paired camera port is occupied. Close the old project launcher before switching.'
}
Push-Location $workspaceRoot
try {
    & $python $runner --profile $profilePath --guest-session $pairPath
    if ($LASTEXITCODE -ne 0) { throw 'Physical profile validation failed; nothing started.' }
    $launchArgs=@('-u',('"'+$runner+'"'),'--profile',('"'+$profilePath+'"'),
                  '--guest-session',('"'+$pairPath+'"'),'--execute')
    $process=Start-Process -FilePath $python -ArgumentList $launchArgs -WorkingDirectory $workspaceRoot `
        -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs 'stdout.log') `
        -RedirectStandardError (Join-Path $logs 'stderr.log') -PassThru
    @{process_id=$process.Id;profile=$profilePath;build_fingerprint=$buildFingerprint;started_utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $recordFile -Encoding utf8
    Write-Output ('Physical iPad service started in the background. PID '+$process.Id+'. Use Start on iPad.')
} finally { Pop-Location }
