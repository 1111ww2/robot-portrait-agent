param([Parameter(Mandatory = $true)][int]$Port)

# 仅停止由工作区管理、占用指定访客端口的仿真启动器。不操作实体机器人、夹爪或无关进程，也不删除文件。
$ErrorActionPreference = 'Stop'
if ($Port -lt 1 -or $Port -gt 65535) { throw 'Invalid guest port' }
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Group-Object OwningProcess | ForEach-Object { $_.Group[0] })
if ($listeners.Count -eq 0) { return }

function Is-Descendant([int]$Child, [int]$Ancestor) {
    for ($depth = 0; $depth -lt 32 -and $Child -gt 0; $depth++) {
        if ($Child -eq $Ancestor) { return $true }
        $process = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $Child)
        if (-not $process) { return $false }
        $Child = [int]$process.ParentProcessId
    }
    return $false
}

$root = $null
$rootCreation = $null
$rootCommand = $null
$expectedExe = [IO.Path]::GetFullPath((Join-Path $workspace 'vision_mvp/.venv/Scripts/python.exe'))
$expectedScript = [IO.Path]::GetFullPath((Join-Path $workspace 'vision_mvp/scripts/run-guidance-live.py'))
foreach ($name in @('reference-live.pid', 'launcher.pid')) {
    $path = Join-Path $workspace ('tmp/guidance-live/' + $name)
    if (-not (Test-Path -LiteralPath $path)) { continue }
    $raw = (Get-Content -LiteralPath $path -Raw).Trim()
    $candidate = 0
    if (-not [int]::TryParse($raw, [ref]$candidate) -or $candidate -le 0) { continue }
    $process = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $candidate)
    if (-not $process -or -not $process.CommandLine -or -not $process.ExecutablePath -or
        -not $process.ExecutablePath.Equals($expectedExe, [StringComparison]::OrdinalIgnoreCase)) { continue }
    $command = $process.CommandLine.Replace('/', '\')
    $absolutePattern = '(?i)(^|\s|["''])' + [regex]::Escape($expectedScript) + '(?=$|\s|["''])'
    $relativePattern = '(?i)(^|\s|["''])vision_mvp\\scripts\\run-guidance-live\.py(?=$|\s|["''])'
    if ($command -notmatch $absolutePattern -and $command -notmatch $relativePattern) { continue }
    $fileTime = (Get-Item -LiteralPath $path).LastWriteTime
    $pidAge = ($fileTime - $process.CreationDate).TotalSeconds
    if ($pidAge -lt -5 -or $pidAge -gt 60) { continue }
    $allOwned = $true
    foreach ($listener in $listeners) {
        if (-not (Is-Descendant ([int]$listener.OwningProcess) $candidate)) { $allOwned = $false; break }
    }
    if ($allOwned) { $root = $candidate; $rootCreation = $process.CreationDate; $rootCommand = $process.CommandLine; break }
}
if ($null -eq $root) { throw 'Guest port is not owned by a verified managed simulator; stop the existing app manually' }
# 只终止已核验的访客监听进程，避免结束整棵进程树。启动器监测子进程退出并清理桥接服务。
$verified = @()
foreach ($listener in $listeners) {
    $id = [int]$listener.OwningProcess
    $info = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $id)
    if (-not $info -or -not $info.CommandLine -or -not $info.ExecutablePath -or
        [IO.Path]::GetFileName($info.ExecutablePath) -notmatch '(?i)^pythonw?\.exe$' -or
        $info.CommandLine -notmatch '(?i)(^|\s)-m\s+leica_tracker(?=\s|$)' -or
        $info.CommandLine -notmatch '(?i)(^|\s)--guest-display(?=\s|$)' -or
        -not (Is-Descendant $id $root)) { throw 'Unrecognized guest listener; no process stopped' }
    $handle = Get-Process -Id $id -ErrorAction Stop
    if ([math]::Abs(($handle.StartTime - $info.CreationDate).TotalSeconds) -gt 1) {
        throw 'Guest listener identity changed; no process stopped'
    }
    # 先获取操作系统进程句柄，再核验身份，防止 PID 被重用后误杀其他进程。
    $null = $handle.Handle
    $again = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $id)
    if (-not $again -or $again.CreationDate -ne $info.CreationDate -or
        $again.CommandLine -ne $info.CommandLine -or -not (Is-Descendant $id $root)) {
        throw 'Guest listener identity changed; no process stopped'
    }
    $verified += $handle
}
$currentRoot = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $root)
if (-not $currentRoot -or $currentRoot.CreationDate -ne $rootCreation -or $currentRoot.CommandLine -ne $rootCommand -or
    -not $currentRoot.ExecutablePath.Equals($expectedExe, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Managed launcher changed; no process stopped'
}
$currentOwners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
                   Select-Object -ExpandProperty OwningProcess -Unique | Sort-Object)
$expectedOwners = @($listeners | ForEach-Object { [int]$_.OwningProcess } | Sort-Object)
if (($currentOwners -join ',') -ne ($expectedOwners -join ',')) {
    throw 'Guest port ownership changed; no process stopped'
}
foreach ($handle in $verified) { $handle.Kill(); $handle.WaitForExit(3000) | Out-Null }
Write-Output 'Stopped the verified managed simulator guest app.'
