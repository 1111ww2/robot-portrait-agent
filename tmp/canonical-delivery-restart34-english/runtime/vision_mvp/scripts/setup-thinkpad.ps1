$ErrorActionPreference = "Stop"

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "This project requires 64-bit Windows."
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed. Run: winget install --id=astral-sh.uv -e"
}

$ProjectDirectory = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectDirectory

Write-Host "[1/4] Syncing the locked Python 3.12 environment..."
uv sync --frozen --python 3.12 --extra robot
if ($LASTEXITCODE -ne 0) { throw "uv sync failed." }

Write-Host "[2/4] Running the full test suite..."
uv run --frozen --extra robot pytest -q
if ($LASTEXITCODE -ne 0) { throw "pytest failed." }

Write-Host "[3/4] Compiling source and tests..."
uv run --frozen --extra robot python -m compileall -q src tests
if ($LASTEXITCODE -ne 0) { throw "compileall failed." }

Write-Host "[4/4] Checking the Leica tracker command..."
uv run --frozen --extra robot leica-tracker --help | Out-Null
if ($LASTEXITCODE -ne 0) { throw "leica-tracker CLI check failed." }

uv run --frozen --extra robot python -c "import rtde_io, rtde_receive; from importlib.metadata import version; assert version('ur-rtde') == '1.6.5'"
if ($LASTEXITCODE -ne 0) { throw "ur-rtde import/version check failed." }

Write-Host "READY: software environment passed. Connect Pocket 3 in Webcam mode next."
