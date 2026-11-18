@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment missing. Run scripts\setup-thinkpad.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m leica_tracker %*
if errorlevel 1 (
  echo Tracker failed. Read the error above and check the camera index and Webcam mode.
  pause
  exit /b 1
)
