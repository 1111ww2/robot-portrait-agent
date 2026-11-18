@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0vision_mvp\scripts\start-canonical-ipad.ps1" -RobotModel ur10e -RuntimeRoot "%~dp0." %*
exit /b %ERRORLEVEL%
