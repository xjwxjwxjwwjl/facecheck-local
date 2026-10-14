@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "local_web\launch.py"
if errorlevel 1 pause
