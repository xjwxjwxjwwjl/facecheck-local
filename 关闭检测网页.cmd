@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "local_web\launch.py" --stop
if errorlevel 1 pause
