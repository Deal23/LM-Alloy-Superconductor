@echo off
chcp 65001 >nul
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo Run scripts\install.bat first.
    exit /b 1
)
set PYTHONDONTWRITEBYTECODE=1
.venv\Scripts\python.exe -B run.py %*
