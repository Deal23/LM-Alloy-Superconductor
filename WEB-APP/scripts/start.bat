@echo off
chcp 65001 >nul
cd /d "%~dp0.."
for /f "usebackq delims=" %%i in (`python scripts\venv_path.py`) do set "VENV_DIR=%%i"
if not defined VENV_DIR exit /b 1
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Run scripts\install.bat first.
    exit /b 1
)
set PYTHONDONTWRITEBYTECODE=1
"%VENV_DIR%\Scripts\python.exe" -B run.py %*
