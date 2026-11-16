@echo off
chcp 65001 >nul
cd /d "%~dp0.."
python -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required'"
if errorlevel 1 exit /b 1
for /f "usebackq delims=" %%i in (`python scripts\venv_path.py`) do set "VENV_DIR=%%i"
if not defined VENV_DIR exit /b 1
python -m venv "%VENV_DIR%"
if errorlevel 1 exit /b 1
"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
"%VENV_DIR%\Scripts\python.exe" -B scripts\register_lmsc.py
if errorlevel 1 echo Warning: lmsc registration failed. Use scripts\start.bat instead.
echo Installed. Run scripts\start.bat
echo Environment: %VENV_DIR%
echo Optional Excel: "%VENV_DIR%\Scripts\python.exe" -m pip install "openpyxl>=3.1.0"
