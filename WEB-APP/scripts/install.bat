@echo off
chcp 65001 >nul
cd /d "%~dp0.."
python -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required'"
if errorlevel 1 exit /b 1
python -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo Installed. Run scripts\start.bat
echo Optional Excel: .venv\Scripts\python.exe -m pip install "openpyxl>=3.1.0"
