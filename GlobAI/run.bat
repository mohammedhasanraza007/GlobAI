@echo off
setlocal

echo ================================
echo   GlobAI Runtime Launcher
echo ================================

:: Use current directory instead of hardcoded E: drive
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "VENV_DIR=%PROJECT_DIR%\nexarag_env"
set "RUNTIME_PY=%PROJECT_DIR%\runtime\pythonw.exe"

:: Detect which python to use
set "TARGET_PY="
if exist "%RUNTIME_PY%" (
    set "TARGET_PY=%RUNTIME_PY%"
) else if exist "%VENV_DIR%\Scripts\pythonw.exe" (
    set "TARGET_PY=%VENV_DIR%\Scripts\pythonw.exe"
)

if not defined TARGET_PY (
    echo [ERROR] Runtime not found. 
    echo         Please run build.bat first.
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"
start "" "%TARGET_PY%" ui\desktop_app.py

exit
