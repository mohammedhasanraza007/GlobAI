@echo off
setlocal

echo ================================
echo   GlobAI Debug Runtime
echo ================================

:: Use current directory instead of hardcoded E: drive
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "VENV_DIR=%PROJECT_DIR%\nexarag_env"
set "RUNTIME_PY=%PROJECT_DIR%\runtime\python.exe"

:: Detect which python to use
set "TARGET_PY="
set "IS_VENV=0"
if exist "%RUNTIME_PY%" (
    set "TARGET_PY=%RUNTIME_PY%"
) else if exist "%VENV_DIR%\Scripts\python.exe" (
    set "TARGET_PY=%VENV_DIR%\Scripts\python.exe"
    set IS_VENV=1
)

if not defined TARGET_PY (
    echo [ERROR] Runtime not found. 
    echo         Run build.bat first.
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"
if "%IS_VENV%"=="1" call "%VENV_DIR%\Scripts\activate.bat"
"%TARGET_PY%" ui\desktop_app.py

echo.
echo [DEBUG SESSION ENDED]
pause