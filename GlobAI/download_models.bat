@echo off
setlocal

echo ============================================
echo   GlobAI Model Downloader
echo ============================================

:: Use current directory
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "VENV_DIR=%PROJECT_DIR%\nexarag_env"
set "RUNTIME_PY=%PROJECT_DIR%\runtime\python.exe"

:: Detect which python to use
set "TARGET_PY="
if exist "%RUNTIME_PY%" (
    set "TARGET_PY=%RUNTIME_PY%"
) else if exist "%VENV_DIR%\Scripts\python.exe" (
    set "TARGET_PY=%VENV_DIR%\Scripts\python.exe"
)

if not defined TARGET_PY (
    echo [ERROR] Runtime not found. 
    echo         Run build.bat first to set up the environment.
    pause
    exit /b 1
)

echo [OK] Using Python: %TARGET_PY%
echo [INFO] Starting model download...
echo.

"%TARGET_PY%" scripts\download_models.py

if errorlevel 1 (
    echo.
    echo [ERROR] Download failed.
) else (
    echo.
    echo [DONE] All models checked/downloaded.
)

pause
