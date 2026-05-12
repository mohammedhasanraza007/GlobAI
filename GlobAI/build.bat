@echo off
setlocal EnableDelayedExpansion

echo ============================================
echo   GlobAI Environment Builder (SMART)
echo ============================================

:: Use current directory instead of hardcoded drives
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

set "VENV_DIR=%PROJECT_DIR%\nexarag_env"

cd /d "%PROJECT_DIR%"

echo [0/5] Checking portable runtime...

:: -------------------------------------------------
:: AUTO RUN RUNTIME SETUP IF MISSING
:: -------------------------------------------------

if not exist "%PROJECT_DIR%\runtime\python.exe" (
    echo [INFO] Runtime missing.
    echo [INFO] Running setup_runtime.ps1...

    powershell -ExecutionPolicy Bypass -File "%PROJECT_DIR%\setup_runtime.ps1"

    if not exist "%PROJECT_DIR%\runtime\python.exe" (
        echo [ERROR] Runtime installation failed.
        pause
        exit /b 1
    )
)

echo [1/5] Searching for Python...

set PYTHON_EXE=
set IS_PORTABLE=0

:: -------------------------------------------------
:: PORTABLE RUNTIME FIRST
:: -------------------------------------------------

if exist "%PROJECT_DIR%\runtime\python.exe" (
    set "PYTHON_EXE=%PROJECT_DIR%\runtime\python.exe"
    set IS_PORTABLE=1
    echo [OK] Found portable runtime.
)

:: -------------------------------------------------
:: COMMON INSTALL LOCATIONS
:: -------------------------------------------------

if not defined PYTHON_EXE if exist "C:\Python310\python.exe" set "PYTHON_EXE=C:\Python310\python.exe"
if not defined PYTHON_EXE if exist "C:\Python311\python.exe" set "PYTHON_EXE=C:\Python311\python.exe"

if not defined PYTHON_EXE if exist "D:\Python310\python.exe" set "PYTHON_EXE=D:\Python310\python.exe"
if not defined PYTHON_EXE if exist "D:\Python311\python.exe" set "PYTHON_EXE=D:\Python311\python.exe"

if not defined PYTHON_EXE if exist "E:\Python310\python.exe" set "PYTHON_EXE=E:\Python310\python.exe"
if not defined PYTHON_EXE if exist "E:\Python311\python.exe" set "PYTHON_EXE=E:\Python311\python.exe"

:: -------------------------------------------------
:: SYSTEM PATH FALLBACK
:: -------------------------------------------------

if not defined PYTHON_EXE (
    where python >nul 2>nul
    if %errorlevel% equ 0 set "PYTHON_EXE=python"
)

:: -------------------------------------------------
:: FAILURE
:: -------------------------------------------------

if not defined PYTHON_EXE (
    echo [ERROR] Python not found.
    echo Please install Python 3.10 or 3.11.
    pause
    exit /b 1
)

echo [OK] Using Python: %PYTHON_EXE%

:: -------------------------------------------------
:: PORTABLE MODE (NO VENV)
:: -------------------------------------------------

if "%IS_PORTABLE%"=="1" (
    echo [2/5] Portable runtime detected.
    echo [INFO] Embedded Python does not support venv.
    echo [INFO] Using runtime directly...
    set "ENV_PYTHON=%PYTHON_EXE%"
    goto INSTALL_DEPS
)

:: -------------------------------------------------
:: NORMAL VENV MODE
:: -------------------------------------------------

if not exist "%VENV_DIR%" (
    echo [2/5] Creating virtual environment...
    "%PYTHON_EXE%" -m venv "%VENV_DIR%"

    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [2/5] Virtual environment already exists.
)

set "ENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

:: -------------------------------------------------
:: INSTALL DEPENDENCIES
:: -------------------------------------------------

:INSTALL_DEPS

echo [3/5] Installing dependencies...

"%ENV_PYTHON%" -m pip install --upgrade pip
"%ENV_PYTHON%" -m pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

:: -------------------------------------------------
:: DOWNLOAD MODELS
:: -------------------------------------------------

echo [4/5] Downloading models...

"%ENV_PYTHON%" scripts\download_models.py

if errorlevel 1 (
    echo [ERROR] Model download failed.
    echo Check internet connection or model script.
    pause
    exit /b 1
)

:: -------------------------------------------------
:: COMPLETE
:: -------------------------------------------------

echo.
echo ============================================
echo   Environment Ready
echo ============================================
echo.

echo [5/5] Launching GlobAI...

call run.bat

pause
exit /b 0