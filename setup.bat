@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  faceswap-stream_DFM - Windows environment setup
echo ============================================================

:: --- Resolve a REAL python.exe (not a pyenv/WindowsApps shim) ---
:: Strategy:
::   1. If pyenv is available, use `pyenv which python` (most reliable on this box).
::   2. Else try py launcher: `py -3.11` / `py -3.10` / `py -3.12`.
::   3. Else try common installer paths.
::   4. Last resort: first `python.exe` on PATH.

set "PYEXE="

:: --- 1. pyenv-win ---
where pyenv >nul 2>&1
if not errorlevel 1 (
    echo [INFO] pyenv detected, resolving real python.exe ...
    for /f "delims=" %%A in ('pyenv which python 2^>nul') do (
        set "PYEXE=%%A"
    )
    if defined PYEXE (
        echo [INFO] pyenv provides: !PYEXE!
    )
)

:: --- 2. py launcher ---
if not defined PYEXE (
    for %%V in (3.11 3.10 3.12) do (
        if not defined PYEXE (
            py -%%V --version >nul 2>&1
            if not errorlevel 1 (
                for /f "delims=" %%A in ('py -%%V -c "import sys; print(sys.executable)"') do (
                    set "PYEXE=%%A"
                )
            )
        )
    )
    if defined PYEXE echo [INFO] py launcher provides: !PYEXE!
)

:: --- 3. Common install paths ---
if not defined PYEXE (
    for %%P in (
        "C:\Python311\python.exe"
        "C:\Python310\python.exe"
        "C:\Python312\python.exe"
        "C:\Program Files\Python311\python.exe"
        "C:\Program Files\Python310\python.exe"
        "C:\Program Files\Python312\python.exe"
    ) do (
        if exist "%%~P" (
            if not defined PYEXE set "PYEXE=%%~P"
        )
    )
)

:: --- 4. Fallback: first non-shim python.exe on PATH ---
if not defined PYEXE (
    for /f "delims=" %%A in ('where python.exe 2^>nul') do (
        if not defined PYEXE (
            echo %%A | findstr /i "shims WindowsApps" >nul
            if errorlevel 1 set "PYEXE=%%A"
        )
    )
)

if not defined PYEXE (
    echo [ERROR] Could not resolve a real python.exe.
    echo Install Python 3.10 / 3.11 / 3.12 from https://www.python.org/downloads/
    exit /b 1
)

:: --- Verify it's actually a .exe, not a .bat shim ---
if /i not "%PYEXE:~-4%"==".exe" (
    echo [ERROR] Resolved Python is not a .exe: %PYEXE%
    echo This is likely a shim. Aborting to avoid hang.
    exit /b 1
)

if not exist "%PYEXE%" (
    echo [ERROR] Resolved Python does not exist: %PYEXE%
    exit /b 1
)

echo.
echo Using Python: %PYEXE%
"%PYEXE%" --version
if errorlevel 1 (
    echo [ERROR] Python exited non-zero.
    exit /b 1
)

:: --- Create venv ---
if not exist ".venv" (
    echo.
    echo [1/3] Creating virtual environment in .venv ...
    "%PYEXE%" -m venv .venv
    if errorlevel 1 (
        echo [ERROR] venv creation failed.
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment .venv already exists - skipping creation.
)

set "VENV_PY=%CD%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [ERROR] Expected %VENV_PY% not found.
    exit /b 1
)

echo.
echo [2/3] Upgrading pip + wheel + setuptools inside venv ...
"%VENV_PY%" -m pip install --upgrade pip wheel setuptools
if errorlevel 1 (
    echo [ERROR] pip upgrade failed.
    exit /b 1
)

echo.
echo [3/3] Installing requirements into venv ...
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] requirements installation failed.
    exit /b 1
)

echo.
echo Verifying install target ...
"%VENV_PY%" -c "import sys; print('  python exe :', sys.executable); print('  version    :', sys.version.split()[0])"
"%VENV_PY%" -m pip list --format=columns | findstr /i "onnxruntime opencv pyvirtualcam insightface mediapipe structlog"

echo.
echo ============================================================
echo  Setup complete.
echo.
echo  Next steps:
echo    1. Activate venv:    .venv\Scripts\activate
echo    2. Run tests:        pytest -v
echo    3. Run pipeline:     python -m src.main --profile live
echo ============================================================

endlocal
