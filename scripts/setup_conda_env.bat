@echo off
::
:: Main project conda environment setup
:: =====================================
:: Creates the 'faceswap' conda environment and installs all
:: project dependencies from requirements.txt.
::
:: Prerequisites:
::   Miniconda  https://docs.conda.io/en/latest/miniconda.html
::   CUDA 12.x  (for onnxruntime-gpu)
::
:: Usage (run once from the project root):
::   scripts\setup_conda_env.bat
::
:: After setup:
::   conda activate faceswap
::   python -m src.main
::

setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
for %%i in ("%PROJECT_DIR%") do set "PROJECT_DIR=%%~fi"

set "ENV_NAME=faceswap"

echo.
echo Project conda environment setup
echo ================================
echo Env  : %ENV_NAME%
echo Root : %PROJECT_DIR%
echo.

:: ----------------------------------------------------------------
:: 0. Verify conda
:: ----------------------------------------------------------------
where conda >nul 2>&1
if errorlevel 1 (
    echo ERROR: conda not found in PATH.
    echo Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

:: ----------------------------------------------------------------
:: 1. Create environment
:: ----------------------------------------------------------------
conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [1/3] Conda env '%ENV_NAME%' already exists - skipping creation.
) else (
    echo [1/3] Creating conda env '%ENV_NAME%' with Python 3.10...
    conda create -n %ENV_NAME% python=3.10 -y
    if errorlevel 1 (
        echo ERROR: conda create failed.
        exit /b 1
    )
)

:: ----------------------------------------------------------------
:: 2. Install dependencies
:: ----------------------------------------------------------------
echo.
echo [2/3] Installing project requirements...
conda run -n %ENV_NAME% pip install -r "%PROJECT_DIR%\requirements.txt"
if errorlevel 1 (
    echo ERROR: pip install failed.
    exit /b 1
)

:: ----------------------------------------------------------------
:: 3. Quick verification
:: ----------------------------------------------------------------
echo.
echo [3/3] Verifying key packages...
conda run -n %ENV_NAME% python -c "import cv2, numpy, onnxruntime, mediapipe, pyvirtualcam, yaml, pydantic; print('All core packages OK')"
if errorlevel 1 (
    echo WARNING: Some packages may be missing - check the output above.
) else (
    echo.
    echo ================================================================
    echo  Setup complete!
    echo ================================================================
    echo.
    echo Activate the environment:
    echo   conda activate %ENV_NAME%
    echo.
    echo Run the pipeline:
    echo   python -m src.main
    echo   python -m src.main --profile debug --no-vcam
    echo   python -m src.main --profile live
    echo.
    echo Run scripts:
    echo   python scripts\augment_dst_faces.py
    echo   python scripts\animate_dst_liveportrait.py
    echo   python scripts\extract_src_frames.py
    echo   python scripts\deploy_model.py --dfm path\to\momo.dfm
    echo.
    echo Run tests:
    echo   pytest
    echo ================================================================
)
endlocal
