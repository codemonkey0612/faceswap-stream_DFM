@echo off
::
:: Phase 3 - DeepFaceLab SAEHD training setup (streaming PC)
:: ==========================================================
:: Sets up DeepFaceLab workspace alongside this project and
:: copies DST + SRC frames into DFL's expected directory layout.
::
:: RUN THIS ON THE STREAMING PC (RTX 5080) after:
::   - Phase 2a complete: training_data\dst_augmented\ has frames
::   - Phase 2b complete: training_data\src_frames\ has frames
::
:: DeepFaceLab is a self-contained Windows package (no pip).
:: Download the latest release BEFORE running this script:
::   https://github.com/iperov/DeepFaceLab/releases
::   -> DeepFaceLab_NVIDIA_RTX_build.zip  (or latest CUDA 12 build)
::   -> Extract to: F:\DeepFaceLab\
::   -> Verify:     F:\DeepFaceLab\DeepFaceLab.exe exists
::
:: After this script: run the DFL GUI or batch scripts inside
::   F:\DeepFaceLab\workspace\
::

setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
for %%i in ("%PROJECT_DIR%") do set "PROJECT_DIR=%%~fi"

set "DFL_DIR=F:\DeepFaceLab"
set "DFL_WORKSPACE=%DFL_DIR%\workspace"
set "DFL_DST=%DFL_WORKSPACE%\data_dst"
set "DFL_SRC=%DFL_WORKSPACE%\data_src"

set "SRC_FRAMES=%PROJECT_DIR%\training_data\src_frames"
set "DST_FRAMES=%PROJECT_DIR%\training_data\dst_augmented"
set "TRAINED_MODEL=%DFL_WORKSPACE%\model"

echo.
echo Phase 3 - DFL SAEHD training setup
echo ====================================
echo DFL dir   : %DFL_DIR%
echo Project   : %PROJECT_DIR%
echo.

:: ----------------------------------------------------------------
:: 1. Verify DFL is installed
:: ----------------------------------------------------------------
if not exist "%DFL_DIR%\DeepFaceLab.exe" (
    echo ERROR: DeepFaceLab not found at %DFL_DIR%\DeepFaceLab.exe
    echo.
    echo Download the latest NVIDIA RTX build from:
    echo   https://github.com/iperov/DeepFaceLab/releases
    echo Extract it to F:\DeepFaceLab\ then re-run this script.
    exit /b 1
)
echo [1/4] DeepFaceLab found.

:: ----------------------------------------------------------------
:: 2. Verify source frames exist
:: ----------------------------------------------------------------
echo.
echo [2/4] Checking frame counts...

set DST_COUNT=0
for %%f in ("%DST_FRAMES%\*.jpg") do set /a DST_COUNT+=1
echo   DST frames (dst_augmented) : %DST_COUNT%

set SRC_COUNT=0
for %%f in ("%SRC_FRAMES%\*.jpg") do set /a SRC_COUNT+=1
echo   SRC frames (src_frames)    : %SRC_COUNT%

if %DST_COUNT% LSS 1000 (
    echo WARNING: DST frame count is low ^(%DST_COUNT%^). Run Phase 2a first.
)
if %SRC_COUNT% LSS 5000 (
    echo WARNING: SRC frame count is low ^(%SRC_COUNT%^). Target is 5000+.
    echo   Record Momo's face and run: .venv\Scripts\python.exe scripts\extract_src_frames.py
)

:: ----------------------------------------------------------------
:: 3. Create DFL workspace and copy frames
:: ----------------------------------------------------------------
echo.
echo [3/4] Setting up DFL workspace...

mkdir "%DFL_DST%" 2>nul
mkdir "%DFL_SRC%" 2>nul
mkdir "%TRAINED_MODEL%" 2>nul

echo   Copying DST frames to %DFL_DST% ...
xcopy /Y /Q "%DST_FRAMES%\*.jpg" "%DFL_DST%\"
if errorlevel 1 (
    echo ERROR: Failed to copy DST frames.
    exit /b 1
)

echo   Copying SRC frames to %DFL_SRC% ...
xcopy /Y /Q "%SRC_FRAMES%\*.jpg" "%DFL_SRC%\"
if errorlevel 1 (
    echo ERROR: Failed to copy SRC frames.
    exit /b 1
)

echo   Done.

:: ----------------------------------------------------------------
:: 4. Print training instructions
:: ----------------------------------------------------------------
echo.
echo [4/4] Ready to train!
echo.
echo ================================================================
echo  TRAINING INSTRUCTIONS
echo ================================================================
echo.
echo Open DeepFaceLab and train SAEHD with these recommended settings:
echo.
echo   Resolution  : 192  (or 256 if VRAM allows)
echo   Batch size  : 8    (increase if RTX 5080 allows)
echo   Encoder dim : 64
echo   Decoder dim : 64
echo   Train until : Loss G ^< 0.03  (both src and dst sides)
echo   Eyes prio   : ON  (helps with eye alignment in swaps)
echo   Mouth prio  : ON  (critical for speech animation)
echo.
echo Run DFL training:
echo   %DFL_DIR%\  -- use DFL's own "4) train SAEHD.bat"
echo.
echo After training converges (loss ^< 0.03):
echo   Use DFL's "6) export DFM.bat"
echo   Copy the exported .dfm to:
echo     %PROJECT_DIR%\models\momo.dfm
echo.
echo Then update config:
echo   config\default.yaml  ->  swap.dfm_path: models/momo.dfm
echo.
echo ================================================================
endlocal
