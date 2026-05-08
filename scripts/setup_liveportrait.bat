@echo off
::
:: LivePortrait setup - Phase 2a DST augmentation (streaming PC)
:: =============================================================
:: Clones KlingAIResearch/LivePortrait and creates a dedicated conda
:: environment with the correct PyTorch + CUDA version.
::
:: RUN THIS ON THE STREAMING PC (RTX 5080, CUDA 12.6+).
:: Dev PC (GTX 1050 / 4 GB) is too small for comfortable inference.
::
:: Prerequisites (install before running this script):
::   Git         https://git-scm.com/download/win
::   Miniconda   https://docs.conda.io/en/latest/miniconda.html
::   CUDA 12.6+  https://developer.nvidia.com/cuda-downloads
::               (RTX 5080 / Blackwell requires >= 12.6)
::
:: After setup, run from the PROJECT directory:
::   conda activate faceswap
::   python scripts\animate_dst_liveportrait.py
::

setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
set "LP_BASE=%PROJECT_DIR%\..\LivePortrait_env"
set "LP_REPO=%LP_BASE%\repo"
set "LP_ENV=liveportrait"

for %%i in ("%LP_BASE%")     do set "LP_BASE=%%~fi"
for %%i in ("%LP_REPO%")     do set "LP_REPO=%%~fi"
for %%i in ("%PROJECT_DIR%") do set "PROJECT_DIR=%%~fi"

echo.
echo LivePortrait setup (conda)
echo ==========================
echo Repo     : %LP_REPO%
echo Conda env: %LP_ENV%
echo.

:: ----------------------------------------------------------------
:: 0. Verify conda is available
:: ----------------------------------------------------------------
where conda >nul 2>&1
if errorlevel 1 (
    echo ERROR: conda not found in PATH.
    echo Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo Then reopen this terminal.
    exit /b 1
)

:: ----------------------------------------------------------------
:: 1. Clone (or update) LivePortrait
:: ----------------------------------------------------------------
if exist "%LP_REPO%\.git" (
    echo [1/5] LivePortrait repo already present - pulling latest...
    git -C "%LP_REPO%" pull --ff-only
) else (
    echo [1/5] Cloning KlingAIResearch/LivePortrait...
    mkdir "%LP_BASE%" 2>nul
    git clone https://github.com/KlingAIResearch/LivePortrait.git "%LP_REPO%"
    if errorlevel 1 (
        echo ERROR: git clone failed. Check network / git installation.
        exit /b 1
    )
)

:: ----------------------------------------------------------------
:: 2. Detect GPU
:: ----------------------------------------------------------------
echo.
echo [2/5] Detecting GPU...
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>nul
if errorlevel 1 (
    echo ERROR: nvidia-smi not found. Ensure NVIDIA drivers are installed.
    exit /b 1
)

set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
echo PyTorch wheel index: %TORCH_INDEX%

:: ----------------------------------------------------------------
:: 3. Create conda environment
:: ----------------------------------------------------------------
echo.
echo [3/5] Creating conda environment '%LP_ENV%'...
conda env list | findstr /C:"%LP_ENV%" >nul 2>&1
if not errorlevel 1 (
    echo   Environment '%LP_ENV%' already exists - skipping creation.
) else (
    conda create -n %LP_ENV% python=3.10 -y
    if errorlevel 1 (
        echo ERROR: conda env creation failed.
        exit /b 1
    )
)

:: ----------------------------------------------------------------
:: 4. Install PyTorch + LivePortrait requirements
:: ----------------------------------------------------------------
echo.
echo [4/5] Installing PyTorch 2.6+ with CUDA 12.6...
conda run -n %LP_ENV% pip install torch torchvision torchaudio --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo ERROR: PyTorch install failed.
    exit /b 1
)

echo Installing LivePortrait requirements...
conda run -n %LP_ENV% pip install -r "%LP_REPO%\requirements.txt"
if errorlevel 1 (
    echo ERROR: LivePortrait requirements install failed.
    exit /b 1
)

conda run -n %LP_ENV% pip install huggingface_hub imageio-ffmpeg
if errorlevel 1 (
    echo WARNING: huggingface_hub/imageio-ffmpeg install failed - continuing.
)

:: ----------------------------------------------------------------
:: 5. Download model weights from HuggingFace (~3.6 GB)
:: ----------------------------------------------------------------
echo.
echo [5/5] Downloading LivePortrait model weights (~3.6 GB)...
echo This may take several minutes on the first run.

conda run -n %LP_ENV% python -c "from huggingface_hub import snapshot_download; import os; w=os.path.join(r'%LP_REPO%','pretrained_weights'); snapshot_download(repo_id='KlingTeam/LivePortrait',local_dir=w,ignore_patterns=['*.md','*.txt']); print('Weights saved to:',w)"
if errorlevel 1 (
    echo.
    echo ERROR: Weight download failed.
    echo Manual alternative:
    echo   1. Visit https://huggingface.co/KlingTeam/LivePortrait
    echo   2. Download all files into %LP_REPO%\pretrained_weights\
    exit /b 1
)

:: ----------------------------------------------------------------
:: Verify
:: ----------------------------------------------------------------
echo.
echo Verifying installation...
conda run -n %LP_ENV% python -c "import torch; print('torch      :', torch.__version__); print('CUDA avail :', torch.cuda.is_available()); g=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'; print('GPU        :', g)"

echo.
echo ================================================================
echo  Setup complete!
echo ================================================================
echo.
echo NEXT STEP - get a driving video:
echo   Any frontal talking-head clip (30 sec - 5 min) works.
echo   The driver's identity does NOT transfer, only motion does.
echo.
echo   Option A - use LP bundled sample (already downloaded):
echo     The animate script finds it automatically.
echo.
echo   Option B - record yourself talking for 2-3 minutes:
echo     Save as: %PROJECT_DIR%\training_data\driving\driving.mp4
echo.
echo   Option C - any free stock video (Pexels, Pixabay "woman talking"):
echo     Save as: %PROJECT_DIR%\training_data\driving\driving.mp4
echo.
echo Then run from the PROJECT directory:
echo.
echo   conda activate faceswap
echo   python scripts\animate_dst_liveportrait.py
echo.
echo   Or with a custom driver:
echo   python scripts\animate_dst_liveportrait.py
echo       --driving training_data\driving\driving.mp4
echo.
echo ================================================================
endlocal
