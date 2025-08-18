@echo off
echo ========================================
echo Installing MaskAnyone-Temporal Dependencies
echo ========================================
echo.

echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    echo.
    echo Trying alternative installation method...
    pip install --upgrade pip
    pip install -r requirements.txt --no-deps
    pip install numpy opencv-python pillow tqdm torch torchvision torchaudio
    pip install scikit-image imageio pandas matplotlib PyYAML
    pip install fastapi uvicorn pathlib2 requests
)

echo.
echo Testing installation...
python test_cuda.py
if %errorlevel% neq 0 (
    echo WARNING: CUDA test failed, but basic installation may still work
)

echo.
echo ========================================
echo Installation completed!
echo ========================================
echo.
echo To activate the environment in the future:
echo   venv\Scripts\activate.bat
echo.
echo To run the project:
echo   python evaluation/davis_baseline_eval.py --help
echo.
pause
