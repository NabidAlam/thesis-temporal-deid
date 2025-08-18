#!/bin/bash

echo "========================================"
echo "Installing MaskAnyone-Temporal Dependencies"
echo "========================================"
echo

echo "Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python3 is not installed or not in PATH"
    exit 1
fi

echo
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

echo
echo "Upgrading pip..."
python -m pip install --upgrade pip

echo
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    echo
    echo "Trying alternative installation method..."
    pip install --upgrade pip
    pip install -r requirements.txt --no-deps
    pip install numpy opencv-python pillow tqdm torch torchvision torchaudio
    pip install scikit-image imageio pandas matplotlib PyYAML
    pip install fastapi uvicorn pathlib2 requests
fi

echo
echo "Testing installation..."
python test_cuda.py
if [ $? -ne 0 ]; then
    echo "WARNING: CUDA test failed, but basic installation may still work"
fi

echo
echo "========================================"
echo "Installation completed!"
echo "========================================"
echo
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo
echo "To run the project:"
echo "  python evaluation/davis_baseline_eval.py --help"
echo
