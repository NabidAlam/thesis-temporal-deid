# Troubleshooting Guide

This guide helps resolve common issues when setting up and running MaskAnyone-Temporal.

## Installation Issues

### 1. ImportError: No module named 'cv2'

**Problem**: OpenCV is not installed or not accessible.

**Solutions**:
```bash
# Option 1: Install via pip
pip install opencv-python

# Option 2: Install via conda
conda install opencv

# Option 3: If you get permission errors
pip install --user opencv-python
```

### 2. CUDA/GPU Issues

**Problem**: PyTorch can't access GPU or CUDA is not available.

**Solutions**:
```bash
# Check CUDA availability
python test_cuda.py

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Python Version Compatibility

**Problem**: Some packages don't support your Python version.

**Solutions**:
```bash
# Check Python version
python --version

# Recommended: Use Python 3.8-3.10
# If using Python 3.11+, some packages may need manual installation

# Create a new environment with specific Python version
conda create -n thesis python=3.9
conda activate thesis
```

### 4. Memory Issues

**Problem**: Out of memory errors during model loading or inference.

**Solutions**:
```bash
# Reduce batch size in configs
# Use smaller model variants (tiny, small instead of large)
# Enable gradient checkpointing
# Use CPU if GPU memory is insufficient
```

## Runtime Issues

### 1. Missing Model Checkpoints

**Problem**: Scripts fail because model files are not found.

**Solutions**:
```bash
# Check if checkpoints exist
ls samurai/sam2/checkpoints/
ls tspsam/model_checkpoint/

# Download required models (see README for links)
# Update paths in configuration files
```

### 2. Path Issues

**Problem**: Scripts can't find input/output directories.

**Solutions**:
```bash
# Use absolute paths
# Check directory structure
# Ensure paths use correct separators for your OS
# Windows: Use \\ or /, Linux/Mac: Use /
```

### 3. Permission Errors

**Problem**: Can't write to output directories or create files.

**Solutions**:
```bash
# Check directory permissions
# Run as administrator (Windows)
# Use sudo (Linux/Mac)
# Change output directory to user-writable location
```

## Performance Issues

### 1. Slow Processing

**Problem**: Models take too long to process videos.

**Solutions**:
```bash
# Use smaller model variants
# Reduce input resolution
# Process fewer frames
# Use GPU acceleration
# Enable mixed precision (torch.autocast)
```

### 2. High Memory Usage

**Problem**: Scripts consume too much RAM/VRAM.

**Solutions**:
```bash
# Process videos in chunks
# Use frame skipping
# Enable gradient checkpointing
# Use CPU offloading
# Reduce batch size
```

## Dataset Issues

### 1. DAVIS Dataset Not Found

**Problem**: Scripts can't locate DAVIS-2017 dataset.

**Solutions**:
```bash
# Download DAVIS-2017 from official website
# Check directory structure matches expected layout
# Update paths in configuration files
# Use relative paths from project root
```

### 2. Missing Bounding Box Files

**Problem**: SAMURAI can't find bounding box annotations.

**Solutions**:
```bash
# Check bbox files exist in input/davis2017/bboxes/
# Ensure file naming matches expected format
# Generate bboxes using generate_bbox.py if needed
```

## Common Error Messages and Solutions

### "ModuleNotFoundError: No module named 'X'"
```bash
pip install X
# or
conda install X
```

### "CUDA out of memory"
```bash
# Reduce batch size
# Use smaller models
# Process shorter sequences
# Enable memory optimization
```

### "FileNotFoundError: [Errno 2] No such file or directory"
```bash
# Check file paths
# Use absolute paths
# Verify file existence
# Check file permissions
```

### "RuntimeError: Expected all tensors to be on the same device"
```bash
# Ensure all tensors are on same device (CPU or GPU)
# Use .to(device) consistently
# Check model and input device placement
```

## Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look for error messages in terminal output
2. **Verify installation**: Run `python test_cuda.py` and `pip list`
3. **Check versions**: Ensure package versions are compatible
4. **Search issues**: Check GitHub issues for similar problems
5. **Create minimal example**: Reproduce issue with minimal code
6. **Provide details**: Include error messages, Python version, OS, etc.

## Environment Verification

Run this to verify your setup:
```bash
python -c "
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
import cv2
print(f'OpenCV: {cv2.__version__}')
import numpy as np
print(f'NumPy: {np.__version__}')
"
```

## Quick Fix Commands

```bash
# Complete reinstall
pip uninstall -y torch torchvision torchaudio opencv-python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python

# Update all packages
pip install --upgrade -r requirements.txt

# Clean environment and reinstall
conda env remove -n thesis-temporal-deid
conda env create -f environment.yml
```
