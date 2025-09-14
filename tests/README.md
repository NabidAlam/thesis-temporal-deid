# Test Organization

This directory contains all test files organized by type:

## Directory Structure

- **`unit/`** - Unit tests for individual components
  - `test_config_loading.py` - Tests for configuration loading functionality
  - `test_opencv.py` - Tests for OpenCV functionality
  - `test_cuda.py` - Tests for CUDA availability and functionality

- **`baseline/`** - Tests for baseline model implementations
  - `test_run_samurai_baseline.py` - Tests for SAMURAI baseline
  - `test_run_tsp_sam_baseline.py` - Tests for TSP-SAM baseline

- **`pipeline/`** - Tests for pipeline integration
  - `test_integrated_temporal_pipeline_hybrid.py` - Tests for hybrid pipeline
  - `test_chunked_processing.py` - Tests for chunked video processing
  - `test_integrated_pipeline_components.py` - Tests for pipeline components
  - `test_integrated_pipeline.py` - Tests for integrated pipeline
  - `test_intelligent_fusion.py` - Tests for intelligent fusion mechanism

- **`integration/`** - Integration tests
  - `test_comparison_script.py` - Tests for comparison scripts
  - `test_temporal_demo.py` - Tests for temporal demo functionality
  - `test_temporal_consistency.py` - Tests for temporal consistency
  - `test_temporal_integration.py` - Tests for temporal integration

## Running Tests

To run all tests:
```bash
python -m pytest tests/
```

To run specific test categories:
```bash
# Unit tests
python -m pytest tests/unit/

# Baseline tests
python -m pytest tests/baseline/

# Pipeline tests
python -m pytest tests/pipeline/

# Integration tests
python -m pytest tests/integration/
```

## Test Requirements

Make sure you have the required dependencies installed:
```bash
pip install pytest
```

And activate the conda environment:
```bash
conda activate marbeit
```
