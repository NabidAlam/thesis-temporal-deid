# MaskAnyone-Temporal: A Hybrid Fusion Approach for Robust Video De-identification

This repository contains the official codebase for my Master's thesis in collaboration with Hasso-Plattner-Institut and University of Potsdam.

The project builds on the [MaskAnyone](https://github.com/MaskAnyone/MaskAnyone) framework and integrates temporal segmentation techniques via [TSP-SAM](https://github.com/WenjunHui1/TSP-SAM) and [SAMURAI](https://github.com/yangchris11/samurai).

## Project Overview

This thesis addresses the critical challenge of **temporal identity leakage** in video de-identification. While traditional methods hide identities in single frames, behavioral video still leaks who we are through gait signatures, gesture kinetics, and pose-to-pose motion trails.

<!-- **MaskAnyone-Temporal (MAT)** integrates two state-of-the-art temporal mechanisms:
- **TSP-SAM**: Frequency-domain motion prompts for temporal coherence
- **SAMURAI**: Kalman-tracked memory for long-range temporal consistency
- **Enhanced Motion Detection**: Intelligent fusion and fallback mechanisms -->

## Installation

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended)
- Git with submodule support

### Step 1: Clone the Repository
```bash
git clone --recursive https://github.com/yourusername/thesis-temporal-deid.git
cd thesis-temporal-deid
```

### Step 2: Set Up Environment

**Option 1: Using conda (Recommended for CUDA support)**
```bash
conda env create -f environment.yml
conda activate marbeit
```

**Option 2: Using pip**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python testing/test_cuda.py
```

## Quick Start

### Run Hybrid Pipeline
```bash
python integrated_temporal_pipeline_hybrid.py --input_path "input/ted/sequence_name" --output_path "output/integrated/sequence_name"
```

### Run Individual Baselines

**SAMURAI Baseline:**
```bash
python run_samurai_baseline.py --sequence hike --output_dir output/samurai/davis
```

**TSP-SAM Baseline:**
```bash
python run_tsp_sam_baseline.py --sequence hike --output_dir output/tsp_sam/davis
```

**Resume Processing (Skip Completed Sequences):**
```bash
python run_samurai_baseline.py --sequence skate-park --output_dir output/samurai_all_sequences --skip-completed
python run_tsp_sam_baseline.py --sequence skate-park --output_dir output/tsp_sam_all_sequences --skip-completed
```

### Configuration-Based Processing
```bash
python run_with_config.py --config configs/datasets/ted_talks.yaml
```

### Run All Sequences Comparison
```bash
python run_all_sequences_comparison.py
```
## Project Structure

```
thesis-temporal-deid/
├── integrated_temporal_pipeline_hybrid.py    # Main hybrid pipeline
├── comprehensive_baseline_comparison.py      # Comprehensive comparison script
├── run_all_sequences_comparison.py           # All sequences comparison runner
├── run_hybrid_on_davis.py                    # Hybrid pipeline runner
├── run_with_config.py                        # Configuration-based runner
├── run_samurai_baseline.py                   # SAMURAI baseline runner
├── run_tsp_sam_baseline.py                   # TSP-SAM baseline runner
├── tests/                                    # Organized test files
│   ├── unit/                                 # Unit tests
│   ├── baseline/                             # Baseline model tests
│   ├── pipeline/                             # Pipeline integration tests
│   └── integration/                          # Integration tests
├── scripts/                                  # Utility and analysis scripts
├── evaluation/                               # Evaluation framework
├── envisionbox_integration/                  # GUI integration components
├── samurai_official/                         # SAMURAI implementation
├── tsp_sam_official/                         # TSP-SAM implementation  
├── maskanyone/                               # MaskAnyone implementation
├── input/                                    # Datasets (DAVIS2017, TED, Team Ten, Tragic Talkers)
├── output/                                   # Generated masks and results
├── configs/                                  # Configuration files
│   └── datasets/                             # Dataset-specific configurations
│       ├── default.yaml                      # Default configuration
│       ├── ted_talks.yaml                    # TED talks configuration
│       ├── team_ten.yaml                     # Team Ten configuration
│       └── tragic_talkers.yaml               # Tragic Talkers configuration
├── docs/                                     # Documentation
└── wandb/                                    # Experiment tracking
``` -->
## Key Features

### Skip-Completed Functionality
- **Resume Processing**: Automatically skip sequences that already have output files
- **Progress Tracking**: Know exactly which sequences remain to be processed
- **Efficient Resumption**: Continue from where you left off without duplicating work

### Configuration System
- **Multi-Dataset Support**: Easy configuration for different datasets
- **Parameter Management**: Centralized configuration for all processing parameters
- **Flexible Processing**: Switch between datasets with simple config changes

### Hybrid Pipeline
- **Intelligent Mask Fusion**: Quality-based selection and temporal smoothing
- **Motion Detection**: Kalman filtering and motion history analysis
- **Scene Adaptation**: Automatic lighting detection and person prioritization
- **Memory Management**: Efficient GPU memory handling and chunked processing

## Testing

### Run Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/          # Unit tests
python -m pytest tests/baseline/      # Baseline model tests
python -m pytest tests/pipeline/      # Pipeline integration tests
python -m pytest tests/integration/   # Integration tests
```

## Scripts

Utility and analysis scripts are organized in the `scripts/` directory:

```bash
# Run utility scripts
python scripts/cleanup_test_output.py
python scripts/show_table_data.py
python scripts/generate_baseline_report.py
```

See `scripts/README.md` for detailed usage information.

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'cv2'**
   ```bash
   pip install opencv-python
   ```

2. **CUDA not available**
   - Use the 'marbeit' conda environment: `conda activate marbeit`
   - Check GPU drivers
   - Run `python testing/test_cuda.py` to verify

3. **Missing model checkpoints**
   - Download required model files
   - Update paths in configuration files

4. **Memory issues with large videos**
   - Reduce batch size in configuration
   - Use frame sampling for very long sequences

### Performance Optimization
- **GPU memory**: Monitor with `nvidia-smi`
- **Batch processing**: Use appropriate batch sizes for your hardware
- **Frame sampling**: Skip frames for faster processing when acceptable

## Contributing

This is a research project for my Master's thesis. For questions or issues, please open an issue on GitHub.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---


