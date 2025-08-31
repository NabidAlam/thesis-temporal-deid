# MaskAnyone-Temporal: Temporally-Consistent Privacy Protection for Behavioural-Science Video

<!-- This repository contains the official codebase for my Master's thesis at the University of Potsdam.  -->

The project builds on the [MaskAnyone](https://github.com/MaskAnyone/MaskAnyone) framework and integrates temporal segmentation techniques via [TSP-SAM](https://github.com/WenjunHui1/TSP-SAM) and [SAMURAI](https://github.com/yangchris11/samurai).

## Project Overview

This thesis addresses the critical challenge of **temporal identity leakage** in video de-identification. While traditional methods hide identities in single frames, behavioral video still leaks who we are through gait signatures, gesture kinetics, and pose-to-pose motion trails.

**MaskAnyone-Temporal (MAT)** integrates two state-of-the-art temporal mechanisms:
- **TSP-SAM**: Frequency-domain motion prompts for temporal coherence
- **SAMURAI**: Kalman-tracked memory for long-range temporal consistency
- **Enhanced Motion Detection**: Intelligent fusion and fallback mechanisms

<!-- ## Recent Developments (v2.0)

### TED Talk Optimizations
- **Drift threshold optimization**: Reduced from 0.85 to 0.50 for better TED talk handling
- **Quality-based reference storage**: Smart mask caching based on quality metrics
- **Scene type detection**: Automatic adaptation for stage lighting, high contrast, and standard scenes
- **Speaker prioritization**: Centrality, size, and aspect ratio-based person detection
- **Gesture movement handling**: Temporal smoothing for natural motion
- **TED-optimized fusion**: 60% SAMURAI + 40% TSP-SAM for optimal results

### Enhanced Pipeline Features
- **Integrated Temporal Pipeline**: Complete end-to-end processing with intelligent fallbacks
- **Motion-Aware Memory**: Kalman filtering for object tracking across frames
- **Quality-Based Mask Selection**: Automatic selection of best available masks
- **Comprehensive Evaluation Framework**: Multi-metric analysis across 90 DAVIS sequences -->

## Installation

### Option 1: Using pip (Recommended)

1. **Clone the repository:**
```bash
git clone --recursive https://github.com/yourusername/thesis-temporal-deid.git
cd thesis-temporal-deid
```

2. **Create a virtual environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Option 2: Using conda (Recommended for CUDA support)

1. **Create conda environment:**
```bash
conda env create -f environment.yml
conda activate marbeit  # Use 'marbeit' environment for CUDA support
```

## Quick Start

### Test CUDA Availability
```bash
python testing/test_cuda.py
```

### Run Integrated Temporal Pipeline
```bash
python integrated_temporal_pipeline.py --input_path "input/ted/sequence_name" --output_path "output/integrated/sequence_name"
```

### Run Individual Baselines

**SAMURAI on DAVIS Dataset:**
```bash
python run_samurai_baseline.py --sequence hike --output_dir output/samurai/davis
```

**TSP-SAM on DAVIS Dataset:**
```bash
python run_tsp_sam_baseline.py --sequence hike --output_dir output/tsp_sam/davis
```

**TED Talks Processing:**
```bash
python run_ted_talks_baseline.py --input_dir input/ted --output_dir output/ted
```

### Comprehensive Evaluation
```bash
# Run full evaluation across all sequences
python evaluation/run_full_evaluation.py

# Generate comprehensive analysis reports
python evaluation/generate_comprehensive_analysis.py

# Visualize results
python evaluation/visualize_baseline_results.py
```
<!-- 
## Current Results

### DAVIS-2017 Dataset (90 Sequences)

**Overall Performance:**
- **SAMURAI**: Mean IoU 85.4%, Mean Temporal Consistency 69.3%
- **TSP-SAM**: Mean IoU 84.1%, Mean Temporal Consistency 67.7%
- **Performance Gap**: SAMURAI leads by 1.4% IoU and 1.6% temporal consistency

**Key Findings:**
- **89 out of 90 sequences**: SAMURAI outperforms TSP-SAM
- **High correlation (0.997)**: Both models share similar architectural limitations
- **Legitimate failures**: Zero IoU values represent genuine model limitations on challenging sequences

**Top Performing Sequences:**
- **hike**: SAMURAI 94.9% IoU, TSP-SAM 58.8% IoU
- **dog**: SAMURAI 96.5% IoU, TSP-SAM 77.5% IoU  
- **camel**: SAMURAI 97.7% IoU, TSP-SAM 85.7% IoU

### Behavioral Science Videos
- **TED Talks**: Optimized processing with speaker prioritization
- **Team Ten**: Custom interaction dataset support
- **Tragic Talkers**: Custom interaction dataset support -->

## Project Structure

```
thesis-temporal-deid/
├── integrated_temporal_pipeline.py          # Main integrated pipeline
├── enhanced_motion_detection.py             # Motion detection algorithms
├── run_all_sequences_comparison.py          # Comprehensive comparison script
├── samurai_official/                        # SAMURAI implementation
├── tsp_sam_official/                        # TSP-SAM implementation  
├── temporal/                                # Integration modules
├── evaluation/                              # Comprehensive evaluation framework
│   ├── run_full_evaluation.py               # Full evaluation runner
│   ├── generate_comprehensive_analysis.py   # Analysis report generator
│   ├── visualize_baseline_results.py        # Results visualization
│   └── failure_analysis/                    # Failure analysis tools
├── testing/                                 # Testing framework
├── input/                                   # Datasets (DAVIS2017, TED, Team Ten, Tragic Talkers)
├── output/                                  # Generated masks and results
├── fig/                                     # Performance visualization figures
├── configs/                                 # Configuration files
├── docs/                                    # Comprehensive documentation
└── wandb/                                   # Experiment tracking
```

## Evaluation Framework

### Metrics Tracked
- **IoU (Intersection over Union)**: Segmentation accuracy
- **Temporal Consistency**: Frame-to-frame mask stability
- **Performance Distribution**: Statistical analysis across sequences
- **Motion Complexity Analysis**: Performance vs. motion difficulty
- **Failure Analysis**: Detailed breakdown of model limitations

### Visualization Tools
- **Performance Dashboard**: Interactive baseline comparison
- **Sequence Analysis**: Per-sequence performance breakdown
- **Temporal Consistency**: Frame-by-frame stability analysis
- **Motion Complexity**: Performance correlation with motion difficulty

## Advanced Features

### Intelligent Mask Fusion
- **Quality-based selection**: Automatic best mask identification
- **Temporal smoothing**: Frame-to-frame consistency enhancement
- **Fallback mechanisms**: Graceful degradation when primary methods fail

### Motion Detection
- **Kalman filtering**: Predictive object tracking
- **Motion history**: Temporal motion pattern analysis
- **Drift compensation**: Automatic correction for tracking drift

### Scene Adaptation
- **Lighting detection**: Automatic stage lighting adaptation
- **Scene classification**: Optimal parameter selection per scene type
- **Person prioritization**: Speaker-focused processing for TED talks

<!-- ## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **QUICK_REFERENCE.md**: Essential commands and workflows
- **TESTING_README.md**: Testing framework and procedures
- **THESIS_METRICS_GUIDE.md**: Detailed metrics explanation
- **BASELINE_TO_SOLUTION_MAPPING.md**: Problem-solution mapping
- **TED_OPTIMIZATIONS_SUMMARY.md**: TED talk specific optimizations
- **WANDB_FIXES_SUMMARY.md**: Experiment tracking solutions -->

## Testing

### Test Coverage
- **Unit tests**: Individual component testing
- **Integration tests**: Pipeline end-to-end testing
- **Performance tests**: CUDA availability and performance validation
- **Baseline tests**: Individual baseline method validation

### Running Tests
```bash
# Run all tests
python -m pytest testing/

# Test specific components
python test_integrated_temporal_pipeline.py
python test_enhanced_motion_detection.py
```

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

<!-- ## Citation

If you use this code in your research, please cite: -->

<!-- ```bibtex
@misc{maskanyone_temporal_2024,
  title={MaskAnyone-Temporal: Temporally-Consistent Privacy Protection for Behavioural-Science Video},
  author={Your Name},
  year={2024},
  note={Master's Thesis, University of Potsdam}
}
``` -->

---

<!-- **Last Updated**: December 2024  
**Version**: 2.0  
**Status**: Active Development - Comprehensive evaluation complete, TED optimizations implemented -->

