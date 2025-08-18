# MaskAnyone-Temporal: Temporally-Consistent Privacy Protection for Behavioural-Science Video

This repository contains the official codebase for my Master's thesis at the University of Potsdam. The project builds on the [MaskAnyone](https://github.com/MaskAnyone/MaskAnyone) framework and integrates temporal segmentation techniques via [TSP-SAM](https://github.com/WenjunHui1/TSP-SAM) and [SAMURAI](https://github.com/yangchris11/samurai).

## Project Overview

This thesis addresses the critical challenge of **temporal identity leakage** in video de-identification. While traditional methods hide identities in single frames, behavioral video still leaks who we are through gait signatures, gesture kinetics, and pose-to-pose motion trails.

**MaskAnyone-Temporal (MAT)** integrates two state-of-the-art temporal mechanisms:
- **TSP-SAM**: Frequency-domain motion prompts for temporal coherence
- **SAMURAI**: Kalman-tracked memory for long-range temporal consistency

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

### Option 2: Using conda

1. **Create conda environment:**
```bash
conda env create -f environment.yml
conda activate thesis-temporal-deid
```

## Quick Start

### Test CUDA Availability
```bash
python test_cuda.py
```

### Run SAMURAI on DAVIS Dataset
```bash
cd samurai
python samurai/scripts/demo.py --video_path "input/davis2017/JPEGImages/480p/hike" --txt_path "input/davis2017/bboxes/bbox_hike.txt" --model_path "sam2/checkpoints/sam2.1_hiera_base_plus.pt"
```

### Run TSP-SAM on DAVIS Dataset
```bash
cd tspsam
python main.py --dataset DAVIS --test_path ../input/davis2017/JPEGImages/480p --output_dir ../output/tsp_sam/davis --resume model_checkpoint/best_checkpoint.pth --sequence hike --gpu_ids 0
```

### Evaluate Results
```bash
python evaluation/davis_baseline_eval.py --method samurai --sequences hike dog camel
python evaluation/davis_baseline_eval.py --method tsp-sam --sequences hike dog camel
```

## Project Structure

```
thesis-temporal-deid/
├── samurai/              # SAMURAI implementation
├── tspsam/               # TSP-SAM implementation  
├── temporal/             # Integration modules
├── evaluation/           # Evaluation scripts
├── input/                # DAVIS-2017 dataset
├── output/               # Generated masks
├── configs/              # Configuration files
└── docs/                 # Documentation
```

## Datasets

- **DAVIS-2017**: Video object segmentation benchmark (90 sequences)
- **TED Talks**: Behavioral science videos
- **Team Ten**: Custom interaction dataset
- **Tragic Talkers**: Custom interaction dataset

## Current Results

**SAMURAI Performance:**
- hike: 94.9% IoU, 90.6% temporal consistency
- dog: 96.5% IoU, 78.7% temporal consistency  
- camel: 97.7% IoU, 91.4% temporal consistency

**TSP-SAM Performance:**
- hike: 58.8% IoU, 59.3% temporal consistency
- dog: 77.5% IoU, 35.4% temporal consistency
- camel: 85.7% IoU, 70.6% temporal consistency

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'cv2'**
   ```bash
   pip install opencv-python
   ```

2. **CUDA not available**
   - Check GPU drivers
   - Install PyTorch with CUDA support
   - Run `python test_cuda.py` to verify

3. **Missing model checkpoints**
   - Download required model files
   - Update paths in configuration files

## Contributing

This is a research project for my Master's thesis. For questions or issues, please open an issue on GitHub.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

<!-- ## Citation

If you use this code in your research, please cite:

```bibtex
@misc{maskanyone_temporal_2024,
  title={MaskAnyone-Temporal: Temporally-Consistent Privacy Protection for Behavioural-Science Video},
  author={Your Name},
  year={2024},
  note={Master's Thesis, University of Potsdam}
}
``` -->

