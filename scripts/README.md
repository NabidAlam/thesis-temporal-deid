# Scripts Directory

This directory contains utility and analysis scripts for the thesis project.

## Scripts Overview

### **Utility Scripts**
- **`cleanup_test_output.py`** - Utility for cleaning up test output directories and temporary files
- **`show_table_data.py`** - Utility for displaying and analyzing table data from experiments

### **Analysis Scripts**
- **`extract_and_plot_wandb_data.py`** - Script for extracting and plotting data from Weights & Biases (WANDB) experiments
- **`generate_baseline_report.py`** - Script for generating comprehensive baseline analysis reports
- **`video_processing_simulation.py`** - Script for simulating video processing scenarios and performance analysis

## Usage

### Running Scripts
All scripts can be run from the project root directory:

```bash
# Clean up test outputs
python scripts/cleanup_test_output.py

# Show table data
python scripts/show_table_data.py

# Extract and plot WANDB data
python scripts/extract_and_plot_wandb_data.py

# Generate baseline report
python scripts/generate_baseline_report.py

# Run video processing simulation
python scripts/video_processing_simulation.py
```

### Script-Specific Usage

#### **cleanup_test_output.py**
```bash
python scripts/cleanup_test_output.py [--output_dir OUTPUT_DIR] [--dry_run]
```

#### **extract_and_plot_wandb_data.py**
```bash
python scripts/extract_and_plot_wandb_data.py [--project PROJECT_NAME] [--output_dir OUTPUT_DIR]
```

#### **generate_baseline_report.py**
```bash
python scripts/generate_baseline_report.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]
```

#### **show_table_data.py**
```bash
python scripts/show_table_data.py [--data_file DATA_FILE] [--format FORMAT]
```

#### **video_processing_simulation.py**
```bash
python scripts/video_processing_simulation.py [--scenarios SCENARIOS] [--output_dir OUTPUT_DIR]
```

## Requirements

Make sure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

And activate the conda environment:
```bash
conda activate marbeit
```

## Notes

- These scripts are utility tools and analysis helpers
- They are separate from the core system files (which remain in the root directory)
- Each script is designed to be run independently
- Check individual script files for specific command-line arguments and options
