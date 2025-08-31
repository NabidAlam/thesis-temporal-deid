#!/usr/bin/env python3
"""
Multi-Sequence Comparison Runner for TSP-SAM vs SAMURAI Baselines
Processes all DAVIS 2017 sequences with both baselines for comprehensive comparison
Organized according to German Master Thesis standards with proper experiment tracking
"""

import os
import sys
import argparse
import subprocess
import time
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
import traceback
import platform
import psutil
import torch

# Global debug flag
DEBUG_MODE = True

def debug_print(message, level="INFO"):
    """Centralized debug printing with timestamps and levels"""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[DEBUG-{level}] {timestamp} - {message}")

def get_system_info():
    """Capture system information for reproducibility"""
    debug_print("Starting system information collection...")
    try:
        system_info = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        }
        debug_print(f"System info collected successfully: {system_info['platform']}, CUDA: {system_info['cuda_available']}")
        return system_info
    except Exception as e:
        debug_print(f"Error collecting system info: {e}", "ERROR")
        return {"timestamp": datetime.now().isoformat(), "error": str(e)}

def create_experiment_structure(base_output_path, experiment_name):
    """Create organized experiment directory structure"""
    debug_print(f"Creating experiment structure: {base_output_path} -> {experiment_name}")
    
    try:
        experiment_dir = Path(base_output_path) / "experiments" / experiment_name
        debug_print(f"Full experiment path: {experiment_dir.absolute()}")
        
        # Create main structure
        directories = [
            "config",
            "config/model_configs", 
            "results/tsp_sam/sequences",
            "results/samurai/sequences",
            "results/comparison/performance_plots",
            "logs",
            "metadata",
            "temp"
        ]
        
        for dir_path in directories:
            full_dir = experiment_dir / dir_path
            full_dir.mkdir(parents=True, exist_ok=True)
            debug_print(f"Created directory: {full_dir}")
        
        debug_print(f"Experiment structure created successfully at: {experiment_dir}")
        return experiment_dir
        
    except Exception as e:
        debug_print(f"Error creating experiment structure: {e}", "ERROR")
        raise e

def save_experiment_config(experiment_dir, args, system_info):
    """Save complete experiment configuration"""
    debug_print(f"Saving experiment configuration to: {experiment_dir}")
    
    try:
        config = {
            "experiment_info": {
                "name": experiment_dir.name,
                "description": "Multi-sequence comparison of TSP-SAM vs SAMURAI baselines",
                "start_time": system_info["timestamp"],
                "input_path": args.input_path,
                "max_frames": args.max_frames,
                "wandb_enabled": args.wandb
            },
            "system_info": system_info,
            "arguments": vars(args)
        }
        
        # Save main config
        config_file = experiment_dir / "config" / "experiment_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        debug_print(f"Main config saved to: {config_file}")
        
        # Save system info separately
        system_info_file = experiment_dir / "metadata" / "system_info.json"
        with open(system_info_file, 'w') as f:
            json.dump(system_info, f, indent=2)
        debug_print(f"System info saved to: {system_info_file}")
        
        # Create reproducibility info
        repro_file = experiment_dir / "metadata" / "reproducibility_info.txt"
        with open(repro_file, 'w') as f:
            f.write(f"Experiment: {experiment_dir.name}\n")
            f.write(f"Date: {system_info['timestamp']}\n")
            f.write(f"Platform: {system_info['platform']}\n")
            f.write(f"Python: {system_info['python_version']}\n")
            f.write(f"CUDA: {system_info['cuda_version']}\n")
            f.write(f"GPU: {', '.join(system_info['gpu_names'])}\n\n")
            f.write("To reproduce this experiment:\n")
            f.write("1. Use the same environment (see environment.yml)\n")
            f.write("2. Run with identical arguments\n")
            f.write("3. Ensure same dataset version\n")
        debug_print(f"Reproducibility info saved to: {repro_file}")
        
    except Exception as e:
        debug_print(f"Error saving experiment config: {e}", "ERROR")
        raise e

def get_available_sequences(input_path):
    """Get all available sequences from DAVIS dataset"""
    debug_print(f"Looking for sequences in: {input_path}")
    
    try:
        sequences_path = Path(input_path) / "JPEGImages" / "480p"
        debug_print(f"Full sequences path: {sequences_path.absolute()}")
        
        if not sequences_path.exists():
            debug_print(f"ERROR: Sequences path not found: {sequences_path}", "ERROR")
            return []
        
        sequences = [d.name for d in sequences_path.iterdir() if d.is_dir()]
        sequences.sort()  # Sort for consistent ordering
        
        debug_print(f"Found {len(sequences)} sequences")
        debug_print(f"First 5 sequences: {sequences[:5]}")
        debug_print(f"Last 5 sequences: {sequences[-5:]}")
        
        return sequences
        
    except Exception as e:
        debug_print(f"Error getting available sequences: {e}", "ERROR")
        return []

def run_baseline_script(script_name, args_dict, sequence_name, max_frames, output_dir, experiment_dir):
    """Run a baseline script with given arguments"""
    debug_print(f"Starting baseline script: {script_name} for sequence: {sequence_name}")
    debug_print(f"Args dict: {args_dict}")
    debug_print(f"Output dir: {output_dir}")
    debug_print(f"Experiment dir: {experiment_dir}")
    
    try:
        # Create sequence-specific output directory
        baseline_name = script_name.replace('.py', '').replace('run_', '').replace('_baseline', '')
        debug_print(f"Baseline name extracted: {baseline_name}")
        
        # For SAMURAI, don't add sequence name to output path (it adds it internally)
        # For TSP-SAM, add sequence name to output path
        if "samurai" in script_name:
            seq_output_dir = output_dir  # SAMURAI will create sequence subdirectory
            debug_print(f"SAMURAI detected, using base output dir: {seq_output_dir}")
        else:
            seq_output_dir = output_dir / sequence_name  # TSP-SAM expects sequence subdirectory
            debug_print(f"TSP-SAM detected, using sequence-specific output dir: {seq_output_dir}")
        
        seq_output_dir.mkdir(parents=True, exist_ok=True)
        debug_print(f"Output directory created/verified: {seq_output_dir}")
        
        # Build command
        cmd = [
            "python", script_name,
            "--input_path", args_dict["input_path"],
            "--output_path", str(seq_output_dir),
            "--sequence", sequence_name,
            "--max_frames", str(max_frames)
        ]
        
        # Add WANDB arguments
        if args_dict.get("wandb", False):
            cmd.extend(["--wandb", "--experiment-name", f"{baseline_name}_{sequence_name}"])
            debug_print(f"WANDB enabled, experiment name: {baseline_name}_{sequence_name}")
        
        # Add TSP-SAM specific arguments
        if "tsp_sam" in script_name:
            cmd.extend([
                "--enable-advanced-metrics",
                "--enable-memory-monitoring", 
                "--enable-failure-analysis"
            ])
            debug_print("TSP-SAM specific arguments added")
        
        debug_print(f"Full command: {' '.join(cmd)}")
        debug_print(f"Working directory: {os.getcwd()}")
        
        # Execute command
        start_time = time.time()
        debug_print(f"Starting subprocess execution at: {datetime.fromtimestamp(start_time)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        end_time = time.time()
        
        execution_time = end_time - start_time
        debug_print(f"Subprocess completed in {execution_time:.2f}s with return code: {result.returncode}")
        
        # Save execution log
        log_file = experiment_dir / "logs" / f"{baseline_name}_{sequence_name}_execution.log"
        debug_print(f"Saving execution log to: {log_file}")
        
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.fromtimestamp(start_time).isoformat()}\n")
            f.write(f"End time: {datetime.fromtimestamp(end_time).isoformat()}\n")
            f.write(f"Execution time: {execution_time:.2f}s\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Working directory: {os.getcwd()}\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
        
        debug_print(f"Execution log saved successfully")
        
        if result.returncode == 0:
            debug_print(f"{script_name} completed successfully")
            return {
                "status": "success",
                "execution_time": execution_time,
                "output": result.stdout,
                "error": result.stderr,
                "log_file": str(log_file)
            }
        else:
            debug_print(f"{script_name} failed with return code {result.returncode}", "ERROR")
            debug_print(f"STDERR content: {result.stderr}", "ERROR")
            return {
                "status": "failed",
                "execution_time": execution_time,
                "return_code": result.returncode,
                "output": result.stdout,
                "error": result.stderr,
                "log_file": str(log_file)
            }
            
    except subprocess.TimeoutExpired:
        debug_print(f"{script_name} timed out after 1 hour", "ERROR")
        return {
            "status": "timeout",
            "execution_time": 3600,
            "error": "Script execution timed out after 1 hour"
        }
    except Exception as e:
        debug_print(f"{script_name} encountered error: {e}", "ERROR")
        debug_print(f"Full traceback: {traceback.format_exc()}", "ERROR")
        return {
            "status": "error",
            "execution_time": 0,
            "error": str(e)
        }

def organize_baseline_outputs(experiment_dir, baseline_name):
    """Organize baseline outputs into proper structure"""
    debug_print(f"Organizing outputs for baseline: {baseline_name}")
    
    try:
        baseline_dir = experiment_dir / "results" / baseline_name
        debug_print(f"Baseline results directory: {baseline_dir}")
        
        # Create organized structure
        sequences_dir = baseline_dir / "sequences"
        metrics_dir = baseline_dir / "metrics"
        visualizations_dir = baseline_dir / "visualizations"
        
        for dir_path in [metrics_dir, visualizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            debug_print(f"Created directory: {dir_path}")
        
        # Move and organize sequence outputs from baseline-specific temp directory
        baseline_temp_dir = experiment_dir / "temp" / baseline_name
        debug_print(f"Looking for temp directory: {baseline_temp_dir}")
        
        if baseline_temp_dir.exists():
            debug_print(f"Found temp directory, organizing contents...")
            for item in baseline_temp_dir.iterdir():
                debug_print(f"Processing temp item: {item}")
                if item.is_dir():
                    # This is a sequence directory
                    seq_name = item.name
                    target_seq_dir = sequences_dir / seq_name
                    if target_seq_dir.exists():
                        debug_print(f"Removing existing target directory: {target_seq_dir}")
                        shutil.rmtree(target_seq_dir)
                    debug_print(f"Moving {seq_name} to {target_seq_dir}")
                    shutil.move(str(item), str(target_seq_dir))
                    
                    # Create metrics and visualizations subdirectories
                    (target_seq_dir / "metrics").mkdir(exist_ok=True)
                    (target_seq_dir / "visualizations").mkdir(exist_ok=True)
                    debug_print(f"Created subdirectories for {seq_name}")
        else:
            debug_print(f"Warning: {baseline_name} temp directory not found: {baseline_temp_dir}", "WARNING")
        
        debug_print(f"Output organization completed for {baseline_name}")
        return baseline_dir
        
    except Exception as e:
        debug_print(f"Error organizing {baseline_name} outputs: {e}", "ERROR")
        raise e

def process_sequences(args, experiment_dir):
    """Process all sequences with both baselines"""
    debug_print("Starting sequence processing...")
    
    try:
        # Get available sequences
        print("Discovering available sequences...")
        sequences = get_available_sequences(args.input_path)
        
        if not sequences:
            debug_print("No sequences found!", "ERROR")
            return None
        
        debug_print(f"Total sequences found: {len(sequences)}")
        
        # Filter sequences if specific ones are requested
        if hasattr(args, 'sequences') and args.sequences:
            debug_print(f"Filtering to specific sequences: {args.sequences}")
            sequences = [seq for seq in sequences if seq in args.sequences]
            debug_print(f"Filtered sequences: {sequences}")
        
        print(f"Found {len(sequences)} sequences: {sequences[:5]}... (showing first 5)")
        
        # Create baseline output directories
        tsp_sam_output = experiment_dir / "temp" / "tsp_sam"
        samurai_output = experiment_dir / "temp" / "samurai"
        
        tsp_sam_output.mkdir(parents=True, exist_ok=True)
        samurai_output.mkdir(parents=True, exist_ok=True)
        
        debug_print(f"TSP-SAM temp output: {tsp_sam_output}")
        debug_print(f"SAMURAI temp output: {samurai_output}")
        
        print(f"TSP-SAM output: {tsp_sam_output}")
        print(f"SAMURAI output: {samurai_output}")
        
        # Results tracking
        results = {
            "timestamp": datetime.now().isoformat(),
            "input_path": args.input_path,
            "max_frames": args.max_frames,
            "total_sequences": len(sequences),
            "sequences": {}
        }
        
        # Process each sequence
        successful_tsp_sam = 0
        successful_samurai = 0
        total_start_time = time.time()
        
        for seq_idx, sequence_name in enumerate(sequences):
            debug_print(f"Starting sequence {seq_idx + 1}/{len(sequences)}: {sequence_name}")
            
            print(f"\n{'='*80}")
            print(f"Processing sequence {seq_idx + 1}/{len(sequences)}: {sequence_name}")
            print(f"{'='*80}")
            
            sequence_results = {
                "sequence_name": sequence_name,
                "tsp_sam": {},
                "samurai": {},
                "start_time": datetime.now().isoformat()
            }
            
            try:
                # Run TSP-SAM baseline
                print(f"\nRunning TSP-SAM baseline on {sequence_name}...")
                debug_print(f"Starting TSP-SAM for sequence: {sequence_name}")
                
                tsp_sam_result = run_baseline_script(
                    "run_tsp_sam_baseline.py",
                    {
                        "input_path": args.input_path,
                        "wandb": args.wandb
                    },
                    sequence_name,
                    args.max_frames,
                    tsp_sam_output,
                    experiment_dir
                )
                
                sequence_results["tsp_sam"] = tsp_sam_result
                if tsp_sam_result["status"] == "success":
                    successful_tsp_sam += 1
                    debug_print(f"TSP-SAM succeeded for {sequence_name}")
                else:
                    debug_print(f"TSP-SAM failed for {sequence_name}: {tsp_sam_result['status']}", "ERROR")
                
                # Run SAMURAI baseline
                print(f"\nRunning SAMURAI baseline on {sequence_name}...")
                debug_print(f"Starting SAMURAI for sequence: {sequence_name}")
                
                samurai_result = run_baseline_script(
                    "run_samurai_baseline.py",
                    {
                        "input_path": args.input_path,
                        "wandb": args.wandb
                    },
                    sequence_name,
                    args.max_frames,
                    samurai_output,
                    experiment_dir
                )
                
                sequence_results["samurai"] = samurai_result
                if samurai_result["status"] == "success":
                    successful_samurai += 1
                    debug_print(f"SAMURAI succeeded for {sequence_name}")
                else:
                    debug_print(f"SAMURAI failed for {sequence_name}: {samurai_result['status']}", "ERROR")
                
                # Store results
                results["sequences"][sequence_name] = sequence_results
                
                # Progress update
                print(f"\nProgress: {seq_idx + 1}/{len(sequences)} sequences completed")
                print(f"   TSP-SAM: {successful_tsp_sam} successful")
                print(f"   SAMURAI: {successful_samurai} successful")
                
                debug_print(f"Sequence {sequence_name} completed. TSP-SAM: {successful_tsp_sam}, SAMURAI: {successful_samurai}")
                
            except Exception as e:
                debug_print(f"Error processing sequence {sequence_name}: {e}", "ERROR")
                debug_print(f"Full traceback: {traceback.format_exc()}", "ERROR")
                print(f"Error processing sequence {sequence_name}: {e}")
                traceback.print_exc()
                sequence_results["error"] = str(e)
                results["sequences"][sequence_name] = sequence_results
                continue
        
        # Final summary
        total_time = time.time() - total_start_time
        debug_print(f"All sequences processed. Total time: {total_time:.2f}s")
        
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Total sequences processed: {len(sequences)}")
        print(f"TSP-SAM successful: {successful_tsp_sam}/{len(sequences)} ({successful_tsp_sam/len(sequences)*100:.1f}%)")
        print(f"SAMURAI successful: {successful_samurai}/{len(sequences)} ({successful_samurai/len(sequences)*100:.1f}%)")
        print(f"Total processing time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        
        # Save results to JSON
        results_file = experiment_dir / "results" / "comparison_results.json"
        debug_print(f"Saving results to: {results_file}")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        debug_print("Results saved successfully")
        
        # Organize outputs
        print("\nOrganizing outputs...")
        debug_print("Starting output organization...")
        
        tsp_sam_organized = organize_baseline_outputs(experiment_dir, "tsp_sam")
        samurai_organized = organize_baseline_outputs(experiment_dir, "samurai")
        
        debug_print("Output organization completed")
        
        # Create experiment README
        create_experiment_readme(experiment_dir, results, total_time)
        
        # Clean up temp directories after organization
        print("\nCleaning up temporary directories...")
        temp_dir = experiment_dir / "temp"
        if temp_dir.exists():
            try:
                debug_print(f"Removing temp directory: {temp_dir}")
                shutil.rmtree(temp_dir)
                print("  Temporary directories cleaned up successfully")
            except Exception as e:
                debug_print(f"Warning: Could not clean up temp directory: {e}", "WARNING")
                print(f"  Warning: Could not clean up temp directory: {e}")
        
        print(f"\nDetailed results saved to: {results_file}")
        debug_print("Sequence processing completed successfully")
        
        return results
        
    except Exception as e:
        debug_print(f"Error in process_sequences: {e}", "ERROR")
        debug_print(f"Full traceback: {traceback.format_exc()}", "ERROR")
        raise e

def create_experiment_readme(experiment_dir, results, total_time):
    """Create comprehensive experiment documentation"""
    debug_print("Creating experiment README...")
    
    try:
        readme_content = f"""# Experiment: {experiment_dir.name}

## Overview
Multi-sequence comparison of TSP-SAM vs SAMURAI baselines on DAVIS-2017 dataset.

## Results Summary
- **Total Sequences**: {results['total_sequences']}
- **TSP-SAM Success Rate**: {sum(1 for seq in results['sequences'].values() if seq.get('tsp_sam', {}).get('status') == 'success')}/{results['total_sequences']}
- **SAMURAI Success Rate**: {sum(1 for seq in results['sequences'].values() if seq.get('samurai', {}).get('status') == 'success')}/{results['total_sequences']}
- **Total Processing Time**: {total_time/3600:.2f} hours

## Directory Structure
```
{experiment_dir.name}/
├── config/                    # Experiment configuration files
├── results/                   # All experimental results
│   ├── tsp_sam/             # TSP-SAM baseline results
│   ├── samurai/             # SAMURAI baseline results
│   └── comparison/          # Comparative analysis
├── logs/                     # Execution logs and error reports
├── metadata/                 # System info and reproducibility data
└── README.md                 # This file
```

## Reproducibility
See `metadata/reproducibility_info.txt` for system details and reproduction instructions.

## Analysis
Results are organized by baseline and sequence for easy comparison and analysis.
"""
        
        readme_file = experiment_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        debug_print(f"README created at: {readme_file}")
        
    except Exception as e:
        debug_print(f"Error creating README: {e}", "ERROR")

def main():
    debug_print("Starting main function...")
    
    parser = argparse.ArgumentParser(
        description='Multi-Sequence Comparison Runner for TSP-SAM vs SAMURAI Baselines'
    )
    
    parser.add_argument(
        '--input_path', 
        type=str, 
        default='input/davis2017',
        help='Path to DAVIS 2017 dataset'
    )
    
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='output',
        help='Base output path for results'
    )
    
    parser.add_argument(
        '--max_frames', 
        type=int, 
        default=50,
        help='Maximum frames to process per sequence (use 0 for all frames)'
    )
    
    parser.add_argument(
        '--wandb', 
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    
    parser.add_argument(
        '--sequences', 
        type=str, 
        nargs='+',
        help='Specific sequences to process (default: all sequences)'
    )
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        help='Custom experiment name (default: timestamp-based)'
    )
    
    args = parser.parse_args()
    debug_print(f"Arguments parsed: {vars(args)}")
    
    # Generate experiment name
    if not args.experiment_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"comparison_run_{timestamp}"
        debug_print(f"Generated experiment name: {args.experiment_name}")
    
    print("Multi-Sequence Comparison Runner")
    print("=" * 50)
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print(f"Experiment name: {args.experiment_name}")
    print(f"Max frames: {args.max_frames if args.max_frames > 0 else 'ALL'}")
    print(f"WANDB logging: {'Enabled' if args.wandb else 'Disabled'}")
    
    if args.sequences:
        print(f"Target sequences: {args.sequences}")
        debug_print(f"Processing specific sequences: {args.sequences}")
    else:
        print("Target sequences: ALL available sequences")
        debug_print("Processing all available sequences")
    
    try:
        # Get system information
        debug_print("Collecting system information...")
        system_info = get_system_info()
        
        # Create experiment structure
        debug_print("Creating experiment directory structure...")
        experiment_dir = create_experiment_structure(args.output_path, args.experiment_name)
        print(f"\nExperiment directory: {experiment_dir}")
        
        # Save configuration
        debug_print("Saving experiment configuration...")
        save_experiment_config(experiment_dir, args, system_info)
        
        # Confirm before proceeding
        if args.max_frames == 0:
            print("\nWARNING: Processing ALL frames for all sequences will take a very long time!")
            print("   Estimated time: 60 sequences × ~100 frames × 2 baselines × ~30s = ~50+ hours")
            confirm = input("\nDo you want to continue? (yes/no): ")
            if confirm.lower() not in ['yes', 'y']:
                debug_print("User cancelled operation")
                print("Operation cancelled.")
                return
        
        # Process sequences
        debug_print("Starting sequence processing...")
        results = process_sequences(args, experiment_dir)
        
        if results:
            print(f"\nMulti-sequence comparison completed!")
            print(f"Check the experiment directory for organized results: {experiment_dir}")
            
            # Create symlink to latest
            latest_link = Path(args.output_path) / "latest_results"
            if latest_link.exists():
                debug_print(f"Removing existing latest link: {latest_link}")
                latest_link.unlink()
            
            try:
                latest_link.symlink_to(experiment_dir, target_is_directory=True)
                debug_print(f"Created latest results symlink: {latest_link}")
                print(f"Latest results symlink created: {latest_link}")
            except Exception as e:
                debug_print(f"Could not create symlink (Windows limitation): {e}", "WARNING")
                print(f"Note: Could not create symlink (Windows limitation): {e}")
        else:
            debug_print("Sequence processing failed", "ERROR")
            print("\nMulti-sequence comparison failed!")
            
    except Exception as e:
        debug_print(f"Critical error in main: {e}", "ERROR")
        debug_print(f"Full traceback: {traceback.format_exc()}", "ERROR")
        print(f"\nCritical error: {e}")
        traceback.print_exc()
        return 1
    
    debug_print("Main function completed successfully")
    return 0

if __name__ == "__main__":
    debug_print("Script started")
    exit_code = main()
    debug_print(f"Script finished with exit code: {exit_code}")
    sys.exit(exit_code)
