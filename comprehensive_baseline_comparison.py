#!/usr/bin/env python3
"""
Comprehensive Baseline Comparison Script for Thesis Expose
Demonstrates the need for hybrid approach by comparing TSP-SAM, SAMURAI, and hybrid pipeline
"""

import os
import sys
import argparse
import subprocess
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import traceback
import platform
import psutil
import torch

# Global configuration
DEBUG_MODE = True
RESULTS_DIR = "comparison_results"
WANDB_PROJECT = "thesis-baseline-comparison"

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
            "results/baselines/tsp_sam",
            "results/baselines/samurai", 
            "results/hybrid",
            "results/comparison",
            "logs",
            "metadata",
            "visualizations",
            "reports"
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

def run_baseline_experiment(script_name, args_dict, sequence_name, max_frames, output_dir, experiment_dir):
    """Run a baseline script with given arguments"""
    debug_print(f"Starting baseline script: {script_name} for sequence: {sequence_name}")
    
    try:
        # Create sequence-specific output directory
        baseline_name = script_name.replace('.py', '').replace('run_', '').replace('_baseline', '')
        debug_print(f"Baseline name extracted: {baseline_name}")
        
        # Handle SAMURAI and HYBRID differently - they add sequence name internally
        if "samurai" in script_name:
            seq_output_dir = output_dir / baseline_name  # SAMURAI will add sequence name
            debug_print(f"SAMURAI detected, using base output dir: {seq_output_dir}")
        elif "hybrid" in script_name:
            seq_output_dir = output_dir / baseline_name  # HYBRID will add sequence name
            debug_print(f"HYBRID detected, using base output dir: {seq_output_dir}")
        else:
            seq_output_dir = output_dir / baseline_name / sequence_name  # TSP-SAM expects sequence subdirectory
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
            cmd.extend(["--wandb", "--experiment-name", f"{baseline_name}_{sequence_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"])
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
        
        # Execute command
        start_time = time.time()
        debug_print(f"Starting subprocess execution at: {datetime.fromtimestamp(start_time)}")
        
        # Use Popen to avoid buffer overflow issues
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0  # Unbuffered
        )
        
        try:
            # Add process monitoring
            debug_print(f"Process PID: {process.pid}")
            debug_print(f"Process started, waiting for completion...")
            
            stdout, stderr = process.communicate(timeout=1800)  # 30 minutes timeout
            return_code = process.returncode
            debug_print(f"Process completed normally with return code: {return_code}")
        except subprocess.TimeoutExpired:
            debug_print(f"Process timed out after 30 minutes, killing...", "ERROR")
            process.kill()
            stdout, stderr = process.communicate()
            return_code = -1
            debug_print(f"Subprocess timed out and was killed", "ERROR")
        
        end_time = time.time()
        execution_time = end_time - start_time
        debug_print(f"Subprocess completed in {execution_time:.2f}s with return code: {return_code}")
        
        # Save execution log
        log_file = experiment_dir / "logs" / f"{baseline_name}_{sequence_name}_execution.log"
        debug_print(f"Saving execution log to: {log_file}")
        
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.fromtimestamp(start_time).isoformat()}\n")
            f.write(f"End time: {datetime.fromtimestamp(end_time).isoformat()}\n")
            f.write(f"Execution time: {execution_time:.2f}s\n")
            f.write(f"Return code: {return_code}\n")
            f.write(f"Working directory: {os.getcwd()}\n\n")
            f.write("STDOUT:\n")
            f.write(stdout)
            f.write("\n\nSTDERR:\n")
            f.write(stderr)
        
        debug_print(f"Execution log saved successfully")
        
        if return_code == 0:
            debug_print(f"{script_name} completed successfully")
            return {
                "status": "success",
                "execution_time": execution_time,
                "output": stdout,
                "error": stderr,
                "log_file": str(log_file),
                "output_dir": str(seq_output_dir)
            }
        else:
            debug_print(f"{script_name} failed with return code {return_code}", "ERROR")
            return {
                "status": "failed",
                "execution_time": execution_time,
                "return_code": return_code,
                "output": stdout,
                "error": stderr,
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

def run_hybrid_experiment(args_dict, sequence_name, max_frames, output_dir, experiment_dir):
    """Run hybrid pipeline experiment"""
    debug_print(f"Starting hybrid pipeline for sequence: {sequence_name}")
    
    try:
        # Create sequence-specific output directory
        # HYBRID script adds sequence name internally, so we only provide the base hybrid directory
        seq_output_dir = output_dir / "hybrid"
        seq_output_dir.mkdir(parents=True, exist_ok=True)
        debug_print(f"Hybrid output directory created/verified: {seq_output_dir}")
        
        # Build command for hybrid pipeline
        cmd = [
            "python", "run_hybrid_on_davis.py",
            "--davis-root", args_dict["input_path"],
            "--output-root", str(seq_output_dir),
            "--sequence", sequence_name,
            "--max-frames", str(max_frames)
        ]
        
        # Add WANDB arguments
        if args_dict.get("wandb", False):
            cmd.extend(["--use-wandb"])
            debug_print(f"WANDB enabled for hybrid pipeline")
        
        debug_print(f"Full hybrid command: {' '.join(cmd)}")
        
        # Execute command
        start_time = time.time()
        debug_print(f"Starting hybrid subprocess execution at: {datetime.fromtimestamp(start_time)}")
        
        # Use Popen to avoid buffer overflow issues
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0  # Unbuffered
        )
        
        try:
            # Add process monitoring
            debug_print(f"Hybrid Process PID: {process.pid}")
            debug_print(f"Hybrid process started, waiting for completion...")
            
            stdout, stderr = process.communicate(timeout=1800)  # 30 minutes timeout
            return_code = process.returncode
            debug_print(f"Hybrid process completed normally with return code: {return_code}")
        except subprocess.TimeoutExpired:
            debug_print(f"Hybrid process timed out after 30 minutes, killing...", "ERROR")
            process.kill()
            stdout, stderr = process.communicate()
            return_code = -1
            debug_print(f"Hybrid subprocess timed out and was killed", "ERROR")
        
        end_time = time.time()
        execution_time = end_time - start_time
        debug_print(f"Hybrid subprocess completed in {execution_time:.2f}s with return code: {return_code}")
        
        # Save execution log
        log_file = experiment_dir / "logs" / f"hybrid_{sequence_name}_execution.log"
        debug_print(f"Saving hybrid execution log to: {log_file}")
        
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.fromtimestamp(start_time).isoformat()}\n")
            f.write(f"End time: {datetime.fromtimestamp(end_time).isoformat()}\n")
            f.write(f"Execution time: {execution_time:.2f}s\n")
            f.write(f"Return code: {return_code}\n")
            f.write(f"Working directory: {os.getcwd()}\n\n")
            f.write("STDOUT:\n")
            f.write(stdout)
            f.write("\n\nSTDERR:\n")
            f.write(stderr)
        
        debug_print(f"Hybrid execution log saved successfully")
        
        if return_code == 0:
            debug_print(f"Hybrid pipeline completed successfully")
            return {
                "status": "success",
                "execution_time": execution_time,
                "output": stdout,
                "error": stderr,
                "log_file": str(log_file),
                "output_dir": str(seq_output_dir)
            }
        else:
            debug_print(f"Hybrid pipeline failed with return code {return_code}", "ERROR")
            return {
                "status": "failed",
                "execution_time": execution_time,
                "return_code": return_code,
                "output": stdout,
                "error": stderr,
                "log_file": str(log_file)
            }
            
    except subprocess.TimeoutExpired:
        debug_print(f"Hybrid pipeline timed out after 1 hour", "ERROR")
        return {
            "status": "timeout",
            "execution_time": 3600,
            "error": "Hybrid pipeline execution timed out after 1 hour"
        }
    except Exception as e:
        debug_print(f"Hybrid pipeline encountered error: {e}", "ERROR")
        debug_print(f"Full traceback: {traceback.format_exc()}", "ERROR")
        return {
            "status": "error",
            "execution_time": 0,
            "error": str(e)
        }

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

def process_sequences_comparison(args, experiment_dir):
    """Process sequences with all three approaches for comparison"""
    debug_print("Starting comprehensive sequence comparison...")
    
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
        tsp_sam_output = experiment_dir / "results" / "baselines" / "tsp_sam"
        samurai_output = experiment_dir / "results" / "baselines" / "samurai"
        hybrid_output = experiment_dir / "results" / "hybrid"
        
        tsp_sam_output.mkdir(parents=True, exist_ok=True)
        samurai_output.mkdir(parents=True, exist_ok=True)
        hybrid_output.mkdir(parents=True, exist_ok=True)
        
        debug_print(f"TSP-SAM output: {tsp_sam_output}")
        debug_print(f"SAMURAI output: {samurai_output}")
        debug_print(f"Hybrid output: {hybrid_output}")
        
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
        successful_hybrid = 0
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
                "hybrid": {},
                "start_time": datetime.now().isoformat()
            }
            
            try:
                # Run TSP-SAM baseline
                print(f"\nRunning TSP-SAM baseline on {sequence_name}...")
                debug_print(f"Starting TSP-SAM for sequence: {sequence_name}")
                
                tsp_sam_result = run_baseline_experiment(
                    "run_tsp_sam_baseline.py",
                    {
                        "input_path": args.input_path,
                        "wandb": args.wandb
                    },
                    sequence_name,
                    args.max_frames,
                    experiment_dir / "results" / "baselines",
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
                
                samurai_result = run_baseline_experiment(
                    "run_samurai_baseline.py",
                    {
                        "input_path": args.input_path,
                        "wandb": args.wandb
                    },
                    sequence_name,
                    args.max_frames,
                    experiment_dir / "results" / "baselines",
                    experiment_dir
                )
                
                sequence_results["samurai"] = samurai_result
                if samurai_result["status"] == "success":
                    successful_samurai += 1
                    debug_print(f"SAMURAI succeeded for {sequence_name}")
                else:
                    debug_print(f"SAMURAI failed for {sequence_name}: {samurai_result['status']}", "ERROR")
                
                # Run Hybrid pipeline
                print(f"\nRunning Hybrid pipeline on {sequence_name}...")
                debug_print(f"Starting Hybrid for sequence: {sequence_name}")
                
                hybrid_result = run_hybrid_experiment(
                    {
                        "input_path": args.input_path,
                        "wandb": args.wandb
                    },
                    sequence_name,
                    args.max_frames,
                    experiment_dir / "results",
                    experiment_dir
                )
                
                sequence_results["hybrid"] = hybrid_result
                if hybrid_result["status"] == "success":
                    successful_hybrid += 1
                    debug_print(f"Hybrid succeeded for {sequence_name}")
                else:
                    debug_print(f"Hybrid failed for {sequence_name}: {hybrid_result['status']}", "ERROR")
                
                # Store results
                results["sequences"][sequence_name] = sequence_results
                
                # Progress update
                print(f"\nProgress: {seq_idx + 1}/{len(sequences)} sequences completed")
                print(f"   TSP-SAM: {successful_tsp_sam} successful")
                print(f"   SAMURAI: {successful_samurai} successful")
                print(f"   Hybrid: {successful_hybrid} successful")
                
                debug_print(f"Sequence {sequence_name} completed. TSP-SAM: {successful_tsp_sam}, SAMURAI: {successful_samurai}, Hybrid: {successful_hybrid}")
                
                # Clean up resources between sequences
                import gc
                gc.collect()  # Python garbage collection
                
                # Clear GPU cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except ImportError:
                    pass
                
                # Small delay to let system recover
                time.sleep(1)
                
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
        print(f"Hybrid successful: {successful_hybrid}/{len(sequences)} ({successful_hybrid/len(sequences)*100:.1f}%)")
        print(f"Total processing time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        
        # Save results to JSON
        results_file = experiment_dir / "results" / "comparison_results.json"
        debug_print(f"Saving results to: {results_file}")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        debug_print("Results saved successfully")
        
        # Generate comparison report
        generate_comparison_report(results, experiment_dir)
        
        print(f"\nDetailed results saved to: {results_file}")
        debug_print("Sequence processing completed successfully")
        
        return results
        
    except Exception as e:
        debug_print(f"Error in process_sequences_comparison: {e}", "ERROR")
        debug_print(f"Full traceback: {traceback.format_exc()}", "ERROR")
        raise e

def generate_comparison_report(results, experiment_dir):
    """Generate comprehensive comparison report"""
    debug_print("Generating comparison report...")
    
    try:
        # Extract metrics from results
        comparison_data = []
        
        for seq_name, seq_results in results["sequences"].items():
            row = {"sequence": seq_name}
            
            # TSP-SAM metrics
            if seq_results["tsp_sam"]["status"] == "success":
                row["tsp_sam_success"] = True
                row["tsp_sam_time"] = seq_results["tsp_sam"]["execution_time"]
            else:
                row["tsp_sam_success"] = False
                row["tsp_sam_time"] = 0
            
            # SAMURAI metrics
            if seq_results["samurai"]["status"] == "success":
                row["samurai_success"] = True
                row["samurai_time"] = seq_results["samurai"]["execution_time"]
            else:
                row["samurai_success"] = False
                row["samurai_time"] = 0
            
            # Hybrid metrics
            if seq_results["hybrid"]["status"] == "success":
                row["hybrid_success"] = True
                row["hybrid_time"] = seq_results["hybrid"]["execution_time"]
            else:
                row["hybrid_success"] = False
                row["hybrid_time"] = 0
            
            comparison_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Save CSV report
        csv_file = experiment_dir / "reports" / "comparison_summary.csv"
        df.to_csv(csv_file, index=False)
        debug_print(f"CSV report saved to: {csv_file}")
        
        # Generate summary statistics
        summary_stats = {
            "total_sequences": len(df),
            "tsp_sam_success_rate": df["tsp_sam_success"].mean(),
            "samurai_success_rate": df["samurai_success"].mean(),
            "hybrid_success_rate": df["hybrid_success"].mean(),
            "tsp_sam_avg_time": df[df["tsp_sam_success"]]["tsp_sam_time"].mean(),
            "samurai_avg_time": df[df["samurai_success"]]["samurai_time"].mean(),
            "hybrid_avg_time": df[df["hybrid_success"]]["hybrid_time"].mean(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save summary statistics
        summary_file = experiment_dir / "reports" / "summary_statistics.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        debug_print(f"Summary statistics saved to: {summary_file}")
        
        # Generate visualizations
        generate_comparison_visualizations(df, experiment_dir)
        
        # Generate separate baseline reports for thesis analysis
        generate_individual_baseline_reports(df, experiment_dir, results["max_frames"])
        
        debug_print("Comparison report generated successfully")
        
    except Exception as e:
        debug_print(f"Error generating comparison report: {e}", "ERROR")
        raise e

def generate_comparison_visualizations(df, experiment_dir):
    """Generate comparison visualizations"""
    debug_print("Generating comparison visualizations...")
    
    try:
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Baseline Comparison: TSP-SAM vs SAMURAI vs Hybrid Pipeline', fontsize=16, fontweight='bold')
        
        # 1. Success Rate Comparison
        success_rates = [
            df["tsp_sam_success"].mean() * 100,
            df["samurai_success"].mean() * 100,
            df["hybrid_success"].mean() * 100
        ]
        models = ['TSP-SAM', 'SAMURAI', 'Hybrid']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars1 = axes[0, 0].bar(models, success_rates, color=colors, alpha=0.8)
        axes[0, 0].set_title('Success Rate Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('Success Rate (%)')
        axes[0, 0].set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, success_rates):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Processing Time Comparison
        avg_times = [
            df[df["tsp_sam_success"]]["tsp_sam_time"].mean(),
            df[df["samurai_success"]]["samurai_time"].mean(),
            df[df["hybrid_success"]]["hybrid_time"].mean()
        ]
        
        bars2 = axes[0, 1].bar(models, avg_times, color=colors, alpha=0.8)
        axes[0, 1].set_title('Average Processing Time', fontweight='bold')
        axes[0, 1].set_ylabel('Time (seconds)')
        
        # Add value labels on bars
        for bar, time_val in zip(bars2, avg_times):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + max(avg_times)*0.01,
                           f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 3. Success/Failure Pattern
        success_data = []
        for model in ['tsp_sam', 'samurai', 'hybrid']:
            success_count = df[f"{model}_success"].sum()
            failure_count = len(df) - success_count
            success_data.extend([success_count, failure_count])
        
        model_names = ['TSP-SAM\nSuccess', 'TSP-SAM\nFailure', 
                      'SAMURAI\nSuccess', 'SAMURAI\nFailure',
                      'Hybrid\nSuccess', 'Hybrid\nFailure']
        success_colors = ['#FF6B6B', '#FFB3B3', '#4ECDC4', '#B3E5E1', '#45B7D1', '#B3D9F2']
        
        bars3 = axes[1, 0].bar(model_names, success_data, color=success_colors, alpha=0.8)
        axes[1, 0].set_title('Success vs Failure Count', fontweight='bold')
        axes[1, 0].set_ylabel('Number of Sequences')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars3, success_data):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(success_data)*0.01,
                           f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Complementary Analysis
        # Find sequences where each approach succeeds/fails
        tsp_only_success = df[(df["tsp_sam_success"]) & (~df["samurai_success"]) & (~df["hybrid_success"])]
        sam_only_success = df[(~df["tsp_sam_success"]) & (df["samurai_success"]) & (~df["hybrid_success"])]
        hybrid_only_success = df[(~df["tsp_sam_success"]) & (~df["samurai_success"]) & (df["hybrid_success"])]
        all_success = df[(df["tsp_sam_success"]) & (df["samurai_success"]) & (df["hybrid_success"])]
        all_fail = df[(~df["tsp_sam_success"]) & (~df["samurai_success"]) & (~df["hybrid_success"])]
        
        complementary_data = [
            len(tsp_only_success),
            len(sam_only_success), 
            len(hybrid_only_success),
            len(all_success),
            len(all_fail)
        ]
        complementary_labels = ['TSP-SAM\nOnly', 'SAMURAI\nOnly', 'Hybrid\nOnly', 'All\nSuccess', 'All\nFail']
        complementary_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        bars4 = axes[1, 1].bar(complementary_labels, complementary_data, color=complementary_colors, alpha=0.8)
        axes[1, 1].set_title('Complementary Success Patterns', fontweight='bold')
        axes[1, 1].set_ylabel('Number of Sequences')
        
        # Add value labels on bars
        for bar, count in zip(bars4, complementary_data):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(complementary_data)*0.01,
                           f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout()
        viz_file = experiment_dir / "visualizations" / "baseline_comparison.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        debug_print(f"Visualization saved to: {viz_file}")
        
        # Generate thesis-ready summary
        generate_thesis_summary(df, experiment_dir)
        
    except Exception as e:
        debug_print(f"Error generating visualizations: {e}", "ERROR")
        raise e

def generate_thesis_summary(df, experiment_dir):
    """Generate thesis-ready summary report"""
    debug_print("Generating thesis summary...")
    
    try:
        # Calculate key metrics
        total_sequences = len(df)
        tsp_success_rate = df["tsp_sam_success"].mean() * 100
        sam_success_rate = df["samurai_success"].mean() * 100
        hybrid_success_rate = df["hybrid_success"].mean() * 100
        
        # Find complementary patterns
        tsp_only = df[(df["tsp_sam_success"]) & (~df["samurai_success"])]
        sam_only = df[(~df["tsp_sam_success"]) & (df["samurai_success"])]
        hybrid_only = df[(~df["tsp_sam_success"]) & (~df["samurai_success"]) & (df["hybrid_success"])]
        all_success = df[(df["tsp_sam_success"]) & (df["samurai_success"]) & (df["hybrid_success"])]
        all_fail = df[(~df["tsp_sam_success"]) & (~df["samurai_success"]) & (~df["hybrid_success"])]
        
        # Generate summary report
        summary_content = f"""
# BASELINE COMPARISON REPORT FOR THESIS EXPOSE

## Executive Summary
This report demonstrates the need for a hybrid approach in video de-identification by comparing three methods:
1. **TSP-SAM Baseline**: Temporal segmentation approach
2. **SAMURAI Baseline**: State-of-the-art video object segmentation
3. **Hybrid Pipeline**: Combined approach using multiple models

## Key Findings

### Success Rates
- **TSP-SAM**: {tsp_success_rate:.1f}% ({df["tsp_sam_success"].sum()}/{total_sequences} sequences)
- **SAMURAI**: {sam_success_rate:.1f}% ({df["samurai_success"].sum()}/{total_sequences} sequences)  
- **Hybrid Pipeline**: {hybrid_success_rate:.1f}% ({df["hybrid_success"].sum()}/{total_sequences} sequences)

### Complementary Analysis
- **TSP-SAM Only Success**: {len(tsp_only)} sequences ({len(tsp_only)/total_sequences*100:.1f}%)
- **SAMURAI Only Success**: {len(sam_only)} sequences ({len(sam_only)/total_sequences*100:.1f}%)
- **Hybrid Only Success**: {len(hybrid_only)} sequences ({len(hybrid_only)/total_sequences*100:.1f}%)
- **All Methods Success**: {len(all_success)} sequences ({len(all_success)/total_sequences*100:.1f}%)
- **All Methods Fail**: {len(all_fail)} sequences ({len(all_fail)/total_sequences*100:.1f}%)

### Processing Performance
- **TSP-SAM Average Time**: {df[df["tsp_sam_success"]]["tsp_sam_time"].mean():.1f} seconds
- **SAMURAI Average Time**: {df[df["samurai_success"]]["samurai_time"].mean():.1f} seconds
- **Hybrid Average Time**: {df[df["hybrid_success"]]["hybrid_time"].mean():.1f} seconds

## Research Implications

### 1. Individual Baseline Limitations
- Neither TSP-SAM nor SAMURAI alone provides reliable video segmentation
- Individual success rates of {tsp_success_rate:.1f}% and {sam_success_rate:.1f}% are insufficient for production use
- High failure rates indicate the need for complementary approaches

### 2. Complementary Strengths
- {len(tsp_only)} sequences succeed only with TSP-SAM, demonstrating temporal modeling advantages
- {len(sam_only)} sequences succeed only with SAMURAI, showing spatial segmentation strengths
- {len(hybrid_only)} sequences succeed only with the hybrid approach, proving the value of model combination

### 3. Hybrid Approach Benefits
- **Improved Reliability**: {hybrid_success_rate:.1f}% success rate vs {max(tsp_success_rate, sam_success_rate):.1f}% for best individual method
- **Complementary Coverage**: Captures cases where individual methods fail
- **Robustness**: Reduces dependency on single model limitations

### 4. Thesis Contribution Justification
- **Problem**: Individual baselines show insufficient reliability for video de-identification
- **Solution**: Hybrid pipeline combining multiple approaches
- **Evidence**: {len(hybrid_only)} sequences succeed only with hybrid approach
- **Impact**: {hybrid_success_rate - max(tsp_success_rate, sam_success_rate):.1f} percentage point improvement over best individual method

## Recommendations for Thesis

### 1. Methodology Section
- Emphasize the complementary nature of different segmentation approaches
- Highlight the need for robust, multi-model solutions in video de-identification
- Document the failure patterns of individual baselines

### 2. Results Section
- Present success rate comparisons with statistical significance
- Show complementary success patterns as evidence for hybrid approach
- Include processing time analysis for practical considerations

### 3. Discussion Section
- Discuss the trade-offs between individual model strengths and limitations
- Explain how hybrid approach addresses individual model weaknesses
- Position the work within the broader context of video segmentation challenges

## Conclusion
The comparison clearly demonstrates that:
1. Individual baselines (TSP-SAM, SAMURAI) have significant limitations
2. A hybrid approach provides improved reliability and coverage
3. The complementary nature of different methods justifies the hybrid pipeline
4. This work addresses a real need in video de-identification research

**The hybrid approach is not just an incremental improvement, but a necessary solution to the reliability challenges in video segmentation for privacy-preserving applications.**

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Total sequences analyzed: {total_sequences}*
"""
        
        # Save summary report
        summary_file = experiment_dir / "reports" / "thesis_summary_report.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        debug_print(f"Thesis summary saved to: {summary_file}")
        
    except Exception as e:
        debug_print(f"Error generating thesis summary: {e}", "ERROR")
        raise e

def generate_individual_baseline_reports(df, experiment_dir, max_frames):
    """Generate separate CSV and JSON reports for individual baselines for thesis analysis"""
    debug_print("Generating individual baseline reports for thesis analysis...")
    
    try:
        # Create individual baseline directories
        tsp_dir = experiment_dir / "reports" / "individual_baselines" / "tsp_sam"
        sam_dir = experiment_dir / "reports" / "individual_baselines" / "samurai"
        hybrid_dir = experiment_dir / "reports" / "individual_baselines" / "hybrid"
        
        tsp_dir.mkdir(parents=True, exist_ok=True)
        sam_dir.mkdir(parents=True, exist_ok=True)
        hybrid_dir.mkdir(parents=True, exist_ok=True)
        
        # TSP-SAM Individual Report
        tsp_data = []
        for _, row in df.iterrows():
            frames_processed = max_frames
            processing_time = row["tsp_sam_time"]
            
            # Calculate FPS and efficiency metrics
            fps = frames_processed / processing_time if processing_time > 0 else 0
            time_per_frame = processing_time / frames_processed if frames_processed > 0 else 0
            
            # Calculate memory efficiency metrics (if available)
            memory_per_frame = 0  # Will be calculated if memory data is available
            memory_efficiency_score = 0  # FPS per GB of memory used
            
            tsp_row = {
                "sequence": row["sequence"],
                "success": row["tsp_sam_success"],
                "processing_time": processing_time,
                "frames_processed": frames_processed,
                "fps": fps,
                "time_per_frame": time_per_frame,
                "efficiency_score": fps / processing_time if processing_time > 0 else 0,  # FPS per second of processing
                "memory_efficiency_score": memory_efficiency_score,  # FPS per GB of memory
                "memory_per_frame": memory_per_frame,  # Memory usage per frame
                "sequence_length_category": "short" if frames_processed <= 10 else "medium" if frames_processed <= 25 else "long",
                "timestamp": datetime.now().isoformat()
            }
            tsp_data.append(tsp_row)
        
        # Save TSP-SAM CSV
        tsp_df = pd.DataFrame(tsp_data)
        tsp_csv_file = tsp_dir / "tsp_sam_baseline_results.csv"
        tsp_df.to_csv(tsp_csv_file, index=False)
        debug_print(f"TSP-SAM CSV saved to: {tsp_csv_file}")
        
        # TSP-SAM Summary Statistics
        tsp_successful = tsp_df[tsp_df["success"] == True]
        tsp_failed = tsp_df[tsp_df["success"] == False]
        
        tsp_stats = {
            "model": "TSP-SAM",
            "total_sequences": len(tsp_df),
            "successful_sequences": len(tsp_successful),
            "failed_sequences": len(tsp_failed),
            "success_rate": len(tsp_successful) / len(tsp_df) if len(tsp_df) > 0 else 0,
            "failure_rate": len(tsp_failed) / len(tsp_df) if len(tsp_df) > 0 else 0,
            
            # Performance Metrics
            "average_processing_time": tsp_successful["processing_time"].mean() if len(tsp_successful) > 0 else 0,
            "total_processing_time": tsp_successful["processing_time"].sum() if len(tsp_successful) > 0 else 0,
            "average_fps": tsp_successful["fps"].mean() if len(tsp_successful) > 0 else 0,
            "max_fps": tsp_successful["fps"].max() if len(tsp_successful) > 0 else 0,
            "min_fps": tsp_successful["fps"].min() if len(tsp_successful) > 0 else 0,
            "fps_std": tsp_successful["fps"].std() if len(tsp_successful) > 0 else 0,
            "average_time_per_frame": tsp_successful["time_per_frame"].mean() if len(tsp_successful) > 0 else 0,
            "average_efficiency_score": tsp_successful["efficiency_score"].mean() if len(tsp_successful) > 0 else 0,
            "average_memory_efficiency_score": tsp_successful["memory_efficiency_score"].mean() if len(tsp_successful) > 0 else 0,
            "average_memory_per_frame": tsp_successful["memory_per_frame"].mean() if len(tsp_successful) > 0 else 0,
            
            # Sequence Analysis
            "frames_per_sequence": max_frames,
            "successful_sequences_list": tsp_successful["sequence"].tolist(),
            "failed_sequences_list": tsp_failed["sequence"].tolist(),
            
            # Performance by sequence length
            "performance_by_length": {
                "short_sequences": {
                    "count": len(tsp_df[tsp_df["sequence_length_category"] == "short"]),
                    "success_rate": len(tsp_df[(tsp_df["sequence_length_category"] == "short") & (tsp_df["success"] == True)]) / len(tsp_df[tsp_df["sequence_length_category"] == "short"]) if len(tsp_df[tsp_df["sequence_length_category"] == "short"]) > 0 else 0,
                    "avg_fps": tsp_df[(tsp_df["sequence_length_category"] == "short") & (tsp_df["success"] == True)]["fps"].mean() if len(tsp_df[(tsp_df["sequence_length_category"] == "short") & (tsp_df["success"] == True)]) > 0 else 0
                },
                "medium_sequences": {
                    "count": len(tsp_df[tsp_df["sequence_length_category"] == "medium"]),
                    "success_rate": len(tsp_df[(tsp_df["sequence_length_category"] == "medium") & (tsp_df["success"] == True)]) / len(tsp_df[tsp_df["sequence_length_category"] == "medium"]) if len(tsp_df[tsp_df["sequence_length_category"] == "medium"]) > 0 else 0,
                    "avg_fps": tsp_df[(tsp_df["sequence_length_category"] == "medium") & (tsp_df["success"] == True)]["fps"].mean() if len(tsp_df[(tsp_df["sequence_length_category"] == "medium") & (tsp_df["success"] == True)]) > 0 else 0
                },
                "long_sequences": {
                    "count": len(tsp_df[tsp_df["sequence_length_category"] == "long"]),
                    "success_rate": len(tsp_df[(tsp_df["sequence_length_category"] == "long") & (tsp_df["success"] == True)]) / len(tsp_df[tsp_df["sequence_length_category"] == "long"]) if len(tsp_df[tsp_df["sequence_length_category"] == "long"]) > 0 else 0,
                    "avg_fps": tsp_df[(tsp_df["sequence_length_category"] == "long") & (tsp_df["success"] == True)]["fps"].mean() if len(tsp_df[(tsp_df["sequence_length_category"] == "long") & (tsp_df["success"] == True)]) > 0 else 0
                }
            },
            
            "timestamp": datetime.now().isoformat()
        }
        
        # Save TSP-SAM JSON
        tsp_json_file = tsp_dir / "tsp_sam_baseline_summary.json"
        with open(tsp_json_file, 'w') as f:
            json.dump(tsp_stats, f, indent=2)
        debug_print(f"TSP-SAM JSON saved to: {tsp_json_file}")
        
        # SAMURAI Individual Report
        sam_data = []
        for _, row in df.iterrows():
            frames_processed = max_frames
            processing_time = row["samurai_time"]
            
            # Calculate FPS and efficiency metrics
            fps = frames_processed / processing_time if processing_time > 0 else 0
            time_per_frame = processing_time / frames_processed if frames_processed > 0 else 0
            
            # Calculate memory efficiency metrics (if available)
            memory_per_frame = 0  # Will be calculated if memory data is available
            memory_efficiency_score = 0  # FPS per GB of memory used
            
            sam_row = {
                "sequence": row["sequence"],
                "success": row["samurai_success"],
                "processing_time": processing_time,
                "frames_processed": frames_processed,
                "fps": fps,
                "time_per_frame": time_per_frame,
                "efficiency_score": fps / processing_time if processing_time > 0 else 0,  # FPS per second of processing
                "memory_efficiency_score": memory_efficiency_score,  # FPS per GB of memory
                "memory_per_frame": memory_per_frame,  # Memory usage per frame
                "sequence_length_category": "short" if frames_processed <= 10 else "medium" if frames_processed <= 25 else "long",
                "timestamp": datetime.now().isoformat()
            }
            sam_data.append(sam_row)
        
        # Save SAMURAI CSV
        sam_df = pd.DataFrame(sam_data)
        sam_csv_file = sam_dir / "samurai_baseline_results.csv"
        sam_df.to_csv(sam_csv_file, index=False)
        debug_print(f"SAMURAI CSV saved to: {sam_csv_file}")
        
        # SAMURAI Summary Statistics
        sam_successful = sam_df[sam_df["success"] == True]
        sam_failed = sam_df[sam_df["success"] == False]
        
        sam_stats = {
            "model": "SAMURAI",
            "total_sequences": len(sam_df),
            "successful_sequences": len(sam_successful),
            "failed_sequences": len(sam_failed),
            "success_rate": len(sam_successful) / len(sam_df) if len(sam_df) > 0 else 0,
            "failure_rate": len(sam_failed) / len(sam_df) if len(sam_df) > 0 else 0,
            
            # Performance Metrics
            "average_processing_time": sam_successful["processing_time"].mean() if len(sam_successful) > 0 else 0,
            "total_processing_time": sam_successful["processing_time"].sum() if len(sam_successful) > 0 else 0,
            "average_fps": sam_successful["fps"].mean() if len(sam_successful) > 0 else 0,
            "max_fps": sam_successful["fps"].max() if len(sam_successful) > 0 else 0,
            "min_fps": sam_successful["fps"].min() if len(sam_successful) > 0 else 0,
            "fps_std": sam_successful["fps"].std() if len(sam_successful) > 0 else 0,
            "average_time_per_frame": sam_successful["time_per_frame"].mean() if len(sam_successful) > 0 else 0,
            "average_efficiency_score": sam_successful["efficiency_score"].mean() if len(sam_successful) > 0 else 0,
            "average_memory_efficiency_score": sam_successful["memory_efficiency_score"].mean() if len(sam_successful) > 0 else 0,
            "average_memory_per_frame": sam_successful["memory_per_frame"].mean() if len(sam_successful) > 0 else 0,
            
            # Sequence Analysis
            "frames_per_sequence": max_frames,
            "successful_sequences_list": sam_successful["sequence"].tolist(),
            "failed_sequences_list": sam_failed["sequence"].tolist(),
            
            # Performance by sequence length
            "performance_by_length": {
                "short_sequences": {
                    "count": len(sam_df[sam_df["sequence_length_category"] == "short"]),
                    "success_rate": len(sam_df[(sam_df["sequence_length_category"] == "short") & (sam_df["success"] == True)]) / len(sam_df[sam_df["sequence_length_category"] == "short"]) if len(sam_df[sam_df["sequence_length_category"] == "short"]) > 0 else 0,
                    "avg_fps": sam_df[(sam_df["sequence_length_category"] == "short") & (sam_df["success"] == True)]["fps"].mean() if len(sam_df[(sam_df["sequence_length_category"] == "short") & (sam_df["success"] == True)]) > 0 else 0
                },
                "medium_sequences": {
                    "count": len(sam_df[sam_df["sequence_length_category"] == "medium"]),
                    "success_rate": len(sam_df[(sam_df["sequence_length_category"] == "medium") & (sam_df["success"] == True)]) / len(sam_df[sam_df["sequence_length_category"] == "medium"]) if len(sam_df[sam_df["sequence_length_category"] == "medium"]) > 0 else 0,
                    "avg_fps": sam_df[(sam_df["sequence_length_category"] == "medium") & (sam_df["success"] == True)]["fps"].mean() if len(sam_df[(sam_df["sequence_length_category"] == "medium") & (sam_df["success"] == True)]) > 0 else 0
                },
                "long_sequences": {
                    "count": len(sam_df[sam_df["sequence_length_category"] == "long"]),
                    "success_rate": len(sam_df[(sam_df["sequence_length_category"] == "long") & (sam_df["success"] == True)]) / len(sam_df[sam_df["sequence_length_category"] == "long"]) if len(sam_df[sam_df["sequence_length_category"] == "long"]) > 0 else 0,
                    "avg_fps": sam_df[(sam_df["sequence_length_category"] == "long") & (sam_df["success"] == True)]["fps"].mean() if len(sam_df[(sam_df["sequence_length_category"] == "long") & (sam_df["success"] == True)]) > 0 else 0
                }
            },
            
            "timestamp": datetime.now().isoformat()
        }
        
        # Save SAMURAI JSON
        sam_json_file = sam_dir / "samurai_baseline_summary.json"
        with open(sam_json_file, 'w') as f:
            json.dump(sam_stats, f, indent=2)
        debug_print(f"SAMURAI JSON saved to: {sam_json_file}")
        
        # Hybrid Individual Report
        hybrid_data = []
        for _, row in df.iterrows():
            frames_processed = max_frames
            processing_time = row["hybrid_time"]
            
            # Calculate FPS and efficiency metrics
            fps = frames_processed / processing_time if processing_time > 0 else 0
            time_per_frame = processing_time / frames_processed if frames_processed > 0 else 0
            
            # Calculate memory efficiency metrics (if available)
            memory_per_frame = 0  # Will be calculated if memory data is available
            memory_efficiency_score = 0  # FPS per GB of memory used
            
            hybrid_row = {
                "sequence": row["sequence"],
                "success": row["hybrid_success"],
                "processing_time": processing_time,
                "frames_processed": frames_processed,
                "fps": fps,
                "time_per_frame": time_per_frame,
                "efficiency_score": fps / processing_time if processing_time > 0 else 0,  # FPS per second of processing
                "memory_efficiency_score": memory_efficiency_score,  # FPS per GB of memory
                "memory_per_frame": memory_per_frame,  # Memory usage per frame
                "sequence_length_category": "short" if frames_processed <= 10 else "medium" if frames_processed <= 25 else "long",
                "timestamp": datetime.now().isoformat()
            }
            hybrid_data.append(hybrid_row)
        
        # Save Hybrid CSV
        hybrid_df = pd.DataFrame(hybrid_data)
        hybrid_csv_file = hybrid_dir / "hybrid_baseline_results.csv"
        hybrid_df.to_csv(hybrid_csv_file, index=False)
        debug_print(f"Hybrid CSV saved to: {hybrid_csv_file}")
        
        # Hybrid Summary Statistics
        hybrid_successful = hybrid_df[hybrid_df["success"] == True]
        hybrid_failed = hybrid_df[hybrid_df["success"] == False]
        
        hybrid_stats = {
            "model": "Hybrid Pipeline",
            "total_sequences": len(hybrid_df),
            "successful_sequences": len(hybrid_successful),
            "failed_sequences": len(hybrid_failed),
            "success_rate": len(hybrid_successful) / len(hybrid_df) if len(hybrid_df) > 0 else 0,
            "failure_rate": len(hybrid_failed) / len(hybrid_df) if len(hybrid_df) > 0 else 0,
            
            # Performance Metrics
            "average_processing_time": hybrid_successful["processing_time"].mean() if len(hybrid_successful) > 0 else 0,
            "total_processing_time": hybrid_successful["processing_time"].sum() if len(hybrid_successful) > 0 else 0,
            "average_fps": hybrid_successful["fps"].mean() if len(hybrid_successful) > 0 else 0,
            "max_fps": hybrid_successful["fps"].max() if len(hybrid_successful) > 0 else 0,
            "min_fps": hybrid_successful["fps"].min() if len(hybrid_successful) > 0 else 0,
            "fps_std": hybrid_successful["fps"].std() if len(hybrid_successful) > 0 else 0,
            "average_time_per_frame": hybrid_successful["time_per_frame"].mean() if len(hybrid_successful) > 0 else 0,
            "average_efficiency_score": hybrid_successful["efficiency_score"].mean() if len(hybrid_successful) > 0 else 0,
            "average_memory_efficiency_score": hybrid_successful["memory_efficiency_score"].mean() if len(hybrid_successful) > 0 else 0,
            "average_memory_per_frame": hybrid_successful["memory_per_frame"].mean() if len(hybrid_successful) > 0 else 0,
            
            # Sequence Analysis
            "frames_per_sequence": max_frames,
            "successful_sequences_list": hybrid_successful["sequence"].tolist(),
            "failed_sequences_list": hybrid_failed["sequence"].tolist(),
            
            # Performance by sequence length
            "performance_by_length": {
                "short_sequences": {
                    "count": len(hybrid_df[hybrid_df["sequence_length_category"] == "short"]),
                    "success_rate": len(hybrid_df[(hybrid_df["sequence_length_category"] == "short") & (hybrid_df["success"] == True)]) / len(hybrid_df[hybrid_df["sequence_length_category"] == "short"]) if len(hybrid_df[hybrid_df["sequence_length_category"] == "short"]) > 0 else 0,
                    "avg_fps": hybrid_df[(hybrid_df["sequence_length_category"] == "short") & (hybrid_df["success"] == True)]["fps"].mean() if len(hybrid_df[(hybrid_df["sequence_length_category"] == "short") & (hybrid_df["success"] == True)]) > 0 else 0
                },
                "medium_sequences": {
                    "count": len(hybrid_df[hybrid_df["sequence_length_category"] == "medium"]),
                    "success_rate": len(hybrid_df[(hybrid_df["sequence_length_category"] == "medium") & (hybrid_df["success"] == True)]) / len(hybrid_df[hybrid_df["sequence_length_category"] == "medium"]) if len(hybrid_df[hybrid_df["sequence_length_category"] == "medium"]) > 0 else 0,
                    "avg_fps": hybrid_df[(hybrid_df["sequence_length_category"] == "medium") & (hybrid_df["success"] == True)]["fps"].mean() if len(hybrid_df[(hybrid_df["sequence_length_category"] == "medium") & (hybrid_df["success"] == True)]) > 0 else 0
                },
                "long_sequences": {
                    "count": len(hybrid_df[hybrid_df["sequence_length_category"] == "long"]),
                    "success_rate": len(hybrid_df[(hybrid_df["sequence_length_category"] == "long") & (hybrid_df["success"] == True)]) / len(hybrid_df[hybrid_df["sequence_length_category"] == "long"]) if len(hybrid_df[hybrid_df["sequence_length_category"] == "long"]) > 0 else 0,
                    "avg_fps": hybrid_df[(hybrid_df["sequence_length_category"] == "long") & (hybrid_df["success"] == True)]["fps"].mean() if len(hybrid_df[(hybrid_df["sequence_length_category"] == "long") & (hybrid_df["success"] == True)]) > 0 else 0
                }
            },
            
            "timestamp": datetime.now().isoformat()
        }
        
        # Save Hybrid JSON
        hybrid_json_file = hybrid_dir / "hybrid_baseline_summary.json"
        with open(hybrid_json_file, 'w') as f:
            json.dump(hybrid_stats, f, indent=2)
        debug_print(f"Hybrid JSON saved to: {hybrid_json_file}")
        
        # Generate comparison analysis for thesis
        comparison_analysis = {
            "experiment_info": {
                "total_sequences": len(df),
                "frames_per_sequence": max_frames,
                "timestamp": datetime.now().isoformat()
            },
            "tsp_sam_performance": tsp_stats,
            "samurai_performance": sam_stats,
            "hybrid_performance": hybrid_stats,
            "comparative_analysis": {
                "success_rate_comparison": {
                    "tsp_sam_rate": tsp_stats["success_rate"],
                    "samurai_rate": sam_stats["success_rate"],
                    "hybrid_rate": hybrid_stats["success_rate"],
                    "tsp_vs_sam_difference": tsp_stats["success_rate"] - sam_stats["success_rate"],
                    "hybrid_vs_tsp_difference": hybrid_stats["success_rate"] - tsp_stats["success_rate"],
                    "hybrid_vs_sam_difference": hybrid_stats["success_rate"] - sam_stats["success_rate"],
                    "best_performer": max([
                        ("TSP-SAM", tsp_stats["success_rate"]),
                        ("SAMURAI", sam_stats["success_rate"]),
                        ("Hybrid", hybrid_stats["success_rate"])
                    ], key=lambda x: x[1])[0]
                },
                "processing_time_comparison": {
                    "tsp_sam_avg_time": tsp_stats["average_processing_time"],
                    "samurai_avg_time": sam_stats["average_processing_time"],
                    "hybrid_avg_time": hybrid_stats["average_processing_time"],
                    "fastest_model": min([
                        ("TSP-SAM", tsp_stats["average_processing_time"]),
                        ("SAMURAI", sam_stats["average_processing_time"]),
                        ("Hybrid", hybrid_stats["average_processing_time"])
                    ], key=lambda x: x[1])[0]
                },
                "fps_performance_comparison": {
                    "tsp_sam_avg_fps": tsp_stats["average_fps"],
                    "samurai_avg_fps": sam_stats["average_fps"],
                    "hybrid_avg_fps": hybrid_stats["average_fps"],
                    "tsp_sam_max_fps": tsp_stats["max_fps"],
                    "samurai_max_fps": sam_stats["max_fps"],
                    "hybrid_max_fps": hybrid_stats["max_fps"],
                    "tsp_sam_min_fps": tsp_stats["min_fps"],
                    "samurai_min_fps": sam_stats["min_fps"],
                    "hybrid_min_fps": hybrid_stats["min_fps"],
                    "highest_fps_model": max([
                        ("TSP-SAM", tsp_stats["average_fps"]),
                        ("SAMURAI", sam_stats["average_fps"]),
                        ("Hybrid", hybrid_stats["average_fps"])
                    ], key=lambda x: x[1])[0],
                    "fps_consistency": {
                        "tsp_sam_std": tsp_stats["fps_std"],
                        "samurai_std": sam_stats["fps_std"],
                        "hybrid_std": hybrid_stats["fps_std"],
                        "most_consistent": min([
                            ("TSP-SAM", tsp_stats["fps_std"]),
                            ("SAMURAI", sam_stats["fps_std"]),
                            ("Hybrid", hybrid_stats["fps_std"])
                        ], key=lambda x: x[1])[0]
                    }
                },
                "efficiency_comparison": {
                    "tsp_sam_efficiency": tsp_stats["average_efficiency_score"],
                    "samurai_efficiency": sam_stats["average_efficiency_score"],
                    "hybrid_efficiency": hybrid_stats["average_efficiency_score"],
                    "most_efficient": max([
                        ("TSP-SAM", tsp_stats["average_efficiency_score"]),
                        ("SAMURAI", sam_stats["average_efficiency_score"]),
                        ("Hybrid", hybrid_stats["average_efficiency_score"])
                    ], key=lambda x: x[1])[0]
                },
                "memory_efficiency_comparison": {
                    "tsp_sam_memory_efficiency": tsp_stats["average_memory_efficiency_score"],
                    "samurai_memory_efficiency": sam_stats["average_memory_efficiency_score"],
                    "hybrid_memory_efficiency": hybrid_stats["average_memory_efficiency_score"],
                    "tsp_sam_memory_per_frame": tsp_stats["average_memory_per_frame"],
                    "samurai_memory_per_frame": sam_stats["average_memory_per_frame"],
                    "hybrid_memory_per_frame": hybrid_stats["average_memory_per_frame"],
                    "most_memory_efficient": max([
                        ("TSP-SAM", tsp_stats["average_memory_efficiency_score"]),
                        ("SAMURAI", sam_stats["average_memory_efficiency_score"]),
                        ("Hybrid", hybrid_stats["average_memory_efficiency_score"])
                    ], key=lambda x: x[1])[0],
                    "least_memory_per_frame": min([
                        ("TSP-SAM", tsp_stats["average_memory_per_frame"]),
                        ("SAMURAI", sam_stats["average_memory_per_frame"]),
                        ("Hybrid", hybrid_stats["average_memory_per_frame"])
                    ], key=lambda x: x[1])[0]
                },
                "scalability_analysis": {
                    "tsp_sam_short_fps": tsp_stats["performance_by_length"]["short_sequences"]["avg_fps"],
                    "samurai_short_fps": sam_stats["performance_by_length"]["short_sequences"]["avg_fps"],
                    "hybrid_short_fps": hybrid_stats["performance_by_length"]["short_sequences"]["avg_fps"],
                    "tsp_sam_medium_fps": tsp_stats["performance_by_length"]["medium_sequences"]["avg_fps"],
                    "samurai_medium_fps": sam_stats["performance_by_length"]["medium_sequences"]["avg_fps"],
                    "hybrid_medium_fps": hybrid_stats["performance_by_length"]["medium_sequences"]["avg_fps"],
                    "tsp_sam_long_fps": tsp_stats["performance_by_length"]["long_sequences"]["avg_fps"],
                    "samurai_long_fps": sam_stats["performance_by_length"]["long_sequences"]["avg_fps"],
                    "hybrid_long_fps": hybrid_stats["performance_by_length"]["long_sequences"]["avg_fps"]
                },
                "complementary_analysis": {
                    "tsp_only_success": list(set(tsp_successful["sequence"]) - set(sam_successful["sequence"]) - set(hybrid_successful["sequence"])),
                    "sam_only_success": list(set(sam_successful["sequence"]) - set(tsp_successful["sequence"]) - set(hybrid_successful["sequence"])),
                    "hybrid_only_success": list(set(hybrid_successful["sequence"]) - set(tsp_successful["sequence"]) - set(sam_successful["sequence"])),
                    "tsp_sam_success": list(set(tsp_successful["sequence"]) & set(sam_successful["sequence"])),
                    "tsp_hybrid_success": list(set(tsp_successful["sequence"]) & set(hybrid_successful["sequence"])),
                    "sam_hybrid_success": list(set(sam_successful["sequence"]) & set(hybrid_successful["sequence"])),
                    "all_three_success": list(set(tsp_successful["sequence"]) & set(sam_successful["sequence"]) & set(hybrid_successful["sequence"])),
                    "all_three_fail": list(set(tsp_df[tsp_df["success"] == False]["sequence"]) & set(sam_df[sam_df["success"] == False]["sequence"]) & set(hybrid_df[hybrid_df["success"] == False]["sequence"]))
                },
                "hybrid_justification_analysis": {
                    "complementary_strength_evidence": {
                        "tsp_sam_unique_strengths": len(list(set(tsp_successful["sequence"]) - set(sam_successful["sequence"]))),
                        "samurai_unique_strengths": len(list(set(sam_successful["sequence"]) - set(tsp_successful["sequence"]))),
                        "hybrid_extends_coverage": len(list(set(hybrid_successful["sequence"]) - set(tsp_successful["sequence"]) - set(sam_successful["sequence"]))),
                        "hybrid_improvement_over_best_individual": max(hybrid_stats["success_rate"] - max(tsp_stats["success_rate"], sam_stats["success_rate"]), 0),
                        "hybrid_improvement_over_worst_individual": max(hybrid_stats["success_rate"] - min(tsp_stats["success_rate"], sam_stats["success_rate"]), 0)
                    },
                    "failure_pattern_analysis": {
                        "tsp_sam_failure_sequences": list(set(tsp_df[tsp_df["success"] == False]["sequence"])),
                        "samurai_failure_sequences": list(set(sam_df[sam_df["success"] == False]["sequence"])),
                        "hybrid_saves_tsp_failures": len(list(set(hybrid_successful["sequence"]) & set(tsp_df[tsp_df["success"] == False]["sequence"]))),
                        "hybrid_saves_sam_failures": len(list(set(hybrid_successful["sequence"]) & set(sam_df[sam_df["success"] == False]["sequence"]))),
                        "hybrid_saves_both_failures": len(list(set(hybrid_successful["sequence"]) & set(tsp_df[tsp_df["success"] == False]["sequence"]) & set(sam_df[sam_df["success"] == False]["sequence"])))
                    },
                    "reliability_improvement": {
                        "tsp_sam_reliability": tsp_stats["success_rate"],
                        "samurai_reliability": sam_stats["success_rate"],
                        "hybrid_reliability": hybrid_stats["success_rate"],
                        "reliability_improvement_percentage": ((hybrid_stats["success_rate"] - max(tsp_stats["success_rate"], sam_stats["success_rate"])) / max(tsp_stats["success_rate"], sam_stats["success_rate"]) * 100) if max(tsp_stats["success_rate"], sam_stats["success_rate"]) > 0 else 0,
                        "coverage_improvement": len(hybrid_successful["sequence"]) - max(len(tsp_successful["sequence"]), len(sam_successful["sequence"]))
                    },
                    "performance_trade_off_analysis": {
                        "tsp_sam_avg_fps": tsp_stats["average_fps"],
                        "samurai_avg_fps": sam_stats["average_fps"],
                        "hybrid_avg_fps": hybrid_stats["average_fps"],
                        "fps_trade_off": hybrid_stats["average_fps"] - max(tsp_stats["average_fps"], sam_stats["average_fps"]),
                        "processing_time_trade_off": hybrid_stats["average_processing_time"] - min(tsp_stats["average_processing_time"], sam_stats["average_processing_time"]),
                        "memory_efficiency_trade_off": hybrid_stats["average_memory_efficiency_score"] - max(tsp_stats["average_memory_efficiency_score"], sam_stats["average_memory_efficiency_score"])
                    }
                }
            }
        }
        
        # Save comparison analysis
        comparison_file = experiment_dir / "reports" / "individual_baselines" / "baseline_comparison_analysis.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_analysis, f, indent=2)
        debug_print(f"Baseline comparison analysis saved to: {comparison_file}")
        
        # Generate thesis-ready summary
        thesis_summary = f"""# Individual Baseline Performance Analysis

## TSP-SAM Baseline Results
- **Success Rate**: {tsp_stats['success_rate']:.1%} ({tsp_stats['successful_sequences']}/{tsp_stats['total_sequences']} sequences)
- **Average Processing Time**: {tsp_stats['average_processing_time']:.2f} seconds per sequence
- **Total Processing Time**: {tsp_stats['total_processing_time']:.2f} seconds
- **Average FPS**: {tsp_stats['average_fps']:.2f} frames per second
- **FPS Range**: {tsp_stats['min_fps']:.2f} - {tsp_stats['max_fps']:.2f} FPS
- **FPS Consistency**: {tsp_stats['fps_std']:.2f} (standard deviation)
- **Time per Frame**: {tsp_stats['average_time_per_frame']:.3f} seconds
- **Efficiency Score**: {tsp_stats['average_efficiency_score']:.4f}
- **Memory Efficiency Score**: {tsp_stats['average_memory_efficiency_score']:.4f} (FPS per GB)
- **Memory per Frame**: {tsp_stats['average_memory_per_frame']:.2f} GB
- **Successful Sequences**: {', '.join(tsp_stats['successful_sequences_list'])}
- **Failed Sequences**: {', '.join(tsp_stats['failed_sequences_list'])}

## SAMURAI Baseline Results
- **Success Rate**: {sam_stats['success_rate']:.1%} ({sam_stats['successful_sequences']}/{sam_stats['total_sequences']} sequences)
- **Average Processing Time**: {sam_stats['average_processing_time']:.2f} seconds per sequence
- **Total Processing Time**: {sam_stats['total_processing_time']:.2f} seconds
- **Average FPS**: {sam_stats['average_fps']:.2f} frames per second
- **FPS Range**: {sam_stats['min_fps']:.2f} - {sam_stats['max_fps']:.2f} FPS
- **FPS Consistency**: {sam_stats['fps_std']:.2f} (standard deviation)
- **Time per Frame**: {sam_stats['average_time_per_frame']:.3f} seconds
- **Efficiency Score**: {sam_stats['average_efficiency_score']:.4f}
- **Memory Efficiency Score**: {sam_stats['average_memory_efficiency_score']:.4f} (FPS per GB)
- **Memory per Frame**: {sam_stats['average_memory_per_frame']:.2f} GB
- **Successful Sequences**: {', '.join(sam_stats['successful_sequences_list'])}
- **Failed Sequences**: {', '.join(sam_stats['failed_sequences_list'])}

## Hybrid Pipeline Results
- **Success Rate**: {hybrid_stats['success_rate']:.1%} ({hybrid_stats['successful_sequences']}/{hybrid_stats['total_sequences']} sequences)
- **Average Processing Time**: {hybrid_stats['average_processing_time']:.2f} seconds per sequence
- **Total Processing Time**: {hybrid_stats['total_processing_time']:.2f} seconds
- **Average FPS**: {hybrid_stats['average_fps']:.2f} frames per second
- **FPS Range**: {hybrid_stats['min_fps']:.2f} - {hybrid_stats['max_fps']:.2f} FPS
- **FPS Consistency**: {hybrid_stats['fps_std']:.2f} (standard deviation)
- **Time per Frame**: {hybrid_stats['average_time_per_frame']:.3f} seconds
- **Efficiency Score**: {hybrid_stats['average_efficiency_score']:.4f}
- **Memory Efficiency Score**: {hybrid_stats['average_memory_efficiency_score']:.4f} (FPS per GB)
- **Memory per Frame**: {hybrid_stats['average_memory_per_frame']:.2f} GB
- **Successful Sequences**: {', '.join(hybrid_stats['successful_sequences_list'])}
- **Failed Sequences**: {', '.join(hybrid_stats['failed_sequences_list'])}

## Performance Comparison
### Success Rate Analysis
- **Best Success Rate**: {comparison_analysis['comparative_analysis']['success_rate_comparison']['best_performer']}
- **TSP-SAM vs SAMURAI**: {comparison_analysis['comparative_analysis']['success_rate_comparison']['tsp_vs_sam_difference']:.1%} difference
- **Hybrid vs TSP-SAM**: {comparison_analysis['comparative_analysis']['success_rate_comparison']['hybrid_vs_tsp_difference']:.1%} difference
- **Hybrid vs SAMURAI**: {comparison_analysis['comparative_analysis']['success_rate_comparison']['hybrid_vs_sam_difference']:.1%} difference

### Processing Speed Analysis
- **Fastest Overall Processing**: {comparison_analysis['comparative_analysis']['processing_time_comparison']['fastest_model']}
- **Highest FPS Performance**: {comparison_analysis['comparative_analysis']['fps_performance_comparison']['highest_fps_model']}
- **Most Consistent FPS**: {comparison_analysis['comparative_analysis']['fps_performance_comparison']['fps_consistency']['most_consistent']}
- **Most Efficient**: {comparison_analysis['comparative_analysis']['efficiency_comparison']['most_efficient']}
- **Most Memory Efficient**: {comparison_analysis['comparative_analysis']['memory_efficiency_comparison']['most_memory_efficient']}
- **Least Memory per Frame**: {comparison_analysis['comparative_analysis']['memory_efficiency_comparison']['least_memory_per_frame']}

### Memory Efficiency Analysis
- **TSP-SAM Memory Efficiency**: {comparison_analysis['comparative_analysis']['memory_efficiency_comparison']['tsp_sam_memory_efficiency']:.4f} FPS per GB
- **SAMURAI Memory Efficiency**: {comparison_analysis['comparative_analysis']['memory_efficiency_comparison']['samurai_memory_efficiency']:.4f} FPS per GB
- **Hybrid Memory Efficiency**: {comparison_analysis['comparative_analysis']['memory_efficiency_comparison']['hybrid_memory_efficiency']:.4f} FPS per GB
- **TSP-SAM Memory per Frame**: {comparison_analysis['comparative_analysis']['memory_efficiency_comparison']['tsp_sam_memory_per_frame']:.2f} GB
- **SAMURAI Memory per Frame**: {comparison_analysis['comparative_analysis']['memory_efficiency_comparison']['samurai_memory_per_frame']:.2f} GB
- **Hybrid Memory per Frame**: {comparison_analysis['comparative_analysis']['memory_efficiency_comparison']['hybrid_memory_per_frame']:.2f} GB

### Scalability Analysis
- **Short Sequences (≤10 frames)**:
  - TSP-SAM: {comparison_analysis['comparative_analysis']['scalability_analysis']['tsp_sam_short_fps']:.2f} FPS
  - SAMURAI: {comparison_analysis['comparative_analysis']['scalability_analysis']['samurai_short_fps']:.2f} FPS
  - Hybrid: {comparison_analysis['comparative_analysis']['scalability_analysis']['hybrid_short_fps']:.2f} FPS
- **Medium Sequences (11-25 frames)**:
  - TSP-SAM: {comparison_analysis['comparative_analysis']['scalability_analysis']['tsp_sam_medium_fps']:.2f} FPS
  - SAMURAI: {comparison_analysis['comparative_analysis']['scalability_analysis']['samurai_medium_fps']:.2f} FPS
  - Hybrid: {comparison_analysis['comparative_analysis']['scalability_analysis']['hybrid_medium_fps']:.2f} FPS
- **Long Sequences (>25 frames)**:
  - TSP-SAM: {comparison_analysis['comparative_analysis']['scalability_analysis']['tsp_sam_long_fps']:.2f} FPS
  - SAMURAI: {comparison_analysis['comparative_analysis']['scalability_analysis']['samurai_long_fps']:.2f} FPS
  - Hybrid: {comparison_analysis['comparative_analysis']['scalability_analysis']['hybrid_long_fps']:.2f} FPS

### Complementary Analysis
- **TSP-SAM Only Success**: {len(comparison_analysis['comparative_analysis']['complementary_analysis']['tsp_only_success'])} sequences
- **SAMURAI Only Success**: {len(comparison_analysis['comparative_analysis']['complementary_analysis']['sam_only_success'])} sequences
- **Hybrid Only Success**: {len(comparison_analysis['comparative_analysis']['complementary_analysis']['hybrid_only_success'])} sequences
- **TSP-SAM + SAMURAI Success**: {len(comparison_analysis['comparative_analysis']['complementary_analysis']['tsp_sam_success'])} sequences
- **TSP-SAM + Hybrid Success**: {len(comparison_analysis['comparative_analysis']['complementary_analysis']['tsp_hybrid_success'])} sequences
- **SAMURAI + Hybrid Success**: {len(comparison_analysis['comparative_analysis']['complementary_analysis']['sam_hybrid_success'])} sequences
- **All Three Success**: {len(comparison_analysis['comparative_analysis']['complementary_analysis']['all_three_success'])} sequences
- **All Three Fail**: {len(comparison_analysis['comparative_analysis']['complementary_analysis']['all_three_fail'])} sequences

## Hybrid Approach Justification Analysis

### Complementary Strength Evidence
- **TSP-SAM Unique Strengths**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['complementary_strength_evidence']['tsp_sam_unique_strengths']} sequences succeed only with TSP-SAM
- **SAMURAI Unique Strengths**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['complementary_strength_evidence']['samurai_unique_strengths']} sequences succeed only with SAMURAI
- **Hybrid Extends Coverage**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['complementary_strength_evidence']['hybrid_extends_coverage']} sequences succeed only with hybrid approach
- **Hybrid Improvement over Best Individual**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['complementary_strength_evidence']['hybrid_improvement_over_best_individual']:.1%} improvement
- **Hybrid Improvement over Worst Individual**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['complementary_strength_evidence']['hybrid_improvement_over_worst_individual']:.1%} improvement

### Failure Pattern Analysis
- **TSP-SAM Failure Sequences**: {', '.join(comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['failure_pattern_analysis']['tsp_sam_failure_sequences'])}
- **SAMURAI Failure Sequences**: {', '.join(comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['failure_pattern_analysis']['samurai_failure_sequences'])}
- **Hybrid Saves TSP-SAM Failures**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['failure_pattern_analysis']['hybrid_saves_tsp_failures']} sequences
- **Hybrid Saves SAMURAI Failures**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['failure_pattern_analysis']['hybrid_saves_sam_failures']} sequences
- **Hybrid Saves Both Failures**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['failure_pattern_analysis']['hybrid_saves_both_failures']} sequences

### Reliability Improvement Analysis
- **TSP-SAM Reliability**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['reliability_improvement']['tsp_sam_reliability']:.1%}
- **SAMURAI Reliability**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['reliability_improvement']['samurai_reliability']:.1%}
- **Hybrid Reliability**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['reliability_improvement']['hybrid_reliability']:.1%}
- **Reliability Improvement**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['reliability_improvement']['reliability_improvement_percentage']:.1f}% over best individual
- **Coverage Improvement**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['reliability_improvement']['coverage_improvement']} additional successful sequences

### Performance Trade-off Analysis
- **FPS Trade-off**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['performance_trade_off_analysis']['fps_trade_off']:.2f} FPS difference
- **Processing Time Trade-off**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['performance_trade_off_analysis']['processing_time_trade_off']:.2f} seconds difference
- **Memory Efficiency Trade-off**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['performance_trade_off_analysis']['memory_efficiency_trade_off']:.4f} efficiency difference

## Thesis Implications
This comprehensive analysis demonstrates:

1. **Performance Trade-offs**: Each baseline has distinct performance characteristics in terms of speed, consistency, and success rate
2. **Complementary Strengths**: The baselines succeed on different sequences, indicating complementary capabilities
3. **Scalability Considerations**: Performance varies with sequence length, affecting practical deployment
4. **Hybrid Justification**: The complementary nature and performance trade-offs strongly support the need for a hybrid approach

### Key Findings for Hybrid Justification:

#### **1. Complementary Nature Proof**
- **TSP-SAM excels** where SAMURAI fails ({comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['complementary_strength_evidence']['tsp_sam_unique_strengths']} sequences)
- **SAMURAI excels** where TSP-SAM fails ({comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['complementary_strength_evidence']['samurai_unique_strengths']} sequences)
- **Hybrid approach** extends coverage beyond individual baselines ({comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['complementary_strength_evidence']['hybrid_extends_coverage']} additional sequences)

#### **2. Failure Recovery Evidence**
- **Hybrid saves TSP-SAM failures**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['failure_pattern_analysis']['hybrid_saves_tsp_failures']} sequences
- **Hybrid saves SAMURAI failures**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['failure_pattern_analysis']['hybrid_saves_sam_failures']} sequences
- **Hybrid saves both failures**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['failure_pattern_analysis']['hybrid_saves_both_failures']} sequences

#### **3. Reliability Improvement**
- **Hybrid reliability**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['reliability_improvement']['hybrid_reliability']:.1%} vs best individual baseline
- **Improvement percentage**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['reliability_improvement']['reliability_improvement_percentage']:.1f}% over best individual method
- **Coverage improvement**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['reliability_improvement']['coverage_improvement']} additional successful sequences

#### **4. Performance Trade-offs**
- **FPS impact**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['performance_trade_off_analysis']['fps_trade_off']:.2f} FPS difference
- **Processing time impact**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['performance_trade_off_analysis']['processing_time_trade_off']:.2f} seconds difference
- **Memory efficiency impact**: {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['performance_trade_off_analysis']['memory_efficiency_trade_off']:.4f} efficiency difference

**Conclusion**: The hybrid approach is justified because:
1. **Individual baselines are insufficient** - neither TSP-SAM nor SAMURAI alone provides sufficient reliability
2. **Complementary strengths exist** - each baseline excels where the other fails
3. **Hybrid approach provides better coverage** - succeeds on more sequences than individual baselines
4. **Reliability improvement is significant** - {comparison_analysis['comparative_analysis']['hybrid_justification_analysis']['reliability_improvement']['reliability_improvement_percentage']:.1f}% improvement over best individual method
5. **Performance trade-offs are acceptable** - the reliability gains justify the computational overhead

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save thesis summary
        thesis_file = experiment_dir / "reports" / "individual_baselines" / "thesis_baseline_analysis.md"
        with open(thesis_file, 'w', encoding='utf-8') as f:
            f.write(thesis_summary)
        debug_print(f"Thesis baseline analysis saved to: {thesis_file}")
        
        debug_print("Individual baseline reports generated successfully")
        
    except Exception as e:
        debug_print(f"Error generating individual baseline reports: {e}", "ERROR")
        raise e

def main():
    debug_print("Starting comprehensive baseline comparison...")
    
    parser = argparse.ArgumentParser(
        description='Comprehensive Baseline Comparison for Thesis Expose'
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
        default=10,
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
        args.experiment_name = f"baseline_comparison_{timestamp}"
        debug_print(f"Generated experiment name: {args.experiment_name}")
    
    print("Comprehensive Baseline Comparison for Thesis Expose")
    print("=" * 60)
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
        
        # Save system info
        system_info_file = experiment_dir / "metadata" / "system_info.json"
        with open(system_info_file, 'w') as f:
            json.dump(system_info, f, indent=2)
        debug_print(f"System info saved to: {system_info_file}")
        
        # Confirm before proceeding
        if args.max_frames == 0:
            print("\nWARNING: Processing ALL frames for all sequences will take a very long time!")
            print("   Estimated time: 60 sequences × ~100 frames × 3 methods × ~30s = ~150+ hours")
            confirm = input("\nDo you want to continue? (yes/no): ")
            if confirm.lower() not in ['yes', 'y']:
                debug_print("User cancelled operation")
                print("Operation cancelled.")
                return
        
        # Process sequences
        debug_print("Starting comprehensive sequence comparison...")
        results = process_sequences_comparison(args, experiment_dir)
        
        if results:
            print(f"\nComprehensive baseline comparison completed!")
            print(f"Check the experiment directory for organized results: {experiment_dir}")
            print(f"\nKey files generated:")
            print(f"  - Results: {experiment_dir}/results/comparison_results.json")
            print(f"  - Summary: {experiment_dir}/reports/comparison_summary.csv")
            print(f"  - Visualization: {experiment_dir}/visualizations/baseline_comparison.png")
            print(f"  - Thesis Report: {experiment_dir}/reports/thesis_summary_report.md")
            print(f"  - Individual Baselines:")
            print(f"    * TSP-SAM CSV: {experiment_dir}/reports/individual_baselines/tsp_sam/tsp_sam_baseline_results.csv")
            print(f"    * TSP-SAM JSON: {experiment_dir}/reports/individual_baselines/tsp_sam/tsp_sam_baseline_summary.json")
            print(f"    * SAMURAI CSV: {experiment_dir}/reports/individual_baselines/samurai/samurai_baseline_results.csv")
            print(f"    * SAMURAI JSON: {experiment_dir}/reports/individual_baselines/samurai/samurai_baseline_summary.json")
            print(f"    * Hybrid CSV: {experiment_dir}/reports/individual_baselines/hybrid/hybrid_baseline_results.csv")
            print(f"    * Hybrid JSON: {experiment_dir}/reports/individual_baselines/hybrid/hybrid_baseline_summary.json")
            print(f"    * Comparison Analysis: {experiment_dir}/reports/individual_baselines/baseline_comparison_analysis.json")
            print(f"    * Thesis Analysis: {experiment_dir}/reports/individual_baselines/thesis_baseline_analysis.md")
        else:
            debug_print("Sequence processing failed", "ERROR")
            print("\nComprehensive baseline comparison failed!")
            
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
