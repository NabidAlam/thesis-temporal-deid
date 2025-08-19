#!/usr/bin/env python3
"""
Comprehensive Results Analysis Script
Shows IoU, Temporal Consistency, Processing Time, and Memory Usage metrics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

def analyze_pipeline_results(output_dir: str):
    """Analyze the results from the pipeline with comprehensive metrics."""
    
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Output directory {output_dir} not found!")
        return
    
    print("=" * 80)
    print("COMPREHENSIVE PIPELINE RESULTS ANALYSIS")
    print("=" * 80)
    
    # Load results if available
    results_file = output_path / "pipeline_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"✓ Loaded results from {results_file}")
    else:
        print("⚠ No results file found - analyzing from frame history")
        results = []
    
    # Analyze frame history if available
    frame_history_file = output_path / "frame_history.json"
    if frame_history_file.exists():
        with open(frame_history_file, 'r') as f:
            frame_history = json.load(f)
        print(f"✓ Loaded frame history from {frame_history_file}")
        
        # Extract metrics from frame history
        metrics_data = []
        for frame_data in frame_history:
            if 'metrics' in frame_data:
                metrics = frame_data['metrics']
                metrics_data.append({
                    'frame_idx': frame_data.get('frame_idx', 0),
                    'processing_time_ms': metrics.get('processing_time_ms', 0),
                    'cpu_memory_mb': metrics.get('cpu_memory_mb', 0),
                    'gpu_memory_gb': metrics.get('gpu_memory_gb', 0),
                    'iou_score': frame_data.get('iou_metrics', {}).get('average_iou', 0),
                    'temporal_consistency': frame_data.get('temporal_consistency', 0),
                    'method': frame_data.get('method', 'unknown')
                })
        
        if metrics_data:
            print(f"✓ Extracted metrics for {len(metrics_data)} frames")
            analyze_metrics(metrics_data, output_path)
        else:
            print("⚠ No metrics found in frame history")
    
    # Check for mask files
    masks_dir = output_path / "masks"
    if masks_dir.exists():
        mask_files = list(masks_dir.glob("*.png"))
        print(f"✓ Found {len(mask_files)} mask files")
        analyze_mask_files(mask_files, output_path)
    else:
        print("No masks directory found")

def analyze_metrics(metrics_data: list, output_path: Path):
    """Analyze the comprehensive metrics."""
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE METRICS ANALYSIS")
    print("=" * 60)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(metrics_data)
    
    # 1. IoU Analysis
    print("\nIoU (Intersection over Union) Analysis:")
    print(f"   Average IoU: {df['iou_score'].mean():.3f}")
    print(f"   IoU Std Dev: {df['iou_score'].std():.3f}")
    print(f"   Min IoU: {df['iou_score'].min():.3f}")
    print(f"   Max IoU: {df['iou_score'].max():.3f}")
    
    # IoU breakdown
    iou_excellent = len(df[df['iou_score'] >= 0.8])
    iou_good = len(df[(df['iou_score'] >= 0.6) & (df['iou_score'] < 0.8)])
    iou_fair = len(df[(df['iou_score'] >= 0.4) & (df['iou_score'] < 0.6)])
    iou_poor = len(df[df['iou_score'] < 0.4])
    
    print(f"   Excellent (≥0.8): {iou_excellent} frames")
    print(f"   Good (0.6-0.8): {iou_good} frames")
    print(f"   Fair (0.4-0.6): {iou_fair} frames")
    print(f"   Poor (<0.4): {iou_poor} frames")
    
    # 2. Temporal Consistency Analysis
    print("\nTemporal Consistency Analysis:")
    print(f"   Average Consistency: {df['temporal_consistency'].mean():.3f}")
    print(f"   Consistency Std Dev: {df['temporal_consistency'].std():.3f}")
    print(f"   Min Consistency: {df['temporal_consistency'].min():.3f}")
    print(f"   Max Consistency: {df['temporal_consistency'].max():.3f}")
    
    # Consistency breakdown
    tc_excellent = len(df[df['temporal_consistency'] >= 0.8])
    tc_good = len(df[(df['temporal_consistency'] >= 0.6) & (df['temporal_consistency'] < 0.8)])
    tc_fair = len(df[(df['temporal_consistency'] >= 0.4) & (df['temporal_consistency'] < 0.6)])
    tc_poor = len(df[df['temporal_consistency'] < 0.4])
    
    print(f"   Excellent (≥0.8): {tc_excellent} frames")
    print(f"   Good (0.6-0.8): {tc_good} frames")
    print(f"   Fair (0.4-0.6): {tc_fair} frames")
    print(f"   Poor (<0.4): {tc_poor} frames")
    
    # 3. Processing Performance Analysis
    print("\nProcessing Performance Analysis:")
    print(f"   Average Processing Time: {df['processing_time_ms'].mean():.1f} ms")
    print(f"   Total Processing Time: {df['processing_time_ms'].sum():.1f} ms")
    print(f"   Average FPS: {1000/df['processing_time_ms'].mean():.2f}")
    print(f"   Min Processing Time: {df['processing_time_ms'].min():.1f} ms")
    print(f"   Max Processing Time: {df['processing_time_ms'].max():.1f} ms")
    
    # 4. Memory Usage Analysis
    print("\nMemory Usage Analysis:")
    print(f"   Average CPU Memory: {df['cpu_memory_mb'].mean():.1f} MB")
    print(f"   Peak CPU Memory: {df['cpu_memory_mb'].max():.1f} MB")
    
    if df['gpu_memory_gb'].notna().any():
        gpu_memory = df['gpu_memory_gb'].dropna()
        if len(gpu_memory) > 0:
            print(f"   Average GPU Memory: {gpu_memory.mean():.3f} GB")
            print(f"   Peak GPU Memory: {gpu_memory.max():.3f} GB")
    
    # 5. Method Distribution
    print("\nMethod Distribution:")
    method_counts = df['method'].value_counts()
    for method, count in method_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {method}: {count} frames ({percentage:.1f}%)")
    
    # Create visualizations
    create_metrics_visualizations(df, output_path)
    
    # Save detailed analysis
    save_detailed_analysis(df, output_path)

def analyze_mask_files(mask_files: list, output_path: Path):
    """Analyze the generated mask files."""
    
    print("\n" + "=" * 60)
    print("MASK FILES ANALYSIS")
    print("=" * 60)
    
    # Analyze file sizes
    file_sizes = []
    for mask_file in mask_files:
        size_kb = mask_file.stat().st_size / 1024
        file_sizes.append(size_kb)
    
    print(f"   Total mask files: {len(mask_files)}")
    print(f"   Average file size: {np.mean(file_sizes):.1f} KB")
    print(f"   Min file size: {np.min(file_sizes):.1f} KB")
    print(f"   Max file size: {np.max(file_sizes):.1f} KB")
    print(f"   Total size: {np.sum(file_sizes):.1f} KB")
    
    # Check for size consistency (indicates quality)
    size_std = np.std(file_sizes)
    if size_std < 5:
        print("   Excellent size consistency (likely high quality)")
    elif size_std < 15:
        print("   Good size consistency (good quality)")
    elif size_std < 30:
        print("   Moderate size consistency (mixed quality)")
    else:
        print("   Poor size consistency (likely artifacts present)")
    
    # Check for circle artifacts (large files)
    large_files = [f for f in mask_files if f.stat().st_size > 50000]  # >50KB
    if large_files:
        print(f"   Found {len(large_files)} potentially problematic files (>50KB)")
        for f in large_files[:3]:  # Show first 3
            print(f"     - {f.name}: {f.stat().st_size/1024:.1f} KB")
    else:
        print("   No large files detected (circle artifacts eliminated)")

def create_metrics_visualizations(df: pd.DataFrame, output_path: Path):
    """Create comprehensive visualizations of the metrics."""
    
    print("\nCreating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Pipeline Metrics Analysis', fontsize=16, fontweight='bold')
    
    # 1. IoU Distribution
    axes[0, 0].hist(df['iou_score'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('IoU Score Distribution')
    axes[0, 0].set_xlabel('IoU Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['iou_score'].mean(), color='red', linestyle='--', label=f'Mean: {df["iou_score"].mean():.3f}')
    axes[0, 0].legend()
    
    # 2. Temporal Consistency Distribution
    axes[0, 1].hist(df['temporal_consistency'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Temporal Consistency Distribution')
    axes[0, 1].set_xlabel('Consistency Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(df['temporal_consistency'].mean(), color='red', linestyle='--', label=f'Mean: {df["temporal_consistency"].mean():.3f}')
    axes[0, 1].legend()
    
    # 3. Processing Time Over Frames
    axes[0, 2].plot(df['frame_idx'], df['processing_time_ms'], marker='o', alpha=0.7)
    axes[0, 2].set_title('Processing Time per Frame')
    axes[0, 2].set_xlabel('Frame Index')
    axes[0, 2].set_ylabel('Processing Time (ms)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. CPU Memory Usage Over Frames
    axes[1, 0].plot(df['frame_idx'], df['cpu_memory_mb'], marker='o', alpha=0.7, color='green')
    axes[1, 0].set_title('CPU Memory Usage per Frame')
    axes[1, 0].set_xlabel('Frame Index')
    axes[1, 0].set_ylabel('CPU Memory (MB)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. IoU vs Temporal Consistency Scatter
    axes[1, 1].scatter(df['temporal_consistency'], df['iou_score'], alpha=0.7)
    axes[1, 1].set_title('IoU vs Temporal Consistency')
    axes[1, 1].set_xlabel('Temporal Consistency')
    axes[1, 1].set_ylabel('IoU Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Processing Time vs Memory Usage
    axes[1, 2].scatter(df['cpu_memory_mb'], df['processing_time_ms'], alpha=0.7, color='orange')
    axes[1, 2].set_title('Processing Time vs Memory Usage')
    axes[1, 2].set_xlabel('CPU Memory (MB)')
    axes[1, 2].set_ylabel('Processing Time (ms)')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    viz_path = output_path / "metrics_visualizations.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved visualizations to {viz_path}")
    plt.close()

def save_detailed_analysis(df: pd.DataFrame, output_path: Path):
    """Save detailed analysis results."""
    
    # Create comprehensive analysis
    analysis = {
        'summary': {
            'total_frames': len(df),
            'average_iou': float(df['iou_score'].mean()),
            'average_temporal_consistency': float(df['temporal_consistency'].mean()),
            'average_processing_time_ms': float(df['processing_time_ms'].mean()),
            'average_cpu_memory_mb': float(df['cpu_memory_mb'].mean()),
            'total_processing_time_seconds': float(df['processing_time_ms'].sum() / 1000),
            'average_fps': float(1000 / df['processing_time_ms'].mean())
        },
        'quality_metrics': {
            'iou_breakdown': {
                'excellent': int(len(df[df['iou_score'] >= 0.8])),
                'good': int(len(df[(df['iou_score'] >= 0.6) & (df['iou_score'] < 0.8)])),
                'fair': int(len(df[(df['iou_score'] >= 0.4) & (df['iou_score'] < 0.6)])),
                'poor': int(len(df[df['iou_score'] < 0.4]))
            },
            'temporal_consistency_breakdown': {
                'excellent': int(len(df[df['temporal_consistency'] >= 0.8])),
                'good': int(len(df[(df['temporal_consistency'] >= 0.6) & (df['temporal_consistency'] < 0.8)])),
                'fair': int(len(df[(df['temporal_consistency'] >= 0.4) & (df['temporal_consistency'] < 0.6)])),
                'poor': int(len(df[df['temporal_consistency'] < 0.4]))
            }
        },
        'performance_metrics': {
            'processing_time_stats': {
                'min_ms': float(df['processing_time_ms'].min()),
                'max_ms': float(df['processing_time_ms'].max()),
                'std_ms': float(df['processing_time_ms'].std())
            },
            'memory_stats': {
                'min_cpu_mb': float(df['cpu_memory_mb'].min()),
                'max_cpu_mb': float(df['cpu_memory_mb'].max()),
                'std_cpu_mb': float(df['cpu_memory_mb'].std())
            }
        }
    }
    
    # Save analysis
    analysis_path = output_path / "detailed_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"   Saved detailed analysis to {analysis_path}")
    
    # Save metrics CSV
    csv_path = output_path / "metrics_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"   Saved metrics data to {csv_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        # Default to most recent output
        output_dirs = [d for d in Path(".").glob("output/*") if d.is_dir()]
        if output_dirs:
            output_dir = str(max(output_dirs, key=lambda x: x.stat().st_mtime))
            print(f"Using most recent output directory: {output_dir}")
        else:
            print("No output directories found. Please specify one:")
            print("python analyze_results.py <output_directory>")
            sys.exit(1)
    
    analyze_pipeline_results(output_dir)
