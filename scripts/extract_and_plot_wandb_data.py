#!/usr/bin/env python3
"""
Extract and plot data from W&B runs for baseline analysis
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def extract_wandb_data():
    """Extract data from W&B run folders"""
    
    wandb_dir = Path("wandb")
    data = []
    
    for run_dir in wandb_dir.glob("run-*"):
        try:
            # Extract run ID and name from folder name
            run_id = run_dir.name.split('-')[2]  # e.g., "8n42x1lb"
            
            # Read wandb-metadata.json
            metadata_file = run_dir / "files" / "wandb-metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    # Extract run name from args
                    args = metadata.get('args', [])
                    run_name = 'unknown'
                    sequence = 'unknown'
                    model = 'unknown'
                    
                    # Find experiment name and sequence from args
                    for i, arg in enumerate(args):
                        if arg == '--experiment-name' and i+1 < len(args):
                            run_name = args[i+1]
                        elif arg == '--sequence' and i+1 < len(args):
                            sequence = args[i+1]
                    
                    # Determine model from program path
                    program = metadata.get('program', '')
                    if 'tsp_sam' in program:
                        model = 'tsp_sam'
                    elif 'samurai' in program:
                        model = 'samurai'
                    elif 'maskanyone' in program:
                        model = 'maskanyone'
            
            # Read wandb-summary.json
            summary_file = run_dir / "files" / "wandb-summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    
                    # Extract key metrics with correct field names
                    row = {
                        'run_id': run_id,
                        'run_name': run_name,
                        'model': model,
                        'sequence': sequence,
                        'iou': summary.get('experiment/overall_avg_iou', summary.get('eval/iou', 0)),
                        'dice': summary.get('experiment/overall_avg_dice', summary.get('eval/dice', 0)),
                        'temporal_consistency': summary.get('eval/temporal_iou', 0),
                        'failure_rate': summary.get('experiment/overall_failure_rate', 0),
                        'processing_time': summary.get('_runtime', 0),
                        'total_frames': summary.get('experiment/total_frames_processed', 0),
                        'precision': summary.get('eval/precision', 0),
                        'recall': summary.get('eval/recall', 0)
                    }
                    data.append(row)
                    
        except Exception as e:
            print(f"Error processing {run_dir}: {e}")
            continue
    
    return pd.DataFrame(data)

def plot_baseline_comparison(df):
    """Create comprehensive baseline comparison plots"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('TSP-SAM vs SAMURAI Baseline Comparison', fontsize=16, fontweight='bold')
    
    # 1. IoU Comparison by Sequence
    ax1 = axes[0, 0]
    sequences = df['sequence'].unique()
    tsp_iou = [df[(df['sequence'] == seq) & (df['model'] == 'tsp_sam')]['iou'].iloc[0] 
               if len(df[(df['sequence'] == seq) & (df['model'] == 'tsp_sam')]) > 0 else 0 
               for seq in sequences]
    samurai_iou = [df[(df['sequence'] == seq) & (df['model'] == 'samurai')]['iou'].iloc[0] 
                   if len(df[(df['sequence'] == seq) & (df['model'] == 'samurai')]) > 0 else 0 
                   for seq in sequences]
    
    x = np.arange(len(sequences))
    width = 0.35
    
    ax1.bar(x - width/2, tsp_iou, width, label='TSP-SAM', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, samurai_iou, width, label='SAMURAI', alpha=0.8, color='lightcoral')
    ax1.set_xlabel('Sequence')
    ax1.set_ylabel('IoU Score')
    ax1.set_title('IoU Comparison by Sequence')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sequences, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Failure Rate Comparison
    ax2 = axes[0, 1]
    tsp_failure = [df[(df['sequence'] == seq) & (df['model'] == 'tsp_sam')]['failure_rate'].iloc[0] 
                   if len(df[(df['sequence'] == seq) & (df['model'] == 'tsp_sam')]) > 0 else 0 
                   for seq in sequences]
    samurai_failure = [df[(df['sequence'] == seq) & (df['model'] == 'samurai')]['failure_rate'].iloc[0] 
                       if len(df[(df['sequence'] == seq) & (df['model'] == 'samurai')]) > 0 else 0 
                       for seq in sequences]
    
    ax2.bar(x - width/2, tsp_failure, width, label='TSP-SAM', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, samurai_failure, width, label='SAMURAI', alpha=0.8, color='lightcoral')
    ax2.set_xlabel('Sequence')
    ax2.set_ylabel('Failure Rate')
    ax2.set_title('Failure Rate Comparison by Sequence')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sequences, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Temporal Consistency Comparison
    ax3 = axes[0, 2]
    tsp_temp = [df[(df['sequence'] == seq) & (df['model'] == 'tsp_sam')]['temporal_consistency'].iloc[0] 
                if len(df[(df['sequence'] == seq) & (df['model'] == 'tsp_sam')]) > 0 else 0 
                for seq in sequences]
    samurai_temp = [df[(df['sequence'] == seq) & (df['model'] == 'samurai')]['temporal_consistency'].iloc[0] 
                    if len(df[(df['sequence'] == seq) & (df['model'] == 'samurai')]) > 0 else 0 
                    for seq in sequences]
    
    ax3.bar(x - width/2, tsp_temp, width, label='TSP-SAM', alpha=0.8, color='skyblue')
    ax3.bar(x + width/2, samurai_temp, width, label='SAMURAI', alpha=0.8, color='lightcoral')
    ax3.set_xlabel('Sequence')
    ax3.set_ylabel('Temporal Consistency')
    ax3.set_title('Temporal Consistency Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sequences, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Overall Performance Distribution
    ax4 = axes[1, 0]
    tsp_data = df[df['model'] == 'tsp_sam']['iou'].dropna()
    samurai_data = df[df['model'] == 'samurai']['iou'].dropna()
    
    ax4.hist(tsp_data, bins=10, alpha=0.7, label='TSP-SAM', color='skyblue', density=True)
    ax4.hist(samurai_data, bins=10, alpha=0.7, label='SAMURAI', color='lightcoral', density=True)
    ax4.set_xlabel('IoU Score')
    ax4.set_ylabel('Density')
    ax4.set_title('IoU Score Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance vs Processing Time
    ax5 = axes[1, 1]
    tsp_times = df[df['model'] == 'tsp_sam']['processing_time'].dropna()
    tsp_perf = df[df['model'] == 'tsp_sam']['iou'].dropna()
    samurai_times = df[df['model'] == 'samurai']['processing_time'].dropna()
    samurai_perf = df[df['model'] == 'samurai']['iou'].dropna()
    
    ax5.scatter(tsp_times, tsp_perf, alpha=0.7, label='TSP-SAM', color='skyblue', s=60)
    ax5.scatter(samurai_times, samurai_perf, alpha=0.7, label='SAMURAI', color='lightcoral', s=60)
    ax5.set_xlabel('Processing Time (seconds)')
    ax5.set_ylabel('IoU Score')
    ax5.set_title('Performance vs Processing Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary Statistics Table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate summary statistics
    summary_stats = {
        'Metric': ['Mean IoU', 'Std IoU', 'Mean Failure Rate', 'Mean Temporal Consistency'],
        'TSP-SAM': [
            f"{df[df['model'] == 'tsp_sam']['iou'].mean():.3f}",
            f"{df[df['model'] == 'tsp_sam']['iou'].std():.3f}",
            f"{df[df['model'] == 'tsp_sam']['failure_rate'].mean():.3f}",
            f"{df[df['model'] == 'tsp_sam']['temporal_consistency'].mean():.3f}"
        ],
        'SAMURAI': [
            f"{df[df['model'] == 'samurai']['iou'].mean():.3f}",
            f"{df[df['model'] == 'samurai']['iou'].std():.3f}",
            f"{df[df['model'] == 'samurai']['failure_rate'].mean():.3f}",
            f"{df[df['model'] == 'samurai']['temporal_consistency'].mean():.3f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    table = ax6.table(cellText=summary_df.values, colLabels=summary_df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax6.set_title('Summary Statistics', pad=20)
    
    plt.tight_layout()
    plt.savefig('baseline_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid GUI issues
    
    return df

def create_failure_analysis_plot(df):
    """Create detailed failure analysis plot"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Failure Analysis: TSP-SAM vs SAMURAI', fontsize=16, fontweight='bold')
    
    # 1. Failure Rate by Sequence
    ax1 = axes[0]
    sequences = df['sequence'].unique()
    tsp_failure = []
    samurai_failure = []
    
    for seq in sequences:
        tsp_data = df[(df['sequence'] == seq) & (df['model'] == 'tsp_sam')]
        samurai_data = df[(df['sequence'] == seq) & (df['model'] == 'samurai')]
        
        tsp_failure.append(tsp_data['failure_rate'].iloc[0] if len(tsp_data) > 0 else 0)
        samurai_failure.append(samurai_data['failure_rate'].iloc[0] if len(samurai_data) > 0 else 0)
    
    x = np.arange(len(sequences))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, tsp_failure, width, label='TSP-SAM', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, samurai_failure, width, label='SAMURAI', alpha=0.8, color='lightcoral')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Sequence')
    ax1.set_ylabel('Failure Rate')
    ax1.set_title('Failure Rate by Sequence')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sequences, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Complementary Failure Analysis
    ax2 = axes[1]
    
    # Find sequences where one model fails and the other succeeds
    complementary_cases = []
    for seq in sequences:
        tsp_data = df[(df['sequence'] == seq) & (df['model'] == 'tsp_sam')]
        samurai_data = df[(df['sequence'] == seq) & (df['model'] == 'samurai')]
        
        if len(tsp_data) > 0 and len(samurai_data) > 0:
            tsp_fail = tsp_data['failure_rate'].iloc[0]
            samurai_fail = samurai_data['failure_rate'].iloc[0]
            
            if (tsp_fail > 0.5 and samurai_fail < 0.5) or (tsp_fail < 0.5 and samurai_fail > 0.5):
                complementary_cases.append({
                    'sequence': seq,
                    'tsp_failure': tsp_fail,
                    'samurai_failure': samurai_fail,
                    'hybrid_potential': min(tsp_fail, samurai_fail)
                })
    
    if complementary_cases:
        comp_df = pd.DataFrame(complementary_cases)
        x_comp = np.arange(len(comp_df))
        
        ax2.bar(x_comp - width/2, comp_df['tsp_failure'], width, label='TSP-SAM', alpha=0.8, color='skyblue')
        ax2.bar(x_comp + width/2, comp_df['samurai_failure'], width, label='SAMURAI', alpha=0.8, color='lightcoral')
        ax2.bar(x_comp, comp_df['hybrid_potential'], width/2, label='Hybrid Potential', alpha=0.8, color='green')
        
        ax2.set_xlabel('Sequence')
        ax2.set_ylabel('Failure Rate')
        ax2.set_title('Complementary Failure Cases\n(Hybrid System Potential)')
        ax2.set_xticks(x_comp)
        ax2.set_xticklabels(comp_df['sequence'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No complementary failure cases found', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Complementary Failure Analysis')
    
    plt.tight_layout()
    plt.savefig('failure_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid GUI issues

def main():
    """Main function to extract and plot W&B data"""
    
    print("Extracting data from W&B runs...")
    df = extract_wandb_data()
    
    if df.empty:
        print("No data found in W&B runs!")
        return
    
    print(f"Extracted data for {len(df)} runs")
    print(f"Models: {df['model'].unique()}")
    print(f"Sequences: {df['sequence'].unique()}")
    
    # Save raw data
    df.to_csv('wandb_baseline_data.csv', index=False)
    print("Raw data saved to 'wandb_baseline_data.csv'")
    
    # Create plots
    print("Creating baseline comparison plots...")
    plot_baseline_comparison(df)
    
    print("Creating failure analysis plots...")
    create_failure_analysis_plot(df)
    
    # Print summary
    print("\n" + "="*50)
    print("BASELINE ANALYSIS SUMMARY")
    print("="*50)
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        print(f"\n{model.upper()}:")
        print(f"  Mean IoU: {model_data['iou'].mean():.3f} ± {model_data['iou'].std():.3f}")
        print(f"  Mean Failure Rate: {model_data['failure_rate'].mean():.3f}")
        print(f"  Mean Temporal Consistency: {model_data['temporal_consistency'].mean():.3f}")
        print(f"  Processing Time: {model_data['processing_time'].mean():.1f}s")
    
    print("\nPlots saved as:")
    print("  - baseline_comparison_analysis.png")
    print("  - failure_analysis.png")
    print("  - wandb_baseline_data.csv")

if __name__ == "__main__":
    main()
