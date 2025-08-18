"""
Baseline Results Visualization Script
Creates comprehensive visualizations for TSP-SAM vs SAMURAI baselines
Organized structure for future dataset additions
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import cv2
from tqdm import tqdm

# Set style for professional plots
plt.style.use('default')  # Use default style for compatibility
sns.set_palette("husl")

class BaselineVisualizer:
    def __init__(self, results_dir="evaluation/results", output_dir="evaluation/visualizations"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        
        # Create organized output structure
        self.create_output_structure()
        
    def create_output_structure(self):
        """Create organized visualization directory structure"""
        # Main visualization directories
        (self.output_dir / "baseline_comparison").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "performance_metrics").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "temporal_analysis").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "sequence_breakdown").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "method_comparison").mkdir(parents=True, exist_ok=True)
        
        # Dataset-specific subdirectories
        for dataset in ["davis", "ted"]:
            (self.output_dir / "baseline_comparison" / dataset).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "performance_metrics" / dataset).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "temporal_analysis" / dataset).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "sequence_breakdown" / dataset).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "method_comparison" / dataset).mkdir(parents=True, exist_ok=True)
    
    def load_evaluation_results(self, method, dataset="davis"):
        """Load evaluation results for a method and dataset"""
        # Try to find the most recent results
        possible_paths = [
            self.results_dir / f"{method}_corrected" / method / dataset,
            self.results_dir / "baseline_results" / method / dataset,
            self.results_dir / f"{method}_baseline" / dataset
        ]
        
        for path in possible_paths:
            if path.exists():
                result_files = list(path.glob("baseline_results_*.json"))
                if result_files:
                    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_file, 'r') as f:
                        return json.load(f)
        
        print(f"No results found for {method} on {dataset}")
        return None
    
    def create_performance_comparison_plot(self, dataset="davis"):
        """Create performance comparison plot between methods"""
        print(f"Creating performance comparison plot for {dataset}...")
        
        # Load results for both methods
        samurai_results = self.load_evaluation_results("samurai", dataset)
        tsp_sam_results = self.load_evaluation_results("tsp-sam", dataset)
        
        if not samurai_results or not tsp_sam_results:
            print("Cannot create comparison plot - missing results")
            return
        
        # Extract metrics
        methods = ["SAMURAI", "TSP-SAM"]
        metrics = ["mean_iou", "mean_dice", "temporal_consistency"]
        metric_labels = ["IoU", "Dice", "Temporal Consistency"]
        
        # Prepare data
        data = []
        for i, method in enumerate([samurai_results, tsp_sam_results]):
            for j, metric in enumerate(metrics):
                if metric in method[0]:  # Check if metric exists
                    data.append({
                        'Method': methods[i],
                        'Metric': metric_labels[j],
                        'Value': method[0][metric]
                    })
        
        if not data:
            print("No valid metrics found for comparison")
            return
        
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar plot
        ax = sns.barplot(data=df, x='Metric', y='Value', hue='Method', 
                        palette=['#FF6B6B', '#4ECDC4'])
        
        # Customize plot
        plt.title(f'Performance Comparison: SAMURAI vs TSP-SAM on {dataset.upper()}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=14, fontweight='bold')
        plt.legend(title='Method', title_fontsize=12, fontsize=11)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=10)
        
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "method_comparison" / dataset / f"performance_comparison_{dataset}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance comparison plot saved to: {output_path}")
    
    def create_temporal_consistency_plot(self, dataset="davis"):
        """Create temporal consistency analysis plot"""
        print(f"Creating temporal consistency plot for {dataset}...")
        
        # Load results
        samurai_results = self.load_evaluation_results("samurai", dataset)
        tsp_sam_results = self.load_evaluation_results("tsp-sam", dataset)
        
        if not samurai_results or not tsp_sam_results:
            print("Cannot create temporal plot - missing results")
            return
        
        # Extract temporal consistency data
        samurai_temporal = [seq['temporal_consistency'] for seq in samurai_results]
        tsp_sam_temporal = [seq['temporal_consistency'] for seq in tsp_sam_results]
        sequences = [seq['sequence'] for seq in samurai_results]
        
        # Create plot
        plt.figure(figsize=(14, 8))
        
        x = np.arange(len(sequences))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, samurai_temporal, width, label='SAMURAI', 
                        color='#FF6B6B', alpha=0.8)
        bars2 = plt.bar(x + width/2, tsp_sam_temporal, width, label='TSP-SAM', 
                        color='#4ECDC4', alpha=0.8)
        
        # Customize plot
        plt.xlabel('Sequences', fontsize=14, fontweight='bold')
        plt.ylabel('Temporal Consistency', fontsize=14, fontweight='bold')
        plt.title(f'Temporal Consistency Comparison: {dataset.upper()}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(x, sequences, rotation=45, ha='right')
        plt.legend(fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.ylim(0, 1.0)
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "temporal_analysis" / dataset / f"temporal_consistency_{dataset}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Temporal consistency plot saved to: {output_path}")
    
    def create_sequence_performance_heatmap(self, dataset="davis"):
        """Create heatmap showing performance across sequences"""
        print(f"Creating sequence performance heatmap for {dataset}...")
        
        # Load results
        samurai_results = self.load_evaluation_results("samurai", dataset)
        tsp_sam_results = self.load_evaluation_results("tsp-sam", dataset)
        
        if not samurai_results or not tsp_sam_results:
            print("Cannot create heatmap - missing results")
            return
        
        # Prepare data for heatmap
        sequences = [seq['sequence'] for seq in samurai_results]
        metrics = ['mean_iou', 'mean_dice', 'temporal_consistency']
        metric_labels = ['IoU', 'Dice', 'Temporal Consistency']
        
        # Create data matrix
        data_matrix = np.zeros((len(metrics), len(sequences)))
        
        for i, metric in enumerate(metrics):
            for j, seq in enumerate(sequences):
                # Find corresponding sequence in results
                samurai_seq = next((s for s in samurai_results if s['sequence'] == seq), None)
                tsp_sam_seq = next((s for s in tsp_sam_results if s['sequence'] == seq), None)
                
                if samurai_seq and tsp_sam_seq and metric in samurai_seq:
                    # Average of both methods
                    samurai_val = samurai_seq[metric]
                    tsp_sam_val = tsp_sam_seq[metric]
                    data_matrix[i, j] = (samurai_val + tsp_sam_val) / 2
        
        # Create heatmap
        plt.figure(figsize=(16, 8))
        
        # Create heatmap
        sns.heatmap(data_matrix, 
                    xticklabels=sequences, 
                    yticklabels=metric_labels,
                    annot=True, 
                    fmt='.3f',
                    cmap='RdYlBu_r',
                    cbar_kws={'label': 'Score'},
                    linewidths=0.5)
        
        plt.title(f'Sequence Performance Heatmap: {dataset.upper()}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Sequences', fontsize=14, fontweight='bold')
        plt.ylabel('Metrics', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "sequence_breakdown" / dataset / f"performance_heatmap_{dataset}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance heatmap saved to: {output_path}")
    
    def create_metric_distribution_plots(self, dataset="davis"):
        """Create distribution plots for key metrics"""
        print(f"Creating metric distribution plots for {dataset}...")
        
        # Load results
        samurai_results = self.load_evaluation_results("samurai", dataset)
        tsp_sam_results = self.load_evaluation_results("tsp-sam", dataset)
        
        if not samurai_results or not tsp_sam_results:
            print("Cannot create distribution plots - missing results")
            return
        
        # Extract metrics
        metrics = ['mean_iou', 'mean_dice', 'temporal_consistency']
        metric_labels = ['IoU', 'Dice', 'Temporal Consistency']
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Metric Distributions: {dataset.upper()}', fontsize=16, fontweight='bold')
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            if metric in samurai_results[0]:
                samurai_vals = [seq[metric] for seq in samurai_results]
                tsp_sam_vals = [seq[metric] for seq in tsp_sam_results]
                
                # Create violin plot
                data = []
                labels = []
                for val in samurai_vals:
                    data.append(val)
                    labels.append('SAMURAI')
                for val in tsp_sam_vals:
                    data.append(val)
                    labels.append('TSP-SAM')
                
                df = pd.DataFrame({'Value': data, 'Method': labels})
                
                sns.violinplot(data=df, x='Method', y='Value', ax=axes[i], 
                              palette=['#FF6B6B', '#4ECDC4'])
                axes[i].set_title(f'{label} Distribution', fontweight='bold')
                axes[i].set_ylabel(label)
                axes[i].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "performance_metrics" / dataset / f"metric_distributions_{dataset}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Metric distribution plots saved to: {output_path}")
    
    def create_summary_dashboard(self, dataset="davis"):
        """Create a comprehensive summary dashboard"""
        print(f"Creating summary dashboard for {dataset}...")
        
        # Load results
        samurai_results = self.load_evaluation_results("samurai", dataset)
        tsp_sam_results = self.load_evaluation_results("tsp-sam", dataset)
        
        if not samurai_results or not tsp_sam_results:
            print("Cannot create dashboard - missing results")
            return
        
        # Create dashboard
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'Baseline Evaluation Dashboard: {dataset.upper()}', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Performance comparison (top left)
        ax1 = plt.subplot(2, 3, 1)
        methods = ["SAMURAI", "TSP-SAM"]
        metrics = ["mean_iou", "mean_dice", "temporal_consistency"]
        metric_labels = ["IoU", "Dice", "Temporal Consistency"]
        
        data = []
        for i, method in enumerate([samurai_results, tsp_sam_results]):
            for j, metric in enumerate(metrics):
                if metric in method[0]:
                    data.append({
                        'Method': methods[i],
                        'Metric': metric_labels[j],
                        'Value': method[0][metric]
                    })
        
        if data:
            df = pd.DataFrame(data)
            sns.barplot(data=df, x='Metric', y='Value', hue='Method', 
                        palette=['#FF6B6B', '#4ECDC4'], ax=ax1)
            ax1.set_title('Performance Comparison', fontweight='bold')
            ax1.set_ylim(0, 1.1)
            ax1.grid(axis='y', alpha=0.3)
        
        # 2. Temporal consistency by sequence (top middle)
        ax2 = plt.subplot(2, 3, 2)
        sequences = [seq['sequence'] for seq in samurai_results]
        samurai_temporal = [seq['temporal_consistency'] for seq in samurai_results]
        tsp_sam_temporal = [seq['temporal_consistency'] for seq in tsp_sam_results]
        
        x = np.arange(len(sequences))
        width = 0.35
        ax2.bar(x - width/2, samurai_temporal, width, label='SAMURAI', 
                color='#FF6B6B', alpha=0.8)
        ax2.bar(x + width/2, tsp_sam_temporal, width, label='TSP-SAM', 
                color='#4ECDC4', alpha=0.8)
        ax2.set_title('Temporal Consistency by Sequence', fontweight='bold')
        ax2.set_xlabel('Sequences')
        ax2.set_ylabel('Temporal Consistency')
        ax2.set_xticks(x)
        ax2.set_xticklabels(sequences, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Metric distributions (top right)
        ax3 = plt.subplot(2, 3, 3)
        if 'mean_iou' in samurai_results[0]:
            samurai_ious = [seq['mean_iou'] for seq in samurai_results]
            tsp_sam_ious = [seq['mean_iou'] for seq in tsp_sam_results]
            
            data = []
            labels = []
            for val in samurai_ious:
                data.append(val)
                labels.append('SAMURAI')
            for val in tsp_sam_ious:
                data.append(val)
                labels.append('TSP-SAM')
            
            df = pd.DataFrame({'IoU': data, 'Method': labels})
            sns.boxplot(data=df, x='Method', y='IoU', ax=ax3, 
                        palette=['#FF6B6B', '#4ECDC4'])
            ax3.set_title('IoU Distribution', fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
        
        # 4. Summary statistics (bottom left)
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        
        # Calculate summary statistics
        if samurai_results and 'mean_iou' in samurai_results[0]:
            samurai_avg_iou = np.mean([seq['mean_iou'] for seq in samurai_results])
            samurai_avg_dice = np.mean([seq['mean_dice'] for seq in samurai_results])
            samurai_avg_temp = np.mean([seq['temporal_consistency'] for seq in samurai_results])
            
            tsp_sam_avg_iou = np.mean([seq['mean_iou'] for seq in tsp_sam_results])
            tsp_sam_avg_dice = np.mean([seq['mean_dice'] for seq in tsp_sam_results])
            tsp_sam_avg_temp = np.mean([seq['temporal_consistency'] for seq in tsp_sam_results])
            
            summary_text = f"""SUMMARY STATISTICS

SAMURAI:
• Avg IoU: {samurai_avg_iou:.3f}
• Avg Dice: {samurai_avg_dice:.3f}
• Avg Temporal: {samurai_avg_temp:.3f}

TSP-SAM:
• Avg IoU: {tsp_sam_avg_iou:.3f}
• Avg Dice: {tsp_sam_avg_dice:.3f}
• Avg Temporal: {tsp_sam_avg_temp:.3f}

Sequences: {len(samurai_results)}
Dataset: {dataset.upper()}"""
            
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                     fontsize=12, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # 5. Performance heatmap (bottom middle)
        ax5 = plt.subplot(2, 3, 5)
        if 'mean_iou' in samurai_results[0]:
            # Create simple heatmap
            sequences_short = [seq[:8] + '...' if len(seq) > 8 else seq for seq in sequences]
            data_matrix = np.zeros((3, len(sequences)))
            
            for i, metric in enumerate(['mean_iou', 'mean_dice', 'temporal_consistency']):
                for j, seq in enumerate(sequences):
                    samurai_seq = next((s for s in samurai_results if s['sequence'] == seq), None)
                    if samurai_seq and metric in samurai_seq:
                        data_matrix[i, j] = samurai_seq[metric]
            
            im = ax5.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
            ax5.set_xticks(range(len(sequences_short)))
            ax5.set_xticklabels(sequences_short, rotation=45, ha='right')
            ax5.set_yticks(range(3))
            ax5.set_yticklabels(['IoU', 'Dice', 'Temporal'])
            ax5.set_title('Performance Heatmap (SAMURAI)', fontweight='bold')
            plt.colorbar(im, ax=ax5, label='Score')
        
        # 6. Method comparison (bottom right)
        ax6 = plt.subplot(2, 3, 6)
        if 'mean_iou' in samurai_results[0]:
            # Create radar chart-like comparison
            metrics_radar = ['IoU', 'Dice', 'Temporal']
            samurai_vals = [samurai_avg_iou, samurai_avg_dice, samurai_avg_temp]
            tsp_sam_vals = [tsp_sam_avg_iou, tsp_sam_avg_dice, tsp_sam_avg_temp]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
            samurai_vals += samurai_vals[:1]  # Close the plot
            tsp_sam_vals += tsp_sam_vals[:1]
            angles += angles[:1]
            
            ax6.plot(angles, samurai_vals, 'o-', linewidth=2, label='SAMURAI', color='#FF6B6B')
            ax6.fill(angles, samurai_vals, alpha=0.25, color='#FF6B6B')
            ax6.plot(angles, tsp_sam_vals, 'o-', linewidth=2, label='TSP-SAM', color='#4ECDC4')
            ax6.fill(angles, tsp_sam_vals, alpha=0.25, color='#4ECDC4')
            
            ax6.set_xticks(angles[:-1])
            ax6.set_xticklabels(metrics_radar)
            ax6.set_ylim(0, 1)
            ax6.set_title('Method Comparison Radar', fontweight='bold')
            ax6.legend()
            ax6.grid(True)
        
        plt.tight_layout()
        
        # Save dashboard
        output_path = self.output_dir / "baseline_comparison" / dataset / f"summary_dashboard_{dataset}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary dashboard saved to: {output_path}")
    
    def generate_all_visualizations(self, dataset="davis"):
        """Generate all visualizations for a dataset"""
        print(f"Generating all visualizations for {dataset}...")
        
        # Create all visualization types
        self.create_performance_comparison_plot(dataset)
        self.create_temporal_consistency_plot(dataset)
        self.create_sequence_performance_heatmap(dataset)
        self.create_metric_distribution_plots(dataset)
        self.create_summary_dashboard(dataset)
        
        print(f"All visualizations for {dataset} completed!")
        print(f"Check the {self.output_dir} directory for results.")

def main():
    parser = argparse.ArgumentParser(description="Generate baseline visualization plots")
    parser.add_argument("--dataset", default="davis", choices=["davis", "ted"],
                       help="Dataset to visualize")
    parser.add_argument("--results_dir", default="evaluation/results",
                       help="Directory containing evaluation results")
    parser.add_argument("--output_dir", default="evaluation/visualizations",
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = BaselineVisualizer(args.results_dir, args.output_dir)
    
    # Generate all visualizations
    visualizer.generate_all_visualizations(args.dataset)
    
    print(f"\n🎉 Visualization generation completed!")
    print(f"📁 Check {args.output_dir} for organized visualization results")
    print(f"📊 Ready to show your professor the visual evidence of your work!")

if __name__ == "__main__":
    main()
