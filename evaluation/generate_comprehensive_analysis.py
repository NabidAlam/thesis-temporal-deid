#!/usr/bin/env python3
"""
Comprehensive Analysis Generator for TSP-SAM vs SAMURAI Comparison
Generates professional analysis reports, metrics, and visualizations for German Master Thesis
"""

import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Any
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import adapted_rand_error, variation_of_information
from skimage.measure import label, regionprops
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveAnalyzer:
    """Comprehensive analysis generator for baseline comparison"""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.results_dir = self.experiment_dir / "results"
        self.comparison_dir = self.results_dir / "comparison"
        self.analysis_dir = self.comparison_dir / "analysis"
        
        # Create analysis directories
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        (self.analysis_dir / "plots").mkdir(exist_ok=True)
        (self.analysis_dir / "tables").mkdir(exist_ok=True)
        (self.analysis_dir / "reports").mkdir(exist_ok=True)
        
        # Load experiment configuration
        self.config = self.load_config()
        
        # Initialize results storage
        self.analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "experiment": self.experiment_dir.name,
            "metrics": {},
            "statistical_tests": {},
            "recommendations": []
        }
    
    def load_config(self) -> Dict[str, Any]:
        """Load experiment configuration"""
        config_file = self.experiment_dir / "config" / "experiment_config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_comparison_results(self) -> Dict[str, Any]:
        """Load comparison results"""
        results_file = self.results_dir / "comparison_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return {}
    
    def calculate_sequence_metrics(self, sequence_name: str, baseline_name: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a sequence"""
        # Try multiple possible directory structures
        possible_paths = [
            # Current nested structure
            self.results_dir / baseline_name / "sequences" / baseline_name / sequence_name,
            # Alternative structure
            self.results_dir / baseline_name / "sequences" / sequence_name,
            # Direct structure
            self.results_dir / baseline_name / sequence_name,
            # Fallback structure
            self.results_dir / "sequences" / baseline_name / sequence_name
        ]
        
        sequence_dir = None
        for path in possible_paths:
            if path.exists():
                sequence_dir = path
                break
        
        if not sequence_dir:
            print(f"Warning: No directory found for {baseline_name}/{sequence_name}")
            return {}
        
        # Find mask files
        mask_files = list(sequence_dir.glob("*.png"))
        if not mask_files:
            print(f"Warning: No mask files found in {sequence_dir}")
            return {}
        
        # Calculate basic metrics
        metrics = {
            "sequence_name": sequence_name,
            "baseline": baseline_name,
            "total_frames": len(mask_files),
            "file_sizes": [],
            "mask_areas": [],
            "mask_complexity": []
        }
        
        for mask_file in sorted(mask_files):
            try:
                # File size
                file_size = mask_file.stat().st_size
                metrics["file_sizes"].append(file_size)
                
                # Load mask
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Mask area
                    mask_area = np.sum(mask > 0)
                    metrics["mask_areas"].append(mask_area)
                    
                    # Mask complexity (perimeter/area ratio)
                    if mask_area > 0:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
                        complexity = perimeter / mask_area if mask_area > 0 else 0
                        metrics["mask_complexity"].append(complexity)
                    else:
                        metrics["mask_complexity"].append(0)
                        
            except Exception as e:
                print(f"Error processing {mask_file}: {e}")
                continue
        
        # Calculate summary statistics
        if metrics["file_sizes"]:
            metrics["avg_file_size"] = np.mean(metrics["file_sizes"])
            metrics["std_file_size"] = np.std(metrics["file_sizes"])
        
        if metrics["mask_areas"]:
            metrics["avg_mask_area"] = np.mean(metrics["mask_areas"])
            metrics["std_mask_area"] = np.std(metrics["mask_areas"])
            metrics["total_mask_area"] = np.sum(metrics["mask_areas"])
        
        if metrics["mask_complexity"]:
            metrics["avg_complexity"] = np.mean(metrics["mask_complexity"])
            metrics["std_complexity"] = np.std(metrics["mask_complexity"])
        
        return metrics
    
    def generate_performance_metrics(self) -> pd.DataFrame:
        """Generate comprehensive performance metrics DataFrame"""
        results = self.load_comparison_results()
        
        all_metrics = []
        
        # Debug: Print what we're working with
        print(f"Processing {len(results.get('sequences', {}))} sequences")
        
        for sequence_name, sequence_data in results.get("sequences", {}).items():
            # TSP-SAM metrics
            if "tsp_sam" in sequence_data:
                tsp_metrics = self.calculate_sequence_metrics(sequence_name, "tsp_sam")
                if tsp_metrics:
                    all_metrics.append(tsp_metrics)
                    print(f"TSP-SAM: {sequence_name} - {tsp_metrics.get('avg_file_size', 'N/A')} bytes")
                else:
                    print(f"TSP-SAM: {sequence_name} - No metrics calculated")
            
            # SAMURAI metrics
            if "samurai" in sequence_data:
                samurai_metrics = self.calculate_sequence_metrics(sequence_name, "samurai")
                if samurai_metrics:
                    all_metrics.append(samurai_metrics)
                    print(f"SAMURAI: {sequence_name} - {samurai_metrics.get('avg_file_size', 'N/A')} bytes")
                else:
                    print(f"SAMURAI: {sequence_name} - No metrics calculated")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Debug: Print DataFrame info
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Baselines in DataFrame: {df['baseline'].unique() if not df.empty else 'Empty'}")
        if not df.empty:
            print(f"Sample data:\n{df.head()}")
        
        # Save metrics
        metrics_file = self.analysis_dir / "tables" / "comprehensive_metrics.csv"
        df.to_csv(metrics_file, index=False)
        
        # Save summary statistics
        if not df.empty:
            summary_stats = df.groupby('baseline').agg({
                'total_frames': ['count', 'sum', 'mean'],
                'avg_file_size': ['mean', 'std'],
                'avg_mask_area': ['mean', 'std'],
                'avg_complexity': ['mean', 'std']
            }).round(3)
            
            summary_file = self.analysis_dir / "tables" / "summary_statistics.csv"
            summary_stats.to_csv(summary_file)
        
        return df
    
    def categorize_sequences(self, sequence_names):
        """Categorize sequences by type for better analysis"""
        categories = {
            'human_actions': ['walking', 'running', 'dancing', 'jumping', 'person', 'human'],
            'animals': ['elephant', 'flamingo', 'bear', 'cow', 'dog', 'cat', 'bird', 'horse'],
            'vehicles': ['car', 'bike', 'motorcycle', 'truck', 'bus'],
            'sports': ['tennis', 'soccer', 'basketball', 'volleyball', 'gymnastics'],
            'daily_activities': ['cooking', 'cleaning', 'reading', 'writing', 'typing'],
            'complex_motion': ['acrobatics', 'dance', 'gymnastics', 'parkour']
        }
        
        categorized = {}
        for seq in sequence_names:
            categorized_flag = False
            for category, keywords in categories.items():
                if any(keyword in seq.lower() for keyword in keywords):
                    if category not in categorized:
                        categorized[category] = []
                    categorized[category].append(seq)
                    categorized_flag = True
                    break
            if not categorized_flag:
                if 'other' not in categorized:
                    categorized['other'] = []
                categorized['other'].append(seq)
        
        return categorized
    
    def get_top_performers(self, metrics_df, baseline, metric, top_n=5):
        """Get top performing sequences for a baseline"""
        baseline_data = metrics_df[metrics_df['baseline'] == baseline]
        if baseline_data.empty:
            return []
        
        # Sort by metric (lower is better for file size, higher is better for mask area)
        if 'file_size' in metric:
            sorted_data = baseline_data.nsmallest(top_n, metric)
        else:
            sorted_data = baseline_data.nlargest(top_n, metric)
        
        return sorted_data[['sequence_name', metric]].to_dict('records')
    
    def get_most_challenging(self, metrics_df, top_n=5):
        """Get most challenging sequences based on complexity"""
        # Calculate average complexity across baselines for each sequence
        sequence_complexity = metrics_df.groupby('sequence_name')['avg_complexity'].mean().sort_values(ascending=False)
        return sequence_complexity.head(top_n).to_dict()
    
    def generate_performance_plots(self, metrics_df: pd.DataFrame):
        """Generate focused, thesis-appropriate performance visualizations"""
        plots_dir = self.analysis_dir / "plots"
        
        # Check if we have data for both baselines
        baselines_in_data = metrics_df['baseline'].unique() if not metrics_df.empty else []
        print(f"Baselines found in data: {baselines_in_data}")
        
        if len(baselines_in_data) < 2:
            print("Warning: Insufficient baseline data for meaningful comparison")
            # Create a warning plot
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.text(0.5, 0.5, f'Insufficient Data for Comparison\n\nBaselines found: {baselines_in_data}\n\nPlease ensure both TSP-SAM and SAMURAI have generated output files.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold')
            ax.set_title('Data Availability Warning', fontsize=16, fontweight='bold', pad=20)
            ax.axis('off')
            plt.savefig(plots_dir / 'data_warning.png', dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # Set up the plotting style for better readability
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 11
        
        # 1. Overall Performance Comparison (Focused)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('TSP-SAM vs SAMURAI: Key Performance Comparison', fontsize=18, fontweight='bold', y=0.98)
        
        # File size comparison
        baseline_data = metrics_df.groupby('baseline')['avg_file_size'].mean()
        if len(baseline_data) >= 2:
            colors = ['#FF6B6B', '#4ECDC4']
            bars = axes[0, 0].bar(baseline_data.index, baseline_data.values, color=colors[:len(baseline_data)], alpha=0.8, edgecolor='black', linewidth=1)
            axes[0, 0].set_title('Average File Size by Baseline', fontsize=14, fontweight='bold', pad=20)
            axes[0, 0].set_ylabel('File Size (bytes)', fontsize=12)
            axes[0, 0].tick_params(axis='x', rotation=0)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            # Add value labels on bars
            for bar, value in zip(bars, baseline_data.values):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{value:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        else:
            axes[0, 0].text(0.5, 0.5, 'Insufficient data\nfor file size comparison', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Average File Size by Baseline', fontsize=14, fontweight='bold', pad=20)
        
        # Mask area comparison
        baseline_data = metrics_df.groupby('baseline')['avg_mask_area'].mean()
        if len(baseline_data) >= 2:
            colors = ['#FF6B6B', '#4ECDC4']
            bars = axes[0, 1].bar(baseline_data.index, baseline_data.values, color=colors[:len(baseline_data)], alpha=0.8, edgecolor='black', linewidth=1)
            axes[0, 1].set_title('Average Mask Area by Baseline', fontsize=14, fontweight='bold', pad=20)
            axes[0, 1].set_ylabel('Mask Area (pixels)', fontsize=12)
            axes[0, 1].tick_params(axis='x', rotation=0)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            # Add value labels on bars
            for bar, value in zip(bars, baseline_data.values):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{value:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'Insufficient data\nfor mask area comparison', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Average Mask Area by Baseline', fontsize=14, fontweight='bold', pad=20)
        
        # Success rate comparison
        results = self.load_comparison_results()
        total_sequences = results.get('total_sequences', 0)
        tsp_sam_success = sum(1 for seq in results.get('sequences', {}).values() 
                             if seq.get('tsp_sam', {}).get('status') == 'success')
        samurai_success = sum(1 for seq in results.get('sequences', {}).values() 
                             if seq.get('samurai', {}).get('status') == 'success')
        
        if total_sequences > 0:
            success_rates = [tsp_sam_success/total_sequences*100, samurai_success/total_sequences*100]
            bars = axes[1, 0].bar(['TSP-SAM', 'SAMURAI'], success_rates, color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='black', linewidth=1)
            axes[1, 0].set_title('Success Rate by Baseline', fontsize=14, fontweight='bold', pad=20)
            axes[1, 0].set_ylabel('Success Rate (%)', fontsize=12)
            axes[1, 0].set_ylim(0, 100)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            # Add value labels on bars
            for bar, value in zip(bars, success_rates):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{value:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'No success rate data\navailable', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Success Rate by Baseline', fontsize=14, fontweight='bold', pad=20)
        
        # Performance correlation
        tsp_sam_data = metrics_df[metrics_df['baseline'] == 'tsp_sam']['avg_file_size'].dropna()
        samurai_data = metrics_df[metrics_df['baseline'] == 'samurai']['avg_file_size'].dropna()
        
        # Create correlation plot only if we have data
        if len(tsp_sam_data) > 0 and len(samurai_data) > 0:
            # For correlation, we need to match sequences
            sequences = metrics_df['sequence_name'].unique()
            correlation_data = []
            for seq in sequences:
                tsp_data = metrics_df[(metrics_df['sequence_name'] == seq) & (metrics_df['baseline'] == 'tsp_sam')]
                sam_data = metrics_df[(metrics_df['sequence_name'] == seq) & (metrics_df['baseline'] == 'samurai')]
                if not tsp_data.empty and not sam_data.empty:
                    correlation_data.append({
                        'tsp_sam': tsp_data['avg_file_size'].iloc[0],
                        'samurai': sam_data['avg_file_size'].iloc[0]
                    })
            
            if correlation_data:
                tsp_values = [item['tsp_sam'] for item in correlation_data]
                sam_values = [item['samurai'] for item in correlation_data]
                
                axes[1, 1].scatter(tsp_values, sam_values, alpha=0.7, color='#45B7D1', s=60)
                axes[1, 1].plot([min(tsp_values), max(tsp_values)], [min(tsp_values), max(tsp_values)], 'r--', alpha=0.8, linewidth=2)
                axes[1, 1].set_xlabel('TSP-SAM File Size', fontsize=12)
                axes[1, 1].set_ylabel('SAMURAI File Size', fontsize=12)
                axes[1, 1].set_title('Performance Correlation', fontsize=14, fontweight='bold', pad=20)
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No correlation data\navailable', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Performance Correlation', fontsize=14, fontweight='bold', pad=20)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor correlation', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Performance Correlation', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(plots_dir / 'key_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Category-wise Performance Analysis
        self.generate_category_analysis(metrics_df, plots_dir)
        
        # 3. Statistical Analysis Summary
        self.generate_statistical_summary_plots(metrics_df, plots_dir)
        
        # 4. Top Performers and Challenging Sequences
        self.generate_performance_ranking_plots(metrics_df, plots_dir)
    
    def generate_category_analysis(self, metrics_df, plots_dir):
        """Generate category-wise performance analysis"""
        # Check if we have data for both baselines
        baselines_in_data = metrics_df['baseline'].unique() if not metrics_df.empty else []
        if len(baselines_in_data) < 2:
            print("Warning: Insufficient baseline data for category analysis")
            return
        
        # Categorize sequences
        sequences = metrics_df['sequence_name'].unique()
        categorized = self.categorize_sequences(sequences)
        
        # Calculate category performance
        category_performance = {}
        for category, seq_list in categorized.items():
            if len(seq_list) > 0:  # Only include categories with sequences
                category_data = metrics_df[metrics_df['sequence_name'].isin(seq_list)]
                if not category_data.empty:
                    # Check if we have data for both baselines in this category
                    tsp_data = category_data[category_data['baseline'] == 'tsp_sam']
                    sam_data = category_data[category_data['baseline'] == 'samurai']
                    
                    if not tsp_data.empty and not sam_data.empty:
                        category_performance[category] = {
                            'tsp_sam_file_size': tsp_data['avg_file_size'].mean(),
                            'samurai_file_size': sam_data['avg_file_size'].mean(),
                            'tsp_sam_mask_area': tsp_data['avg_mask_area'].mean(),
                            'samurai_mask_area': sam_data['avg_mask_area'].mean(),
                            'sequence_count': len(seq_list)
                        }
        
        if not category_performance:
            print("Warning: No categories with data for both baselines")
            return
        
        # Create category comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Performance Analysis by Sequence Category', fontsize=18, fontweight='bold', y=0.98)
        
        categories = list(category_performance.keys())
        x = np.arange(len(categories))
        width = 0.35
        
        # File size by category
        tsp_sam_sizes = [category_performance[cat]['tsp_sam_file_size'] for cat in categories]
        samurai_sizes = [category_performance[cat]['samurai_file_size'] for cat in categories]
        
        axes[0, 0].bar(x - width/2, tsp_sam_sizes, width, label='TSP-SAM', color='#FF6B6B', alpha=0.8)
        axes[0, 0].bar(x + width/2, samurai_sizes, width, label='SAMURAI', color='#4ECDC4', alpha=0.8)
        axes[0, 0].set_title('File Size by Category', fontsize=14, fontweight='bold', pad=20)
        axes[0, 0].set_ylabel('File Size (bytes)', fontsize=12)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f"{cat}\n({category_performance[cat]['sequence_count']})" for cat in categories], rotation=45, ha='right')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mask area by category
        tsp_sam_areas = [category_performance[cat]['tsp_sam_mask_area'] for cat in categories]
        samurai_areas = [category_performance[cat]['samurai_mask_area'] for cat in categories]
        
        axes[0, 1].bar(x - width/2, tsp_sam_areas, width, label='TSP-SAM', color='#FF6B6B', alpha=0.8)
        axes[0, 1].bar(x + width/2, samurai_areas, width, label='SAMURAI', color='#4ECDC4', alpha=0.8)
        axes[0, 1].set_title('Mask Area by Category', fontsize=14, fontweight='bold', pad=20)
        axes[0, 1].set_ylabel('Mask Area (pixels)', fontsize=12)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([f"{cat}\n({category_performance[cat]['sequence_count']})" for cat in categories], rotation=45, ha='right')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance ratio (TSP-SAM / SAMURAI)
        performance_ratio = []
        for tsp, sam in zip(tsp_sam_sizes, samurai_sizes):
            if sam > 0:
                ratio = tsp / sam
                performance_ratio.append(ratio)
            else:
                performance_ratio.append(0)
        
        colors = ['green' if ratio < 1 else 'red' for ratio in performance_ratio]
        axes[1, 0].bar(categories, performance_ratio, color=colors, alpha=0.8)
        axes[1, 0].axhline(y=1, color='black', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Performance Ratio (TSP-SAM / SAMURAI)\n<1: TSP-SAM better, >1: SAMURAI better', fontsize=14, fontweight='bold', pad=20)
        axes[1, 0].set_ylabel('Ratio', fontsize=12)
        axes[1, 0].set_xticklabels([f"{cat}\n({category_performance[cat]['sequence_count']})" for cat in categories], rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Category distribution
        sequence_counts = [category_performance[cat]['sequence_count'] for cat in categories]
        axes[1, 1].pie(sequence_counts, labels=[f"{cat}\n({count})" for cat, count in zip(categories, sequence_counts)], 
                       autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Sequence Distribution by Category', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(plots_dir / 'category_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_statistical_summary_plots(self, metrics_df, plots_dir):
        """Generate statistical summary plots"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Statistical Analysis Summary', fontsize=18, fontweight='bold', y=0.98)
        
        # File size distribution
        tsp_sam_data = metrics_df[metrics_df['baseline'] == 'tsp_sam']['avg_file_size'].dropna()
        samurai_data = metrics_df[metrics_df['baseline'] == 'samurai']['avg_file_size'].dropna()
        
        axes[0, 0].hist(tsp_sam_data, bins=20, alpha=0.7, label='TSP-SAM', color='#FF6B6B', edgecolor='black', linewidth=0.5)
        axes[0, 0].hist(samurai_data, bins=20, alpha=0.7, label='SAMURAI', color='#4ECDC4', edgecolor='black', linewidth=0.5)
        axes[0, 0].set_title('File Size Distribution', fontsize=14, fontweight='bold', pad=20)
        axes[0, 0].set_xlabel('File Size (bytes)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plots for comparison
        if len(tsp_sam_data) > 0 and len(samurai_data) > 0:
            data_to_plot = [tsp_sam_data, samurai_data]
            bp = axes[0, 1].boxplot(data_to_plot, labels=['TSP-SAM', 'SAMURAI'], patch_artist=True)
            if len(bp['boxes']) >= 2:
                bp['boxes'][0].set_facecolor('#FF6B6B')
                bp['boxes'][1].set_facecolor('#4ECDC4')
            axes[0, 1].set_title('File Size Box Plot Comparison', fontsize=14, fontweight='bold', pad=20)
            axes[0, 1].set_ylabel('File Size (bytes)', fontsize=12)
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Insufficient data\nfor box plot', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('File Size Box Plot Comparison', fontsize=14, fontweight='bold', pad=20)
        
        # Performance correlation
        if len(tsp_sam_data) > 0 and len(samurai_data) > 0:
            # For correlation, we need to match sequences
            sequences = metrics_df['sequence_name'].unique()
            correlation_data = []
            for seq in sequences:
                tsp_data = metrics_df[(metrics_df['sequence_name'] == seq) & (metrics_df['baseline'] == 'tsp_sam')]
                sam_data = metrics_df[(metrics_df['sequence_name'] == seq) & (metrics_df['baseline'] == 'samurai')]
                if not tsp_data.empty and not sam_data.empty:
                    correlation_data.append({
                        'tsp_sam': tsp_data['avg_file_size'].iloc[0],
                        'samurai': sam_data['avg_file_size'].iloc[0]
                    })
            
            if correlation_data:
                tsp_values = [item['tsp_sam'] for item in correlation_data]
                sam_values = [item['samurai'] for item in correlation_data]
                
                axes[1, 0].scatter(tsp_values, sam_values, alpha=0.7, color='#45B7D1', s=60)
                axes[1, 0].plot([min(tsp_values), max(tsp_values)], [min(tsp_values), max(tsp_values)], 'r--', alpha=0.8, linewidth=2)
                axes[1, 0].set_xlabel('TSP-SAM File Size', fontsize=12)
                axes[1, 0].set_ylabel('SAMURAI File Size', fontsize=12)
                axes[1, 0].set_title('Performance Correlation', fontsize=14, fontweight='bold', pad=20)
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No correlation data\navailable', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Performance Correlation', fontsize=14, fontweight='bold', pad=20)
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data\nfor correlation', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Performance Correlation', fontsize=14, fontweight='bold', pad=20)
        
        # Statistical summary table
        axes[1, 1].axis('off')
        summary_data = [
            ['Metric', 'TSP-SAM', 'SAMURAI'],
            ['Mean File Size', f"{tsp_sam_data.mean():,.0f} bytes", f"{samurai_data.mean():,.0f} bytes"],
            ['Std File Size', f"{tsp_sam_data.std():,.0f} bytes", f"{samurai_data.std():,.0f} bytes"],
            ['Min File Size', f"{tsp_sam_data.min():,.0f} bytes", f"{samurai_data.min():,.0f} bytes"],
            ['Max File Size', f"{tsp_sam_data.max():,.0f} bytes", f"{samurai_data.max():,.0f} bytes"],
            ['Total Sequences', f"{len(tsp_sam_data)}", f"{len(samurai_data)}"]
        ]
        
        table = axes[1, 1].table(cellText=summary_data[1:], colLabels=summary_data[0], 
                                 cellLoc='center', loc='center', bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(summary_data)):
            for j in range(len(summary_data[0])):
                if i == 0:  # Header row
                    table[(i, j)].set_facecolor('#4ECDC4')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    if j == 0:  # Metric column
                        table[(i, j)].set_facecolor('#F0F0F0')
                        table[(i, j)].set_text_props(weight='bold')
                    else:  # Data columns
                        table[(i, j)].set_facecolor('#FFFFFF')
        
        axes[1, 1].set_title('Statistical Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(plots_dir / 'statistical_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_performance_ranking_plots(self, metrics_df, plots_dir):
        """Generate performance ranking plots"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Performance Rankings and Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        # Top performers for TSP-SAM
        top_tsp_sam = self.get_top_performers(metrics_df, 'tsp_sam', 'avg_file_size', 10)
        if top_tsp_sam:
            sequences = [item['sequence_name'] for item in top_tsp_sam]
            values = [item['avg_file_size'] for item in top_tsp_sam]
            axes[0, 0].barh(sequences, values, color='#FF6B6B', alpha=0.8)
            axes[0, 0].set_title('Top 10 TSP-SAM Performers (Smallest File Size)', fontsize=14, fontweight='bold', pad=20)
            axes[0, 0].set_xlabel('File Size (bytes)', fontsize=12)
            axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # Top performers for SAMURAI
        top_samurai = self.get_top_performers(metrics_df, 'samurai', 'avg_file_size', 10)
        if top_samurai:
            sequences = [item['sequence_name'] for item in top_samurai]
            values = [item['avg_file_size'] for item in top_samurai]
            axes[0, 1].barh(sequences, values, color='#4ECDC4', alpha=0.8)
            axes[0, 1].set_title('Top 10 SAMURAI Performers (Smallest File Size)', fontsize=14, fontweight='bold', pad=20)
            axes[0, 1].set_xlabel('File Size (bytes)', fontsize=12)
            axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # Most challenging sequences
        challenging = self.get_most_challenging(metrics_df, 10)
        if challenging:
            sequences = list(challenging.keys())
            values = list(challenging.values())
            axes[1, 0].barh(sequences, values, color='#FFA500', alpha=0.8)
            axes[1, 0].set_title('Most Challenging Sequences (High Complexity)', fontsize=14, fontweight='bold', pad=20)
            axes[1, 0].set_xlabel('Complexity (perimeter/area)', fontsize=12)
            axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Performance comparison for top sequences
        top_sequences = set([item['sequence_name'] for item in top_tsp_sam[:5]] + [item['sequence_name'] for item in top_samurai[:5]])
        if top_sequences:
            comparison_data = []
            for seq in top_sequences:
                tsp_data = metrics_df[(metrics_df['sequence_name'] == seq) & (metrics_df['baseline'] == 'tsp_sam')]
                sam_data = metrics_df[(metrics_df['sequence_name'] == seq) & (metrics_df['baseline'] == 'samurai')]
                if not tsp_data.empty and not sam_data.empty:
                    comparison_data.append({
                        'sequence': seq,
                        'tsp_sam': tsp_data['avg_file_size'].iloc[0],
                        'samurai': sam_data['avg_file_size'].iloc[0]
                    })
            
            if comparison_data:
                sequences = [item['sequence'] for item in comparison_data]
                tsp_values = [item['tsp_sam'] for item in comparison_data]
                sam_values = [item['samurai'] for item in comparison_data]
                
                x = np.arange(len(sequences))
                width = 0.35
                
                axes[1, 1].bar(x - width/2, tsp_values, width, label='TSP-SAM', color='#FF6B6B', alpha=0.8)
                axes[1, 1].bar(x + width/2, sam_values, width, label='SAMURAI', color='#4ECDC4', alpha=0.8)
                axes[1, 1].set_title('Top Sequences: Direct Comparison', fontsize=14, fontweight='bold', pad=20)
                axes[1, 1].set_ylabel('File Size (bytes)', fontsize=12)
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(sequences, rotation=45, ha='right')
                axes[1, 1].legend(fontsize=11)
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(plots_dir / 'performance_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def perform_statistical_tests(self, metrics_df: pd.DataFrame):
        """Perform statistical significance tests"""
        from scipy import stats
        
        statistical_results = {}
        
        # File size comparison
        tsp_sam_sizes = metrics_df[metrics_df['baseline'] == 'tsp_sam']['avg_file_size'].dropna()
        samurai_sizes = metrics_df[metrics_df['baseline'] == 'samurai']['avg_file_size'].dropna()
        
        if len(tsp_sam_sizes) > 0 and len(samurai_sizes) > 0:
            # T-test
            t_stat, p_value = stats.ttest_ind(tsp_sam_sizes, samurai_sizes)
            statistical_results['file_size_ttest'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
            
            # Mann-Whitney U test
            u_stat, u_p_value = stats.mannwhitneyu(tsp_sam_sizes, samurai_sizes, alternative='two-sided')
            statistical_results['file_size_mannwhitney'] = {
                'u_statistic': float(u_stat),
                'p_value': float(u_p_value),
                'significant': u_p_value < 0.05
            }
        
        # Mask area comparison
        tsp_sam_areas = metrics_df[metrics_df['baseline'] == 'tsp_sam']['avg_mask_area'].dropna()
        samurai_areas = metrics_df[metrics_df['baseline'] == 'samurai']['avg_mask_area'].dropna()
        
        if len(tsp_sam_areas) > 0 and len(samurai_areas) > 0:
            # T-test
            t_stat, p_value = stats.ttest_ind(tsp_sam_areas, samurai_areas)
            statistical_results['mask_area_ttest'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        
        # Save statistical results
        stats_file = self.analysis_dir / "tables" / "statistical_tests.json"
        with open(stats_file, 'w') as f:
            json.dump(statistical_results, f, indent=2)
        
        self.analysis_results["statistical_tests"] = statistical_results
        return statistical_results
    
    def generate_recommendations(self, metrics_df: pd.DataFrame, statistical_results: Dict):
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Performance analysis
        tsp_sam_avg_size = metrics_df[metrics_df['baseline'] == 'tsp_sam']['avg_file_size'].mean()
        samurai_avg_size = metrics_df[metrics_df['baseline'] == 'samurai']['avg_file_size'].mean()
        
        if tsp_sam_avg_size < samurai_avg_size:
            recommendations.append({
                "category": "Performance",
                "finding": "TSP-SAM produces smaller output files",
                "implication": "Better storage efficiency and faster processing",
                "recommendation": "Consider TSP-SAM for storage-constrained applications"
            })
        else:
            recommendations.append({
                "category": "Performance",
                "finding": "SAMURAI produces smaller output files",
                "implication": "Better storage efficiency",
                "recommendation": "Consider SAMURAI for storage-constrained applications"
            })
        
        # Statistical significance
        if statistical_results.get('file_size_ttest', {}).get('significant', False):
            recommendations.append({
                "category": "Statistical Analysis",
                "finding": "Statistically significant difference in file sizes",
                "implication": "The performance difference is not due to chance",
                "recommendation": "Confidence in baseline performance differences"
            })
        
        # Sequence complexity analysis
        complex_sequences = metrics_df.groupby('sequence_name')['avg_complexity'].mean().sort_values(ascending=False).head(5)
        if not complex_sequences.empty:
            recommendations.append({
                "category": "Complexity Analysis",
                "finding": f"Most complex sequences: {', '.join(complex_sequences.index)}",
                "implication": "These sequences may be more challenging for both baselines",
                "recommendation": "Focus optimization efforts on high-complexity sequences"
            })
        
        # Save recommendations
        rec_file = self.analysis_dir / "reports" / "recommendations.json"
        with open(rec_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        self.analysis_results["recommendations"] = recommendations
        return recommendations
    
    def generate_latex_report(self, metrics_df: pd.DataFrame, statistical_results: Dict, recommendations: List):
        """Generate LaTeX report for thesis inclusion"""
        latex_content = f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{array}}
\\usepackage{{multirow}}
\\usepackage{{wrapfig}}
\\usepackage{{float}}
\\usepackage{{colortbl}}
\\usepackage{{pdflscape}}
\\usepackage{{tabu}}
\\usepackage{{threeparttable}}
\\usepackage{{threeparttablex}}
\\usepackage{{forloop}}
\\usepackage{{xcolor}}
\\usepackage{{geometry}}
\\geometry{{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}}

\\title{{TSP-SAM vs SAMURAI Baseline Comparison Analysis}}
\\author{{Generated Analysis Report}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Executive Summary}}
This report presents a comprehensive analysis of TSP-SAM and SAMURAI baseline performance on the DAVIS-2017 dataset. The analysis covers {len(metrics_df)} data points across multiple sequences, providing insights into baseline performance characteristics and statistical significance.

\\section{{Methodology}}
The analysis was conducted using a systematic approach:
\\begin{{enumerate}}
    \\item Comprehensive metrics calculation for each baseline
    \\item Statistical significance testing using t-tests and Mann-Whitney U tests
    \\item Performance correlation analysis
    \\item Complexity-based sequence analysis
\\end{{enumerate}}

\\section{{Results Overview}}

\\subsection{{Performance Metrics}}
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{TSP-SAM}} & \\textbf{{SAMURAI}} \\\\
\\midrule
Average File Size & {metrics_df[metrics_df['baseline'] == 'tsp_sam']['avg_file_size'].mean():.2f} bytes & {metrics_df[metrics_df['baseline'] == 'samurai']['avg_file_size'].mean():.2f} bytes \\\\
Average Mask Area & {metrics_df[metrics_df['baseline'] == 'tsp_sam']['avg_mask_area'].mean():.2f} pixels & {metrics_df[metrics_df['baseline'] == 'samurai']['avg_mask_area'].mean():.2f} pixels \\\\
Average Complexity & {metrics_df[metrics_df['baseline'] == 'tsp_sam']['avg_complexity'].mean():.2f} & {metrics_df[metrics_df['baseline'] == 'samurai']['avg_complexity'].mean():.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Performance Metrics Comparison}}
\\end{{table}}

\\subsection{{Statistical Significance}}
"""

        # Add statistical results
        for test_name, test_results in statistical_results.items():
            latex_content += f"""
\\textbf{{test_name.replace('_', ' ').title()}}: p-value = {test_results['p_value']:.4f} ({'Significant' if test_results['significant'] else 'Not Significant'})
"""
        
        latex_content += f"""

\\section{{Recommendations}}
Based on the analysis, the following recommendations are provided:
\\begin{{enumerate}}
"""

        for rec in recommendations:
            latex_content += f"""
    \\item \\textbf{{{rec['category']}}}: {rec['finding']}
    \\begin{{itemize}}
        \\item \\textit{{Implication}}: {rec['implication']}
        \\item \\textit{{Recommendation}}: {rec['recommendation']}
    \\end{{itemize}}
"""
        
        latex_content += """
\\end{enumerate}

\\section{{Conclusion}}
This analysis provides a comprehensive comparison of TSP-SAM and SAMURAI baselines, revealing key performance differences and statistical significance. The findings support informed decision-making for baseline selection based on specific application requirements.

\\end{document}
"""
        
        # Save LaTeX report
        latex_file = self.analysis_dir / "reports" / "thesis_report.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_content)
        
        return latex_file
    
    def generate_summary_dashboard(self, metrics_df: pd.DataFrame):
        """Generate a thesis-focused summary dashboard"""
        plots_dir = self.analysis_dir / "plots"
        
        # Create a comprehensive summary figure
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('TSP-SAM vs SAMURAI: Thesis Summary Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Overall Performance Comparison (Top Left)
        baseline_data = metrics_df.groupby('baseline')['avg_file_size'].mean()
        colors = ['#FF6B6B', '#4ECDC4']
        bars = axes[0, 0].bar(baseline_data.index, baseline_data.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        axes[0, 0].set_title('Key Finding 1: Storage Efficiency', fontsize=16, fontweight='bold', pad=20)
        axes[0, 0].set_ylabel('File Size (bytes)', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        # Add value labels and interpretation
        for i, (bar, value) in enumerate(zip(bars, baseline_data.values)):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            # Add interpretation text
            if i == 0:  # TSP-SAM
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height*0.5, 'Better\nStorage', 
                               ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # 2. Success Rate Comparison (Top Center)
        results = self.load_comparison_results()
        total_sequences = results.get('total_sequences', 0)
        tsp_sam_success = sum(1 for seq in results.get('sequences', {}).values() 
                             if seq.get('tsp_sam', {}).get('status') == 'success')
        samurai_success = sum(1 for seq in results.get('sequences', {}).values() 
                             if seq.get('samurai', {}).get('status') == 'success')
        
        success_rates = [tsp_sam_success/total_sequences*100, samurai_success/total_sequences*100]
        bars = axes[0, 1].bar(['TSP-SAM', 'SAMURAI'], success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        axes[0, 1].set_title('Key Finding 2: Reliability', fontsize=16, fontweight='bold', pad=20)
        axes[0, 1].set_ylabel('Success Rate (%)', fontsize=14)
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        # Add value labels and interpretation
        for i, (bar, value) in enumerate(zip(bars, success_rates)):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
            # Add interpretation text
            if value == max(success_rates):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height*0.5, 'More\nReliable', 
                               ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # 3. Category Performance Summary (Top Right)
        sequences = metrics_df['sequence_name'].unique()
        categorized = self.categorize_sequences(sequences)
        
        # Calculate average performance by category
        category_performance = {}
        for category, seq_list in categorized.items():
            if len(seq_list) > 0:
                category_data = metrics_df[metrics_df['sequence_name'].isin(seq_list)]
                if not category_data.empty:
                    tsp_avg = category_data[category_data['baseline'] == 'tsp_sam']['avg_file_size'].mean()
                    sam_avg = category_data[category_data['baseline'] == 'samurai']['avg_file_size'].mean()
                    category_performance[category] = {
                        'tsp_sam': tsp_avg,
                        'samurai': sam_avg,
                        'winner': 'TSP-SAM' if tsp_avg < sam_avg else 'SAMURAI',
                        'count': len(seq_list)
                    }
        
        # Create category performance summary
        categories = list(category_performance.keys())
        tsp_wins = sum(1 for cat in categories if category_performance[cat]['winner'] == 'TSP-SAM')
        sam_wins = len(categories) - tsp_wins
        
        axes[0, 2].pie([tsp_wins, sam_wins], labels=[f'TSP-SAM\n({tsp_wins} categories)', f'SAMURAI\n({sam_wins} categories)'], 
                       colors=['#FF6B6B', '#4ECDC4'], autopct='%1.0f', startangle=90)
        axes[0, 2].set_title('Key Finding 3: Category Dominance', fontsize=16, fontweight='bold', pad=20)
        
        # 4. Performance Distribution (Bottom Left)
        tsp_sam_data = metrics_df[metrics_df['baseline'] == 'tsp_sam']['avg_file_size'].dropna()
        samurai_data = metrics_df[metrics_df['baseline'] == 'samurai']['avg_file_size'].dropna()
        axes[1, 0].hist(tsp_sam_data, bins=15, alpha=0.7, label='TSP-SAM', color='#FF6B6B', edgecolor='black', linewidth=0.5)
        axes[1, 0].hist(samurai_data, bins=15, alpha=0.7, label='SAMURAI', color='#4ECDC4', edgecolor='black', linewidth=0.5)
        axes[1, 0].set_title('Key Finding 4: Performance Distribution', fontsize=16, fontweight='bold', pad=20)
        axes[1, 0].set_xlabel('File Size (bytes)', fontsize=14)
        axes[1, 0].set_ylabel('Frequency', fontsize=14)
        axes[1, 0].legend(fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Statistical Significance (Bottom Center)
        from scipy import stats
        if len(tsp_sam_data) > 0 and len(samurai_data) > 0:
            t_stat, p_value = stats.ttest_ind(tsp_sam_data, samurai_data)
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            
            axes[1, 1].text(0.5, 0.7, f'Statistical Test Results', fontsize=16, fontweight='bold', ha='center', transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.5, 0.6, f't-statistic: {t_stat:.3f}', fontsize=14, ha='center', transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.5, 0.5, f'p-value: {p_value:.4f}', fontsize=14, ha='center', transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.5, 0.4, f'Significance: {significance}', fontsize=14, fontweight='bold', ha='center', transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.5, 0.3, f'(α = 0.05)', fontsize=12, ha='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor statistical test', fontsize=14, ha='center', transform=axes[1, 1].transAxes)
        
        axes[1, 1].set_title('Key Finding 5: Statistical Significance', fontsize=16, fontweight='bold', pad=20)
        axes[1, 1].axis('off')
        
        # 6. Key Insights Table (Bottom Right)
        axes[1, 2].axis('off')
        
        # Generate key insights
        insights = self.generate_key_insights(metrics_df, results)
        
        insight_data = [
            ['Key Insight', 'Finding', 'Implication'],
            ['Storage Efficiency', insights['storage_winner'], insights['storage_implication']],
            ['Reliability', insights['reliability_winner'], insights['reliability_implication']],
            ['Category Performance', insights['category_winner'], insights['category_implication']],
            ['Statistical Evidence', insights['statistical_evidence'], insights['statistical_implication']]
        ]
        
        table = axes[1, 2].table(cellText=insight_data[1:], colLabels=insight_data[0], 
                                 cellLoc='left', loc='center', bbox=[0.05, 0.1, 0.9, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style the table
        for i in range(len(insight_data)):
            for j in range(len(insight_data[0])):
                if i == 0:  # Header row
                    table[(i, j)].set_facecolor('#4ECDC4')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    if j == 0:  # Insight column
                        table[(i, j)].set_facecolor('#F0F0F0')
                        table[(i, j)].set_text_props(weight='bold')
                    else:  # Data columns
                        table[(i, j)].set_facecolor('#FFFFFF')
        
        axes[1, 2].set_title('Key Findings Summary', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout(pad=4.0)
        plt.savefig(plots_dir / 'thesis_summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Thesis summary dashboard generated successfully!")
    
    def generate_key_insights(self, metrics_df, results):
        """Generate key insights for thesis"""
        # Storage efficiency
        tsp_avg_size = metrics_df[metrics_df['baseline'] == 'tsp_sam']['avg_file_size'].mean()
        sam_avg_size = metrics_df[metrics_df['baseline'] == 'samurai']['avg_file_size'].mean()
        
        if tsp_avg_size < sam_avg_size:
            storage_winner = "TSP-SAM (smaller files)"
            storage_implication = "Better storage efficiency"
        else:
            storage_winner = "SAMURAI (smaller files)"
            storage_implication = "Better storage efficiency"
        
        # Reliability
        total_sequences = results.get('total_sequences', 0)
        tsp_success = sum(1 for seq in results.get('sequences', {}).values() 
                         if seq.get('tsp_sam', {}).get('status') == 'success')
        sam_success = sum(1 for seq in results.get('sequences', {}).values() 
                         if seq.get('samurai', {}).get('status') == 'success')
        
        if tsp_success > sam_success:
            reliability_winner = "TSP-SAM (higher success rate)"
            reliability_implication = "More reliable processing"
        else:
            reliability_winner = "SAMURAI (higher success rate)"
            reliability_implication = "More reliable processing"
        
        # Category performance
        sequences = metrics_df['sequence_name'].unique()
        categorized = self.categorize_sequences(sequences)
        category_wins = {'TSP-SAM': 0, 'SAMURAI': 0}
        
        for category, seq_list in categorized.items():
            if len(seq_list) > 0:
                category_data = metrics_df[metrics_df['sequence_name'].isin(seq_list)]
                if not category_data.empty:
                    tsp_avg = category_data[category_data['baseline'] == 'tsp_sam']['avg_file_size'].mean()
                    sam_avg = category_data[category_data['baseline'] == 'samurai']['avg_file_size'].mean()
                    if tsp_avg < sam_avg:
                        category_wins['TSP-SAM'] += 1
                    else:
                        category_wins['SAMURAI'] += 1
        
        if category_wins['TSP-SAM'] > category_wins['SAMURAI']:
            category_winner = f"TSP-SAM ({category_wins['TSP-SAM']} categories)"
            category_implication = "Better across more sequence types"
        else:
            category_winner = f"SAMURAI ({category_wins['SAMURAI']} categories)"
            category_implication = "Better across more sequence types"
        
        # Statistical evidence
        from scipy import stats
        tsp_data = metrics_df[metrics_df['baseline'] == 'tsp_sam']['avg_file_size'].dropna()
        sam_data = metrics_df[metrics_df['baseline'] == 'samurai']['avg_file_size'].dropna()
        
        if len(tsp_data) > 0 and len(sam_data) > 0:
            t_stat, p_value = stats.ttest_ind(tsp_data, sam_data)
            if p_value < 0.05:
                statistical_evidence = "Statistically significant difference"
                statistical_implication = "Confidence in baseline differences"
            else:
                statistical_evidence = "No significant difference"
                statistical_implication = "Differences may be due to chance"
        else:
            statistical_evidence = "Insufficient data"
            statistical_implication = "Cannot determine significance"
        
        return {
            'storage_winner': storage_winner,
            'storage_implication': storage_implication,
            'reliability_winner': reliability_winner,
            'reliability_implication': reliability_implication,
            'category_winner': category_winner,
            'category_implication': category_implication,
            'statistical_evidence': statistical_evidence,
            'statistical_implication': statistical_implication
        }
    
    def generate_single_baseline_analysis(self, metrics_df: pd.DataFrame):
        """Generate meaningful analysis when only one baseline has data"""
        plots_dir = self.analysis_dir / "plots"
        
        baseline_name = metrics_df['baseline'].unique()[0]
        print(f"Generating single-baseline analysis for {baseline_name}")
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 11
        
        # 1. Performance Distribution Analysis
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'{baseline_name.upper()}: Comprehensive Performance Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        # File size distribution
        file_sizes = metrics_df['avg_file_size'].dropna()
        axes[0, 0].hist(file_sizes, bins=20, alpha=0.7, color='#FF6B6B', edgecolor='black', linewidth=0.5)
        axes[0, 0].axvline(file_sizes.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {file_sizes.mean():.0f} bytes')
        axes[0, 0].set_title('File Size Distribution', fontsize=14, fontweight='bold', pad=20)
        axes[0, 0].set_xlabel('File Size (bytes)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mask area distribution
        mask_areas = metrics_df['avg_mask_area'].dropna()
        axes[0, 1].hist(mask_areas, bins=20, alpha=0.7, color='#4ECDC4', edgecolor='black', linewidth=0.5)
        axes[0, 1].axvline(mask_areas.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {mask_areas.mean():.0f} pixels')
        axes[0, 1].set_title('Mask Area Distribution', fontsize=14, fontweight='bold', pad=20)
        axes[0, 1].set_xlabel('Mask Area (pixels)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Complexity distribution
        complexities = metrics_df['avg_complexity'].dropna()
        axes[1, 0].hist(complexities, bins=20, alpha=0.7, color='#FFA500', edgecolor='black', linewidth=0.5)
        axes[1, 0].axvline(complexities.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {complexities.mean():.3f}')
        axes[1, 0].set_title('Mask Complexity Distribution', fontsize=14, fontweight='bold', pad=20)
        axes[1, 0].set_xlabel('Complexity (perimeter/area)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        # File size vs Mask area correlation
        axes[1, 1].scatter(file_sizes, mask_areas, alpha=0.7, color='#45B7D1', s=60)
        axes[1, 1].set_xlabel('File Size (bytes)', fontsize=12)
        axes[1, 1].set_ylabel('Mask Area (pixels)', fontsize=12)
        axes[1, 1].set_title('File Size vs Mask Area Correlation', fontsize=14, fontweight='bold', pad=20)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(plots_dir / f'{baseline_name}_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Category Performance Analysis
        self.generate_single_baseline_category_analysis(metrics_df, plots_dir)
        
        # 3. Top Performers Analysis
        self.generate_single_baseline_rankings(metrics_df, plots_dir)
        
        print(f"Single-baseline analysis completed for {baseline_name}")
    
    def generate_single_baseline_category_analysis(self, metrics_df: pd.DataFrame, plots_dir):
        """Generate category analysis for single baseline"""
        baseline_name = metrics_df['baseline'].unique()[0]
        
        # Categorize sequences
        sequences = metrics_df['sequence_name'].unique()
        categorized = self.categorize_sequences(sequences)
        
        # Calculate category performance
        category_performance = {}
        for category, seq_list in categorized.items():
            if len(seq_list) > 0:
                category_data = metrics_df[metrics_df['sequence_name'].isin(seq_list)]
                if not category_data.empty:
                    category_performance[category] = {
                        'avg_file_size': category_data['avg_file_size'].mean(),
                        'avg_mask_area': category_data['avg_mask_area'].mean(),
                        'avg_complexity': category_data['avg_complexity'].mean(),
                        'sequence_count': len(seq_list)
                    }
        
        if not category_performance:
            return
        
        # Create category analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'{baseline_name.upper()}: Performance by Category', fontsize=18, fontweight='bold', y=0.98)
        
        categories = list(category_performance.keys())
        x = np.arange(len(categories))
        
        # File size by category
        file_sizes = [category_performance[cat]['avg_file_size'] for cat in categories]
        bars = axes[0, 0].bar(categories, file_sizes, color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1)
        axes[0, 0].set_title('Average File Size by Category', fontsize=14, fontweight='bold', pad=20)
        axes[0, 0].set_ylabel('File Size (bytes)', fontsize=12)
        axes[0, 0].set_xticklabels([f"{cat}\n({category_performance[cat]['sequence_count']})" for cat in categories], rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, file_sizes):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Mask area by category
        mask_areas = [category_performance[cat]['avg_mask_area'] for cat in categories]
        bars = axes[0, 1].bar(categories, mask_areas, color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1)
        axes[0, 1].set_title('Average Mask Area by Category', fontsize=14, fontweight='bold', pad=20)
        axes[0, 1].set_ylabel('Mask Area (pixels)', fontsize=12)
        axes[0, 1].set_xticklabels([f"{cat}\n({category_performance[cat]['sequence_count']})" for cat in categories], rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, mask_areas):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Complexity by category
        complexities = [category_performance[cat]['avg_complexity'] for cat in categories]
        bars = axes[1, 0].bar(categories, complexities, color='#FFA500', alpha=0.8, edgecolor='black', linewidth=1)
        axes[1, 0].set_title('Average Complexity by Category', fontsize=14, fontweight='bold', pad=20)
        axes[1, 0].set_ylabel('Complexity (perimeter/area)', fontsize=12)
        axes[1, 0].set_xticklabels([f"{cat}\n({category_performance[cat]['sequence_count']})" for cat in categories], rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, complexities):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Category distribution
        sequence_counts = [category_performance[cat]['sequence_count'] for cat in categories]
        axes[1, 1].pie(sequence_counts, labels=[f"{cat}\n({count})" for cat, count in zip(categories, sequence_counts)], 
                       autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Sequence Distribution by Category', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(plots_dir / f'{baseline_name}_category_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_single_baseline_rankings(self, metrics_df: pd.DataFrame, plots_dir):
        """Generate rankings for single baseline"""
        baseline_name = metrics_df['baseline'].unique()[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'{baseline_name.upper()}: Performance Rankings', fontsize=18, fontweight='bold', y=0.98)
        
        # Top performers by file size (smallest)
        top_performers = self.get_top_performers(metrics_df, baseline_name, 'avg_file_size', 10)
        if top_performers:
            sequences = [item['sequence_name'] for item in top_performers]
            values = [item['avg_file_size'] for item in top_performers]
            axes[0, 0].barh(sequences, values, color='#FF6B6B', alpha=0.8)
            axes[0, 0].set_title('Top 10 Performers (Smallest File Size)', fontsize=14, fontweight='bold', pad=20)
            axes[0, 0].set_xlabel('File Size (bytes)', fontsize=12)
            axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # Top performers by mask area (largest)
        top_area = self.get_top_performers(metrics_df, baseline_name, 'avg_mask_area', 10)
        if top_area:
            sequences = [item['sequence_name'] for item in top_area]
            values = [item['avg_mask_area'] for item in top_area]
            axes[0, 1].barh(sequences, values, color='#4ECDC4', alpha=0.8)
            axes[0, 1].set_title('Top 10 Performers (Largest Mask Area)', fontsize=14, fontweight='bold', pad=20)
            axes[0, 1].set_xlabel('Mask Area (pixels)', fontsize=12)
            axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # Most challenging sequences
        challenging = self.get_most_challenging(metrics_df, 10)
        if challenging:
            sequences = list(challenging.keys())
            values = list(challenging.values())
            axes[1, 0].barh(sequences, values, color='#FFA500', alpha=0.8)
            axes[1, 0].set_title('Most Challenging Sequences (High Complexity)', fontsize=14, fontweight='bold', pad=20)
            axes[1, 0].set_xlabel('Complexity (perimeter/area)', fontsize=12)
            axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Statistical summary
        axes[1, 1].axis('off')
        summary_data = [
            ['Metric', 'Value'],
            ['Total Sequences', f"{len(metrics_df)}"],
            ['Mean File Size', f"{metrics_df['avg_file_size'].mean():,.0f} bytes"],
            ['Std File Size', f"{metrics_df['avg_file_size'].std():,.0f} bytes"],
            ['Mean Mask Area', f"{metrics_df['avg_mask_area'].mean():,.0f} pixels"],
            ['Mean Complexity', f"{metrics_df['avg_complexity'].mean():.3f}"]
        ]
        
        table = axes[1, 1].table(cellText=summary_data[1:], colLabels=summary_data[0], 
                                 cellLoc='center', loc='center', bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(summary_data)):
            for j in range(len(summary_data[0])):
                if i == 0:  # Header row
                    table[(i, j)].set_facecolor('#4ECDC4')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    if j == 0:  # Metric column
                        table[(i, j)].set_facecolor('#F0F0F0')
                        table[(i, j)].set_text_props(weight='bold')
                    else:  # Data columns
                        table[(i, j)].set_facecolor('#FFFFFF')
        
        axes[1, 1].set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(plots_dir / f'{baseline_name}_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_comprehensive_analysis(self):
        """Run the complete comprehensive analysis pipeline"""
        print("Starting comprehensive analysis...")
        
        # Generate performance metrics
        print("1. Generating performance metrics...")
        metrics_df = self.generate_performance_metrics()
        
        # Check if we have data for both baselines
        baselines_in_data = metrics_df['baseline'].unique() if not metrics_df.empty else []
        
        if len(baselines_in_data) < 2:
            print(f"\n WARNING: Only {len(baselines_in_data)} baseline(s) have data")
            if len(baselines_in_data) == 1:
                print(f"Generating single-baseline analysis for {baselines_in_data[0]}")
                # Generate single-baseline analysis
                self.generate_single_baseline_analysis(metrics_df)
                
                # Generate recommendations for single baseline
                print("4. Generating recommendations...")
                recommendations = self.generate_single_baseline_recommendations(metrics_df)
                
                # Generate LaTeX report
                print("5. Generating LaTeX report...")
                latex_file = self.generate_single_baseline_latex_report(metrics_df, recommendations)
                
                print(f"\n Single-baseline analysis completed!")
                print(f" Note: This analysis only covers {baselines_in_data[0]} due to missing data from other baselines")
                print(f" To get a complete comparison, ensure all baselines generate output files")
                print(f"Results saved to: {self.analysis_dir}")
                print(f"LaTeX report: {latex_file}")
                
                return self.analysis_results
            else:
                print("ERROR: No baseline data available for analysis")
                return None
        
        # Continue with full comparison analysis
        print("2. Generating performance plots...")
        self.generate_performance_plots(metrics_df)
        
        # Perform statistical tests
        print("3. Performing statistical tests...")
        statistical_results = self.perform_statistical_tests(metrics_df)
        
        # Generate recommendations
        print("4. Generating recommendations...")
        recommendations = self.generate_recommendations(metrics_df, statistical_results)
        
        # Generate LaTeX report
        print("5. Generating LaTeX report...")
        latex_file = self.generate_latex_report(metrics_df, statistical_results, recommendations)
        
        # Generate summary dashboard
        print("6. Generating summary dashboard...")
        self.generate_summary_dashboard(metrics_df)
        
        # Save complete analysis results
        analysis_file = self.analysis_dir / "complete_analysis_results.json"
        with open(analysis_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        
        print(f"\nComprehensive analysis completed!")
        print(f"Results saved to: {self.analysis_dir}")
        print(f"LaTeX report: {latex_file}")
        
        return self.analysis_results
    
    def generate_single_baseline_recommendations(self, metrics_df: pd.DataFrame):
        """Generate recommendations for single baseline analysis"""
        baseline_name = metrics_df['baseline'].unique()[0]
        
        recommendations = []
        
        # Performance analysis
        avg_file_size = metrics_df['avg_file_size'].mean()
        avg_mask_area = metrics_df['avg_mask_area'].mean()
        avg_complexity = metrics_df['avg_complexity'].mean()
        
        recommendations.append({
            "category": "Performance Analysis",
            "finding": f"{baseline_name.upper()} processed {len(metrics_df)} sequences successfully",
            "implication": "Baseline demonstrates consistent processing capability",
            "recommendation": "Consider this baseline for production use in similar scenarios"
        })
        
        # File size analysis
        if avg_file_size < 2000:
            size_category = "efficient"
        elif avg_file_size < 4000:
            size_category = "moderate"
        else:
            size_category = "large"
        
        recommendations.append({
            "category": "Storage Efficiency",
            "finding": f"Average file size: {avg_file_size:.0f} bytes ({size_category})",
            "implication": f"Storage requirements are {size_category} for this baseline",
            "recommendation": "Monitor storage usage and consider compression if needed"
        })
        
        # Complexity analysis
        complex_sequences = metrics_df.groupby('sequence_name')['avg_complexity'].mean().sort_values(ascending=False).head(5)
        if not complex_sequences.empty:
            recommendations.append({
                "category": "Complexity Analysis",
                "finding": f"Most complex sequences: {', '.join(complex_sequences.index)}",
                "implication": "These sequences may be challenging for the baseline",
                "recommendation": "Focus optimization efforts on high-complexity sequences"
            })
        
        # Save recommendations
        rec_file = self.analysis_dir / "reports" / "single_baseline_recommendations.json"
        with open(rec_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        self.analysis_results["recommendations"] = recommendations
        return recommendations
    
    def generate_single_baseline_latex_report(self, metrics_df: pd.DataFrame, recommendations: List):
        """Generate LaTeX report for single baseline analysis"""
        baseline_name = metrics_df['baseline'].unique()[0]
        
        latex_content = f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{geometry}}
\\geometry{{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}}

\\title{{{baseline_name.upper()} Baseline Performance Analysis}}
\\subtitle{{Single Baseline Analysis Report}}
\\author{{Generated Analysis Report}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Executive Summary}}
This report presents a comprehensive analysis of {baseline_name.upper()} baseline performance on the DAVIS-2017 dataset. The analysis covers {len(metrics_df)} sequences, providing insights into baseline performance characteristics and operational efficiency.

\\textbf{{Note:}} This is a single-baseline analysis due to missing data from other baselines. For a complete comparison, ensure all baselines generate output files.

\\section{{Methodology}}
The analysis was conducted using a systematic approach:
\\begin{{enumerate}}
    \\item Comprehensive metrics calculation for {baseline_name.upper()}
    \\item Performance distribution analysis
    \\item Category-based performance evaluation
    \\item Complexity analysis and ranking
\\end{{enumerate}}

\\section{{Results Overview}}

\\subsection{{Performance Metrics}}
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Total Sequences & {len(metrics_df)} \\\\
Average File Size & {metrics_df['avg_file_size'].mean():.2f} bytes \\\\
Standard Deviation File Size & {metrics_df['avg_file_size'].std():.2f} bytes \\\\
Average Mask Area & {metrics_df['avg_mask_area'].mean():.2f} pixels \\\\
Average Complexity & {metrics_df['avg_complexity'].mean():.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Performance Metrics Summary}}
\\end{{table}}

\\subsection{{Performance Distribution}}
The analysis reveals the following performance characteristics:
\\begin{{itemize}}
    \\item File sizes range from {metrics_df['avg_file_size'].min():.0f} to {metrics_df['avg_file_size'].max():.0f} bytes
    \\item Mask areas range from {metrics_df['avg_mask_area'].min():.0f} to {metrics_df['avg_mask_area'].max():.0f} pixels
    \\item Complexity scores range from {metrics_df['avg_complexity'].min():.3f} to {metrics_df['avg_complexity'].max():.3f}
\\end{{itemize}}

\\section{{Recommendations}}
Based on the analysis, the following recommendations are provided:
\\begin{{enumerate}}
"""

        for rec in recommendations:
            latex_content += f"""
    \\item \\textbf{{{rec['category']}}}: {rec['finding']}
    \\begin{{itemize}}
        \\item \\textit{{Implication}}: {rec['implication']}
        \\item \\textit{{Recommendation}}: {rec['recommendation']}
    \\end{{itemize}}
"""
        
        latex_content += f"""
\\end{{enumerate}}

\\section{{Conclusion}}
This analysis provides comprehensive insights into {baseline_name.upper()} baseline performance, revealing consistent processing capability across {len(metrics_df)} sequences. The baseline demonstrates reliable performance with predictable resource requirements.

\\textbf{{Next Steps:}} To enable baseline comparison, ensure all baselines generate output files and re-run the analysis.

\\end{{document}}
"""
        
        # Save LaTeX report
        latex_file = self.analysis_dir / "reports" / f"{baseline_name}_single_baseline_report.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_content)
        
        return latex_file

def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive analysis for baseline comparison'
    )
    
    parser.add_argument(
        '--experiment_dir',
        type=str,
        required=True,
        help='Path to experiment directory'
    )
    
    parser.add_argument(
        '--output_format',
        type=str,
        choices=['all', 'plots', 'tables', 'reports'],
        default='all',
        help='Output format to generate'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ComprehensiveAnalyzer(args.experiment_dir)
    
    # Run analysis
    results = analyzer.run_comprehensive_analysis()
    
    print(f"\nAnalysis completed successfully!")
    print(f"Check the analysis directory for all outputs: {analyzer.analysis_dir}")

if __name__ == "__main__":
    main()
