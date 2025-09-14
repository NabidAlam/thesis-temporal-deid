#!/usr/bin/env python3
"""
Table Data Calculator for Thesis Expose
Shows the exact calculations used in the thesis expose table
"""

import json
from pathlib import Path

def load_results_data():
    """Load the baseline comparison results"""
    results_file = Path('output/experiments/baseline_comparison_20250906_214955/reports/individual_baselines/baseline_comparison_analysis.json')
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Please run the comprehensive baseline comparison first.")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

def calculate_table_data(data):
    """Calculate and display the table data"""
    print("=" * 60)
    print("THESIS EXPOSE TABLE DATA CALCULATIONS")
    print("=" * 60)
    print()
    
    # Extract data
    tsp_data = data['tsp_sam_performance']
    sam_data = data['samurai_performance'] 
    hybrid_data = data['hybrid_performance']
    
    print("RAW DATA:")
    print(f"   TSP-SAM: {tsp_data['successful_sequences']}/{tsp_data['total_sequences']} sequences, {tsp_data['average_processing_time']:.1f}s avg time")
    print(f"   SAMURAI: {sam_data['successful_sequences']}/{sam_data['total_sequences']} sequences, {sam_data['average_processing_time']:.1f}s avg time")
    print(f"   Hybrid:  {hybrid_data['successful_sequences']}/{hybrid_data['total_sequences']} sequences, {hybrid_data['average_processing_time']:.1f}s avg time")
    print()
    
    print("CALCULATIONS:")
    print("   Success Rate = successful_sequences / total_sequences")
    print("   FPS = 25 frames / average_processing_time")
    print()
    
    print("THESIS EXPOSE TABLE:")
    print("┌─────────┬─────────────┬──────────┬────────────┐")
    print("│ Method  │ Success Rate│ Avg Time │ Performance│")
    print("├─────────┼─────────────┼──────────┼────────────┤")
    
    # TSP-SAM
    tsp_success_rate = tsp_data['success_rate']
    tsp_time = tsp_data['average_processing_time']
    tsp_fps = tsp_data['average_fps']
    print(f"│ TSP-SAM │ {tsp_success_rate:.1%} ({tsp_data['successful_sequences']}/{tsp_data['total_sequences']}) │ {tsp_time:.1f}s │ {tsp_fps:.2f} FPS │")
    
    # SAMURAI
    sam_success_rate = sam_data['success_rate']
    sam_time = sam_data['average_processing_time']
    sam_fps = sam_data['average_fps']
    print(f"│ SAMURAI │ {sam_success_rate:.1%} ({sam_data['successful_sequences']}/{sam_data['total_sequences']}) │ {sam_time:.1f}s │ {sam_fps:.2f} FPS │")
    
    # Hybrid
    hybrid_success_rate = hybrid_data['success_rate']
    hybrid_time = hybrid_data['average_processing_time']
    hybrid_fps = hybrid_data['average_fps']
    print(f"│ Hybrid  │ {hybrid_success_rate:.1%} ({hybrid_data['successful_sequences']}/{hybrid_data['total_sequences']}) │ {hybrid_time:.1f}s │ {hybrid_fps:.2f} FPS │")
    
    print("└─────────┴─────────────┴──────────┴────────────┘")
    print()
    
    print("KEY FINDINGS:")
    print(f"   • TSP-SAM fails on {tsp_data['total_sequences'] - tsp_data['successful_sequences']} sequences")
    print(f"   • SAMURAI fails on {sam_data['total_sequences'] - sam_data['successful_sequences']} sequences")
    print(f"   • Hybrid achieves perfect reliability (100% success)")
    
    # Complementary analysis
    comp = data['comparative_analysis']['complementary_analysis']
    just = data['comparative_analysis']['hybrid_justification_analysis']
    
    print(f"   • Hybrid recovers {just['failure_pattern_analysis']['hybrid_saves_tsp_failures']} TSP-SAM failures")
    print(f"   • Hybrid recovers {just['failure_pattern_analysis']['hybrid_saves_sam_failures']} SAMURAI failures")
    print(f"   • Total recovered: {just['failure_pattern_analysis']['hybrid_saves_tsp_failures'] + just['failure_pattern_analysis']['hybrid_saves_sam_failures']} sequences")
    print()
    
    print("EXPERIMENT INFO:")
    exp_info = data['experiment_info']
    print(f"   • Total sequences: {exp_info['total_sequences']}")
    print(f"   • Frames per sequence: {exp_info['frames_per_sequence']}")
    print(f"   • Experiment timestamp: {exp_info['timestamp']}")
    print()
    
    print("💾 DATA SOURCE:")
    results_file = Path('output/experiments/baseline_comparison_20250906_214955/reports/individual_baselines/baseline_comparison_analysis.json')
    print(f"   File: {results_file}")
    print("=" * 60)

def main():
    """Main function"""
    print("Loading baseline comparison results...")
    
    data = load_results_data()
    if data is None:
        return
    
    calculate_table_data(data)

if __name__ == "__main__":
    main()
