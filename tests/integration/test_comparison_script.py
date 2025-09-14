#!/usr/bin/env python3
"""
Test script for the comprehensive baseline comparison
Demonstrates how to run the comparison with a small subset of sequences
"""

import subprocess
import sys
import os
from pathlib import Path

def test_comparison_script():
    """Test the comparison script with a small subset"""
    
    print("Testing Comprehensive Baseline Comparison Script")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        "comprehensive_baseline_comparison.py",
        "run_tsp_sam_baseline.py", 
        "run_samurai_baseline.py",
        "run_hybrid_on_davis.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"ERROR: Missing required files: {missing_files}")
        return False
    
    print("✓ All required files found")
    
    # Check if DAVIS dataset exists
    davis_path = "input/davis2017"
    if not os.path.exists(davis_path):
        print(f"WARNING: DAVIS dataset not found at {davis_path}")
        print("Please ensure the DAVIS 2017 dataset is available")
        return False
    
    print("✓ DAVIS dataset found")
    
    # Run comparison with a small subset (2 sequences, 5 frames each)
    print("\nRunning comparison test with small subset...")
    print("This will test TSP-SAM, SAMURAI, and Hybrid approaches")
    
    cmd = [
        "python", "comprehensive_baseline_comparison.py",
        "--input_path", davis_path,
        "--output_path", "test_output",
        "--max_frames", "5",
        "--sequences", "blackswan", "bmx-trees",  # Test with 2 sequences
        "--experiment_name", "test_comparison"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the comparison
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            print("✓ Comparison test completed successfully!")
            print("\nOutput:")
            print(result.stdout)
            
            # Check if output files were created
            test_output_dir = Path("test_output/experiments/test_comparison")
            expected_files = [
                "results/comparison_results.json",
                "reports/comparison_summary.csv", 
                "reports/thesis_summary_report.md",
                "visualizations/baseline_comparison.png"
            ]
            
            print(f"\nChecking output files in: {test_output_dir}")
            for file_path in expected_files:
                full_path = test_output_dir / file_path
                if full_path.exists():
                    print(f"✓ {file_path}")
                else:
                    print(f"✗ {file_path} (missing)")
            
            return True
            
        else:
            print("✗ Comparison test failed!")
            print(f"Return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Comparison test timed out (30 minutes)")
        return False
    except Exception as e:
        print(f"✗ Error running comparison test: {e}")
        return False

def show_usage_examples():
    """Show usage examples for the comparison script"""
    
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    examples = [
        {
            "description": "Quick test with 2 sequences, 5 frames each",
            "command": "python comprehensive_baseline_comparison.py --sequences blackswan bmx-trees --max_frames 5"
        },
        {
            "description": "Full comparison with all sequences, 10 frames each",
            "command": "python comprehensive_baseline_comparison.py --max_frames 10"
        },
        {
            "description": "Comparison with WANDB logging enabled",
            "command": "python comprehensive_baseline_comparison.py --max_frames 10 --wandb"
        },
        {
            "description": "Custom experiment name and output directory",
            "command": "python comprehensive_baseline_comparison.py --output_path my_results --experiment_name my_experiment --max_frames 5"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}:")
        print(f"   {example['command']}")

def main():
    """Main test function"""
    
    print("Comprehensive Baseline Comparison - Test Script")
    print("=" * 60)
    
    # Show usage examples first
    show_usage_examples()
    
    # Ask user if they want to run the test
    print(f"\n{'='*60}")
    response = input("Do you want to run a quick test? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        success = test_comparison_script()
        
        if success:
            print(f"\n{'='*60}")
            print("TEST COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("The comparison script is working correctly.")
            print("You can now run it with your desired parameters.")
            print("\nFor your thesis expose, consider running:")
            print("python comprehensive_baseline_comparison.py --max_frames 10 --wandb")
        else:
            print(f"\n{'='*60}")
            print("TEST FAILED!")
            print("=" * 60)
            print("Please check the error messages above and ensure:")
            print("1. All required files are present")
            print("2. DAVIS dataset is available")
            print("3. Required dependencies are installed")
    else:
        print("\nTest skipped. You can run the comparison script directly when ready.")
    
    print(f"\n{'='*60}")
    print("For more information, see the generated thesis_summary_report.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
