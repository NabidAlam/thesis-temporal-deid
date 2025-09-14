#!/usr/bin/env python3
"""
Clean up test output directory to test the fixed comparison script
"""

import shutil
import os
from pathlib import Path

def cleanup_test_output():
    """Remove the test output directory"""
    
    test_output_dir = Path("test_output")
    
    if test_output_dir.exists():
        print(f"Removing existing test output directory: {test_output_dir}")
        shutil.rmtree(test_output_dir)
        print("Test output directory removed")
    else:
        print("No existing test output directory found")
    
    print("\nYou can now run the test script again to verify the fix:")
    print("python test_comparison_script.py")

if __name__ == "__main__":
    cleanup_test_output()
