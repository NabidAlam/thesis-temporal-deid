#!/usr/bin/env python3
"""
Simple test script to demonstrate configuration loading
"""

import yaml
from pathlib import Path

def test_config_loading():
    """Test loading all configuration files."""
    
    configs_dir = Path("configs/datasets")
    print("Testing Configuration File Loading")
    print("=" * 50)
    
    for config_file in configs_dir.glob("*.yaml"):
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            dataset_info = config['dataset']
            print(f"{config_file.name}")
            print(f"   Name: {dataset_info['name']}")
            print(f"   Type: {dataset_info['type']}")
            print(f"   Description: {dataset_info['description']}")
            
            # Show key settings
            if 'tsp_sam' in dataset_info:
                tsp_config = dataset_info['tsp_sam']
                print(f"   TSP-SAM: {tsp_config.get('input_sizes', 'N/A')} input sizes")
                print(f"   Temporal: {tsp_config.get('temporal_processing', 'N/A')}")
            
            if 'samurai' in dataset_info:
                samurai_config = dataset_info['samurai']
                print(f"   SAMURAI: {samurai_config.get('confidence_threshold', 'N/A')} confidence")
                print(f"   Max Persons: {samurai_config.get('max_persons', 'N/A')}")
            
            print()
            
        except Exception as e:
            print(f"Error reading {config_file}: {e}")
            print()

if __name__ == "__main__":
    test_config_loading()
