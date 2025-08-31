#!/usr/bin/env python3
"""
Script to run the Hybrid Temporal Pipeline with different configuration files
"""

import argparse
import yaml
import os
import sys
from pathlib import Path

# Add the current directory to path for imports
sys.path.append('.')

from integrated_temporal_pipeline_hybrid import HybridTemporalPipeline

def load_config(config_path):
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration: {config['dataset']['name']}")
        return config
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        return None

def list_available_configs():
    """List all available configuration files."""
    configs_dir = Path("configs/datasets")
    if not configs_dir.exists():
        print("No configs directory found")
        return
    
    print("\nAvailable Configuration Files:")
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
                print()
        except Exception as e:
            print(f"Error reading {config_file}: {e}")
            print()

def run_pipeline_with_config(input_video, output_dir, config_path, **kwargs):
    """Run the pipeline with a specific configuration."""
    
    # Load configuration
    config = load_config(config_path)
    if config is None:
        return False
    
    # Extract dataset name for logging
    dataset_name = config['dataset']['name']
    print(f"Starting pipeline with {dataset_name} configuration")
    
    # Initialize pipeline with config
    try:
        pipeline = HybridTemporalPipeline(
            input_video=input_video,
            output_dir=output_dir,
            dataset_config=config,
            **kwargs
        )
        
        print(f"Pipeline initialized successfully with {dataset_name} config")
        
        # Process video
        if kwargs.get('enable_chunked_processing'):
            print("Using chunked processing mode")
            pipeline.process_video_chunked()
        else:
            print("Using standard processing mode")
            pipeline.process_video()
        
        print(f"Pipeline completed successfully with {dataset_name} configuration")
        return True
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        if kwargs.get('debug_mode'):
            import traceback
            traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run Hybrid Temporal Pipeline with configuration files"
    )
    
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("output_dir", help="Path to output directory")
    parser.add_argument("--config", "-c", 
                       default="configs/datasets/default.yaml",
                       help="Path to configuration file (default: default.yaml)")
    
    # Pipeline options
    parser.add_argument("--chunked", action="store_true", 
                       help="Enable chunked processing for long videos")
    parser.add_argument("--chunk-size", type=int, default=100,
                       help="Number of frames per chunk (default: 100)")
    parser.add_argument("--max-frames", type=int,
                       help="Maximum number of frames to process")
    parser.add_argument("--start-time", type=float,
                       help="Start time in seconds")
    parser.add_argument("--end-time", type=float,
                       help="End time in seconds")
    
    # Processing options
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--list-configs", action="store_true",
                       help="List all available configuration files")
    
    args = parser.parse_args()
    
    # List available configs if requested
    if args.list_configs:
        list_available_configs()
        return
    
    # Check if input video exists
    if not os.path.exists(args.input_video):
        print(f"Input video not found: {args.input_video}")
        return
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Use --list-configs to see available configurations")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare pipeline arguments
    pipeline_kwargs = {
        'debug_mode': args.debug,
        'use_wandb': args.wandb,
        'enable_chunked_processing': args.chunked,
        'chunk_size': args.chunk_size,
        'start_time': args.start_time,
        'end_time': args.end_time
    }
    
    # Run pipeline
    success = run_pipeline_with_config(
        args.input_video,
        args.output_dir,
        args.config,
        **pipeline_kwargs
    )
    
    if success:
        print("\nPipeline completed successfully!")
        print(f"Output saved to: {args.output_dir}")
    else:
        print("\nPipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
