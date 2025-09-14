#!/usr/bin/env python3
"""
FAST TEST script for chunked video processing to prevent memory issues with long videos.
This demonstrates how to use the new chunked processing mode with reduced testing time.
"""

import os
import sys
import time

def test_chunked_processing_fast():
    """Test the chunked processing functionality with reduced scope for faster testing."""
    print("="*60)
    print("FAST TESTING CHUNKED VIDEO PROCESSING")
    print("="*60)
    
    # Test video path (use video5.mp4 as example)
    test_video = "input/ted/video5.mp4"
    output_dir = "output/video5_chunked_fast_test"
    
    if not os.path.exists(test_video):
        print(f"Test video not found: {test_video}")
        print("Please ensure video5.mp4 exists in input/ted/")
        return False
    
    print(f"Test video: {test_video}")
    print(f"Output directory: {output_dir}")
    
    # Test with smaller chunk sizes and fewer frames for faster testing
    chunk_sizes = [25, 50]  # Reduced from [50, 100, 200]
    
    for chunk_size in chunk_sizes:
        print(f"\n{'='*40}")
        print(f"Testing with chunk size: {chunk_size}")
        print(f"{'='*40}")
        
        # Create unique output directory for this test
        test_output_dir = f"{output_dir}_chunk{chunk_size}"
        
        # Build command with reduced frame count for faster testing
        cmd = f'python integrated_temporal_pipeline_hybrid.py "{test_video}" "{test_output_dir}" --max-frames 100 --chunked --chunk-size {chunk_size} --debug'
        
        print(f"Command: {cmd}")
        print(f"Processing 100 frames in chunks of {chunk_size}...")
        
        # Run the command
        start_time = time.time()
        result = os.system(cmd)
        end_time = time.time()
        
        if result == 0:
            print(f"Chunked processing with chunk size {chunk_size} completed successfully!")
            print(f"Total time: {end_time - start_time:.2f} seconds")
            
            # Check output files
            if os.path.exists(test_output_dir):
                files = os.listdir(test_output_dir)
                frame_files = [f for f in files if f.startswith("frame_") and f.endswith("_deidentified.png")]
                print(f"Output directory: {test_output_dir}")
                print(f"Generated frames: {len(frame_files)}")
                
                # Check for performance report
                perf_report = os.path.join(test_output_dir, "performance_report.json")
                if os.path.exists(perf_report):
                    print(f"Performance report: {perf_report}")
                    
                    # Read and display key metrics
                    try:
                        import json
                        with open(perf_report, 'r') as f:
                            data = json.load(f)
                        
                        print(f"Processing stats:")
                        print(f"   - Total frames: {data['processing_stats']['total_frames']}")
                        print(f"   - Success rate: {data['processing_stats']['successful_frames']}/{data['processing_stats']['total_frames']}")
                        print(f"   - Total time: {data['processing_stats']['total_processing_time']:.2f}s")
                        print(f"   - Avg time per frame: {data['processing_stats']['average_time_per_frame']:.2f}s")
                        
                        if 'memory_usage' in data:
                            print(f"Memory usage:")
                            print(f"   - Peak memory: {data['memory_usage']['peak_memory_mb']:.1f} MB")
                            print(f"   - Memory warnings: {len(data['memory_usage']['memory_warnings'])}")
                            print(f"   - Chunked processing: {data['pipeline_info']['chunked_processing']}")
                            print(f"   - Chunk size: {data['pipeline_info']['chunk_size']}")
                    
                    except Exception as e:
                        print(f"Could not read performance report: {e}")
            else:
                print(f"Output directory not created: {test_output_dir}")
        else:
            print(f"Chunked processing with chunk size {chunk_size} failed!")
            print(f"Exit code: {result}")
        
        print(f"\n{'='*40}")
    
    print(f"\n{'='*60}")
    print("FAST CHUNKED PROCESSING TEST COMPLETED")
    print(f"{'='*60}")
    
    return True

def compare_processing_modes_fast():
    """Compare standard vs chunked processing modes with reduced scope."""
    print("\n" + "="*60)
    print("FAST COMPARING PROCESSING MODES")
    print("="*60)
    
    test_video = "input/ted/video5.mp4"
    
    if not os.path.exists(test_video):
        print(f"Test video not found: {test_video}")
        return False
    
    # Test standard processing (25 frames) - reduced from 50
    print("\n1. Testing STANDARD processing (25 frames):")
    print("   This mode loads all frames into memory")
    
    cmd_standard = f'python integrated_temporal_pipeline_hybrid.py "{test_video}" "output/video5_standard_fast" --max-frames 25 --debug'
    print(f"Command: {cmd_standard}")
    
    start_time = time.time()
    result_standard = os.system(cmd_standard)
    end_time = time.time()
    
    if result_standard == 0:
        print(f"Standard processing completed in {end_time - start_time:.2f}s")
    else:
        print(f"Standard processing failed")
    
    # Test chunked processing (25 frames in chunks of 10) - reduced scope
    print("\n2. Testing CHUNKED processing (25 frames in chunks of 10):")
    print("   This mode processes frames in small chunks to save memory")
    
    cmd_chunked = f'python integrated_temporal_pipeline_hybrid.py "{test_video}" "output/video5_chunked_fast" --max-frames 25 --chunked --chunk-size 10 --debug'
    print(f"Command: {cmd_chunked}")
    
    start_time = time.time()
    result_chunked = os.system(cmd_chunked)
    end_time = time.time()
    
    if result_chunked == 0:
        print(f"Chunked processing completed in {end_time - start_time:.2f}s")
    else:
        print(f"Chunked processing failed")
    
    print("\n" + "="*60)
    print("FAST COMPARISON COMPLETED")
    print("="*60)
    
    return True

if __name__ == "__main__":
    print("FAST Chunked Processing Test Script")
    print("This script tests the new chunked processing functionality")
    print("with reduced scope for faster validation.")
    print("\nEstimated total time: 2-5 minutes (vs 10-15 minutes for full test)")
    
    # Test chunked processing (fast version)
    test_chunked_processing_fast()
    
    # Compare processing modes (fast version)
    compare_processing_modes_fast()
    
    print("\nFast test script completed!")
    print("\nSummary of what was tested:")
    print("   - Chunked processing with chunk sizes: 25, 50")
    print("   - Standard vs chunked processing comparison")
    print("   - Memory management validation")
    print("   - Performance metrics analysis")
    print("\nTo test with your own video:")
    print("python integrated_temporal_pipeline_hybrid.py <input_video> <output_dir> --chunked --chunk-size 100")
    print("\nFor very long videos (>5 minutes), use:")
    print("python integrated_temporal_pipeline_hybrid.py <input_video> <output_dir> --chunked --chunk-size 50")
