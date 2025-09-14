import cv2
import numpy as np
import os
from integrated_temporal_pipeline_hybrid import HybridTemporalPipeline

def test_temporal_capabilities():
    """Test temporal capabilities for professor demonstration."""
    print("="*80)
    print("TESTING TEMPORAL CAPABILITIES")
    print("="*80)
    
    # Initialize pipeline
    pipeline = HybridTemporalPipeline('input/ted/video5.mp4', 'output/temporal_demo')
    
    # Read test frames
    cap = cv2.VideoCapture('input/ted/video5.mp4')
    test_frames = []
    for i in range(5):  # Test with 5 frames
        ret, frame = cap.read()
        if ret:
            test_frames.append(frame)
            print(f"Loaded frame {i}: {frame.shape}")
        else:
            break
    cap.release()
    
    print(f"Loaded {len(test_frames)} test frames")
    
    # Demonstrate temporal capabilities
    try:
        results = pipeline.demonstrate_temporal_capabilities(test_frames)
        
        # Save temporal processing results
        print("\n" + "="*60)
        print("SAVING TEMPORAL PROCESSING RESULTS")
        print("="*60)
        
        # Create output directory
        output_dir = 'output/temporal_demo'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original frames
        for i, frame in enumerate(test_frames):
            cv2.imwrite(f'{output_dir}/frame_{i:02d}_original.jpg', frame)
            print(f"Saved original frame {i}: frame_{i:02d}_original.jpg")
        
        # Save temporal analysis results
        if 'temporal_analysis' in results:
            temporal_data = results['temporal_analysis']
            
            # Save temporal consistency plot
            if 'consistency_scores' in temporal_data:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(range(len(temporal_data['consistency_scores'])), 
                        temporal_data['consistency_scores'], 'bo-')
                plt.xlabel('Frame Index')
                plt.ylabel('Temporal Consistency Score')
                plt.title('Temporal Consistency Across Frames')
                plt.grid(True)
                plt.savefig(f'{output_dir}/temporal_consistency_plot.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved temporal consistency plot: temporal_consistency_plot.png")
            
            # Save temporal memory usage
            if 'memory_usage' in temporal_data:
                with open(f'{output_dir}/temporal_memory_usage.txt', 'w') as f:
                    f.write("TEMPORAL MEMORY USAGE ANALYSIS\n")
                    f.write("="*50 + "\n")
                    f.write(f"Total frames processed: {len(test_frames)}\n")
                    f.write(f"Frames using temporal memory: {temporal_data.get('frames_with_memory', 0)}\n")
                    f.write(f"Temporal memory depth: {temporal_data.get('memory_depth', 0)} frames\n")
                    f.write(f"Average consistency: {temporal_data.get('avg_consistency', 0):.3f}\n")
                print(f"Saved temporal memory analysis: temporal_memory_usage.txt")
        
        # Save summary results
        with open(f'{output_dir}/demonstration_summary.txt', 'w') as f:
            f.write("TEMPORAL CAPABILITIES DEMONSTRATION SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Temporal Consistency Score: {results.get('temporal_consistency', 'N/A')}\n")
            f.write(f"Processing Time: {results.get('processing_time', 'N/A')}\n")
            f.write(f"Mask Quality: {results.get('mask_quality', 'N/A')}\n")
            f.write(f"Temporal Memory Usage: {results.get('temporal_memory_usage', 'N/A')}\n")
            f.write(f"Total frames processed: {len(test_frames)}\n")
            f.write(f"Output saved to: {output_dir}\n")
        
        print(f"Saved demonstration summary: demonstration_summary.txt")
        print(f"All results saved to: {output_dir}")
        
        print("\n" + "="*60)
        print("TEMPORAL CAPABILITIES DEMONSTRATION RESULTS")
        print("="*60)
        print(f"Temporal Consistency Score: {results.get('temporal_consistency', 'N/A')}")
        print(f"Processing Time: {results.get('processing_time', 'N/A')}")
        print(f"Mask Quality: {results.get('mask_quality', 'N/A')}")
        print(f"Temporal Memory Usage: {results.get('temporal_memory_usage', 'N/A')}")
        return True
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_temporal_capabilities()
    if success:
        print("\nTemporal capabilities demonstration completed successfully!")
        print("Check the output/temporal_demo folder for saved results!")
    else:
        print("\nTemporal capabilities demonstration failed!")
