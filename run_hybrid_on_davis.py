#!/usr/bin/env python3
"""
Run Hybrid Pipeline on DAVIS Dataset Sequences
Processes individual frame sequences from DAVIS dataset using the hybrid pipeline
"""

import os
import sys
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import time
from datetime import datetime

# Add the hybrid pipeline to path
sys.path.append('.')

try:
    from integrated_temporal_pipeline_hybrid import HybridTemporalPipeline
    print("Successfully imported HybridTemporalPipeline")
except ImportError as e:
    print(f"Error importing HybridTemporalPipeline: {e}")
    sys.exit(1)

class DavisSequenceProcessor:
    """Process DAVIS dataset sequences using the hybrid pipeline"""
    
    def __init__(self, davis_root: str, output_root: str, sequence_name: str = None):
        self.davis_root = Path(davis_root)
        self.output_root = Path(output_root)
        self.sequence_name = sequence_name
        
        # DAVIS dataset structure (480p resolution)
        self.images_dir = self.davis_root / "JPEGImages" / "480p"
        self.annotations_dir = self.davis_root / "Annotations" / "480p"
        
        # Validate DAVIS structure
        if not self.images_dir.exists():
            raise ValueError(f"DAVIS images directory not found: {self.images_dir}")
        
        # Get available sequences
        self.available_sequences = self._get_available_sequences()
        print(f"Found {len(self.available_sequences)} DAVIS sequences")
        
        # Create output directory
        self.output_root.mkdir(parents=True, exist_ok=True)
        
    def _get_available_sequences(self) -> List[str]:
        """Get list of available DAVIS sequences"""
        sequences = []
        if self.images_dir.exists():
            for item in self.images_dir.iterdir():
                if item.is_dir():
                    sequences.append(item.name)
        return sorted(sequences)
    
    def _get_sequence_frames(self, sequence_name: str, max_frames: int = None) -> List[Path]:
        """Get ordered list of frame paths for a sequence"""
        sequence_dir = self.images_dir / sequence_name
        if not sequence_dir.exists():
            raise ValueError(f"Sequence directory not found: {sequence_dir}")
        
        # Get all image files and sort them
        image_extensions = ['.jpg', '.jpeg', '.png']
        frames = []
        for ext in image_extensions:
            frames.extend(sequence_dir.glob(f"*{ext}"))
        
        # Sort frames by name (DAVIS uses sequential naming)
        frames.sort(key=lambda x: x.name)
        
        if not frames:
            raise ValueError(f"No frames found in sequence: {sequence_name}")
        
        # Limit frames if max_frames is specified
        if max_frames is not None and max_frames > 0:
            frames = frames[:max_frames]
            print(f"Limited to {len(frames)} frames (max_frames={max_frames}) in sequence: {sequence_name}")
        else:
            print(f"Found {len(frames)} frames in sequence: {sequence_name}")
        
        return frames
    
    def _create_video_from_frames(self, frames: List[Path], output_path: Path, fps: int = 30) -> Path:
        """Create a temporary video file from frame sequence"""
        if not frames:
            raise ValueError("No frames provided")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frames[0]))
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {frames[0]}")
        
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = output_path / f"temp_{self.sequence_name}.mp4"
        
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        # Write frames
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                out.write(frame)
        
        out.release()
        print(f"Created temporary video: {video_path}")
        return video_path
    
    def _process_sequence(self, sequence_name: str, use_wandb: bool = False, experiment_name: str = None, max_frames: int = None) -> Dict:
        """Process a single DAVIS sequence using the hybrid pipeline"""
        print(f"\nProcessing sequence: {sequence_name}")
        print("=" * 60)
        
        # Get sequence frames
        frames = self._get_sequence_frames(sequence_name, max_frames)
        
        # Create sequence output directory
        sequence_output = self.output_root / sequence_name
        sequence_output.mkdir(parents=True, exist_ok=True)
        
        # Create temporary video from frames
        temp_video_path = self._create_video_from_frames(frames, sequence_output)
        
        try:
            # Initialize hybrid pipeline
            print(f"Initializing hybrid pipeline for {sequence_name}...")
            
            # Create experiment name if not provided
            if not experiment_name:
                experiment_name = f"hybrid_davis_{sequence_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize pipeline
            pipeline = HybridTemporalPipeline(
                input_video=str(temp_video_path),
                output_dir=str(sequence_output),
                debug_mode=True,
                dataset_config=None,  # Use default config for DAVIS
                use_wandb=use_wandb,
                experiment_name=experiment_name,
                enable_chunked_processing=False,  # DAVIS sequences are short
                chunk_size=50,
                deidentification_strategy='blurring'
            )
            
            print(f"Pipeline initialized successfully")
            
            # Process the sequence
            print(f"Processing {len(frames)} frames...")
            start_time = time.time()
            
            # Actually run the hybrid pipeline processing
            print(f"Running hybrid pipeline on {len(frames)} frames...")
            pipeline.process_video()
            
            # Wait a moment for video creation to complete
            time.sleep(2)
            
            processing_time = time.time() - start_time
            print(f"Processing completed in {processing_time:.2f} seconds")
            
            # Collect results
            results = {
                'sequence_name': sequence_name,
                'total_frames': len(frames),
                'processing_time': processing_time,
                'output_directory': str(sequence_output),
                'temp_video_path': str(temp_video_path),
                'pipeline_status': 'completed'
            }
            
            # Save results
            results_file = sequence_output / 'processing_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            print(f"Error processing sequence {sequence_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error results
            error_results = {
                'sequence_name': sequence_name,
                'total_frames': len(frames),
                'error': str(e),
                'pipeline_status': 'failed'
            }
            
            error_file = sequence_output / 'error_results.json'
            with open(error_file, 'w') as f:
                json.dump(error_results, f, indent=2)
            
            return error_results
        
        finally:
            # Clean up temporary video
            if temp_video_path.exists():
                temp_video_path.unlink()
                print(f"Cleaned up temporary video: {temp_video_path}")
    
    def run_all_sequences(self, use_wandb: bool = False, max_frames: int = None) -> List[Dict]:
        """Run hybrid pipeline on all available DAVIS sequences"""
        results = []
        
        if self.sequence_name:
            # Process specific sequence
            print(f"Processing specific sequence: {self.sequence_name}")
            if self.sequence_name in self.available_sequences:
                result = self._process_sequence(self.sequence_name, use_wandb, max_frames=max_frames)
                results.append(result)
            else:
                print(f"Sequence '{self.sequence_name}' not found in available sequences")
                print(f"Available sequences: {self.available_sequences}")
        else:
            # Process all sequences
            print(f"Processing all {len(self.available_sequences)} DAVIS sequences...")
            for i, sequence in enumerate(self.available_sequences, 1):
                print(f"\nProgress: {i}/{len(self.available_sequences)}")
                result = self._process_sequence(sequence, use_wandb, max_frames=max_frames)
                results.append(result)
                
                # Small delay between sequences
                time.sleep(1)
        
        return results
    
    def generate_summary_report(self, results: List[Dict]) -> None:
        """Generate a summary report of all processing results"""
        summary_file = self.output_root / 'davis_processing_summary.json'
        
        # Calculate statistics
        total_sequences = len(results)
        successful_sequences = len([r for r in results if r.get('pipeline_status') == 'completed'])
        failed_sequences = total_sequences - successful_sequences
        
        # Calculate average processing time
        processing_times = [r.get('processing_time', 0) for r in results if r.get('pipeline_status') == 'completed']
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        # Total frames processed
        total_frames = sum([r.get('total_frames', 0) for r in results])
        
        summary = {
            'summary': {
                'total_sequences': total_sequences,
                'successful_sequences': successful_sequences,
                'failed_sequences': failed_sequences,
                'success_rate': f"{(successful_sequences/total_sequences)*100:.1f}%" if total_sequences > 0 else "0%",
                'total_frames_processed': total_frames,
                'average_processing_time_per_sequence': f"{avg_processing_time:.2f}s",
                'total_processing_time': f"{sum(processing_times):.2f}s"
            },
            'sequence_results': results,
            'timestamp': datetime.now().isoformat(),
            'davis_root': str(self.davis_root),
            'output_root': str(self.output_root)
        }
        
        # Save summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSUMMARY REPORT")
        print("=" * 60)
        print(f"Total Sequences: {total_sequences}")
        print(f"Successful: {successful_sequences}")
        print(f"Failed: {failed_sequences}")
        print(f"Success Rate: {summary['summary']['success_rate']}")
        print(f"Total Frames: {total_frames}")
        print(f"Avg Time/Sequence: {summary['summary']['average_processing_time_per_sequence']}")
        print(f"Total Time: {summary['summary']['total_processing_time']}")
        print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Run Hybrid Pipeline on DAVIS Dataset")
    parser.add_argument("--davis-root", type=str, default="input/davis2017",
                       help="Path to DAVIS dataset root directory")
    parser.add_argument("--output-root", type=str, default="output/davis_hybrid_results",
                       help="Path to output directory")
    parser.add_argument("--sequence", type=str, default=None,
                       help="Process specific sequence (optional, processes all if not specified)")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--list-sequences", action="store_true",
                       help="List available sequences and exit")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum number of frames to process per sequence")
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = DavisSequenceProcessor(
            davis_root=args.davis_root,
            output_root=args.output_root,
            sequence_name=args.sequence
        )
        
        # List sequences if requested
        if args.list_sequences:
            print("Available DAVIS sequences:")
            for i, seq in enumerate(processor.available_sequences, 1):
                print(f"  {i:2d}. {seq}")
            return
        
        # Run processing
        results = processor.run_all_sequences(use_wandb=args.use_wandb, max_frames=args.max_frames)
        
        # Generate summary
        processor.generate_summary_report(results)
        
        print(f"\nDAVIS processing completed!")
        print(f"Results saved to: {args.output_root}")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
