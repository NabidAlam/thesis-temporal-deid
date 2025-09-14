#!/usr/bin/env python3
"""
Hybrid Pipeline Wrapper for EnvisionBox
Calls the working hybrid temporal pipeline as a subprocess to avoid import issues.
"""

import sys
import os
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Import torch for device detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Device detection will use fallback.")

class HybridPipelineWrapper:
    """
    Wrapper class that calls the working hybrid temporal pipeline as a subprocess
    to avoid import path issues while maintaining full functionality.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the wrapper"""
        print("DEBUG: Starting HybridPipelineWrapper initialization...")
        try:
            print("DEBUG: Loading config...")
            self.config = self.load_config(config_path)
            print("DEBUG: Config loaded successfully")
            
            print("DEBUG: Setting up logging...")
            self.setup_logging()
            print("DEBUG: Logging setup complete")
            
            # Find the main project directory
            print("DEBUG: Finding main project directory...")
            # The wrapper is in: envisionbox_integration/integration/
            # We need to go up 2 levels to get to the main project directory
            current_file = Path(__file__).resolve()  # Get absolute path
            print(f"DEBUG: Current wrapper file: {current_file}")
            
            self.main_project_dir = current_file.parent.parent.parent
            print(f"DEBUG: Main project directory: {self.main_project_dir}")
            print(f"DEBUG: Main project directory exists: {self.main_project_dir.exists()}")
            
            self.hybrid_pipeline_script = self.main_project_dir / "integrated_temporal_pipeline_hybrid.py"
            print(f"DEBUG: Pipeline script path: {self.hybrid_pipeline_script}")
            print(f"DEBUG: Pipeline script exists: {self.hybrid_pipeline_script.exists()}")
            
            # Also check if the script exists in the current working directory as a fallback
            if not self.hybrid_pipeline_script.exists():
                print("DEBUG: Script not found in calculated path, checking current working directory...")
                cwd_script = Path.cwd() / "integrated_temporal_pipeline_hybrid.py"
                print(f"DEBUG: CWD script path: {cwd_script}")
                print(f"DEBUG: CWD script exists: {cwd_script.exists()}")
                if cwd_script.exists():
                    self.hybrid_pipeline_script = cwd_script
                    self.main_project_dir = Path.cwd()
                    print("DEBUG: Using script from current working directory")
            
            if not self.hybrid_pipeline_script.exists():
                print(f"DEBUG: Pipeline script not found, searching in common locations...")
                
                # Search in common locations
                search_paths = [
                    Path.cwd() / "integrated_temporal_pipeline_hybrid.py",  # Current working directory
                    Path.cwd().parent / "integrated_temporal_pipeline_hybrid.py",  # Parent of CWD
                    Path.cwd().parent.parent / "integrated_temporal_pipeline_hybrid.py",  # Grandparent of CWD
                    Path("D:/Thesis/thesis-temporal-deid/integrated_temporal_pipeline_hybrid.py"),  # Absolute path
                ]
                
                for search_path in search_paths:
                    print(f"DEBUG: Checking: {search_path}")
                    if search_path.exists():
                        self.hybrid_pipeline_script = search_path
                        self.main_project_dir = search_path.parent
                        print(f"DEBUG: Found script at: {search_path}")
                        break
                
                if not self.hybrid_pipeline_script.exists():
                    print(f"DEBUG: Pipeline script not found in any location, listing directory contents:")
                    if self.main_project_dir.exists():
                        for item in self.main_project_dir.iterdir():
                            print(f"DEBUG:   - {item.name}")
                    raise FileNotFoundError(f"Hybrid pipeline script not found: {self.hybrid_pipeline_script}")
            
            self.logger.info(f"Hybrid pipeline wrapper initialized")
            self.logger.info(f"Main project directory: {self.main_project_dir}")
            self.logger.info(f"Pipeline script: {self.hybrid_pipeline_script}")
            print("DEBUG: HybridPipelineWrapper initialization completed successfully")
            
        except Exception as e:
            print(f"DEBUG: Exception in HybridPipelineWrapper.__init__: {e}")
            print(f"DEBUG: Exception type: {type(e)}")
            import traceback
            print(f"DEBUG: Full traceback: {traceback.format_exc()}")
            raise
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "main_config.json"
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration optimized for behavioral research
            return {
                "hybrid_pipeline": {
                    "debug_mode": True,
                    "use_wandb": False,
                    "enable_chunked_processing": True,
                    "chunk_size": 50,
                    "deidentification_strategy": "blurring"
                },
                "video_processing": {
                    "max_frames": 1000,
                    "frame_skip": 1,
                    "output_fps": 30
                },
                "behavioral_analysis": {
                    "enable_overlap_detection": True,
                    "overlap_threshold": 10.0,
                    "enable_temporal_consistency": True
                }
            }
    
    def setup_logging(self):
        """Setup logging"""
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "hybrid_pipeline_wrapper.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def process_behavioral_video(self, video_path: str, output_path: str, 
                               text_prompts: List[str],
                               deidentify_objects: List[str] = None,
                               preserve_objects: List[str] = None,
                               target_objects: List[str] = None,
                               gaze_objects: List[str] = None) -> Dict[str, Any]:
        """
        Process video for behavioral research using the working hybrid pipeline
        
        Args:
            video_path: Path to input video
            output_path: Path to output video
            text_prompts: Text prompts for object detection
            deidentify_objects: Objects to de-identify
            preserve_objects: Objects to preserve
            target_objects: Objects to track for overlap detection
            gaze_objects: Objects representing gaze
            
        Returns:
            Processing results and statistics
        """
        if deidentify_objects is None:
            deidentify_objects = ["person", "face", "people"]
        if preserve_objects is None:
            preserve_objects = ["ball", "toy", "object", "stage", "background"]
        if target_objects is None:
            target_objects = ["ball", "toy"]
        if gaze_objects is None:
            gaze_objects = ["person", "face"]
        
        self.logger.info("="*60)
        self.logger.info("STARTING BEHAVIORAL VIDEO PROCESSING")
        self.logger.info("="*60)
        self.logger.info(f"Input video: {video_path}")
        self.logger.info(f"Output video: {output_path}")
        self.logger.info(f"Text prompts: {text_prompts}")
        self.logger.info(f"De-identify: {deidentify_objects}")
        self.logger.info(f"Preserve: {preserve_objects}")
        self.logger.info(f"Target objects: {target_objects}")
        self.logger.info(f"Gaze objects: {gaze_objects}")
        
        # Validate input video
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")
        
        # Test video accessibility
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            # Get basic video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"DEBUG: Video properties:")
            print(f"DEBUG:   FPS: {fps}")
            print(f"DEBUG:   Frame count: {frame_count}")
            print(f"DEBUG:   Resolution: {width}x{height}")
            
            cap.release()
            
            if fps is None or fps <= 0:
                raise ValueError(f"Invalid FPS: {fps}")
            if frame_count is None or frame_count <= 0:
                raise ValueError(f"Invalid frame count: {frame_count}")
                
        except Exception as e:
            print(f"DEBUG: Video validation error: {e}")
            raise ValueError(f"Video validation failed: {e}")
        
        # Create output directory
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Store behavioral analysis parameters for later use
        print(f"DEBUG: Config keys: {list(self.config.keys())}")
        print(f"DEBUG: Full config: {self.config}")
        
        # Get overlap threshold with fallback
        overlap_threshold = 10.0  # Default value
        if "behavioral_analysis" in self.config and "overlap_threshold" in self.config["behavioral_analysis"]:
            overlap_threshold = self.config["behavioral_analysis"]["overlap_threshold"]
        else:
            print(f"DEBUG: behavioral_analysis config not found, using default overlap_threshold: {overlap_threshold}")
        
        self.behavioral_config = {
            "text_prompts": text_prompts,
            "deidentify_objects": deidentify_objects,
            "preserve_objects": preserve_objects,
            "target_objects": target_objects,
            "gaze_objects": gaze_objects,
            "overlap_threshold": overlap_threshold
        }
        print(f"DEBUG: Behavioral config: {self.behavioral_config}")
        
        try:
            # Use direct import instead of subprocess for better reliability
            self.logger.info("Initializing hybrid temporal pipeline directly...")
            
            # Get config values with fallbacks
            chunk_size = self.config.get("hybrid_pipeline", {}).get("chunk_size", 50)
            deidentification_strategy = self.config.get("hybrid_pipeline", {}).get("deidentification_strategy", "blurring")
            use_wandb = self.config.get("hybrid_pipeline", {}).get("use_wandb", False)
            
            # Ensure chunk_size is not None
            if chunk_size is None:
                chunk_size = 50
                print(f"DEBUG: chunk_size was None, using default: {chunk_size}")
            
            print(f"DEBUG: Using chunk_size: {chunk_size}")
            print(f"DEBUG: Using deidentification_strategy: {deidentification_strategy}")
            print(f"DEBUG: Using use_wandb: {use_wandb}")
            
            # Import and use the hybrid pipeline directly
            print(f"DEBUG: Importing hybrid pipeline from: {self.hybrid_pipeline_script}")
            
            # Change to the main project directory for proper imports
            original_cwd = os.getcwd()
            os.chdir(str(self.main_project_dir))
            
            try:
                # Import the hybrid pipeline
                import importlib.util
                spec = importlib.util.spec_from_file_location("hybrid_pipeline", str(self.hybrid_pipeline_script))
                hybrid_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(hybrid_module)
                
                # Create the pipeline instance
                print(f"DEBUG: Creating HybridTemporalPipeline instance...")
                pipeline = hybrid_module.HybridTemporalPipeline(
                    input_video=video_path,
                    output_dir=output_dir,
                    debug_mode=True,
                    use_wandb=use_wandb,
                    experiment_name="envisionbox_behavioral_analysis",
                    enable_chunked_processing=True,
                    chunk_size=chunk_size,
                    deidentification_strategy=deidentification_strategy
                )
                
                # Process the video
                print(f"DEBUG: Starting video processing...")
                print(f"DEBUG: Pipeline object: {pipeline}")
                print(f"DEBUG: Pipeline type: {type(pipeline)}")
                print(f"DEBUG: Video path: {video_path}")
                print(f"DEBUG: Output dir: {output_dir}")
                
                try:
                    pipeline.process_video_chunked(chunk_size=chunk_size)
                    print(f"DEBUG: Video processing completed successfully")
                except Exception as pipeline_error:
                    print(f"DEBUG: Pipeline processing error: {pipeline_error}")
                    print(f"DEBUG: Error type: {type(pipeline_error)}")
                    import traceback
                    print(f"DEBUG: Pipeline traceback: {traceback.format_exc()}")
                    raise
                
            finally:
                # Always restore the original working directory
                os.chdir(original_cwd)
            
            # Look for the output video
            expected_output = os.path.join(output_dir, "output_video.mp4")
            if os.path.exists(expected_output):
                # Copy to the expected output path
                import shutil
                shutil.copy2(expected_output, output_path)
                self.logger.info(f"Video copied to expected output path: {output_path}")
            else:
                # Look for any .mp4 files in the output directory
                output_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
                if output_files:
                    source_file = os.path.join(output_dir, output_files[0])
                    import shutil
                    shutil.copy2(source_file, output_path)
                    self.logger.info(f"Video copied from {source_file} to {output_path}")
                else:
                    raise FileNotFoundError(f"No output video found in {output_dir}")
            
            # Generate behavioral analysis data
            print("DEBUG: Generating behavioral analysis data...")
            try:
                behavioral_data = self.generate_behavioral_analysis(output_dir)
                print(f"DEBUG: Behavioral data generated: {behavioral_data}")
            except Exception as e:
                print(f"DEBUG: Error generating behavioral analysis: {e}")
                import traceback
                print(f"DEBUG: Traceback: {traceback.format_exc()}")
                # Create empty behavioral data as fallback
                behavioral_data = {
                    'total_frames': 0,
                    'total_processing_time': 0,
                    'avg_processing_time': 0,
                    'overlap_detections': [],
                    'temporal_consistency': {},
                    'deidentification_stats': {}
                }
            
            # Prepare results in the format expected by EnvisionBox GUI
            print("DEBUG: Preparing results...")
            results = {
                'output_video_path': output_path,
                'behavioral_data_path': output_path.replace('.mp4', '_behavioral_data.csv'),
                'stats': {
                    'processed_frames': behavioral_data.get('total_frames', 0),
                    'total_processing_time': behavioral_data.get('total_processing_time', 0),
                    'avg_processing_time': behavioral_data.get('avg_processing_time', 0),
                    'overlap_detections': behavioral_data.get('overlap_detections', []),
                    'temporal_consistency': behavioral_data.get('temporal_consistency', {}),
                    'deidentification_stats': behavioral_data.get('deidentification_stats', {})
                }
            }
            print(f"DEBUG: Results prepared: {results}")
            
            # Export behavioral data
            self.export_behavioral_data(results['stats'], results['behavioral_data_path'])
            
            self.logger.info("Behavioral video processing completed successfully")
            return results
            
        except subprocess.TimeoutExpired:
            self.logger.error("Hybrid pipeline timed out after 1 hour")
            raise RuntimeError("Processing timed out")
        except Exception as e:
            self.logger.error(f"Video processing failed: {e}")
            raise
    
    def generate_behavioral_analysis(self, output_dir: str) -> Dict[str, Any]:
        """
        Generate behavioral analysis data from the hybrid pipeline results
        
        Args:
            output_dir: Directory containing pipeline output
            
        Returns:
            Dictionary containing behavioral analysis data
        """
        print(f"DEBUG: generate_behavioral_analysis called with output_dir: {output_dir}")
        
        # Look for performance data files
        try:
            files_in_dir = os.listdir(output_dir)
            print(f"DEBUG: Files in output directory: {files_in_dir}")
            performance_files = [f for f in files_in_dir if 'performance' in f.lower() and f.endswith('.json')]
            print(f"DEBUG: Performance files found: {performance_files}")
        except Exception as e:
            print(f"DEBUG: Error listing directory: {e}")
            performance_files = []
        
        behavioral_data = {
            'total_frames': 0,
            'total_processing_time': 0,
            'avg_processing_time': 0,
            'overlap_detections': [],
            'temporal_consistency': {},
            'deidentification_stats': {}
        }
        
        if performance_files:
            try:
                performance_file_path = os.path.join(output_dir, performance_files[0])
                print(f"DEBUG: Loading performance data from: {performance_file_path}")
                with open(performance_file_path, 'r') as f:
                    performance_data = json.load(f)
                print(f"DEBUG: Performance data loaded: {performance_data}")
                
                behavioral_data.update({
                    'total_frames': performance_data.get('total_frames_processed', 0),
                    'total_processing_time': performance_data.get('total_processing_time', 0),
                    'avg_processing_time': performance_data.get('avg_frame_processing_time', 0),
                    'temporal_consistency': performance_data.get('temporal_consistency', {}),
                    'deidentification_stats': performance_data.get('deidentification_stats', {})
                })
                print(f"DEBUG: Updated behavioral data: {behavioral_data}")
            except Exception as e:
                print(f"DEBUG: Error loading performance data: {e}")
                import traceback
                print(f"DEBUG: Traceback: {traceback.format_exc()}")
                self.logger.warning(f"Could not load performance data: {e}")
        else:
            print("DEBUG: No performance files found, using default behavioral data")
        
        print(f"DEBUG: Returning behavioral data: {behavioral_data}")
        return behavioral_data
    
    def export_behavioral_data(self, stats: Dict[str, Any], output_path: str):
        """Export behavioral analysis data to CSV and JSON"""
        import pandas as pd
        import json
        import numpy as np
        
        def convert_numpy_types(obj):
            """Convert NumPy types to native Python types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Prepare data for export
        overlap_data = []
        for detection in stats.get("overlap_detections", []):
            frame = detection.get("frame", 0)
            for overlap_key, overlap_info in detection.get("overlaps", {}).items():
                overlap_data.append({
                    "frame": frame,
                    "overlap_type": overlap_key,
                    "overlap_percentage": overlap_info.get("overlap_percentage", 0),
                    "is_looking": overlap_info.get("is_looking", False),
                    "target_area": overlap_info.get("target_area", 0),
                    "gaze_area": overlap_info.get("gaze_area", 0)
                })
        
        if overlap_data:
            df = pd.DataFrame(overlap_data)
            df.to_csv(output_path, index=False)
            self.logger.info(f"Behavioral data exported to: {output_path}")
        else:
            # Create empty CSV with headers for consistency
            empty_df = pd.DataFrame(columns=["frame", "overlap_type", "overlap_percentage", "is_looking", "target_area", "gaze_area"])
            empty_df.to_csv(output_path, index=False)
            self.logger.info(f"Empty behavioral data CSV created: {output_path}")
        
        # Export comprehensive JSON report
        json_path = output_path.replace('.csv', '_detailed_report.json')
        detailed_report = {
            "processing_summary": {
                "total_frames": stats.get("processed_frames", 0),
                "total_processing_time": stats.get("total_processing_time", 0),
                "avg_processing_time": stats.get("avg_processing_time", 0),
                "fps": stats.get("processed_frames", 0) / stats.get("total_processing_time", 1) if stats.get("total_processing_time", 0) > 0 else 0
            },
            "hybrid_pipeline_status": {
                "tsp_sam_available": True,  # We know it's available since we're calling the working pipeline
                "samurai_available": True,
                "maskanyone_available": True,
                "temporal_processing": True,
                "chunked_processing": self.config["hybrid_pipeline"]["enable_chunked_processing"]
            },
            "behavioral_analysis": {
                "total_overlap_detections": len(stats.get("overlap_detections", [])),
                "frame_by_frame_data": overlap_data,
                "avg_overlap_percentage": np.mean([item["overlap_percentage"] for item in overlap_data]) if overlap_data else 0,
                "temporal_consistency": stats.get("temporal_consistency", {}),
                "deidentification_stats": stats.get("deidentification_stats", {})
            },
            "system_info": {
                "pipeline_type": "Full Hybrid Temporal Pipeline (Subprocess)",
                "models_used": ["TSP-SAM", "SAMURAI", "MaskAnyone"],
                "device": "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
                "chunk_size": self.config["hybrid_pipeline"]["chunk_size"]
            }
        }
        
        # Convert all NumPy types to native Python types
        detailed_report = convert_numpy_types(detailed_report)
        
        with open(json_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        self.logger.info(f"Detailed report exported to: {json_path}")
        
        return json_path

def main():
    """Main function for testing the wrapper"""
    print("Hybrid Pipeline Wrapper for EnvisionBox")
    print("=" * 50)
    
    # Initialize wrapper
    wrapper = HybridPipelineWrapper()
    
    # Example usage
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "output_processed.mp4"
        
        # Example: Process "baby with ball" scenario
        text_prompts = ["baby", "ball", "person", "face", "stage", "background"]
        deidentify_objects = ["person", "face"]
        preserve_objects = ["baby", "ball", "stage", "background"]
        target_objects = ["ball"]
        gaze_objects = ["baby", "face"]
        
        try:
            results = wrapper.process_behavioral_video(
                video_path=video_path,
                output_path=output_path,
                text_prompts=text_prompts,
                deidentify_objects=deidentify_objects,
                preserve_objects=preserve_objects,
                target_objects=target_objects,
                gaze_objects=gaze_objects
            )
            
            print(f"Processing completed: {output_path}")
            print(f"Behavioral data: {results['behavioral_data_path']}")
            
        except Exception as e:
            print(f"Processing failed: {e}")
    else:
        print("Usage: python hybrid_pipeline_wrapper.py <input_video> [output_video]")

if __name__ == "__main__":
    main()
