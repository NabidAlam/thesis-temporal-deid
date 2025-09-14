#!/usr/bin/env python3
"""
Standalone Hybrid EnvisionObjectAnnotator Integration
A standalone version that doesn't depend on the main hybrid pipeline.
"""

import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
import logging

class StandaloneHybridIntegration:
    """
    Standalone integration class that provides EnvisionObjectAnnotator capabilities
    for behavioral research without depending on the main hybrid pipeline.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the standalone integration"""
        self.config = self.load_config(config_path)
        self.sam2_predictor = None
        self.setup_logging()
        
        # Initialize SAM2
        self.initialize_sam2()
        
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "main_config.json"
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "sam2": {
                    "model_type": "sam2_hiera_large",
                    "checkpoint_path": "checkpoints/sam2_hiera_large.pt",
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                },
                "video_processing": {
                    "max_frames": 1000,
                    "frame_skip": 1,
                    "output_fps": 30
                },
                "hybrid_integration": {
                    "enable_text_prompts": True,
                    "enable_selective_deidentification": True,
                    "confidence_threshold": 0.5
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
                logging.FileHandler(log_dir / "standalone_integration.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_sam2(self):
        """Initialize SAM2 for text-guided segmentation using wrapper"""
        try:
            from sam2_wrapper import SAM2Wrapper
            
            # Check if checkpoint exists
            checkpoint_path = Path(__file__).parent.parent / self.config["sam2"]["checkpoint_path"]
            
            if not checkpoint_path.exists():
                self.logger.warning(f"SAM2 checkpoint not found: {checkpoint_path}")
                self.logger.info("SAM2 functionality will be limited without checkpoints")
                return
            
            # Initialize SAM2 wrapper
            model_cfg = self.config["sam2"]["model_type"]
            device = self.config["sam2"]["device"]
            
            self.sam2_predictor = SAM2Wrapper(str(checkpoint_path), model_cfg, device)
            
            if self.sam2_predictor.is_initialized():
                self.logger.info(f"SAM2 initialized with {model_cfg} on {device}")
            else:
                self.logger.warning("SAM2 initialization failed - using mock segmentation")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SAM2: {e}")
            self.sam2_predictor = None
    
    def segment_with_text_prompts(self, frame: np.ndarray, text_prompts: List[str]) -> Dict[str, np.ndarray]:
        """
        Segment objects using text prompts (SAMURAI capability)
        
        Args:
            frame: Input frame
            text_prompts: List of text descriptions (e.g., ["person on stage", "ball"])
            
        Returns:
            Dictionary mapping text prompts to segmentation masks
        """
        if self.sam2_predictor is None:
            self.logger.warning("SAM2 predictor not initialized - using mock segmentation")
            return self.mock_segmentation(frame, text_prompts)
        
        try:
            # Use the wrapper's segmentation method
            return self.sam2_predictor.segment_with_text_prompts(frame, text_prompts)
            
        except Exception as e:
            self.logger.error(f"SAM2 segmentation failed: {e}")
            return self.mock_segmentation(frame, text_prompts)
    
    def mock_segmentation(self, frame: np.ndarray, text_prompts: List[str]) -> Dict[str, np.ndarray]:
        """Create mock segmentation masks for testing without SAM2"""
        results = {}
        height, width = frame.shape[:2]
        
        for i, prompt in enumerate(text_prompts):
            # Create a simple mock mask based on prompt
            mask = np.zeros((height, width), dtype=bool)
            
            if "person" in prompt.lower():
                # Mock person mask (center-left area)
                mask[height//4:3*height//4, width//8:width//4] = True
            elif "ball" in prompt.lower():
                # Mock ball mask (center area)
                center_x, center_y = width//2, height//2
                y, x = np.ogrid[:height, :width]
                mask = (x - center_x)**2 + (y - center_y)**2 <= 30**2
            elif "face" in prompt.lower():
                # Mock face mask (upper center)
                mask[height//6:height//3, width//3:2*width//3] = True
            else:
                # Default mask (small area)
                mask[height//3:2*height//3, width//3:2*width//3] = True
            
            results[prompt] = mask
        
        self.logger.info(f"Created mock segmentation for {len(text_prompts)} prompts")
        return results
    
    def selective_deidentification(self, frame: np.ndarray, masks: Dict[str, np.ndarray], 
                                 deidentify_objects: List[str], preserve_objects: List[str]) -> np.ndarray:
        """
        Apply selective de-identification based on object types
        
        Args:
            frame: Input frame
            masks: Dictionary of object masks
            deidentify_objects: Objects to de-identify (blur)
            preserve_objects: Objects to preserve (keep clear)
            
        Returns:
            Frame with selective de-identification applied
        """
        result_frame = frame.copy()
        deidentification_stats = {
            "objects_processed": 0,
            "objects_deidentified": 0,
            "objects_preserved": 0,
            "deidentified_areas": {}
        }
        
        for obj_type, mask in masks.items():
            deidentification_stats["objects_processed"] += 1
            
            if obj_type in deidentify_objects:
                # Apply blur to de-identify
                blurred_region = cv2.GaussianBlur(frame, (15, 15), 0)
                result_frame[mask] = blurred_region[mask]
                
                # Calculate de-identified area
                deidentified_pixels = np.sum(mask)
                total_pixels = mask.shape[0] * mask.shape[1]
                deidentified_percentage = (deidentified_pixels / total_pixels) * 100
                
                deidentification_stats["objects_deidentified"] += 1
                deidentification_stats["deidentified_areas"][obj_type] = {
                    "pixels": int(deidentified_pixels),
                    "percentage": round(deidentified_percentage, 2)
                }
                
                self.logger.info(f"Applied de-identification to '{obj_type}' ({deidentified_percentage:.1f}% of frame)")
                
            elif obj_type in preserve_objects:
                # Keep object clear for behavioral analysis
                deidentification_stats["objects_preserved"] += 1
                self.logger.info(f"Preserved '{obj_type}' for analysis")
        
        # Store stats for later export
        if not hasattr(self, 'deidentification_stats'):
            self.deidentification_stats = []
        self.deidentification_stats.append(deidentification_stats)
        
        return result_frame
    
    def detect_object_overlaps(self, masks: Dict[str, np.ndarray], 
                             target_objects: List[str], 
                             gaze_objects: List[str]) -> Dict[str, Any]:
        """
        Detect overlaps between target objects and gaze objects
        (EnvisionObjectAnnotator functionality)
        
        Args:
            masks: Dictionary of object masks
            target_objects: Objects to track (e.g., ["ball", "toy"])
            gaze_objects: Objects representing gaze (e.g., ["person", "face"])
            
        Returns:
            Dictionary with overlap detection results
        """
        overlap_results = {}
        
        for target in target_objects:
            if target not in masks:
                continue
                
            target_mask = masks[target]
            target_area = np.sum(target_mask)
            
            for gaze in gaze_objects:
                if gaze not in masks:
                    continue
                
                gaze_mask = masks[gaze]
                gaze_area = np.sum(gaze_mask)
                
                # Calculate overlap
                overlap_mask = np.logical_and(target_mask, gaze_mask)
                overlap_area = np.sum(overlap_mask)
                
                # Calculate overlap percentage
                if target_area > 0:
                    overlap_percentage = (overlap_area / target_area) * 100
                else:
                    overlap_percentage = 0
                
                overlap_key = f"{gaze}_looking_at_{target}"
                overlap_results[overlap_key] = {
                    "overlap_area": overlap_area,
                    "overlap_percentage": overlap_percentage,
                    "is_looking": overlap_percentage > 10,  # Threshold for "looking at"
                    "target_area": target_area,
                    "gaze_area": gaze_area
                }
                
                self.logger.info(f"{gaze} looking at {target}: {overlap_percentage:.1f}%")
        
        return overlap_results
    
    def process_behavioral_video(self, video_path: str, output_path: str, 
                               text_prompts: List[str],
                               deidentify_objects: List[str] = None,
                               preserve_objects: List[str] = None,
                               target_objects: List[str] = None,
                               gaze_objects: List[str] = None) -> Dict[str, Any]:
        """
        Process video for behavioral research with selective de-identification
        
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
            preserve_objects = ["ball", "toy", "object"]
        if target_objects is None:
            target_objects = ["ball", "toy"]
        if gaze_objects is None:
            gaze_objects = ["person", "face"]
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing statistics
        stats = {
            "total_frames": total_frames,
            "processed_frames": 0,
            "overlap_detections": [],
            "segmentation_confidence": [],
            "processing_time": []
        }
        
        frame_count = 0
        
        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Text prompts: {text_prompts}")
        self.logger.info(f"De-identify: {deidentify_objects}")
        self.logger.info(f"Preserve: {preserve_objects}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start_time = cv2.getTickCount()
                
                # Segment objects using text prompts
                masks = self.segment_with_text_prompts(frame, text_prompts)
                
                # Detect overlaps for behavioral analysis
                if target_objects and gaze_objects:
                    overlaps = self.detect_object_overlaps(masks, target_objects, gaze_objects)
                    stats["overlap_detections"].append({
                        "frame": frame_count,
                        "overlaps": overlaps
                    })
                
                # Apply selective de-identification
                processed_frame = self.selective_deidentification(
                    frame, masks, deidentify_objects, preserve_objects
                )
                
                # Write processed frame
                out.write(processed_frame)
                
                # Update statistics
                frame_time = (cv2.getTickCount() - frame_start_time) / cv2.getTickFrequency()
                stats["processing_time"].append(frame_time)
                stats["processed_frames"] += 1
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    self.logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        finally:
            cap.release()
            out.release()
        
        # Calculate final statistics
        stats["avg_processing_time"] = np.mean(stats["processing_time"])
        stats["total_processing_time"] = np.sum(stats["processing_time"])
        
        self.logger.info(f"Video processing completed: {output_path}")
        self.logger.info(f"Processed {stats['processed_frames']} frames in {stats['total_processing_time']:.2f}s")
        
        # Export behavioral data to CSV
        csv_path = output_path.replace('.mp4', '_behavioral_data.csv')
        self.export_behavioral_data(stats, csv_path)
        
        # Return results in expected format for GUI
        return {
            'output_video_path': output_path,
            'behavioral_data_path': csv_path,
            'stats': stats
        }
    
    def export_behavioral_data(self, stats: Dict[str, Any], output_path: str):
        """Export behavioral analysis data to CSV and JSON"""
        import pandas as pd
        import json
        
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
        for detection in stats["overlap_detections"]:
            frame = detection["frame"]
            for overlap_key, overlap_info in detection["overlaps"].items():
                overlap_data.append({
                    "frame": frame,
                    "overlap_type": overlap_key,
                    "overlap_percentage": overlap_info["overlap_percentage"],
                    "is_looking": overlap_info["is_looking"],
                    "target_area": overlap_info["target_area"],
                    "gaze_area": overlap_info["gaze_area"]
                })
        
        if overlap_data:
            df = pd.DataFrame(overlap_data)
            df.to_csv(output_path, index=False)
            self.logger.info(f"Behavioral data exported to: {output_path}")
        else:
            self.logger.warning("No overlap data to export")
        
        # Export comprehensive JSON report
        json_path = output_path.replace('.csv', '_detailed_report.json')
        detailed_report = {
            "processing_summary": {
                "total_frames": stats.get("processed_frames", 0),
                "total_processing_time": stats.get("total_processing_time", 0),
                "avg_processing_time": stats.get("avg_processing_time", 0),
                "fps": stats.get("processed_frames", 0) / stats.get("total_processing_time", 1) if stats.get("total_processing_time", 0) > 0 else 0
            },
            "deidentification_stats": {
                "objects_deidentified": stats.get("deidentified_objects", []),
                "objects_preserved": stats.get("preserved_objects", []),
                "deidentification_applied_frames": stats.get("deidentification_frames", 0)
            },
            "behavioral_analysis": {
                "total_overlap_detections": len(stats.get("overlap_detections", [])),
                "frame_by_frame_data": overlap_data,
                "avg_overlap_percentage": np.mean([item["overlap_percentage"] for item in overlap_data]) if overlap_data else 0
            },
            "processing_timeline": stats.get("processing_time", []),
            "system_info": {
                "sam2_initialized": self.sam2_predictor is not None and self.sam2_predictor.is_initialized(),
                "device": self.config["sam2"]["device"],
                "model_type": self.config["sam2"]["model_type"]
            }
        }
        
        # Convert all NumPy types to native Python types
        detailed_report = convert_numpy_types(detailed_report)
        
        with open(json_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        self.logger.info(f"Detailed report exported to: {json_path}")
        
        return json_path

def main():
    """Main function for testing the standalone integration"""
    print("Standalone Hybrid EnvisionObjectAnnotator Integration")
    print("=" * 60)
    
    # Initialize integration
    integration = StandaloneHybridIntegration()
    
    # Example usage
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "output_processed.mp4"
        
        # Example: Process "baby with ball" scenario
        text_prompts = ["baby", "ball", "person", "face"]
        deidentify_objects = ["person", "face"]
        preserve_objects = ["ball", "baby"]
        target_objects = ["ball"]
        gaze_objects = ["baby", "face"]
        
        try:
            stats = integration.process_behavioral_video(
                video_path=video_path,
                output_path=output_path,
                text_prompts=text_prompts,
                deidentify_objects=deidentify_objects,
                preserve_objects=preserve_objects,
                target_objects=target_objects,
                gaze_objects=gaze_objects
            )
            
            # Export behavioral data
            csv_path = output_path.replace('.mp4', '_behavioral_data.csv')
            integration.export_behavioral_data(stats, csv_path)
            
            print(f"Processing completed: {output_path}")
            print(f"Behavioral data: {csv_path}")
            
        except Exception as e:
            print(f"Processing failed: {e}")
    else:
        print("Usage: python standalone_hybrid_integration.py <input_video> [output_video]")

if __name__ == "__main__":
    main()
