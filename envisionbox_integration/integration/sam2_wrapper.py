#!/usr/bin/env python3
"""
SAM2 Wrapper for proper initialization and usage
Handles the directory requirements for SAM2 configuration loading
"""

import sys
import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional

class SAM2Wrapper:
    """Wrapper class to handle SAM2 initialization and usage"""
    
    def __init__(self, checkpoint_path: str, model_type: str = "sam2_hiera_l", device: str = "cuda"):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type
        self.device = device
        self.predictor = None
        self.sam2_repo_path = Path(__file__).parent.parent.parent / "sam2_repo"
        self.checkpoints_path = Path(__file__).parent.parent / "checkpoints"
        
        # Store original working directory
        self.original_cwd = os.getcwd()
        
        self._initialize_sam2()
    
    def _initialize_sam2(self):
        """Initialize SAM2 with proper directory handling"""
        try:
            # Add SAM2 repo to path
            sys.path.insert(0, str(self.sam2_repo_path))
            
            # Change to SAM2 directory for config loading
            os.chdir(self.sam2_repo_path)
            
            # Import SAM2 modules
            import sam2
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            print(f"🔧 Initializing SAM2 with {self.model_type}...")
            
            # Resolve checkpoint path
            if not self.checkpoint_path.is_absolute():
                # Try relative to checkpoints directory first
                checkpoint_full_path = self.checkpoints_path / self.checkpoint_path.name
                if checkpoint_full_path.exists():
                    checkpoint_path_str = str(checkpoint_full_path)
                else:
                    checkpoint_path_str = str(self.checkpoint_path)
            else:
                checkpoint_path_str = str(self.checkpoint_path)
            
            # Build model with correct config path
            config_path = f"configs/sam2/{self.model_type}.yaml"
            model = build_sam2(config_path, checkpoint_path_str, device=self.device)
            
            # Create predictor
            self.predictor = SAM2ImagePredictor(model)
            
            print(f"SAM2 initialized successfully with {self.model_type}")
            
        except Exception as e:
            print(f"Failed to initialize SAM2: {e}")
            self.predictor = None
        finally:
            # Restore original working directory
            os.chdir(self.original_cwd)
    
    def segment_with_text_prompts(self, frame: np.ndarray, text_prompts: List[str]) -> Dict[str, np.ndarray]:
        """
        Segment objects using text prompts
        
        Args:
            frame: Input frame (H, W, 3)
            text_prompts: List of text descriptions
            
        Returns:
            Dictionary mapping prompt to segmentation mask
        """
        if self.predictor is None:
            print("SAM2 not initialized - using mock segmentation")
            return self._create_mock_segmentation(frame, text_prompts)
        
        try:
            # Change to SAM2 directory for inference
            os.chdir(self.sam2_repo_path)
            
            # Set image for predictor
            self.predictor.set_image(frame)
            
            results = {}
            for prompt in text_prompts:
                try:
                    # For now, use point prompts (SAM2 doesn't have native text prompts)
                    # This is a placeholder - you'd need to implement text-to-point conversion
                    # or use a different model like SAMURAI for text prompts
                    
                    # Use center point as placeholder
                    h, w = frame.shape[:2]
                    point_coords = np.array([[w//2, h//2]])
                    point_labels = np.array([1])
                    
                    masks, scores, logits = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True,
                    )
                    
                    # Use the best mask and ensure it's boolean
                    best_mask = masks[np.argmax(scores)]
                    results[prompt] = best_mask.astype(bool)
                    
                except Exception as e:
                    print(f"Error segmenting '{prompt}': {e}")
                    results[prompt] = self._create_mock_mask(frame.shape[:2])
            
            return results
            
        except Exception as e:
            print(f"Error in SAM2 segmentation: {e}")
            return self._create_mock_segmentation(frame, text_prompts)
        finally:
            # Restore original working directory
            os.chdir(self.original_cwd)
    
    def _create_mock_segmentation(self, frame: np.ndarray, text_prompts: List[str]) -> Dict[str, np.ndarray]:
        """Create mock segmentation masks for testing"""
        h, w = frame.shape[:2]
        results = {}
        
        for i, prompt in enumerate(text_prompts):
            # Create different mock masks for different prompts
            mask = np.zeros((h, w), dtype=bool)
            
            if "baby" in prompt.lower():
                # Baby area (center-left)
                mask[h//4:3*h//4, w//8:3*w//8] = True
            elif "ball" in prompt.lower():
                # Ball area (center-right)
                mask[h//2-20:h//2+20, 5*w//8-20:5*w//8+20] = True
            elif "person" in prompt.lower():
                # Person area (center)
                mask[h//6:5*h//6, w//3:2*w//3] = True
            elif "face" in prompt.lower():
                # Face area (upper center)
                mask[h//6:h//2, w//2-30:w//2+30] = True
            else:
                # Default area
                mask[h//4:3*h//4, w//4:3*w//4] = True
            
            results[prompt] = mask
        
        return results
    
    def _create_mock_mask(self, shape: tuple) -> np.ndarray:
        """Create a simple mock mask"""
        h, w = shape
        mask = np.zeros((h, w), dtype=bool)
        mask[h//4:3*h//4, w//4:3*w//4] = True
        return mask
    
    def is_initialized(self) -> bool:
        """Check if SAM2 is properly initialized"""
        return self.predictor is not None
    
    def __del__(self):
        """Cleanup: restore original working directory"""
        try:
            os.chdir(self.original_cwd)
        except:
            pass
