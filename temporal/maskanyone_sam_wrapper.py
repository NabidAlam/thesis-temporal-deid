import os
import sys
import numpy as np
import cv2
from PIL import Image

# Add your custom SAM2 client
from temporal.my_sam2_client import MySAM2Client as SAM2Client

# Add MaskAnyone root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "maskanyone"))
sys.path.append(ROOT_DIR)

class MaskAnyoneSAMWrapper:
    def __init__(self, sam_server_url="http://localhost:8081"):
        self.client = SAM2Client(server_url=sam_server_url)
        # Try to use local SAM if available
        try:
            from segment_anything import sam_model_registry, SamPredictor
            import torch
            
            # Check if SAM checkpoint exists
            sam_checkpoint = "temporal/checkpoints/sam_vit_b_01ec64.pth"
            if os.path.exists(sam_checkpoint):
                print("[SAM] Loading local SAM model...")
                self.sam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
                self.sam_model.to("cuda" if torch.cuda.is_available() else "cpu")
                self.sam_predictor = SamPredictor(self.sam_model)
                self.use_local_sam = True
                print("[SAM] Local SAM model loaded successfully")
            else:
                print(f"[SAM] Checkpoint not found at {sam_checkpoint}, will use external server")
                self.use_local_sam = False
        except Exception as e:
            print(f"[SAM] Failed to load local SAM: {e}")
            self.use_local_sam = False
    
    def segment_with_box(self, image, bbox_str):
        """Segment image using bounding box prompt"""
        try:
            # Parse bbox string "[x1, y1, x2, y2]"
            bbox = eval(bbox_str)
            if not isinstance(bbox, list) or len(bbox) != 4:
                print(f"[SAM] Invalid bbox format: {bbox_str}")
                return np.zeros(image.shape[:2], dtype=np.uint8)
            
            x1, y1, x2, y2 = bbox
            
            if self.use_local_sam:
                # Use local SAM model
                return self._segment_local_sam(image, bbox)
            else:
                # Try external server
                return self._segment_external_sam(image, bbox)
                
        except Exception as e:
            print(f"[SAM] Error in segment_with_box: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def _segment_local_sam(self, image, bbox):
        """Use local SAM model for segmentation"""
        try:
            # Convert image to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                if image.dtype == np.uint8:
                    image_rgb = image
                else:
                    image_rgb = (image * 255).astype(np.uint8)
            else:
                print("[SAM] Invalid image format for local SAM")
                return np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Set image in predictor
            self.sam_predictor.set_image(image_rgb)
            
            # Convert bbox to SAM format [x1, y1, x2, y2]
            input_box = np.array(bbox)
            
            # Generate mask
            masks, scores, logits = self.sam_predictor.predict(
                box=input_box,
                multimask_output=False
            )
            
            # Return the mask
            mask = masks[0].astype(np.uint8) * 255
            print(f"[SAM] Local SAM generated mask with area: {np.sum(mask > 0)}")
            return mask
            
        except Exception as e:
            print(f"[SAM] Local SAM failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def _segment_external_sam(self, image, bbox):
        """Try external SAM server"""
        try:
            # Convert image to PIL
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Use SAM2 client
            masks = self.client.predict_box(pil_image, bbox)
            if masks and masks[0] is not None:
                mask = masks[0].astype(np.uint8) * 255
                print(f"[SAM] External SAM generated mask with area: {np.sum(mask > 0)}")
                return mask
            else:
                print("[SAM] External SAM returned no masks")
                return np.zeros(image.shape[:2], dtype=np.uint8)
                
        except Exception as e:
            print(f"[SAM] External SAM failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def segment_with_points(self, image, points, labels):
        """Segment image using point prompts"""
        try:
            if self.use_local_sam:
                return self._segment_local_sam_points(image, points, labels)
            else:
                return self._segment_external_sam_points(image, points, labels)
        except Exception as e:
            print(f"[SAM] Error in segment_with_points: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def _segment_local_sam_points(self, image, points, labels):
        """Use local SAM model with point prompts"""
        try:
            # Convert image to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                if image.dtype == np.uint8:
                    image_rgb = image
                else:
                    image_rgb = (image * 255).astype(np.uint8)
            else:
                print("[SAM] Invalid image format for local SAM points")
                return np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Set image in predictor
            self.sam_predictor.set_image(image_rgb)
            
            # Convert points to numpy array
            input_points = np.array(points)
            input_labels = np.array(labels)
            
            # Generate mask
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False
            )
            
            # Return the mask
            mask = masks[0].astype(np.uint8) * 255
            print(f"[SAM] Local SAM points generated mask with area: {np.sum(mask > 0)}")
            return mask
            
        except Exception as e:
            print(f"[SAM] Local SAM points failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def _segment_external_sam_points(self, image, points, labels):
        """Try external SAM server with points"""
        try:
            # Convert image to PIL
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Use SAM2 client
            masks = self.client.predict_points(pil_image, points, labels)
            if masks and masks[0] is not None:
                mask = masks[0].astype(np.uint8) * 255
                print(f"[SAM] External SAM points generated mask with area: {np.sum(mask > 0)}")
                return mask
            else:
                print("[SAM] External SAM points returned no masks")
                return np.zeros(image.shape[:2], dtype=np.uint8)
                
        except Exception as e:
            print(f"[SAM] External SAM points failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)
