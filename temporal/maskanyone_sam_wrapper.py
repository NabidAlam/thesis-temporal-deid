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

    def segment_with_box(self, image_np, bbox_str, mode="best"):
        """
        Run SAM2 segmentation on a given image and bounding box.

        Parameters:
            image_np (np.ndarray): Input RGB image (HWC, uint8)
            bbox_str (str): Bounding box as a string, e.g., "[480, 216, 1440, 972]"
            mode (str): SAM mode — "best", "everything", etc.

        Returns:
            mask (np.ndarray): Binary mask (uint8) from SAM, same size as input image
        """
        if not isinstance(bbox_str, str):
            raise ValueError(f"bbox must be a string like '[x1, y1, x2, y2]', got: {type(bbox_str)}")

        if image_np.dtype != np.uint8:
            raise ValueError("Image must be in uint8 format")
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError("Image must be RGB (HWC, 3 channels)")

        image_pil = Image.fromarray(image_np)

        try:
            masks = self.client.predict(image=image_pil, boxes=bbox_str, mode=mode)
        except Exception as e:
            print(f"[ERROR] SAM2 prediction failed: {e}")
            return np.zeros(image_np.shape[:2], dtype=np.uint8)

        if not masks or masks[0] is None:
            print("[WARN] SAM2 returned no mask for box:", bbox_str)
            return np.zeros(image_np.shape[:2], dtype=np.uint8)

        return (np.array(masks[0]).astype(np.uint8)) * 255
