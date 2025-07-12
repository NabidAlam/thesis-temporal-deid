import numpy as np
import torch
import cv2

class SAMURAIClient:
    def __init__(self, checkpoint_path, device="cuda"):
        # TODO: load SAMURAI model and weights here
        self.device = device
        self.model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path):
        # Implement this by reading the SAMURAI scripts
        # Usually involves loading a config and a torch.load()
        pass

    def predict(self, frame_np: np.ndarray, prev_mask: np.ndarray = None) -> np.ndarray:
        """
        Args:
            frame_np: RGB image as ndarray
            prev_mask: Optional temporal mask from previous frame
        Returns:
            Binary mask as ndarray
        """
        # TODO: resize, normalize, to tensor, forward pass
        pass
