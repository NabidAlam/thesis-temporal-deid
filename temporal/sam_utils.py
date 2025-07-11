import torch
import numpy as np
import cv2

from segment_anything import sam_model_registry, SamPredictor

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose.Pose(static_image_mode=True)
except ImportError:
    mp_pose = None
    print("[WARN] MediaPipe not found — bounding box detection will not work.")


def load_sam_model(model_type="vit_b", checkpoint="sam_vit_b.pth", device="cuda"):
    print(f"Loading SAM model ({model_type})...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


def get_pose_bounding_boxes(frame):
    if mp_pose is None:
        return []

    results = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes = []
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape
        xs = [l.x * w for l in landmarks]
        ys = [l.y * h for l in landmarks]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        boxes.append([x_min, y_min, x_max, y_max])
    return boxes


def run_sam_with_openpose(frame, predictor, device="cuda"):
    predictor.set_image(frame)
    boxes = get_pose_bounding_boxes(frame)
    if not boxes:
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    # Convert to SAM input format
    input_boxes = torch.tensor(boxes, device=device, dtype=torch.float)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, frame.shape[:2])
    masks, _, _ = predictor.predict_torch(boxes=transformed_boxes, multimask_output=False)

    # Combine masks if multiple persons detected
    combined_mask = torch.any(masks.squeeze(1), dim=0).cpu().numpy().astype(np.uint8) * 255
    return combined_mask


def fuse_masks(mask1, mask2, method="union"):
    if method == "union":
        return np.logical_or(mask1 > 0, mask2 > 0).astype(np.uint8) * 255
    elif method == "intersection":
        return np.logical_and(mask1 > 0, mask2 > 0).astype(np.uint8) * 255
    elif method == "sam_priority":
        return np.where(mask2 > 0, mask2, mask1)
    else:
        raise ValueError(f"Unknown fusion method: {method}")
