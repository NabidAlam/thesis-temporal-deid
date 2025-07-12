import os
import sys
import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath("samurai"))

from sam2.sam2.build_sam import build_sam2

from utils import resize_frame, save_mask_and_frame
from pose_extractor import extract_pose_keypoints


def scale_keypoints(keypoints, original_shape, target_shape=(512, 512)):
    orig_h, orig_w = original_shape
    target_h, target_w = target_shape
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    return [[int(x * scale_x), int(y * scale_y)] for x, y in keypoints]


def run_samurai(input_path, output_path_base, config_path):
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    infer_cfg = config["inference"]
    output_cfg = config["output"]

    frame_stride = infer_cfg.get("frame_stride", 6)
    min_area = infer_cfg.get("min_area", 100)
    suppress_bottom = infer_cfg.get("suppress_bottom_text", False)

    device = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model_type = model_cfg.get("model_type", "hiera_base_plus")
    checkpoint_path = model_cfg["checkpoint_path"]

    print(f"Loading SAMURAI model ({model_type}) from: {checkpoint_path}")
    samurai_model = build_sam2(model_type, checkpoint_path, device=device)

    video_name = Path(input_path).stem
    output_path = Path(output_path_base) / video_name
    if output_path.exists():
        print(f"Cleaning previous run: {output_path}")
        import shutil
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    overlay_dir = output_path / "overlays"
    composite_dir = output_path / "composites"
    mask_dir = output_path / "masks"
    debug_dir = output_path / "pose_debug"
    for d in [overlay_dir, composite_dir, mask_dir, debug_dir]:
        d.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    pose_json = {}
    pose_json_path = output_path / "pose_keypoints.json"

    with tqdm(total=total_frames // frame_stride) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            frame_resized = resize_frame(frame, infer_cfg)
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            keypoints = extract_pose_keypoints(rgb, draw_debug=True, frame_idx=frame_idx, debug_dir=str(debug_dir))
            pose_json[frame_idx] = keypoints

            if keypoints:
                scaled_kpts = scale_keypoints(keypoints, frame_resized.shape[:2])
                point_labels = [1] * len(scaled_kpts)
                mask = samurai_model.predict_from_points(rgb, scaled_kpts, point_labels)
            else:
                mask = np.zeros((512, 512), dtype=np.uint8)

            if np.sum(mask) > min_area:
                resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                save_mask_and_frame(
                    frame,
                    resized_mask,
                    str(output_path),
                    frame_idx,
                    save_overlay=output_cfg.get("save_overlay", True),
                    overlay_alpha=output_cfg.get("overlay_alpha", 0.5),
                    save_frames=output_cfg.get("save_frames", False),
                    save_composite=True
                )

            frame_idx += 1
            pbar.update(1)

    with open(pose_json_path, "w") as f_json:
        json.dump(pose_json, f_json, indent=2)

    cap.release()
    print(f"\nFinished processing {frame_idx} frames.")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python samurai_runner.py <input_video> <output_base_dir> <config_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_base = sys.argv[2]
    config_path = sys.argv[3]
    run_samurai(input_path, output_base, config_path)
