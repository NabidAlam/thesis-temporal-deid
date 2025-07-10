import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root and samurai path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samurai")))

from sam2.build_sam import build_sam2_video_predictor
from temporal.utils import save_mask_and_frame

# Hydra imports
from hydra import initialize_config_dir
from hydra.compose import compose
from hydra.core.global_hydra import GlobalHydra

def run_samurai(input_path, output_base):
    video_name = Path(input_path).stem
    output_path = Path(output_base) / video_name
    output_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading SAM2VideoPredictor from local config and checkpoint...")
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samurai", "sam2", "sam2"))
    config_file = "sam2_hiera_b+.yaml"
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samurai", "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt"))

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, job_name="samurai_job", version_base=None):
        cfg = compose(config_name=config_file)

    predictor = build_sam2_video_predictor(
        config_file=cfg,
        checkpoint=ckpt_path,
        load_checkpoint=True,
    )

    print(f"[INFO] Initializing inference state for video: {input_path}")
    state = predictor.init_state(input_path)
    print(f"[INFO] Video loaded with {state['num_frames']} frames, resolution: {state['video_width']}x{state['video_height']}")

    frame_height = state["video_height"]
    frame_width = state["video_width"]
    center_box = [
        frame_width * 0.25, frame_height * 0.25,
        frame_width * 0.75, frame_height * 0.75
    ]
    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        box=center_box,
    )

    print("[INFO] Starting mask propagation...")
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        print(f"[Frame {frame_idx}] Found {len(masks)} masks for objects: {obj_ids}")
        if masks is None or len(masks) == 0:
            print(f"[WARN] No masks for frame {frame_idx}")
            continue
        for i, mask in enumerate(masks):
            mask_np = (mask.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
            print(f"  - Saving mask {i} with {np.sum(mask_np > 0)} non-zero pixels")
            save_mask_and_frame(
                None,
                mask_np,
                str(output_path),
                frame_idx,
                save_overlay=True,
                overlay_alpha=0.5,
                save_frames=True,
                save_masks=True
            )

    print("\n[INFO] Finished processing video")
    print(f"[INFO] Output written to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python samurai_runner.py <input_video> <output_base_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_base = sys.argv[2]

    run_samurai(input_path, output_base)
