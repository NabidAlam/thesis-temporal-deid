# ----------------------------------------------------------
# TSP-SAM + SAM + Pose Fusion Runner (MaskAnyone-Temporal)
# ----------------------------------------------------------
# Description:
# This script runs temporally-consistent video segmentation
# using TSP-SAM as the primary model, and optionally fuses it
# with SAM (via bounding box prompts) and SAM2 (via pose keypoints).
#
# Features:
# - TSP-SAM-based core segmentation with adaptive thresholding
# - Optional SAM integration using TSP-based bounding box prompting
# - Optional pose-guided mask generation using OpenPose + SAM2
# - Configurable mask fusion logic via YAML: "tsp_only", "union", "pose_only", "tsp+sam", "tsp+pose"
# - Dynamic morphological kernel sizing for dataset adaptability
# - Smart warmup-based pose mask thresholding with percentile control
# - Aspect-ratio-aware relaxed pose filtering (e.g., for TED talkers vs full body)
# - Frame-wise dominant source diagnostic logging (TSP/SAM/Pose)
# - Optional temporal smoothing with rolling mask memory
# - Connected component analysis for evaluating segmentation stability
# - Full DAVIS and TED dataset support (video or folder input)
# - Output:
#     • Binary masks
#     • Overlay images
#     • Composite debug visualizations
#     • Frame-by-frame statistics in `debug_stats.csv`
#
# Usage:
# python temporal/tsp_sam_complete.py <input_path> <output_dir> <config.yaml> [--force]


# python temporal/tsp_sam_complete.py input/davis2017/JPEGImages/480p/kid-football output/tsp_sam/davis configs/tsp_sam_davis.yaml --force
# python temporal/tsp_sam_complete.py input/ted/video1.mp4 output/tsp_sam/ted configs/tsp_sam_ted.yaml --force


# -------------------------------------------------------------
# Recommended DAVIS Sequences for Baseline Testing (TSP-SAM)
# -------------------------------------------------------------
#  Person-centric:
#   - dog (animal with dynamic pose)
#   - dance-twirl (human motion, fast rotation)
#   - dance-jump (vertical motion, kinematic trails)
#   - breakdance / breakdance-flare (fast whole-body motion)
#   - disc-jockey (occlusion + crowd)
#   - boxing-fisheye (distorted field, ego-motion)

#  Behavioral Relevance (Privacy Stress Test):
#   - car-turn (foreground-background separation)
#   - camel (sparse scene, silhouette privacy)
#   - bus (large rigid object, track masking consistency)
#   - color-run (multiple moving objects in frame)
#   - dog-agility (pose kinematics on quadruped)

#  Skip: object-centric or static cases like 'bear', 'boat', 'blackswan'

import os
import sys
import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath("tsp_sam"))
sys.path.append(os.path.abspath("temporal"))

from tsp_sam.lib.pvtv2_afterTEM import Network
from utils import save_mask_and_frame, resize_frame
from maskanyone_sam_wrapper import MaskAnyoneSAMWrapper
from pose_extractor import extract_pose_keypoints
from my_sam2_client import MySAM2Client

def scale_keypoints(keypoints, original_shape, target_shape=(512, 512)):
    orig_h, orig_w = original_shape
    target_h, target_w = target_shape
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    return [[int(x * scale_x), int(y * scale_y)] for x, y in keypoints]

def get_dynamic_kernel_size(mask_shape, base_divisor=100, max_kernel=15):
    h, w = mask_shape[:2]
    avg_dim = (h + w) / 2
    k = max(3, int(avg_dim // base_divisor))
    k = k + (k % 2 == 0)
    return min(k, max_kernel)

def post_process_fused_mask(fused_mask, min_area=100, kernel_size=None):
    if kernel_size is None:
        kernel_size = get_dynamic_kernel_size(fused_mask.shape)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    cleaned = cv2.morphologyEx(fused_mask, cv2.MORPH_OPEN, kernel)
    
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(closed)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(filtered, [cnt], -1, 255, -1)
    return cv2.dilate(filtered, kernel, iterations=1)




def extract_bbox_from_mask(mask, margin_ratio=0.05):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    x1 = max(x - margin_x, 0)
    y1 = max(y - margin_y, 0)
    x2 = min(x + w + margin_x, mask.shape[1])
    y2 = min(y + h + margin_y, mask.shape[0])
    return [x1, y1, x2, y2]

def get_adaptive_threshold(prob_np, percentile=96):
    return np.percentile(prob_np.flatten(), percentile)

def model_infer_real(model, frame, infer_cfg, debug_save_dir=None, frame_idx=None):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        if output.dim() == 4:
            output = output[0]
        output = output.squeeze()
        prob = torch.sigmoid(output).cpu().numpy()

    percentile = infer_cfg.get("adaptive_percentile", 96)
    min_area = infer_cfg.get("min_area", 1500)
    suppress_bottom = infer_cfg.get("suppress_bottom_text", False)
    adaptive_thresh = get_adaptive_threshold(prob, percentile=percentile)
    raw_mask = (prob > adaptive_thresh).astype(np.uint8)
    mask_denoised = median_filter(raw_mask, size=3)
    mask_denoised = cv2.GaussianBlur(mask_denoised, (5, 5), 0)

    kernel = np.ones((5, 5), np.uint8)
    mask_open = cv2.morphologyEx(mask_denoised, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    if suppress_bottom:
        h = mask_cleaned.shape[0]
        mask_cleaned[int(h * 0.9):, :] = 0

    num_pixels = np.sum(mask_cleaned)
    final_mask = mask_cleaned * 255 if num_pixels >= min_area else np.zeros_like(mask_cleaned, dtype=np.uint8)

    stats = {
        "mean": float(prob.mean()), "max": float(prob.max()), "min": float(prob.min()),
        "adaptive_thresh": float(adaptive_thresh), "mask_area": int(num_pixels)
    }

    if debug_save_dir and frame_idx is not None:
        os.makedirs(debug_save_dir, exist_ok=True)
        for thresh in [0.3, 0.5, 0.7, 0.9]:
            temp_mask = (prob > thresh).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(debug_save_dir, f"{frame_idx:05d}_th{int(thresh * 100)}.png"), temp_mask)
        cv2.imwrite(os.path.join(debug_save_dir, f"{frame_idx:05d}_adaptive_{int(adaptive_thresh * 100)}.png"), final_mask)
        plt.hist(prob.ravel(), bins=50)
        plt.title(f"Pixel Probabilities (frame {frame_idx})")
        plt.savefig(os.path.join(debug_save_dir, f"{frame_idx:05d}_hist.png"))
        plt.close()

    return final_mask, stats

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_base_dir")
    parser.add_argument("config_path")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()

def run_tsp_sam(input_path, output_path_base, config_path, force=False):
    import shutil
    from collections import deque

    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    infer_cfg = config["inference"]
    output_cfg = config["output"]
    dataset_mode = config.get("dataset", {}).get("mode", "ted").lower()
    fusion_method = config.get("fusion_method", "tsp_only")

    fusion_cfg = config.get("fusion", {})
    enable_sam = fusion_cfg.get("enable_sam", True)
    enable_pose = fusion_cfg.get("enable_pose", True)

    use_sam = enable_sam and fusion_method in ("sam_only", "union", "tsp+sam")
    use_pose = enable_pose and dataset_mode != "davis" and fusion_method in ("pose_only", "union", "tsp+pose")
    temporal_smoothing = infer_cfg.get("temporal_smoothing", False)

    # Load model
    opt = type("opt", (object,), {})()
    opt.resume = model_cfg["checkpoint_path"]
    opt.device = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    opt.gpu_ids = [0] if opt.device == "cuda" else []
    opt.channel = model_cfg.get("channel", 32)

    model = Network(opt)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids).to(opt.device)
    model.eval()
    weights = torch.load(opt.resume, map_location=opt.device)
    model.module.feat_net.pvtv2_en.load_state_dict(weights, strict=False)

    sam_wrapper = MaskAnyoneSAMWrapper()
    sam2_client = MySAM2Client()

    video_name = Path(input_path).stem
    output_path = Path(output_path_base) / video_name
    if output_path.exists() and force:
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    frame_stride = infer_cfg.get("frame_stride", 2)
    min_area = infer_cfg.get("min_area", 1500)

    if dataset_mode == "davis":
        image_paths = sorted(Path(input_path).glob("*.jpg"))
        frame_iter = enumerate(image_paths)
        total_frames = len(image_paths)
        dynamic_pose_thresh = 0
    else:
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_iter = enumerate(range(total_frames))

        warmup_areas = []
        for _ in range(50):
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = resize_frame(frame, infer_cfg)
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            keypoints = extract_pose_keypoints(rgb)
            if keypoints:
                scaled = scale_keypoints(keypoints, frame_resized.shape[:2])
                masks = sam2_client.predict_points(Image.fromarray(rgb), scaled, [1]*len(scaled))
                if masks and masks[0] is not None:
                    pose_mask = cv2.resize(masks[0], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    warmup_areas.append(np.sum(pose_mask > 0))
        
    
            
        # Get dynamic threshold settings
        pose_thresh_p = infer_cfg.get("pose_thresh_percentile", 10)
        max_thresh = infer_cfg.get("max_dynamic_thresh", 160000)

        if warmup_areas:
            percentile_value = np.percentile(warmup_areas, pose_thresh_p)
            dynamic_pose_thresh = min(percentile_value, max_thresh)
            print(f"[WARMUP] min={min(warmup_areas)}, p{pose_thresh_p}={percentile_value:.1f}, capped={dynamic_pose_thresh}")
        else:
            dynamic_pose_thresh = 300
        
                

        # dynamic_pose_thresh = np.percentile(warmup_areas, 10) if warmup_areas else 300

    with open(output_path / "debug_stats.csv", "w", newline="") as dbgfile:
        writer = csv.writer(dbgfile)
        writer.writerow([
            "frame", "mean", "max", "min", "adaptive_thresh",
            "tsp_area", "sam_area", "pose_area", "fused_area", "max_region"
        ])

        mask_memory = deque(maxlen=5)
        prev_valid_mask = None

        with tqdm(total=total_frames // frame_stride) as pbar:
            for idx, frame_data in frame_iter:
                if idx % frame_stride != 0:
                    continue

                if dataset_mode == "davis":
                    frame = cv2.imread(str(frame_data))
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break

                frame_resized = resize_frame(frame, infer_cfg)
                rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                tsp_mask, stats = model_infer_real(
                    model, frame_resized, infer_cfg,
                    debug_save_dir=output_path / "tsp_thresh", frame_idx=idx
                )

                sam_mask = np.zeros_like(tsp_mask)
                pose_mask = np.zeros_like(tsp_mask)

                if use_sam:
                    bbox = extract_bbox_from_mask(tsp_mask)
                    if bbox:
                        sam_raw = sam_wrapper.segment_with_box(rgb, str(bbox))
                        sam_mask = cv2.resize(sam_raw, tsp_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
                            
                if use_pose:
                    keypoints = extract_pose_keypoints(rgb)
                    if keypoints:
                        visible_kps = [kp for kp in keypoints if kp[0] > 0 and kp[1] > 0]
                        if len(visible_kps) < 8:  # Require at least 8 visible keypoints
                            print(f"Frame {idx}: Skipping due to insufficient keypoints ({len(visible_kps)})")
                            pbar.update(1)
                            continue
                        scaled = scale_keypoints(keypoints, frame_resized.shape[:2])
                        masks = sam2_client.predict_points(Image.fromarray(rgb), scaled, [1]*len(scaled))
                        if masks and masks[0] is not None:
                            pose_mask = cv2.resize(masks[0], tsp_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)


                pose_area = int(np.sum(pose_mask > 0))
                sam_area = int(np.sum(sam_mask > 0))
                tsp_area = int(np.sum(tsp_mask > 0))
                
                print(f"[DEBUG] Frame {idx}: tsp_area={tsp_area}, sam_area={sam_area}, pose_area={pose_area}")
                
                dominant = "pose" if pose_area > sam_area and pose_area > tsp_area else \
                           "sam" if sam_area > tsp_area else "tsp"
                print(f"Frame {idx}: Dominant source = {dominant}")


                if dataset_mode != "davis" and use_pose:
                    h, w = frame.shape[:2]
                    aspect_ratio = h / w
                    relaxed_thresh = dynamic_pose_thresh * 0.5 if aspect_ratio > 1.4 else dynamic_pose_thresh
                    if pose_area < relaxed_thresh:
                        print(f"Frame {idx}: Pose area {pose_area} < relaxed threshold {relaxed_thresh}, skipping. Dominant: {dominant}")
                        # print(f"Frame {idx}: Pose area {pose_area} < relaxed threshold {relaxed_thresh}, skipping.")
                        pbar.update(1)
                        continue

                # Mask Fusion Logic
                print(f"Frame {idx}: Fusion method = {fusion_method}")

                fused_mask = tsp_mask
                if fusion_method == "union":
                    fused_mask = cv2.bitwise_or(fused_mask, sam_mask)
                    if use_pose:
                        fused_mask = cv2.bitwise_or(fused_mask, pose_mask)
                elif fusion_method == "sam_only":
                    fused_mask = sam_mask
                elif fusion_method == "pose_only":
                    fused_mask = pose_mask
                elif fusion_method == "tsp+sam":
                    fused_mask = cv2.bitwise_or(tsp_mask, sam_mask)
                elif fusion_method == "tsp+pose":
                    fused_mask = cv2.bitwise_or(tsp_mask, pose_mask)

                # Optional Temporal Smoothing
                if temporal_smoothing:
                    mask_memory.append(fused_mask.astype(np.float32))
                    smoothed_mask = np.mean(mask_memory, axis=0)
                    fused_mask = (smoothed_mask > 127).astype(np.uint8) * 255

                fused_mask = post_process_fused_mask(fused_mask, min_area=min_area)

                fused_area = int(np.sum(fused_mask > 0))
                num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(fused_mask)
                num_regions = num_labels - 1  # exclude background
                print(f"Frame {idx}: Connected regions = {num_regions}")

                max_region = 0 if len(stats_cc) <= 1 else np.max(stats_cc[1:, cv2.CC_STAT_AREA])

                if fused_area == 0 and prev_valid_mask is not None:
                    fused_mask = prev_valid_mask.copy()
                    fused_area = int(np.sum(fused_mask > 0))

                if fused_area > 0:
                    prev_valid_mask = fused_mask.copy()
                    mask_resized = cv2.resize(fused_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    save_mask_and_frame(frame, mask_resized, str(output_path), idx,
                                        save_overlay=output_cfg.get("save_overlay", True),
                                        overlay_alpha=output_cfg.get("overlay_alpha", 0.5),
                                        save_frames=output_cfg.get("save_frames", False),
                                        save_composite=True)

                writer.writerow([
                    idx, stats['mean'], stats['max'], stats['min'], stats['adaptive_thresh'],
                    tsp_area, sam_area, pose_area, fused_area, max_region
                ])
                pbar.update(1)

    if dataset_mode != "davis":
        cap.release()
    print(f"Done. Results saved to: {output_path}")



if __name__ == "__main__":
    args = parse_args()
    run_tsp_sam(args.input_path, args.output_base_dir, args.config_path, force=args.force)

    
    