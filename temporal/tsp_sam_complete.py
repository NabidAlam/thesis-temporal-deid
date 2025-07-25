# ----------------------------------------------------------
# TSP-SAM + SAM + Pose Fusion Runner (MaskAnyone–Temporal)
# ----------------------------------------------------------
# Description:
# This script runs temporally-consistent video segmentation
# using TSP-SAM as the primary model, optionally fused with:
#   • SAM (via bounding box prompting)
#   • SAM2 (via OpenPose keypoints)
#
# It supports GDPR-compliant privacy masking with shape-aware,
# temporally-aware, and behavior-preserving mask refinement.
#
# Features:
# - TSP-SAM core segmentation with adaptive percentile thresholding
# - Optional SAM integration via TSP-generated bounding boxes
# - Optional SAM2 (pose-guided) mask generation via OpenPose prompts
# - Configurable fusion logic: "tsp_only", "union", "pose_only", "tsp+sam", "tsp+pose"
# - Dynamic pose fusion weighting based on keypoint area (TED- vs DAVIS-aware)
# - Warmup-based percentile thresholding for pose mask filtering
# - Aspect-ratio-aware pose threshold relaxation (e.g., TED talkers)
# - Smart fusion diagnostics (dominant source per frame)
#
# - Dataset-aware postprocessing:
#     • min_area, extent, solidity, aspect_ratio filtering
#     • extent-based suppression for thin artifacts
#     • automatic DAVIS threshold override
#     • resolution-scaled filtering
#     • mask hole filling + dilation
#
# - Temporal smoothing:
#     • Rolling mask memory (deque)
#     • reset_memory_every: control adaptive temporal refresh
#     • IOU drift checks with fallback to previous valid mask
#
# - Overlay generation:
#     • Edge-fade option for visual smoothness
#     • Composite overlay + debug region visualization
#
# - Debug logging:
#     • Frame-wise stats in `debug_stats.csv`
#     • Debug masks for each thresholding stage
#     • Pose mask skipped frame logs
#
# - Config-driven:
#     • Fully YAML-configurable (model, inference, fusion, postprocess, output)
#     • Dataset-specific config overrides (DAVIS vs TED)
#
# Output:
#     • Binary segmentation masks
#     • Optional overlay images
#     • Composite debug visualizations
#     • Frame-by-frame statistics in `debug_stats.csv`
#
# Usage:
# python temporal/tsp_sam_complete.py <input_path> <output_dir> <config.yaml> [--force]
#
# Example:
# python temporal/tsp_sam_complete.py input/davis2017/JPEGImages/480p/camel output/tsp_sam/davis configs/tsp_sam_davis.yaml --force
# python temporal/tsp_sam_complete.py input/ted/talk1.mp4 output/tsp_sam/ted configs/tsp_sam_ted.yaml --force


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


def fade_edges(mask, width=20):
    h, w = mask.shape
    fade = np.ones_like(mask, dtype=np.float32)
    fade[:width, :] *= np.linspace(0, 1, width)[:, None]
    fade[-width:, :] *= np.linspace(1, 0, width)[:, None]
    fade[:, :width] *= np.linspace(0, 1, width)[None, :]
    fade[:, -width:] *= np.linspace(1, 0, width)[None, :]
    return (mask.astype(np.float32) * fade).astype(np.uint8)


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


# def post_process_fused_mask(fused_mask, min_area=100, kernel_size=None):
# def post_process_fused_mask(fused_mask, min_area=1200, kernel_size=None):
#     if kernel_size is None:
#         kernel_size = get_dynamic_kernel_size(fused_mask.shape)
        
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
#     cleaned = cv2.morphologyEx(fused_mask, cv2.MORPH_OPEN, kernel)
    
#     closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
#     contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     filtered = np.zeros_like(closed)
#     for cnt in contours:
#         if cv2.contourArea(cnt) >= min_area:
#             cv2.drawContours(filtered, [cnt], -1, 255, -1)
#     return cv2.dilate(filtered, kernel, iterations=1)

def fill_mask_holes(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for i in range(len(contours)):
            if hierarchy[0][i][3] != -1:  # has parent => it's a hole
                cv2.drawContours(mask, contours, i, 255, -1)
    return mask



# def post_process_fused_mask(
#     fused_mask,
#     min_area=1200,
#     kernel_size=None,
#     dilation_iters=1,
#     return_debug=False,
#     large_area_thresh=50000,
#     min_extent=0.15,
#     prev_valid_mask=None
# ):
#     if kernel_size is None:
#         kernel_size = get_dynamic_kernel_size(fused_mask.shape)

#     # Ensure 0–255 uint8 binary
#     if fused_mask.max() == 1:
#         fused_mask = (fused_mask * 255).astype(np.uint8)
#     else:
#         fused_mask = fused_mask.astype(np.uint8)

#     # Morphological cleaning
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
#     opened = cv2.morphologyEx(fused_mask, cv2.MORPH_OPEN, kernel)
#     closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

#     # Component filtering
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
#     filtered = np.zeros_like(closed)
#     debug_vis = np.zeros((*fused_mask.shape, 3), dtype=np.uint8)

#     h_img, w_img = fused_mask.shape
#     border_margin = 10

#     for i in range(1, num_labels):  # skip background
#         x, y, w, h, area = stats[i, :5]
#         aspect_ratio = w / h if h > 0 else 0
#         extent = area / (w * h + 1e-6)
#         label_mask = (labels == i)

#         if return_debug:
#             print(f"[Region {i}] area={area}, aspect_ratio={aspect_ratio:.2f}, extent={extent:.2f}")

#         if area >= large_area_thresh:
#             filtered[label_mask] = 255
#             if return_debug:
#                 debug_vis[label_mask] = [0, 255, 0]  # green: very large
#         elif area >= min_area:
#             if aspect_ratio < 3.0 and extent > min_extent and h > 15 and w > 15:
#                 filtered[label_mask] = 255
#                 if return_debug:
#                     debug_vis[label_mask] = [255, 255, 255]  # white: normal pass
#             else:
#                 if return_debug:
#                     debug_vis[label_mask] = [0, 165, 255]  # orange: borderline
#         else:
#             if return_debug:
#                 debug_vis[label_mask] = [0, 0, 255]  # red: too small

#         # Optional area label
#         if return_debug:
#             cv2.putText(
#                 debug_vis, f"{area}", (x, y + 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
#             )

#     # Remove borders and bottom subtitles
#     filtered[:border_margin, :] = 0
#     filtered[-border_margin:, :] = 0
#     filtered[:, :border_margin] = 0
#     filtered[:, -border_margin:] = 0
#     filtered[int(0.9 * h_img):, :] = 0

#     # Fallback to previous valid mask if all rejected
#     # if np.sum(filtered) == 0 and prev_valid_mask is not None:
#     #     print("[Fallback] Using previous valid mask due to empty post-process")
#     #     filtered = prev_valid_mask.copy()
#     if np.sum(filtered) == 0:
#         print("[Warning] Empty postprocessed mask. Raw TSP or previous mask used.")
#         if prev_valid_mask is not None:
#             print("[Fallback] Using previous valid mask due to empty post-process")
#             filtered = prev_valid_mask.copy()
#         else:
#             print("[Fallback] All regions filtered. Using raw TSP mask.")
#             filtered = fused_mask.copy()


    
#     # Fill holes before final dilation
#     filtered = fill_mask_holes(filtered)

#     if dilation_iters > 0:
#         filtered = cv2.dilate(filtered, kernel, iterations=dilation_iters)

#     return (filtered, debug_vis) if return_debug else filtered

# def post_process_fused_mask(
#     fused_mask,
#     min_area=1200,
#     kernel_size=None,
#     dilation_iters=1,
#     return_debug=False,
#     large_area_thresh=50000,
#     min_extent=0.15,
#     prev_valid_mask=None,
#     config=None
# ):

#     import inspect
#     caller_locals = inspect.currentframe().f_back.f_locals
#     use_solidity_filter = caller_locals.get("post_cfg", {}).get("use_solidity_filter", False)
#     # use_solidity_filter = config.get("use_solidity_filter", False) if config else False


#     dataset_mode = caller_locals.get("dataset_mode", "ted").lower()

#     # Override for DAVIS to improve small-object recall
#     if dataset_mode == "davis":
#         print("[Override] Postprocess thresholds adjusted for DAVIS")
#         min_area = 400
#         min_extent = 0.03
#         dilation_iters = 1

#     if kernel_size is None:
#         kernel_size = get_dynamic_kernel_size(fused_mask.shape)

#     # Resolution-aware scaling of area thresholds (relative to 512x512)
#     h_img, w_img = fused_mask.shape
#     scale_factor = (h_img * w_img) / (512 * 512)
#     min_area = int(min_area * scale_factor)
#     large_area_thresh = int(large_area_thresh * scale_factor)

#     # Ensure 0–255 uint8 binary
#     if fused_mask.max() == 1:
#         fused_mask = (fused_mask * 255).astype(np.uint8)
#     else:
#         fused_mask = fused_mask.astype(np.uint8)

#     # Morphological cleaning
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
#     opened = cv2.morphologyEx(fused_mask, cv2.MORPH_OPEN, kernel)
#     closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

#     # Component filtering
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
#     filtered = np.zeros_like(closed)
#     debug_vis = np.zeros((*fused_mask.shape, 3), dtype=np.uint8)

#     # border_margin = 10
#     border_margin = max(10, int(0.02 * h_img))  # ~2% of height


#     for i in range(1, num_labels):  # skip background
#         x, y, w, h, area = stats[i, :5]
#         aspect_ratio = w / h if h > 0 else 0
#         extent = area / (w * h + 1e-6)
#         label_mask = (labels == i)
#         if use_solidity_filter:
#             cnt = np.array(np.where(label_mask)).T
#             cnt = np.flip(cnt, axis=1).reshape(-1, 1, 2).astype(np.int32)
#             if len(cnt) < 3:
#                 continue

#             hull = cv2.convexHull(cnt)
#             hull_area = cv2.contourArea(hull)
#             solidity = float(area) / (hull_area + 1e-6)

#             if solidity < 0.3 and extent < 0.5:
#                 if return_debug:
#                     print(f"[Region {i}] Skipped due to low solidity {solidity:.2f} and extent {extent:.2f}")
#                 continue



        
#         # cnt = np.array(np.where(label_mask)).T
#         # cnt = np.flip(cnt, axis=1).reshape(-1, 1, 2).astype(np.int32)
#         # if len(cnt) < 3:
#         #     continue
#         # hull = cv2.convexHull(cnt)
#         # hull_area = cv2.contourArea(hull)
#         # solidity = float(area) / (hull_area + 1e-6)
#         # if solidity < 0.3:
#         #     if return_debug:
#         #         print(f"[Region {i}] Skipped due to low solidity {solidity:.2f}")
#         #     continue

        
#          # Optional: Ignore high aspect ratio + low area regions as noise
#         if area < 100 and aspect_ratio > 4.0:
#             if return_debug:
#                 print(f"[Region {i}] Skipped due to high aspect ratio and low area")
#             continue

#         if return_debug:
#             print(f"[Region {i}] area={area}, aspect_ratio={aspect_ratio:.2f}, extent={extent:.2f}")

#         if area >= large_area_thresh:
#             filtered[label_mask] = 255
#             if return_debug:
#                 debug_vis[label_mask] = [0, 255, 0]  # green: very large
#         elif area >= min_area:
#             # Accept normal and thin blobs adaptively
#             if (
#                 (extent > min_extent and aspect_ratio < 3.0 and h > 5 and w > 5)
#                 or (extent > 0.015 and aspect_ratio > 2.0 and area >= min_area * 0.7)
#             ):
#                 filtered[label_mask] = 255
#                 if return_debug:
#                     debug_vis[label_mask] = [255, 255, 255]  # white: normal pass
#             else:
#                 if return_debug:
#                     debug_vis[label_mask] = [0, 165, 255]  # orange: borderline

#         # elif area >= min_area:
#         #     if aspect_ratio < 3.0 and extent > min_extent and h > 5 and w > 5:
#         #         filtered[label_mask] = 255
#         #         if return_debug:
#         #             debug_vis[label_mask] = [255, 255, 255]  # white: normal pass
#         #     else:
#         #         if return_debug:
#         #             debug_vis[label_mask] = [0, 165, 255]  # orange: borderline
#         else:
#             if return_debug:
#                 debug_vis[label_mask] = [0, 0, 255]  # red: too small

#         # Optional area label
#         if return_debug:
#             cv2.putText(
#                 debug_vis, f"{area}", (x, y + 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
#             )
            
#     if return_debug:
#         print(f"[Summary] {np.sum(filtered > 0)} pixels kept across {np.max(labels)} regions")


#     # Remove borders and bottom subtitles
#     filtered[:border_margin, :] = 0
#     filtered[-border_margin:, :] = 0
#     filtered[:, :border_margin] = 0
#     filtered[:, -border_margin:] = 0
#     filtered[int(0.9 * h_img):, :] = 0

#     # Fallback to previous valid mask if all rejected
#     if np.sum(filtered) == 0:
#         print("[Warning] Empty postprocessed mask. Raw TSP or previous mask used.")
#         if prev_valid_mask is not None:
#             print("[Fallback] Using previous valid mask due to empty post-process")
#             filtered = prev_valid_mask.copy()
#         else:
#             print("[Fallback] All regions filtered. Using raw TSP mask.")
#             filtered = fused_mask.copy()

#     # Fill holes before final dilation
#     filtered = fill_mask_holes(filtered)

#     if dilation_iters > 0:
#         filtered = cv2.dilate(filtered, kernel, iterations=dilation_iters)

#     return (filtered, debug_vis) if return_debug else filtered

def post_process_fused_mask(
    fused_mask,
    min_area=1200,
    kernel_size=None,
    dilation_iters=1,
    return_debug=False,
    large_area_thresh=50000,
    min_extent=0.15,
    prev_valid_mask=None,
    max_aspect_ratio=6.0,
    min_solidity=0.25,
    border_margin_pct=0.03,
    use_extent_suppression=True
):
    import inspect
    caller_locals = inspect.currentframe().f_back.f_locals
    post_cfg = caller_locals.get("post_cfg", {})
    use_solidity_filter = post_cfg.get("use_solidity_filter", False)

    dataset_mode = caller_locals.get("dataset_mode", "ted").lower()

    if dataset_mode == "davis":
        print("[Override] Postprocess thresholds adjusted for DAVIS")
        min_area = 400
        min_extent = 0.03
        dilation_iters = 1

    if kernel_size is None:
        kernel_size = get_dynamic_kernel_size(fused_mask.shape)

    h_img, w_img = fused_mask.shape
    scale_factor = (h_img * w_img) / (512 * 512)
    min_area = int(min_area * scale_factor)
    large_area_thresh = int(large_area_thresh * scale_factor)

    if fused_mask.max() == 1:
        fused_mask = (fused_mask * 255).astype(np.uint8)
    else:
        fused_mask = fused_mask.astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(fused_mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    filtered = np.zeros_like(closed)
    debug_vis = np.zeros((*fused_mask.shape, 3), dtype=np.uint8)

    border_margin = max(10, int(border_margin_pct * h_img))

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i, :5]
        aspect_ratio = w / h if h > 0 else 0
        extent = area / (w * h + 1e-6)
        label_mask = (labels == i)

        # Aspect ratio + low-area suppression
        if area < 100 and aspect_ratio > 4.0:
            if return_debug:
                print(f"[Region {i}] Skipped: high aspect ratio and low area")
            continue

        if aspect_ratio > max_aspect_ratio:
            if return_debug:
                print(f"[Region {i}] Skipped: aspect_ratio={aspect_ratio:.2f} exceeds max {max_aspect_ratio}")
            continue

        if use_solidity_filter:
            cnt = np.array(np.where(label_mask)).T
            cnt = np.flip(cnt, axis=1).reshape(-1, 1, 2).astype(np.int32)
            if len(cnt) < 3:
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / (hull_area + 1e-6)
            if solidity < min_solidity:
                if return_debug:
                    print(f"[Region {i}] Skipped: solidity={solidity:.2f} below min {min_solidity}")
                continue

        if use_extent_suppression and extent < 0.1 and area < 1.5 * min_area:
            if return_debug:
                print(f"[Region {i}] Skipped: tight extent suppression (extent={extent:.2f})")
            continue

        if return_debug:
            print(f"[Region {i}] area={area}, aspect_ratio={aspect_ratio:.2f}, extent={extent:.2f}")


    
        if area >= large_area_thresh:
            filtered[label_mask] = 255
            if return_debug:
                debug_vis[label_mask] = [0, 255, 0]  # green
        elif area >= min_area:
            if (
                (extent > min_extent and aspect_ratio < 3.0 and h > 5 and w > 5) or
                (extent > 0.015 and aspect_ratio > 2.0 and area >= 0.7 * min_area)
            ):
                filtered[label_mask] = 255
                if return_debug:
                    debug_vis[label_mask] = [255, 255, 255]  # white
            else:
                if return_debug:
                    debug_vis[label_mask] = [0, 165, 255]  # orange
        else:
            if return_debug:
                debug_vis[label_mask] = [0, 0, 255]  # red

        if return_debug:
            cv2.putText(
                debug_vis, f"{area}", (x, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
            )

    if return_debug:
        print(f"[Summary] {np.sum(filtered > 0)} pixels kept across {np.max(labels)} regions")

    # Edge trimming
    filtered[:border_margin, :] = 0
    filtered[-border_margin:, :] = 0
    filtered[:, :border_margin] = 0
    filtered[:, -border_margin:] = 0
    filtered[int(0.9 * h_img):, :] = 0

    if np.sum(filtered) == 0:
        print("[Warning] Empty postprocessed mask. Raw TSP or previous mask used.")
        if prev_valid_mask is not None:
            print("[Fallback] Using previous valid mask due to empty post-process")
            filtered = prev_valid_mask.copy()
        else:
            print("[Fallback] All regions filtered. Using raw TSP mask.")
            filtered = fused_mask.copy()

    filtered = fill_mask_holes(filtered)

    if dilation_iters > 0:
        filtered = cv2.dilate(filtered, kernel, iterations=dilation_iters)

    return (filtered, debug_vis) if return_debug else filtered



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

from glob import glob

def load_davis_annotations(input_root, seq_name):
    """
    Load DAVIS ground-truth segmentation masks.

    Args:
        input_root (str or Path): Root directory of DAVIS dataset.
        seq_name (str): Name of the sequence (e.g., "dog", "camel").

    Returns:
        Dict[str, np.ndarray]: Mapping from frame name (e.g., "00000") to grayscale annotation mask.
    """
    anno_dir = Path(input_root) / "Annotations" / "480p" / seq_name
    mask_paths = sorted(anno_dir.glob("*.png"))

    # Use filename stem (e.g., "00000") as key
    masks = {p.stem: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in mask_paths}

    return masks


def run_tsp_sam(input_path, output_path_base, config_path, force=False):
    import shutil
    from collections import deque

    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        #newly added
        post_cfg = config.get("postprocess", {})
        use_edge_fade = post_cfg.get("edge_fade", False)

        post_min_area = post_cfg.get("min_area", 1200)
        post_large_thresh = post_cfg.get("large_area_threshold", 50000)
        post_min_extent = post_cfg.get("min_extent", 0.15)
        post_dilation_iters = post_cfg.get("dilation_iters", 1)
        post_max_aspect_ratio = post_cfg.get("max_aspect_ratio", 6.0)
        post_min_solidity = post_cfg.get("min_solidity", 0.25)
        post_border_margin_pct = post_cfg.get("border_margin", 0.03)
        post_extent_suppression = post_cfg.get("tight_extent_check", True)

        
        
    model_cfg = config["model"]
    infer_cfg = config["inference"]
    output_cfg = config["output"]
    dataset_mode = config.get("dataset", {}).get("mode", "ted").lower()
    
        # --- Dataset-specific post-processing tweaks ---
    if dataset_mode == "davis":
        print("[CONFIG OVERRIDE] Relaxing post-process for DAVIS")
        post_min_area = 400
        post_min_extent = 0.05
        post_dilation_iters = 1
        if "reset_memory_every" not in infer_cfg:
            infer_cfg["reset_memory_every"] = 8
            print("[CONFIG OVERRIDE] Setting reset_memory_every = 8 for DAVIS")

        
        

    fusion_method = config.get("fusion_method", "tsp_only")

    fusion_cfg = config.get("fusion", {})
    enable_sam = fusion_cfg.get("enable_sam", True)
    enable_pose = fusion_cfg.get("enable_pose", True)

    use_sam = enable_sam and fusion_method in ("sam_only", "union", "tsp+sam")
    # use_pose = enable_pose and dataset_mode != "davis" and fusion_method in ("pose_only", "union", "tsp+pose")
    # use_pose = enable_pose and fusion_method in ("pose_only", "union", "tsp+pose")
    use_pose = enable_pose and fusion_method in ("pose_only", "union", "tsp+pose") and dataset_mode != "davis"


    temporal_smoothing = infer_cfg.get("temporal_smoothing", False)
    reset_memory_every = infer_cfg.get("reset_memory_every", None)
    
    print(f"[CONFIG] Dataset: {dataset_mode.upper()} | Fusion: {fusion_method} | use_sam={use_sam} | use_pose={use_pose} | temporal_smoothing={temporal_smoothing}")



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
    
        
    # Load DAVIS GT annotations if in DAVIS mode
    seq_name = Path(input_path).stem
    gt_masks = {}

    if dataset_mode == "davis":
        gt_masks = load_davis_annotations("input/davis2017", seq_name)
        print(f"[DEBUG] Loaded {len(gt_masks)} ground-truth annotation frames for DAVIS sequence: '{seq_name}'")



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

        # mask_memory = deque(maxlen=5)
        temporal_window = 10 if dataset_mode == "ted" else 5
        mask_memory = deque(maxlen=temporal_window)


        prev_valid_mask = None

        with tqdm(total=total_frames // frame_stride) as pbar:
            for idx, frame_data in frame_iter:
                
                if temporal_smoothing and reset_memory_every and idx % reset_memory_every == 0:
                    print(f"[DEBUG] Resetting memory at frame {idx}")
                    mask_memory.clear()
                
                if idx % frame_stride != 0:
                    continue

                if dataset_mode == "davis":
                    frame = cv2.imread(str(frame_data))
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break

                frame_resized = resize_frame(frame, infer_cfg)
                print(f"[Model Infer] Running TSP-SAM on frame {idx}")

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
                            print(f"[Pose] Frame {idx}: SAM2 generated pose mask, area = {np.sum(pose_mask > 0)}")



                pose_area = int(np.sum(pose_mask > 0))
                sam_area = int(np.sum(sam_mask > 0))
                tsp_area = int(np.sum(tsp_mask > 0))
                
                print(f"[DEBUG] Frame {idx}: tsp_area={tsp_area}, sam_area={sam_area}, pose_area={pose_area}")
                
                dominant = "pose" if pose_area > sam_area and pose_area > tsp_area else \
                           "sam" if sam_area > tsp_area else "tsp"
                print(f"Frame {idx}: Dominant source = {dominant}")


                # if dataset_mode != "davis" and use_pose:
                #     h, w = frame.shape[:2]
                #     aspect_ratio = h / w
                #     relaxed_thresh = dynamic_pose_thresh * 0.5 if aspect_ratio > 1.4 else dynamic_pose_thresh
                #     # if pose_area < relaxed_thresh:
                #     #     print(f"Frame {idx}: Pose area {pose_area} < relaxed threshold {relaxed_thresh}, skipping. Dominant: {dominant}")
                #     #     # print(f"Frame {idx}: Pose area {pose_area} < relaxed threshold {relaxed_thresh}, skipping.")
                        
                #     #     pbar.update(1)
                #     #     continue
                    
                #     if pose_area < relaxed_thresh:
                #         print(f"Frame {idx}: Pose area {pose_area} < relaxed threshold {relaxed_thresh}, skipping. Dominant: {dominant}")
                        
                #         # Save the skipped pose mask for inspection
                #         skipped_pose_dir = output_path / "pose_skipped"
                #         skipped_pose_dir.mkdir(exist_ok=True)
                        
                #         debug_pose = cv2.resize(pose_mask, (frame.shape[1], frame.shape[0]))
                #         cv2.imwrite(str(skipped_pose_dir / f"{idx:05d}_pose_skipped.png"), debug_pose)

                #         pbar.update(1)
                #         continue
                
                if use_pose:
                    h, w = frame.shape[:2]
                    aspect_ratio = h / w

                    # Adjust threshold based on dataset
                    if dataset_mode == "davis":
                        relaxed_thresh = dynamic_pose_thresh * 0.3  # Looser threshold for DAVIS (pose weaker)
                    else:
                        relaxed_thresh = dynamic_pose_thresh * 0.5 if aspect_ratio > 1.4 else dynamic_pose_thresh

                    if pose_area < relaxed_thresh:
                        print(f"Frame {idx}: Pose area {pose_area} < relaxed threshold {relaxed_thresh}, skipping. Dominant: {dominant}")
                        
                        # Save the skipped pose mask for inspection
                        skipped_pose_dir = output_path / "pose_skipped"
                        skipped_pose_dir.mkdir(exist_ok=True)

                        debug_pose = cv2.resize(pose_mask, (frame.shape[1], frame.shape[0]))
                        cv2.imwrite(str(skipped_pose_dir / f"{idx:05d}_pose_skipped.png"), debug_pose)

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
                # elif fusion_method == "tsp+pose":
                #     fused_mask = cv2.bitwise_or(tsp_mask, pose_mask)
                elif fusion_method == "tsp+pose":
                    pose_area = np.sum(pose_mask > 0)
                    relaxed_thresh = dynamic_pose_thresh * 1.2
                    pose_weight = 0.5 if pose_area < relaxed_thresh else 1.0
                    pose_weighted = (pose_mask.astype(np.float32) * pose_weight).astype(np.uint8)
                    fused_mask = cv2.bitwise_or(tsp_mask, pose_weighted)
                # Save intermediate masks for debugging
                cv2.imwrite(str(output_path / f"{idx:05d}_tsp_mask.png"), tsp_mask)
                cv2.imwrite(str(output_path / f"{idx:05d}_sam_mask.png"), sam_mask)



                    


                # if temporal_smoothing:
                #     if idx < 5:
                #         # Warmup memory with first few masks to stabilize early temporal smoothing
                #         for _ in range(5):
                #             mask_memory.append(fused_mask.astype(np.float32))
                #     else:
                #         mask_memory.append(fused_mask.astype(np.float32))

                #     # Apply temporal smoothing
                #     smoothed_mask = np.mean(mask_memory, axis=0)
                #     fused_mask = (smoothed_mask > 127).astype(np.uint8) * 255
                    
                if temporal_smoothing:
                    if idx < mask_memory.maxlen:
                        # Pre-fill memory to avoid cold start flicker
                        for _ in range(mask_memory.maxlen - idx):
                            mask_memory.append(fused_mask.astype(np.float32))
                    mask_memory.append(fused_mask.astype(np.float32))
                    smoothed_mask = np.mean(mask_memory, axis=0)
                    fused_mask = (smoothed_mask > 127).astype(np.uint8) * 255
                    print(f"[Temporal Smoothing] Applied on frame {idx}, memory size: {len(mask_memory)}")



                # # Optional Temporal Smoothing
                # if temporal_smoothing:
                #     mask_memory.append(fused_mask.astype(np.float32))
                #     smoothed_mask = np.mean(mask_memory, axis=0)
                #     fused_mask = (smoothed_mask > 127).astype(np.uint8) * 255

                # fused_mask = post_process_fused_mask(fused_mask, min_area=min_area)
                
                # UPDATED: Post-process with dataset-aware logic
                # fused_mask = post_process_fused_mask(
                #     fused_mask,
                #     min_area=min_area,
                #     dataset=dataset_mode
                # )
                
                # fused_mask, debug_vis = post_process_fused_mask(
                #     fused_mask,
                #     min_area=post_min_area,
                #     dilation_iters=post_dilation_iters,
                #     return_debug=True,
                #     large_area_thresh=post_large_thresh,
                #     min_extent=post_min_extent,
                #     prev_valid_mask=prev_valid_mask
                # )
                
                print(f"[Fusion Debug] Frame {idx}: fusion_method={fusion_method}, tsp_area={tsp_area}, sam_area={sam_area}, pose_area={pose_area}")

                
                fused_mask, debug_vis = post_process_fused_mask(
                    fused_mask,
                    min_area=post_min_area,
                    dilation_iters=post_dilation_iters,
                    return_debug=True,
                    large_area_thresh=post_large_thresh,
                    min_extent=post_min_extent,
                    prev_valid_mask=prev_valid_mask,
                    max_aspect_ratio=post_max_aspect_ratio,
                    min_solidity=post_min_solidity,
                    border_margin_pct=post_border_margin_pct,
                    use_extent_suppression=post_extent_suppression
                )
                
  

                cv2.imwrite(str(output_path / f"{idx:05d}_debug_post.png"), debug_vis)

                fused_area = int(np.sum(fused_mask > 0))
                
                iou = -1  # default if not computable
                if dataset_mode == "davis":
                    frame_name = Path(frame_data).stem
                    if frame_name in gt_masks:
                        gt = (gt_masks[frame_name] > 0).astype(np.uint8)
                        pred_resized = cv2.resize(fused_mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                        pred = (pred_resized > 0).astype(np.uint8)
                        intersection = np.logical_and(gt, pred).sum()
                        union = np.logical_or(gt, pred).sum()
                        iou = intersection / (union + 1e-6)
                        print(f"[IoU] Frame {frame_name}: IoU with GT = {iou:.3f}")
                    else:
                        print(f"[IoU] Frame {frame_name}: No ground-truth annotation available")

                        
                # IOU drift warning (before potential fallback, only if enabled)
                if post_cfg.get("iou_check", False) and prev_valid_mask is not None:
                    iou = np.sum((fused_mask > 0) & (prev_valid_mask > 0)) / (np.sum((fused_mask > 0) | (prev_valid_mask > 0)) + 1e-6)
                    if iou < 0.3:
                        print(f"[Drift Warning] Frame {idx}: IOU with previous = {iou:.2f}")

                # Fallback to previous valid mask if current is empty
                if fused_area == 0 and prev_valid_mask is not None:
                    fused_mask = prev_valid_mask.copy()
                    fused_area = int(np.sum(fused_mask > 0))

                # Connected component stats
                num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(fused_mask)
                num_regions = num_labels - 1  # exclude background
                print(f"Frame {idx}: Connected regions = {num_regions}")
                max_region = 0 if len(stats_cc) <= 1 else np.max(stats_cc[1:, cv2.CC_STAT_AREA])
                print(f"[Output] Frame {idx}: Mask saved, fused_area={fused_area}, max_region={max_region}, dominant={dominant}")


                # Save outputs only if mask is valid
                if fused_area > 0:
                    prev_valid_mask = fused_mask.copy()
                    mask_resized = cv2.resize(fused_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                    # Save binary mask
                    cv2.imwrite(str(output_path / f"{idx:05d}_mask.png"), mask_resized)

                    # Create overlay mask with optional edge fading
                    if use_edge_fade:
                        overlay_mask = fade_edges(mask_resized)
                    else:
                        overlay_mask = mask_resized.copy()
                    overlay_mask = cv2.GaussianBlur(overlay_mask, (7, 7), 0)

                    save_mask_and_frame(
                        frame,
                        overlay_mask,
                        str(output_path),
                        idx,
                        save_overlay=output_cfg.get("save_overlay", True),
                        overlay_alpha=output_cfg.get("overlay_alpha", 0.5),
                        save_frames=output_cfg.get("save_frames", False),
                        save_composite=True
                    )
                    
                    print(f"[Output] Frame {idx}: Mask saved, fused_area={fused_area}, max_region={max_region}, dominant={dominant}")



                # Log stats to CSV
                # writer.writerow([
                #     idx, stats['mean'], stats['max'], stats['min'], stats['adaptive_thresh'],
                #     tsp_area, sam_area, pose_area, fused_area, max_region
                # ])
                
                writer.writerow([
                    idx, stats['mean'], stats['max'], stats['min'], stats['adaptive_thresh'],
                    tsp_area, sam_area, pose_area, fused_area, max_region, round(iou, 4)
                ])

                pbar.update(1)


    if dataset_mode != "davis":
        cap.release()
    print(f"Done. Results saved to: {output_path}")



if __name__ == "__main__":
    args = parse_args()
    run_tsp_sam(args.input_path, args.output_base_dir, args.config_path, force=args.force)

    
    