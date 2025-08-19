# ----------------------------------------------------------
# TSP-SAM + SAM + Pose Fusion Runner (MaskAnyone–Temporal)
# ----------------------------------------------------------
# Description:
# This script performs temporally-consistent video segmentation
# using TSP-SAM as the core model, with optional fusion from:
#   • SAM   — via TSP-derived bounding box prompts
#   • SAM2  — via OpenPose keypoint-based prompts
#
# Designed for GDPR-compliant de-identification of behavioral video,
# it balances privacy protection with utility preservation (e.g., gestures).
#
# Features:
# - TSP-SAM core segmentation with adaptive percentile thresholding
# - Optional SAM integration via bounding box prompting
# - Optional SAM2 (pose-guided) fusion via OpenPose keypoints
# - Fusion logic configurable: "tsp_only", "union", "pose_only", "tsp+sam", "tsp+pose"
# - Dynamic pose mask weighting based on warmup-estimated area thresholds
# - Relaxed pose thresholds for TED (talker detection), stricter for DAVIS
# - Pose mask rejection logging and visual output
#
# Postprocessing:
# - Morphological open/close, hole filling, optional dilation
# - Contour filtering: min_area, extent, aspect_ratio, solidity
# - Optional tight extent suppression for streaky noise
# - Resolution-aware and dataset-aware parameter scaling
# - Border + subtitle crop (TED) and DAVIS-specific threshold overrides
#
# Temporal Smoothing:
# - Rolling mask memory using a fixed-size deque
# - reset_memory_every to periodically clear memory in dynamic scenes
# - IOU-based drift detection and fallback to last valid mask if necessary
#
# Debug and Output:
# - Frame-by-frame logs written to `debug_stats.csv` (incl. IoU, mask areas, region size)
# - Overlay and composite debug visualizations
# - Per-frame debug masks (adaptive thresholding, post-filter diagnostics)
# - Ground-truth comparison and IoU stats for DAVIS dataset
#
# Configuration:
# - YAML-driven pipeline: model, inference, fusion, postprocess, output
# - Dataset-specific config overrides for TED and DAVIS modes
#
# Output:
# - Binary segmentation masks (.png)
# - Optional overlay masks (edge-faded)
# - Composite debug images
# - Frame-wise stats in `debug_stats.csv`
#
# Usage:
# python temporal/tsp_sam_complete.py <input_path> <output_dir> <config.yaml> [--force]
#
# Examples:
# python temporal/tsp_sam_complete.py input/davis2017/JPEGImages/480p/camel output/tsp_sam/davis configs/tsp_sam_davis.yaml --force
# python temporal/tsp_sam_complete.py input/ted/video2.mp4 output/tsp_sam/ted configs/tsp_sam_ted.yaml --force



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

# tsp sam, understand temporal segmentaion and memory drift   
# samurai evaluate prompt tracking with kalman memory
# reproduce these two baselines
# maskanyone, contrast agains frame only performance
# apply tspsam to davis first
# apply tsp sam and samurai to ted and team ten
# camera cuts, partial visibility pose absense
# find out failures and try to fix them 

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
# sys.path.append(os.path.abspath("tsp_sam"))
sys.path.append(os.path.abspath("tsp_sam_official"))
sys.path.append(os.path.abspath("temporal"))

# from tsp_sam.lib.pvtv2_afterTEM import Network
from tsp_sam_official.lib.pvtv2_afterTEM import Network
# from TSP_SPAM.lib.pvtv2_afterTEM import Network

from utils import save_mask_and_frame, resize_frame
from maskanyone_sam_wrapper import MaskAnyoneSAMWrapper
# from pose_extractor import extract_pose_keypoints  # Temporarily disabled due to MediaPipe compatibility
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

def fill_mask_holes(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for i in range(len(contours)):
            if hierarchy[0][i][3] != -1:  # has parent => it's a hole
                cv2.drawContours(mask, contours, i, 255, -1)
    return mask


def compute_temporal_consistency_loss(current_mask, previous_mask):
    """
    Compute temporal consistency between consecutive frames.
    
    Args:
        current_mask: Current frame mask (binary)
        previous_mask: Previous frame mask (binary)
    
    Returns:
        float: Temporal consistency score (0.0 = no consistency, 1.0 = perfect consistency)
    """
    if previous_mask is None:
        return 1.0  # First frame has perfect consistency
    
    # Ensure masks are binary
    current_binary = (current_mask > 0).astype(np.uint8)
    previous_binary = (previous_mask > 0).astype(np.uint8)
    
    # IoU-based consistency
    intersection = np.logical_and(current_binary, previous_binary).sum()
    union = np.logical_or(current_binary, previous_binary).sum()
    iou = intersection / (union + 1e-6)
    
    # Area change penalty (penalize large area changes)
    current_area = current_binary.sum()
    previous_area = previous_binary.sum()
    area_ratio = min(current_area, previous_area) / max(current_area, previous_area) if max(current_area, previous_area) > 0 else 1.0
    
    # Centroid distance penalty (penalize large centroid shifts)
    if current_area > 0 and previous_area > 0:
        current_centroid = np.mean(np.where(current_binary > 0), axis=1)
        previous_centroid = np.mean(np.where(previous_binary > 0), axis=1)
        centroid_distance = np.linalg.norm(current_centroid - previous_centroid)
        max_distance = np.sqrt(current_binary.shape[0]**2 + current_binary.shape[1]**2)
        centroid_penalty = 1.0 - (centroid_distance / max_distance)
    else:
        centroid_penalty = 1.0
    
    # Combined consistency score
    consistency_score = 0.5 * iou + 0.3 * area_ratio + 0.2 * centroid_penalty
    
    return max(0.0, min(1.0, consistency_score))


def compute_motion_consistency(current_mask, previous_mask, flow_threshold=5.0):
    """
    Compute motion consistency using optical flow between masks.
    
    Args:
        current_mask: Current frame mask
        previous_mask: Previous frame mask
        flow_threshold: Threshold for motion magnitude
    
    Returns:
        float: Motion consistency score
    """
    if previous_mask is None:
        return 1.0
    
    try:
        # Convert masks to grayscale for optical flow
        prev_gray = previous_mask.astype(np.uint8)
        curr_gray = current_mask.astype(np.uint8)
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                          pyr_scale=0.5, levels=3, winsize=15, 
                                          iterations=3, poly_n=5, poly_sigma=1.2, 
                                          flags=0)
        
        # Compute motion magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Motion consistency: lower motion = higher consistency
        avg_motion = np.mean(magnitude)
        motion_consistency = max(0.0, 1.0 - (avg_motion / flow_threshold))
        
        return motion_consistency
        
    except Exception as e:
        print(f"[WARNING] Optical flow computation failed: {e}")
        return 1.0  # Fallback to perfect consistency


def temporal_mask_smoothing(mask_sequence, window_size=3):
    """
    Apply temporal smoothing to a sequence of masks.
    
    Args:
        mask_sequence: List of masks to smooth
        window_size: Size of temporal window for smoothing
    
    Returns:
        List: Smoothed masks
    """
    if len(mask_sequence) < window_size:
        return mask_sequence
    
    smoothed_masks = []
    
    for i in range(len(mask_sequence)):
        # Get temporal window
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(mask_sequence), i + window_size // 2 + 1)
        window_masks = mask_sequence[start_idx:end_idx]
        
        # Temporal averaging with consistency weighting
        if len(window_masks) > 1:
            # Weight masks by their temporal distance from current frame
            weights = []
            for j, mask in enumerate(window_masks):
                distance = abs(i - (start_idx + j))
                weight = 1.0 / (1.0 + distance)
                weights.append(weight)
            
            # Normalize weights
            weights = np.array(weights) / np.sum(weights)
            
            # Weighted average
            smoothed_mask = np.zeros_like(mask_sequence[i], dtype=np.float32)
            for j, (mask, weight) in enumerate(zip(window_masks, weights)):
                smoothed_mask += weight * mask.astype(np.float32)
            
            # Convert back to binary
            smoothed_mask = (smoothed_mask > 127).astype(np.uint8) * 255
        else:
            smoothed_mask = mask_sequence[i]
        
        smoothed_masks.append(smoothed_mask)
    
    return smoothed_masks


def compute_mask_confidence(mask, mask_type="tsp", previous_mask=None, temporal_context=None):
    """
    Compute confidence score for a mask based on multiple quality metrics.
    
    Args:
        mask: The mask to evaluate
        mask_type: Type of mask ("tsp", "sam", "pose")
        previous_mask: Previous frame mask for temporal consistency
        temporal_context: Additional temporal context information
    
    Returns:
        float: Confidence score (0.0 = low confidence, 1.0 = high confidence)
    """
    if mask is None or np.sum(mask > 0) == 0:
        return 0.0
    
    # Ensure mask is binary for analysis
    binary_mask = (mask > 0).astype(np.uint8)
    
    # 1. Area-based confidence (normalized by image size)
    total_pixels = mask.shape[0] * mask.shape[1]
    mask_area = np.sum(binary_mask)
    area_confidence = min(1.0, mask_area / (total_pixels * 0.3))  # Cap at 30% of image
    
    # 2. Shape quality confidence
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Perimeter-to-area ratio (lower is better for compact shapes)
    perimeter = cv2.arcLength(largest_contour, True)
    area = cv2.contourArea(largest_contour)
    if area > 0:
        compactness = 4 * np.pi * area / (perimeter * perimeter)
        shape_confidence = min(1.0, compactness * 2)  # Scale to 0-1
    else:
        shape_confidence = 0.0
    
    # 3. Boundary smoothness confidence
    # Use morphological operations to measure boundary roughness
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    
    boundary_pixels = np.logical_xor(binary_mask, eroded).sum()
    total_boundary = np.logical_xor(binary_mask, dilated).sum()
    
    if total_boundary > 0:
        smoothness = 1.0 - (boundary_pixels / total_boundary)
        boundary_confidence = max(0.0, smoothness)
    else:
        boundary_confidence = 1.0
    
    # 4. Temporal consistency confidence (if previous mask available)
    temporal_confidence = 1.0
    if previous_mask is not None:
        temporal_confidence = compute_temporal_consistency_loss(mask, previous_mask)
    
    # 5. Mask-type specific confidence adjustments
    type_confidence = 1.0
    if mask_type == "tsp":
        # TSP-SAM: High confidence for primary segmentation
        type_confidence = 1.0
    elif mask_type == "sam":
        # SAM: Slightly lower confidence (auxiliary)
        type_confidence = 0.9
    elif mask_type == "pose":
        # Pose: Lower confidence due to potential noise
        type_confidence = 0.8
    
    # 6. Connected component confidence
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
    if num_labels > 1:  # More than just background
        # Prefer masks with fewer, larger components
        component_sizes = stats[1:, cv2.CC_STAT_AREA]
        largest_component_ratio = np.max(component_sizes) / mask_area if mask_area > 0 else 0
        component_confidence = largest_component_ratio
    else:
        component_confidence = 0.0
    
    # Combine all confidence scores with weights
    confidence_weights = {
        'area': 0.25,
        'shape': 0.20,
        'boundary': 0.15,
        'temporal': 0.25,
        'type': 0.10,
        'component': 0.05
    }
    
    total_confidence = (
        confidence_weights['area'] * area_confidence +
        confidence_weights['shape'] * shape_confidence +
        confidence_weights['boundary'] * boundary_confidence +
        confidence_weights['temporal'] * temporal_confidence +
        confidence_weights['type'] * type_confidence +
        confidence_weights['component'] * component_confidence
    )
    
    return max(0.0, min(1.0, total_confidence))


def intelligent_fusion(tsp_mask, sam_mask, pose_mask, previous_mask, fusion_method="adaptive"):
    """
    Intelligently fuse masks based on confidence scores and temporal consistency.
    
    Args:
        tsp_mask: TSP-SAM generated mask
        sam_mask: SAM generated mask
        pose_mask: Pose-based mask
        previous_mask: Previous frame mask for temporal context
        fusion_method: Fusion strategy ("adaptive", "confidence", "temporal", "hybrid")
    
    Returns:
        tuple: (fused_mask, fusion_weights, fusion_method_used)
    """
    # Compute confidence scores for each mask
    tsp_conf = compute_mask_confidence(tsp_mask, "tsp", previous_mask)
    sam_conf = compute_mask_confidence(sam_mask, "sam", previous_mask)
    pose_conf = compute_mask_confidence(pose_mask, "pose", previous_mask)
    
    print(f"[Fusion] Confidence scores - TSP: {tsp_conf:.3f}, SAM: {sam_conf:.3f}, Pose: {pose_conf:.3f}")
    
    if fusion_method == "adaptive":
        # Adaptive fusion: choose best method based on confidence scores
        max_conf = max(tsp_conf, sam_conf, pose_conf)
        
        if max_conf < 0.3:
            # Low confidence: use temporal smoothing
            fusion_method_used = "temporal_fallback"
            if previous_mask is not None:
                fused_mask = previous_mask.copy()
                fusion_weights = {"tsp": 0.0, "sam": 0.0, "pose": 0.0, "previous": 1.0}
            else:
                # No previous mask, use TSP as fallback
                fused_mask = tsp_mask.copy()
                fusion_weights = {"tsp": 1.0, "sam": 0.0, "pose": 0.0, "previous": 0.0}
        
        elif max_conf < 0.6:
            # Medium confidence: use weighted fusion
            fusion_method_used = "weighted_fusion"
            # Normalize confidence scores
            total_conf = tsp_conf + sam_conf + pose_conf
            if total_conf > 0:
                tsp_weight = tsp_conf / total_conf
                sam_weight = sam_conf / total_conf
                pose_weight = pose_conf / total_conf
            else:
                tsp_weight, sam_weight, pose_weight = 1.0, 0.0, 0.0
            
            # Weighted combination
            fused_mask = np.zeros_like(tsp_mask, dtype=np.float32)
            if tsp_weight > 0:
                fused_mask += tsp_weight * tsp_mask.astype(np.float32)
            if sam_weight > 0:
                fused_mask += sam_weight * sam_mask.astype(np.float32)
            if pose_weight > 0:
                fused_mask += pose_weight * pose_mask.astype(np.float32)
            
            fused_mask = (fused_mask > 127).astype(np.uint8) * 255
            fusion_weights = {"tsp": tsp_weight, "sam": sam_weight, "pose": pose_weight, "previous": 0.0}
        
        else:
            # High confidence: use best mask with minor blending
            fusion_method_used = "best_with_blending"
            if tsp_conf >= sam_conf and tsp_conf >= pose_conf:
                best_mask = tsp_mask
                best_conf = tsp_conf
                second_mask = sam_mask if sam_conf > pose_conf else pose_mask
                second_conf = max(sam_conf, pose_conf)
            elif sam_conf >= pose_conf:
                best_mask = sam_mask
                best_conf = sam_conf
                second_mask = tsp_mask
                second_conf = tsp_conf
            else:
                best_mask = pose_mask
                best_conf = pose_conf
                second_mask = tsp_mask
                second_conf = tsp_conf
            
            # Blend best mask with second best for stability
            blend_factor = 0.8
            fused_mask = (blend_factor * best_mask + (1 - blend_factor) * second_mask).astype(np.uint8)
            
            # Set weights based on which masks were used
            fusion_weights = {"tsp": 0.0, "sam": 0.0, "pose": 0.0, "previous": 0.0}
            if best_mask is tsp_mask:
                fusion_weights["tsp"] = blend_factor
                if second_mask is sam_mask:
                    fusion_weights["sam"] = 1 - blend_factor
                else:
                    fusion_weights["pose"] = 1 - blend_factor
            elif best_mask is sam_mask:
                fusion_weights["sam"] = blend_factor
                if second_mask is tsp_mask:
                    fusion_weights["tsp"] = 1 - blend_factor
                else:
                    fusion_weights["pose"] = 1 - blend_factor
            else:  # pose_mask
                fusion_weights["pose"] = blend_factor
                if second_mask is tsp_mask:
                    fusion_weights["tsp"] = 1 - blend_factor
                else:
                    fusion_weights["sam"] = 1 - blend_factor
    
    elif fusion_method == "confidence":
        # Pure confidence-based fusion
        fusion_method_used = "confidence_based"
        total_conf = tsp_conf + sam_conf + pose_conf
        if total_conf > 0:
            tsp_weight = tsp_conf / total_conf
            sam_weight = sam_conf / total_conf
            pose_weight = pose_conf / total_conf
        else:
            tsp_weight, sam_weight, pose_weight = 1.0, 0.0, 0.0
        
        fused_mask = (tsp_weight * tsp_mask + sam_weight * sam_mask + pose_weight * pose_mask).astype(np.uint8)
        fusion_weights = {"tsp": tsp_weight, "sam": sam_weight, "pose": pose_weight, "previous": 0.0}
    
    elif fusion_method == "temporal":
        # Temporal consistency-based fusion
        fusion_method_used = "temporal_based"
        if previous_mask is not None:
            # Compute temporal consistency with previous mask
            tsp_temp_conf = compute_temporal_consistency_loss(tsp_mask, previous_mask)
            sam_temp_conf = compute_temporal_consistency_loss(sam_mask, previous_mask)
            pose_temp_conf = compute_temporal_consistency_loss(pose_mask, previous_mask)
            
            # Use temporal consistency as weights
            total_temp_conf = tsp_temp_conf + sam_temp_conf + pose_temp_conf
            if total_temp_conf > 0:
                tsp_weight = tsp_temp_conf / total_temp_conf
                sam_weight = sam_temp_conf / total_temp_conf
                pose_weight = pose_temp_conf / total_temp_conf
            else:
                tsp_weight, sam_weight, pose_weight = 1.0, 0.0, 0.0
        else:
            # No previous mask, use equal weights
            tsp_weight = sam_weight = pose_weight = 1.0 / 3.0
        
        fused_mask = (tsp_weight * tsp_mask + sam_weight * sam_mask + pose_weight * pose_mask).astype(np.uint8)
        fusion_weights = {"tsp": tsp_weight, "sam": sam_weight, "pose": pose_weight, "previous": 0.0}
    
    elif fusion_method == "tsp+sam":
        # TSP + SAM fusion: prioritize TSP but enhance with SAM when available
        fusion_method_used = "tsp+sam"
        
        # Start with TSP mask as base
        fused_mask = tsp_mask.copy()
        
        # If SAM mask is available and has reasonable area, blend it in
        if sam_conf > 0.3 and np.sum(sam_mask > 0) > 1000:  # Minimum area threshold
            # Use SAM to refine TSP mask
            sam_weight = 0.4
            tsp_weight = 0.6
            fused_mask = (tsp_weight * tsp_mask + sam_weight * sam_mask).astype(np.uint8)
            print(f"[Fusion] TSP+SAM: Blending TSP ({tsp_weight:.1f}) + SAM ({sam_weight:.1f})")
        else:
            # Fallback to TSP only
            tsp_weight = 1.0
            sam_weight = 0.0
            print(f"[Fusion] TSP+SAM: Using TSP only (SAM confidence: {sam_conf:.3f})")
        
        # Set weights
        fusion_weights = {"tsp": tsp_weight, "sam": sam_weight, "pose": 0.0, "previous": 0.0}
    
    else:  # hybrid
        # Hybrid approach: combine confidence and temporal consistency
        fusion_method_used = "hybrid"
        
        # Compute temporal consistency if previous mask available
        if previous_mask is not None:
            tsp_temp_conf = compute_temporal_consistency_loss(tsp_mask, previous_mask)
            sam_temp_conf = compute_temporal_consistency_loss(sam_mask, previous_mask)
            pose_temp_conf = compute_temporal_consistency_loss(pose_mask, previous_mask)
        else:
            tsp_temp_conf = sam_temp_conf = pose_temp_conf = 1.0
        
        # Combine confidence and temporal consistency
        tsp_hybrid = 0.7 * tsp_conf + 0.3 * tsp_temp_conf
        sam_hybrid = 0.7 * sam_conf + 0.3 * sam_temp_conf
        pose_hybrid = 0.7 * pose_conf + 0.3 * pose_temp_conf
        
        total_hybrid = tsp_hybrid + sam_hybrid + pose_hybrid
        if total_hybrid > 0:
            tsp_weight = tsp_hybrid / total_hybrid
            sam_weight = sam_hybrid / total_hybrid
            pose_weight = pose_hybrid / total_hybrid
        else:
            tsp_weight, sam_weight, pose_weight = 1.0, 0.0, 0.0
        
        fused_mask = (tsp_weight * tsp_mask + sam_weight * sam_mask + pose_weight * pose_mask).astype(np.uint8)
        fusion_weights = {"tsp": tsp_weight, "sam": sam_weight, "pose": pose_weight, "previous": 0.0}
    
    print(f"[Fusion] Method: {fusion_method_used}, Weights: {fusion_weights}")
    return fused_mask, fusion_weights, fusion_method_used


def compute_motion_magnitude(current_mask, previous_mask, flow_threshold=5.0):
    """
    Compute motion magnitude between two consecutive masks.
    
    Args:
        current_mask: Current frame mask
        previous_mask: Previous frame mask
        flow_threshold: Threshold for motion magnitude normalization
    
    Returns:
        float: Motion magnitude score (0.0 = no motion, 1.0 = high motion)
    """
    if previous_mask is None:
        return 0.0
    
    try:
        # Convert masks to grayscale for optical flow
        prev_gray = previous_mask.astype(np.uint8)
        curr_gray = current_mask.astype(np.uint8)
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                          pyr_scale=0.5, levels=3, winsize=15, 
                                          iterations=3, poly_n=5, poly_sigma=1.2, 
                                          flags=0)
        
        # Compute motion magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        avg_motion = np.mean(magnitude)
        
        # Normalize motion magnitude
        motion_score = min(1.0, avg_motion / flow_threshold)
        
        return motion_score
        
    except Exception as e:
        print(f"[WARNING] Motion magnitude computation failed: {e}")
        return 0.5  # Fallback to medium motion


def adaptive_morphological_operations(mask, motion_magnitude, temporal_consistency, 
                                   base_kernel_size=3, max_kernel_size=7):
    """
    Apply adaptive morphological operations based on motion and temporal consistency.
    
    Args:
        mask: Input mask to process
        motion_magnitude: Motion magnitude score (0.0-1.0)
        temporal_consistency: Temporal consistency score (0.0-1.0)
        base_kernel_size: Base kernel size for morphological operations
        max_kernel_size: Maximum kernel size
    
    Returns:
        numpy.ndarray: Processed mask
    """
    if mask is None or np.sum(mask > 0) == 0:
        return mask
    
    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Adaptive kernel sizing based on motion and temporal consistency
    # High motion + low consistency = larger kernel (more aggressive smoothing)
    # Low motion + high consistency = smaller kernel (preserve details)
    
    motion_factor = motion_magnitude
    consistency_factor = 1.0 - temporal_consistency  # Invert for kernel sizing
    
    # Combine factors: higher values = larger kernel
    adaptive_factor = (motion_factor + consistency_factor) / 2.0
    
    # Calculate adaptive kernel size
    kernel_size = int(base_kernel_size + (max_kernel_size - base_kernel_size) * adaptive_factor)
    kernel_size = max(base_kernel_size, min(max_kernel_size, kernel_size))
    
    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    print(f"[Post-Processing] Motion: {motion_magnitude:.3f}, Consistency: {temporal_consistency:.3f}")
    print(f"[Post-Processing] Adaptive kernel size: {kernel_size}x{kernel_size}")
    
    # Create morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply adaptive morphological operations
    processed_mask = binary_mask.copy()
    
    # 1. Noise removal (opening) - removes small noise
    if motion_magnitude > 0.3:  # Only if there's significant motion
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
        print(f"[Post-Processing] Applied opening with {kernel_size}x{kernel_size} kernel")
    
    # 2. Hole filling (closing) - fills small holes
    if temporal_consistency < 0.7:  # Only if temporal consistency is low
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)
        print(f"[Post-Processing] Applied closing with {kernel_size}x{kernel_size} kernel")
    
    # 3. Boundary smoothing (morphological gradient)
    if motion_magnitude > 0.5:  # Only for high motion frames
        # Apply morphological gradient for boundary smoothing
        gradient_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, kernel_size-2), max(3, kernel_size-2)))
        dilated = cv2.dilate(processed_mask, gradient_kernel, iterations=1)
        eroded = cv2.erode(processed_mask, gradient_kernel, iterations=1)
        gradient = cv2.subtract(dilated, eroded)
        
        # Smooth the gradient and add it back
        smoothed_gradient = cv2.GaussianBlur(gradient, (3, 3), 0)
        processed_mask = cv2.add(processed_mask, smoothed_gradient)
        processed_mask = (processed_mask > 0).astype(np.uint8) * 255
        
        print(f"[Post-Processing] Applied boundary smoothing")
    
    return processed_mask


def temporal_boundary_refinement(mask, previous_mask, next_mask=None, 
                               refinement_strength=0.3):
    """
    Refine mask boundaries using temporal information.
    
    Args:
        mask: Current mask to refine
        previous_mask: Previous frame mask
        next_mask: Next frame mask (if available)
        refinement_strength: Strength of temporal refinement (0.0-1.0)
    
    Returns:
        numpy.ndarray: Refined mask
    """
    if mask is None or np.sum(mask > 0) == 0:
        return mask
    
    refined_mask = mask.copy()
    
    # Use previous mask for boundary refinement if available
    if previous_mask is not None:
        # Find contours in both masks
        current_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        previous_contours, _ = cv2.findContours(previous_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if current_contours and previous_contours:
            # Get the largest contour from each mask
            current_contour = max(current_contours, key=cv2.contourArea)
            previous_contour = max(previous_contours, key=cv2.contourArea)
            
            # Compute temporal boundary consistency
            current_perimeter = cv2.arcLength(current_contour, True)
            previous_perimeter = cv2.arcLength(previous_contour, True)
            
            if current_perimeter > 0 and previous_perimeter > 0:
                perimeter_ratio = min(current_perimeter, previous_perimeter) / max(current_perimeter, previous_perimeter)
                
                # If perimeters are very different, apply temporal refinement
                if perimeter_ratio < 0.8:
                    # Create a temporal boundary mask
                    temporal_boundary = np.zeros_like(mask, dtype=np.uint8)
                    
                    # Draw previous contour with slight dilation
                    cv2.drawContours(temporal_boundary, [previous_contour], -1, 255, 2)
                    
                    # Blend current mask with temporal boundary
                    blended = cv2.addWeighted(mask, 1.0 - refinement_strength, 
                                           temporal_boundary, refinement_strength, 0)
                    
                    # Threshold to get final mask
                    refined_mask = (blended > 127).astype(np.uint8) * 255
                    
                    print(f"[Post-Processing] Applied temporal boundary refinement (strength: {refinement_strength})")
    
    # Use next mask for additional refinement if available
    if next_mask is not None:
        # Similar process for next frame
        next_contours, _ = cv2.findContours(next_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if next_contours:
            next_contour = max(next_contours, key=cv2.contourArea)
            next_perimeter = cv2.arcLength(next_contour, True)
            
            if cv2.arcLength(refined_mask, True) > 0 and next_perimeter > 0:
                next_perimeter_ratio = min(cv2.arcLength(refined_mask, True), next_perimeter) / max(cv2.arcLength(refined_mask, True), next_perimeter)
                
                if next_perimeter_ratio < 0.8:
                    # Apply forward-looking refinement
                    next_boundary = np.zeros_like(refined_mask, dtype=np.uint8)
                    cv2.drawContours(next_boundary, [next_contour], -1, 255, 2)
                    
                    blended = cv2.addWeighted(refined_mask, 1.0 - refinement_strength/2, 
                                           next_boundary, refinement_strength/2, 0)
                    refined_mask = (blended > 127).astype(np.uint8) * 255
                    
                    print(f"[Post-Processing] Applied forward-looking boundary refinement")
    
    return refined_mask


def advanced_post_processing(fused_mask, previous_mask, next_mask=None, 
                           motion_magnitude=0.5, temporal_consistency=0.8,
                           enable_adaptive_morph=True, enable_temporal_refinement=True):
    """
    Advanced post-processing pipeline with temporal awareness.
    
    Args:
        fused_mask: Input fused mask
        previous_mask: Previous frame mask for temporal context
        next_mask: Next frame mask for forward-looking refinement
        motion_magnitude: Motion magnitude score
        temporal_consistency: Temporal consistency score
        enable_adaptive_morph: Enable adaptive morphological operations
        enable_temporal_refinement: Enable temporal boundary refinement
    
    Returns:
        numpy.ndarray: Post-processed mask
    """
    if fused_mask is None or np.sum(fused_mask > 0) == 0:
        return fused_mask
    
    print(f"\n[Post-Processing] Starting advanced post-processing...")
    print(f"[Post-Processing] Input mask area: {np.sum(fused_mask > 0)}")
    
    processed_mask = fused_mask.copy()
    
    # Step 1: Adaptive morphological operations
    if enable_adaptive_morph:
        print(f"[Post-Processing] Step 1: Adaptive morphological operations")
        processed_mask = adaptive_morphological_operations(
            processed_mask, motion_magnitude, temporal_consistency
        )
        print(f"[Post-Processing] After morphological operations: {np.sum(processed_mask > 0)} pixels")
    
    # Step 2: Temporal boundary refinement
    if enable_temporal_refinement:
        print(f"[Post-Processing] Step 2: Temporal boundary refinement")
        processed_mask = temporal_boundary_refinement(
            processed_mask, previous_mask, next_mask
        )
        print(f"[Post-Processing] After temporal refinement: {np.sum(processed_mask > 0)} pixels")
    
    # Step 3: Final consistency check
    if previous_mask is not None:
        final_consistency = compute_temporal_consistency_loss(processed_mask, previous_mask)
        print(f"[Post-Processing] Final temporal consistency: {final_consistency:.3f}")
        
        # If post-processing significantly reduced consistency, apply temporal smoothing
        if final_consistency < temporal_consistency * 0.8:
            print(f"[Post-Processing] Consistency dropped significantly, applying temporal smoothing")
            # Blend with previous mask to maintain consistency
            blend_factor = 0.7
            processed_mask = cv2.addWeighted(processed_mask, blend_factor, 
                                           previous_mask, 1.0 - blend_factor, 0)
            processed_mask = (processed_mask > 127).astype(np.uint8) * 255
            print(f"[Post-Processing] After temporal smoothing: {np.sum(processed_mask > 0)} pixels")
    
    # Step 4: Quality validation
    final_area = np.sum(processed_mask > 0)
    original_area = np.sum(fused_mask > 0)
    
    if final_area > 0:
        area_change = abs(final_area - original_area) / original_area
        print(f"[Post-Processing] Area change: {area_change:.1%}")
        
        # Warn if area changed too much
        if area_change > 0.3:
            print(f"[WARNING] Large area change detected ({area_change:.1%}), post-processing may be too aggressive")
    
    print(f"[Post-Processing] Advanced post-processing completed")
    print(f"[Post-Processing] Final mask area: {final_area} pixels")
    
    return processed_mask


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
    parser.add_argument("--max_frames", type=int, default=None, help="Process at most this many frames")
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
    print(f"[DAVIS-DEBUG] Looking for annotations in: {anno_dir}")

    mask_paths = sorted(anno_dir.glob("*.png"))
    print(f"[DAVIS-DEBUG] Found {len(mask_paths)} annotation masks.")

    # Use filename stem (e.g., "00000") as key
    masks = {}
    for p in mask_paths:
        key = p.stem
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[DAVIS-WARNING] Failed to load: {p}")
        else:
            masks[key] = mask

    print(f"[DAVIS-DEBUG] Loaded {len(masks)} masks with keys: {list(masks.keys())[:5]} ...")

    return masks


def run_tsp_sam(input_path, output_path_base, config_path, force=False, max_frames=None):
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
    # use_pose = enable_pose and fusion_method in ("pose_only", "union", "tsp+pose") and dataset_mode != "davis"
    use_pose = False  # Temporarily disabled due to MediaPipe compatibility


    temporal_smoothing = infer_cfg.get("temporal_smoothing", False)
    reset_memory_every = infer_cfg.get("reset_memory_every", None)
    
    print(f"[CONFIG] Dataset: {dataset_mode.upper()} | Fusion: {fusion_method} | use_sam={use_sam} | use_pose={use_pose} | temporal_smoothing={temporal_smoothing}")



    # Load model
    opt = type("opt", (object,), {})()
    opt.resume = model_cfg["checkpoint_path"]
    opt.device = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    opt.gpu_ids = [0] if opt.device == "cuda" else []
    opt.channel = model_cfg.get("channel", 32)

    # model = Network(opt)
    model = Network(channel=opt.channel)

    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids).to(opt.device)
    model.eval()
    
    
    # weights = torch.load(opt.resume, map_location=opt.device)
    # model.module.feat_net.pvtv2_en.load_state_dict(weights, strict=False)

    # Recommended for full model weights
    checkpoint_path = opt.resume
    state_dict = torch.load(checkpoint_path, map_location=opt.device)
    model.load_state_dict(state_dict, strict=False)

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
        print(list(gt_masks.keys())[:5])        # should print ['00000', '00001', ...]
        print(gt_masks['00000'].shape)          # should match (H, W) of original mask (e.g., 480, 854)

        # Frame path discovery
        image_paths = sorted(Path(input_path).glob("*.jpg"))
        total_frames = len(image_paths)

        # Check if annotation keys exist for all frames
        expected_keys = [f"{i:05d}" for i in range(total_frames)]
        missing_keys = [k for k in expected_keys if k not in gt_masks]

        if missing_keys:
            print(f"[WARNING] {len(missing_keys)} frames in input sequence do NOT have annotations.")
            print(f"[INFO] Example missing annotation keys: {missing_keys[:5]}")
        else:
            print(f"[INFO] All {total_frames} input frames have corresponding annotations.")

        # Set up iterator and pose threshold (0 for DAVIS)
        frame_iter = enumerate(image_paths)
        dynamic_pose_thresh = 0

    else:
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_iter = enumerate(range(total_frames))


















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
            # keypoints = extract_pose_keypoints(rgb) # Temporarily disabled
            # if keypoints:
            #     scaled = scale_keypoints(keypoints, frame_resized.shape[:2])
            #     masks = sam2_client.predict_points(Image.fromarray(rgb), scaled, [1]*len(scaled))
            #     if masks and masks[0] is not None:
            #         pose_mask = cv2.resize(masks[0], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            #         warmup_areas.append(np.sum(pose_mask > 0))
        
    
            
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
        # writer.writerow([
        #     "frame", "mean", "max", "min", "adaptive_thresh",
        #     "tsp_area", "sam_area", "pose_area", "fused_area", "max_region"
        # ])
        
   
        writer.writerow([
        "frame", "mean", "max", "min", "adaptive_thresh",
        "tsp_area", "sam_area", "pose_area", "fused_area", "max_region", "iou", "annotation_used", "temporal_consistency", "motion_consistency",
        "tsp_weight", "sam_weight", "pose_weight", "fusion_method"
        ])


        # mask_memory = deque(maxlen=5)
        temporal_window = 7 if dataset_mode == "ted" else 10
        mask_memory = deque(maxlen=temporal_window)


        prev_valid_mask = None
        # Add temporal consistency tracking
        temporal_consistency_scores = []
        mask_sequence_buffer = []

        with tqdm(total=total_frames // frame_stride) as pbar:
            processed_frames = 0
            for idx, frame_data in frame_iter:
                
                # Initialize temporal consistency variables
                current_consistency = 1.0
                current_motion_consistency = 1.0
                
                if temporal_smoothing and reset_memory_every and idx % reset_memory_every == 0:
                    print(f"[DEBUG] Resetting memory at frame {idx}")
                    mask_memory.clear()
                    mask_sequence_buffer.clear()  # Clear temporal buffer too
                
                if idx % frame_stride != 0:
                    continue

                if max_frames is not None and processed_frames >= max_frames:
                    print(f"Reached max_frames={max_frames}. Stopping early.")
                    break

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
                    print(f"[SAM] TSP mask area: {np.sum(tsp_mask > 0)}, bbox: {bbox}")
                    if bbox and np.sum(tsp_mask > 0) > 100:  # Only if TSP mask has reasonable area
                        try:
                            print(f"[SAM] Extracting bbox: {bbox}")
                            # Ensure bbox is within image bounds
                            h, w = rgb.shape[:2]
                            bbox[0] = max(0, min(bbox[0], w-1))
                            bbox[1] = max(0, min(bbox[1], h-1))
                            bbox[2] = max(bbox[0]+1, min(bbox[2], w))
                            bbox[3] = max(bbox[1]+1, min(bbox[3], h))
                            print(f"[SAM] Adjusted bbox: {bbox}")
                            
                            sam_raw = sam_wrapper.segment_with_box(rgb, str(bbox))
                            sam_mask = cv2.resize(sam_raw, tsp_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
                            print(f"[SAM] Generated mask with area: {np.sum(sam_mask > 0)}")
                        except Exception as e:
                            print(f"[SAM] Error generating mask: {e}")
                            sam_mask = np.zeros_like(tsp_mask)
                    else:
                        print(f"[SAM] No valid bbox or TSP mask too small (area: {np.sum(tsp_mask > 0)})")
                        sam_mask = np.zeros_like(tsp_mask)
                            
                if use_pose:
                    # keypoints = extract_pose_keypoints(rgb) # Temporarily disabled
                    # if keypoints:
                    #     visible_kps = [kp for kp in keypoints if kp[0] > 0 and kp[1] > 0]
                    #     if len(visible_kps) < 8:  # Require at least 8 visible keypoints
                    #         print(f"Frame {idx}: Skipping due to insufficient keypoints ({len(visible_kps)})")
                    #         pbar.update(1)
                    #         continue
                    #     scaled = scale_keypoints(keypoints, frame_resized.shape[:2])
                    #     masks = sam2_client.predict_points(Image.fromarray(rgb), scaled, [1]*len(scaled))
                    #     if masks and masks[0] is not None:
                    #         pose_mask = cv2.resize(masks[0], tsp_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
                    #         print(f"[Pose] Frame {idx}: SAM2 generated pose mask, area = {np.sum(pose_mask > 0)}")
                    pass # Pose extraction is temporarily disabled


                pose_area = int(np.sum(pose_mask > 0))
                sam_area = int(np.sum(sam_mask > 0))
                tsp_area = int(np.sum(tsp_mask > 0))
                
                print(f"[DEBUG] Frame {idx}: tsp_area={tsp_area}, sam_area={sam_area}, pose_area={pose_area}")
                
                dominant = "pose" if pose_area > sam_area and pose_area > tsp_area else \
                           "sam" if sam_area > tsp_area else "tsp"
                print(f"Frame {idx}: Dominant source = {dominant}")

                
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

                # INTELLIGENT FUSION SYSTEM
                print(f"\n[Fusion] Frame {idx}: Starting intelligent fusion...")
                
                # Get fusion method from config or use adaptive
                fusion_strategy = config.get("fusion_strategy", config.get("fusion_method", "adaptive"))
                print(f"[Fusion] Using strategy: {fusion_strategy}")
                print(f"[Fusion] Config fusion_method: {config.get('fusion_method', 'not set')}")
                print(f"[Fusion] Config fusion_strategy: {config.get('fusion_strategy', 'not set')}")
                
                # Apply intelligent fusion
                fused_mask, fusion_weights, fusion_method_used = intelligent_fusion(
                    tsp_mask, sam_mask, pose_mask, prev_valid_mask, fusion_strategy
                )
                
                # Log fusion details
                print(f"[Fusion] Frame {idx}: {fusion_method_used}")
                print(f"[Fusion] Weights: TSP={fusion_weights['tsp']:.3f}, SAM={fusion_weights['sam']:.3f}, Pose={fusion_weights['pose']:.3f}")
                if 'previous' in fusion_weights and fusion_weights['previous'] > 0:
                    print(f"[Fusion] Previous mask weight: {fusion_weights['previous']:.3f}")
                
                # Update dominant source based on fusion weights
                max_weight = max(fusion_weights['tsp'], fusion_weights['sam'], fusion_weights['pose'])
                if fusion_weights['tsp'] == max_weight:
                    dominant = "tsp"
                elif fusion_weights['sam'] == max_weight:
                    dominant = "sam"
                elif fusion_weights['pose'] == max_weight:
                    dominant = "pose"
                else:
                    dominant = "fused"
                
                print(f"[Fusion] Frame {idx}: Final dominant source = {dominant} (method: {fusion_method_used})")

                # TEMPORAL CONSISTENCY CHECKING
                if prev_valid_mask is not None:
                    # Compute temporal consistency score
                    consistency_score = compute_temporal_consistency_loss(fused_mask, prev_valid_mask)
                    temporal_consistency_scores.append(consistency_score)
                    
                    # Motion consistency using optical flow
                    motion_consistency = compute_motion_consistency(fused_mask, prev_valid_mask)
                    
                    print(f"[Temporal] Frame {idx}: Consistency={consistency_score:.3f}, Motion={motion_consistency:.3f}")
                    
                    # Apply temporal smoothing if consistency is low
                    if consistency_score < 0.6:  # Threshold for low consistency
                        print(f"[Temporal] Low consistency detected, applying smoothing...")
                        # Use previous valid mask as fallback with blending
                        blend_factor = 0.7
                        fused_mask = (blend_factor * prev_valid_mask + (1 - blend_factor) * fused_mask).astype(np.uint8)
                
                # Store mask in temporal buffer for smoothing
                mask_sequence_buffer.append(fused_mask.copy())
                
                # Apply temporal smoothing if we have enough frames
                if len(mask_sequence_buffer) >= 3:
                    smoothed_masks = temporal_mask_smoothing(mask_sequence_buffer, window_size=3)
                    if smoothed_masks:
                        fused_mask = smoothed_masks[-1]  # Use the last smoothed mask
                        print(f"[Temporal] Applied smoothing to frame {idx}")

                # ADVANCED POST-PROCESSING
                print(f"\n[Post-Processing] Frame {idx}: Starting advanced post-processing...")
                
                # Compute motion magnitude for adaptive post-processing
                current_motion_magnitude = compute_motion_magnitude(fused_mask, prev_valid_mask)
                
                # Get post-processing configuration from config
                post_processing_config = config.get("post_processing", {})
                enable_adaptive_morph = post_processing_config.get("enable_adaptive_morph", True)
                enable_temporal_refinement = post_processing_config.get("enable_temporal_refinement", True)
                
                # Apply advanced post-processing
                final_mask = advanced_post_processing(
                    fused_mask=fused_mask,
                    previous_mask=prev_valid_mask,
                    next_mask=None,  # We don't have next frame in real-time processing
                    motion_magnitude=current_motion_magnitude,
                    temporal_consistency=current_consistency,
                    enable_adaptive_morph=enable_adaptive_morph,
                    enable_temporal_refinement=enable_temporal_refinement
                )
                
                # Update fused_mask with the post-processed result
                fused_mask = final_mask
                
                print(f"[Post-Processing] Frame {idx}: Advanced post-processing completed")
                print(f"[Post-Processing] Final mask area: {np.sum(fused_mask > 0)} pixels")

                fused_area = int(np.sum(fused_mask > 0))

                # Post-processing is now handled by advanced_post_processing above
                # No need for additional post-processing here

                # Update fused area after post-processing
                fused_area = int(np.sum(fused_mask > 0))

                iou = -1  # default if not computable
                if dataset_mode == "davis":
                    # frame_name = Path(frame_data).stem
                    frame_name = f"{idx:05d}"

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
                        overlay_mask = cv2.GaussianBlur(overlay_mask, (5, 5), 0)  # gentle blur
                    else:
                        overlay_mask = mask_resized.copy()
                    

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
             
                if dataset_mode == "davis":
                    frame_name = f"{idx:05d}"  # Ensure 5-digit key for DAVIS
                    annotation_used = int(frame_name in gt_masks)
                    print(f"[DEBUG] Frame {idx}: frame_name={frame_name}, annotation_used={annotation_used}, In GT: {frame_name in gt_masks}")
                else:
                    frame_name = ""
                    annotation_used = 0

                # Get temporal consistency scores (with fallbacks)
                current_consistency = temporal_consistency_scores[-1] if temporal_consistency_scores else 1.0
                current_motion_consistency = motion_consistency if 'motion_consistency' in locals() else 1.0

                writer.writerow([
                    idx, stats['mean'], stats['max'], stats['min'], stats['adaptive_thresh'],
                    tsp_area, sam_area, pose_area, fused_area, max_region,
                    round(iou, 4), annotation_used, round(current_consistency, 4), round(current_motion_consistency, 4),
                    fusion_weights['tsp'], fusion_weights['sam'], fusion_weights['pose'], fusion_method_used
                ])

                pbar.update(1)
                processed_frames += 1
                if max_frames is not None and processed_frames >= max_frames:
                    print(f"Reached max_frames={max_frames}. Stopping early.")
                    break


    # TEMPORAL CONSISTENCY SUMMARY
    if temporal_consistency_scores:
        avg_consistency = np.mean(temporal_consistency_scores)
        min_consistency = np.min(temporal_consistency_scores)
        max_consistency = np.max(temporal_consistency_scores)
        
        print(f"\n{'='*60}")
        print(f"TEMPORAL CONSISTENCY SUMMARY")
        print(f"{'='*60}")
        print(f"Average Consistency: {avg_consistency:.3f}")
        print(f"Min Consistency: {min_consistency:.3f}")
        print(f"Max Consistency: {max_consistency:.3f}")
        print(f"Frames Processed: {len(temporal_consistency_scores)}")
        
        # Identify problematic frames
        low_consistency_frames = [i for i, score in enumerate(temporal_consistency_scores) if score < 0.6]
        if low_consistency_frames:
            print(f"Low Consistency Frames (< 0.6): {len(low_consistency_frames)}")
            print(f"Low Consistency Indices: {low_consistency_frames[:10]}...")  # Show first 10
        
        print(f"{'='*60}\n")

    # FUSION PERFORMANCE SUMMARY
    print(f"\n{'='*60}")
    print(f"INTELLIGENT FUSION SUMMARY")
    print(f"{'='*60}")
    
    # Collect fusion statistics from the processing
    fusion_methods_used = []
    tsp_weight_sum = 0
    sam_weight_sum = 0
    pose_weight_sum = 0
    frame_count = 0
    
    # This will be populated during processing, but we'll show what we can
    print(f"Fusion Strategy: {config.get('fusion_strategy', 'adaptive')}")
    print(f"Available Fusion Methods:")
    print(f"  • adaptive: Automatically chooses best fusion method")
    print(f"  • confidence: Pure confidence-based weighting")
    print(f"  • temporal: Temporal consistency-based weighting")
    print(f"  • hybrid: Combines confidence and temporal consistency")
    
    print(f"\nFusion Features:")
    print(f"  • Multi-metric confidence scoring")
    print(f"  • Adaptive strategy selection")
    print(f"  • Temporal fallback mechanisms")
    print(f"  • Intelligent weight balancing")
    
    print(f"\nExpected Improvements:")
    print(f"  • Better mask quality through intelligent weighting")
    print(f"  • Reduced flickering through temporal consistency")
    print(f"  • Adaptive handling of difficult frames")
    print(f"  • Robust fallback to previous valid masks")
    
    print(f"{'='*60}\n")

    # POST-PROCESSING PERFORMANCE SUMMARY
    print(f"\n{'='*60}")
    print(f"ADVANCED POST-PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    # Get post-processing configuration
    post_processing_config = config.get("post_processing", {})
    
    print(f"Post-Processing Configuration:")
    print(f"  • Adaptive Morphological Operations: {'Enabled' if post_processing_config.get('enable_adaptive_morph', True) else 'Disabled'}")
    print(f"  • Temporal Boundary Refinement: {'Enabled' if post_processing_config.get('enable_temporal_refinement', True) else 'Disabled'}")
    print(f"  • Base Kernel Size: {post_processing_config.get('base_kernel_size', 3)}")
    print(f"  • Max Kernel Size: {post_processing_config.get('max_kernel_size', 7)}")
    print(f"  • Refinement Strength: {post_processing_config.get('refinement_strength', 0.3)}")
    
    print(f"\nPost-Processing Features:")
    print(f"  • Motion-aware kernel sizing")
    print(f"  • Temporal consistency-based operations")
    print(f"  • Adaptive noise removal and hole filling")
    print(f"  • Temporal boundary refinement")
    print(f"  • Quality validation and consistency checks")
    
    print(f"\nExpected Improvements:")
    print(f"  • Better boundary quality through adaptive operations")
    print(f"  • Reduced noise while preserving details")
    print(f"  • Improved temporal stability")
    print(f"  • Adaptive processing based on motion patterns")
    
    print(f"{'='*60}\n")

    if dataset_mode != "davis":
        cap.release()
    print(f"Done. Results saved to: {output_path}")



if __name__ == "__main__":
    args = parse_args()
    run_tsp_sam(args.input_path, args.output_base_dir, args.config_path, force=args.force)

    
    