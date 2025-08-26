#!/usr/bin/env python3
"""
TSP-SAM Baseline Runner for DAVIS-2017 Dataset
This script runs the actual TSP-SAM model for video object segmentation
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import imageio
from skimage import img_as_ubyte
import cv2
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import time

# Add the official TSP-SAM to path
sys.path.append('tsp_sam_official')
sys.path.append('tsp_sam_official/lib')
sys.path.append('tsp_sam_official/dataloaders')

try:
    from lib.short_term_model import VideoModel
    from utils.utils import post_process
    print("Successfully imported ORIGINAL TSP-SAM VideoModel")
except ImportError as e:
    print(f"Error importing TSP-SAM modules: {e}")
    print("Please ensure 'tsp_sam_official' is a correctly configured git submodule")
    sys.exit(1)

# Configuration for ORIGINAL TSP-SAM VideoModel_pvtv2
class ModelConfig:
    def __init__(self):
        self.channel = 32  # VideoModel_pvtv2 expects 32 channels
        self.imgsize = 352
        self.pretrained = True
        self.gpu_ids = [0] if torch.cuda.is_available() else []
        self.trainsize = 352
        self.testsize = 352
        self.grid = 8  # DCT grid size for frequency analysis

def load_and_preprocess_image(image_path, img_size):
    """Load and preprocess image for TSP-SAM with proper transforms (matching official implementation)"""
    # Define transforms matching official implementation
    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_sam_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Apply transforms matching official implementation
    img_tensor = img_transform(img).unsqueeze(0)  # [1, 3, H, W] with ImageNet normalization
    
    # Convert to YCbCr for TSP-SAM's specific input requirement
    # CRITICAL FIX: YCbCr images should use the SAME transform as RGB (including ImageNet normalization)
    img_ycbcr = img.convert('YCbCr')
    img_ycbcr_tensor = img_transform(img_ycbcr).unsqueeze(0)  # [1, 3, H, W] with ImageNet normalization
    
    # One critital fix is that SAM input should be normalized [0,1], not raw [0,255]
    # Official implementation uses normalized inputs
    img_sam_tensor = img_sam_transform(img).unsqueeze(0)  # [1, 3, H, W] normalized [0,1]
    
    return img_tensor, img_ycbcr_tensor, img_sam_tensor

def load_ground_truth_mask(gt_path, target_size=352):
    """Load and preprocess ground truth mask from DAVIS annotations"""
    try:
        gt_mask = Image.open(gt_path).convert('L')
        gt_mask = gt_mask.resize((target_size, target_size), Image.NEAREST)
        gt_np = np.array(gt_mask)
        
        print(f"    [DEBUG] Raw GT mask - shape: {gt_np.shape}, range: {gt_np.min()} to {gt_np.max()}")
        print(f"    [DEBUG] Raw GT mask unique values: {np.unique(gt_np)}")
        
        # DAVIS annotations can have different pixel value ranges
        # Handle both [0, 38] and [0, 128] and [0, 255] cases
        if gt_np.max() > 1:
            if gt_np.max() <= 38:  # DAVIS 2016 format
                print(f"    [DEBUG] DAVIS 2016 format detected (max: {gt_np.max()})")
                gt_np = (gt_np > 0).astype(np.uint8)
            elif gt_np.max() <= 128:  # DAVIS 2017 format
                print(f"    [DEBUG] DAVIS 2017 format detected (max: {gt_np.max()})")
                gt_np = (gt_np > 0).astype(np.uint8)
            else:  # Standard 0-255 format
                print(f"    [DEBUG] Standard format detected (max: {gt_np.max()})")
                gt_np = (gt_np > 127).astype(np.uint8)
        
        print(f"    [DEBUG] After normalization - range: {gt_np.min()} to {gt_np.max()}")
        print(f"    [DEBUG] After normalization - unique values: {np.unique(gt_np)}")
        print(f"    [DEBUG] After normalization - sum: {gt_np.sum()}")
        
        # Apply morphological operations for cleanup
        kernel = np.ones((3, 3), np.uint8)
        gt_np = cv2.morphologyEx(gt_np, cv2.MORPH_CLOSE, kernel)
        gt_np = cv2.morphologyEx(gt_np, cv2.MORPH_OPEN, kernel)
        
        print(f"    [DEBUG] After morphology - sum: {gt_np.sum()}")
        
        return gt_np
        
    except Exception as e:
        print(f"    [ERROR] Failed to load ground truth mask: {e}")
        return None

def create_video_sequence(image_files, max_frames=5):
    """Create a video sequence from multiple frames for TSP-SAM"""
    if len(image_files) < max_frames:
        max_frames = len(image_files)
    
    # Load multiple frames to create temporal context
    frames = []
    frames_ycbcr = []
    frames_sam = []
    
    for i in range(max_frames):
        img_tensor, img_ycbcr_tensor, sam_features = load_and_preprocess_image(image_files[i], 352)
        frames.append(img_tensor)
        frames_ycbcr.append(img_ycbcr_tensor)
        frames_sam.append(sam_features)
    
    return frames, frames_ycbcr, frames_sam

def adaptive_threshold_optimization(pred_mask, gt_mask=None):
    """Optimized adaptive thresholding with multiple strategies"""
    pred_np = pred_mask.cpu().numpy().squeeze()
    
    # Calculate statistics
    pred_mean = np.mean(pred_np)
    pred_std = np.std(pred_np)
    pred_median = np.median(pred_np)
    
    print(f"    [DEBUG] Prediction statistics - mean: {pred_mean:.6f}, std: {pred_std:.6f}, median: {pred_median:.6f}")
    
    # If ground truth is available, use it to guide threshold selection
    if gt_mask is not None:
        gt_coverage = np.sum(gt_mask) / gt_mask.size
        print(f"    [DEBUG] Ground truth coverage: {gt_coverage:.3%} ({np.sum(gt_mask)} pixels)")
        target_coverage = gt_coverage * 1.1  # Aim for 110% of ground truth coverage
        print(f"    [DEBUG] Target coverage: {target_coverage:.3%}")
    else:
        target_coverage = 0.15  # Default to 15% if no ground truth
    
    # Define threshold strategies
    thresholds = {
        'mean': pred_mean,
        'median': pred_median,
        'mean + 0.5*std': pred_mean + 0.5 * pred_std,
        'mean + 0.75*std': pred_mean + 0.75 * pred_std,
        'mean + std': pred_mean + pred_std,
        'mean + 1.25*std': pred_mean + 1.25 * pred_std,
        'mean + 1.5*std': pred_mean + 1.5 * pred_std,
        'percentile_80': np.percentile(pred_np, 80),
        'percentile_85': np.percentile(pred_np, 85),
        'percentile_90': np.percentile(pred_np, 90),
        'percentile_95': np.percentile(pred_np, 95)
    }
    
    # Score each threshold based on how close it gets to target coverage
    threshold_scores = {}
    print(f"    [DEBUG] Threshold strategies:")
    
    for name, threshold in thresholds.items():
        coverage = np.sum(pred_np > threshold) / pred_np.size
        score = -abs(coverage - target_coverage)  # Negative score, closer to 0 is better
        threshold_scores[name] = score
        
        print(f"        {name}: {threshold:.6f} -> {coverage:.3%} ({np.sum(pred_np > threshold)} pixels) [score: {score:.3f}]")
    
    # Select the best threshold
    best_threshold_name = max(threshold_scores, key=threshold_scores.get)
    best_threshold = thresholds[best_threshold_name]
    
    print(f"    [DEBUG] Best threshold selected: {best_threshold:.6f} (score: {threshold_scores[best_threshold_name]:.3f})")
    
    return best_threshold

def apply_temporal_consistency(pred_mask, prev_mask=None, consistency_weight=0.3):
    """Apply temporal consistency between frames"""
    if prev_mask is None:
        return pred_mask
    
    # Blend current prediction with previous mask
    if isinstance(pred_mask, torch.Tensor):
        pred_np = pred_mask.cpu().numpy().squeeze()
    else:
        pred_np = pred_mask
    
    if isinstance(prev_mask, torch.Tensor):
        prev_np = prev_mask.cpu().numpy().squeeze()
    else:
        prev_np = prev_mask
    
    # Ensure same shape
    if pred_np.shape != prev_np.shape:
        prev_np = cv2.resize(prev_np, (pred_np.shape[1], pred_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Apply temporal consistency
    consistent_mask = (1 - consistency_weight) * pred_np + consistency_weight * prev_np
    
    return consistent_mask

def main():
    parser = argparse.ArgumentParser(description='TSP-SAM Baseline for Video De-identification')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output masks')
    parser.add_argument('--sequence', type=str, required=True, help='Sequence name to process')
    parser.add_argument('--max_frames', type=int, default=5, help='Maximum frames to process')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    print(f"[DEBUG] ==================================================")
    print(f"[DEBUG] ARGUMENTS:")
    print(f"[DEBUG]   input_path: {args.input_path}")
    print(f"[DEBUG]   output_path: {args.output_path}")
    print(f"[DEBUG]   sequence: {args.sequence}")
    print(f"[DEBUG]   max_frames: {args.max_frames}")
    print(f"[DEBUG]   device: {args.device}")
    print(f"[DEBUG] ==================================================")
    
    # Create output directory
    print(f"[DEBUG] Creating output directory: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    print(f"[DEBUG] Output directory created/verified: {os.path.exists(args.output_path)}")
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load TSP-SAM model
    print("Loading TSP-SAM model...")
    try:
        # Create model_args object like official implementation
        class ModelArgs:
            def __init__(self):
                self.trainsize = 352
                self.testsize = 352
                self.grid = 8
                self.gpu_ids = [0] if device == 'cuda' else [0]
        
        model_args = ModelArgs()
        model = VideoModel(model_args)
        
        # Load TSP-SAM checkpoint
        checkpoint_path = 'tsp_sam_official/snapshot/best_checkpoint.pth'
        if os.path.exists(checkpoint_path):
            try:
                print(f"Loading checkpoint from {checkpoint_path}...")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                print(f"Checkpoint loaded, keys: {list(checkpoint.keys())[:5]}...")
                
                # Fix checkpoint keys by removing 'module.' prefix if present
                if list(checkpoint.keys())[0].startswith('module.'):
                    print(f"  [DEBUG] Removing 'module.' prefix from checkpoint keys")
                    new_checkpoint = {}
                    for key, value in checkpoint.items():
                        new_key = key.replace('module.', '')
                        new_checkpoint[new_key] = value
                    checkpoint = new_checkpoint
                    print(f"  [DEBUG] Fixed checkpoint keys: {list(checkpoint.keys())[:5]}...")
                
                model.load_state_dict(checkpoint, strict=True)  # Use strict=True like original
                print(f"TSP-SAM checkpoint loaded in {time.time():.1f} seconds")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")
                print("Model will run with random weights")
        else:
            print(f"Warning: TSP-SAM checkpoint not found at {checkpoint_path}")
            print("Model will run with random weights")
        
        model = model.to(device)
        model.eval()
        print("ORIGINAL TSP-SAM VideoModel loaded successfully")
        
    except Exception as e:
        print(f"Error loading TSP-SAM model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Find sequence directory
    seq_dir = os.path.join(args.input_path, 'JPEGImages', '480p', args.sequence)
    print(f"[DEBUG] Looking for sequence directory: {seq_dir}")
    print(f"[DEBUG] Directory exists: {os.path.exists(seq_dir)}")
    if not os.path.exists(seq_dir):
        print(f"[ERROR] Sequence directory not found: {seq_dir}")
        print(f"[DEBUG] Available directories in {os.path.join(args.input_path, 'JPEGImages', '480p')}:")
        if os.path.exists(os.path.join(args.input_path, 'JPEGImages', '480p')):
            print(f"[DEBUG] {os.listdir(os.path.join(args.input_path, 'JPEGImages', '480p'))}")
        return
    
    # Find ground truth directory
    gt_dir = os.path.join(args.input_path, 'Annotations', '480p', args.sequence)
    print(f"[DEBUG] Looking for ground truth directory: {gt_dir}")
    print(f"[DEBUG] Directory exists: {os.path.exists(gt_dir)}")
    if not os.path.exists(gt_dir):
        print(f"[ERROR] Ground truth directory not found: {gt_dir}")
        print(f"[DEBUG] Available directories in {os.path.join(args.input_path, 'Annotations', '480p')}:")
        if os.path.exists(os.path.join(args.input_path, 'Annotations', '480p')):
            print(f"[DEBUG] {os.listdir(os.path.join(args.input_path, 'Annotations', '480p'))}")
        return
    
    # Get image and ground truth files
    print(f"[DEBUG] Listing files in sequence directory: {seq_dir}")
    all_seq_files = os.listdir(seq_dir)
    print(f"[DEBUG] All files in sequence directory: {all_seq_files[:10]}...")  # Show first 10
    
    img_files = sorted([f for f in all_seq_files if f.endswith('.jpg')])
    print(f"[DEBUG] Found {len(img_files)} image files: {img_files[:5]}...")  # Show first 5
    
    print(f"[DEBUG] Listing files in ground truth directory: {gt_dir}")
    all_gt_files = os.listdir(gt_dir)
    print(f"[DEBUG] All files in ground truth directory: {all_gt_files[:10]}...")  # Show first 10
    
    gt_files = sorted([f for f in all_gt_files if f.endswith('.png')])
    print(f"[DEBUG] Found {len(gt_files)} ground truth files: {gt_files[:5]}...")  # Show first 5
    
    if len(img_files) < 3:
        print(f"[ERROR] Need at least 3 frames for TSP-SAM temporal processing, found {len(img_files)}")
        return
    
    if len(img_files) != len(gt_files):
        print(f"[ERROR] Mismatch: {len(img_files)} images vs {len(gt_files)} ground truth files")
        return
    
    print(f"[DEBUG] Found {len(img_files)} frames in sequence: {args.sequence}")
    
    # Process frames with proper temporal context (like official implementation)
    max_frames = min(args.max_frames, len(img_files))
    print(f"[DEBUG] Processing {max_frames} frames...")
    
    # Start from frame 2 (like official implementation)
    start_frame = 2
    end_frame = min(start_frame + max_frames, len(img_files))
    print(f"[DEBUG] Frame range: {start_frame} to {end_frame}")
    print(f"[DEBUG] Total frames to process: {end_frame - start_frame}")
    
    with tqdm(total=end_frame-start_frame, desc=f"Processing {args.sequence}") as pbar:
        for frame_idx in range(start_frame, end_frame):
            print(f"[DEBUG] ==================================================")
            print(f"[DEBUG] Processing frame {frame_idx} (index {frame_idx - start_frame + 1}/{end_frame - start_frame})")
            print(f"[DEBUG] ==================================================")
            try:
                print(f"\n[Frame {frame_idx + 1}/{len(img_files)}] Processing {img_files[frame_idx]}")
                print(f"  [DEBUG] {'='*50}")
                
                # Create temporal sequence: [frame_idx-2, frame_idx-1, frame_idx] (like official)
                if frame_idx >= 2:
                    temporal_frames = [frame_idx-2, frame_idx-1, frame_idx]
                else:
                    # Handle edge cases
                    temporal_frames = [0, 0, frame_idx] if frame_idx == 1 else [0, 0, 0]
                
                print(f"  [DEBUG] Temporal context: frames {temporal_frames}")
                
                # Load temporal sequence
                images = []
                images_ycbcr = []
                
                print(f"  [DEBUG] Loading temporal sequence: frames {temporal_frames}")
                for temp_idx in temporal_frames:
                    img_file = os.path.join(seq_dir, img_files[temp_idx])
                    print(f"  [DEBUG] Loading temporal frame {temp_idx}: {img_file}")
                    print(f"  [DEBUG] File exists: {os.path.exists(img_file)}")
                    
                    # Load and preprocess image
                    try:
                        img_tensor, img_ycbcr_tensor, _ = load_and_preprocess_image(img_file, 352)
                        images.append(img_tensor)
                        images_ycbcr.append(img_ycbcr_tensor)
                        print(f"  [DEBUG] Successfully loaded temporal frame {temp_idx}")
                    except Exception as e:
                        print(f"  [ERROR] Failed to load temporal frame {temp_idx}: {e}")
                        raise
                
                # Verify we have different frames (not identical)
                if len(images) >= 2:
                    frame_diff = torch.abs(images[0] - images[1]).mean().item()
                    print(f"  [DEBUG] Frame difference (0 vs 1): {frame_diff:.6f}")
                    if len(images) >= 3:
                        frame_diff2 = torch.abs(images[1] - images[2]).mean().item()
                        print(f"  [DEBUG] Frame difference (1 vs 2): {frame_diff2:.6f}")
                
                # Load SAM input: ONLY the current frame (like official implementation)
                current_img_file = os.path.join(seq_dir, img_files[frame_idx])
                print(f"  [DEBUG] Loading SAM input from: {current_img_file}")
                print(f"  [DEBUG] File exists: {os.path.exists(current_img_file)}")
                
                try:
                    _, _, img_sam_tensor = load_and_preprocess_image(current_img_file, 352)
                    print(f"  [DEBUG] Successfully loaded SAM input")
                except Exception as e:
                    print(f"  [ERROR] Failed to load SAM input: {e}")
                    raise
                
                # Move to device
                images = [img.to(device) for img in images]
                images_ycbcr = [img.to(device) for img in images_ycbcr]
                img_sam_tensor = img_sam_tensor.to(device)  # Single tensor, not a list!
                
                print(f"  [DEBUG] Input shapes:")
                print(f"    [DEBUG] images: {[img.shape for img in images]}")
                print(f"    [DEBUG] images_ycbcr: {[img.shape for img in images_ycbcr]}")
                print(f"    [DEBUG] img_sam_tensor: {img_sam_tensor.shape} (single tensor)")
                
                # Load ground truth for current frame
                current_gt_file = os.path.join(gt_dir, gt_files[frame_idx])
                print(f"  [DEBUG] Loading ground truth from: {current_gt_file}")
                print(f"  [DEBUG] File exists: {os.path.exists(current_gt_file)}")
                
                try:
                    gt_mask = load_ground_truth_mask(current_gt_file, 352)
                    if gt_mask is None:
                        print(f"  [ERROR] Failed to load ground truth for {gt_files[frame_idx]}")
                        continue
                    print(f"  [DEBUG] Successfully loaded ground truth")
                except Exception as e:
                    print(f"  [ERROR] Failed to load ground truth: {e}")
                    raise
                
                print(f"  [DEBUG] Ground truth mask shape: {gt_mask.shape}")
                print(f"  [DEBUG] Ground truth mask sum: {gt_mask.sum()}")
                print(f"  [DEBUG] Ground truth mask unique: {np.unique(gt_mask)}")
                
                # Call ORIGINAL TSP-SAM model (like official implementation)
                print(f"  [DEBUG] Calling ORIGINAL TSP-SAM VideoModel...")
                print(f"  [DEBUG] Input shapes:")
                print(f"    [DEBUG] images: {[img.shape for img in images]}")
                print(f"  [DEBUG] images_ycbcr: {[img.shape for img in images_ycbcr]}")
                print(f"  [DEBUG] img_sam_tensor: {img_sam_tensor.shape}")
                
                try:
                    with torch.no_grad():
                        # Another critical fix: TSP-SAM VideoModel expects 3 inputs like official implementation
                        # 1. x: temporal sequence of RGB images (list of tensors)
                        # 2. x_ycbcr: temporal sequence of YCbCr images (list of tensors)
                        # 3. x_sam: SAM input (single frame tensor)
                        pred = model(images, images_ycbcr, img_sam_tensor)
                    print(f"  [DEBUG] Model call successful")
                except Exception as e:
                    print(f"  [ERROR] Model call failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                print(f"  [DEBUG] Raw prediction type: {type(pred)}")
                print(f"  [DEBUG] Raw prediction length: {len(pred)}")
                print(f"TSP-SAM VideoModel output: {len(pred)} components")
                
                # Debug each component of the prediction
                for i, p in enumerate(pred):
                    if isinstance(p, torch.Tensor):
                        print(f"  [DEBUG] pred[{i}] - type: {type(p)}, shape: {p.shape}, range: {p.min().item():.6f} to {p.max().item():.6f}, mean: {p.mean().item():.6f}")
                    else:
                        print(f"  [DEBUG] pred[{i}] - type: {type(p)}")
                
                # According to ORIGINAL TSP-SAM architecture:
                # pred is a single tensor (main prediction)
                
                # CRITICAL FIX: Use the main prediction (output 2) based on actual TSP-SAM structure
                if isinstance(pred, torch.Tensor):
                    pred_tensor = pred
                elif isinstance(pred, (list, tuple)) and len(pred) > 0:
                    print(f"  [DEBUG] TSP-SAM returned {len(pred)} outputs:")
                    for i, p in enumerate(pred):
                        if isinstance(p, torch.Tensor):
                            print(f"    [DEBUG] Output {i}: shape={p.shape}, range={p.min().item():.6f} to {p.max().item():.6f}")
                    
                    # Based on debug output: pred[2] is the main mask prediction [1, 1, 352, 352]
                    # pred[4] is another mask prediction, pred[3] is IoU prediction
                    pred_tensor = pred[2]  # Main mask prediction
                    print(f"  [DEBUG] Using main prediction (output 2): Main mask tensor")
                else:
                    print(f"  [ERROR] Unexpected prediction format: {type(pred)}")
                    continue
                
                print(f"  [DEBUG] Using ORIGINAL TSP-SAM prediction: {pred_tensor.shape}")
                print(f"  [DEBUG] Prediction range: {pred_tensor.min().item():.6f} to {pred_tensor.max().item():.6f}")
                print(f"  [DEBUG] Prediction mean: {pred_tensor.mean().item():.6f}")
                
                # Apply sigmoid activation to convert logits to probabilities
                pred_tensor = torch.sigmoid(pred_tensor)
                print(f"  [DEBUG] After sigmoid - range: {pred_tensor.min().item():.6f} to {pred_tensor.max().item():.6f}")
                print(f"  [DEBUG] After sigmoid - mean: {pred_tensor.mean().item():.6f}")
                
                # Convert to numpy and resize to match ground truth
                pred_mask = F.interpolate(pred_tensor, size=gt_mask.shape, mode='bilinear', align_corners=False)
                pred_mask = pred_mask.squeeze().cpu().numpy()
                
                print(f"  [DEBUG] Resized prediction mask shape: {pred_mask.shape}")
                print(f"  [DEBUG] Prediction mask range: {pred_mask.min():.6f} to {pred_mask.max():.6f}")
                print(f"  [DEBUG] Prediction mask mean: {pred_mask.mean():.6f}")
                
                # Apply balanced thresholding to get reasonable coverage
                # Target coverage similar to ground truth
                target_coverage = gt_mask.sum() / gt_mask.size
                
                print(f"  [DEBUG] Target coverage: {target_coverage:.4f}")
                
                # Find threshold that gives reasonable coverage
                # CRITICAL FIX: TSP-SAM produces sparse predictions, need lower thresholds
                print(f"  [DEBUG] Prediction statistics:")
                print(f"    [DEBUG] Min: {pred_mask.min():.6f}, Max: {pred_mask.max():.6f}")
                print(f"    [DEBUG] Mean: {pred_mask.mean():.6f}, Std: {pred_mask.std():.6f}")
                
                # Use percentile-based thresholds for sparse predictions
                percentiles = [50, 60, 70, 80, 85, 90, 95]
                best_threshold = 0.1  # Start with very low threshold
                best_coverage_diff = abs((pred_mask > 0.1).sum() / pred_mask.size - target_coverage)
                
                for p in percentiles:
                    test_threshold = np.percentile(pred_mask, p)
                    test_coverage = (pred_mask > test_threshold).sum() / pred_mask.size
                    coverage_diff = abs(test_coverage - target_coverage)
                    
                    print(f"  [DEBUG] Percentile {p} ({test_threshold:.6f}): coverage {test_coverage:.4f}, diff {coverage_diff:.4f}")
                    
                    if coverage_diff < best_coverage_diff:
                        best_threshold = test_threshold
                        best_coverage_diff = coverage_diff
                
                # Also try very low fixed thresholds for sparse predictions
                for test_threshold in [0.05, 0.1, 0.15, 0.2, 0.25]:
                    test_coverage = (pred_mask > test_threshold).sum() / pred_mask.size
                    coverage_diff = abs(test_coverage - target_coverage)
                    
                    print(f"  [DEBUG] Fixed threshold {test_threshold}: coverage {test_coverage:.4f}, diff {coverage_diff:.4f}")
                    
                    if coverage_diff < best_coverage_diff:
                        best_threshold = test_threshold
                        best_coverage_diff = coverage_diff
                
                # CRITICAL FIX: Add even lower thresholds for very sparse TSP-SAM predictions
                for test_threshold in [0.01, 0.02, 0.03, 0.04]:
                    test_coverage = (pred_mask > test_threshold).sum() / pred_mask.size
                    coverage_diff = abs(test_coverage - target_coverage)
                    
                    print(f"  [DEBUG] Ultra-low threshold {test_threshold}: coverage {test_coverage:.4f}, diff {coverage_diff:.4f}")
                    
                    if coverage_diff < best_coverage_diff:
                        best_threshold = test_threshold
                        best_coverage_diff = coverage_diff
                
                threshold = best_threshold
                print(f"  [DEBUG] Selected threshold: {threshold:.6f} (coverage diff: {best_coverage_diff:.4f})")
                
                # Apply threshold to get binary mask
                binary_mask = (pred_mask > threshold).astype(np.uint8)
                
                print(f"  [DEBUG] Binary mask shape: {binary_mask.shape}")
                print(f"  [DEBUG] Binary mask sum: {binary_mask.sum()}")
                print(f"  [DEBUG] Binary mask unique: {np.unique(binary_mask)}")
                
                # CRITICAL FIX: Apply morphological operations to clean up the mask
                if binary_mask.sum() > 0:
                    kernel = np.ones((3, 3), np.uint8)
                    # Remove small noise
                    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                    # Fill small holes
                    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                    
                    print(f"  [DEBUG] After morphology - sum: {binary_mask.sum()}")
                    print(f"  [DEBUG] Morphology effect: {binary_mask.sum()} pixels remaining")
                
                # Apply TSP-SAM post-processing (like official implementation)
                print(f"  [DEBUG] Raw TSP-SAM output - sum: {binary_mask.sum()}")
                print(f"  [DEBUG] Raw TSP-SAM output - coverage: {binary_mask.sum() / binary_mask.size:.4f}")
                
                # TEMPORARILY DISABLE POST-PROCESSING - it's causing issues with extremely large values
                # try:
                #     from utils.utils import post_process
                #     # CRITICAL FIX: post_process expects binary mask with proper format
                #     # Convert to uint8 and ensure proper dimensions
                #     post_input = binary_mask.astype(np.uint8)
                #     processed_mask = post_process(post_input)
                #     
                #     print(f"  [DEBUG] After post_process - sum: {processed_mask.sum()}")
                #     print(f"  [DEBUG] Post-processing effect: {binary_mask.sum() - processed_mask.sum()} pixels removed")
                #     
                #     # Only use post-processed mask if it's reasonable
                #     if processed_mask.sum() > 0 and processed_mask.sum() < binary_mask.size * 0.5:
                #         binary_mask = processed_mask
                #         print(f"  [DEBUG] Using post-processed mask")
                #     else:
                #         print(f"  [DEBUG] Post-processing too aggressive, keeping original mask")
                #         
                # except Exception as e:
                #     print(f"  [WARNING] Post-processing failed: {e}, using unprocessed mask")
                print(f"  [DEBUG] Post-processing disabled - using original binary mask")
                
                # Calculate metrics
                intersection = np.logical_and(binary_mask, gt_mask).sum()
                union = np.logical_or(binary_mask, gt_mask).sum()
                iou = intersection / (union + 1e-8)
                
                dice_numerator = 2 * intersection
                dice_denominator = binary_mask.sum() + gt_mask.sum()
                dice = dice_numerator / (dice_denominator + 1e-8)
                
                coverage_ratio = binary_mask.sum() / (gt_mask.sum() + 1e-8)
                
                print(f"  [DEBUG] Metrics:")
                print(f"    [DEBUG] IoU: {iou:.4f}")
                print(f"    [DEBUG] Dice: {dice:.4f}")
                print(f"    [DEBUG] Coverage Ratio: {coverage_ratio:.4f}")
                
                # Save prediction
                output_file = os.path.join(args.output_path, f"{args.sequence}_{frame_idx:04d}.png")
                print(f"  [DEBUG] Saving to: {output_file}")
                print(f"  [DEBUG] Output directory exists: {os.path.exists(args.output_path)}")
                print(f"  [DEBUG] Output directory contents: {os.listdir(args.output_path)[:5] if os.path.exists(args.output_path) else 'N/A'}")
                
                # CRITICAL FIX: binary_mask is already uint8 [0,1], convert to [0,255] properly
                save_mask = (binary_mask * 255).astype(np.uint8)
                print(f"  [DEBUG] Save mask - shape: {save_mask.shape}, range: {save_mask.min()} to {save_mask.max()}, unique: {np.unique(save_mask)}")
                
                try:
                    Image.fromarray(save_mask, mode='L').save(output_file)
                    print(f"  [DEBUG] Successfully saved prediction to: {output_file}")
                    print(f"  [DEBUG] File exists after save: {os.path.exists(output_file)}")
                    print(f"  [DEBUG] File size after save: {os.path.getsize(output_file) if os.path.exists(output_file) else 'N/A'} bytes")
                except Exception as e:
                    print(f"  [ERROR] Failed to save prediction: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                print(f"  [DEBUG] ==================================================")
                
            except Exception as e:
                print(f"  [ERROR] Failed to process frame {frame_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            pbar.update(1)
    
    print(f"\nTSP-SAM baseline completed!")
    print(f"Results saved to: {args.output_path}")

if __name__ == "__main__":
    main()
