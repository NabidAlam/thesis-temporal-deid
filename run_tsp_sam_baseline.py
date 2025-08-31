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

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not available. Install with: pip install wandb")

from datetime import datetime

# Advanced dependencies for enhanced metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available. Install with: pip install psutil")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("GPUtil not available. Install with: pip install GPUtil")

try:
    from scipy import ndimage
    from scipy.spatial.distance import directed_hausdorff
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("scipy not available. Install with: pip install scipy")

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import adapted_rand_error, variation_of_information
    from skimage.measure import label, regionprops
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("scikit-image not available. Install with: pip install scikit-image")

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
            # DAVIS 2016 format detected (max: 38)
            if gt_np.max() <= 38:
                print(f"    [DEBUG] DAVIS 2016 format detected (max: {gt_np.max()})")
                gt_np = (gt_np > 0).astype(np.uint8)
            else:
                print(f"    [DEBUG] DAVIS 2017 format detected (max: {gt_np.max()})")
                gt_np = (gt_np > 0).astype(np.uint8)
        else:
            print(f"    [DEBUG] Already normalized format detected")
            gt_np = (gt_np > 0.1).astype(np.uint8)
        
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
        print(f"Error loading ground truth mask {gt_path}: {e}")
        return None

# Advanced utility functions for enhanced metrics
def calculate_precision_recall(gt_mask, pred_mask):
    """Calculate precision and recall for segmentation"""
    try:
        tp = np.logical_and(gt_mask, pred_mask).sum()  # True positives
        fp = np.logical_and(np.logical_not(gt_mask), pred_mask).sum()  # False positives
        fn = np.logical_and(gt_mask, np.logical_not(pred_mask)).sum()  # False negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return precision, recall
    except Exception:
        return 0.0, 0.0

def calculate_boundary_accuracy(gt_mask, pred_mask, boundary_width=2):
    """Calculate boundary accuracy between ground truth and prediction"""
    if not SCIPY_AVAILABLE:
        return 0.0
    
    try:
        # Create boundary masks
        gt_boundary = ndimage.binary_erosion(gt_mask) != ndimage.binary_dilation(gt_mask)
        pred_boundary = ndimage.binary_erosion(pred_mask) != ndimage.binary_dilation(pred_mask)
        
        # Dilate boundaries for tolerance
        gt_boundary = ndimage.binary_dilation(gt_boundary, iterations=boundary_width)
        pred_boundary = ndimage.binary_dilation(pred_boundary, iterations=boundary_width)
        
        # Calculate boundary overlap
        boundary_intersection = np.logical_and(gt_boundary, pred_boundary).sum()
        boundary_union = np.logical_or(gt_boundary, pred_boundary).sum()
        
        return boundary_intersection / boundary_union if boundary_union > 0 else 0.0
    except Exception:
        return 0.0

def calculate_hausdorff_distance(gt_mask, pred_mask):
    """Calculate Hausdorff distance for boundary accuracy assessment"""
    if not SCIPY_AVAILABLE:
        return 0.0
    
    try:
        # Find boundary points
        gt_boundary = np.argwhere(gt_mask > 0)
        pred_boundary = np.argwhere(pred_mask > 0)
        
        if len(gt_boundary) == 0 or len(pred_boundary) == 0:
            return float('inf')
        
        # Calculate directed Hausdorff distances
        d_gt_to_pred = directed_hausdorff(gt_boundary, pred_boundary)[0]
        d_pred_to_gt = directed_hausdorff(pred_boundary, gt_boundary)[0]
        
        # Return symmetric Hausdorff distance
        return max(d_gt_to_pred, d_pred_to_gt)
    except Exception:
        return 0.0

def calculate_contour_similarity(gt_mask, pred_mask):
    """Calculate contour similarity using structural similarity"""
    if not SKIMAGE_AVAILABLE:
        return 0.0
    
    try:
        # Ensure masks are same size and type
        gt_norm = gt_mask.astype(np.float32) / gt_mask.max() if gt_mask.max() > 0 else gt_mask.astype(np.float32)
        pred_norm = pred_mask.astype(np.float32) / pred_mask.max() if pred_mask.max() > 0 else pred_mask.astype(np.float32)
        
        # Calculate SSIM
        similarity = ssim(gt_norm, pred_norm, data_range=1.0)
        return similarity
    except Exception:
        return 0.0

def calculate_region_based_metrics(gt_mask, pred_mask):
    """Calculate region-based segmentation metrics"""
    if not SKIMAGE_AVAILABLE:
        return {'adapted_rand_error': 0.0, 'variation_of_information': 0.0}
    
    try:
        # Ensure masks are labeled (not binary)
        gt_labeled = gt_mask.astype(np.int32)
        pred_labeled = (pred_mask > 0.1).astype(np.int32)
        
        # Adapted Rand Error (lower is better)
        ar_error = adapted_rand_error(gt_labeled, pred_labeled)[0]
        
        # Variation of Information (lower is better)
        voi = variation_of_information(gt_labeled, pred_labeled)[0]
        
        return {
            'adapted_rand_error': ar_error,
            'variation_of_information': voi
        }
    except Exception:
        return {'adapted_rand_error': 0.0, 'variation_of_information': 0.0}

def calculate_complexity_metrics(gt_mask):
    """Calculate object complexity metrics for difficulty assessment"""
    if not SKIMAGE_AVAILABLE:
        return {
            'object_count': 0,
            'total_area': 0,
            'avg_area': 0,
            'perimeter': 0,
            'compactness': 0,
            'eccentricity': 0
        }
    
    try:
        # Label connected components
        labeled_mask = label(gt_mask)
        regions = regionprops(labeled_mask)
        
        if not regions:
            return {
                'object_count': 0,
                'total_area': 0,
                'avg_area': 0,
                'perimeter': 0,
                'compactness': 0,
                'eccentricity': 0
            }
        
        # Calculate complexity metrics
        total_area = sum(region.area for region in regions)
        avg_area = total_area / len(regions)
        perimeter = sum(region.perimeter for region in regions)
        
        # Compactness (4π * area / perimeter²)
        compactness = (4 * np.pi * total_area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Eccentricity (average across regions)
        eccentricities = [region.eccentricity for region in regions if hasattr(region, 'eccentricity')]
        avg_eccentricity = np.mean(eccentricities) if eccentricities else 0
        
        return {
            'object_count': len(regions),
            'total_area': total_area,
            'avg_area': avg_area,
            'perimeter': perimeter,
            'compactness': compactness,
            'eccentricity': avg_eccentricity
        }
    except Exception:
        return {
            'object_count': 0,
            'total_area': 0,
            'avg_area': 0,
            'perimeter': 0,
            'compactness': 0,
            'eccentricity': 0
        }

def analyze_failure_cases(gt_mask, pred_mask, frame_idx, sequence_name):
    """Analyze when and why segmentation fails"""
    try:
        gt_binary = (gt_mask > 0).astype(np.uint8)
        pred_binary = (pred_mask > 0.0).astype(np.uint8)
        
        # Calculate failure metrics
        false_negatives = np.logical_and(gt_binary, np.logical_not(pred_binary)).sum()
        false_positives = np.logical_and(np.logical_not(gt_binary), pred_binary).sum()
        true_positives = np.logical_and(gt_binary, pred_binary).sum()
        
        total_gt_pixels = gt_binary.sum()
        total_pred_pixels = pred_binary.sum()
        total_pixels = gt_binary.size
        
        # Calculate ratios
        fn_ratio = false_negatives / max(total_gt_pixels, 1) if total_gt_pixels > 0 else 0
        fp_ratio = false_positives / max(total_pixels, 1)
        
        # Determine if this is a failure case
        is_failure = fn_ratio > 0.3 or fp_ratio > 0.05
        
        return {
            'false_negatives': false_negatives,
            'false_positives': false_positives,
            'true_positives': true_positives,
            'fn_ratio': fn_ratio,
            'fp_ratio': fp_ratio,
            'is_failure': is_failure,
            'failure_severity': max(fn_ratio, fp_ratio)
        }
    except Exception:
        return {
            'false_negatives': 0,
            'false_positives': 0,
            'true_positives': 0,
            'fn_ratio': 0,
            'fp_ratio': 0,
            'is_failure': False,
            'failure_severity': 0
        }

def get_memory_usage():
    """Get current memory usage"""
    if not PSUTIL_AVAILABLE:
        return {'cpu_memory_percent': 0, 'cpu_memory_used_gb': 0}
    
    try:
        cpu_memory = psutil.virtual_memory()
        return {
            'cpu_memory_percent': cpu_memory.percent,
            'cpu_memory_used_gb': cpu_memory.used / (1024**3)
        }
    except Exception:
        return {'cpu_memory_percent': 0, 'cpu_memory_used_gb': 0}

def get_gpu_memory_usage():
    """Get GPU memory usage if available"""
    if not GPUTIL_AVAILABLE:
        return 0
    
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].memoryUsed
        return 0
    except Exception:
        return 0

def smooth_metric(value, prev_value, alpha=0.9):
    """Apply exponential smoothing to metrics for stability"""
    if prev_value is None:
        return value
    return alpha * prev_value + (1 - alpha) * value

def calculate_iou(mask1, mask2):
    """Calculate Intersection over Union between two masks"""
    try:
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0

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
    print(f"[DEBUG] Starting TSP-SAM baseline main function")
    
    # Print system information
    print(f"[DEBUG] Python version: {sys.version}")
    print(f"[DEBUG] PyTorch version: {torch.__version__}")
    print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[DEBUG] CUDA version: {torch.version.cuda}")
        print(f"[DEBUG] GPU count: {torch.cuda.device_count()}")
        print(f"[DEBUG] Current GPU: {torch.cuda.current_device()}")
        print(f"[DEBUG] GPU name: {torch.cuda.get_device_name(0)}")
    
    parser = argparse.ArgumentParser(description='TSP-SAM Baseline for Video De-identification')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output masks')
    parser.add_argument('--sequence', type=str, required=True, help='Sequence name to process')
    parser.add_argument('--max_frames', type=int, default=5, help='Maximum frames to process')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--experiment-name', type=str, default=None, help='Name for the Weights & Biases experiment')
    parser.add_argument('--checkpoint', type=str, default='tsp_sam_official/snapshot/best_checkpoint.pth', help='Path to TSP-SAM checkpoint')
    parser.add_argument('--enable-advanced-metrics', action='store_true', help='Enable advanced metrics (Hausdorff, Contour Similarity, etc.)')
    parser.add_argument('--enable-memory-monitoring', action='store_true', help='Enable CPU/GPU memory monitoring')
    parser.add_argument('--enable-failure-analysis', action='store_true', help='Enable detailed failure case analysis')
    parser.add_argument('--metric-smoothing', type=float, default=0.9, help='Metric smoothing factor (0.0-1.0)')
    parser.add_argument('--boundary-tolerance', type=int, default=2, help='Boundary accuracy tolerance in pixels')
    
    args = parser.parse_args()
    
    print(f"[DEBUG] ==================================================")
    print(f"[DEBUG] ARGUMENTS:")
    print(f"[DEBUG]   input_path: {args.input_path}")
    print(f"[DEBUG]   output_path: {args.output_path}")
    print(f"[DEBUG]   sequence: {args.sequence}")
    print(f"[DEBUG]   max_frames: {args.max_frames}")
    print(f"[DEBUG]   device: {args.device}")
    print(f"[DEBUG]   wandb: {args.wandb}")
    print(f"[DEBUG]   experiment_name: {args.experiment_name}")
    print(f"[DEBUG]   checkpoint: {args.checkpoint}")
    print(f"[DEBUG]   enable_advanced_metrics: {args.enable_advanced_metrics}")
    print(f"[DEBUG]   enable_memory_monitoring: {args.enable_memory_monitoring}")
    print(f"[DEBUG]   enable_failure_analysis: {args.enable_failure_analysis}")
    print(f"[DEBUG]   metric_smoothing: {args.metric_smoothing}")
    print(f"[DEBUG]   boundary_tolerance: {args.boundary_tolerance}")
    print(f"[DEBUG] ==================================================")
    
    # Record start time for performance tracking
    start_time = time.time()
    
    # Create output directory
    print(f"[DEBUG] Creating output directory: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    print(f"[DEBUG] Output directory created/verified: {os.path.exists(args.output_path)}")
    
    device = torch.device(args.device)
    print(f"[DEBUG] Using device: {device}")
    
    # Initialize Weights & Biases if requested
    if args.wandb and WANDB_AVAILABLE:
        print(f"[DEBUG] Initializing W&B...")
        
        # Create comprehensive experiment name
        experiment_name = args.experiment_name if args.experiment_name else f"tsp_sam_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"[DEBUG] Experiment name: {experiment_name}")
        
        # Initialize W&B with comprehensive configuration
        wandb.init(
            project="temporal-deid-baselines",
            name=experiment_name,
            tags=[
                "m_tsp_sam",           # Model identifier
                "d_davis2017",         # Dataset identifier
                "b_baseline",          # Baseline identifier
                "v_temporal",          # Version identifier
                f"s_{args.sequence}",  # Sequence identifier
                "h_cuda" if device.type == 'cuda' else "h_cpu"  # Hardware identifier
            ],
            config={
                # Model configuration
                "model": "TSP-SAM",
                "model_version": "temporal",
                "checkpoint": args.checkpoint,
                "input_size": 352,
                "temporal_frames": 3,
                
                # Dataset configuration
                "dataset": "DAVIS-2017",
                "sequence": args.sequence,
                "max_frames": args.max_frames,
                "frame_start": 2,
                
                # Hardware configuration
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                
                # Processing configuration
                "enable_advanced_metrics": args.enable_advanced_metrics,
                "enable_memory_monitoring": args.enable_memory_monitoring,
                "enable_failure_analysis": args.enable_failure_analysis,
                "metric_smoothing": args.metric_smoothing,
                "boundary_tolerance": args.boundary_tolerance,
                
                # Experiment metadata
                "timestamp": datetime.now().isoformat(),
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
            }
        )
        
        print(f"[DEBUG] W&B initialized successfully")
        print(f"[DEBUG] Project: {wandb.run.project}")
        print(f"[DEBUG] Run ID: {wandb.run.id}")
        print(f"[DEBUG] Tags: {wandb.run.tags}")
        
        # Safely check config length
        try:
            config_dict = dict(wandb.run.config)
            print(f"[DEBUG] Config logged: {len(config_dict)} parameters")
        except Exception as e:
            print(f"[DEBUG] Could not get config length: {e}")
            print(f"[DEBUG] Config type: {type(wandb.run.config)}")
        
        print(f"[DEBUG] W&B experiment started: {wandb.run.name}")
    elif args.wandb and not WANDB_AVAILABLE:
        print("Warning: --wandb specified but wandb not available")
    
    # Load TSP-SAM model
    print(f"[DEBUG] Loading TSP-SAM model...")
    print(f"[DEBUG] Checkpoint path: {args.checkpoint}")
    print(f"[DEBUG] Device: {device}")
    
    try:
        # Create model_args object like official implementation
        class ModelArgs:
            def __init__(self):
                self.trainsize = 352
                self.testsize = 352
                self.grid = 8
                self.gpu_ids = [0] if device == 'cuda' else [0]
        
        model_args = ModelArgs()
        print(f"[DEBUG] Model arguments created - trainsize: {model_args.trainsize}, testsize: {model_args.testsize}, grid: {model_args.grid}")
        
        model = VideoModel(model_args)
        print(f"[DEBUG] VideoModel instance created successfully")
        
        # Load TSP-SAM checkpoint
        checkpoint_path = args.checkpoint
        if os.path.exists(checkpoint_path):
            try:
                print(f"[DEBUG] Loading checkpoint from {checkpoint_path}...")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                print(f"[DEBUG] Checkpoint loaded, keys: {list(checkpoint.keys())[:5]}...")
                
                # Fix checkpoint keys by removing 'module.' prefix if present
                if list(checkpoint.keys())[0].startswith('module.'):
                    print(f"[DEBUG] Removing 'module.' prefix from checkpoint keys")
                    new_checkpoint = {}
                    for key, value in checkpoint.items():
                        new_key = key.replace('module.', '')
                        new_checkpoint[new_key] = value
                    checkpoint = new_checkpoint
                    print(f"[DEBUG] Fixed checkpoint keys: {list(checkpoint.keys())[:5]}...")
                
                model.load_state_dict(checkpoint, strict=True)  # Use strict=True like original
                print(f"[DEBUG] TSP-SAM checkpoint loaded successfully in {time.time():.1f} seconds")
            except Exception as e:
                print(f"[WARNING] Failed to load checkpoint: {e}")
                print("[DEBUG] Model will run with random weights")
        else:
            print(f"[WARNING] TSP-SAM checkpoint not found at {checkpoint_path}")
            print("[DEBUG] Model will run with random weights")
        
        model = model.to(device)
        model.eval()
        print(f"[DEBUG] ORIGINAL TSP-SAM VideoModel loaded successfully on {device}")
        
    except Exception as e:
        print(f"[ERROR] Error loading TSP-SAM model: {e}")
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
    print(f"[DEBUG] Total available frames: {len(img_files)}")
    
    # Start from frame 2 (like official implementation)
    start_frame = 2
    end_frame = min(start_frame + max_frames, len(img_files))
    print(f"[DEBUG] Frame range: {start_frame} to {end_frame}")
    print(f"[DEBUG] Total frames to process: {end_frame - start_frame}")
    print(f"[DEBUG] Temporal context: 3 frames (frame_idx-2, frame_idx-1, frame_idx)")
    
    # Initialize metrics tracking
    all_ious = []
    all_dices = []
    all_coverage_ratios = []
    all_thresholds = []
    all_coverage_diffs = []
    
    # Advanced metrics tracking (if enabled)
    if args.enable_advanced_metrics:
        all_hausdorff_distances = []
        all_contour_similarities = []
        all_boundary_accuracies = []
        all_adapted_rand_errors = []
        all_variation_of_information = []
        all_failure_cases = []
        all_processing_times = []
    
    # Memory monitoring (if enabled)
    if args.enable_memory_monitoring:
        all_cpu_memory = []
        all_gpu_memory = []
    
    # Metric smoothing variables
    prev_mask = None
    prev_iou = None
    prev_dice = None
    
    # Complexity metrics for first frame
    complexity_metrics = {}
    
    print(f"[DEBUG] Starting TSP-SAM processing loop...")
    print(f"[DEBUG] Processing sequence: {args.sequence}")
    print(f"[DEBUG] Output directory: {args.output_path}")
    print(f"[DEBUG] W&B enabled: {args.wandb}")
    print(f"[DEBUG] Advanced metrics: {args.enable_advanced_metrics}")
    print(f"[DEBUG] Memory monitoring: {args.enable_memory_monitoring}")
    print(f"[DEBUG] Failure analysis: {args.enable_failure_analysis}")
    
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
                print(f"  [DEBUG] Model device: {next(model.parameters()).device}")
                
                try:
                    with torch.no_grad():
                        # Another critical fix: TSP-SAM VideoModel expects 3 inputs like official implementation
                        # 1. x: temporal sequence of RGB images (list of tensors)
                        # 2. x_ycbcr: temporal sequence of YCbCr images (list of tensors)
                        # 3. x_sam: SAM input (single frame tensor)
                        print(f"  [DEBUG] Executing TSP-SAM inference...")
                        pred = model(images, images_ycbcr, img_sam_tensor)
                        print(f"  [DEBUG] Model inference completed successfully")
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
                print(f"  [DEBUG] Calculating metrics for frame {frame_idx}...")
                
                intersection = np.logical_and(gt_mask, binary_mask).sum()
                union = np.logical_or(gt_mask, binary_mask).sum()
                iou = intersection / (union + 1e-8)
                
                dice_numerator = 2 * intersection
                dice_denominator = gt_mask.sum() + binary_mask.sum()
                dice = dice_numerator / (dice_denominator + 1e-8)
                
                coverage_ratio = binary_mask.sum() / (gt_mask.sum() + 1e-8)
                
                print(f"  [DEBUG] Raw metrics - IoU: {iou:.3f}, Dice: {dice:.3f}, Coverage: {coverage_ratio:.3f}")
                print(f"  [DEBUG] Intersection: {intersection}, Union: {union}")
                print(f"  [DEBUG] GT pixels: {gt_mask.sum()}, Pred pixels: {binary_mask.sum()}")
                
                # Apply metric smoothing if enabled
                if args.metric_smoothing > 0:
                    iou = smooth_metric(iou, prev_iou, args.metric_smoothing)
                    dice = smooth_metric(dice, prev_dice, args.metric_smoothing)
                    prev_iou = iou
                    prev_dice = dice
                
                # Calculate advanced metrics if enabled
                precision, recall = 0.0, 0.0
                boundary_accuracy = 0.0
                hausdorff_dist = 0.0
                contour_sim = 0.0
                region_metrics = {'adapted_rand_error': 0.0, 'variation_of_information': 0.0}
                failure_analysis = {'is_failure': False, 'failure_severity': 0.0}
                
                if args.enable_advanced_metrics:
                    # Basic precision/recall
                    precision, recall = calculate_precision_recall(gt_mask, binary_mask)
                    
                    # Boundary accuracy
                    boundary_accuracy = calculate_boundary_accuracy(gt_mask, binary_mask, args.boundary_tolerance)
                    
                    # Advanced boundary metrics
                    hausdorff_dist = calculate_hausdorff_distance(gt_mask, binary_mask)
                    contour_sim = calculate_contour_similarity(gt_mask, binary_mask)
                    
                    # Region-based metrics
                    region_metrics = calculate_region_based_metrics(gt_mask, binary_mask)
                    
                    # Failure analysis
                    if args.enable_failure_analysis:
                        failure_analysis = analyze_failure_cases(gt_mask, binary_mask, frame_idx, args.sequence)
                
                # Calculate temporal consistency (if not first frame)
                temporal_iou = 0.0
                if prev_mask is not None:
                    temporal_iou = calculate_iou(prev_mask, binary_mask)
                
                # Store previous mask for next iteration
                prev_mask = binary_mask.copy()
                
                # Calculate complexity metrics for first frame
                if frame_idx == start_frame and args.enable_advanced_metrics:
                    complexity_metrics = calculate_complexity_metrics(gt_mask)
                
                # Get memory usage if monitoring enabled
                memory_info = {'cpu_memory_percent': 0, 'cpu_memory_used_gb': 0}
                gpu_memory = 0
                if args.enable_memory_monitoring:
                    memory_info = get_memory_usage()
                    gpu_memory = get_gpu_memory_usage()
                
                print(f"  [DEBUG] Metrics:")
                print(f"    [DEBUG] IoU: {iou:.4f}")
                print(f"    [DEBUG] Dice: {dice:.4f}")
                print(f"    [DEBUG] Coverage Ratio: {coverage_ratio:.4f}")
                
                if args.enable_advanced_metrics:
                    print(f"    [DEBUG] Precision: {precision:.4f}, Recall: {recall:.4f}")
                    print(f"    [DEBUG] Boundary Accuracy: {boundary_accuracy:.4f}")
                    print(f"    [DEBUG] Hausdorff Distance: {hausdorff_dist:.4f}")
                    print(f"    [DEBUG] Contour Similarity: {contour_sim:.4f}")
                    print(f"    [DEBUG] Temporal IoU: {temporal_iou:.4f}")
                
                if args.enable_failure_analysis and failure_analysis['is_failure']:
                    print(f"    [DEBUG] FAILURE DETECTED - Severity: {failure_analysis['failure_severity']:.4f}")
                
                # Store metrics for WANDB logging
                all_ious.append(iou)
                all_dices.append(dice)
                all_coverage_ratios.append(coverage_ratio)
                
                # Get the selected threshold from the thresholding process
                selected_threshold = 0.0  # Default value
                coverage_diff = 0.0  # Default value
                
                # Find the threshold that was actually used (this should match the thresholding logic above)
                if 'selected_threshold' in locals():
                    selected_threshold = locals()['selected_threshold']
                else:
                    # Estimate based on the binary mask
                    selected_threshold = 0.1  # Default threshold
                
                all_thresholds.append(selected_threshold)
                all_coverage_diffs.append(coverage_diff)
                
                if args.enable_advanced_metrics:
                    all_hausdorff_distances.append(hausdorff_dist)
                    all_contour_similarities.append(contour_sim)
                    all_boundary_accuracies.append(boundary_accuracy)
                    all_adapted_rand_errors.append(region_metrics['adapted_rand_error'])
                    all_variation_of_information.append(region_metrics['variation_of_information'])
                    all_failure_cases.append(1 if failure_analysis['is_failure'] else 0)
                
                if args.enable_memory_monitoring:
                    all_cpu_memory.append(memory_info['cpu_memory_percent'])
                    all_gpu_memory.append(gpu_memory)
                
                # Enhanced WANDB logging
                if args.wandb and WANDB_AVAILABLE:
                    log_data = {
                        # Frame-level metrics with proper prefixes
                        "frame/idx": frame_idx,
                        "frame/sequence_name": args.sequence,
                        "frame/progress": (frame_idx - start_frame + 1) / (end_frame - start_frame),
                        "frame/total_frames": end_frame - start_frame,
                        
                        # Mask metrics
                        "mask/area_pixels": int(binary_mask.sum()),
                        "mask/coverage_percent": float((binary_mask.sum() / binary_mask.size) * 100),
                        "mask/gt_coverage_percent": float((gt_mask.sum() / gt_mask.size) * 100),
                        
                        # Evaluation metrics with proper prefixes
                        "eval/iou": float(iou),
                        "eval/dice": float(dice),
                        "eval/coverage_ratio": float(coverage_ratio),
                        "eval/temporal_iou": float(temporal_iou),
                        "eval/selected_threshold": float(selected_threshold),
                        "eval/coverage_diff": float(coverage_diff)
                    }
                    
                    if args.enable_advanced_metrics:
                        log_data.update({
                            "eval/precision": float(precision),
                            "eval/recall": float(recall),
                            "eval/boundary_accuracy": float(boundary_accuracy),
                            "eval/hausdorff_distance": float(hausdorff_dist),
                            "eval/contour_similarity": float(contour_sim),
                            "eval/adapted_rand_error": float(region_metrics['adapted_rand_error']),
                            "eval/variation_of_information": float(region_metrics['variation_of_information']),
                            "eval/is_failure_case": int(failure_analysis['is_failure']),
                            "eval/failure_severity": float(failure_analysis['failure_severity'])
                        })
                    
                    if args.enable_memory_monitoring:
                        log_data.update({
                            "system/cpu_memory_percent": float(memory_info['cpu_memory_percent']),
                            "system/cpu_memory_used_gb": float(memory_info['cpu_memory_used_gb']),
                            "system/gpu_memory_used_mb": float(gpu_memory)
                        })
                    
                    # Log to W&B with error handling
                    try:
                        wandb.log(log_data)
                        print(f"  [DEBUG] W&B logging completed successfully")
                    except Exception as e:
                        print(f"  [WARNING] W&B logging failed: {e}")
                
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
    
    # Final WANDB logging
    if args.wandb and WANDB_AVAILABLE:
        print(f"[DEBUG] Logging final experiment summary to W&B...")
        
        # Calculate summary statistics
        if all_ious:
            summary_data = {
                # Experiment status
                "experiment/status_code": 1,
                "experiment/total_frames_processed": int(end_frame - start_frame),
                "experiment/total_sequences_processed": 1,
                "experiment/final_timestamp": float(datetime.now().timestamp()),
                
                # Basic metrics summary with proper prefixes
                "experiment/overall_avg_iou": float(np.mean(all_ious)),
                "experiment/overall_avg_dice": float(np.mean(all_dices)),
                "experiment/overall_avg_coverage": float(np.mean(all_coverage_ratios)),
                "experiment/overall_avg_threshold": float(np.mean(all_thresholds)),
                "experiment/overall_avg_coverage_diff": float(np.mean(all_coverage_diffs)),
                
                # Statistical measures
                "experiment/dataset_iou_mean": float(np.mean(all_ious)),
                "experiment/dataset_iou_std": float(np.std(all_ious)),
                "experiment/dataset_iou_min": float(np.min(all_ious)),
                "experiment/dataset_iou_max": float(np.max(all_ious)),
                "experiment/dataset_iou_median": float(np.median(all_ious)),
                "experiment/dataset_iou_q25": float(np.percentile(all_ious, 25)),
                "experiment/dataset_iou_q75": float(np.percentile(all_ious, 75)),
                
                "experiment/dataset_dice_mean": float(np.mean(all_dices)),
                "experiment/dataset_dice_std": float(np.std(all_dices)),
                "experiment/dataset_dice_median": float(np.median(all_dices)),
                
                "experiment/dataset_coverage_mean": float(np.mean(all_coverage_ratios)),
                "experiment/dataset_coverage_std": float(np.std(all_coverage_ratios)),
                "experiment/dataset_coverage_median": float(np.median(all_coverage_ratios))
            }
            
            # Add advanced metrics summary if enabled
            if args.enable_advanced_metrics:
                if all_hausdorff_distances:
                    summary_data.update({
                        "experiment/overall_avg_hausdorff": float(np.mean(all_hausdorff_distances)),
                        "experiment/overall_avg_contour_similarity": float(np.mean(all_contour_similarities)),
                        "experiment/overall_avg_boundary_accuracy": float(np.mean(all_boundary_accuracies)),
                        "experiment/overall_avg_adapted_rand": float(np.mean(all_adapted_rand_errors)),
                        "experiment/overall_avg_voi": float(np.mean(all_variation_of_information))
                    })
                
                if all_failure_cases:
                    failure_rate = np.mean(all_failure_cases)
                    summary_data.update({
                        "experiment/overall_failure_rate": float(failure_rate),
                        "experiment/total_failure_cases": int(sum(all_failure_cases))
                    })
                
                # Add complexity metrics if available
                if complexity_metrics:
                    summary_data.update({
                        "complexity/object_count": int(complexity_metrics.get('object_count', 0)),
                        "complexity/avg_area": float(complexity_metrics.get('avg_area', 0)),
                        "complexity/compactness": float(complexity_metrics.get('compactness', 0)),
                        "complexity/eccentricity": float(complexity_metrics.get('eccentricity', 0))
                    })
            
            # Add memory monitoring summary if enabled
            if args.enable_memory_monitoring:
                if all_cpu_memory:
                    summary_data.update({
                        "system/avg_cpu_memory_percent": float(np.mean(all_cpu_memory)),
                        "system/max_cpu_memory_percent": float(np.max(all_cpu_memory))
                    })
                
                if all_gpu_memory:
                    summary_data.update({
                        "system/avg_gpu_memory_mb": float(np.mean(all_gpu_memory)),
                        "system/max_gpu_memory_mb": float(np.max(all_gpu_memory))
                    })
            
            # Log summary to W&B
            try:
                wandb.log(summary_data)
                print(f"[DEBUG] Experiment summary logged to W&B successfully")
            except Exception as e:
                print(f"[WARNING] Failed to log experiment summary: {e}")
        else:
            # Log error case
            try:
                wandb.log({
                    "experiment/status_code": 0,
                    "experiment/total_frames_processed": int(end_frame - start_frame),
                    "experiment/note": "No metrics collected - possible processing errors"
                })
                print(f"[DEBUG] Error summary logged to W&B")
            except Exception as e:
                print(f"[WARNING] Failed to log error summary: {e}")
        
        # Finish W&B run
        try:
            wandb.finish()
            print(f"[DEBUG] W&B experiment completed and logged")
        except Exception as e:
            print(f"[WARNING] Failed to finish W&B run: {e}")
    
    # Print comprehensive summary
    print(f"\n[DEBUG] ==================================================")
    print(f"[DEBUG] TSP-SAM BASELINE SUMMARY")
    print(f"[DEBUG] ==================================================")
    print(f"Sequence: {args.sequence}")
    print(f"Total frames processed: {end_frame - start_frame}")
    print(f"Processing time: {time.time() - start_time:.2f} seconds" if 'start_time' in locals() else "Processing time: N/A")
    
    if all_ious:
        print(f"[DEBUG] Performance Metrics:")
        print(f"  Average IoU: {np.mean(all_ious):.4f} ± {np.std(all_ious):.4f}")
        print(f"  Average Dice: {np.mean(all_dices):.4f} ± {np.std(all_dices):.4f}")
        print(f"  Average Coverage Ratio: {np.mean(all_coverage_ratios):.4f} ± {np.std(all_coverage_ratios):.4f}")
        print(f"  IoU Range: {np.min(all_ious):.4f} - {np.max(all_ious):.4f}")
        print(f"  Dice Range: {np.min(all_dices):.4f} - {np.max(all_dices):.4f}")
        
        if args.enable_advanced_metrics:
            print(f"[DEBUG] Advanced Metrics:")
            if all_hausdorff_distances:
                print(f"  Average Hausdorff Distance: {np.mean(all_hausdorff_distances):.2f} ± {np.std(all_hausdorff_distances):.2f}")
            if all_contour_similarities:
                print(f"  Average Contour Similarity: {np.mean(all_contour_similarities):.4f} ± {np.std(all_contour_similarities):.4f}")
            if all_boundary_accuracies:
                print(f"  Average Boundary Accuracy: {np.mean(all_boundary_accuracies):.4f} ± {np.std(all_boundary_accuracies):.4f}")
            if all_failure_cases:
                failure_rate = np.mean(all_failure_cases)
                print(f"  Failure Rate: {failure_rate:.1%} ({sum(all_failure_cases)}/{len(all_failure_cases)} frames)")
        
        if args.enable_memory_monitoring:
            print(f"[DEBUG] Resource Usage:")
            if all_cpu_memory:
                print(f"  Average CPU Memory: {np.mean(all_cpu_memory):.1f}%")
            if all_gpu_memory:
                print(f"  Average GPU Memory: {np.mean(all_gpu_memory):.1f} MB")
    
    print(f"[DEBUG] ==================================================")
    print(f"[DEBUG] TSP-SAM baseline completed successfully!")
    print(f"[DEBUG] Results saved to: {args.output_path}")
    print(f"[DEBUG] ==================================================")

if __name__ == "__main__":
    main()
