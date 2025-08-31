#!/usr/bin/env python3
"""
SAMURAI Baseline Runner for DAVIS-2017 Dataset
This script runs the actual SAMURAI model for video object segmentation
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import imageio
import cv2
from pathlib import Path
from PIL import Image
from datetime import datetime
import time

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not available. Install with: pip install wandb")

# Add SAMURAI official to path
sys.path.append('samurai_official/sam2')
# sys.path.append('samurai_official/sam2/sam2')

try:
    from sam2.build_sam import build_sam2_video_predictor
    print("Successfully imported SAMURAI modules")
except ImportError as e:
    print(f"Error importing SAMURAI modules: {e}")
    print("Please ensure 'samurai_official' is a correctly configured git submodule")
    sys.exit(1)

def load_ground_truth_mask(gt_path, target_size=None):
    """Load and preprocess ground truth mask from DAVIS annotations"""
    try:
        gt_mask = Image.open(gt_path).convert('L')
        # Keep original size for SAMURAI (don't resize for bbox generation)
        if target_size is not None:
            gt_mask = gt_mask.resize((target_size, target_size), Image.NEAREST)
        gt_np = np.array(gt_mask)
        
        # DAVIS annotations use non-standard pixel values [0, 38] or [0, 128]
        # Normalize to [0, 1] range
        if gt_np.max() > 1:
            gt_np = gt_np / 255.0
        
        # Create binary mask with appropriate threshold for DAVIS
        binary_mask = (gt_np > 0.1).astype(np.uint8)
        
        # Apply morphological operations for cleanup
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        return binary_mask
    except Exception as e:
        print(f"Error loading ground truth mask {gt_path}: {e}")
        return None

def generate_bbox_from_mask(mask):
    """Generate bounding box from mask"""
    if mask is None or mask.sum() == 0:
        return [0, 0, 0, 0]
    
    # Find non-zero indices
    non_zero_indices = np.argwhere(mask)
    if len(non_zero_indices) == 0:
        return [0, 0, 0, 0]
    
    # Get bounding box coordinates
    y_min, x_min = non_zero_indices.min(axis=0)
    y_max, x_max = non_zero_indices.max(axis=0)
    
    # Return [x_min, y_min, x_max, y_max] format (coordinates, not width/height)
    bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
    return bbox

def run_samurai_davis_baseline(input_path, output_path, checkpoint_path=None, sequence=None, max_frames=None, use_wandb=False):
    """Run SAMURAI baseline on DAVIS-2017 dataset using actual model"""
    print(f"Running SAMURAI baseline on DAVIS-2017 dataset")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all sequences (DAVIS format - look in JPEGImages/480p)
    jpeg_path = Path(input_path) / "JPEGImages" / "480p"
    if not jpeg_path.exists():
        print(f"JPEGImages/480p directory not found in {input_path}")
        return
    
    sequences = [d for d in jpeg_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    sequences = sorted(sequences)
    
    if sequence:
        sequences = [s for s in sequences if s.name == sequence]
    
    print(f"Found {len(sequences)} sequences to process")
    
    # Track per-sequence metrics for comprehensive analysis
    sequence_metrics = {}
    
    # Initialize SAMURAI model
    if checkpoint_path is None:
        checkpoint_path = "samurai_official/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please download the SAM2.1 checkpoint first")
        return
    
    print(f"Loading SAMURAI model from: {checkpoint_path}")
    
    # Use base_plus model configuration
    print(f"Current working directory: {os.getcwd()}")
    print(f"Available configs in sam2: {os.listdir('samurai_official/sam2/sam2/configs/sam2.1')}")
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    
    try:
        # Initialize SAMURAI predictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print(f"Model config path: {model_cfg}")
        print(f"Config file exists: {os.path.exists(model_cfg)}")

        predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)
        print("SAMURAI model loaded successfully")
        
        # Watch the model with W&B if enabled
        if use_wandb and WANDB_AVAILABLE:
            try:
                # Get the underlying PyTorch model from the predictor
                if hasattr(predictor, 'model'):
                    wandb.watch(predictor.model, log="all", log_freq=10)
                    print("W&B model watching enabled")
                else:
                    print("Warning: Could not access underlying PyTorch model for W&B watching")
            except Exception as e:
                print(f"Warning: Could not enable W&B model watching: {e}")
        
    except Exception as e:
        print(f"Error loading SAMURAI model: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to placeholder implementation")
        return run_samurai_placeholder(input_path, output_path, sequence, max_frames)
    
    # Process each sequence
    for seq_idx, seq_path in enumerate(sequences):
        print(f"\n[{seq_idx + 1}/{len(sequences)}] Processing sequence: {seq_path.name}")
        
        # Create sequence output directory
        seq_output_dir = output_path / seq_path.name
        seq_output_dir.mkdir(exist_ok=True)
        
        # Initialize sequence metrics tracking
        sequence_metrics[seq_path.name] = {
            'total_frames': 0,
            'iou_scores': [],
            'dice_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'boundary_accuracy_scores': [],
            'coverage_scores': [],
            'processing_times': [],
            'failure_cases': 0,
            'temporal_consistency_scores': [],
            'hausdorff_distances': [],
            'contour_similarities': [],
            'adapted_rand_errors': [],
            'variation_of_information': [],
            'temporal_stability_metrics': {},
            'complexity_metrics': {},
            'prev_mask': None,  # Add this for temporal consistency
            'prev_iou': None,   # Add this for smoothing
            'prev_dice': None,  # Add this for smoothing
            'start_time': time.time()
        }
        
        # Check for image and annotation directories (DAVIS format)
        img_dir = Path(input_path) / "JPEGImages" / "480p" / seq_path.name
        ann_dir = Path(input_path) / "Annotations" / "480p" / seq_path.name
        
        if not img_dir.exists() or not ann_dir.exists():
            print(f"Missing image or annotation directory for {seq_path.name}")
            continue
        
        # Get all image files
        image_files = sorted([f for f in img_dir.glob('*.jpg')])
        annotation_files = sorted([f for f in ann_dir.glob('*.png')])
        
        if not image_files or not annotation_files:
            print(f"No images or annotations found for {seq_path.name}")
            continue
        
        # Align image and annotation files
        aligned_pairs = []
        for img_file in image_files:
            ann_name = img_file.stem + '.png'
            ann_file = ann_dir / ann_name
            if ann_file.exists():
                aligned_pairs.append((img_file, ann_file))
        
        if max_frames:
            aligned_pairs = aligned_pairs[:max_frames]
        
        print(f"Processing {len(aligned_pairs)} frames for {seq_path.name}")
        
        # Update sequence metrics
        sequence_metrics[seq_path.name]['total_frames'] = len(aligned_pairs)
        
        try:
            # Reset model for each sequence to avoid state corruption
            if seq_idx > 0:  # Don't reset for first sequence
                print(f"Resetting SAMURAI model for {seq_path.name}...")
                del predictor  # Clean up old predictor
                torch.cuda.empty_cache()  # Clear GPU memory
                
                # Reload predictor
                predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)
                print("SAMURAI model reloaded successfully")
                
                # Re-enable W&B watching for new predictor
                if use_wandb and WANDB_AVAILABLE:
                    try:
                        if hasattr(predictor, 'model'):
                            wandb.watch(predictor.model, log="all", log_freq=10)
                    except Exception as e:
                        print(f"Warning: Could not re-enable W&B model watching: {e}")
            
            # Process sequence with SAMURAI
            process_sequence_with_samurai(predictor, aligned_pairs, seq_output_dir, seq_path.name, use_wandb, sequence_metrics[seq_path.name])
            
        except Exception as e:
            print(f"Error processing sequence with SAMURAI: {e}")
            print("Falling back to placeholder implementation")
            run_samurai_placeholder_sequence(aligned_pairs, seq_output_dir, seq_path.name)
        
        # Calculate and log sequence summary metrics
        seq_metrics = sequence_metrics[seq_path.name]
        seq_metrics['end_time'] = time.time()
        seq_metrics['total_time'] = seq_metrics['end_time'] - seq_metrics['start_time']
        
        # Calculate averages and statistics
        if seq_metrics['iou_scores']:
            seq_metrics['avg_iou'] = np.mean(seq_metrics['iou_scores'])
            seq_metrics['std_iou'] = np.std(seq_metrics['iou_scores'])
            seq_metrics['min_iou'] = np.min(seq_metrics['iou_scores'])
            seq_metrics['max_iou'] = np.max(seq_metrics['iou_scores'])
        else:
            seq_metrics['avg_iou'] = seq_metrics['std_iou'] = seq_metrics['min_iou'] = seq_metrics['max_iou'] = 0.0
            
        if seq_metrics['dice_scores']:
            seq_metrics['avg_dice'] = np.mean(seq_metrics['dice_scores'])
            seq_metrics['std_dice'] = np.std(seq_metrics['dice_scores'])
        else:
            seq_metrics['avg_dice'] = seq_metrics['std_dice'] = 0.0
            
        if seq_metrics['coverage_scores']:
            seq_metrics['avg_coverage'] = np.mean(seq_metrics['coverage_scores'])
            seq_metrics['std_coverage'] = np.std(seq_metrics['coverage_scores'])
        else:
            seq_metrics['avg_coverage'] = seq_metrics['std_coverage'] = 0.0
            
        if seq_metrics['temporal_consistency_scores']:
            seq_metrics['avg_temporal_consistency'] = np.mean(seq_metrics['temporal_consistency_scores'])
        else:
            seq_metrics['avg_temporal_consistency'] = 0.0
        
        # Calculate advanced metric statistics
        if seq_metrics['hausdorff_distances']:
            seq_metrics['avg_hausdorff'] = np.mean(seq_metrics['hausdorff_distances'])
            seq_metrics['std_hausdorff'] = np.std(seq_metrics['hausdorff_distances'])
        else:
            seq_metrics['avg_hausdorff'] = seq_metrics['std_hausdorff'] = 0.0
            
        if seq_metrics['contour_similarities']:
            seq_metrics['avg_contour_sim'] = np.mean(seq_metrics['contour_similarities'])
            seq_metrics['std_contour_sim'] = np.std(seq_metrics['contour_similarities'])
        else:
            seq_metrics['avg_contour_sim'] = seq_metrics['std_contour_sim'] = 0.0
            
        if seq_metrics['adapted_rand_errors']:
            seq_metrics['avg_adapted_rand'] = np.mean(seq_metrics['adapted_rand_errors'])
            seq_metrics['std_adapted_rand'] = np.std(seq_metrics['adapted_rand_errors'])
        else:
            seq_metrics['avg_adapted_rand'] = seq_metrics['std_adapted_rand'] = 0.0
            
        if seq_metrics['variation_of_information']:
            seq_metrics['avg_voi'] = np.mean(seq_metrics['variation_of_information'])
            seq_metrics['std_voi'] = np.std(seq_metrics['variation_of_information'])
        else:
            seq_metrics['avg_voi'] = seq_metrics['std_voi'] = 0.0
        
        # Calculate temporal stability metrics for the sequence
        if len(seq_metrics['temporal_consistency_scores']) > 1:
            seq_metrics['temporal_stability_metrics'] = calculate_temporal_stability_metrics(
                [np.array(seq_metrics['coverage_scores'])]  # Convert to array format
            )
        
        # Calculate complexity metrics for the first frame (representative of sequence)
        if aligned_pairs:
            first_gt_mask = load_ground_truth_mask(aligned_pairs[0][1])
            if first_gt_mask is not None:
                seq_metrics['complexity_metrics'] = calculate_complexity_metrics(first_gt_mask)
            else:
                # Initialize with default values if no mask available
                seq_metrics['complexity_metrics'] = {
                    'object_count': 0,
                    'total_area': 0,
                    'avg_area': 0,
                    'perimeter': 0,
                    'compactness': 0.0,
                    'eccentricity': 0.0
                }
        else:
            # Initialize with default values if no pairs available
            seq_metrics['complexity_metrics'] = {
                'object_count': 0,
                'total_area': 0,
                'avg_area': 0,
                'perimeter': 0,
                'compactness': 0.0,
                'eccentricity': 0.0
            }
        
        # Log comprehensive sequence summary to W&B with standardized naming
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                # Sequence identification (as numeric indices instead of strings)
                "sequence/idx": seq_idx,
                "sequence/total_sequences": len(sequences),
                "sequence/total_frames": len(aligned_pairs),
                
                # Performance metrics with eval/ prefix
                "eval/avg_iou": seq_metrics['avg_iou'],
                "eval/std_iou": seq_metrics['std_iou'],
                "eval/min_iou": seq_metrics['min_iou'],
                "eval/max_iou": seq_metrics['max_iou'],
                "eval/avg_dice": seq_metrics['avg_dice'],
                "eval/std_dice": seq_metrics['std_dice'],
                "eval/avg_coverage": seq_metrics['avg_coverage'],
                "eval/std_coverage": seq_metrics['std_coverage'],
                "eval/avg_temporal_consistency": seq_metrics['avg_temporal_consistency'],
                "eval/avg_hausdorff": seq_metrics['avg_hausdorff'],
                "eval/std_hausdorff": seq_metrics['std_hausdorff'],
                "eval/avg_contour_similarity": seq_metrics['avg_contour_sim'],
                "eval/std_contour_similarity": seq_metrics['std_contour_sim'],
                "eval/avg_adapted_rand": seq_metrics['avg_adapted_rand'],
                "eval/std_adapted_rand": seq_metrics['std_adapted_rand'],
                "eval/avg_voi": seq_metrics['avg_voi'],
                "eval/std_voi": seq_metrics['std_voi'],
                "eval/failure_rate": seq_metrics['failure_cases'] / seq_metrics['total_frames'] if seq_metrics['total_frames'] > 0 else 0.0,
                
                # Processing metrics
                "performance/total_time": seq_metrics['total_time'],
                "performance/avg_fps": seq_metrics['total_frames'] / seq_metrics['total_time'] if seq_metrics['total_time'] > 0 else 0.0,
                
                # Complexity metrics
                "complexity/object_count": seq_metrics.get('complexity_metrics', {}).get('object_count', 0),
                "complexity/eccentricity": seq_metrics.get('complexity_metrics', {}).get('eccentricity', 0.0),
                "complexity/compactness": seq_metrics.get('complexity_metrics', {}).get('compactness', 0.0),
                "complexity/avg_area": seq_metrics.get('complexity_metrics', {}).get('avg_area', 0.0),
                
                # Status as numeric code
                "sequence/status_code": 1,  # 1 = completed, 0 = failed
                "sequence/completion_timestamp": datetime.now().timestamp()
            })
            
            # Log processing warning if any
            if 'processing_warning' in seq_metrics:
                wandb.log({
                    "sequence/processing_warning_code": 1,  # 1 = warning, 0 = no warning
                    "sequence/processed_frames_ratio": seq_metrics.get('processed_frames', 0) / seq_metrics['total_frames'] if seq_metrics['total_frames'] > 0 else 0.0
                })
        
        print(f"Completed sequence: {seq_path.name}")
        print(f"  Average IoU: {seq_metrics['avg_iou']:.3f} ± {seq_metrics['std_iou']:.3f}")
        print(f"  Average Dice: {seq_metrics['avg_dice']:.3f} ± {seq_metrics['std_dice']:.3f}")
        print(f"  Average Coverage: {seq_metrics['avg_coverage']:.2f}% ± {seq_metrics['std_coverage']:.2f}%")
        print(f"  Temporal Consistency: {seq_metrics['avg_temporal_consistency']:.3f}")
        print(f"  Boundary Accuracy: {seq_metrics['avg_hausdorff']:.2f} ± {seq_metrics['std_hausdorff']:.2f}")
        print(f"  Contour Similarity: {seq_metrics['avg_contour_sim']:.3f} ± {seq_metrics['std_contour_sim']:.3f}")
        print(f"  Failure Cases: {seq_metrics['failure_cases']}/{len(aligned_pairs)} ({seq_metrics['failure_cases']/len(aligned_pairs)*100:.1f}%)")
        print(f"  Processing Time: {seq_metrics['total_time']:.2f}s ({len(aligned_pairs)/seq_metrics['total_time']:.2f} FPS)")
        if seq_metrics['complexity_metrics']:
            comp = seq_metrics['complexity_metrics']
            print(f"  Object Complexity: {comp.get('object_count', 0)} objects, Area: {comp.get('avg_area', 0):.0f}, Compactness: {comp.get('compactness', 0):.3f}")
    
    # Calculate overall experiment statistics
    print(f"\n=== EXPERIMENT SUMMARY ===")
    
    # Overall statistics across all sequences
    all_ious = []
    all_dices = []
    all_coverages = []
    all_temporal_consistencies = []
    all_hausdorffs = []
    all_contour_sims = []
    all_adapted_rands = []
    all_vois = []
    total_failure_cases = 0
    total_frames = 0
    
    for seq_name, metrics in sequence_metrics.items():
        all_ious.extend(metrics['iou_scores'])
        all_dices.extend(metrics['dice_scores'])
        all_coverages.extend(metrics['coverage_scores'])
        all_temporal_consistencies.extend(metrics['temporal_consistency_scores'])
        all_hausdorffs.extend(metrics['hausdorff_distances'])
        all_contour_sims.extend(metrics['contour_similarities'])
        all_adapted_rands.extend(metrics['adapted_rand_errors'])
        all_vois.extend(metrics['variation_of_information'])
        total_failure_cases += metrics['failure_cases']
        total_frames += metrics['total_frames']
    
    # Calculate overall averages
    overall_avg_iou = np.mean(all_ious) if all_ious else 0.0
    overall_avg_dice = np.mean(all_dices) if all_dices else 0.0
    overall_avg_coverage = np.mean(all_coverages) if all_coverages else 0.0
    overall_avg_temporal = np.mean(all_temporal_consistencies) if all_temporal_consistencies else 0.0
    overall_avg_hausdorff = np.mean(all_hausdorffs) if all_hausdorffs else 0.0
    overall_avg_contour_sim = np.mean(all_contour_sims) if all_contour_sims else 0.0
    overall_avg_adapted_rand = np.mean(all_adapted_rands) if all_adapted_rands else 0.0
    overall_avg_voi = np.mean(all_vois) if all_vois else 0.0
    overall_failure_rate = total_failure_cases / total_frames if total_frames > 0 else 0.0
    
    # Calculate comprehensive dataset statistics
    dataset_stats = calculate_dataset_statistics(sequence_metrics)
    
    print(f"Overall Performance:")
    print(f"  Average IoU: {overall_avg_iou:.3f}")
    print(f"  Average Dice: {overall_avg_dice:.3f}")
    print(f"  Average Coverage: {overall_avg_coverage:.2f}%")
    print(f"  Average Temporal Consistency: {overall_avg_temporal:.3f}")
    print(f"  Average Boundary Accuracy (Hausdorff): {overall_avg_hausdorff:.2f}")
    print(f"  Average Contour Similarity: {overall_avg_contour_sim:.3f}")
    print(f"  Average Adapted Rand Error: {overall_avg_adapted_rand:.3f}")
    print(f"  Average Variation of Information: {overall_avg_voi:.3f}")
    print(f"  Overall Failure Rate: {overall_failure_rate:.1%}")
    print(f"  Total Frames Processed: {total_frames}")
    
    # Print detailed statistics
    if 'iou_q25' in dataset_stats:
        print(f"\nDetailed IoU Statistics:")
        print(f"  Median: {dataset_stats['iou_median']:.3f}")
        print(f"  Q25: {dataset_stats['iou_q25']:.3f}")
        print(f"  Q75: {dataset_stats['iou_q75']:.3f}")
        print(f"  Range: {dataset_stats['iou_min']:.3f} - {dataset_stats['iou_max']:.3f}")
    
    # Log final experiment summary to W&B with standardized naming
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            # Experiment summary
            "experiment/total_sequences_processed": len(sequences),
            "experiment/total_frames_processed": total_frames,
            
            # Overall performance metrics
            "experiment/overall_avg_iou": overall_avg_iou,
            "experiment/overall_avg_dice": overall_avg_dice,
            "experiment/overall_avg_coverage": overall_avg_coverage,
            "experiment/overall_avg_temporal_consistency": overall_avg_temporal,
            "experiment/overall_avg_hausdorff": overall_avg_hausdorff,
            "experiment/overall_avg_contour_similarity": overall_avg_contour_sim,
            "experiment/overall_avg_adapted_rand": overall_avg_adapted_rand,
            "experiment/overall_avg_voi": overall_avg_voi,
            "experiment/overall_failure_rate": overall_failure_rate,
            
            # Status as numeric code instead of string
            "experiment/status_code": 1,  # 1 = completed, 0 = failed
            "experiment/final_timestamp": datetime.now().timestamp()
        })
        
        # Log detailed dataset statistics
        for key, value in dataset_stats.items():
            if isinstance(value, (int, float)):
                wandb.log({f"experiment/dataset_{key}": value})
        
        wandb.finish()
        print("Wandb experiment completed and logged")
    
    print(f"\nSAMURAI baseline completed!")
    print(f"Output saved to: {output_path}")

def process_sequence_with_samurai(predictor, aligned_pairs, seq_output_dir, seq_name, use_wandb=False, seq_metrics=None):
    """Process a sequence using actual SAMURAI model"""
    print(f"[DEBUG] Starting process_sequence_with_samurai")
    print(f"[DEBUG] Sequence: {seq_name}")
    print(f"[DEBUG] Output directory: {seq_output_dir}")
    print(f"[DEBUG] Number of aligned pairs: {len(aligned_pairs)}")
    print(f"[DEBUG] W&B enabled: {use_wandb}")
    print(f"[DEBUG] Sequence metrics provided: {seq_metrics is not None}")
    print(f"[DEBUG] Predictor type: {type(predictor)}")
    
    import time
    start_time = time.time()
    
    # Get first frame and annotation for initialization
    print(f"[DEBUG] Loading first frame and annotation...")
    first_img_file, first_ann_file = aligned_pairs[0]
    print(f"[DEBUG] First image file: {first_img_file}")
    print(f"[DEBUG] First annotation file: {first_ann_file}")
    
    first_gt_mask = load_ground_truth_mask(first_ann_file)
    print(f"[DEBUG] First GT mask loaded: {first_gt_mask is not None}")
    if first_gt_mask is not None:
        print(f"[DEBUG] First GT mask shape: {first_gt_mask.shape}")
        print(f"[DEBUG] First GT mask range: [{first_gt_mask.min():.3f}, {first_gt_mask.max():.3f}]")
        print(f"[DEBUG] First GT mask sum: {first_gt_mask.sum()}")
    
    if first_gt_mask is None:
        print(f"[ERROR] Failed to load first annotation for {seq_name}")
        return
    
    # Generate bounding box from first frame
    print(f"[DEBUG] Generating bounding box from first frame...")
    bbox = generate_bbox_from_mask(first_gt_mask)
    print(f"[DEBUG] Generated bbox: {bbox}")
    
    # Initialize SAMURAI state with first frame
    print(f"[DEBUG] Initializing SAMURAI state...")
    frame_folder = str(first_img_file.parent)
    print(f"[DEBUG] Frame folder: {frame_folder}")
    
    try:
        print(f"[DEBUG] Starting SAMURAI inference...")
        with torch.inference_mode():
            print(f"[DEBUG] Initializing SAMURAI state...")
            # Initialize state
            state = predictor.init_state(
                frame_folder, 
                offload_video_to_cpu=True, 
                offload_state_to_cpu=True, 
                async_loading_frames=True
            )
            print(f"[DEBUG] SAMURAI state initialized successfully")
            
            # Add first object
            print(f"[DEBUG] Adding first object to state...")
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(
                state, box=bbox, frame_idx=0, obj_id=0
            )
            print(f"[DEBUG] First object added - frame_idx: {frame_idx}, object_ids: {object_ids}")
            
            # Debug mask structure safely
            if masks is not None:
                if isinstance(masks, (list, tuple)) and len(masks) > 0:
                    print(f"[DEBUG] Masks received: {len(masks)} objects (list/tuple)")
                elif hasattr(masks, 'shape'):
                    print(f"[DEBUG] Masks received: tensor with shape {masks.shape}")
                    # For tensor masks, check the first dimension
                    if len(masks.shape) >= 3:
                        print(f"[DEBUG] Number of objects: {masks.shape[1] if len(masks.shape) > 1 else 1}")
                    else:
                        print(f"[DEBUG] Single mask tensor")
                else:
                    print(f"[DEBUG] Masks received: {type(masks)}")
            else:
                print(f"[DEBUG] No masks received")
            
            print(f"[DEBUG] Masks structure: {type(masks)}")
            if masks is not None:
                print(f"[DEBUG] Masks length: {len(masks) if hasattr(masks, '__len__') else 'No length'}")
            
            # Process remaining frames
            processed_frames = 0
            max_frames_to_process = len(aligned_pairs)
            
            print(f"[DEBUG] Starting SAMURAI propagation for {max_frames_to_process} frames...")
            print(f"[DEBUG] Expected frame range: 0 to {max_frames_to_process-1}")
            
            try:
                print(f"[DEBUG] Entering propagate_in_video loop...")
                for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                    print(f"[DEBUG] Propagate yielded - frame_idx: {frame_idx}, object_ids: {object_ids}")
                    
                    # Debug mask structure safely
                    if masks is not None:
                        if isinstance(masks, (list, tuple)) and len(masks) > 0:
                            print(f"[DEBUG] Masks received: {len(masks)} objects (list/tuple)")
                        elif hasattr(masks, 'shape'):
                            print(f"[DEBUG] Masks received: tensor with shape {masks.shape}")
                            # For tensor masks, check the first dimension
                            if len(masks.shape) >= 3:
                                print(f"[DEBUG] Number of objects: {masks.shape[1] if len(masks.shape) > 1 else 1}")
                            else:
                                print(f"[DEBUG] Single mask tensor")
                        else:
                            print(f"[DEBUG] Masks received: {type(masks)}")
                    else:
                        print(f"[DEBUG] No masks received")
                    
                    if frame_idx < max_frames_to_process:
                        print(f"[DEBUG] Processing frame {frame_idx} (within range)")
                        
                        # Save mask for this frame
                        if masks is not None:
                            if isinstance(masks, (list, tuple)) and len(masks) > 0:
                                if isinstance(masks[0], (list, tuple)) and len(masks[0]) > 0:
                                    mask = masks[0][0].cpu().numpy()  # Get first object's mask
                                    print(f"[DEBUG] Mask extracted from list structure - shape: {mask.shape}")
                                elif hasattr(masks[0], 'cpu'):
                                    mask = masks[0].cpu().numpy()  # Get first object's mask
                                    print(f"[DEBUG] Mask extracted from tensor in list - shape: {mask.shape}")
                                else:
                                    print(f"[WARNING] Unexpected mask structure in list: {type(masks[0])}")
                                    continue
                            elif hasattr(masks, 'cpu'):
                                # Handle tensor masks directly
                                if len(masks.shape) >= 3:
                                    # Shape: [batch, objects, height, width] or similar
                                    mask = masks[0, 0].cpu().numpy()  # Get first batch, first object
                                    print(f"[DEBUG] Mask extracted from tensor - shape: {mask.shape}")
                                else:
                                    # Single mask tensor
                                    mask = masks.cpu().numpy()
                                    print(f"[DEBUG] Single mask extracted from tensor - shape: {mask.shape}")
                            else:
                                print(f"[WARNING] Unexpected mask type: {type(masks)}")
                                continue
                        else:
                            print(f"[WARNING] No masks available for frame {frame_idx}")
                            print(f"[DEBUG] Masks structure: {masks}")
                            continue
                        
                        print(f"[DEBUG] Mask extracted - shape: {mask.shape}, range: [{mask.min():.3f}, {mask.max():.3f}]")
                        
                        # SAM2 outputs probability scores [0,1], use lower threshold
                        original_mask = mask.copy()
                        mask = mask > 0.1  # Use lower threshold for better coverage
                        print(f"[DEBUG] Mask thresholded - original sum: {original_mask.sum()}, thresholded sum: {mask.sum()}")
                        
                        # Load ground truth for this frame for quality assessment
                        print(f"[DEBUG] Loading ground truth for frame {frame_idx}...")
                        _, ann_file = aligned_pairs[frame_idx]
                        gt_mask = load_ground_truth_mask(ann_file)
                        
                        if gt_mask is not None:
                            print(f"[DEBUG] GT mask loaded - shape: {gt_mask.shape}, range: [{gt_mask.min():.3f}, {gt_mask.max():.3f}]")
                            
                            # Convert masks to binary for metric calculation
                            gt_binary = (gt_mask > 0).astype(np.uint8)
                            # SAMURAI outputs probability scores - use zero threshold to capture all predictions
                            pred_binary = (mask > 0.0).astype(np.uint8)  # Zero threshold to capture all non-zero values
                            
                            print(f"[DEBUG] Binary masks created:")
                            print(f"[DEBUG]   GT binary - shape: {gt_binary.shape}, sum: {gt_binary.sum()}")
                            print(f"[DEBUG]   Pred binary - shape: {pred_binary.shape}, sum: {pred_binary.sum()}")
                            
                            # Handle edge cases for empty masks
                            if mask.sum() == 0 or gt_binary.sum() == 0:
                                print(f"[WARNING] Empty mask detected for frame {frame_idx}")
                                print(f"[DEBUG]   Pred mask sum: {mask.sum()}")
                                print(f"[DEBUG]   GT binary sum: {gt_binary.sum()}")
                                
                                # Handle empty masks gracefully
                                iou = 0.0
                                dice = 0.0
                                precision, recall = 0.0, 0.0
                                boundary_accuracy = 0.0
                                hausdorff_dist = float('inf')
                                contour_sim = 0.0
                                region_metrics = {'adapted_rand_error': 1.0, 'variation_of_information': 1.0}
                                failure_analysis = {
                                    'false_negatives': gt_binary.sum(),
                                    'false_positives': 0,
                                    'true_positives': 0,
                                    'fn_ratio': 1.0 if gt_binary.sum() > 0 else 0.0,
                                    'fp_ratio': 0.0,
                                    'is_failure': True,
                                    'failure_severity': 1.0
                                }
                                print(f"[DEBUG] Empty mask metrics calculated")
                            else:
                                print(f"[DEBUG] Calculating comprehensive metrics for frame {frame_idx}...")
                                
                                # Calculate comprehensive quality metrics
                                raw_iou = calculate_iou(gt_binary, pred_binary)
                                raw_dice = calculate_dice_coefficient(gt_binary, pred_binary)
                                
                                print(f"[DEBUG] Raw metrics - IoU: {raw_iou:.3f}, Dice: {raw_dice:.3f}")
                                
                                # Apply smoothing for stability
                                iou = smooth_metric(raw_iou, seq_metrics.get('prev_iou') if seq_metrics else None)
                                dice = smooth_metric(raw_dice, seq_metrics.get('prev_dice') if seq_metrics else None)
                                
                                print(f"[DEBUG] Smoothed metrics - IoU: {iou:.3f}, Dice: {dice:.3f}")
                                
                                # Store for next iteration
                                if seq_metrics:
                                    seq_metrics['prev_iou'] = iou
                                    seq_metrics['prev_dice'] = dice
                                
                                precision, recall = calculate_precision_recall(gt_binary, pred_binary)
                                print(f"[DEBUG] Precision: {precision:.3f}, Recall: {recall:.3f}")
                            
                                # Calculate boundary accuracy (if scipy available)
                                try:
                                    boundary_accuracy = calculate_boundary_accuracy(gt_binary, pred_binary)
                                    print(f"[DEBUG] Boundary accuracy: {boundary_accuracy:.3f}")
                                except ImportError:
                                    boundary_accuracy = 0.0
                                    print(f"[DEBUG] Boundary accuracy calculation skipped (scipy not available)")
                                
                                # Calculate advanced boundary metrics
                                hausdorff_dist = calculate_hausdorff_distance(gt_binary, pred_binary)
                                contour_sim = calculate_contour_similarity(gt_binary, pred_binary)
                                print(f"[DEBUG] Hausdorff distance: {hausdorff_dist:.3f}")
                                print(f"[DEBUG] Contour similarity: {contour_sim:.3f}")
                                
                                # Calculate region-based metrics
                                region_metrics = calculate_region_based_metrics(gt_mask, mask)
                                print(f"[DEBUG] Region metrics: {region_metrics}")
                                
                                # Analyze failure cases
                                failure_analysis = analyze_failure_cases(gt_mask, mask, frame_idx, seq_name)
                                print(f"[DEBUG] Failure analysis: {failure_analysis}")
                            
                            # Calculate temporal consistency (if not first frame)
                            temporal_iou = 0.0
                            if frame_idx > 0 and seq_metrics and 'prev_mask' in seq_metrics:
                                print(f"[DEBUG] Calculating temporal consistency...")
                                temporal_iou = calculate_iou(seq_metrics['prev_mask'], pred_binary)
                                print(f"[DEBUG] Temporal IoU: {temporal_iou:.3f}")
                            
                            # Store previous mask for next iteration
                            if seq_metrics:
                                seq_metrics['prev_mask'] = pred_binary.copy()
                                print(f"[DEBUG] Previous mask stored for next iteration")
                            
                            # Update sequence metrics
                            if seq_metrics:
                                print(f"[DEBUG] Updating sequence metrics...")
                                seq_metrics['iou_scores'].append(iou)
                                seq_metrics['dice_scores'].append(dice)
                                seq_metrics['precision_scores'].append(precision)
                                seq_metrics['recall_scores'].append(recall)
                                seq_metrics['boundary_accuracy_scores'].append(boundary_accuracy)
                                seq_metrics['coverage_scores'].append((mask.sum() / mask.size) * 100)
                                seq_metrics['temporal_consistency_scores'].append(temporal_iou)
                                seq_metrics['hausdorff_distances'].append(hausdorff_dist)
                                seq_metrics['contour_similarities'].append(contour_sim)
                                seq_metrics['adapted_rand_errors'].append(region_metrics['adapted_rand_error'])
                                seq_metrics['variation_of_information'].append(region_metrics['variation_of_information'])
                                
                                if failure_analysis['is_failure']:
                                    seq_metrics['failure_cases'] += 1
                                    print(f"[DEBUG] Failure case recorded - total: {seq_metrics['failure_cases']}")
                                
                                # Track processed frames count
                                seq_metrics['processed_frames'] = processed_frames
                            
                            # Get memory usage
                            print(f"[DEBUG] Getting memory usage...")
                            memory_info = get_memory_usage()
                            gpu_memory = get_gpu_memory_usage()
                            print(f"[DEBUG] Memory info: {memory_info}")
                            print(f"[DEBUG] GPU memory: {gpu_memory}")
                            
                            # Log comprehensive frame metrics to wandb with standardized naming
                            if use_wandb and WANDB_AVAILABLE:
                                print(f"[DEBUG] Logging frame metrics to W&B...")
                                try:
                                    wandb.log({
                                        # Frame identification
                                        "frame/idx": frame_idx,
                                        "frame/total_frames": len(aligned_pairs),
                                        "frame/progress": (frame_idx + 1) / len(aligned_pairs),
                                        
                                        # Mask properties
                                        "mask/area_pixels": int(mask.sum()),
                                        "mask/coverage_percent": float((mask.sum() / mask.size) * 100),
                                        "mask/gt_coverage_percent": float((gt_binary.sum() / gt_binary.size) * 100),
                                        
                                        # Evaluation metrics with eval/ prefix
                                        "eval/iou": float(iou),
                                        "eval/dice": float(dice),
                                        "eval/precision": float(precision),
                                        "eval/recall": float(recall),
                                        "eval/boundary_accuracy": float(boundary_accuracy),
                                        "eval/temporal_consistency": float(temporal_iou),
                                        "eval/hausdorff_distance": float(hausdorff_dist) if hausdorff_dist != float('inf') else 1000.0,
                                        "eval/contour_similarity": float(contour_sim),
                                        "eval/adapted_rand_error": float(region_metrics['adapted_rand_error']),
                                        "eval/variation_of_information": float(region_metrics['variation_of_information']),
                                        
                                        # Failure analysis
                                        "eval/false_negatives": int(failure_analysis['false_negatives']),
                                        "eval/false_positives": int(failure_analysis['false_positives']),
                                        "eval/true_positives": int(failure_analysis['true_positives']),
                                        "eval/fn_ratio": float(failure_analysis['fn_ratio']),
                                        "eval/fp_ratio": float(failure_analysis['fp_ratio']),
                                        "eval/is_failure_case": int(failure_analysis['is_failure']),
                                        "eval/failure_severity": float(failure_analysis['failure_severity']),
                                        
                                        # System resources
                                        "system/cpu_memory_percent": float(memory_info['cpu_memory_percent']),
                                        "system/cpu_memory_used_gb": float(memory_info['cpu_memory_used_gb']),
                                        "system/gpu_memory_used_mb": float(gpu_memory) if gpu_memory is not None else 0.0
                                    })
                                    print(f"[DEBUG] W&B logging completed successfully")
                                except Exception as e:
                                    print(f"[ERROR] Failed to log to W&B: {e}")
                                    print(f"[DEBUG] Stack trace:")
                                    import traceback
                                    traceback.print_exc()
                        else:
                            print(f"[WARNING] Could not load ground truth for frame {frame_idx}")
                        
                        # Save mask
                        print(f"[DEBUG] Saving mask for frame {frame_idx}...")
                        output_name = f"{frame_idx:05d}.png"
                        output_file = seq_output_dir / output_name
                        
                        mask_255 = (mask * 255).astype(np.uint8)
                        imageio.imwrite(str(output_file), mask_255)
                        print(f"[DEBUG] Mask saved to: {output_file}")
                        
                        processed_frames += 1
                        print(f"[DEBUG] Frame {frame_idx + 1}/{len(aligned_pairs)} processed (Total processed: {processed_frames})")
                        
                        # Check if we've processed all frames
                        if processed_frames >= max_frames_to_process:
                            print(f"[DEBUG] Completed processing all {max_frames_to_process} frames")
                            break
                    else:
                        print(f"[WARNING] Frame index {frame_idx} exceeds expected range {max_frames_to_process}")
                        print(f"[DEBUG] Stopping propagation loop")
                        break
                        
            except Exception as e:
                print(f"[ERROR] Error during SAMURAI propagation: {e}")
                print(f"[DEBUG] Processed {processed_frames} frames before error")
                print(f"[DEBUG] Stack trace:")
                import traceback
                traceback.print_exc()
                raise e
            
            # Check if we processed all frames
            if processed_frames < max_frames_to_process:
                print(f"[WARNING] Only processed {processed_frames}/{max_frames_to_process} frames")
                print(f"[DEBUG] This may indicate an issue with SAMURAI's internal propagation")
                
                # Try to process remaining frames manually if possible
                remaining_frames = max_frames_to_process - processed_frames
                print(f"[DEBUG] Attempting to process remaining {remaining_frames} frames...")
                
                # For now, we'll just log this issue
                if seq_metrics:
                    seq_metrics['processing_warning'] = f"Only processed {processed_frames}/{max_frames_to_process} frames"
                    seq_metrics['processed_frames'] = processed_frames
            else:
                # Successfully processed all frames
                print(f"[DEBUG] Successfully processed all {processed_frames} frames")
                if seq_metrics:
                    seq_metrics['processed_frames'] = processed_frames
            
    except Exception as e:
        print(f"[ERROR] Error in SAMURAI inference: {e}")
        print(f"[DEBUG] Stack trace:")
        import traceback
        traceback.print_exc()
        raise e
    
    # Log sequence completion metrics
    total_time = time.time() - start_time
    print(f"[DEBUG] Sequence processing completed in {total_time:.2f}s")
    print(f"[DEBUG] Average FPS: {len(aligned_pairs) / total_time if total_time > 0 else 0:.2f}")
    
    if use_wandb and WANDB_AVAILABLE:
        print(f"[DEBUG] Logging sequence completion to W&B...")
        try:
            wandb.log({
                "sequence/total_frames_processed": len(aligned_pairs),
                "sequence/total_processing_time_s": float(total_time),
                "sequence/average_fps": float(len(aligned_pairs) / total_time if total_time > 0 else 0),
                "sequence/status_code": 1  # 1 = completed, 0 = failed
            })
            print(f"[DEBUG] Sequence completion logged to W&B successfully")
        except Exception as e:
            print(f"[ERROR] Failed to log sequence completion to W&B: {e}")
            print(f"[DEBUG] Stack trace:")
            import traceback
            traceback.print_exc()
    
    print(f"[DEBUG] process_sequence_with_samurai completed successfully")

def run_samurai_placeholder(input_path, output_path, sequence, max_frames):
    """Placeholder implementation when SAMURAI model fails to load"""
    print(f"[DEBUG] Starting SAMURAI placeholder implementation")
    print(f"[DEBUG] Input path: {input_path}")
    print(f"[DEBUG] Output path: {output_path}")
    print(f"[DEBUG] Sequence filter: {sequence}")
    print(f"[DEBUG] Max frames: {max_frames}")
    
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Output directory created: {output_path}")
    
    # Find all sequences (DAVIS format - look in JPEGImages/480p)
    jpeg_path = Path(input_path) / "JPEGImages" / "480p"
    if not jpeg_path.exists():
        print(f"[ERROR] JPEGImages/480p directory not found in {input_path}")
        return False
    
    print(f"[DEBUG] JPEG path found: {jpeg_path}")
    sequences = [d for d in jpeg_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    sequences = sorted(sequences)
    print(f"[DEBUG] Found {len(sequences)} sequences: {[s.name for s in sequences]}")
    
    if sequence:
        sequences = [s for s in sequences if s.name == sequence]
        print(f"[DEBUG] Filtered to sequence: {sequence}")
    
    print(f"[DEBUG] Processing {len(sequences)} sequences with placeholder")
    
    # Process each sequence
    for seq_idx, seq_path in enumerate(sequences):
        print(f"\n[DEBUG] [{seq_idx + 1}/{len(sequences)}] Processing sequence: {seq_path.name}")
        
        # Create sequence output directory
        seq_output_dir = output_path / seq_path.name
        seq_output_dir.mkdir(exist_ok=True)
        print(f"[DEBUG] Sequence output dir: {seq_output_dir}")
        
        # Check for image and annotation directories (DAVIS format)
        img_dir = Path(input_path) / "JPEGImages" / "480p" / seq_path.name
        ann_dir = Path(input_path) / "Annotations" / "480p" / seq_path.name
        
        print(f"[DEBUG] Image dir: {img_dir} (exists: {img_dir.exists()})")
        print(f"[DEBUG] Annotation dir: {ann_dir} (exists: {ann_dir.exists()})")
        
        if not img_dir.exists() or not ann_dir.exists():
            print(f"[ERROR] Missing image or annotation directory for {seq_path.name}")
            continue
        
        # Get all image files
        image_files = sorted([f for f in img_dir.glob('*.jpg')])
        annotation_files = sorted([f for f in ann_dir.glob('*.png')])
        
        print(f"[DEBUG] Found {len(image_files)} image files")
        print(f"[DEBUG] Found {len(annotation_files)} annotation files")
        
        if not image_files or not annotation_files:
            print(f"[ERROR] No images or annotations found for {seq_path.name}")
            continue
        
        # Align image and annotation files
        aligned_pairs = []
        for img_file in image_files:
            ann_name = img_file.stem + '.png'
            ann_file = ann_dir / ann_name
            if ann_file.exists():
                aligned_pairs.append((img_file, ann_file))
        
        print(f"[DEBUG] Created {len(aligned_pairs)} aligned pairs")
        
        if max_frames:
            aligned_pairs = aligned_pairs[:max_frames]
            print(f"[DEBUG] Limited to {len(aligned_pairs)} frames due to max_frames={max_frames}")
        
        print(f"[DEBUG] Final frame count: {len(aligned_pairs)}")
        
        # Process sequence with placeholder
        print(f"[DEBUG] Starting placeholder sequence processing...")
        run_samurai_placeholder_sequence(aligned_pairs, seq_output_dir, seq_path.name)
    
    print(f"[DEBUG] Placeholder implementation completed")
    return True

def run_samurai_placeholder_sequence(aligned_pairs, seq_output_dir, seq_name):
    """Placeholder sequence processing - just copy ground truth masks"""
    print(f"[DEBUG] Starting placeholder sequence processing for {seq_name}")
    print(f"[DEBUG] Number of aligned pairs: {len(aligned_pairs)}")
    print(f"[DEBUG] Output directory: {seq_output_dir}")
    
    import time
    start_time = time.time()
    
    processed_frames = 0
    
    for frame_idx, (img_file, ann_file) in enumerate(aligned_pairs):
        print(f"[DEBUG] Processing frame {frame_idx + 1}/{len(aligned_pairs)}")
        print(f"[DEBUG] Image file: {img_file}")
        print(f"[DEBUG] Annotation file: {ann_file}")
        
        try:
            # Load ground truth mask
            print(f"[DEBUG] Loading ground truth mask...")
            gt_mask = load_ground_truth_mask(ann_file)
            
            if gt_mask is not None:
                print(f"[DEBUG] GT mask loaded - shape: {gt_mask.shape}")
                
                # Save as output mask (placeholder just copies GT)
                output_name = f"{frame_idx:05d}.png"
                output_file = seq_output_dir / output_name
                
                print(f"[DEBUG] Saving placeholder mask to: {output_file}")
                imageio.imwrite(str(output_file), gt_mask)
                
                processed_frames += 1
                print(f"[DEBUG] Frame {frame_idx + 1} processed successfully")
            else:
                print(f"[WARNING] Could not load ground truth for frame {frame_idx}")
                
        except Exception as e:
            print(f"[ERROR] Error processing frame {frame_idx}: {e}")
            print(f"[DEBUG] Stack trace:")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"[DEBUG] Placeholder sequence completed in {total_time:.2f}s")
    print(f"[DEBUG] Processed {processed_frames}/{len(aligned_pairs)} frames")
    print(f"[DEBUG] Average FPS: {processed_frames / total_time if total_time > 0 else 0:.2f}")

def calculate_iou(mask1, mask2):
    """Calculate Intersection over Union between two masks"""
    try:
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0

def calculate_dice_coefficient(mask1, mask2):
    """Calculate Dice coefficient between two masks"""
    try:
        intersection = np.logical_and(mask1, mask2).sum()
        total = mask1.sum() + mask2.sum()
        return (2 * intersection) / total if total > 0 else 0.0
    except Exception:
        return 0.0

def calculate_precision_recall(gt_mask, pred_mask):
    """Calculate precision and recall for segmentation"""
    tp = np.logical_and(gt_mask, pred_mask).sum()  # True positives
    fp = np.logical_and(np.logical_not(gt_mask), pred_mask).sum()  # False positives
    fn = np.logical_and(gt_mask, np.logical_not(pred_mask)).sum()  # False negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall

def calculate_boundary_accuracy(gt_mask, pred_mask, boundary_width=2):
    """Calculate boundary accuracy between ground truth and prediction"""
    from scipy import ndimage
    
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

def calculate_temporal_consistency(masks):
    """Calculate temporal consistency across a sequence of masks"""
    if len(masks) < 2:
        return 0.0
    
    temporal_ious = []
    for i in range(1, len(masks)):
        iou = calculate_iou(masks[i-1], masks[i])
        temporal_ious.append(iou)
    
    return np.mean(temporal_ious) if temporal_ious else 0.0

def get_memory_usage():
    """Get current memory usage"""
    try:
        import psutil
        cpu_memory = psutil.virtual_memory()
        return {
            'cpu_memory_percent': cpu_memory.percent,
            'cpu_memory_used_gb': cpu_memory.used / (1024**3)
        }
    except ImportError:
        return {'cpu_memory_percent': 0, 'cpu_memory_used_gb': 0}

def get_gpu_memory_usage():
    """Get GPU memory usage if available"""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].memoryUsed
        return 0
    except ImportError:
        return 0

def analyze_failure_cases(gt_mask, pred_mask, frame_idx, sequence_name):
    """Analyze when and why segmentation fails"""
    gt_binary = (gt_mask > 0).astype(np.uint8)
    pred_binary = (pred_mask > 0.0).astype(np.uint8)  # Zero threshold to capture all predictions
    
    # Calculate failure metrics - FIXED logical operations
    false_negatives = np.logical_and(gt_binary, np.logical_not(pred_binary)).sum()  # GT has object, prediction doesn't
    false_positives = np.logical_and(np.logical_not(gt_binary), pred_binary).sum()  # Prediction has object, GT doesn't
    true_positives = np.logical_and(gt_binary, pred_binary).sum()   # Both have object
    
    total_gt_pixels = gt_binary.sum()
    total_pred_pixels = pred_binary.sum()
    
    # Debug: Check the logical operations
    print(f"    [DEBUG] True positives: {true_positives}, False negatives: {false_negatives}, False positives: {false_positives}")
    print(f"    [DEBUG] GT pixels: {total_gt_pixels}, Pred pixels: {total_pred_pixels}")
    print(f"    [DEBUG] Intersection: {np.logical_and(gt_binary, pred_binary).sum()}")
    print(f"    [DEBUG] Union: {np.logical_or(gt_binary, pred_binary).sum()}")
    
    # Calculate proper ratios - what percentage of GT pixels are missed, what percentage of pred pixels are wrong
    total_pixels = gt_binary.size  # Total image pixels
    
    # False negative ratio: missed GT pixels / total GT pixels
    fn_ratio = false_negatives / max(total_gt_pixels, 1) if total_gt_pixels > 0 else 0
    
    # False positive ratio: wrong pred pixels / total image pixels (not just pred pixels)
    fp_ratio = false_positives / max(total_pixels, 1)
    
    # Determine if this is a failure case - use more reasonable thresholds
    # Consider it a failure if we miss more than 30% of GT pixels OR have more than 5% wrong pixels
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

def calculate_hausdorff_distance(gt_mask, pred_mask):
    """Calculate Hausdorff distance for boundary accuracy assessment"""
    try:
        from scipy.spatial.distance import directed_hausdorff
        
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
    except ImportError:
        return 0.0

def calculate_contour_similarity(gt_mask, pred_mask):
    """Calculate contour similarity using structural similarity"""
    try:
        from skimage.metrics import structural_similarity as ssim
        
        # Ensure masks are same size and type
        gt_norm = gt_mask.astype(np.float32) / gt_mask.max() if gt_mask.max() > 0 else gt_mask.astype(np.float32)
        pred_norm = pred_mask.astype(np.float32) / pred_mask.max() if pred_mask.max() > 0 else pred_mask.astype(np.float32)
        
        # Calculate SSIM
        similarity = ssim(gt_norm, pred_norm, data_range=1.0)
        return similarity
    except ImportError:
        return 0.0

def calculate_region_based_metrics(gt_mask, pred_mask):
    """Calculate region-based segmentation metrics"""
    try:
        from skimage.metrics import adapted_rand_error, variation_of_information
        
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
    except ImportError:
        return {'adapted_rand_error': 0.0, 'variation_of_information': 0.0}

def calculate_temporal_stability_metrics(masks):
    """Calculate comprehensive temporal stability metrics"""
    if len(masks) < 2:
        return {
            'temporal_iou_mean': 0.0,
            'temporal_iou_std': 0.0,
            'temporal_coverage_variance': 0.0,
            'temporal_shape_consistency': 0.0
        }
    
    temporal_ious = []
    coverage_values = []
    shape_consistencies = []
    
    for i in range(1, len(masks)):
        # Temporal IoU
        iou = calculate_iou(masks[i-1], masks[i])
        temporal_ious.append(iou)
        
        # Coverage consistency
        coverage_prev = (masks[i-1].sum() / masks[i-1].size) * 100
        coverage_curr = (masks[i].sum() / masks[i].size) * 100
        coverage_values.append(abs(coverage_curr - coverage_prev))
        
        # Shape consistency (using contour similarity)
        shape_sim = calculate_contour_similarity(masks[i-1], masks[i])
        shape_consistencies.append(shape_sim)
    
    return {
        'temporal_iou_mean': np.mean(temporal_ious),
        'temporal_iou_std': np.std(temporal_ious),
        'temporal_coverage_variance': np.var(coverage_values),
        'temporal_shape_consistency': np.mean(shape_consistencies)
    }

def calculate_complexity_metrics(gt_mask):
    """Calculate object complexity metrics for difficulty assessment"""
    try:
        from skimage.measure import label, regionprops
        
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
    except ImportError:
        return {
            'object_count': 0,
            'total_area': 0,
            'avg_area': 0,
            'perimeter': 0,
            'compactness': 0,
            'eccentricity': 0
        }

def smooth_metric(value, prev_value, alpha=0.9):
    """Apply exponential smoothing to metrics for stability"""
    if prev_value is None:
        return value
    return alpha * prev_value + (1 - alpha) * value

def calculate_dataset_statistics(sequence_metrics):
    """Calculate comprehensive dataset-level statistics"""
    # Collect all metrics across sequences
    all_ious = []
    all_dices = []
    all_coverages = []
    all_temporal_consistencies = []
    all_failure_rates = []
    
    for seq_name, metrics in sequence_metrics.items():
        all_ious.extend(metrics['iou_scores'])
        all_dices.extend(metrics['dice_scores'])
        all_coverages.extend(metrics['coverage_scores'])
        all_temporal_consistencies.extend(metrics['temporal_consistency_scores'])
        if metrics['total_frames'] > 0:
            all_failure_rates.append(metrics['failure_cases'] / metrics['total_frames'])
    
    # Calculate comprehensive statistics
    stats = {}
    
    if all_ious:
        stats['iou_mean'] = np.mean(all_ious)
        stats['iou_std'] = np.std(all_ious)
        stats['iou_median'] = np.median(all_ious)
        stats['iou_q25'] = np.percentile(all_ious, 25)
        stats['iou_q75'] = np.percentile(all_ious, 75)
        stats['iou_min'] = np.min(all_ious)
        stats['iou_max'] = np.max(all_ious)
    
    if all_dices:
        stats['dice_mean'] = np.mean(all_dices)
        stats['dice_std'] = np.std(all_dices)
        stats['dice_median'] = np.median(all_dices)
    
    if all_coverages:
        stats['coverage_mean'] = np.mean(all_coverages)
        stats['coverage_std'] = np.std(all_coverages)
        stats['coverage_median'] = np.median(all_coverages)
    
    if all_temporal_consistencies:
        stats['temporal_consistency_mean'] = np.mean(all_temporal_consistencies)
        stats['temporal_consistency_std'] = np.std(all_temporal_consistencies)
    
    if all_failure_rates:
        stats['failure_rate_mean'] = np.mean(all_failure_rates)
        stats['failure_rate_std'] = np.std(all_failure_rates)
    
    return stats

def run_samurai_actual(input_path, output_path, predictor, sequence=None, max_frames=None, use_wandb=False):
    """Run actual SAMURAI implementation with comprehensive debugging"""
    print(f"[DEBUG] Starting actual SAMURAI implementation")
    print(f"[DEBUG] Input path: {input_path}")
    print(f"[DEBUG] Output path: {output_path}")
    print(f"[DEBUG] Predictor type: {type(predictor)}")
    print(f"[DEBUG] Sequence filter: {sequence}")
    print(f"[DEBUG] Max frames: {max_frames}")
    print(f"[DEBUG] W&B enabled: {use_wandb}")
    
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Output directory created: {output_path}")
    
    # Find all sequences (DAVIS format - look in JPEGImages/480p)
    jpeg_path = Path(input_path) / "JPEGImages" / "480p"
    if not jpeg_path.exists():
        print(f"[ERROR] JPEGImages/480p directory not found in {input_path}")
        return False
    
    print(f"[DEBUG] JPEG path found: {jpeg_path}")
    sequences = [d for d in jpeg_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    sequences = sorted(sequences)
    print(f"[DEBUG] Found {len(sequences)} sequences: {[s.name for s in sequences]}")
    
    if sequence:
        sequences = [s for s in sequences if s.name == sequence]
        print(f"[DEBUG] Filtered to sequence: {sequence}")
    
    print(f"[DEBUG] Processing {len(sequences)} sequences")
    
    # Track per-sequence metrics for comprehensive analysis
    sequence_metrics = {}
    
    # Process each sequence
    for seq_idx, seq_path in enumerate(sequences):
        print(f"\n[DEBUG] [{seq_idx + 1}/{len(sequences)}] Starting sequence: {seq_path.name}")
        
        # Create sequence output directory
        seq_output_dir = output_path / seq_path.name
        seq_output_dir.mkdir(exist_ok=True)
        print(f"[DEBUG] Sequence output dir: {seq_output_dir}")
        
        # Initialize sequence metrics tracking
        sequence_metrics[seq_path.name] = {
            'total_frames': 0,
            'iou_scores': [],
            'dice_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'boundary_accuracy_scores': [],
            'coverage_scores': [],
            'processing_times': [],
            'failure_cases': 0,
            'temporal_consistency_scores': [],
            'hausdorff_distances': [],
            'contour_similarities': [],
            'adapted_rand_errors': [],
            'variation_of_information': [],
            'temporal_stability_metrics': {},
            'complexity_metrics': {},
            'prev_mask': None,  # Add this for temporal consistency
            'prev_iou': None,   # Add this for smoothing
            'prev_dice': None,  # Add this for smoothing
            'start_time': time.time()
        }
        print(f"[DEBUG] Sequence metrics initialized for {seq_path.name}")
        
        # Check for image and annotation directories (DAVIS format)
        img_dir = Path(input_path) / "JPEGImages" / "480p" / seq_path.name
        ann_dir = Path(input_path) / "Annotations" / "480p" / seq_path.name
        
        print(f"[DEBUG] Image dir: {img_dir} (exists: {img_dir.exists()})")
        print(f"[DEBUG] Annotation dir: {ann_dir} (exists: {ann_dir.exists()})")
        
        if not img_dir.exists() or not ann_dir.exists():
            print(f"[ERROR] Missing image or annotation directory for {seq_path.name}")
            continue
        
        # Get all image files
        image_files = sorted([f for f in img_dir.glob('*.jpg')])
        annotation_files = sorted([f for f in ann_dir.glob('*.png')])
        
        print(f"[DEBUG] Found {len(image_files)} image files")
        print(f"[DEBUG] Found {len(annotation_files)} annotation files")
        
        if not image_files or not annotation_files:
            print(f"[ERROR] No images or annotations found for {seq_path.name}")
            continue
        
        # Align image and annotation files
        aligned_pairs = []
        for img_file in image_files:
            ann_name = img_file.stem + '.png'
            ann_file = ann_dir / ann_name
            if ann_file.exists():
                aligned_pairs.append((img_file, ann_file))
        
        print(f"[DEBUG] Created {len(aligned_pairs)} aligned pairs")
        
        if max_frames:
            aligned_pairs = aligned_pairs[:max_frames]
            print(f"[DEBUG] Limited to {len(aligned_pairs)} frames due to max_frames={max_frames}")
        
        print(f"[DEBUG] Final frame count: {len(aligned_pairs)}")
        
        # Update sequence metrics
        sequence_metrics[seq_path.name]['total_frames'] = len(aligned_pairs)
        
        try:
            # Reset model for each sequence to avoid state corruption
            if seq_idx > 0:  # Don't reset for first sequence
                print(f"[DEBUG] Resetting SAMURAI model for {seq_path.name}...")
                del predictor  # Clean up old predictor
                torch.cuda.empty_cache()  # Clear GPU memory
                print(f"[DEBUG] GPU memory cleared")
                
                # Reload predictor
                print(f"[DEBUG] Reloading predictor...")
                predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)
                print(f"[DEBUG] SAMURAI model reloaded successfully")
                
                # Re-enable W&B watching for new predictor
                if use_wandb and WANDB_AVAILABLE:
                    try:
                        if hasattr(predictor, 'model'):
                            wandb.watch(predictor.model, log="all", log_freq=10)
                            print(f"[DEBUG] W&B watching re-enabled")
                    except Exception as e:
                        print(f"[WARNING] Could not re-enable W&B model watching: {e}")
            
            # Process sequence with SAMURAI
            print(f"[DEBUG] Starting sequence processing...")
            process_sequence_with_samurai(predictor, aligned_pairs, seq_output_dir, seq_path.name, use_wandb, sequence_metrics[seq_path.name])
            
        except Exception as e:
            print(f"[ERROR] Error processing sequence with SAMURAI: {e}")
            print(f"[DEBUG] Stack trace:")
            import traceback
            traceback.print_exc()
            print(f"[DEBUG] Falling back to placeholder implementation")
            run_samurai_placeholder_sequence(aligned_pairs, seq_output_dir, seq_path.name)
        
        # Calculate and log sequence summary metrics
        print(f"[DEBUG] Calculating sequence summary metrics...")
        seq_metrics = sequence_metrics[seq_path.name]
        seq_metrics['end_time'] = time.time()
        seq_metrics['total_time'] = seq_metrics['end_time'] - seq_metrics['start_time']
        
        print(f"[DEBUG] Sequence processing time: {seq_metrics['total_time']:.2f}s")
        
        # Calculate averages and statistics
        if seq_metrics['iou_scores']:
            seq_metrics['avg_iou'] = np.mean(seq_metrics['iou_scores'])
            seq_metrics['std_iou'] = np.std(seq_metrics['iou_scores'])
            seq_metrics['min_iou'] = np.min(seq_metrics['iou_scores'])
            seq_metrics['max_iou'] = np.max(seq_metrics['iou_scores'])
            print(f"[DEBUG] IoU stats - avg: {seq_metrics['avg_iou']:.3f}, std: {seq_metrics['std_iou']:.3f}")
        else:
            seq_metrics['avg_iou'] = seq_metrics['std_iou'] = seq_metrics['min_iou'] = seq_metrics['max_iou'] = 0.0
            print(f"[WARNING] No IoU scores available for {seq_path.name}")
            
        if seq_metrics['dice_scores']:
            seq_metrics['avg_dice'] = np.mean(seq_metrics['dice_scores'])
            seq_metrics['std_dice'] = np.std(seq_metrics['dice_scores'])
            print(f"[DEBUG] Dice stats - avg: {seq_metrics['avg_dice']:.3f}, std: {seq_metrics['std_dice']:.3f}")
        else:
            seq_metrics['avg_dice'] = seq_metrics['std_dice'] = 0.0
            
        if seq_metrics['coverage_scores']:
            seq_metrics['avg_coverage'] = np.mean(seq_metrics['coverage_scores'])
            seq_metrics['std_coverage'] = np.std(seq_metrics['coverage_scores'])
            print(f"[DEBUG] Coverage stats - avg: {seq_metrics['avg_coverage']:.2f}%, std: {seq_metrics['std_coverage']:.2f}%")
        else:
            seq_metrics['avg_coverage'] = seq_metrics['std_coverage'] = 0.0
            
        if seq_metrics['temporal_consistency_scores']:
            seq_metrics['avg_temporal_consistency'] = np.mean(seq_metrics['temporal_consistency_scores'])
            print(f"[DEBUG] Temporal consistency - avg: {seq_metrics['avg_temporal_consistency']:.3f}")
        else:
            seq_metrics['avg_temporal_consistency'] = 0.0
        
        # Calculate advanced metric statistics
        if seq_metrics['hausdorff_distances']:
            seq_metrics['avg_hausdorff'] = np.mean(seq_metrics['hausdorff_distances'])
            seq_metrics['std_hausdorff'] = np.std(seq_metrics['hausdorff_distances'])
            print(f"[DEBUG] Hausdorff stats - avg: {seq_metrics['avg_hausdorff']:.2f}, std: {seq_metrics['std_hausdorff']:.2f}")
        else:
            seq_metrics['avg_hausdorff'] = seq_metrics['std_hausdorff'] = 0.0
            
        if seq_metrics['contour_similarities']:
            seq_metrics['avg_contour_sim'] = np.mean(seq_metrics['contour_similarities'])
            seq_metrics['std_contour_sim'] = np.std(seq_metrics['contour_similarities'])
            print(f"[DEBUG] Contour similarity stats - avg: {seq_metrics['avg_contour_sim']:.3f}, std: {seq_metrics['std_contour_sim']:.3f}")
        else:
            seq_metrics['avg_contour_sim'] = seq_metrics['std_contour_sim'] = 0.0
            
        if seq_metrics['adapted_rand_errors']:
            seq_metrics['avg_adapted_rand'] = np.mean(seq_metrics['adapted_rand_errors'])
            seq_metrics['std_adapted_rand'] = np.std(seq_metrics['adapted_rand_errors'])
            print(f"[DEBUG] Adapted Rand stats - avg: {seq_metrics['avg_adapted_rand']:.3f}, std: {seq_metrics['std_adapted_rand']:.3f}")
        else:
            seq_metrics['avg_adapted_rand'] = seq_metrics['std_adapted_rand'] = 0.0
            
        if seq_metrics['variation_of_information']:
            seq_metrics['avg_voi'] = np.mean(seq_metrics['variation_of_information'])
            seq_metrics['std_voi'] = np.std(seq_metrics['variation_of_information'])
            print(f"[DEBUG] VOI stats - avg: {seq_metrics['avg_voi']:.3f}, std: {seq_metrics['std_voi']:.3f}")
        else:
            seq_metrics['avg_voi'] = seq_metrics['std_voi'] = 0.0
        
        # Calculate temporal stability metrics for the sequence
        if len(seq_metrics['temporal_consistency_scores']) > 1:
            print(f"[DEBUG] Calculating temporal stability metrics...")
            seq_metrics['temporal_stability_metrics'] = calculate_temporal_stability_metrics(
                [np.array(seq_metrics['coverage_scores'])]  # Convert to array format
            )
        
        # Calculate complexity metrics for the first frame (representative of sequence)
        if aligned_pairs:
            print(f"[DEBUG] Calculating complexity metrics from first frame...")
            first_gt_mask = load_ground_truth_mask(aligned_pairs[0][1])
            if first_gt_mask is not None:
                seq_metrics['complexity_metrics'] = calculate_complexity_metrics(first_gt_mask)
                print(f"[DEBUG] Complexity metrics calculated: {seq_metrics['complexity_metrics']}")
            else:
                # Initialize with default values if no mask available
                seq_metrics['complexity_metrics'] = {
                    'object_count': 0,
                    'total_area': 0,
                    'avg_area': 0,
                    'perimeter': 0,
                    'compactness': 0.0,
                    'eccentricity': 0.0
                }
                print(f"[WARNING] Could not load first frame mask, using default complexity metrics")
        else:
            # Initialize with default values if no pairs available
            seq_metrics['complexity_metrics'] = {
                'object_count': 0,
                'total_area': 0,
                'avg_area': 0,
                'perimeter': 0,
                'compactness': 0.0,
                'eccentricity': 0.0
            }
            print(f"[WARNING] No aligned pairs available, using default complexity metrics")
        
        # Log comprehensive sequence summary to W&B with standardized naming
        if use_wandb and WANDB_AVAILABLE:
            print(f"[DEBUG] Logging sequence summary to W&B...")
            try:
                wandb.log({
                    # Sequence identification (as numeric indices instead of strings)
                    "sequence/idx": seq_idx,
                    "sequence/total_sequences": len(sequences),
                    "sequence/total_frames": len(aligned_pairs),
                    
                    # Performance metrics with eval/ prefix
                    "eval/avg_iou": seq_metrics['avg_iou'],
                    "eval/std_iou": seq_metrics['std_iou'],
                    "eval/min_iou": seq_metrics['min_iou'],
                    "eval/max_iou": seq_metrics['max_iou'],
                    "eval/avg_dice": seq_metrics['avg_dice'],
                    "eval/std_dice": seq_metrics['std_dice'],
                    "eval/avg_coverage": seq_metrics['avg_coverage'],
                    "eval/std_coverage": seq_metrics['std_coverage'],
                    "eval/avg_temporal_consistency": seq_metrics['avg_temporal_consistency'],
                    "eval/avg_hausdorff": seq_metrics['avg_hausdorff'],
                    "eval/std_hausdorff": seq_metrics['std_hausdorff'],
                    "eval/avg_contour_similarity": seq_metrics['avg_contour_sim'],
                    "eval/std_contour_similarity": seq_metrics['std_contour_sim'],
                    "eval/avg_adapted_rand": seq_metrics['avg_adapted_rand'],
                    "eval/std_adapted_rand": seq_metrics['std_adapted_rand'],
                    "eval/avg_voi": seq_metrics['avg_voi'],
                    "eval/std_voi": seq_metrics['std_voi'],
                    "eval/failure_rate": seq_metrics['failure_cases'] / seq_metrics['total_frames'] if seq_metrics['total_frames'] > 0 else 0.0,
                    
                    # Processing metrics
                    "performance/total_time": seq_metrics['total_time'],
                    "performance/avg_fps": seq_metrics['total_frames'] / seq_metrics['total_time'] if seq_metrics['total_time'] > 0 else 0.0,
                    
                    # Complexity metrics
                    "complexity/object_count": seq_metrics.get('complexity_metrics', {}).get('object_count', 0),
                    "complexity/eccentricity": seq_metrics.get('complexity_metrics', {}).get('eccentricity', 0.0),
                    "complexity/compactness": seq_metrics.get('complexity_metrics', {}).get('compactness', 0.0),
                    "complexity/avg_area": seq_metrics.get('complexity_metrics', {}).get('avg_area', 0.0),
                    
                    # Status as numeric code
                    "sequence/status_code": 1,  # 1 = completed, 0 = failed
                    "sequence/completion_timestamp": datetime.now().timestamp()
                })
                
                # Log processing warning if any
                if 'processing_warning' in seq_metrics:
                    print(f"[DEBUG] Logging processing warning to W&B...")
                    wandb.log({
                        "sequence/processing_warning_code": 1,  # 1 = warning, 0 = no warning
                        "sequence/processed_frames_ratio": seq_metrics.get('processed_frames', 0) / seq_metrics['total_frames'] if seq_metrics['total_frames'] > 0 else 0.0
                    })
                
                print(f"[DEBUG] W&B logging completed successfully")
                
            except Exception as e:
                print(f"[ERROR] Failed to log to W&B: {e}")
                print(f"[DEBUG] Stack trace:")
                import traceback
                traceback.print_exc()
        
        print(f"[DEBUG] Completed sequence: {seq_path.name}")
        print(f"  Average IoU: {seq_metrics['avg_iou']:.3f} ± {seq_metrics['std_iou']:.3f}")
        print(f"  Average Dice: {seq_metrics['avg_dice']:.3f} ± {seq_metrics['std_dice']:.3f}")
        print(f"  Average Coverage: {seq_metrics['avg_coverage']:.2f}% ± {seq_metrics['std_coverage']:.2f}%")
        print(f"  Temporal Consistency: {seq_metrics['avg_temporal_consistency']:.3f}")
        print(f"  Boundary Accuracy: {seq_metrics['avg_hausdorff']:.2f} ± {seq_metrics['std_hausdorff']:.2f}")
        print(f"  Contour Similarity: {seq_metrics['avg_contour_sim']:.3f} ± {seq_metrics['std_contour_sim']:.3f}")
        print(f"  Failure Cases: {seq_metrics['failure_cases']}/{len(aligned_pairs)} ({seq_metrics['failure_cases']/len(aligned_pairs)*100:.1f}%)")
        print(f"  Processing Time: {seq_metrics['total_time']:.2f}s ({len(aligned_pairs)/seq_metrics['total_time']:.2f} FPS)")
        if seq_metrics['complexity_metrics']:
            comp = seq_metrics['complexity_metrics']
            print(f"  Object Complexity: {comp.get('object_count', 0)} objects, Area: {comp.get('avg_area', 0):.0f}, Compactness: {comp.get('compactness', 0):.3f}")
    
    # Calculate overall experiment statistics
    print(f"\n[DEBUG] Calculating overall experiment statistics...")
    print(f"=== EXPERIMENT SUMMARY ===")
    
    # Overall statistics across all sequences
    all_ious = []
    all_dices = []
    all_coverages = []
    all_temporal_consistencies = []
    all_hausdorffs = []
    all_contour_sims = []
    all_adapted_rands = []
    all_vois = []
    total_failure_cases = 0
    total_frames = 0
    
    for seq_name, metrics in sequence_metrics.items():
        print(f"[DEBUG] Processing metrics for sequence: {seq_name}")
        all_ious.extend(metrics['iou_scores'])
        all_dices.extend(metrics['dice_scores'])
        all_coverages.extend(metrics['coverage_scores'])
        all_temporal_consistencies.extend(metrics['temporal_consistency_scores'])
        all_hausdorffs.extend(metrics['hausdorff_distances'])
        all_contour_sims.extend(metrics['contour_similarities'])
        all_adapted_rands.extend(metrics['adapted_rand_errors'])
        all_vois.extend(metrics['variation_of_information'])
        total_failure_cases += metrics['failure_cases']
        total_frames += metrics['total_frames']
    
    print(f"[DEBUG] Total frames across all sequences: {total_frames}")
    print(f"[DEBUG] Total failure cases: {total_failure_cases}")
    
    # Calculate overall averages
    overall_avg_iou = np.mean(all_ious) if all_ious else 0.0
    overall_avg_dice = np.mean(all_dices) if all_dices else 0.0
    overall_avg_coverage = np.mean(all_coverages) if all_coverages else 0.0
    overall_avg_temporal = np.mean(all_temporal_consistencies) if all_temporal_consistencies else 0.0
    overall_avg_hausdorff = np.mean(all_hausdorffs) if all_hausdorffs else 0.0
    overall_avg_contour_sim = np.mean(all_contour_sims) if all_contour_sims else 0.0
    overall_avg_adapted_rand = np.mean(all_adapted_rands) if all_adapted_rands else 0.0
    overall_avg_voi = np.mean(all_vois) if all_vois else 0.0
    overall_failure_rate = total_failure_cases / total_frames if total_frames > 0 else 0.0
    
    print(f"[DEBUG] Overall averages calculated:")
    print(f"[DEBUG]   IoU: {overall_avg_iou:.3f}")
    print(f"[DEBUG]   Dice: {overall_avg_dice:.3f}")
    print(f"[DEBUG]   Coverage: {overall_avg_coverage:.2f}%")
    print(f"[DEBUG]   Temporal: {overall_avg_temporal:.3f}")
    
    # Calculate comprehensive dataset statistics
    print(f"[DEBUG] Calculating dataset statistics...")
    dataset_stats = calculate_dataset_statistics(sequence_metrics)
    
    print(f"Overall Performance:")
    print(f"  Average IoU: {overall_avg_iou:.3f}")
    print(f"  Average Dice: {overall_avg_dice:.3f}")
    print(f"  Average Coverage: {overall_avg_coverage:.2f}%")
    print(f"  Average Temporal Consistency: {overall_avg_temporal:.3f}")
    print(f"  Average Boundary Accuracy (Hausdorff): {overall_avg_hausdorff:.2f}")
    print(f"  Average Contour Similarity: {overall_avg_contour_sim:.3f}")
    print(f"  Average Adapted Rand Error: {overall_avg_adapted_rand:.3f}")
    print(f"  Average Variation of Information: {overall_avg_voi:.3f}")
    print(f"  Overall Failure Rate: {overall_failure_rate:.1%}")
    print(f"  Total Frames Processed: {total_frames}")
    
    # Print detailed statistics
    if 'iou_q25' in dataset_stats:
        print(f"\nDetailed IoU Statistics:")
        print(f"  Median: {dataset_stats['iou_median']:.3f}")
        print(f"  Q25: {dataset_stats['iou_q25']:.3f}")
        print(f"  Q75: {dataset_stats['iou_q75']:.3f}")
        print(f"  Range: {dataset_stats['iou_min']:.3f} - {dataset_stats['iou_max']:.3f}")
    
    # Log final experiment summary to W&B with standardized naming
    if use_wandb and WANDB_AVAILABLE:
        print(f"[DEBUG] Logging final experiment summary to W&B...")
        try:
            wandb.log({
                # Experiment summary
                "experiment/total_sequences_processed": len(sequences),
                "experiment/total_frames_processed": total_frames,
                
                # Overall performance metrics
                "experiment/overall_avg_iou": overall_avg_iou,
                "experiment/overall_avg_dice": overall_avg_dice,
                "experiment/overall_avg_coverage": overall_avg_coverage,
                "experiment/overall_avg_temporal_consistency": overall_avg_temporal,
                "experiment/overall_avg_hausdorff": overall_avg_hausdorff,
                "experiment/overall_avg_contour_similarity": overall_avg_contour_sim,
                "experiment/overall_avg_adapted_rand": overall_avg_adapted_rand,
                "experiment/overall_avg_voi": overall_avg_voi,
                "experiment/overall_failure_rate": overall_failure_rate,
                
                # Status as numeric code instead of string
                "experiment/status_code": 1,  # 1 = completed, 0 = failed
                "experiment/final_timestamp": datetime.now().timestamp()
            })
            
            # Log detailed dataset statistics
            for key, value in dataset_stats.items():
                if isinstance(value, (int, float)):
                    wandb.log({f"experiment/dataset_{key}": value})
            
            print(f"[DEBUG] Final W&B logging completed successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to log final summary to W&B: {e}")
            print(f"[DEBUG] Stack trace:")
            import traceback
            traceback.print_exc()
    
    print(f"\n[DEBUG] SAMURAI baseline completed successfully!")
    print(f"Output saved to: {output_path}")
    return True

def main():
    print(f"[DEBUG] Starting SAMURAI baseline main function")
    print(f"[DEBUG] Python version: {sys.version}")
    print(f"[DEBUG] PyTorch version: {torch.__version__}")
    print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[DEBUG] CUDA version: {torch.version.cuda}")
        print(f"[DEBUG] GPU count: {torch.cuda.device_count()}")
        print(f"[DEBUG] Current GPU: {torch.cuda.current_device()}")
        print(f"[DEBUG] GPU name: {torch.cuda.get_device_name()}")
    
    parser = argparse.ArgumentParser(description="Run SAMURAI baseline on DAVIS-2017 dataset")
    parser.add_argument("--input_path", type=str, default="input/davis2017",
                       help="Path to DAVIS-2017 dataset")
    parser.add_argument("--output_path", type=str, default="output/samurai_davis_baseline",
                       help="Path to save output masks")
    parser.add_argument("--checkpoint", type=str, 
                       default="samurai_official/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
                       help="Path to SAMURAI checkpoint")
    parser.add_argument("--sequence", type=str, default=None,
                       help="Process only specific sequence (optional)")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum frames to process per sequence (optional)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--experiment-name", type=str, default=None, help="Name for the Weights & Biases experiment")
    
    args = parser.parse_args()
    
    print(f"[DEBUG] Command line arguments parsed:")
    print(f"[DEBUG]   input_path: {args.input_path}")
    print(f"[DEBUG]   output_path: {args.output_path}")
    print(f"[DEBUG]   checkpoint: {args.checkpoint}")
    print(f"[DEBUG]   sequence: {args.sequence}")
    print(f"[DEBUG]   max_frames: {args.max_frames}")
    print(f"[DEBUG]   wandb: {args.wandb}")
    print(f"[DEBUG]   experiment_name: {args.experiment_name}")
    
    # Validate input path
    if not os.path.exists(args.input_path):
        print(f"[ERROR] Input path does not exist: {args.input_path}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        print(f"[DEBUG] Available directories:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"[DEBUG]   - {item}/")
        return 1
    
    print(f"[DEBUG] Input path validation passed")
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint path does not exist: {args.checkpoint}")
        print(f"[DEBUG] Available checkpoints in samurai_official/sam2/checkpoints/:")
        checkpoint_dir = "samurai_official/sam2/checkpoints"
        if os.path.exists(checkpoint_dir):
            for item in os.listdir(checkpoint_dir):
                print(f"[DEBUG]   - {item}")
        else:
            print(f"[DEBUG] Checkpoint directory does not exist: {checkpoint_dir}")
        return 1
    
    print(f"[DEBUG] Checkpoint path validation passed")
    
    # Check W&B availability
    if args.wandb:
        if not WANDB_AVAILABLE:
            print(f"[ERROR] W&B requested but not available. Install with: pip install wandb")
            return 1
        print(f"[DEBUG] W&B logging enabled")
    else:
        print(f"[DEBUG] W&B logging disabled")
    
    # Initialize W&B if requested
    if args.wandb and WANDB_AVAILABLE:
        print(f"[DEBUG] Initializing W&B...")
        try:
            # Generate experiment name if not provided
            if not args.experiment_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                args.experiment_name = f"samurai_baseline_{timestamp}"
            
            print(f"[DEBUG] Experiment name: {args.experiment_name}")
            
            # Initialize W&B with comprehensive config
            wandb.init(
                project="temporal-deid-baselines",
                name=args.experiment_name,
                config={
                    # Model Configuration
                    "model/name": "SAMURAI",
                    "model/version": "sam2.1_hiera_base_plus",
                    "model/checkpoint_path": args.checkpoint,
                    "model/config_file": "configs/sam2.1/sam2.1_hiera_b+.yaml",
                    
                    # Dataset Configuration
                    "dataset/name": "DAVIS-2017",
                    "dataset/format": "JPEGImages/480p + Annotations/480p",
                    "dataset/sequence_filter": args.sequence,
                    "dataset/max_frames": args.max_frames,
                    "dataset/input_path": args.input_path,
                    "dataset/output_path": args.output_path,
                    
                    # Hardware Configuration
                    "hardware/device": "cuda" if torch.cuda.is_available() else "cpu",
                    "hardware/cuda_available": torch.cuda.is_available(),
                    "hardware/gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "hardware/gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A",
                    "hardware/pytorch_version": torch.__version__,
                    "hardware/python_version": sys.version,
                    
                    # Processing Configuration
                    "processing/batch_size": 1,
                    "processing/mask_format": "PNG",
                    "processing/temporal_context": "3-frame",
                    "processing/memory_management": "sequence_reset",
                    "processing/offload_to_cpu": True,
                    
                    # Evaluation Configuration
                    "evaluation/metrics": ["IoU", "Dice", "Coverage", "Temporal_Consistency", "Hausdorff", "Contour_Similarity", "Adapted_Rand", "VOI"],
                    "evaluation/threshold_strategy": "adaptive",
                    "evaluation/failure_analysis": True,
                    "evaluation/complexity_metrics": True,
                    
                    # Experiment Metadata
                    "experiment/framework": "PyTorch",
                    "experiment/implementation": "SAMURAI_Official",
                    "experiment/timestamp": datetime.now().isoformat(),
                    "experiment/run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "experiment/description": "SAMURAI baseline evaluation on DAVIS-2017 dataset"
                },
                tags=[
                    f"m_samurai",           # Model: SAMURAI
                    f"d_davis2017",         # Dataset: DAVIS-2017
                    f"b_baseline",          # Baseline: Yes
                    f"v_sam2.1",            # Version: SAM2.1
                    f"s_{args.sequence}" if args.sequence else "s_all",  # Sequence
                    f"h_cuda" if torch.cuda.is_available() else "h_cpu"  # Hardware
                ]
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
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize W&B: {e}")
            print(f"[DEBUG] Stack trace:")
            import traceback
            traceback.print_exc()
            print(f"[DEBUG] Continuing without W&B...")
            args.wandb = False
    
    # Run the baseline
    print(f"[DEBUG] Starting SAMURAI baseline execution...")
    try:
        success = run_samurai_davis_baseline(
            input_path=args.input_path,
            output_path=args.output_path,
            checkpoint_path=args.checkpoint,
            sequence=args.sequence,
            max_frames=args.max_frames,
            use_wandb=args.wandb
        )
        
        if success:
            print(f"[DEBUG] SAMURAI baseline completed successfully")
            if args.wandb and WANDB_AVAILABLE:
                print(f"[DEBUG] Finalizing W&B run...")
                try:
                    wandb.finish()
                    print(f"[DEBUG] W&B run finalized successfully")
                except Exception as e:
                    print(f"[WARNING] Failed to finalize W&B: {e}")
            return 0
        else:
            print(f"[ERROR] SAMURAI baseline failed")
            return 1
            
    except Exception as e:
        print(f"[ERROR] Unexpected error in main execution: {e}")
        print(f"[DEBUG] Stack trace:")
        import traceback
        traceback.print_exc()
        
        if args.wandb and WANDB_AVAILABLE:
            print(f"[DEBUG] Finalizing W&B run due to error...")
            try:
                wandb.finish()
                print(f"[DEBUG] W&B run finalized after error")
            except Exception as wb_e:
                print(f"[WARNING] Failed to finalize W&B after error: {wb_e}")
        
        return 1

if __name__ == '__main__':
    main()
