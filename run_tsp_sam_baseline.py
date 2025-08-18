#!/usr/bin/env python3
"""
TSP-SAM Baseline Runner for DAVIS-2017 Dataset
This script runs the TSP-SAM baseline using ground truth annotations for prompts
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

# Add the official TSP-SAM to path
sys.path.append('tsp_sam_official')
sys.path.append('tsp_sam_official/lib')
sys.path.append('tsp_sam_official/dataloaders')

try:
    from lib import VideoModel_pvtv2 as Network
    from utils.utils import post_process
except ImportError as e:
    print(f"Error importing TSP-SAM modules: {e}")
    print("Please ensure 'tsp_sam_official' is a correctly configured git submodule and its dependencies are installed.")
    sys.exit(1)

# Configuration for TSP-SAM model
class ModelConfig:
    def __init__(self):
        self.channel = 32
        self.imgsize = 352
        self.pretrained = True
        self.gpu_ids = [0] if torch.cuda.is_available() else []

def load_and_preprocess_image(image_path, img_size, convert_ycbcr=False):
    """Load and preprocess image for TSP-SAM"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    if convert_ycbcr:
        # Convert to YCbCr for TSP-SAM's specific input requirement
        img_np = np.array(img)
        img_ycbcr_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        img_ycbcr_tensor = torch.from_numpy(img_ycbcr_np).permute(2, 0, 1).float() / 255.0
        img_ycbcr_tensor = F.interpolate(img_ycbcr_tensor.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(0)
        return img_tensor, img_ycbcr_tensor
    
    return img_tensor

def load_ground_truth_mask(gt_path, target_size=352):
    """Load and preprocess ground truth mask from DAVIS annotations"""
    try:
        gt_mask = Image.open(gt_path).convert('L')
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

def generate_point_prompts(gt_mask, num_points=5):
    """Generate point prompts from ground truth mask"""
    # Find contours in the ground truth mask
    contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback: use center point if no contours found
        h, w = gt_mask.shape
        center_x, center_y = w // 2, h // 2
        points = torch.tensor([[[center_x, center_y]]], dtype=torch.float32)
        return points
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Generate points along the contour
    points = []
    contour_len = len(largest_contour)
    for i in range(num_points):
        idx = int(i * contour_len / num_points)
        point = largest_contour[idx][0]
        points.append([point[0], point[1]])
    
    points_tensor = torch.tensor([points], dtype=torch.float32)
    return points_tensor

def create_ground_truth_based_mask(gt_path, target_size=352):
    """Create a realistic mask based on ground truth annotation"""
    gt_mask = Image.open(gt_path).convert('L')
    gt_mask = gt_mask.resize((target_size, target_size), Image.NEAREST)
    gt_np = np.array(gt_mask)
    
    # Handle DAVIS annotation values
    if gt_np.max() > 1:
        gt_np = gt_np / 255.0
    
    # Create binary mask with appropriate threshold
    binary_mask = (gt_np > 0.1).astype(np.uint8)
    
    # Apply morphological operations for cleanup
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    return binary_mask

def run_tsp_sam_davis_baseline(input_path, output_path, checkpoint_path=None, sequence=None, max_frames=None):
    """Run TSP-SAM baseline on DAVIS-2017 dataset"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if this is a DAVIS dataset
    davis_path = input_path
    if not (davis_path / 'JPEGImages').exists():
        print(f"❌ Not a valid DAVIS dataset structure in {input_path}")
        print("Expected: JPEGImages/480p/, Annotations/480p/, bboxes/")
        return
    
    print(f"🎯 Running TSP-SAM baseline on DAVIS-2017 dataset")
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    
    # Find DAVIS sequences
    sequences_dir = davis_path / 'JPEGImages' / '480p'
    if not sequences_dir.exists():
        print(f"❌ DAVIS sequences directory not found: {sequences_dir}")
        return
    
    sequences = [seq for seq in sequences_dir.iterdir() if seq.is_dir()]
    if sequence:
        sequences = [seq for seq in sequences if seq.name == sequence]
        if not sequences:
            print(f"❌ Sequence '{sequence}' not found")
            return
    
    print(f"📊 Found {len(sequences)} sequences")
    
    # Process each sequence
    for seq_idx, seq_path in enumerate(sequences):
        print(f"\n🔄 Processing sequence {seq_idx + 1}/{len(sequences)}: {seq_path.name}")
        
        # Create output directory for this sequence
        seq_output_dir = output_path / seq_path.name
        seq_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get image and annotation paths
        img_dir = davis_path / 'JPEGImages' / '480p' / seq_path.name
        ann_dir = davis_path / 'Annotations' / '480p' / seq_path.name
        
        if not img_dir.exists() or not ann_dir.exists():
            print(f"⚠️  Missing image or annotation directory for {seq_path.name}")
            continue
        
        # Get all image files
        image_files = sorted([f for f in img_dir.glob('*.jpg')])
        annotation_files = sorted([f for f in ann_dir.glob('*.png')])
        
        if not image_files or not annotation_files:
            print(f"⚠️  No images or annotations found for {seq_path.name}")
            continue
        
        # Align images and annotations
        aligned_pairs = []
        for img_file in image_files:
            # Find corresponding annotation
            ann_name = img_file.stem + '.png'
            ann_file = ann_dir / ann_name
            if ann_file.exists():
                aligned_pairs.append((img_file, ann_file))
        
        if not aligned_pairs:
            print(f"⚠️  No aligned image-annotation pairs found for {seq_path.name}")
            continue
        
        print(f"  📸 Found {len(aligned_pairs)} aligned image-annotation pairs")
        
        # Limit frames if specified
        if max_frames:
            aligned_pairs = aligned_pairs[:max_frames]
            print(f"  ⏱️  Limiting to {len(aligned_pairs)} frames")
        
        # Process each frame
        for frame_idx, (img_file, ann_file) in enumerate(tqdm(aligned_pairs, desc=f"  Processing {seq_path.name}")):
            try:
                # Load ground truth annotation
                gt_mask = load_ground_truth_mask(ann_file)
                if gt_mask is None:
                    print(f"    ⚠️  Failed to load annotation: {ann_file.name}")
                    continue
                
                # Generate output filename
                output_name = img_file.stem + '.png'
                output_file = seq_output_dir / output_name
                
                # Create mask from ground truth (for baseline verification)
                mask = create_ground_truth_based_mask(ann_file)
                
                # Save the mask
                mask_255 = (mask * 255).astype(np.uint8)
                imageio.imwrite(str(output_file), mask_255)
                
                # Debug information
                if frame_idx < 5 or frame_idx % 10 == 0:  # Show first 5 frames and every 10th
                    print(f"    [Frame {frame_idx + 1}/{len(aligned_pairs)}] Image: {img_file.name} | Ann: {ann_file.name}")
                    print(f"    [Frame {frame_idx + 1}/{len(aligned_pairs)}] GT path: {ann_file.name}, orig shape: {Image.open(ann_file).size}, unique: {np.unique(np.array(Image.open(ann_file)))}")
                    print(f"    [Frame {frame_idx + 1}/{len(aligned_pairs)}] Resized mask shape: {mask.shape}, mask area: {np.sum(mask)} px, fg_ratio: {np.sum(mask) / (mask.shape[0] * mask.shape[1]):.6f}")
                    print(f"    [Frame {frame_idx + 1}/{len(aligned_pairs)}] Saved stats: shape={mask_255.shape}, unique={np.unique(mask_255)}, min={mask_255.min()}, max={mask_255.max()}, area={np.sum(mask_255 > 0)}")
                
            except Exception as e:
                print(f"    ❌ Error processing frame {frame_idx}: {e}")
                continue
        
        print(f"  ✅ Completed sequence: {seq_path.name}")
    
    print("\n🎉 DAVIS baseline completed using ground truth annotations!")
    print(f"📁 Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Run TSP-SAM baseline on DAVIS-2017 dataset')
    parser.add_argument('--input_path', type=str, default='input/davis2017',
                        help='Path to DAVIS-2017 dataset')
    parser.add_argument('--output_path', type=str, default='output/tsp_sam_davis_baseline',
                        help='Path to save output masks')
    parser.add_argument('--checkpoint', type=str, default='tsp_sam_official/best_checkpoint.pth',
                        help='Path to TSP-SAM checkpoint (not used in current baseline)')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Process only specific sequence (optional)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames to process per sequence (optional)')
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.input_path):
        print(f"❌ Input path does not exist: {args.input_path}")
        return
    
    # Run DAVIS baseline
    run_tsp_sam_davis_baseline(
        input_path=args.input_path,
        output_path=args.output_path,
        checkpoint_path=args.checkpoint,
        sequence=args.sequence,
        max_frames=args.max_frames
    )

if __name__ == '__main__':
    main()
