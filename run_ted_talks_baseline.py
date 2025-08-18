#!/usr/bin/env python3
"""
TED Talks Baseline Runner with Pose-Based Frame Filtering
Applies TSP-SAM and SAMURAI to TED Talks dataset, filtering frames with valid pose keypoints
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
import json

# Add paths for models
sys.path.append('tsp_sam_official')
sys.path.append('samurai_official')

def detect_pose_keypoints(image_path, confidence_threshold=0.5):
    """
    Detect pose keypoints in an image using MediaPipe or OpenPose
    Returns True if valid pose keypoints are detected
    """
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return False
        
        # Simple pose detection using MediaPipe (you can replace with OpenPose)
        import mediapipe as mp
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=confidence_threshold
        )
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Check if we have enough valid keypoints
            valid_keypoints = 0
            for landmark in results.pose_landmarks.landmark:
                if landmark.visibility > confidence_threshold:
                    valid_keypoints += 1
            
            # Consider pose valid if we have at least 10 keypoints
            return valid_keypoints >= 10
        
        return False
        
    except ImportError:
        # Fallback: simple edge detection to estimate if person is present
        print("MediaPipe not available, using fallback detection")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return False
        
        # Simple edge detection
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        # Assume person present if reasonable edge density
        return 0.01 < edge_density < 0.1
    
    except Exception as e:
        print(f"Error in pose detection: {e}")
        return False

def filter_frames_by_pose(sequence_dir, confidence_threshold=0.5):
    """
    Filter frames to only include those with valid pose keypoints
    """
    print(f"🔍 Filtering frames by pose keypoints for {sequence_dir.name}")
    
    image_files = sorted([f for f in sequence_dir.glob('*.jpg')])
    valid_frames = []
    
    for img_file in tqdm(image_files, desc="Detecting poses"):
        if detect_pose_keypoints(img_file, confidence_threshold):
            valid_frames.append(img_file)
    
    print(f"✅ Found {len(valid_frames)} valid frames out of {len(image_files)} total frames")
    return valid_frames

def run_ted_talks_baseline(input_path, output_path, method="both", confidence_threshold=0.5):
    """
    Run baseline evaluation on TED Talks dataset with pose filtering
    """
    
    print(f"🎭 Starting TED Talks baseline evaluation")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Method: {method}")
    print(f"Pose confidence threshold: {confidence_threshold}")
    
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find TED Talks sequences
    ted_path = Path(input_path) / "ted"
    if not ted_path.exists():
        print(f"TED Talks directory not found: {ted_path}")
        return
    
    sequences = [d for d in ted_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    sequences = sorted(sequences)
    
    print(f"Found {len(sequences)} TED Talks sequences")
    
    # Process each sequence
    for seq_idx, seq_path in enumerate(sequences):
        print(f"\n[{seq_idx + 1}/{len(sequences)}] Processing sequence: {seq_path.name}")
        
        # Create sequence output directory
        seq_output_dir = output_path / seq_path.name
        seq_output_dir.mkdir(exist_ok=True)
        
        # Filter frames by pose keypoints
        valid_frames = filter_frames_by_pose(seq_path, confidence_threshold)
        
        if not valid_frames:
            print(f"⚠️  No valid frames found for {seq_path.name}, skipping...")
            continue
        
        # Save frame filtering results
        filtering_info = {
            "sequence": seq_path.name,
            "total_frames": len(list(seq_path.glob('*.jpg'))),
            "valid_frames": len(valid_frames),
            "filtering_ratio": len(valid_frames) / len(list(seq_path.glob('*.jpg'))),
            "confidence_threshold": confidence_threshold,
            "valid_frame_names": [f.stem for f in valid_frames]
        }
        
        with open(seq_output_dir / "filtering_info.json", 'w') as f:
            json.dump(filtering_info, f, indent=2)
        
        print(f"📊 Filtering results: {filtering_info['filtering_ratio']:.2%} frames retained")
        
        # Process valid frames (placeholder for actual segmentation)
        for frame_idx, frame_path in enumerate(tqdm(valid_frames, desc=f"  Processing {seq_path.name}")):
            try:
                # Generate output filename
                output_name = frame_path.stem + '_filtered.png'
                output_file = seq_output_dir / output_name
                
                # For now, just copy the frame as placeholder
                # TODO: Replace with actual TSP-SAM or SAMURAI segmentation
                frame = Image.open(frame_path)
                frame.save(output_file)
                
            except Exception as e:
                print(f"    Error processing frame {frame_idx}: {e}")
                continue
        
        print(f"  ✅ Completed sequence: {seq_path.name}")
    
    print(f"\n🎭 TED Talks baseline completed with pose filtering!")
    print(f"Results saved to: {output_path}")
    print(f"💡 Next: Integrate TSP-SAM and SAMURAI segmentation on filtered frames")

def main():
    parser = argparse.ArgumentParser(description='Run TED Talks baseline with pose filtering')
    parser.add_argument('--input_path', type=str, default='input',
                        help='Path to input datasets')
    parser.add_argument('--output_path', type=str, default='output/ted_talks_baseline',
                        help='Path to save output results')
    parser.add_argument('--method', type=str, default='both', choices=['tsp-sam', 'samurai', 'both'],
                        help='Segmentation method to use')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Pose detection confidence threshold')
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.input_path):
        print(f"Input path does not exist: {args.input_path}")
        return
    
    # Run TED Talks baseline
    run_ted_talks_baseline(
        input_path=args.input_path,
        output_path=args.output_path,
        method=args.method,
        confidence_threshold=args.confidence_threshold
    )

if __name__ == '__main__':
    main()
