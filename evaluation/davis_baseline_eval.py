"""
DAVIS-2017 Baseline Evaluation Script
Evaluates TSP-SAM and SAMURAI performance on DAVIS dataset
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class DAVISEvaluator:
    def __init__(self, davis_root="input/davis2017", output_root="output"):
        self.davis_root = Path(davis_root)
        self.output_root = Path(output_root)
        self.annotations_dir = self.davis_root / "Annotations" / "480p"
        self.images_dir = self.davis_root / "JPEGImages" / "480p"
        
    def load_ground_truth(self, sequence):
        """Load DAVIS ground truth masks for a sequence"""
        gt_dir = self.annotations_dir / sequence
        if not gt_dir.exists():
            print(f"GT directory not found: {gt_dir}")
            return {}
        
        mask_files = sorted(gt_dir.glob("*.png"))
        gt_masks = {}
        
        print(f"Loading {len(mask_files)} GT masks for {sequence}")
        
        for mask_file in mask_files:
            frame_name = mask_file.stem  # e.g., "00000"
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # DAVIS GT: 0=background, >0=objects
                gt_masks[frame_name] = (mask > 0).astype(np.uint8)
        
        return gt_masks
    
    def load_predictions(self, method, sequence):
        """Load predicted masks from TSP-SAM or SAMURAI"""
        if method.lower() == "samurai":
            return self._load_samurai_predictions(sequence)
        elif method.lower() == "tsp-sam":
            return self._load_tspsam_predictions(sequence)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _load_samurai_predictions(self, sequence):
        """Load SAMURAI mask outputs"""
        pred_dir = self.output_root / "samurai" / "davis" / sequence
        
        if not pred_dir.exists():
            print(f"Warning: SAMURAI output not found at {pred_dir}")
            return {}
        
        pred_masks = {}
        mask_files = sorted(pred_dir.glob("mask_obj*_frame*.png"))
        
        print(f"Loading {len(mask_files)} SAMURAI predictions for {sequence}")
        
        for mask_file in mask_files:
            # Parse filename: mask_obj0_frame00000.png
            parts = mask_file.stem.split("_")
            frame_idx = parts[-1].replace("frame", "")  # "00000"
            
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                pred_masks[frame_idx] = (mask > 127).astype(np.uint8)
        
        return pred_masks
    
    # def _load_tspsam_predictions(self, sequence):
    #     """Load TSP-SAM mask outputs (adapt to your TSP-SAM output format)"""
    #     pred_dir = self.output_root / "tsp_sam" / "davis" / sequence
        
    #     if not pred_dir.exists():
    #         print(f"Warning: TSP-SAM output not found at {pred_dir}")
    #         return {}
        
    #     pred_masks = {}
    #     # Adapt this based on your TSP-SAM output naming convention
    #     mask_files = sorted(pred_dir.glob("*_mask.png"))
        
    #     for mask_file in mask_files:
    #         # Extract frame index from filename
    #         frame_idx = mask_file.stem.split("_")[0]  # Adjust as needed
    #         mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    #         if mask is not None:
    #             pred_masks[frame_idx] = (mask > 127).astype(np.uint8)
        
    #     return pred_masks
    
    def _load_tspsam_predictions(self, sequence):
        pred_dir = self.output_root / "tsp_sam" / "davis" / sequence
        print(f"[DEBUG] Checking TSP-SAM prediction directory: {pred_dir}")
        if not pred_dir.exists():
            print(f"Warning: TSP-SAM output not found at {pred_dir}")
            return {}
        pred_masks = {}
        mask_files = sorted(pred_dir.glob("*.png"))
        print(f"[DEBUG] Found {len(mask_files)} TSP-SAM prediction files for {sequence}: {mask_files[:5]}")
        for mask_file in mask_files:
            if "raw_" in mask_file.stem or "overlay_" in mask_file.stem:
                continue
            frame_idx = mask_file.stem.split("_")[0]
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                pred_masks[frame_idx] = (mask > 127).astype(np.uint8)
            else:
                print(f"[DEBUG] Failed to load TSP-SAM prediction: {mask_file}")
        return pred_masks
    
    def compute_iou(self, pred_mask, gt_mask):
        """Compute Intersection over Union"""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def compute_temporal_consistency(self, pred_masks):
        """Compute temporal consistency using frame-to-frame IoU"""
        if len(pred_masks) < 2:
            return 0.0
        
        frame_keys = sorted(pred_masks.keys())
        consistency_scores = []
        
        for i in range(1, len(frame_keys)):
            prev_mask = pred_masks[frame_keys[i-1]]
            curr_mask = pred_masks[frame_keys[i]]
            
            iou = self.compute_iou(curr_mask, prev_mask)
            consistency_scores.append(iou)
        
        return np.mean(consistency_scores)
    
    def evaluate_sequence(self, method, sequence):
        """Evaluate a single sequence"""
        print(f"Evaluating {method} on sequence: {sequence}")
        
        # Load data
        gt_masks = self.load_ground_truth(sequence)
        pred_masks = self.load_predictions(method, sequence)
        
        if not gt_masks:
            print(f"No ground truth found for {sequence}")
            return None
        
        if not pred_masks:
            print(f"No predictions found for {method} on {sequence}")
            return None
        
        # Find common frames
        common_frames = set(gt_masks.keys()) & set(pred_masks.keys())
        
        if not common_frames:
            print(f"No common frames between GT and predictions for {sequence}")
            print(f"GT frames: {sorted(list(gt_masks.keys()))[:5]}...")
            print(f"Pred frames: {sorted(list(pred_masks.keys()))[:5]}...")
            return None
        
        print(f"Evaluating {len(common_frames)} common frames for {sequence}")
        
        # Compute metrics
        frame_results = []
        
        for frame_id in sorted(common_frames):
            gt_mask = gt_masks[frame_id]
            pred_mask = pred_masks[frame_id]
            
            # Resize if necessary
            if gt_mask.shape != pred_mask.shape:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            
            iou = self.compute_iou(pred_mask, gt_mask)
            
            frame_results.append({
                'frame_id': frame_id,
                'iou': float(iou),  # Convert to Python float
                'gt_area': int(gt_mask.sum()),  # Convert to Python int
                'pred_area': int(pred_mask.sum())  # Convert to Python int
            })
        
        # Temporal consistency
        temporal_consistency = self.compute_temporal_consistency(pred_masks)
        
        # Aggregate metrics
        ious = [r['iou'] for r in frame_results]
        
        return {
            'sequence': sequence,
            'method': method,
            'mean_iou': float(np.mean(ious)),
            'temporal_consistency': float(temporal_consistency),
            'num_frames': len(frame_results),
            'min_iou': float(np.min(ious)),
            'max_iou': float(np.max(ious)),
            'std_iou': float(np.std(ious)),
            'frame_results': frame_results
        }
    
    def evaluate_method(self, method, sequences):
        """Evaluate a method on multiple sequences"""
        results = []
        
        for sequence in sequences:
            seq_result = self.evaluate_sequence(method, sequence)
            if seq_result:
                results.append(seq_result)
        
        return results
    
    def save_results(self, results, output_path):
        """Save evaluation results"""
        if not results:
            print("No results to save!")
            return None
            
        # Create summary
        summary = []
        for result in results:
            summary.append({
                'sequence': result['sequence'],
                'method': result['method'],
                'mean_iou': result['mean_iou'],
                'temporal_consistency': result['temporal_consistency'],
                'num_frames': result['num_frames']
            })
        
        # Save to CSV
        df = pd.DataFrame(summary)
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        
        # Convert all results to JSON-serializable format
        results_json = convert_numpy_types(results)
        
        # Save detailed results to JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Results saved to: {csv_path} and {json_path}")
        return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate DAVIS baselines")
    parser.add_argument("--method", required=True, choices=["samurai", "tsp-sam"],
                       help="Method to evaluate")
    parser.add_argument("--sequences", nargs="+", 
                       default=["dog", "camel", "hike", "dance-twirl"],
                       help="Sequences to evaluate")
    parser.add_argument("--davis_root", default="input/davis2017",
                       help="DAVIS dataset root")
    parser.add_argument("--output_root", default="output",
                       help="Output root directory")
    parser.add_argument("--save_path", default="evaluation/results/baseline_results",
                       help="Path to save results")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = DAVISEvaluator(args.davis_root, args.output_root)
    
    # Run evaluation
    results = evaluator.evaluate_method(args.method, args.sequences)
    
    if not results:
        print("No results to save!")
        return
    
    # Save results
    save_path = Path(args.save_path)
    
    # Create method-specific subfolder
    method_folder = save_path.parent / save_path.stem / args.method
    method_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = method_folder / f"baseline_results_{args.method}_{timestamp}"
    
    df = evaluator.save_results(results, final_path)
    
    if df is not None:
        # Print summary
        print(f"\n=== {args.method.upper()} Evaluation Summary ===")
        print(f"Sequences evaluated: {len(results)}")
        print(f"Average IoU: {df['mean_iou'].mean():.3f} ± {df['mean_iou'].std():.3f}")
        print(f"Average Temporal Consistency: {df['temporal_consistency'].mean():.3f}")
        print("\nPer-sequence results:")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
    
    
# # This will now create proper folder organization
# python evaluation/davis_baseline_eval.py --method samurai --sequences hike dog camel

# # When you run TSP-SAM later
# python evaluation/davis_baseline_eval.py --method tsp-sam --sequences hike dog camel

# python evaluation/davis_baseline_eval.py --method samurai --sequences hike dog camel