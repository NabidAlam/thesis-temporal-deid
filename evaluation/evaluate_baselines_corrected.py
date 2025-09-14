"""
Corrected Baseline Evaluation Script
Evaluates TSP-SAM and SAMURAI performance on DAVIS-2017 dataset
Works with our actual output structure
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

class CorrectedVideoEvaluator:
    def __init__(self, dataset_root="input", output_root="output", dataset="davis"):
        self.dataset = dataset
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        
        if dataset == "davis":
            self.annotations_dir = self.dataset_root / "davis2017" / "Annotations" / "480p"
            self.images_dir = self.dataset_root / "davis2017" / "JPEGImages" / "480p"
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
    def load_ground_truth(self, sequence):
        """Load ground truth masks for a sequence (DAVIS only)"""
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
        """Load SAMURAI mask outputs from our actual structure"""
        # Try different possible paths
        possible_paths = [
            self.output_root / "samurai_davis_baseline_all_sequences" / sequence,
            self.output_root / "samurai_davis_baseline" / sequence,
            self.output_root / "samurai_davis_full" / sequence,
            self.output_root / "full_evaluation" / "samurai_davis_baseline" / sequence
        ]
        
        pred_dir = None
        for path in possible_paths:
            if path.exists():
                pred_dir = path
                break
        
        if not pred_dir:
            print(f"Warning: SAMURAI output not found for {sequence}")
            print(f"Tried paths: {[str(p) for p in possible_paths]}")
            return {}
        
        pred_masks = {}
        mask_files = sorted(pred_dir.glob("*.png"))
        
        print(f"Loading {len(mask_files)} SAMURAI predictions for {sequence} from {pred_dir}")
        
        for mask_file in mask_files:
            frame_name = mask_file.stem  # e.g., "00000"
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Convert to binary mask
                pred_masks[frame_name] = (mask > 127).astype(np.uint8)
        
        return pred_masks
    
    def _load_tspsam_predictions(self, sequence):
        """Load TSP-SAM mask outputs from our actual structure"""
        # Try different possible paths
        possible_paths = [
            self.output_root / "tsp_sam_davis_baseline_all_sequences" / sequence,
            self.output_root / "tsp_sam_davis_baseline" / sequence,
            self.output_root / "tsp_sam_davis_full" / sequence,
            self.output_root / "full_evaluation" / "tsp_sam_davis_baseline" / sequence
        ]
        
        pred_dir = None
        for path in possible_paths:
            if path.exists():
                pred_dir = path
                break
        
        if not pred_dir:
            print(f"Warning: TSP-SAM output not found for {sequence}")
            print(f"Tried paths: {[str(p) for p in possible_paths]}")
            return {}
        
        pred_masks = {}
        mask_files = sorted(pred_dir.glob("*.png"))
        
        print(f"Loading {len(mask_files)} TSP-SAM predictions for {sequence} from {pred_dir}")
        
        for mask_file in mask_files:
            frame_name = mask_file.stem  # e.g., "00000"
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Convert to binary mask
                pred_masks[frame_name] = (mask > 127).astype(np.uint8)
        
        return pred_masks
    
    def compute_iou(self, pred_mask, gt_mask):
        """Compute Intersection over Union"""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def compute_dice(self, pred_mask, gt_mask):
        """Compute Dice coefficient"""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        total = pred_mask.sum() + gt_mask.sum()
        
        if total == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return (2 * intersection) / total
    
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
        print(f"\nEvaluating {method} on sequence: {sequence}")
        
        # Load data
        gt_masks = self.load_ground_truth(sequence)
        pred_masks = self.load_predictions(method, sequence)
        
        if not pred_masks:
            print(f"No predictions found for {method} on {sequence}")
            return None
        
        # Find common frames
        common_frames = set(gt_masks.keys()) & set(pred_masks.keys())
        
        if not common_frames:
            print(f"No common frames to evaluate for {sequence}")
            print(f"GT frames: {sorted(list(gt_masks.keys()))[:5]}...")
            print(f"Pred frames: {sorted(list(pred_masks.keys()))[:5]}...")
            return None
        
        print(f"Evaluating {len(common_frames)} frames for {sequence}")
        
        # Compute metrics
        frame_results = []
        
        for frame_id in sorted(common_frames):
            pred_mask = pred_masks[frame_id]
            gt_mask = gt_masks[frame_id]
            
            # Resize if necessary
            if gt_mask.shape != pred_mask.shape:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            
            iou = self.compute_iou(pred_mask, gt_mask)
            dice = self.compute_dice(pred_mask, gt_mask)
            
            metrics = {
                'frame_id': frame_id,
                'iou': float(iou),
                'dice': float(dice),
                'pred_area': int(pred_mask.sum()),
                'gt_area': int(gt_mask.sum())
            }
            
            frame_results.append(metrics)
        
        # Temporal consistency
        temporal_consistency = self.compute_temporal_consistency(pred_masks)
        
        # Aggregate metrics
        ious = [r['iou'] for r in frame_results]
        dices = [r['dice'] for r in frame_results]
        
        agg_metrics = {
            'sequence': sequence,
            'method': method,
            'temporal_consistency': float(temporal_consistency),
            'num_frames': len(frame_results),
            'mean_iou': float(np.mean(ious)),
            'min_iou': float(np.min(ious)),
            'max_iou': float(np.max(ious)),
            'std_iou': float(np.std(ious)),
            'mean_dice': float(np.mean(dices)),
            'min_dice': float(np.min(dices)),
            'max_dice': float(np.max(dices)),
            'std_dice': float(np.std(dices)),
            'frame_results': frame_results
        }
        
        return agg_metrics
    
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
            sum_entry = {
                'sequence': result['sequence'],
                'method': result['method'],
                'temporal_consistency': result['temporal_consistency'],
                'num_frames': result['num_frames'],
                'mean_iou': result['mean_iou'],
                'std_iou': result['std_iou'],
                'mean_dice': result['mean_dice'],
                'std_dice': result['std_dice']
            }
            summary.append(sum_entry)
        
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
    parser = argparse.ArgumentParser(description="Evaluate video baselines with corrected paths")
    parser.add_argument("--method", required=True, choices=["samurai", "tsp-sam"],
                       help="Method to evaluate")
    parser.add_argument("--sequences", nargs="+", 
                       default=["dog", "camel", "hike", "dance-twirl"],
                       help="Sequences to evaluate")
    parser.add_argument("--dataset", default="davis", choices=["davis"],
                       help="Dataset to evaluate on")
    parser.add_argument("--dataset_root", default="input",
                       help="Dataset root directory")
    parser.add_argument("--output_root", default="output",
                       help="Output root directory")
    parser.add_argument("--save_path", default="evaluation/results/corrected_baseline_results",
                       help="Path to save results")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = CorrectedVideoEvaluator(args.dataset_root, args.output_root, args.dataset)
    
    # Run evaluation
    results = evaluator.evaluate_method(args.method, args.sequences)
    
    if not results:
        print("No results to save!")
        return
    
    # Save results
    save_path = Path(args.save_path)
    
    # Create method-specific subfolder
    method_folder = save_path.parent / save_path.stem / args.method / args.dataset
    method_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = method_folder / f"baseline_results_{args.method}_{args.dataset}_{timestamp}"
    
    df = evaluator.save_results(results, final_path)
    
    if df is not None:
        # Print summary
        print(f"\n=== {args.method.upper()} Evaluation Summary on {args.dataset.upper()} ===")
        print(f"Sequences evaluated: {len(results)}")
        print(f"Average Temporal Consistency: {df['temporal_consistency'].mean():.3f}")
        print(f"Average IoU: {df['mean_iou'].mean():.3f} ± {df['mean_iou'].std():.3f}")
        print(f"Average Dice: {df['mean_dice'].mean():.3f} ± {df['mean_dice'].std():.3f}")
        print("\nPer-sequence results:")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
