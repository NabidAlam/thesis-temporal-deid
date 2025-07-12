import os
import numpy as np
import cv2
from glob import glob
from pathlib import Path
import csv

def compute_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union != 0 else np.nan

def binarize_gt_mask(gt_img):
    """Extracts the red-channel-based binary mask (DAVIS-style)."""
    if len(gt_img.shape) == 3 and gt_img.shape[2] == 3:
        red_channel = gt_img[:, :, 0]
        return red_channel > 127
    else:
        return gt_img > 127

def evaluate_davis(pred_dir, gt_dir, sequences=None, log_csv="iou_log.csv"):
    if sequences is None:
        sequences = sorted(os.listdir(pred_dir))

    all_ious = []
    csv_rows = [("sequence", "frame", "iou")]

    for seq in sequences:
        pred_seq_path = Path(pred_dir) / seq
        gt_seq_path = Path(gt_dir) / seq

        if not gt_seq_path.exists():
            print(f"[WARNING] GT not found for: {seq}, skipping...")
            continue

        pred_masks = sorted(glob(str(pred_seq_path / "*.png")))
        gt_masks = sorted(glob(str(gt_seq_path / "*.png")))

        if len(pred_masks) != len(gt_masks):
            print(f"[WARNING] Frame count mismatch in '{seq}': {len(pred_masks)} pred vs {len(gt_masks)} GT")

        ious = []
        for i, (pred_path, gt_path) in enumerate(zip(pred_masks, gt_masks)):
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)

            if pred is None or gt is None:
                print(f"[ERROR] Failed to read: {pred_path} or {gt_path}")
                continue

            if pred.shape != gt.shape[:2]:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

            pred_bin = pred > 127
            gt_bin = binarize_gt_mask(gt)
            iou = compute_iou(pred_bin, gt_bin)
            ious.append(iou)
            csv_rows.append((seq, i, round(iou, 4)))

        if ious:
            seq_mean_iou = np.nanmean(ious)
            all_ious.extend(ious)
            print(f"{seq:>12}: mIoU = {seq_mean_iou:.4f}")
        else:
            print(f"{seq:>12}: ❗ No valid masks for mIoU.")

    print("=" * 40)
    if all_ious:
        print(f"Overall DAVIS mIoU: {np.nanmean(all_ious):.4f}")
    else:
        print("No sequences evaluated. Check input paths and frame consistency.")

    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

if __name__ == "__main__":
    pred_dir = "output/tsp_sam/davis"
    gt_dir = "input/davis2017/Annotations/480p"
    evaluate_davis(pred_dir, gt_dir, sequences=["bus", "camel"])
