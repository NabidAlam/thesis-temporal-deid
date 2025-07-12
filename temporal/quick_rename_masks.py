import os
from pathlib import Path

def rename_masks(pred_root, sequences):
    for seq in sequences:
        mask_dir = Path(pred_root) / seq
        print(f"[INFO] Checking: {mask_dir}")
        if not mask_dir.exists():
            print(f"[WARNING] Directory not found: {mask_dir}")
            continue

        for file in os.listdir(mask_dir):
            if file.endswith("_mask.png"):
                old_path = mask_dir / file
                new_file = file.replace("_mask", "")
                new_path = mask_dir / new_file

                if not new_path.exists():
                    os.rename(old_path, new_path)
                    print(f"[RENAMED] {file} → {new_file}")
                else:
                    print(f"[SKIP] {new_file} already exists.")

if __name__ == "__main__":
    rename_masks("output/tsp_sam/davis", sequences=["camel", "boat"])
