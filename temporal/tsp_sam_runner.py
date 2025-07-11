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

# Add root and module paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath("tsp_sam"))
sys.path.append(os.path.abspath("temporal"))

from tsp_sam.lib.pvtv2_afterTEM import Network
from utils import save_mask_and_frame, resize_frame


def get_adaptive_threshold(prob_np, percentile=98):
    return np.percentile(prob_np.flatten(), percentile)


def model_infer_real(model, frame, debug_save_dir=None, frame_idx=None, min_area=200):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        if output.dim() == 4:
            output = output[0]
        output = output.squeeze()

        prob = torch.sigmoid(output)
        prob_np = prob.cpu().numpy()

    adaptive_thresh = get_adaptive_threshold(prob_np, percentile=98)
    raw_mask = (prob_np > adaptive_thresh).astype(np.uint8)

    # Apply median filter
    mask_denoised = median_filter(raw_mask, size=3)

    # Filter small blobs
    num_pixels = np.sum(mask_denoised)
    if num_pixels < min_area:
        final_mask = np.zeros_like(mask_denoised, dtype=np.uint8)
    else:
        final_mask = mask_denoised * 255

    stats = {
        "mean": float(prob_np.mean()),
        "max": float(prob_np.max()),
        "min": float(prob_np.min()),
        "adaptive_thresh": float(adaptive_thresh),
        "mask_area": int(num_pixels)
    }

    print(f"[DEBUG] Frame {frame_idx} — shape: {prob_np.shape}, max: {stats['max']:.4f}, "
          f"min: {stats['min']:.4f}, mean: {stats['mean']:.4f}, "
          f"adaptive_thresh: {adaptive_thresh:.4f}, area: {num_pixels}")

    # Save debug masks and histogram
    if debug_save_dir and frame_idx is not None:
        os.makedirs(debug_save_dir, exist_ok=True)
        for thresh in [0.3, 0.5, 0.7, 0.9]:
            temp_mask = (prob_np > thresh).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(debug_save_dir, f"{frame_idx:05d}_th{int(thresh * 100)}.png"), temp_mask)

        cv2.imwrite(os.path.join(debug_save_dir, f"{frame_idx:05d}_adaptive_{int(adaptive_thresh * 100)}.png"), final_mask)

        plt.hist(prob_np.ravel(), bins=50)
        plt.title(f"Pixel Probabilities (frame {frame_idx})")
        plt.savefig(os.path.join(debug_save_dir, f"{frame_idx:05d}_hist.png"))
        plt.close()

    return final_mask, stats


def run_tsp_sam(input_path, output_path_base, config_path):
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    infer_cfg = config["inference"]
    output_cfg = config["output"]
    frame_stride = infer_cfg.get("frame_stride", 2)

    class Opt: pass
    opt = Opt()
    opt.resume = model_cfg["checkpoint_path"]
    opt.device = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    opt.gpu_ids = [0] if opt.device == "cuda" and torch.cuda.is_available() else []
    opt.channel = model_cfg.get("channel", 32)

    print(f"Initializing model on device: {opt.device}")
    model = Network(opt)
    if opt.device == "cuda":
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids).cuda()
    else:
        model = model.cpu()

    print(f"Loading weights from: {opt.resume}")
    pretrained_weights = torch.load(opt.resume, map_location=opt.device)
    result = model.module.feat_net.pvtv2_en.load_state_dict(pretrained_weights, strict=False)
    if isinstance(result, dict):
        print("Weights loaded.")
        print("   - Missing keys:", result.get("missing_keys", []))
        print("   - Unexpected keys:", result.get("unexpected_keys", []))

    model.eval()

    video_name = Path(input_path).stem
    output_path = Path(output_path_base) / video_name
    if output_path.exists():
        print(f"Cleaning previous run: {output_path}")
        import shutil
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    debug_csv_path = output_path / "debug_stats.csv"
    debug_file = open(debug_csv_path, mode="w", newline="")
    csv_writer = csv.writer(debug_file)
    csv_writer.writerow(["frame_idx", "mean", "max", "min", "adaptive_thresh", "mask_area"])

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    frame_idx = 0
    saved_frames = 0
    empty_masks = 0

    with tqdm(total=total_frames // frame_stride, desc="Processing Frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            frame_resized = resize_frame(frame, infer_cfg)
            mask, stats = model_infer_real(
                model,
                frame_resized,
                debug_save_dir=output_path / "threshold_debug",
                frame_idx=frame_idx
            )
            
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            csv_writer.writerow([
                frame_idx, stats["mean"], stats["max"], stats["min"],
                stats["adaptive_thresh"], stats["mask_area"]
            ])

            if mask.sum() > 0:
                save_mask_and_frame(
                    frame,
                    mask_resized,
                    str(output_path),
                    frame_idx,
                    save_overlay=output_cfg.get("save_overlay", False),
                    overlay_alpha=output_cfg.get("overlay_alpha", 0.5),
                    save_frames=output_cfg.get("save_frames", False),
                    save_composite=True
                )
                saved_frames += 1
            else:
                empty_masks += 1
                if empty_masks <= 3:
                    print(f"Frame {frame_idx}: Mask is empty.")

            frame_idx += 1
            pbar.update(1)

    cap.release()
    debug_file.close()

    print(f"\nFinished processing {frame_idx} frames.")
    print(f"Masks saved: {saved_frames}")
    print(f"Empty masks: {empty_masks}")
    print(f"Debug stats written to: {debug_csv_path}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python tsp_sam_runner.py <input_video> <output_base_dir> <config_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_base = sys.argv[2]
    config_path = sys.argv[3]

    run_tsp_sam(input_path, output_base, config_path)
