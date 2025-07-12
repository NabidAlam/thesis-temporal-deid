
# python temporal/tsp_sam_runner.py input/ted/video1.mp4 output/tsp_sam/ted configs/tsp_sam_ted.yaml

# python temporal/tsp_sam_davis_runner.py input/davis2017/JPEGImages/480p/camel output/tsp_sam/davis configs/tsp_sam_davis.yaml

# import os
# import sys
# import cv2
# import yaml
# import torch
# import numpy as np
# from pathlib import Path
# from PIL import Image
# import torchvision.transforms as T
# from tqdm import tqdm
# import csv
# import matplotlib.pyplot as plt
# from scipy.ndimage import median_filter
# import json

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# sys.path.append(os.path.abspath("tsp_sam"))
# sys.path.append(os.path.abspath("temporal"))

# from tsp_sam.lib.pvtv2_afterTEM import Network
# from utils import save_mask_and_frame, resize_frame
# from maskanyone_sam_wrapper import MaskAnyoneSAMWrapper
# from pose_extractor import extract_pose_keypoints
# from my_sam2_client import MySAM2Client

# def post_process_fused_mask(fused_mask, min_area=100, kernel_size=5):
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
#     cleaned = cv2.morphologyEx(fused_mask, cv2.MORPH_OPEN, kernel)
#     closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
#     contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     filtered = np.zeros_like(closed)
#     for cnt in contours:
#         if cv2.contourArea(cnt) > min_area:
#             cv2.drawContours(filtered, [cnt], -1, 255, -1)
#     return cv2.dilate(filtered, kernel, iterations=1)

# def extract_bbox_from_mask(mask, margin_ratio=0.05):
#     contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None
#     x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
#     margin_x = int(w * margin_ratio)
#     margin_y = int(h * margin_ratio)
#     x1 = max(x - margin_x, 0)
#     y1 = max(y - margin_y, 0)
#     x2 = min(x + w + margin_x, mask.shape[1])
#     y2 = min(y + h + margin_y, mask.shape[0])
#     return [x1, y1, x2, y2]

# def get_adaptive_threshold(prob_np, percentile=98):
#     return np.percentile(prob_np.flatten(), percentile)

# def model_infer_real(model, frame, debug_save_dir=None, frame_idx=None, min_area=1500, suppress_bottom=False):
#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     transform = T.Compose([
#         T.Resize((512, 512)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

#     with torch.no_grad():
#         output = model(input_tensor)
#         if isinstance(output, tuple):
#             output = output[0]
#         if output.dim() == 4:
#             output = output[0]
#         output = output.squeeze()
#         prob = torch.sigmoid(output).cpu().numpy()

#     adaptive_thresh = get_adaptive_threshold(prob, percentile=98)
#     raw_mask = (prob > adaptive_thresh).astype(np.uint8)
#     mask_denoised = median_filter(raw_mask, size=3)
#     mask_denoised = cv2.GaussianBlur(mask_denoised, (5, 5), 0)

#     kernel = np.ones((5, 5), np.uint8)
#     mask_open = cv2.morphologyEx(mask_denoised, cv2.MORPH_OPEN, kernel)
#     mask_cleaned = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

#     if suppress_bottom:
#         h = mask_cleaned.shape[0]
#         mask_cleaned[int(h * 0.9):, :] = 0

#     num_pixels = np.sum(mask_cleaned)
#     final_mask = mask_cleaned * 255 if num_pixels >= min_area else np.zeros_like(mask_cleaned, dtype=np.uint8)

#     stats = {
#         "mean": float(prob.mean()), "max": float(prob.max()), "min": float(prob.min()),
#         "adaptive_thresh": float(adaptive_thresh), "mask_area": int(num_pixels)
#     }

#     if debug_save_dir and frame_idx is not None:
#         os.makedirs(debug_save_dir, exist_ok=True)
#         for thresh in [0.3, 0.5, 0.7, 0.9]:
#             temp_mask = (prob > thresh).astype(np.uint8) * 255
#             cv2.imwrite(os.path.join(debug_save_dir, f"{frame_idx:05d}_th{int(thresh * 100)}.png"), temp_mask)
#         cv2.imwrite(os.path.join(debug_save_dir, f"{frame_idx:05d}_adaptive_{int(adaptive_thresh * 100)}.png"), final_mask)
#         plt.hist(prob.ravel(), bins=50)
#         plt.title(f"Pixel Probabilities (frame {frame_idx})")
#         plt.savefig(os.path.join(debug_save_dir, f"{frame_idx:05d}_hist.png"))
#         plt.close()

#     return final_mask, stats

# def scale_keypoints(keypoints, original_shape, target_shape=(512, 512)):
#     orig_h, orig_w = original_shape
#     target_h, target_w = target_shape
#     scale_x = target_w / orig_w
#     scale_y = target_h / orig_h
#     return [[int(x * scale_x), int(y * scale_y)] for x, y in keypoints]

# def run_tsp_sam(input_path, output_path_base, config_path):
#     print("Loading configuration...")
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)

#     model_cfg = config["model"]
#     infer_cfg = config["inference"]
#     output_cfg = config["output"]
#     frame_stride = infer_cfg.get("frame_stride", 2)
#     min_area = infer_cfg.get("min_area", 1500)
#     suppress_bottom = infer_cfg.get("suppress_bottom_text", False)

#     class Opt: pass
#     opt = Opt()
#     opt.resume = model_cfg["checkpoint_path"]
#     opt.device = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
#     opt.gpu_ids = [0] if opt.device == "cuda" else []
#     opt.channel = model_cfg.get("channel", 32)

#     print(f"Initializing model on device: {opt.device}")
#     model = Network(opt)
#     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids).to(opt.device)
#     model.eval()

#     print(f"Loading weights from: {opt.resume}")
#     pretrained_weights = torch.load(opt.resume, map_location=opt.device)
#     model.module.feat_net.pvtv2_en.load_state_dict(pretrained_weights, strict=False)

#     sam_wrapper = MaskAnyoneSAMWrapper()
#     sam2_client = MySAM2Client()

#     video_name = Path(input_path).stem
#     output_path = Path(output_path_base) / video_name
#     if output_path.exists():
#         print(f"Cleaning previous run: {output_path}")
#         import shutil
#         shutil.rmtree(output_path)
#     output_path.mkdir(parents=True, exist_ok=True)

#     debug_csv_path = output_path / "debug_stats.csv"
#     pose_csv_path = output_path / "pose_keypoints.csv"
#     pose_json_path = output_path / "pose_keypoints.json"

#     with open(debug_csv_path, "w", newline="") as debug_file, open(pose_csv_path, "w", newline="") as pose_file:
#         csv_writer = csv.writer(debug_file)
#         pose_writer = csv.writer(pose_file)
#         csv_writer.writerow(["frame_idx", "mean", "max", "min", "adaptive_thresh", "mask_area", "sam_area", "pose_area", "fused_area"])
#         pose_writer.writerow(["frame_idx", "keypoints"])

#         pose_json = {}

#         frame_paths = sorted(Path(input_path).glob("*.jpg"))

#         # ---- Warm-up for dynamic pose threshold ----
#         warmup_areas = []
#         for fpath in frame_paths[:50]:
#             frame = cv2.imread(str(fpath))
#             frame_resized = resize_frame(frame, infer_cfg)
#             rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
#             keypoints = extract_pose_keypoints(rgb)
#             if keypoints:
#                 scaled_kpts = scale_keypoints(keypoints, original_shape=frame_resized.shape[:2])
#                 point_labels = [1] * len(scaled_kpts)
#                 pose_masks = sam2_client.predict_points(Image.fromarray(rgb), scaled_kpts, point_labels)
#                 if pose_masks and pose_masks[0] is not None:
#                     pose_mask = cv2.resize(pose_masks[0], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
#                     warmup_areas.append(np.sum(pose_mask > 0))
#         dynamic_pose_thresh = np.percentile(warmup_areas, 10) if warmup_areas else 300
#         print(f"[Dynamic Threshold] pose_area ≥ {int(dynamic_pose_thresh)}")

#         frame_idx = 0
#         with tqdm(total=len(frame_paths) // frame_stride) as pbar:
#             for i, frame_path in enumerate(frame_paths):
#                 if i % frame_stride != 0:
#                     continue

#                 frame = cv2.imread(str(frame_path))
#                 frame_resized = resize_frame(frame, infer_cfg)
#                 rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

#                 tsp_mask, stats = model_infer_real(model, frame_resized, frame_idx=frame_idx, debug_save_dir=output_path / "tsp_thresh", min_area=min_area, suppress_bottom=suppress_bottom)
#                 bbox = extract_bbox_from_mask(tsp_mask)

#                 if bbox:
#                     raw_sam_mask = sam_wrapper.segment_with_box(rgb, str(bbox))
#                     sam_mask = cv2.resize(raw_sam_mask, (tsp_mask.shape[1], tsp_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
#                 else:
#                     sam_mask = np.zeros_like(tsp_mask)

#                 keypoints = extract_pose_keypoints(rgb, draw_debug=True, frame_idx=frame_idx, debug_dir=str(output_path / "pose_debug"))
#                 if keypoints:
#                     scaled_kpts = scale_keypoints(keypoints, original_shape=frame_resized.shape[:2])
#                     point_labels = [1] * len(scaled_kpts)
#                     pose_masks = sam2_client.predict_points(Image.fromarray(rgb), scaled_kpts, point_labels)
#                     raw_pose_mask = pose_masks[0] if pose_masks and pose_masks[0] is not None else np.zeros_like(tsp_mask)
#                     pose_mask = cv2.resize(raw_pose_mask, (tsp_mask.shape[1], tsp_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
#                 else:
#                     pose_mask = np.zeros_like(tsp_mask)

#                 pose_writer.writerow([frame_idx, keypoints])
#                 pose_json[frame_idx] = keypoints
#                 pose_area = int(np.sum(pose_mask > 0))

#                 if pose_area < dynamic_pose_thresh or keypoints is None:
#                     print(f"[Frame {frame_idx}] Skipped: Unreliable or small pose mask")
#                     frame_idx += 1
#                     pbar.update(1)
#                     continue

#                 fused_mask = cv2.bitwise_or(tsp_mask, sam_mask)
#                 fused_mask = cv2.bitwise_or(fused_mask, pose_mask)
#                 fused_mask = post_process_fused_mask(fused_mask)

#                 fused_area = int(np.sum(fused_mask > 0))
#                 sam_area = int(np.sum(sam_mask > 0))

#                 csv_writer.writerow([
#                     frame_idx, stats['mean'], stats['max'], stats['min'],
#                     stats['adaptive_thresh'], stats['mask_area'], sam_area, pose_area, fused_area
#                 ])

#                 if fused_area > 0:
#                     resized_fused_mask = cv2.resize(fused_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
#                     save_mask_and_frame(frame, resized_fused_mask, str(output_path), frame_idx,
#                         save_overlay=True, overlay_alpha=0.5, save_frames=False, save_composite=True)

#                 frame_idx += 1
#                 pbar.update(1)

#         with open(pose_json_path, "w") as f_json:
#             json.dump(pose_json, f_json, indent=2)

#         print(f"\nFinished processing {frame_idx} frames.")
#         print(f"Output written to: {output_path}")

# if __name__ == "__main__":
#     if len(sys.argv) != 4:
#         print("Usage: python tsp_sam_runner.py <input_video> <output_base_dir> <config_path>")
#         sys.exit(1)

#     input_path = sys.argv[1]
#     output_base = sys.argv[2]
#     config_path = sys.argv[3]
#     run_tsp_sam(input_path, output_base, config_path)

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
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath("tsp_sam"))
sys.path.append(os.path.abspath("temporal"))

from tsp_sam.lib.pvtv2_afterTEM import Network
from utils import save_mask_and_frame, resize_frame
from maskanyone_sam_wrapper import MaskAnyoneSAMWrapper
from pose_extractor import extract_pose_keypoints
from my_sam2_client import MySAM2Client

def scale_keypoints(keypoints, original_shape, target_shape=(512, 512)):
    orig_h, orig_w = original_shape
    target_h, target_w = target_shape
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    return [[int(x * scale_x), int(y * scale_y)] for x, y in keypoints]

def get_dynamic_kernel_size(mask_shape, base_divisor=100, max_kernel=15):
    h, w = mask_shape[:2]
    avg_dim = (h + w) / 2
    k = max(3, int(avg_dim // base_divisor))
    k = k + (k % 2 == 0)  # ensure odd
    return min(k, max_kernel)

def post_process_fused_mask(fused_mask, min_area=100, kernel_size=None):
    if kernel_size is None:
        kernel_size = get_dynamic_kernel_size(fused_mask.shape)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(fused_mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(closed)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(filtered, [cnt], -1, 255, -1)
    return cv2.dilate(filtered, kernel, iterations=1)

def extract_bbox_from_mask(mask, margin_ratio=0.05):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    x1 = max(x - margin_x, 0)
    y1 = max(y - margin_y, 0)
    x2 = min(x + w + margin_x, mask.shape[1])
    y2 = min(y + h + margin_y, mask.shape[0])
    return [x1, y1, x2, y2]

def get_adaptive_threshold(prob_np, percentile=98):
    return np.percentile(prob_np.flatten(), percentile)

def model_infer_real(model, frame, debug_save_dir=None, frame_idx=None, min_area=1500, suppress_bottom=False):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        if output.dim() == 4:
            output = output[0]
        output = output.squeeze()
        prob = torch.sigmoid(output).cpu().numpy()

    adaptive_thresh = get_adaptive_threshold(prob)
    raw_mask = (prob > adaptive_thresh).astype(np.uint8)
    mask_denoised = median_filter(raw_mask, size=3)
    mask_denoised = cv2.GaussianBlur(mask_denoised, (5, 5), 0)

    kernel = np.ones((5, 5), np.uint8)
    mask_open = cv2.morphologyEx(mask_denoised, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    if suppress_bottom:
        h = mask_cleaned.shape[0]
        mask_cleaned[int(h * 0.9):, :] = 0

    num_pixels = np.sum(mask_cleaned)
    final_mask = mask_cleaned * 255 if num_pixels >= min_area else np.zeros_like(mask_cleaned, dtype=np.uint8)

    stats = {
        "mean": float(prob.mean()), "max": float(prob.max()), "min": float(prob.min()),
        "adaptive_thresh": float(adaptive_thresh), "mask_area": int(num_pixels)
    }

    if debug_save_dir and frame_idx is not None:
        os.makedirs(debug_save_dir, exist_ok=True)
        for thresh in [0.3, 0.5, 0.7, 0.9]:
            temp_mask = (prob > thresh).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(debug_save_dir, f"{frame_idx:05d}_th{int(thresh * 100)}.png"), temp_mask)
        cv2.imwrite(os.path.join(debug_save_dir, f"{frame_idx:05d}_adaptive_{int(adaptive_thresh * 100)}.png"), final_mask)
        plt.hist(prob.ravel(), bins=50)
        plt.title(f"Pixel Probabilities (frame {frame_idx})")
        plt.savefig(os.path.join(debug_save_dir, f"{frame_idx:05d}_hist.png"))
        plt.close()

    return final_mask, stats

def run_tsp_sam(input_path, output_path_base, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    infer_cfg = config["inference"]
    output_cfg = config["output"]

    class Opt: pass
    opt = Opt()
    opt.resume = model_cfg["checkpoint_path"]
    opt.device = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    opt.gpu_ids = [0] if opt.device == "cuda" else []
    opt.channel = model_cfg.get("channel", 32)

    model = Network(opt)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids).to(opt.device)
    model.eval()
    pretrained_weights = torch.load(opt.resume, map_location=opt.device)
    model.module.feat_net.pvtv2_en.load_state_dict(pretrained_weights, strict=False)

    sam_wrapper = MaskAnyoneSAMWrapper()
    sam2_client = MySAM2Client()

    video_name = Path(input_path).stem
    output_path = Path(output_path_base) / video_name
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    debug_csv_path = output_path / "debug_stats.csv"
    pose_json_path = output_path / "pose_keypoints.json"
    csv_writer = csv.writer(open(debug_csv_path, "w", newline=""))
    csv_writer.writerow(["frame_idx", "mean", "max", "min", "adaptive_thresh", "mask_area", "sam_area", "pose_area", "fused_area"])
    pose_json = {}

    frame_paths = sorted(Path(input_path).glob("*.jpg"))
    
    
    
    # warmup_areas = []
    # for f in frame_paths[:50]:
    #     frame = cv2.imread(str(f))
    #     frame_resized = resize_frame(frame, infer_cfg)
    #     rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    #     kpts = extract_pose_keypoints(rgb)
    #     if kpts:
    #         scaled = scale_keypoints(kpts, frame_resized.shape[:2])
    #         labels = [1] * len(scaled)
    #         masks = sam2_client.predict_points(Image.fromarray(rgb), scaled, labels)
    #         if masks and masks[0] is not None:
    #             pose_mask = cv2.resize(masks[0], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    #             warmup_areas.append(np.sum(pose_mask > 0))
    # pose_thresh = np.percentile(warmup_areas, 10) if warmup_areas else 300
    
    
    warmup_areas = []
    for f in frame_paths[:50]:
        frame = cv2.imread(str(f))
        frame_resized = resize_frame(frame, infer_cfg)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        kpts = extract_pose_keypoints(rgb)
        if kpts:
            scaled = scale_keypoints(kpts, frame_resized.shape[:2])
            labels = [1] * len(scaled)
            masks = sam2_client.predict_points(Image.fromarray(rgb), scaled, labels)
            if masks and masks[0] is not None:
                pose_mask = cv2.resize(masks[0], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                warmup_areas.append(np.sum(pose_mask > 0))
    
    dynamic_pose_thresh = np.percentile(warmup_areas, 10) if warmup_areas else 300
    print(f"[Dynamic Threshold] pose_area ≥ {int(dynamic_pose_thresh)}")


    frame_idx = 0
    with tqdm(total=len(frame_paths)) as pbar:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            frame_resized = resize_frame(frame, infer_cfg)
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            tsp_mask, stats = model_infer_real(model, frame_resized, debug_save_dir=output_path / "tsp_thresh", frame_idx=frame_idx)
            bbox = extract_bbox_from_mask(tsp_mask)
            sam_mask = sam_wrapper.segment_with_box(rgb, str(bbox)) if bbox else np.zeros_like(tsp_mask)
            sam_mask = cv2.resize(sam_mask, tsp_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)

            kpts = extract_pose_keypoints(rgb)
            if kpts:
                scaled = scale_keypoints(kpts, frame_resized.shape[:2])
                labels = [1] * len(scaled)
                masks = sam2_client.predict_points(Image.fromarray(rgb), scaled, labels)
                pose_mask = cv2.resize(masks[0], tsp_mask.shape[::-1], interpolation=cv2.INTER_NEAREST) if masks and masks[0] is not None else np.zeros_like(tsp_mask)
            else:
                pose_mask = np.zeros_like(tsp_mask)

            pose_area = int(np.sum(pose_mask > 0))
            if pose_area < dynamic_pose_thresh:
                pbar.update(1)
                frame_idx += 1
                continue

            fused_mask = cv2.bitwise_or(tsp_mask, sam_mask)
            fused_mask = cv2.bitwise_or(fused_mask, pose_mask)
            fused_mask = post_process_fused_mask(fused_mask)
            fused_area = int(np.sum(fused_mask > 0))

            csv_writer.writerow([frame_idx, stats['mean'], stats['max'], stats['min'], stats['adaptive_thresh'], stats['mask_area'], int(np.sum(sam_mask > 0)), pose_area, fused_area])
            if fused_area > 0:
                resized = cv2.resize(fused_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                save_mask_and_frame(frame, resized, str(output_path), frame_idx, save_overlay=True, overlay_alpha=0.5, save_frames=False, save_composite=True)

            pose_json[frame_idx] = kpts
            frame_idx += 1
            pbar.update(1)

    with open(pose_json_path, "w") as f_json:
        json.dump(pose_json, f_json, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python tsp_sam_runner.py <input_image_folder> <output_base_dir> <config_path>")
        sys.exit(1)
    run_tsp_sam(sys.argv[1], sys.argv[2], sys.argv[3])
