# import os
# import sys
# import cv2
# import yaml
# import torch
# import numpy as np
# from pathlib import Path
# from PIL import Image
# import torchvision.transforms as T
# from tqdm import tqdm  # For progress bar

# # repo paths
# sys.path.append(os.path.abspath("tsp_sam"))
# sys.path.append(os.path.abspath("temporal"))

# from tsp_sam.lib.pvtv2_afterTEM import Network
# from utils import save_mask_and_frame, resize_frame

# def run_tsp_sam(input_path, output_path, config_path):
#     # Loading here config
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)

#     model_cfg = config["model"]
#     infer_cfg = config["inference"]
#     output_cfg = config["output"]

#     # Simulate argparse-like config
#     class Opt: pass
#     opt = Opt()
#     opt.resume = model_cfg["checkpoint_path"]
#     opt.device = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
#     opt.gpu_ids = [0] if opt.device == "cuda" and torch.cuda.is_available() else []
#     opt.channel = model_cfg.get("channel", 32)

#     # Init model
#     model = Network(opt)
#     if opt.device == "cuda" and torch.cuda.is_available():
#         model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids).cuda()
#     else:
#         model = model.cpu()

#     # Load pretrained backbone weights
#     print(f"Load pretrained parameters from {opt.resume}")
#     pretrained_weights = torch.load(opt.resume, map_location=opt.device)
#     result = model.module.feat_net.pvtv2_en.load_state_dict(pretrained_weights, strict=False)
#     if isinstance(result, dict):
#         print("Missing keys:", result.get("missing_keys", []))
#         print("Unexpected keys:", result.get("unexpected_keys", []))

#     model.eval()
#     Path(output_path).mkdir(parents=True, exist_ok=True)

#     # Load video
#     cap = cv2.VideoCapture(input_path)
#     if not cap.isOpened():
#         raise IOError(f"Cannot open video: {input_path}")

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_idx = 0

#     with tqdm(total=total_frames // 2, desc="Processing Frames", unit="frame") as pbar:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             if frame_idx % 2 != 0:
#                 frame_idx += 1
#                 continue

#             frame = resize_frame(frame, infer_cfg)
#             mask = model_infer_real(model, frame)

#             if mask.sum() > 0:
#                 save_mask_and_frame(
#                     frame,
#                     mask,
#                     output_path,
#                     frame_idx,
#                     save_overlay=output_cfg.get("save_overlay", False),
#                     overlay_alpha=output_cfg.get("overlay_alpha", 0.5),
#                     save_frames=output_cfg.get("save_frames", False),
#                     save_masks=output_cfg.get("save_masks", True)
#                 )

#             frame_idx += 1
#             pbar.update(1)

#     cap.release()
#     print(f"[TSP-SAM] Finished processing {frame_idx} frames from: {input_path}")

# def model_infer_real(model, frame):
#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     transform = T.Compose([
#         T.Resize((224, 224)),  # its need to be adjusted based on training
#         T.ToTensor(),
#         T.Normalize(mean=[0.5]*3, std=[0.5]*3)
#     ])
#     input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

#     with torch.no_grad():
#         output = model(input_tensor)
#         if isinstance(output, tuple):
#             output = output[0]
#         if output.dim() == 4:
#             output = output[0]

#         output = output.squeeze().cpu().numpy()

#     mask = (output > 0.5).astype(np.uint8) * 255
#     return mask



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

# Add root path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Add custom module paths
sys.path.append(os.path.abspath("tsp_sam"))
sys.path.append(os.path.abspath("temporal"))

from tsp_sam.lib.pvtv2_afterTEM import Network
from utils import save_mask_and_frame, resize_frame

def run_tsp_sam(input_path, output_path_base, config_path):
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    infer_cfg = config["inference"]
    output_cfg = config["output"]

    class Opt: pass
    opt = Opt()
    opt.resume = model_cfg["checkpoint_path"]
    opt.device = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    opt.gpu_ids = [0] if opt.device == "cuda" and torch.cuda.is_available() else []
    opt.channel = model_cfg.get("channel", 32)

    print(f"Initializing model on device: {opt.device}")
    model = Network(opt)
    if opt.device == "cuda" and torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids).cuda()
    else:
        model = model.cpu()

    print(f"Loading weights from: {opt.resume}")
    pretrained_weights = torch.load(opt.resume, map_location=opt.device)
    result = model.module.feat_net.pvtv2_en.load_state_dict(pretrained_weights, strict=False)

    if isinstance(result, dict):
        print("Weights loaded. Check for missing/unexpected keys:")
        print("  - Missing keys:", result.get("missing_keys", []))
        print("  - Unexpected keys:", result.get("unexpected_keys", []))

    model.eval()

    # Auto-detect video name for folder structure
    video_name = Path(input_path).stem
    output_path = Path(output_path_base) / video_name
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory prepared at: {output_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    frame_idx = 0
    saved_frames = 0

    with tqdm(total=total_frames // 2, desc="Processing Frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 2 != 0:
                frame_idx += 1
                continue

            frame = resize_frame(frame, infer_cfg)
            mask = model_infer_real(model, frame)

            print(f"Frame {frame_idx} — Mask sum: {mask.sum()}")

            if mask.sum() > 0:
                save_mask_and_frame(
                    frame,
                    mask,
                    str(output_path),
                    frame_idx,
                    save_overlay=output_cfg.get("save_overlay", False),
                    overlay_alpha=output_cfg.get("overlay_alpha", 0.5),
                    save_frames=output_cfg.get("save_frames", False),
                    save_masks=output_cfg.get("save_masks", True)
                )
                print(f"✅ Saved mask for frame {frame_idx}")
                saved_frames += 1
            else:
                print(f"⚠️ Skipped frame {frame_idx}: Empty mask.")

            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"\n🌟 Finished processing {frame_idx} frames.")
    print(f"📀 Total frames saved: {saved_frames}")
    print(f"📄 Output written to: {output_path}")

def model_infer_real(model, frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        if output.dim() == 4:
            output = output[0]

        output = output.squeeze().cpu().numpy()

    mask = (output > 0.5).astype(np.uint8) * 255
    return mask

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python tsp_sam_runner.py <input_video> <output_base_dir> <config_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_base = sys.argv[2]
    config_path = sys.argv[3]

    run_tsp_sam(input_path, output_base, config_path)