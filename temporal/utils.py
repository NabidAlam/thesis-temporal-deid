import cv2
import os

def resize_frame(frame, cfg):
    h, w = frame.shape[:2]
    short_side = cfg.get("resize_shorter_side", 720)
    long_side = cfg.get("max_longer_side", 1280)

    scale = short_side / min(h, w)
    if max(h, w) * scale > long_side:
        scale = long_side / max(h, w)

    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


# def save_mask_and_frame(frame, mask, output_path, frame_idx, save_overlay=True, overlay_alpha=0.5):
#     """
#     Saves the binary mask and optionally an overlay image.

#     Args:
#         frame (np.ndarray): Original RGB frame.
#         mask (np.ndarray): Binary mask (same HxW as frame).
#         output_path (str): Folder to save output.
#         frame_idx (int): Frame number.
#         save_overlay (bool): Whether to save overlay visualization.
#         overlay_alpha (float): Blend weight for overlay (0.0–1.0).
#     """
#     # Save binary mask
#     mask_path = os.path.join(output_path, f"mask_{frame_idx:04d}.png")
#     cv2.imwrite(mask_path, (mask * 255).astype("uint8"))

#     if save_overlay:
#         # Convert mask to 3-channel
#         colored_mask = cv2.cvtColor((mask * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
#         overlay = cv2.addWeighted(frame, 1 - overlay_alpha, colored_mask, overlay_alpha, 0)

#         overlay_path = os.path.join(output_path, f"overlay_{frame_idx:04d}.jpg")
#         cv2.imwrite(overlay_path, overlay)

# def save_mask_and_frame(frame, mask, output_path, frame_idx, save_overlay=True, overlay_alpha=0.5):
#     # Ensure mask is same size as frame
#     mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

#     # Save raw mask
#     mask_path = os.path.join(output_path, f"mask_{frame_idx:05d}.png")
#     cv2.imwrite(mask_path, mask)

#     # Optionally overlay and save
#     if save_overlay:
#         # Convert mask to 3-channel BGR (for overlay)
#         if len(mask.shape) == 2:
#             colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#         else:
#             colored_mask = mask

#         overlay = cv2.addWeighted(frame, 1 - overlay_alpha, colored_mask, overlay_alpha, 0)
#         overlay_path = os.path.join(output_path, f"overlay_{frame_idx:05d}.jpg")
#         cv2.imwrite(overlay_path, overlay)


# def save_mask_and_frame(frame, mask, output_dir, frame_idx, save_overlay=True, overlay_alpha=0.5, save_frames=False):
#     frame_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
#     mask_path = os.path.join(output_dir, f"mask_{frame_idx:05d}.png")
#     overlay_path = os.path.join(output_dir, f"overlay_{frame_idx:05d}.jpg")

#     # Save mask (grayscale PNG)
#     cv2.imwrite(mask_path, mask)

#     if save_frames:
#         cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Use JPEG

#     if save_overlay:
#         # Ensure mask has 3 channels
#         if len(mask.shape) == 2:
#             mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#         overlay = cv2.addWeighted(frame, 1 - overlay_alpha, mask_colored, overlay_alpha, 0)
#         cv2.imwrite(overlay_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])


# import os
# import cv2
# import numpy as np

# def save_mask_and_frame(frame, mask, output_dir, frame_idx,
#                         save_overlay=True, overlay_alpha=0.5,
#                         save_frames=False, save_composite=False):
#     """
#     Save mask, (optional) overlay, and (optional) raw frame.
#     If `save_composite` is True, a side-by-side view of original and overlay is saved.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # Ensure mask has 1 channel and matches frame size
#     if mask.shape != frame.shape[:2]:
#         raise ValueError(f"Mask and frame shape mismatch: {mask.shape} vs {frame.shape[:2]}")

#     mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

#     frame_filename = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
#     mask_filename = os.path.join(output_dir, f"{frame_idx:05d}_mask.png")
#     overlay_filename = os.path.join(output_dir, f"{frame_idx:05d}_overlay.jpg")
#     composite_filename = os.path.join(output_dir, f"{frame_idx:05d}_composite.jpg")

#     if save_frames:
#         cv2.imwrite(frame_filename, frame)

#     cv2.imwrite(mask_filename, mask)

#     if save_overlay or save_composite:
#         overlay = cv2.addWeighted(frame, 1 - overlay_alpha, mask_color, overlay_alpha, 0)

#         if save_overlay:
#             cv2.imwrite(overlay_filename, overlay)

#         if save_composite:
#             composite = np.hstack((frame, overlay))
#             cv2.imwrite(composite_filename, composite)


import os
import cv2
import numpy as np

def save_mask_and_frame(frame, mask, output_dir, frame_idx,
                        save_overlay=True, overlay_alpha=0.5,
                        save_frames=False, save_composite=False):
    """
    Save mask, (optional) overlay, and (optional) raw frame.
    If `save_composite` is True, a side-by-side view of original and overlay is saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure mask is single channel and same size as frame
    if mask.shape != frame.shape[:2]:
        raise ValueError(f"Mask and frame shape mismatch: {mask.shape} vs {frame.shape[:2]}")

    # Normalize mask if needed (ensure 0 or 255)
    if mask.max() == 1:
        mask = (mask * 255).astype(np.uint8)

    elif mask.max() > 1 and mask.max() < 255:
        print(f"[WARN] Mask has unusual max value ({mask.max()}). Forcing normalization.")
        mask = ((mask > 127).astype(np.uint8)) * 255

    # Convert to 3-channel color mask
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # File naming
    frame_filename = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
    mask_filename = os.path.join(output_dir, f"{frame_idx:05d}_mask.png")
    overlay_filename = os.path.join(output_dir, f"{frame_idx:05d}_overlay.jpg")
    composite_filename = os.path.join(output_dir, f"{frame_idx:05d}_composite.jpg")

    # Save original frame if needed
    if save_frames:
        cv2.imwrite(frame_filename, frame)

    # Save binary mask
    cv2.imwrite(mask_filename, mask)

    # Save overlay and composite view
    if save_overlay or save_composite:
        overlay = cv2.addWeighted(frame, 1 - overlay_alpha, mask_color, overlay_alpha, 0)

        if save_overlay:
            cv2.imwrite(overlay_filename, overlay)

        if save_composite:
            composite = np.hstack((frame, overlay))
            cv2.imwrite(composite_filename, composite)
