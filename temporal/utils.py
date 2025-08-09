import cv2
import os
import numpy as np

def resize_frame(frame, cfg):
    h, w = frame.shape[:2]
    short_side = cfg.get("resize_shorter_side", 720)
    long_side = cfg.get("max_longer_side", 1280)

    scale = short_side / min(h, w)
    if max(h, w) * scale > long_side:
        scale = long_side / max(h, w)

    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


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



# this is for baseline tsp sams
def post_process(mask, min_area=1000, kernel_size=5):
    """
    Perform morphological operations and small object removal.
    """
    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    final_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            final_mask[labels == i] = 1
    return final_mask
