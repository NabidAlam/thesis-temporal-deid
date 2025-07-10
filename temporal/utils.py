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


def save_mask_and_frame(frame, mask, output_dir, frame_idx, save_overlay=True, overlay_alpha=0.5, save_frames=False):
    frame_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
    mask_path = os.path.join(output_dir, f"mask_{frame_idx:05d}.png")
    overlay_path = os.path.join(output_dir, f"overlay_{frame_idx:05d}.jpg")

    # Save mask (grayscale PNG)
    cv2.imwrite(mask_path, mask)

    if save_frames:
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Use JPEG

    if save_overlay:
        # Ensure mask has 3 channels
        if len(mask.shape) == 2:
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(frame, 1 - overlay_alpha, mask_colored, overlay_alpha, 0)
        cv2.imwrite(overlay_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
