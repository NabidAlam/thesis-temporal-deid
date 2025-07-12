import mediapipe as mp
import cv2
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_pose_keypoints(frame_rgb, draw_debug=False, frame_idx=None, debug_dir=None):
    keypoints = []
    h, w, _ = frame_rgb.shape

    # Use dynamic mode with lower threshold for better detection across frames
    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
        results = pose.process(frame_rgb)

        if not results.pose_landmarks:
            print(f"[DEBUG] Frame {frame_idx}: No pose detected.")
            return []

        for i, lm in enumerate(results.pose_landmarks.landmark):
            x, y = int(lm.x * w), int(lm.y * h)
            keypoints.append((x, y))

        print(f"[DEBUG] Frame {frame_idx}: {len(keypoints)} pose keypoints extracted.")

        # Optional visualization
        if draw_debug and debug_dir and frame_idx is not None:
            vis_frame = frame_rgb.copy()
            mp_drawing.draw_landmarks(vis_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            vis_path = f"{debug_dir}/frame_{frame_idx:05d}_pose_debug.jpg"
            cv2.imwrite(vis_path, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))

        return keypoints
