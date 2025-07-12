# import requests
# import numpy as np
# import cv2
# from io import BytesIO
# from PIL import Image

# class MySAM2Client:
#     def __init__(self, server_url="http://localhost:8081"):
#         self.endpoint = f"{server_url}/segment"

#     def predict(self, image, boxes, mode="best"):
#         image_np = np.array(image.convert("RGB"))
#         _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
#         files = {"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
#         data = {"box": str(boxes).replace(" ", ""), "mode": mode}

#         try:
#             response = requests.post(self.endpoint, files=files, data=data)
#             response.raise_for_status()

#             # Parse PNG mask image from binary
#             mask_img = Image.open(BytesIO(response.content)).convert("L")
#             mask_np = np.array(mask_img)
#             mask_bin = (mask_np > 128).astype(np.uint8) * 255
#             return [mask_bin]
#         except Exception as e:
#             print(f"[MySAM2Client] Request failed: {e}")
#             return [None for _ in boxes]


import requests
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import json

class MySAM2Client:
    def __init__(self, server_url="http://localhost:8081"):
        self.box_endpoint = f"{server_url}/segment"
        self.point_endpoint = f"{server_url}/predict-points"

    def predict(self, image, boxes, mode="best"):
        """
        Segment using bounding boxes.
        """
        image_np = np.array(image.convert("RGB"))
        _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        files = {"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
        data = {"box": str(boxes).replace(" ", ""), "mode": mode}

        try:
            response = requests.post(self.box_endpoint, files=files, data=data)
            response.raise_for_status()
            mask_img = Image.open(BytesIO(response.content)).convert("L")
            mask_np = np.array(mask_img)
            mask_bin = (mask_np > 128).astype(np.uint8) * 255
            return [mask_bin]
        except Exception as e:
            print(f"[MySAM2Client] Request failed (box): {e}")
            return [None for _ in boxes]

    def predict_points(self, image, point_coords, point_labels):
        """
        Segment using keypoint prompts (pose-based).
        """
        if not point_coords or not point_labels or len(point_coords) != len(point_labels):
            raise ValueError("Invalid or empty point coordinates or labels")

        image_np = np.array(image.convert("RGB"))
        _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        files = {"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
        data = {
            "point_coords": json.dumps(point_coords),
            "point_labels": json.dumps(point_labels),
        }

        try:
            response = requests.post(self.point_endpoint, files=files, data=data)
            response.raise_for_status()

            # Assume the server returns a single grayscale PNG mask
            mask_img = Image.open(BytesIO(response.content)).convert("L")
            mask_np = np.array(mask_img)
            mask_bin = (mask_np > 128).astype(np.uint8) * 255
            return [mask_bin]
        except Exception as e:
            print(f"[MySAM2Client] Request failed (points): {e}")
            return [None]
