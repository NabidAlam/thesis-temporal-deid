# import requests
# import numpy as np
# import cv2

# class MySAM2Client:
#     def __init__(self, server_url="http://localhost:8081"):
#         self.endpoint = f"{server_url}/segment"

#     def predict(self, image, boxes, mode="best"):
#         """
#         Sends a segmentation request to the SAM2 server.

#         Parameters:
#             image (PIL.Image): Input image (RGB)
#             boxes (list of tuple): List of bounding boxes [(x1, y1, x2, y2)]
#             mode (str): Segmentation mode — "best", "everything", etc.

#         Returns:
#             list of np.ndarray: List of binary masks (bool or uint8)
#         """
#         image_np = np.array(image.convert("RGB"))
#         _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

#         files = {"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
#         data = {"boxes": str(boxes), "mode": mode}

#         try:
#             response = requests.post(self.endpoint, files=files, data=data)
#             response.raise_for_status()

#             masks = np.load(response.content, allow_pickle=True)
#             return list(masks)
#         except Exception as e:
#             print(f"[MySAM2Client] Request failed: {e}")
#             return [None for _ in boxes]


# import requests
# import numpy as np
# import cv2
# import io
# from PIL import Image

# class MySAM2Client:
#     def __init__(self, server_url="http://localhost:8081"):
#         self.endpoint = f"{server_url}/segment"

#     def predict(self, image, boxes, mode="best"):
#         image_np = np.array(image.convert("RGB"))
#         _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

#         files = {"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
#         data = {"box": boxes, "mode": mode}

#         try:
#             response = requests.post(self.endpoint, files=files, data=data)
#             response.raise_for_status()

#             # New: handle image/png response as mask
#             image_bytes = io.BytesIO(response.content)
#             mask_image = Image.open(image_bytes).convert("L")  # grayscale
#             mask_np = np.array(mask_image)

#             return [mask_np]
#         except Exception as e:
#             print(f"[MySAM2Client] Request failed: {e}")
#             return [None]


import requests
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

class MySAM2Client:
    def __init__(self, server_url="http://localhost:8081"):
        self.endpoint = f"{server_url}/segment"

    def predict(self, image, boxes, mode="best"):
        image_np = np.array(image.convert("RGB"))
        _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        files = {"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
        data = {"box": str(boxes).replace(" ", ""), "mode": mode}

        try:
            response = requests.post(self.endpoint, files=files, data=data)
            response.raise_for_status()

            # Parse PNG mask image from binary
            mask_img = Image.open(BytesIO(response.content)).convert("L")
            mask_np = np.array(mask_img)
            mask_bin = (mask_np > 128).astype(np.uint8) * 255
            return [mask_bin]
        except Exception as e:
            print(f"[MySAM2Client] Request failed: {e}")
            return [None for _ in boxes]
