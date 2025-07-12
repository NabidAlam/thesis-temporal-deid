# # sam2_api_server.py
# from fastapi import FastAPI, UploadFile, Form
# from fastapi.responses import Response
# import numpy as np
# import cv2
# import io
# from PIL import Image
# import torch
# from segment_anything import sam_model_registry, SamPredictor

# app = FastAPI()

# # Device and model setup
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_TYPE = "vit_b"
# CHECKPOINT = r"D:\Thesis\thesis-temporal-deid\temporal\checkpoints\sam_vit_b_01ec64.pth"

# sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
# sam.to(DEVICE)
# predictor = SamPredictor(sam)

# @app.post("/segment")
# async def segment_image(image: UploadFile, box: str = Form(...), mode: str = Form("best")):
#     contents = await image.read()
#     img = Image.open(io.BytesIO(contents)).convert("RGB")
#     img_np = np.array(img)

#     try:
#         box_eval = eval(box)  # Example: [x0, y0, x1, y1]
#         if not isinstance(box_eval, list) or len(box_eval) != 4:
#             raise ValueError("Box must be a list of four coordinates.")
#     except Exception as e:
#         return {"error": f"Invalid box format: {e}"}

#     predictor.set_image(img_np)
#     input_box = torch.tensor([box_eval], dtype=torch.float, device=DEVICE)
#     transformed_box = predictor.transform.apply_boxes_torch(input_box, img_np.shape[:2])
    
#     # Fix: point_labels argument is now required
#     masks, _, _ = predictor.predict_torch(
#         point_coords=None,
#         point_labels=None,
#         boxes=transformed_box,
#         multimask_output=False
#     )

#     mask_np = masks[0][0].cpu().numpy().astype(np.uint8) * 255
#     _, encoded = cv2.imencode(".png", mask_np)
#     return Response(content=encoded.tobytes(), media_type="image/png")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("sam2_api_server:app", host="0.0.0.0", port=8081, reload=True)


# from fastapi import FastAPI, UploadFile, Form
# from fastapi.responses import Response
# import numpy as np
# import cv2
# import io
# from PIL import Image
# import torch
# from segment_anything import sam_model_registry, SamPredictor
# import json

# app = FastAPI()

# # Device and model setup
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_TYPE = "vit_b"
# CHECKPOINT = r"D:\Thesis\thesis-temporal-deid\temporal\checkpoints\sam_vit_b_01ec64.pth"

# sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
# sam.to(DEVICE)
# predictor = SamPredictor(sam)

# @app.post("/segment")
# async def segment_image(image: UploadFile, box: str = Form(...), mode: str = Form("best")):
#     contents = await image.read()
#     img = Image.open(io.BytesIO(contents)).convert("RGB")
#     img_np = np.array(img)

#     try:
#         box_eval = eval(box)
#         if not isinstance(box_eval, list) or len(box_eval) != 4:
#             raise ValueError("Box must be a list of four coordinates.")
#     except Exception as e:
#         return {"error": f"Invalid box format: {e}"}

#     predictor.set_image(img_np)
#     input_box = torch.tensor([box_eval], dtype=torch.float, device=DEVICE)
#     transformed_box = predictor.transform.apply_boxes_torch(input_box, img_np.shape[:2])

#     masks, _, _ = predictor.predict_torch(
#         point_coords=None,
#         point_labels=None,
#         boxes=transformed_box,
#         multimask_output=False
#     )

#     mask_np = masks[0][0].cpu().numpy().astype(np.uint8) * 255
#     _, encoded = cv2.imencode(".png", mask_np)
#     return Response(content=encoded.tobytes(), media_type="image/png")

# @app.post("/predict-points")
# async def predict_from_points(image: UploadFile, point_coords: str = Form(...), point_labels: str = Form(...)):
#     try:
#         contents = await image.read()
#         img = Image.open(io.BytesIO(contents)).convert("RGB")
#         img_np = np.array(img)

#         coords = np.array(json.loads(point_coords))
#         labels = np.array(json.loads(point_labels))

#         predictor.set_image(img_np)
#         masks, scores, logits = predictor.predict_torch(
#             point_coords=torch.tensor(coords, device=DEVICE).unsqueeze(0),
#             point_labels=torch.tensor(labels, device=DEVICE).unsqueeze(0),
#             boxes=None,
#             multimask_output=False
#         )

#         mask_np = masks[0][0].cpu().numpy().astype(np.uint8) * 255
#         _, encoded = cv2.imencode(".png", mask_np)
#         return Response(content=encoded.tobytes(), media_type="image/png")

#     except Exception as e:
#         print(f"[Server ERROR] predict-points: {e}")
#         return Response(content=f"Error: {e}", status_code=500)



# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("sam2_api_server:app", host="0.0.0.0", port=8081, reload=True)


from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import Response
import numpy as np
import cv2
import io
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamPredictor
import json

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(filename="debug.log", level=logging.DEBUG)
logging.debug("Some message")


app = FastAPI()

# Device and model setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "vit_b"
CHECKPOINT = r"D:\Thesis\thesis-temporal-deid\temporal\checkpoints\sam_vit_b_01ec64.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(DEVICE)
predictor = SamPredictor(sam)


@app.post("/segment")
async def segment_image(image: UploadFile, box: str = Form(...), mode: str = Form("best")):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(img)

    try:
        box_eval = eval(box)
        if not isinstance(box_eval, list) or len(box_eval) != 4:
            raise ValueError("Box must be a list of four coordinates.")
    except Exception as e:
        return {"error": f"Invalid box format: {e}"}

    predictor.set_image(img_np)
    input_box = torch.tensor([box_eval], dtype=torch.float, device=DEVICE)
    transformed_box = predictor.transform.apply_boxes_torch(input_box, img_np.shape[:2])

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_box,
        multimask_output=False
    )

    mask_np = masks[0][0].cpu().numpy().astype(np.uint8) * 255
    _, encoded = cv2.imencode(".png", mask_np)
    return Response(content=encoded.tobytes(), media_type="image/png")


@app.post("/predict-points")
async def predict_from_points(image: UploadFile, point_coords: str = Form(...), point_labels: str = Form(...)):
    try:
        print("[DEBUG] Received /predict-points request")
        print("[DEBUG] Raw point_coords:", point_coords)
        print("[DEBUG] Raw point_labels:", point_labels)

        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img)

        coords = np.array(json.loads(point_coords))
        labels = np.array(json.loads(point_labels))

        print("[DEBUG] Parsed coords:", coords)
        print("[DEBUG] Parsed labels:", labels)
        print("[DEBUG] Coords shape:", coords.shape, "| Labels shape:", labels.shape)

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("Each point coordinate must be [x, y] format")

        if coords.shape[0] != labels.shape[0]:
            raise ValueError("Mismatch: number of coords and labels")

        predictor.set_image(img_np)
        masks, scores, logits = predictor.predict_torch(
            point_coords=torch.tensor(coords, device=DEVICE).unsqueeze(0),
            point_labels=torch.tensor(labels, device=DEVICE).unsqueeze(0),
            boxes=None,
            multimask_output=False
        )

        mask_np = masks[0][0].cpu().numpy().astype(np.uint8) * 255
        _, encoded = cv2.imencode(".png", mask_np)
        return Response(content=encoded.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"[Server ERROR] predict-points: {e}")
        return Response(content=f"Error: {e}", status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sam2_api_server:app", host="0.0.0.0", port=8081, reload=True)
