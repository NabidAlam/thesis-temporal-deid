# sam2_api_server.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import Response
import numpy as np
import cv2
import io
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamPredictor

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
        box_eval = eval(box)  # Example: [x0, y0, x1, y1]
        if not isinstance(box_eval, list) or len(box_eval) != 4:
            raise ValueError("Box must be a list of four coordinates.")
    except Exception as e:
        return {"error": f"Invalid box format: {e}"}

    predictor.set_image(img_np)
    input_box = torch.tensor([box_eval], dtype=torch.float, device=DEVICE)
    transformed_box = predictor.transform.apply_boxes_torch(input_box, img_np.shape[:2])
    
    # Fix: point_labels argument is now required
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_box,
        multimask_output=False
    )

    mask_np = masks[0][0].cpu().numpy().astype(np.uint8) * 255
    _, encoded = cv2.imencode(".png", mask_np)
    return Response(content=encoded.tobytes(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sam2_api_server:app", host="0.0.0.0", port=8081, reload=True)
