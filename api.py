from fastapi import FastAPI, UploadFile, File
import cv2, numpy as np
from inference_backend import infer_auto, infer_manual

app = FastAPI(title="PCB Defect Detection API")

@app.post("/detect/auto")
async def detect_auto(pcb_image: UploadFile = File(...)):
    img = cv2.imdecode(np.frombuffer(await pcb_image.read(), np.uint8), 1)
    tpl, detections, _ = infer_auto(img)
    return {
        "mode": "auto",
        "template_used": tpl,
        "defect_count": len(detections),
        "detections": detections
    }

@app.post("/detect/manual")
async def detect_manual(
    pcb_image: UploadFile = File(...),
    template_image: UploadFile = File(...)
):
    pcb = cv2.imdecode(np.frombuffer(await pcb_image.read(), np.uint8), 1)
    tpl = cv2.imdecode(np.frombuffer(await template_image.read(), np.uint8), 1)
    detections, _ = infer_manual(pcb, tpl)
    return {
        "mode": "manual",
        "defect_count": len(detections),
        "detections": detections
    }
