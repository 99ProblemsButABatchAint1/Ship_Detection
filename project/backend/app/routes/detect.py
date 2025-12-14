from fastapi import APIRouter, UploadFile, File
from app.services.detector_stub import detect_stub

from app.deps import detector

router = APIRouter(prefix="/api", tags=["detect"])

@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read file bytes (later you will preprocess these bytes and run the model)
    image_bytes  = await file.read()

    # For now: return a fake detection
    # predictions = detect_stub()
    predictions = detector.detect(image_bytes)

    return {"predictions": predictions}

@router.post("/detect_debug")
async def detect_debug(file: UploadFile = File(...)):
    image_bytes = await file.read()
    preds = detector.detect(image_bytes)

    # if you included gate_prob in preds, this will show it
    gate_prob = preds[0].get("gate_prob") if preds else None
    return {"predictions": preds, "gate_prob": gate_prob}

