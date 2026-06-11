from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from torchvision import models, transforms


ROOT_DIR = Path(__file__).resolve().parent
MODEL_FILES = {
    "pth": ROOT_DIR / "tire_defect_resnet18.pth",
    "scripted": ROOT_DIR / "tire_defect_resnet18_scripted.pt",
    "traced": ROOT_DIR / "tire_defect_resnet18_traced.pt",
}
CLASS_NAMES = ["Defective", "Good"]
DEVICE = torch.device("cpu")

inference_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

app = FastAPI(title="TyreNet Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image data")
    model_name: str = Field("scripted", description="Model selection: scripted, traced, or pth")


def _decode_base64_image(image_base64: str) -> Image.Image:
    payload = image_base64.split(",", 1)[-1]
    try:
        raw_bytes = base64.b64decode(payload)
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc


@lru_cache(maxsize=3)
def load_model(model_name: str):
    if model_name not in MODEL_FILES:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'.")

    model_path = MODEL_FILES[model_name]
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path.name}")

    if model_name in {"scripted", "traced"}:
        model = torch.jit.load(str(model_path), map_location=DEVICE)
        model.eval()
        return model

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(image: Image.Image, model_name: str):
    model = load_model(model_name)
    input_tensor = inference_transforms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    class_name = CLASS_NAMES[predicted_idx.item()]
    probabilities_map = {
        class_label: float(probability.item())
        for class_label, probability in zip(CLASS_NAMES, probabilities)
    }

    return {
        "class_name": class_name,
        "confidence": float(confidence.item() * 100),
        "probabilities": probabilities_map,
        "model_name": model_name,
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    image = _decode_base64_image(request.image_base64)
    try:
        return predict_image(image, request.model_name)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc