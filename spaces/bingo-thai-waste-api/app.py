from __future__ import annotations

import base64
import io
import os
import time
from threading import Lock
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_REPO_ID = os.environ.get(
    "MODEL_REPO_ID", "ChrisSujinlv/bingo-thai-four-bin-waste-vit"
)
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(4 * 1024 * 1024)))

THAI_WASTE_LABELS: dict[str, dict[str, Any]] = {
    "green": {
        "englishName": "Organic / food waste",
        "thaiName": "ขยะอินทรีย์",
        "chineseName": "有机垃圾 / 厨余垃圾",
        "binColor": "Green",
    },
    "yellow": {
        "englishName": "Recyclable waste",
        "thaiName": "ขยะรีไซเคิล",
        "chineseName": "可回收垃圾",
        "binColor": "Yellow",
    },
    "blue": {
        "englishName": "General waste",
        "thaiName": "ขยะทั่วไป",
        "chineseName": "一般垃圾 / 其他垃圾",
        "binColor": "Blue",
    },
    "red": {
        "englishName": "Household hazardous waste",
        "thaiName": "ขยะอันตราย",
        "chineseName": "有害垃圾",
        "binColor": "Red",
    },
}


class ClassifyRequest(BaseModel):
    image: str
    mimeType: str | None = None
    fileName: str | None = None
    source: str | None = None


app = FastAPI(title="BinGo Thai Waste API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_model = None
_processor = None
_loaded_at = None
_load_lock = Lock()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "modelRepoId": MODEL_REPO_ID,
        "loaded": _model is not None,
        "loadedAt": _loaded_at,
    }


@app.post("/classify")
def classify(request: ClassifyRequest) -> dict[str, Any]:
    image_bytes = decode_image(request.image)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processor, model = load_model()

    inputs = processor(images=image, return_tensors="pt")
    started = time.time()
    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).detach().cpu()[0]

    top_count = min(4, int(probabilities.shape[0]))
    top = torch.topk(probabilities, k=top_count)
    id_to_label = model.config.id2label
    best_index = int(top.indices[0].item())
    best_bin = str(id_to_label[best_index])
    confidence = round(float(top.values[0]) * 100, 2)

    return {
        "ok": True,
        "modelRepoId": MODEL_REPO_ID,
        "classifier": "huggingface-space-vit",
        "bin": best_bin,
        "confidence": confidence,
        "needsManualCheck": confidence < 70,
        "labelInfo": THAI_WASTE_LABELS.get(best_bin, {}),
        "top": [
            {
                "bin": str(id_to_label[int(index.item())]),
                "confidence": round(float(value) * 100, 2),
            }
            for value, index in zip(top.values, top.indices)
        ],
        "image": {
            "bytes": len(image_bytes),
            "mimeType": request.mimeType,
            "source": request.source,
        },
        "timingMs": round((time.time() - started) * 1000),
    }


def load_model():
    global _model, _processor, _loaded_at
    if _model is not None and _processor is not None:
        return _processor, _model

    with _load_lock:
        if _model is not None and _processor is not None:
            return _processor, _model

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        _processor = AutoImageProcessor.from_pretrained(MODEL_REPO_ID, token=token)
        _model = AutoModelForImageClassification.from_pretrained(
            MODEL_REPO_ID, token=token
        )
        _model.eval()
        _loaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return _processor, _model


def decode_image(value: str) -> bytes:
    if not value:
        raise HTTPException(status_code=400, detail="Missing image.")

    encoded = value.split(",", 1)[1] if "," in value else value
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Image must be base64.") from exc

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large.")

    return image_bytes
