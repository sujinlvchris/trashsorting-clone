#!/usr/bin/env python3
"""Run local inference with the trained four-bin Hugging Face model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model-dir", default="models/four-bin-waste-vit-v2-target-adapted")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        logits = model(pixel_values=inputs["pixel_values"].to(device)).logits
        probabilities = torch.softmax(logits, dim=-1).detach().cpu()[0]

    id_to_label = model.config.id2label
    top = torch.topk(probabilities, k=min(4, len(probabilities)))
    best_id = int(top.indices[0].item())
    result = {
        "bin": id_to_label[best_id],
        "confidence": round(float(top.values[0]) * 100, 2),
        "model": str(Path(args.model_dir)),
        "top": [
            {
                "bin": id_to_label[int(index.item())],
                "confidence": round(float(value) * 100, 2),
            }
            for value, index in zip(top.values, top.indices)
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "mps":
        return torch.device("mps")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


if __name__ == "__main__":
    main()
