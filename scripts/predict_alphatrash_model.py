#!/usr/bin/env python3
"""Predict one image with the trained AlphaTrash material model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from train_alphatrash_material_model import THAI_BIN_LABELS, extract_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model-dir", default="models/alphatrash-material-v1")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(model_dir / "model.joblib")
    image_size = int(metadata["image_size"])
    image = Image.open(args.image).convert("RGB").resize((image_size, image_size))
    features = extract_features(image).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    labels = list(model.classes_)
    best_index = int(np.argmax(probabilities))
    material = labels[best_index]
    thai_bin = metadata["thai_bin_map"][material]
    result = {
        "material": material,
        "materialConfidence": round(float(probabilities[best_index]) * 100, 2),
        "thaiBin": thai_bin,
        "thaiBinLabel": THAI_BIN_LABELS[thai_bin],
        "probabilities": {
            label: round(float(probabilities[index]) * 100, 2)
            for index, label in enumerate(labels)
        },
        "hazardousRedSupport": metadata["hazardous_red_support"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
