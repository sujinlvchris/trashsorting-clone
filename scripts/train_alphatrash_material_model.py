#!/usr/bin/env python3
"""
Train a lightweight local garbage-classification model from AlphaTrash.

Dataset:
  https://github.com/Patipol-BKK/alphatrash-dataset

This first model predicts the AlphaTrash material classes:
general, metal, organic, paper, plastic.

The report also maps those material classes to the current Thailand bins:
organic -> green
metal/paper/plastic -> yellow
general -> blue

AlphaTrash does not contain hazardous waste, so red-bin recognition requires a
separate hazardous dataset in a later model version.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO = "Patipol-BKK/alphatrash-dataset"
API_ROOT = f"https://api.github.com/repos/{REPO}/contents/trash_dataset"
SOURCE_URL = f"https://github.com/{REPO}"
LABELS = ["general", "metal", "organic", "paper", "plastic"]
THAI_BIN_MAP = {
    "general": "blue",
    "metal": "yellow",
    "organic": "green",
    "paper": "yellow",
    "plastic": "yellow",
}
THAI_BIN_LABELS = {
    "green": "Food / Organic waste",
    "yellow": "Recyclable waste",
    "blue": "General waste",
    "red": "Hazardous waste",
}


@dataclass
class Sample:
    split: str
    label: str
    name: str
    url: str
    image_path: Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data-sources/alphatrash")
    parser.add_argument("--model-dir", default="models/alphatrash-material-v1")
    parser.add_argument("--max-train-per-class", type=int, default=500)
    parser.add_argument("--max-val-per-class", type=int, default=150)
    parser.add_argument("--max-test-per-class", type=int, default=150)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    image_dir = data_dir / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    rng = random.Random(args.seed)
    manifest = load_or_create_manifest(data_dir / "manifest.json", rng)
    samples = select_samples(manifest, image_dir, args, rng)
    download_and_resize_samples(samples, args.concurrency, args.image_size)
    write_dataset_csv(model_dir / "dataset.csv", samples)

    train = [sample for sample in samples if sample.split == "train"]
    val = [sample for sample in samples if sample.split == "val"]
    test = [sample for sample in samples if sample.split == "test"]

    print(f"Feature extraction: train={len(train)} val={len(val)} test={len(test)}", flush=True)
    x_train, y_train = load_features(train, args.image_size)
    x_val, y_val = load_features(val, args.image_size)
    x_test, y_test = load_features(test, args.image_size)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=0.00008,
                    max_iter=2500,
                    tol=1e-4,
                    class_weight="balanced",
                    random_state=args.seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    print("Training model...", flush=True)
    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    val_accuracy = accuracy_score(y_val, val_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    thai_true = [THAI_BIN_MAP[label] for label in y_test]
    thai_pred = [THAI_BIN_MAP[label] for label in test_pred]
    thai_accuracy = accuracy_score(thai_true, thai_pred)

    metrics = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": SOURCE_URL,
        "model_type": "sklearn SGDClassifier + handcrafted image features",
        "labels": LABELS,
        "thai_bin_map": THAI_BIN_MAP,
        "hazardous_red_support": False,
        "image_size": args.image_size,
        "seed": args.seed,
        "counts": count_by_split_label(samples),
        "feature_dim": int(x_train.shape[1]),
        "val_accuracy": round(float(val_accuracy) * 100, 2),
        "test_accuracy": round(float(test_accuracy) * 100, 2),
        "test_thai_bin_accuracy": round(float(thai_accuracy) * 100, 2),
        "elapsed_seconds": round(time.time() - started, 2),
        "classification_report": classification_report(y_test, test_pred, labels=LABELS, output_dict=True, zero_division=0),
        "thai_classification_report": classification_report(thai_true, thai_pred, labels=["green", "yellow", "blue"], output_dict=True, zero_division=0),
        "confusion_matrix": confusion_to_dict(y_test, test_pred, LABELS),
        "thai_confusion_matrix": confusion_to_dict(thai_true, thai_pred, ["green", "yellow", "blue"]),
    }

    joblib.dump(model, model_dir / "model.joblib")
    write_json(model_dir / "metadata.json", build_metadata(args, metrics))
    write_json(model_dir / "metrics.json", metrics)
    write_confusion_csv(model_dir / "confusion_matrix.csv", metrics["confusion_matrix"], LABELS)
    write_confusion_csv(model_dir / "thai_bin_confusion_matrix.csv", metrics["thai_confusion_matrix"], ["green", "yellow", "blue"])
    write_report(model_dir / "report.zh.md", metrics)

    print("\nTraining complete:")
    print(f"- {model_dir / 'model.joblib'}")
    print(f"- {model_dir / 'metadata.json'}")
    print(f"- {model_dir / 'metrics.json'}")
    print(f"- {model_dir / 'report.zh.md'}")


def load_or_create_manifest(path: Path, rng: random.Random) -> dict[str, dict[str, list[dict[str, str]]]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    manifest: dict[str, dict[str, list[dict[str, str]]]] = {}
    for split in ["train", "val", "test"]:
        manifest[split] = {}
        for label in LABELS:
            url = f"{API_ROOT}/{split}/{urllib.parse.quote(label)}?ref=main"
            print(f"Listing {split}/{label}", flush=True)
            rows = request_json(url)
            items = [
                {
                    "name": row["name"],
                    "download_url": row["download_url"],
                    "size": row.get("size", 0),
                }
                for row in rows
                if row.get("download_url") and row["name"].lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
            rng.shuffle(items)
            manifest[split][label] = items
    write_json(path, manifest)
    return manifest


def request_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def select_samples(
    manifest: dict[str, dict[str, list[dict[str, str]]]],
    image_dir: Path,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[Sample]:
    limits = {
        "train": args.max_train_per_class,
        "val": args.max_val_per_class,
        "test": args.max_test_per_class,
    }
    samples: list[Sample] = []
    for split in ["train", "val", "test"]:
        for label in LABELS:
            rows = list(manifest[split][label])
            rng.shuffle(rows)
            for row in rows[: limits[split]]:
                image_path = image_dir / split / label / safe_file_name(row["name"])
                samples.append(
                    Sample(
                        split=split,
                        label=label,
                        name=row["name"],
                        url=row["download_url"],
                        image_path=image_path,
                    )
                )
    return samples


def download_and_resize_samples(samples: list[Sample], concurrency: int, image_size: int) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    missing = [sample for sample in samples if not sample.image_path.exists()]
    print(f"Downloading/resizing {len(missing)} missing images ({len(samples)} selected).", flush=True)
    if not missing:
        return

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [executor.submit(download_and_resize_one, sample, image_size) for sample in missing]
        for index, future in enumerate(as_completed(futures), start=1):
            future.result()
            if index % 50 == 0 or index == len(missing):
                print(f"  {index}/{len(missing)} images ready", flush=True)


def download_and_resize_one(sample: Sample, image_size: int) -> None:
    request = urllib.request.Request(sample.url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()
    sample.image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(PathBytes(data))
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (246, 248, 250))
    offset = ((image_size - image.width) // 2, (image_size - image.height) // 2)
    canvas.paste(image, offset)
    canvas.save(sample.image_path, format="JPEG", quality=88, optimize=True)


class PathBytes:
    def __init__(self, data: bytes):
        self._data = data
        self._offset = 0

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = len(self._data) - self._offset
        chunk = self._data[self._offset : self._offset + n]
        self._offset += len(chunk)
        return chunk

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._offset = offset
        elif whence == 1:
            self._offset += offset
        elif whence == 2:
            self._offset = len(self._data) + offset
        return self._offset

    def tell(self) -> int:
        return self._offset


def load_features(samples: list[Sample], image_size: int) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for sample in samples:
        image = Image.open(sample.image_path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        features.append(extract_features(image))
        labels.append(sample.label)
    return np.vstack(features).astype(np.float32), np.array(labels)


def extract_features(image: Image.Image) -> np.ndarray:
    small = image.resize((32, 32), Image.Resampling.BILINEAR)
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    raw = rgb.reshape(-1)

    full = np.asarray(image, dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    hist_parts = []
    for channel in range(3):
        hist_parts.append(np.histogram(full[:, :, channel], bins=24, range=(0, 1), density=True)[0])
        hist_parts.append(np.histogram(hsv[:, :, channel], bins=24, range=(0, 1), density=True)[0])

    gray = np.asarray(image.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
    hog = []
    grid = 4
    bins = 9
    cell = gray.shape[0] // grid
    for row in range(grid):
        for col in range(grid):
            mag_cell = magnitude[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell]
            angle_cell = angle[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell]
            hist, _ = np.histogram(angle_cell, bins=bins, range=(0, 1), weights=mag_cell)
            hist = hist / (np.linalg.norm(hist) + 1e-6)
            hog.append(hist)

    moments = []
    for channel in range(3):
        values = full[:, :, channel].reshape(-1)
        moments.extend([values.mean(), values.std(), np.percentile(values, 10), np.percentile(values, 50), np.percentile(values, 90)])

    edge_density = np.array([float((magnitude > 0.12).mean()), float(magnitude.mean()), float(magnitude.std())])
    return np.concatenate([raw, *hist_parts, *hog, np.array(moments), edge_density]).astype(np.float32)


def count_by_split_label(samples: list[Sample]) -> dict[str, dict[str, int]]:
    counts = {split: {label: 0 for label in LABELS} for split in ["train", "val", "test"]}
    for sample in samples:
        counts[sample.split][sample.label] += 1
    return counts


def confusion_to_dict(y_true: list[str] | np.ndarray, y_pred: list[str] | np.ndarray, labels: list[str]) -> dict[str, dict[str, int]]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        label: {pred_label: int(matrix[row_index, col_index]) for col_index, pred_label in enumerate(labels)}
        for row_index, label in enumerate(labels)
    }


def build_metadata(args: argparse.Namespace, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "alphatrash-material-v1",
        "created_at": metrics["generated_at"],
        "source": SOURCE_URL,
        "model_file": "model.joblib",
        "framework": "scikit-learn",
        "model_type": metrics["model_type"],
        "labels": LABELS,
        "thai_bin_map": THAI_BIN_MAP,
        "thai_bin_labels": THAI_BIN_LABELS,
        "hazardous_red_support": False,
        "image_size": args.image_size,
        "feature_dim": metrics["feature_dim"],
        "metrics": {
            "val_accuracy": metrics["val_accuracy"],
            "test_accuracy": metrics["test_accuracy"],
            "test_thai_bin_accuracy": metrics["test_thai_bin_accuracy"],
        },
    }


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# AlphaTrash 自训练垃圾分类模型报告",
        "",
        "## 结论",
        "",
        f"- 模型：`alphatrash-material-v1`",
        f"- 数据源：{SOURCE_URL}",
        "- 数据说明：泰国 AlphaTrash 项目真实垃圾图片，官方 train/val/test 划分。",
        "- 训练目标：5 类材料分类 `general / metal / organic / paper / plastic`。",
        "- 泰国桶映射：organic -> green，metal/paper/plastic -> yellow，general -> blue。",
        "- 注意：AlphaTrash 没有 hazardous/red 类，所以此模型暂不支持有害垃圾红桶识别。",
        f"- 特征维度：{metrics['feature_dim']}",
        f"- 验证集准确率：**{metrics['val_accuracy']}%**",
        f"- 测试集材料分类准确率：**{metrics['test_accuracy']}%**",
        f"- 测试集映射到泰国桶准确率：**{metrics['test_thai_bin_accuracy']}%**",
        f"- 训练耗时：{metrics['elapsed_seconds']} 秒",
        "",
        "## 数据量",
        "",
        "| Split | general | metal | organic | paper | plastic |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, row in metrics["counts"].items():
        lines.append(f"| {split} | {row['general']} | {row['metal']} | {row['organic']} | {row['paper']} | {row['plastic']} |")

    lines.extend([
        "",
        "## 测试集分类报告",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for label in LABELS:
        row = metrics["classification_report"][label]
        lines.append(f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1-score']:.3f} | {int(row['support'])} |")

    lines.extend([
        "",
        "## 泰国桶映射报告",
        "",
        "| Bin | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for label in ["green", "yellow", "blue"]:
        row = metrics["thai_classification_report"][label]
        lines.append(f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1-score']:.3f} | {int(row['support'])} |")

    lines.extend([
        "",
        "## 产物",
        "",
        "- `model.joblib`：训练好的 scikit-learn 模型。",
        "- `metadata.json`：模型标签、桶映射、指标和输入尺寸。",
        "- `metrics.json`：完整指标。",
        "- `dataset.csv`：训练/验证/测试样本清单。",
        "- `confusion_matrix.csv`：5 类材料混淆矩阵。",
        "- `thai_bin_confusion_matrix.csv`：泰国桶映射混淆矩阵。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_csv(path: Path, samples: list[Sample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "label", "name", "image_path", "url"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "split": sample.split,
                "label": sample.label,
                "name": sample.name,
                "image_path": str(sample.image_path),
                "url": sample.url,
            })


def write_confusion_csv(path: Path, matrix: dict[str, dict[str, int]], labels: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *labels])
        for label in labels:
            writer.writerow([label, *[matrix[label][pred] for pred in labels]])


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_file_name(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    suffix = Path(value).suffix.lower() or ".jpg"
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(value).stem)[:80]
    return f"{stem}_{digest}{suffix}"


if __name__ == "__main__":
    main()
