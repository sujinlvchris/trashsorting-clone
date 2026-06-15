#!/usr/bin/env python3
"""
Fine-tune a public waste-classification model into the project's 4 Thailand bins.

Base model:
  watersplash/waste-classification

Training data:
  khoaliamle/Garbage_Classification_YOLO

The base model is already a waste image classifier. This script adapts its
classifier head to:
  green  = biological / organic
  yellow = cardboard, glass, metal, paper, plastic
  blue   = clothes, shoes, trash
  red    = battery
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import HfApi
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoImageProcessor, AutoModelForImageClassification


DATASET_REPO = "khoaliamle/Garbage_Classification_YOLO"
BASE_MODEL = "watersplash/waste-classification"
BIN_LABELS = ["green", "yellow", "blue", "red"]
SOURCE_TO_BIN = {
    "battery": "red",
    "biological": "green",
    "cardboard": "yellow",
    "glass": "yellow",
    "metal": "yellow",
    "paper": "yellow",
    "plastic": "yellow",
    "clothes": "blue",
    "shoes": "blue",
    "trash": "blue",
}


@dataclass
class Sample:
    split: str
    source_label: str
    bin_label: str
    repo_path: str
    local_path: Path


class WasteDataset(Dataset):
    def __init__(self, samples: list[Sample], processor: Any):
        self.samples = samples
        self.processor = processor
        self.label_to_id = {label: index for index, label in enumerate(BIN_LABELS)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample.local_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(self.label_to_id[sample.bin_label], dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    parser.add_argument("--data-dir", default="data-sources/hf-garbage-yolo")
    parser.add_argument("--model-dir", default="models/four-bin-waste-vit-v1")
    parser.add_argument("--max-train-per-source-class", type=int, default=320)
    parser.add_argument("--max-test-per-source-class", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--download-workers", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--eval-real-complex", default="test-results/real-complex-100/cases.json")
    args = parser.parse_args()

    started = time.time()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"Device: {device}", flush=True)

    manifest = build_manifest(args.dataset_repo, data_dir / "manifest.json", args.seed)
    train_samples = select_and_download(
        manifest,
        "train",
        args.max_train_per_source_class,
        data_dir,
        args.dataset_repo,
        args.seed,
        args.download_workers,
    )
    test_samples = select_and_download(
        manifest,
        "test",
        args.max_test_per_source_class,
        data_dir,
        args.dataset_repo,
        args.seed + 1,
        args.download_workers,
    )
    write_samples_csv(model_dir / "train_samples.csv", train_samples)
    write_samples_csv(model_dir / "test_samples.csv", test_samples)

    processor = AutoImageProcessor.from_pretrained(args.base_model)
    model = AutoModelForImageClassification.from_pretrained(
        args.base_model,
        num_labels=len(BIN_LABELS),
        id2label={index: label for index, label in enumerate(BIN_LABELS)},
        label2id={label: index for index, label in enumerate(BIN_LABELS)},
        ignore_mismatched_sizes=True,
    ).to(device)

    if args.freeze_backbone:
        freeze_backbone(model)
        print("Backbone frozen; training classifier head only.", flush=True)

    train_dataset = WasteDataset(train_samples, processor)
    test_dataset = WasteDataset(test_samples, processor)
    sampler = build_balanced_sampler(train_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(train_loader)))

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        metrics = evaluate_loader(model, test_loader, device)
        history.append({"epoch": epoch, "loss": round(loss, 5), **metrics})
        print(
            f"epoch {epoch}/{args.epochs} loss={loss:.4f} "
            f"test_acc={metrics['accuracy']}%",
            flush=True,
        )

    final_metrics = evaluate_loader(model, test_loader, device, include_rows=True, samples=test_samples)
    real_complex_metrics = None
    real_complex_path = Path(args.eval_real_complex)
    if real_complex_path.exists():
        real_complex_metrics = evaluate_real_complex(model, processor, real_complex_path, device)

    processor.save_pretrained(model_dir)
    model.save_pretrained(model_dir)

    metadata = {
        "name": "four-bin-waste-vit-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_model": args.base_model,
        "dataset_repo": args.dataset_repo,
        "labels": BIN_LABELS,
        "source_to_bin": SOURCE_TO_BIN,
        "device": str(device),
        "train_count": len(train_samples),
        "test_count": len(test_samples),
        "train_counts": count_samples(train_samples),
        "test_counts": count_samples(test_samples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "freeze_backbone": args.freeze_backbone,
        "elapsed_seconds": round(time.time() - started, 2),
        "final_test_metrics": final_metrics_without_rows(final_metrics),
        "real_complex_metrics": final_metrics_without_rows(real_complex_metrics) if real_complex_metrics else None,
    }
    write_json(model_dir / "metadata.json", metadata)
    write_json(model_dir / "history.json", history)
    write_json(model_dir / "test_metrics.json", final_metrics)
    if real_complex_metrics:
        write_json(model_dir / "real_complex_metrics.json", real_complex_metrics)
    write_report(model_dir / "report.zh.md", metadata, history, final_metrics, real_complex_metrics)

    print("\nModel trained:")
    print(f"- {model_dir}")
    print(f"- {model_dir / 'report.zh.md'}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "mps":
        return torch.device("mps")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def build_manifest(repo: str, path: Path, seed: int) -> dict[str, dict[str, list[str]]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset")
    manifest: dict[str, dict[str, list[str]]] = {"train": defaultdict(list), "test": defaultdict(list)}
    for file_name in files:
        parts = file_name.split("/")
        if len(parts) < 4:
            continue
        root, split, label = parts[:3]
        if root != "classify" or split not in manifest or label not in SOURCE_TO_BIN:
            continue
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        manifest[split][label].append(file_name)
    rng = random.Random(seed)
    for split in manifest:
        for label in manifest[split]:
            rng.shuffle(manifest[split][label])
    normal_manifest = {split: dict(labels) for split, labels in manifest.items()}
    write_json(path, normal_manifest)
    return normal_manifest


def select_and_download(
    manifest: dict[str, dict[str, list[str]]],
    split: str,
    limit: int,
    data_dir: Path,
    repo: str,
    seed: int,
    workers: int,
) -> list[Sample]:
    rng = random.Random(seed)
    samples = []
    selected = []
    for source_label, paths in manifest[split].items():
        choices = list(paths)
        rng.shuffle(choices)
        selected.extend((source_label, path) for path in choices[:limit])
    rng.shuffle(selected)
    print(f"Downloading {split}: {len(selected)} selected images with {workers} workers", flush=True)

    def fetch(item: tuple[str, str]) -> Sample:
        source_label, repo_path = item
        local_path = download_dataset_file(repo, repo_path, data_dir)
        return Sample(
            split=split,
            source_label=source_label,
            bin_label=SOURCE_TO_BIN[source_label],
            repo_path=repo_path,
            local_path=local_path,
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(fetch, item) for item in selected]
        for index, future in enumerate(as_completed(futures), start=1):
            samples.append(future.result())
            if index % 250 == 0 or index == len(selected):
                print(f"  {split}: {index}/{len(selected)}", flush=True)

    samples.sort(key=lambda sample: sample.repo_path)
    return samples


def download_dataset_file(repo: str, repo_path: str, data_dir: Path) -> Path:
    local_path = data_dir / "files" / repo_path
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_path = urllib.parse.quote(repo_path, safe="/")
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{encoded_path}"
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                local_path.write_bytes(response.read())
            if local_path.stat().st_size <= 0:
                raise RuntimeError("downloaded empty file")
            return local_path
        except Exception as exc:  # noqa: BLE001 - retry and surface exact failure.
            last_error = exc
            if local_path.exists():
                local_path.unlink(missing_ok=True)
            time.sleep(0.75 * (attempt + 1))
    raise RuntimeError(f"Failed to download {repo_path}: {last_error}")


def build_balanced_sampler(samples: list[Sample]) -> WeightedRandomSampler:
    counts = Counter(sample.bin_label for sample in samples)
    weights = [1.0 / counts[sample.bin_label] for sample in samples]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


def freeze_backbone(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("classifier."):
            param.requires_grad = False


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: Any, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for step, batch in enumerate(loader, start=1):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total += batch_size
        if step % 25 == 0 or step == len(loader):
            print(f"    step {step}/{len(loader)} loss={total_loss / max(1, total):.4f}", flush=True)
    return total_loss / max(1, total)


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    include_rows: bool = False,
    samples: list[Sample] | None = None,
) -> dict[str, Any]:
    model.eval()
    actual = []
    predicted = []
    confidences = []
    rows = []
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            logits = model(pixel_values=pixel_values).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu()
            preds = torch.argmax(probs, dim=-1)
            for offset, (label_id, pred_id) in enumerate(zip(labels.detach().cpu().tolist(), preds.tolist())):
                actual_label = BIN_LABELS[int(label_id)]
                pred_label = BIN_LABELS[int(pred_id)]
                confidence = float(probs[offset, pred_id]) * 100
                actual.append(actual_label)
                predicted.append(pred_label)
                confidences.append(confidence)
                if include_rows and samples:
                    sample = samples[cursor]
                    rows.append({
                        "split": sample.split,
                        "source_label": sample.source_label,
                        "actual": actual_label,
                        "predicted": pred_label,
                        "confidence": round(confidence, 4),
                        "repo_path": sample.repo_path,
                        "correct": actual_label == pred_label,
                    })
                cursor += 1
    return build_metrics(actual, predicted, confidences, rows if include_rows else None)


def evaluate_real_complex(model: torch.nn.Module, processor: Any, cases_path: Path, device: torch.device) -> dict[str, Any]:
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    actual = []
    predicted = []
    confidences = []
    rows = []
    model.eval()
    with torch.inference_mode():
        for case in data["cases"]:
            image = Image.open(case["image_path"]).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            probs = torch.softmax(model(pixel_values=pixel_values).logits, dim=-1).detach().cpu()[0]
            pred_id = int(torch.argmax(probs).item())
            pred_label = BIN_LABELS[pred_id]
            confidence = float(probs[pred_id]) * 100
            actual.append(case["expected_bin"])
            predicted.append(pred_label)
            confidences.append(confidence)
            rows.append({
                "case_id": case["case_id"],
                "source_dataset": case["source_dataset"],
                "source_category": case["source_category"],
                "actual": case["expected_bin"],
                "predicted": pred_label,
                "confidence": round(confidence, 4),
                "image_path": case["image_path"],
                "correct": case["expected_bin"] == pred_label,
            })
    return build_metrics(actual, predicted, confidences, rows)


def build_metrics(actual: list[str], predicted: list[str], confidences: list[float], rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    total = len(actual)
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    by_label = {}
    for label in BIN_LABELS:
        indices = [index for index, item in enumerate(actual) if item == label]
        label_correct = sum(1 for index in indices if predicted[index] == label)
        by_label[label] = {
            "total": len(indices),
            "correct": label_correct,
            "accuracy": round(label_correct / len(indices) * 100, 2) if indices else 0,
        }
    confusion = {label: {pred: 0 for pred in BIN_LABELS} for label in BIN_LABELS}
    for a, p in zip(actual, predicted):
        confusion[a][p] += 1
    result = {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / max(1, total) * 100, 2),
        "avg_confidence": round(sum(confidences) / max(1, len(confidences)), 2),
        "by_label": by_label,
        "confusion_matrix": confusion,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def count_samples(samples: list[Sample]) -> dict[str, dict[str, int]]:
    by_bin: dict[str, int] = Counter(sample.bin_label for sample in samples)
    by_source: dict[str, int] = Counter(sample.source_label for sample in samples)
    return {"by_bin": dict(by_bin), "by_source": dict(by_source)}


def final_metrics_without_rows(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics:
        return None
    return {key: value for key, value in metrics.items() if key != "rows"}


def write_samples_csv(path: Path, samples: list[Sample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "source_label", "bin_label", "repo_path", "local_path"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "split": sample.split,
                "source_label": sample.source_label,
                "bin_label": sample.bin_label,
                "repo_path": sample.repo_path,
                "local_path": str(sample.local_path),
            })


def write_report(
    path: Path,
    metadata: dict[str, Any],
    history: list[dict[str, Any]],
    test_metrics: dict[str, Any],
    real_complex_metrics: dict[str, Any] | None,
) -> None:
    lines = [
        "# 四桶垃圾分类模型训练报告",
        "",
        "## 结论",
        "",
        f"- 模型：`{metadata['name']}`",
        f"- 初始化权重：`{metadata['base_model']}`",
        f"- 训练数据：`{metadata['dataset_repo']}`",
        f"- 标签：`green / yellow / blue / red`",
        f"- 训练样本：{metadata['train_count']}",
        f"- 测试样本：{metadata['test_count']}",
        f"- Epochs：{metadata['epochs']}",
        f"- 测试集准确率：**{test_metrics['accuracy']}%**",
        f"- 测试集平均置信度：**{test_metrics['avg_confidence']}**",
    ]
    if real_complex_metrics:
        lines.extend([
            f"- 真实复杂 100 图准确率：**{real_complex_metrics['accuracy']}%**",
            f"- 真实复杂 100 图平均置信度：**{real_complex_metrics['avg_confidence']}**",
        ])
    lines.extend([
        "",
        "## 训练历史",
        "",
        "| Epoch | Loss | Test accuracy | Avg confidence |",
        "| ---: | ---: | ---: | ---: |",
    ])
    for row in history:
        lines.append(f"| {row['epoch']} | {row['loss']} | {row['accuracy']}% | {row['avg_confidence']} |")
    lines.extend(["", "## 测试集分桶", "", "| Bin | Total | Correct | Accuracy |", "| --- | ---: | ---: | ---: |"])
    for label in BIN_LABELS:
        row = test_metrics["by_label"][label]
        lines.append(f"| {label} | {row['total']} | {row['correct']} | {row['accuracy']}% |")
    if real_complex_metrics:
        lines.extend(["", "## 真实复杂 100 图分桶", "", "| Bin | Total | Correct | Accuracy |", "| --- | ---: | ---: | ---: |"])
        for label in BIN_LABELS:
            row = real_complex_metrics["by_label"][label]
            lines.append(f"| {label} | {row['total']} | {row['correct']} | {row['accuracy']}% |")
    lines.extend([
        "",
        "## 说明",
        "",
        "- 这是用公开模型 `watersplash/waste-classification` 作为初始化权重微调出来的四桶模型。",
        "- 数据集原始类别映射：battery -> red，biological -> green，cardboard/glass/metal/paper/plastic -> yellow，clothes/shoes/trash -> blue。",
        "- 这个模型是可本地推理的开源模型路线，不依赖 OpenAI API。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
