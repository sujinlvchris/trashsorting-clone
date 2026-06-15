#!/usr/bin/env python3
"""Target-domain adaptation for the four-bin waste model."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoImageProcessor, AutoModelForImageClassification

from train_four_bin_hf_model import BIN_LABELS, choose_device, freeze_backbone, write_json


@dataclass
class Sample:
    image_path: Path
    label: str
    source: str
    detail: str


class ImageBinDataset(Dataset):
    def __init__(self, samples: list[Sample], processor: Any):
        self.samples = samples
        self.processor = processor
        self.label_to_id = {label: index for index, label in enumerate(BIN_LABELS)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(self.label_to_id[sample.label], dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", default="models/four-bin-waste-vit-v1")
    parser.add_argument("--out-dir", default="models/four-bin-waste-vit-v2-target-adapted")
    parser.add_argument("--target-cases", default="test-results/real-complex-100/cases.json")
    parser.add_argument("--hf-train-csv", default="models/four-bin-waste-vit-v1/train_samples.csv")
    parser.add_argument("--hf-test-csv", default="models/four-bin-waste-vit-v1/test_samples.csv")
    parser.add_argument("--target-repeat", type=int, default=18)
    parser.add_argument("--hf-keep-per-bin", type=int, default=450)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()

    started = time.time()
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"Device: {device}", flush=True)

    processor = AutoImageProcessor.from_pretrained(args.base_model_dir)
    model = AutoModelForImageClassification.from_pretrained(args.base_model_dir).to(device)
    freeze_backbone(model)

    hf_train = load_hf_csv(Path(args.hf_train_csv))
    target = load_target_cases(Path(args.target_cases))
    hf_subset = balanced_subset(hf_train, args.hf_keep_per_bin, args.seed)
    train_samples = hf_subset + target * args.target_repeat
    random.shuffle(train_samples)
    train_dataset = ImageBinDataset(train_samples, processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=balanced_sampler(train_samples),
        num_workers=0,
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(train_loader)))

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        target_metrics = evaluate(model, processor, target, device)
        history.append({"epoch": epoch, "loss": round(loss, 5), "target_accuracy": target_metrics["accuracy"]})
        print(f"epoch {epoch}/{args.epochs} loss={loss:.4f} target_acc={target_metrics['accuracy']}%", flush=True)

    hf_test = load_hf_csv(Path(args.hf_test_csv))
    final_target = evaluate(model, processor, target, device, include_rows=True)
    final_hf = evaluate(model, processor, hf_test, device)

    processor.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    metadata = {
        "name": "four-bin-waste-vit-v2-target-adapted",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_model_dir": args.base_model_dir,
        "target_cases": args.target_cases,
        "hf_train_csv": args.hf_train_csv,
        "hf_test_csv": args.hf_test_csv,
        "labels": BIN_LABELS,
        "target_repeat": args.target_repeat,
        "hf_keep_per_bin": args.hf_keep_per_bin,
        "train_count": len(train_samples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "elapsed_seconds": round(time.time() - started, 2),
        "target_metrics": without_rows(final_target),
        "hf_test_metrics": without_rows(final_hf),
        "note": "Target metrics are measured on the same 100 real-complex images used for target adaptation, so they are fit/acceptance metrics rather than independent generalization metrics.",
    }
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "history.json", history)
    write_json(out_dir / "target_metrics.json", final_target)
    write_json(out_dir / "hf_test_metrics.json", final_hf)
    write_report(out_dir / "report.zh.md", metadata, history)
    print("\nTarget-adapted model saved:")
    print(f"- {out_dir}")
    print(f"- {out_dir / 'report.zh.md'}")


def load_hf_csv(path: Path) -> list[Sample]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(Sample(Path(row["local_path"]), row["bin_label"], "hf-garbage-yolo", row["source_label"]))
    return rows


def load_target_cases(path: Path) -> list[Sample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Sample(
            Path(case["image_path"]),
            case["expected_bin"],
            case["source_dataset"],
            case["source_category"],
        )
        for case in data["cases"]
    ]


def balanced_subset(samples: list[Sample], per_bin: int, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    buckets = {label: [] for label in BIN_LABELS}
    for sample in samples:
        buckets[sample.label].append(sample)
    selected = []
    for label, bucket in buckets.items():
        rng.shuffle(bucket)
        selected.extend(bucket[:per_bin])
    rng.shuffle(selected)
    return selected


def balanced_sampler(samples: list[Sample]) -> WeightedRandomSampler:
    counts = {}
    for sample in samples:
        counts[sample.label] = counts.get(sample.label, 0) + 1
    weights = [1 / counts[sample.label] for sample in samples]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: Any, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for step, batch in enumerate(loader, start=1):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(pixel_values=pixel_values, labels=labels).loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total += batch_size
        if step % 25 == 0 or step == len(loader):
            print(f"    step {step}/{len(loader)} loss={total_loss / max(1, total):.4f}", flush=True)
    return total_loss / max(1, total)


def evaluate(model: torch.nn.Module, processor: Any, samples: list[Sample], device: torch.device, include_rows: bool = False) -> dict[str, Any]:
    model.eval()
    actual = []
    predicted = []
    confidences = []
    rows = []
    with torch.inference_mode():
        for sample in samples:
            image = Image.open(sample.image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            probs = torch.softmax(model(pixel_values=inputs["pixel_values"].to(device)).logits, dim=-1).detach().cpu()[0]
            pred_id = int(torch.argmax(probs).item())
            pred = BIN_LABELS[pred_id]
            conf = float(probs[pred_id]) * 100
            actual.append(sample.label)
            predicted.append(pred)
            confidences.append(conf)
            if include_rows:
                rows.append({
                    "image_path": str(sample.image_path),
                    "source": sample.source,
                    "detail": sample.detail,
                    "actual": sample.label,
                    "predicted": pred,
                    "confidence": round(conf, 4),
                    "correct": sample.label == pred,
                })
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    by_label = {}
    for label in BIN_LABELS:
        indices = [i for i, a in enumerate(actual) if a == label]
        label_correct = sum(1 for i in indices if predicted[i] == label)
        by_label[label] = {
            "total": len(indices),
            "correct": label_correct,
            "accuracy": round(label_correct / len(indices) * 100, 2) if indices else 0,
        }
    confusion = {label: {pred: 0 for pred in BIN_LABELS} for label in BIN_LABELS}
    for a, p in zip(actual, predicted):
        confusion[a][p] += 1
    result = {
        "total": len(actual),
        "correct": correct,
        "accuracy": round(correct / max(1, len(actual)) * 100, 2),
        "avg_confidence": round(sum(confidences) / max(1, len(confidences)), 2),
        "by_label": by_label,
        "confusion_matrix": confusion,
    }
    if include_rows:
        result["rows"] = rows
    return result


def without_rows(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "rows"}


def write_report(path: Path, metadata: dict[str, Any], history: list[dict[str, Any]]) -> None:
    target = metadata["target_metrics"]
    hf = metadata["hf_test_metrics"]
    lines = [
        "# 四桶垃圾分类目标域适配模型报告",
        "",
        "## 结论",
        "",
        f"- 模型：`{metadata['name']}`",
        f"- 基础模型：`{metadata['base_model_dir']}`",
        f"- 目标域样本：`{metadata['target_cases']}`",
        f"- 训练样本总量：{metadata['train_count']}",
        f"- Target repeat：{metadata['target_repeat']}",
        f"- 目标复杂 100 图拟合准确率：**{target['accuracy']}%**",
        f"- 目标复杂 100 图平均置信度：**{target['avg_confidence']}**",
        f"- 公开 HF 测试集准确率：**{hf['accuracy']}%**",
        f"- 公开 HF 测试集平均置信度：**{hf['avg_confidence']}**",
        "",
        "> 注意：目标复杂 100 图参与了适配训练，所以这里是目标域拟合/验收指标，不是独立泛化指标。",
        "",
        "## 训练历史",
        "",
        "| Epoch | Loss | Target accuracy |",
        "| ---: | ---: | ---: |",
    ]
    for row in history:
        lines.append(f"| {row['epoch']} | {row['loss']} | {row['target_accuracy']}% |")
    lines.extend(["", "## 目标复杂 100 图分桶", "", "| Bin | Total | Correct | Accuracy |", "| --- | ---: | ---: | ---: |"])
    for label in BIN_LABELS:
        row = target["by_label"][label]
        lines.append(f"| {label} | {row['total']} | {row['correct']} | {row['accuracy']}% |")
    lines.extend(["", "## 公开 HF 测试集分桶", "", "| Bin | Total | Correct | Accuracy |", "| --- | ---: | ---: | ---: |"])
    for label in BIN_LABELS:
        row = hf["by_label"][label]
        lines.append(f"| {label} | {row['total']} | {row['correct']} | {row['accuracy']}% |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
