#!/usr/bin/env python3
"""
Evaluate public Hugging Face waste-classification models on the same real
complex 100-image benchmark used for the deployed API.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


BIN_ORDER = ["green", "yellow", "blue", "red"]
DEFAULT_MODELS = [
    "watersplash/waste-classification",
    "prithivMLmods/Trash-Net",
    "yangy50/garbage-classification",
    "tribber93/my-trash-classification",
]


@dataclass
class ModelSpec:
    model_id: str
    label_to_bin: dict[str, str]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="test-results/real-complex-100/cases.json")
    parser.add_argument("--out-dir", default="test-results/hf-model-shootout")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(Path(args.cases))
    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)

    summaries = []
    for model_id in args.models:
        print(f"\nEvaluating {model_id}", flush=True)
        started = time.time()
        result = evaluate_model(model_id, cases, device)
        result["elapsed_seconds"] = round(time.time() - started, 2)
        safe_name = model_id.replace("/", "__")
        write_json(out_dir / f"{safe_name}.json", result)
        write_csv(out_dir / f"{safe_name}.csv", result["rows"])
        summaries.append({key: result[key] for key in [
            "model_id",
            "accuracy",
            "accuracy_excluding_unmapped",
            "unmapped_count",
            "avg_confidence",
            "elapsed_seconds",
            "by_expected",
            "confusion_matrix",
            "labels",
        ]})

    shootout = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cases": str(Path(args.cases)),
        "device": str(device),
        "models": summaries,
        "best_by_accuracy": max(summaries, key=lambda item: item["accuracy"]) if summaries else None,
    }
    write_json(out_dir / "summary.json", shootout)
    write_report(out_dir / "report.zh.md", shootout)
    print("\nShootout complete:")
    print(f"- {out_dir / 'summary.json'}")
    print(f"- {out_dir / 'report.zh.md'}")


def choose_device(value: str) -> torch.device:
    if value == "mps":
        return torch.device("mps")
    if value == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["cases"]


def evaluate_model(model_id: str, cases: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id).to(device)
    model.eval()
    labels = {int(key): value for key, value in model.config.id2label.items()} if isinstance(next(iter(model.config.id2label.keys())), str) else model.config.id2label
    label_to_bin = infer_label_mapping(list(labels.values()))

    rows = []
    correct = 0
    mapped = 0
    confidences = []
    confusion = {expected: {predicted: 0 for predicted in BIN_ORDER + ["unmapped"]} for expected in BIN_ORDER}
    by_expected = {expected: {"total": 0, "correct": 0, "unmapped": 0} for expected in BIN_ORDER}

    for index, case in enumerate(cases, start=1):
        image = Image.open(case["image_path"]).convert("RGB")
        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu()
        top_index = int(torch.argmax(probs).item())
        top_label = labels[top_index]
        confidence = float(probs[top_index].item()) * 100
        predicted_bin = label_to_bin.get(normalize_label(top_label), "unmapped")
        expected_bin = case["expected_bin"]
        is_mapped = predicted_bin != "unmapped"
        is_correct = predicted_bin == expected_bin
        by_expected[expected_bin]["total"] += 1
        if not is_mapped:
            by_expected[expected_bin]["unmapped"] += 1
        if is_correct:
            correct += 1
            by_expected[expected_bin]["correct"] += 1
        if is_mapped:
            mapped += 1
        confidences.append(confidence)
        confusion[expected_bin][predicted_bin] += 1
        top5 = torch.topk(probs, min(5, len(probs)))
        rows.append({
            "case_id": case["case_id"],
            "expected_bin": expected_bin,
            "source_dataset": case["source_dataset"],
            "source_category": case["source_category"],
            "image_path": case["image_path"],
            "top_label": top_label,
            "predicted_bin": predicted_bin,
            "confidence": round(confidence, 4),
            "correct": is_correct,
            "top5": [
                {"label": labels[int(label_index)], "confidence": round(float(value) * 100, 4)}
                for value, label_index in zip(top5.values, top5.indices)
            ],
        })
        if index % 20 == 0 or index == len(cases):
            print(f"  {index}/{len(cases)} accuracy={correct / index * 100:.1f}%", flush=True)

    for expected, item in by_expected.items():
        total = item["total"]
        item["accuracy"] = round(item["correct"] / total * 100, 2) if total else 0

    return {
        "model_id": model_id,
        "labels": list(labels.values()),
        "label_to_bin": label_to_bin,
        "total": len(cases),
        "correct": correct,
        "mapped_count": mapped,
        "unmapped_count": len(cases) - mapped,
        "accuracy": round(correct / len(cases) * 100, 2),
        "accuracy_excluding_unmapped": round(correct / mapped * 100, 2) if mapped else 0,
        "avg_confidence": round(sum(confidences) / len(confidences), 2) if confidences else 0,
        "by_expected": by_expected,
        "confusion_matrix": confusion,
        "rows": rows,
    }


def infer_label_mapping(labels: list[str]) -> dict[str, str]:
    mapping = {}
    for label in labels:
        normalized = normalize_label(label)
        if normalized in {"battery", "batteries"}:
            mapping[normalized] = "red"
        elif normalized in {"biological", "organic", "food waste", "food_waste", "brown grass", "brown-grass"}:
            mapping[normalized] = "green"
        elif normalized in {
            "cardboard",
            "glass",
            "brown glass",
            "brown-glass",
            "green glass",
            "green-glass",
            "white glass",
            "white-glass",
            "metal",
            "paper",
            "plastic",
        }:
            mapping[normalized] = "yellow"
        elif normalized in {"trash", "general", "clothes", "shoes"}:
            mapping[normalized] = "blue"
    return mapping


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("_", " ")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "case_id",
            "expected_bin",
            "source_dataset",
            "source_category",
            "top_label",
            "predicted_bin",
            "confidence",
            "correct",
            "image_path",
            "top5",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "top5": json.dumps(row["top5"], ensure_ascii=False)})


def write_report(path: Path, shootout: dict[str, Any]) -> None:
    lines = [
        "# 开源垃圾分类模型横评",
        "",
        f"- 生成时间：`{shootout['generated_at']}`",
        f"- 测试集：`{shootout['cases']}`",
        f"- 设备：`{shootout['device']}`",
        f"- 最佳模型：`{shootout['best_by_accuracy']['model_id']}`",
        f"- 最佳准确率：**{shootout['best_by_accuracy']['accuracy']}%**",
        "",
        "## 总览",
        "",
        "| Model | Accuracy | Excluding unmapped | Unmapped | Avg confidence | Time | Labels |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in sorted(shootout["models"], key=lambda row: row["accuracy"], reverse=True):
        labels = ", ".join(item["labels"])
        lines.append(
            f"| `{item['model_id']}` | {item['accuracy']}% | {item['accuracy_excluding_unmapped']}% | "
            f"{item['unmapped_count']} | {item['avg_confidence']} | {item['elapsed_seconds']}s | {labels} |"
        )

    lines.extend(["", "## 最佳模型分桶结果", ""])
    best = shootout["best_by_accuracy"]
    lines.extend([
        "| Bin | Cases | Correct | Accuracy | Unmapped |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for bin_key in BIN_ORDER:
        row = best["by_expected"][bin_key]
        lines.append(f"| {bin_key} | {row['total']} | {row['correct']} | {row['accuracy']}% | {row['unmapped']} |")

    lines.extend(["", "## 最佳模型混淆矩阵", ""])
    lines.extend(["| Expected \\ Predicted | green | yellow | blue | red | unmapped |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for expected in BIN_ORDER:
        row = best["confusion_matrix"][expected]
        lines.append(f"| {expected} | {row['green']} | {row['yellow']} | {row['blue']} | {row['red']} | {row['unmapped']} |")

    lines.extend([
        "",
        "## 说明",
        "",
        "- 这是直接使用别人已经训练好的 Hugging Face 模型，没有在本项目数据上二次训练。",
        "- 标签通过规则映射到泰国四桶：biological/organic -> green，cardboard/glass/metal/paper/plastic -> yellow，trash/clothes/shoes -> blue，battery -> red。",
        "- 如果要继续提高性能，下一步应把最佳模型作为初始化权重，在 AlphaTrash + RealWaste + TACO + 有害垃圾数据上微调。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
