#!/usr/bin/env python3
"""
Download real waste images from public datasets and benchmark the deployed API.

Sources:
- TACO: real litter in context, COCO annotations, Flickr-hosted images.
- RealWaste: real landfill/waste facility images from the UCI/GitHub release.

The API request uses neutral file names such as real_case_001.jpg so expected
labels are not leaked to the model.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import random
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


DEFAULT_ENDPOINT = "https://bingo-sorter.vercel.app/api/classify"
TACO_ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
TACO_REPO_URL = "https://github.com/pedropro/TACO"
REALWASTE_REPO_URL = "https://github.com/sam-single/realwaste"
REALWASTE_UCI_URL = "https://archive.ics.uci.edu/dataset/908/realwaste"
REALWASTE_API_ROOT = "https://api.github.com/repos/sam-single/realwaste/contents/RealWaste"

BIN_ORDER = ["green", "yellow", "blue", "red"]
BIN_LABELS = {
    "green": "Food / Organic waste",
    "yellow": "Recyclable waste",
    "blue": "General waste",
    "red": "Hazardous waste",
}

TACO_RED = {"Battery", "Aerosol", "Aluminium blister pack", "Carded blister pack"}
TACO_GREEN = {"Food waste"}
TACO_YELLOW = {
    "Aluminium foil",
    "Other plastic bottle",
    "Clear plastic bottle",
    "Glass bottle",
    "Metal bottle cap",
    "Food Can",
    "Drink can",
    "Toilet tube",
    "Other carton",
    "Egg carton",
    "Drink carton",
    "Corrugated carton",
    "Meal carton",
    "Pizza box",
    "Glass cup",
    "Glass jar",
    "Metal lid",
    "Magazine paper",
    "Wrapping paper",
    "Normal paper",
    "Paper bag",
    "Pop tab",
    "Scrap metal",
}
TACO_BLUE = {
    "Plastic bottle cap",
    "Broken glass",
    "Paper cup",
    "Disposable plastic cup",
    "Foam cup",
    "Other plastic cup",
    "Plastic lid",
    "Other plastic",
    "Tissues",
    "Plastified paper bag",
    "Plastic film",
    "Six pack rings",
    "Garbage bag",
    "Other plastic wrapper",
    "Single-use carrier bag",
    "Polypropylene bag",
    "Crisp packet",
    "Spread tub",
    "Tupperware",
    "Disposable food container",
    "Foam food container",
    "Other plastic container",
    "Plastic glooves",
    "Plastic utensils",
    "Rope & strings",
    "Shoe",
    "Squeezable tube",
    "Plastic straw",
    "Paper straw",
    "Styrofoam piece",
    "Unlabeled litter",
    "Cigarette",
}

REALWASTE_MAP = {
    "Food Organics": "green",
    "Vegetation": "green",
    "Cardboard": "yellow",
    "Glass": "yellow",
    "Metal": "yellow",
    "Paper": "yellow",
    "Plastic": "yellow",
    "Miscellaneous Trash": "blue",
    "Textile Trash": "blue",
}


@dataclass
class Case:
    case_id: str
    expected_bin: str
    source_dataset: str
    source_category: str
    source_detail: str
    image_path: Path
    request_file_name: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--out-dir", default="test-results/real-complex-100")
    parser.add_argument("--data-dir", default="data-sources/real-complex")
    parser.add_argument("--cases-per-bin", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--rebuild-images", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    images_dir = out_dir / "images"
    raw_dir = out_dir / "raw"
    for path in [out_dir, data_dir, images_dir, raw_dir]:
        path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    taco = load_taco(data_dir / "taco_annotations.json")
    cases = build_real_cases(taco, data_dir, images_dir, args.cases_per_bin, rng, args.rebuild_images)
    write_metadata(out_dir / "cases.json", cases, args)

    started = time.time()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {
            executor.submit(classify_case, case, args.endpoint, args.timeout, args.retries, raw_dir): case
            for case in cases
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            status = "OK" if result["correct"] else "MISS"
            if result.get("error"):
                status = "ERR"
            print(
                f"[{index:03d}/{len(cases)}] {status} {result['case_id']} "
                f"src={result['source_dataset']} expected={result['expected_bin']} "
                f"predicted={result.get('predicted_bin') or '-'} "
                f"conf={result.get('confidence') if result.get('confidence') is not None else '-'} "
                f"latency={result['latency_ms']}ms",
                flush=True,
            )

    results.sort(key=lambda item: item["case_id"])
    elapsed = time.time() - started
    summary = build_summary(results, elapsed, args, cases)

    write_json(out_dir / "results.json", results)
    write_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "results.csv", results)
    write_report(out_dir / "report.md", summary, results, args)
    write_report_zh(out_dir / "report.zh.md", summary, results, args)

    print("\nReport written:")
    print(f"- {out_dir / 'report.zh.md'}")
    print(f"- {out_dir / 'report.md'}")
    print(f"- {out_dir / 'results.csv'}")
    print(f"- {out_dir / 'summary.json'}")


def load_taco(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(f"Downloading TACO annotations: {TACO_ANNOTATIONS_URL}", flush=True)
        urllib.request.urlretrieve(TACO_ANNOTATIONS_URL, path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_real_cases(
    taco: dict[str, Any],
    data_dir: Path,
    images_dir: Path,
    cases_per_bin: int,
    rng: random.Random,
    rebuild_images: bool,
) -> list[Case]:
    plan = {
        "green": {"realwaste": 21, "taco": 4},
        "yellow": {"realwaste": 10, "taco": 15},
        "blue": {"realwaste": 10, "taco": 15},
        "red": {"realwaste": 0, "taco": 25},
    }
    for bin_key in BIN_ORDER:
        planned = plan[bin_key]["realwaste"] + plan[bin_key]["taco"]
        if planned != cases_per_bin:
            raise ValueError(f"Plan for {bin_key} has {planned} cases, expected {cases_per_bin}.")

    taco_candidates = prepare_taco_candidates(taco, rng)
    realwaste_candidates = prepare_realwaste_candidates(rng)

    cases: list[Case] = []
    serial = 1
    for bin_key in BIN_ORDER:
        selected_realwaste = realwaste_candidates[bin_key][: plan[bin_key]["realwaste"]]
        selected_taco = sample_with_repeats(taco_candidates[bin_key], plan[bin_key]["taco"], rng)

        for source_case in selected_realwaste:
            case_id = f"{serial:03d}_{bin_key}_realwaste"
            image_path = images_dir / f"{case_id}.jpg"
            if rebuild_images or not image_path.exists():
                raw_path = download_realwaste_image(data_dir, source_case)
                normalize_photo(raw_path, image_path)
            cases.append(
                Case(
                    case_id=case_id,
                    expected_bin=bin_key,
                    source_dataset="RealWaste",
                    source_category=source_case["category"],
                    source_detail=source_case["name"],
                    image_path=image_path,
                    request_file_name=f"real_case_{serial:03d}.jpg",
                )
            )
            serial += 1

        for repeat_index, source_case in enumerate(selected_taco):
            case_id = f"{serial:03d}_{bin_key}_taco"
            image_path = images_dir / f"{case_id}.jpg"
            if rebuild_images or not image_path.exists():
                raw_path = download_taco_image(data_dir, source_case)
                crop_taco_object(raw_path, source_case, image_path, repeat_index)
            cases.append(
                Case(
                    case_id=case_id,
                    expected_bin=bin_key,
                    source_dataset="TACO",
                    source_category=source_case["category_name"],
                    source_detail=f"image_id={source_case['image_id']} annotation_id={source_case['annotation_id']}",
                    image_path=image_path,
                    request_file_name=f"real_case_{serial:03d}.jpg",
                )
            )
            serial += 1

    return cases


def prepare_taco_candidates(taco: dict[str, Any], rng: random.Random) -> dict[str, list[dict[str, Any]]]:
    categories = {category["id"]: category for category in taco["categories"]}
    images = {image["id"]: image for image in taco["images"]}
    candidates = {bin_key: [] for bin_key in BIN_ORDER}

    for annotation in taco["annotations"]:
        category = categories[annotation["category_id"]]
        bin_key = map_taco_category(category["name"])
        if not bin_key:
            continue
        image = images[annotation["image_id"]]
        if not image.get("flickr_640_url") and not image.get("flickr_url"):
            continue
        area = float(annotation.get("area") or (annotation.get("bbox") or [0, 0, 0, 0])[2] * (annotation.get("bbox") or [0, 0, 0, 0])[3])
        if area <= 0:
            continue
        candidates[bin_key].append(
            {
                "bin": bin_key,
                "annotation_id": annotation["id"],
                "image_id": image["id"],
                "category_name": category["name"],
                "supercategory": category.get("supercategory", ""),
                "bbox": annotation["bbox"],
                "original_width": image["width"],
                "original_height": image["height"],
                "file_name": image["file_name"],
                "url": image.get("flickr_640_url") or image.get("flickr_url"),
                "fallback_url": image.get("flickr_url") or image.get("flickr_640_url"),
                "area": area,
            }
        )

    for bin_key, rows in candidates.items():
        rows.sort(key=lambda item: item["area"], reverse=True)
        top = rows[: max(80, min(len(rows), 200))]
        rng.shuffle(top)
        candidates[bin_key] = top
    return candidates


def map_taco_category(name: str) -> str | None:
    if name in TACO_RED:
        return "red"
    if name in TACO_GREEN:
        return "green"
    if name in TACO_YELLOW:
        return "yellow"
    if name in TACO_BLUE:
        return "blue"
    return None


def prepare_realwaste_candidates(rng: random.Random) -> dict[str, list[dict[str, Any]]]:
    candidates = {bin_key: [] for bin_key in BIN_ORDER}
    for category, bin_key in REALWASTE_MAP.items():
        rows = list_realwaste_category(category)
        rng.shuffle(rows)
        for row in rows:
            candidates[bin_key].append(
                {
                    "bin": bin_key,
                    "category": category,
                    "name": row["name"],
                    "download_url": row["download_url"],
                }
            )
    for rows in candidates.values():
        rng.shuffle(rows)
    return candidates


def list_realwaste_category(category: str) -> list[dict[str, Any]]:
    encoded = urllib.parse.quote(category)
    url = f"{REALWASTE_API_ROOT}/{encoded}?ref=main"
    with urllib.request.urlopen(url, timeout=60) as response:
        rows = json.loads(response.read().decode("utf-8"))
    return [
        {"name": row["name"], "download_url": row["download_url"]}
        for row in rows
        if row.get("download_url") and row["name"].lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]


def sample_with_repeats(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("No candidates available for requested bin.")
    if len(rows) >= count:
        return rows[:count]
    selected = list(rows)
    while len(selected) < count:
        selected.append(rng.choice(rows))
    return selected


def download_realwaste_image(data_dir: Path, source_case: dict[str, Any]) -> Path:
    target_dir = data_dir / "realwaste" / safe_name(source_case["category"])
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / safe_name(source_case["name"])
    if not target.exists():
        download_url(source_case["download_url"], target)
    return target


def download_taco_image(data_dir: Path, source_case: dict[str, Any]) -> Path:
    target_dir = data_dir / "taco_images"
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(source_case["file_name"]).suffix or ".jpg"
    target = target_dir / f"{source_case['image_id']:06d}{suffix}"
    if not target.exists():
        try:
            download_url(source_case["url"], target)
        except Exception:
            if source_case.get("fallback_url") and source_case["fallback_url"] != source_case["url"]:
                download_url(source_case["fallback_url"], target)
            else:
                raise
    return target


def download_url(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        target.write_bytes(response.read())


def normalize_photo(input_path: Path, output_path: Path) -> None:
    image = Image.open(input_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((900, 900), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=88, optimize=True)


def crop_taco_object(input_path: Path, source_case: dict[str, Any], output_path: Path, repeat_index: int) -> None:
    image = Image.open(input_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    sx = width / float(source_case["original_width"])
    sy = height / float(source_case["original_height"])
    x, y, w, h = source_case["bbox"]
    x *= sx
    y *= sy
    w *= sx
    h *= sy

    scale = [2.0, 2.4, 1.7, 2.8][repeat_index % 4]
    cx = x + w / 2
    cy = y + h / 2
    side = max(w, h) * scale
    min_side = min(width, height, max(360, side))
    side = max(side, min_side)
    left = max(0, cx - side / 2)
    top = max(0, cy - side / 2)
    right = min(width, cx + side / 2)
    bottom = min(height, cy + side / 2)

    if right - left < side:
        left = max(0, right - side)
        right = min(width, left + side)
    if bottom - top < side:
        top = max(0, bottom - side)
        bottom = min(height, top + side)

    crop = image.crop((int(left), int(top), int(right), int(bottom)))
    crop.thumbnail((900, 900), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, format="JPEG", quality=88, optimize=True)


def classify_case(case: Case, endpoint: str, timeout: int, retries: int, raw_dir: Path) -> dict[str, Any]:
    started = time.time()
    payload = {
        "image": image_to_data_url(case.image_path),
        "mimeType": "image/jpeg",
        "fileName": case.request_file_name,
        "source": "upload",
    }

    error = ""
    data: dict[str, Any] = {}
    for attempt in range(retries + 1):
        try:
            data = post_json(endpoint, payload, timeout)
            break
        except Exception as exc:  # noqa: BLE001 - report exact API failure.
            error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                data = {}

    latency_ms = int(round((time.time() - started) * 1000))
    predicted_bin = data.get("bin") if isinstance(data, dict) else None
    confidence = data.get("confidence") if isinstance(data, dict) else None
    correct = predicted_bin == case.expected_bin
    raw_path = raw_dir / f"{case.case_id}.json"
    write_json(raw_path, data if data else {"error": error})

    return {
        "case_id": case.case_id,
        "expected_bin": case.expected_bin,
        "expected_label": BIN_LABELS[case.expected_bin],
        "source_dataset": case.source_dataset,
        "source_category": case.source_category,
        "source_detail": case.source_detail,
        "request_file_name": case.request_file_name,
        "image_path": str(case.image_path),
        "predicted_bin": predicted_bin,
        "predicted_label": data.get("label") if isinstance(data, dict) else None,
        "category": data.get("category") if isinstance(data, dict) else None,
        "confidence": confidence,
        "needs_manual_check": data.get("needsManualCheck") if isinstance(data, dict) else None,
        "classifier": data.get("classifier") if isinstance(data, dict) else None,
        "algorithm_version": data.get("algorithmVersion") if isinstance(data, dict) else None,
        "model": data.get("model") if isinstance(data, dict) else None,
        "ai_item_name": (data.get("ai") or {}).get("itemName") if isinstance(data, dict) else None,
        "ai_material": (data.get("ai") or {}).get("material") if isinstance(data, dict) else None,
        "evidence": " | ".join(data.get("evidence") or []) if isinstance(data, dict) else "",
        "disposal": data.get("disposal") if isinstance(data, dict) else None,
        "latency_ms": latency_ms,
        "correct": bool(correct),
        "error": error if not data else "",
        "raw_path": str(raw_path),
    }


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def post_json(endpoint: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc


def build_summary(results: list[dict[str, Any]], elapsed: float, args: argparse.Namespace, cases: list[Case]) -> dict[str, Any]:
    total = len(results)
    errors = [item for item in results if item.get("error")]
    valid = [item for item in results if not item.get("error")]
    correct = [item for item in valid if item.get("correct")]
    confidence_values = [float(item["confidence"]) for item in valid if isinstance(item.get("confidence"), (int, float))]
    latencies = [int(item["latency_ms"]) for item in results if isinstance(item.get("latency_ms"), int)]

    by_expected = {}
    for bin_key in BIN_ORDER:
        rows = [item for item in results if item["expected_bin"] == bin_key]
        ok_rows = [item for item in rows if item.get("correct")]
        conf_rows = [float(item["confidence"]) for item in rows if isinstance(item.get("confidence"), (int, float))]
        by_expected[bin_key] = {
            "total": len(rows),
            "correct": len(ok_rows),
            "accuracy": safe_ratio(len(ok_rows), len(rows)),
            "avg_confidence": round(statistics.mean(conf_rows), 2) if conf_rows else None,
            "manual_check_count": sum(1 for item in rows if item.get("needs_manual_check") is True),
        }

    by_source = {}
    for source in sorted({item["source_dataset"] for item in results}):
        rows = [item for item in results if item["source_dataset"] == source]
        ok_rows = [item for item in rows if item.get("correct")]
        by_source[source] = {
            "total": len(rows),
            "correct": len(ok_rows),
            "accuracy": safe_ratio(len(ok_rows), len(rows)),
        }

    confusion = {
        expected: {predicted: 0 for predicted in BIN_ORDER + ["error", "missing"]}
        for expected in BIN_ORDER
    }
    for item in results:
        expected = item["expected_bin"]
        predicted = "error" if item.get("error") else item.get("predicted_bin") or "missing"
        confusion[expected][predicted] += 1

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "endpoint": args.endpoint,
        "seed": args.seed,
        "total_cases": total,
        "source_counts": count_by(cases, "source_dataset"),
        "valid_responses": len(valid),
        "error_count": len(errors),
        "correct_count": len(correct),
        "accuracy": safe_ratio(len(correct), total),
        "accuracy_excluding_errors": safe_ratio(len(correct), len(valid)),
        "avg_confidence": round(statistics.mean(confidence_values), 2) if confidence_values else None,
        "median_confidence": round(statistics.median(confidence_values), 2) if confidence_values else None,
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "p95_latency_ms": percentile(latencies, 95) if latencies else None,
        "elapsed_seconds": round(elapsed, 2),
        "concurrency": args.concurrency,
        "cases_per_bin": args.cases_per_bin,
        "by_expected": by_expected,
        "by_source": by_source,
        "confusion_matrix": confusion,
        "sources": {
            "TACO": TACO_REPO_URL,
            "RealWaste": REALWASTE_REPO_URL,
            "RealWaste_UCI": REALWASTE_UCI_URL,
        },
    }


def count_by(cases: list[Case], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        key = str(getattr(case, attr))
        counts[key] = counts.get(key, 0) + 1
    return counts


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator * 100, 2)


def percentile(values: list[int], pct: int) -> int:
    ordered = sorted(values)
    index = math.ceil((pct / 100) * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def write_metadata(path: Path, cases: list[Case], args: argparse.Namespace) -> None:
    write_json(
        path,
        {
            "endpoint": args.endpoint,
            "seed": args.seed,
            "cases_per_bin": args.cases_per_bin,
            "case_count": len(cases),
            "note": "Request file names are neutral and do not contain expected labels.",
            "sources": {
                "TACO": TACO_REPO_URL,
                "RealWaste": REALWASTE_REPO_URL,
                "RealWaste_UCI": REALWASTE_UCI_URL,
            },
            "cases": [
                {
                    "case_id": case.case_id,
                    "expected_bin": case.expected_bin,
                    "source_dataset": case.source_dataset,
                    "source_category": case.source_category,
                    "source_detail": case.source_detail,
                    "image_path": str(case.image_path),
                    "request_file_name": case.request_file_name,
                }
                for case in cases
            ],
        },
    )


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "expected_bin",
        "expected_label",
        "source_dataset",
        "source_category",
        "source_detail",
        "request_file_name",
        "predicted_bin",
        "predicted_label",
        "category",
        "confidence",
        "needs_manual_check",
        "classifier",
        "algorithm_version",
        "model",
        "ai_item_name",
        "ai_material",
        "latency_ms",
        "correct",
        "error",
        "image_path",
        "raw_path",
        "evidence",
        "disposal",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_report(path: Path, summary: dict[str, Any], results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    wrong = [item for item in results if not item.get("correct")]
    low_conf = [item for item in results if isinstance(item.get("confidence"), (int, float)) and float(item["confidence"]) < 80]
    manual = [item for item in results if item.get("needs_manual_check") is True]
    lines = [
        "# Real Complex Image Benchmark",
        "",
        "## Summary",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Endpoint: `{summary['endpoint']}`",
        f"- Sources: TACO ({summary['source_counts'].get('TACO', 0)} cases), RealWaste ({summary['source_counts'].get('RealWaste', 0)} cases)",
        f"- Total cases: **{summary['total_cases']}**",
        f"- Valid responses: **{summary['valid_responses']}**",
        f"- Errors: **{summary['error_count']}**",
        f"- Correct: **{summary['correct_count']}**",
        f"- Accuracy: **{summary['accuracy']}%**",
        f"- Average confidence: **{summary['avg_confidence']}**",
        f"- Median confidence: **{summary['median_confidence']}**",
        f"- Average latency: **{summary['avg_latency_ms']} ms**",
        f"- P95 latency: **{summary['p95_latency_ms']} ms**",
        f"- Wall time: **{summary['elapsed_seconds']} s** with concurrency `{summary['concurrency']}`",
        "",
        "Expected labels are mapped from dataset annotations/classes to the Thailand bins. TACO object crops use real image content around annotated objects; RealWaste uses real full-frame facility images.",
        "",
        "## Per-Bin Accuracy",
        "",
        "| Expected bin | Cases | Correct | Accuracy | Avg confidence | Manual checks |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    append_bin_rows(lines, summary)
    append_confusion(lines, summary)
    append_source_rows(lines, summary)
    append_wrong_cases(lines, wrong)
    append_low_confidence(lines, low_conf)
    lines.extend([
        "",
        "## Manual Check Flags",
        "",
        f"- Manual check count: **{len(manual)}**",
        "",
        "## Source Links",
        "",
        f"- TACO: {TACO_REPO_URL}",
        f"- RealWaste GitHub: {REALWASTE_REPO_URL}",
        f"- RealWaste UCI: {REALWASTE_UCI_URL}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_zh(path: Path, summary: dict[str, Any], results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    wrong = [item for item in results if not item.get("correct")]
    low_conf = [item for item in results if isinstance(item.get("confidence"), (int, float)) and float(item["confidence"]) < 80]
    manual = [item for item in results if item.get("needs_manual_check") is True]
    lines = [
        "# 真实复杂图片 100 张测试报告",
        "",
        "## 测试结论",
        "",
        f"- 测试时间：`{summary['generated_at']}`",
        f"- 测试接口：`{summary['endpoint']}`",
        "- 测试模式：线上 OpenAI Direct API，后端不做本地规则二次改判。",
        f"- 数据来源：TACO {summary['source_counts'].get('TACO', 0)} 张，RealWaste {summary['source_counts'].get('RealWaste', 0)} 张。",
        f"- 总测试数：**{summary['total_cases']}**",
        f"- 有效返回：**{summary['valid_responses']}**",
        f"- 请求错误：**{summary['error_count']}**",
        f"- 分类正确：**{summary['correct_count']}**",
        f"- 总准确率：**{summary['accuracy']}%**",
        f"- 平均置信度：**{summary['avg_confidence']}**",
        f"- 中位置信度：**{summary['median_confidence']}**",
        f"- 平均请求耗时：**{summary['avg_latency_ms']} ms**",
        f"- P95 请求耗时：**{summary['p95_latency_ms']} ms**",
        f"- 总运行时间：**{summary['elapsed_seconds']} s**，并发数 `{summary['concurrency']}`",
        "",
        "这次不是合成图。TACO 图片是真实野外垃圾场景，包含道路、树林、海滩等复杂背景；RealWaste 图片是真实垃圾处理设施中的垃圾材料照片。请求给后端的文件名是 `real_case_001.jpg` 这类中性名字，没有把 expected bin 或类别名传给模型。",
        "",
        "## 分类别结果",
        "",
        "| 期望类别 | 图片数 | 正确数 | 准确率 | 平均置信度 | 人工确认 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    append_bin_rows(lines, summary, zh=True)
    append_confusion(lines, summary, zh=True)
    append_source_rows(lines, summary, zh=True)
    append_wrong_cases(lines, wrong, zh=True)
    append_low_confidence(lines, low_conf, zh=True)
    lines.extend([
        "",
        "## 人工确认",
        "",
        f"- 触发人工确认：**{len(manual)}**",
        "",
        "## 数据映射说明",
        "",
        "- TACO：根据 COCO 标注类别映射到泰国四类垃圾；例如塑料瓶/玻璃瓶/易拉罐/纸箱映射 yellow，包装袋/烟头/泡沫/脏纸类映射 blue，食物垃圾映射 green，电池/喷雾罐/药品泡罩映射 red。",
        "- RealWaste：Food Organics 和 Vegetation 映射 green；Cardboard/Glass/Metal/Paper/Plastic 映射 yellow；Miscellaneous Trash 和 Textile Trash 映射 blue。",
        "- TACO 用标注框附近的真实图像裁剪，让目标物体更明确；RealWaste 使用真实完整图片。",
        "",
        "## 数据源",
        "",
        f"- TACO：{TACO_REPO_URL}",
        f"- RealWaste GitHub：{REALWASTE_REPO_URL}",
        f"- RealWaste UCI：{REALWASTE_UCI_URL}",
        "",
        "## 产物位置",
        "",
        "- `test-results/real-complex-100/images/`：测试图片。",
        "- `test-results/real-complex-100/raw/`：每张图片的原始 API 返回。",
        "- `test-results/real-complex-100/results.csv`：明细表。",
        "- `test-results/real-complex-100/summary.json`：汇总数据。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_bin_rows(lines: list[str], summary: dict[str, Any], zh: bool = False) -> None:
    labels = {
        "green": "green / 厨余有机垃圾" if zh else "green (Food / Organic waste)",
        "yellow": "yellow / 可回收垃圾" if zh else "yellow (Recyclable waste)",
        "blue": "blue / 一般垃圾" if zh else "blue (General waste)",
        "red": "red / 有害垃圾" if zh else "red (Hazardous waste)",
    }
    for bin_key in BIN_ORDER:
        row = summary["by_expected"][bin_key]
        lines.append(
            f"| {labels[bin_key]} | {row['total']} | {row['correct']} | "
            f"{row['accuracy']}% | {row['avg_confidence']} | {row['manual_check_count']} |"
        )


def append_confusion(lines: list[str], summary: dict[str, Any], zh: bool = False) -> None:
    title = "## 混淆矩阵" if zh else "## Confusion Matrix"
    header = "| 期望 \\ 预测 | green | yellow | blue | red | error | missing |" if zh else "| Expected \\ Predicted | green | yellow | blue | red | error | missing |"
    lines.extend(["", title, "", header, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for expected in BIN_ORDER:
        row = summary["confusion_matrix"][expected]
        lines.append(f"| {expected} | {row['green']} | {row['yellow']} | {row['blue']} | {row['red']} | {row['error']} | {row['missing']} |")


def append_source_rows(lines: list[str], summary: dict[str, Any], zh: bool = False) -> None:
    title = "## 分数据源结果" if zh else "## Per-Source Accuracy"
    lines.extend(["", title, "", "| Source | Cases | Correct | Accuracy |", "| --- | ---: | ---: | ---: |"])
    for source, row in summary["by_source"].items():
        lines.append(f"| {source} | {row['total']} | {row['correct']} | {row['accuracy']}% |")


def append_wrong_cases(lines: list[str], wrong: list[dict[str, Any]], zh: bool = False) -> None:
    title = "## 错误或异常样例" if zh else "## Wrong Or Error Cases"
    lines.extend(["", title, ""])
    if not wrong:
        lines.append("没有错误或异常样例。" if zh else "No wrong or error cases.")
        return
    lines.extend([
        "| Case | Source | Category | Expected | Predicted | Confidence | Evidence / Error |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ])
    for item in wrong:
        evidence = item.get("error") or item.get("evidence") or ""
        lines.append(
            f"| `{item['case_id']}` | {item['source_dataset']} | {safe_md(item['source_category'])} | "
            f"{item['expected_bin']} | {item.get('predicted_bin') or '-'} | {item.get('confidence') or '-'} | "
            f"{safe_md(shorten(evidence, 220))} |"
        )


def append_low_confidence(lines: list[str], low_conf: list[dict[str, Any]], zh: bool = False) -> None:
    title = "## 低置信度样例（< 80）" if zh else "## Low Confidence Cases (< 80)"
    lines.extend(["", title, ""])
    if not low_conf:
        lines.append("没有低于 80 置信度的样例。" if zh else "No cases below 80 confidence.")
        return
    lines.extend([
        "| Case | Source | Category | Expected | Predicted | Confidence | Manual check |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ])
    for item in low_conf:
        lines.append(
            f"| `{item['case_id']}` | {item['source_dataset']} | {safe_md(item['source_category'])} | "
            f"{item['expected_bin']} | {item.get('predicted_bin') or '-'} | {item.get('confidence')} | "
            f"{item.get('needs_manual_check')} |"
        )


def shorten(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def safe_md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in value).strip().replace(" ", "_")


if __name__ == "__main__":
    main()
