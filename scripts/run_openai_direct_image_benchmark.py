#!/usr/bin/env python3
"""
Generate about 100 synthetic waste images, send them to the deployed classifier,
and write a reproducible benchmark report.

This is a functional regression benchmark for the deployed OpenAI-direct API.
It is not a substitute for a real-world photo dataset.
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
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageFilter


DEFAULT_ENDPOINT = "https://bingo-sorter.vercel.app/api/classify"
BIN_ORDER = ["green", "yellow", "blue", "red"]
BIN_LABELS = {
    "green": "Food / Organic waste",
    "yellow": "Recyclable waste",
    "blue": "General waste",
    "red": "Hazardous waste",
}
ITEMS = {
    "green": ["banana_peel", "apple_core", "leftover_rice", "vegetable_scraps", "coffee_grounds"],
    "yellow": ["pet_bottle", "glass_bottle", "aluminum_can", "cardboard_box", "newspaper"],
    "blue": ["snack_wrapper", "plastic_bag", "dirty_tissue", "foam_box", "face_mask"],
    "red": ["aa_battery", "light_bulb", "aerosol_can", "medicine_blister", "ewaste_phone"],
}


@dataclass
class Case:
    case_id: str
    expected_bin: str
    item_type: str
    image_path: Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--out-dir", default="test-results/openai-direct-100")
    parser.add_argument("--cases-per-bin", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=100)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--skip-existing-images", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    cases = build_cases(images_dir, args.cases_per_bin, rng, args.skip_existing_images)
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
                f"expected={result['expected_bin']} predicted={result.get('predicted_bin') or '-'} "
                f"conf={result.get('confidence') if result.get('confidence') is not None else '-'} "
                f"latency={result['latency_ms']}ms",
                flush=True,
            )

    results.sort(key=lambda item: item["case_id"])
    elapsed = time.time() - started
    summary = build_summary(results, elapsed, args)

    write_json(out_dir / "results.json", results)
    write_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "results.csv", results)
    write_report(out_dir / "report.md", summary, results, cases, args)

    print("\nReport written:")
    print(f"- {out_dir / 'report.md'}")
    print(f"- {out_dir / 'results.csv'}")
    print(f"- {out_dir / 'results.json'}")
    print(f"- {out_dir / 'summary.json'}")


def build_cases(images_dir: Path, cases_per_bin: int, rng: random.Random, skip_existing: bool) -> list[Case]:
    cases: list[Case] = []
    serial = 1
    for expected_bin in BIN_ORDER:
        item_types = ITEMS[expected_bin]
        for index in range(cases_per_bin):
            item_type = item_types[index % len(item_types)]
            case_id = f"{serial:03d}_{expected_bin}_{item_type}_{index + 1:02d}"
            image_path = images_dir / f"{case_id}.png"
            if not (skip_existing and image_path.exists()):
                image = draw_waste_image(expected_bin, item_type, index, rng)
                image.save(image_path, format="PNG", optimize=True)
            cases.append(Case(case_id, expected_bin, item_type, image_path))
            serial += 1
    return cases


def draw_waste_image(expected_bin: str, item_type: str, variant: int, rng: random.Random) -> Image.Image:
    width, height = 640, 640
    bg = rng.choice(["#f8fafc", "#f7f3e8", "#eef6ff", "#f6f8ef", "#f9f5f7"])
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    add_noise_background(draw, width, height, rng)

    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    drawer = DRAWERS[item_type]
    drawer(d, rng, variant)

    angle = rng.uniform(-7, 7)
    layer = layer.rotate(angle, resample=Image.Resampling.BICUBIC, center=(width // 2, height // 2))
    image = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))
    return image


def add_noise_background(draw: ImageDraw.ImageDraw, width: int, height: int, rng: random.Random) -> None:
    for _ in range(18):
        x = rng.randint(0, width)
        y = rng.randint(0, height)
        radius = rng.randint(18, 60)
        color = rng.choice(["#e2e8f0", "#e5e7eb", "#fef3c7", "#dcfce7", "#dbeafe"])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill: str, size: int) -> None:
    fnt = font(size)
    bbox = draw.textbbox((0, 0), text, font=fnt)
    x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) / 2
    y = box[1] + (box[3] - box[1] - (bbox[3] - bbox[1])) / 2
    draw.text((x, y), text, fill=fill, font=fnt)


def draw_banana_peel(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.arc((145, 190, 500, 520), 205, 335, fill="#facc15", width=42)
    draw.arc((175, 210, 465, 490), 205, 335, fill="#fde68a", width=24)
    draw.polygon([(305, 250), (330, 175), (354, 253)], fill="#facc15", outline="#a16207")
    draw.line((318, 250, 325, 178, 345, 254), fill="#92400e", width=4)
    draw.ellipse((210, 385, 250, 425), fill="#7c2d12")
    text_center(draw, (180, 540, 460, 600), "BANANA PEEL", "#3f3f46", 30)


def draw_apple_core(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.ellipse((210, 160, 430, 430), fill="#fca5a5", outline="#b91c1c", width=6)
    draw.rectangle((275, 190, 365, 405), fill="#fde68a")
    draw.ellipse((270, 138, 370, 210), fill="#f8fafc")
    draw.ellipse((270, 385, 370, 455), fill="#f8fafc")
    draw.line((320, 150, 335, 95), fill="#854d0e", width=8)
    draw.ellipse((338, 86, 390, 122), fill="#65a30d")
    for y in [250, 290, 330]:
        draw.ellipse((310, y, 326, y + 24), fill="#7c2d12")
    text_center(draw, (190, 515, 450, 590), "APPLE CORE", "#3f3f46", 32)


def draw_leftover_rice(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.ellipse((150, 360, 490, 510), fill="#94a3b8", outline="#475569", width=6)
    draw.ellipse((175, 250, 465, 430), fill="#fef3c7", outline="#d97706", width=5)
    for _ in range(80):
        x = rng.randint(205, 435)
        y = rng.randint(280, 405)
        draw.ellipse((x, y, x + 7, y + 4), fill=rng.choice(["#fff7ed", "#fde68a", "#fefce8"]))
    for _ in range(12):
        x = rng.randint(215, 420)
        y = rng.randint(285, 395)
        draw.rectangle((x, y, x + 18, y + 9), fill=rng.choice(["#22c55e", "#ef4444", "#f97316"]))
    text_center(draw, (170, 525, 470, 590), "LEFTOVER RICE", "#3f3f46", 30)


def draw_vegetable_scraps(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    for _ in range(18):
        cx = rng.randint(170, 470)
        cy = rng.randint(190, 440)
        rx = rng.randint(28, 60)
        ry = rng.randint(16, 38)
        color = rng.choice(["#16a34a", "#22c55e", "#84cc16", "#65a30d"])
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color, outline="#166534", width=3)
        draw.line((cx - rx + 8, cy, cx + rx - 8, cy), fill="#dcfce7", width=2)
    draw.rectangle((185, 440, 455, 480), fill="#a3e635", outline="#4d7c0f", width=4)
    text_center(draw, (165, 520, 480, 590), "VEGETABLE SCRAPS", "#3f3f46", 28)


def draw_coffee_grounds(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.ellipse((165, 350, 475, 500), fill="#b45309", outline="#78350f", width=5)
    draw.ellipse((190, 230, 450, 410), fill="#7c2d12", outline="#451a03", width=5)
    for _ in range(140):
        x = rng.randint(205, 435)
        y = rng.randint(250, 395)
        r = rng.randint(2, 5)
        draw.ellipse((x, y, x + r, y + r), fill=rng.choice(["#451a03", "#713f12", "#92400e"]))
    draw.arc((450, 320, 540, 430), -90, 110, fill="#78350f", width=10)
    text_center(draw, (175, 525, 465, 590), "COFFEE GROUNDS", "#3f3f46", 28)


def draw_pet_bottle(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.rounded_rectangle((280, 80, 360, 145), radius=15, fill="#94a3b8", outline="#475569", width=4)
    draw.rounded_rectangle((210, 130, 430, 545), radius=70, fill="#dff7ff", outline="#0ea5e9", width=8)
    draw.rounded_rectangle((245, 260, 395, 360), radius=18, fill="#ffffff", outline="#0369a1", width=4)
    draw.line((255, 170, 275, 510), fill="#ffffff", width=9)
    draw.line((390, 175, 405, 510), fill="#bae6fd", width=6)
    text_center(draw, (250, 266, 390, 314), "WATER", "#0369a1", 27)
    text_center(draw, (260, 315, 380, 352), "PET", "#0369a1", 24)


def draw_glass_bottle(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.rounded_rectangle((285, 80, 355, 210), radius=18, fill="#bbf7d0", outline="#15803d", width=6)
    draw.rounded_rectangle((220, 190, 420, 540), radius=55, fill="#dcfce7", outline="#15803d", width=8)
    draw.rounded_rectangle((250, 300, 390, 375), radius=10, fill="#f8fafc", outline="#166534", width=4)
    text_center(draw, (255, 305, 385, 365), "GLASS", "#166534", 25)


def draw_aluminum_can(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.ellipse((220, 120, 420, 190), fill="#e5e7eb", outline="#6b7280", width=5)
    draw.rectangle((220, 155, 420, 500), fill="#d1d5db", outline="#6b7280", width=5)
    draw.ellipse((220, 465, 420, 535), fill="#e5e7eb", outline="#6b7280", width=5)
    draw.rectangle((240, 250, 400, 360), fill="#ef4444")
    text_center(draw, (245, 260, 395, 350), "SODA CAN", "#ffffff", 26)
    draw.arc((285, 122, 355, 170), 0, 360, fill="#374151", width=3)


def draw_cardboard_box(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.polygon([(180, 210), (330, 145), (470, 210), (320, 280)], fill="#d97706", outline="#92400e")
    draw.polygon([(180, 210), (320, 280), (320, 520), (180, 445)], fill="#b45309", outline="#92400e")
    draw.polygon([(470, 210), (320, 280), (320, 520), (470, 445)], fill="#c2410c", outline="#92400e")
    draw.line((330, 145, 320, 280, 320, 520), fill="#78350f", width=4)
    text_center(draw, (215, 365, 425, 445), "CARDBOARD", "#fff7ed", 27)


def draw_newspaper(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.rectangle((160, 120, 480, 510), fill="#f8fafc", outline="#64748b", width=6)
    draw.line((320, 135, 320, 495), fill="#cbd5e1", width=3)
    text_center(draw, (180, 140, 460, 200), "DAILY NEWS", "#111827", 30)
    for x0, x1 in [(180, 300), (340, 460)]:
        for y in range(230, 470, 36):
            draw.line((x0, y, x1, y), fill="#64748b", width=4)
    draw.rectangle((190, 235, 295, 315), fill="#bfdbfe", outline="#1d4ed8", width=3)


def draw_snack_wrapper(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.polygon([(130, 230), (500, 170), (535, 415), (165, 480)], fill="#f97316", outline="#9a3412")
    draw.polygon([(130, 230), (180, 245), (165, 480)], fill="#fed7aa")
    draw.polygon([(500, 170), (485, 400), (535, 415)], fill="#fed7aa")
    text_center(draw, (210, 260, 470, 380), "CHIPS", "#ffffff", 52)
    text_center(draw, (220, 380, 455, 430), "EMPTY WRAPPER", "#7c2d12", 22)


def draw_plastic_bag(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.arc((230, 110, 320, 245), 180, 360, fill="#64748b", width=10)
    draw.arc((320, 110, 410, 245), 180, 360, fill="#64748b", width=10)
    draw.rounded_rectangle((190, 205, 450, 515), radius=30, fill="#e0f2fe", outline="#64748b", width=6)
    draw.line((225, 260, 420, 460), fill="#f8fafc", width=8)
    draw.line((420, 260, 225, 460), fill="#f8fafc", width=8)
    text_center(draw, (220, 520, 420, 585), "PLASTIC BAG", "#334155", 28)


def draw_dirty_tissue(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    for _ in range(7):
        x = rng.randint(150, 390)
        y = rng.randint(180, 410)
        draw.ellipse((x, y, x + rng.randint(90, 160), y + rng.randint(55, 120)), fill="#f8fafc", outline="#cbd5e1", width=4)
    for _ in range(12):
        x = rng.randint(230, 410)
        y = rng.randint(250, 430)
        draw.ellipse((x, y, x + 18, y + 12), fill=rng.choice(["#a16207", "#92400e", "#ef4444"]))
    text_center(draw, (190, 525, 450, 590), "DIRTY TISSUE", "#334155", 30)


def draw_foam_box(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.rounded_rectangle((150, 220, 500, 485), radius=30, fill="#f8fafc", outline="#94a3b8", width=8)
    draw.line((165, 300, 485, 300), fill="#cbd5e1", width=5)
    draw.line((235, 300, 235, 470), fill="#cbd5e1", width=4)
    draw.ellipse((270, 335, 450, 430), fill="#fed7aa", outline="#ea580c", width=4)
    text_center(draw, (180, 520, 470, 585), "FOAM FOOD BOX", "#334155", 28)


def draw_face_mask(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.rounded_rectangle((170, 220, 470, 420), radius=45, fill="#bae6fd", outline="#0369a1", width=6)
    for y in [265, 315, 365]:
        draw.line((205, y, 435, y), fill="#0ea5e9", width=5)
    draw.arc((105, 250, 210, 400), 80, 280, fill="#64748b", width=7)
    draw.arc((430, 250, 535, 400), -100, 100, fill="#64748b", width=7)
    text_center(draw, (210, 500, 430, 575), "USED MASK", "#334155", 32)


def draw_aa_battery(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.rounded_rectangle((145, 250, 485, 400), radius=35, fill="#111827", outline="#374151", width=7)
    draw.rectangle((465, 285, 520, 365), fill="#e5e7eb", outline="#6b7280", width=5)
    draw.rectangle((180, 270, 315, 380), fill="#facc15")
    text_center(draw, (185, 278, 310, 335), "AA", "#111827", 42)
    text_center(draw, (320, 290, 455, 360), "BATTERY", "#f8fafc", 27)
    text_center(draw, (190, 465, 450, 530), "USED BATTERY", "#334155", 30)


def draw_light_bulb(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.ellipse((210, 115, 430, 345), fill="#fef3c7", outline="#d97706", width=7)
    draw.rectangle((270, 330, 370, 440), fill="#cbd5e1", outline="#64748b", width=6)
    for y in [350, 378, 406]:
        draw.line((275, y, 365, y), fill="#64748b", width=5)
    draw.line((260, 210, 320, 270, 380, 210), fill="#f59e0b", width=5)
    text_center(draw, (200, 500, 440, 570), "LIGHT BULB", "#334155", 33)


def draw_aerosol_can(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.rectangle((270, 85, 370, 145), fill="#64748b", outline="#334155", width=5)
    draw.ellipse((220, 125, 420, 190), fill="#e5e7eb", outline="#475569", width=5)
    draw.rectangle((220, 155, 420, 520), fill="#ef4444", outline="#991b1b", width=7)
    draw.ellipse((220, 485, 420, 555), fill="#dc2626", outline="#991b1b", width=5)
    text_center(draw, (245, 245, 395, 330), "SPRAY", "#ffffff", 38)
    text_center(draw, (250, 350, 390, 410), "AEROSOL", "#ffffff", 24)
    draw.polygon([(330, 430), (350, 470), (310, 470)], fill="#facc15", outline="#7c2d12")


def draw_medicine_blister(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.rounded_rectangle((150, 160, 490, 470), radius=24, fill="#e5e7eb", outline="#64748b", width=7)
    for row in range(3):
        for col in range(4):
            x = 190 + col * 75
            y = 205 + row * 78
            draw.ellipse((x, y, x + 46, y + 46), fill="#f8fafc", outline="#94a3b8", width=4)
            if (row + col + variant) % 3 == 0:
                draw.line((x + 8, y + 23, x + 38, y + 23), fill="#ef4444", width=4)
    text_center(draw, (185, 495, 455, 570), "MEDICINE PACK", "#334155", 28)


def draw_ewaste_phone(draw: ImageDraw.ImageDraw, rng: random.Random, variant: int) -> None:
    draw.rounded_rectangle((210, 100, 430, 520), radius=35, fill="#111827", outline="#374151", width=8)
    draw.rectangle((235, 145, 405, 440), fill="#1e293b", outline="#475569", width=4)
    draw.circle((320, 480), 18, fill="#475569")
    draw.line((245, 180, 395, 390), fill="#22c55e", width=5)
    draw.line((395, 180, 245, 390), fill="#ef4444", width=5)
    text_center(draw, (240, 250, 400, 330), "E-WASTE", "#e2e8f0", 25)


DRAWERS = {
    "banana_peel": draw_banana_peel,
    "apple_core": draw_apple_core,
    "leftover_rice": draw_leftover_rice,
    "vegetable_scraps": draw_vegetable_scraps,
    "coffee_grounds": draw_coffee_grounds,
    "pet_bottle": draw_pet_bottle,
    "glass_bottle": draw_glass_bottle,
    "aluminum_can": draw_aluminum_can,
    "cardboard_box": draw_cardboard_box,
    "newspaper": draw_newspaper,
    "snack_wrapper": draw_snack_wrapper,
    "plastic_bag": draw_plastic_bag,
    "dirty_tissue": draw_dirty_tissue,
    "foam_box": draw_foam_box,
    "face_mask": draw_face_mask,
    "aa_battery": draw_aa_battery,
    "light_bulb": draw_light_bulb,
    "aerosol_can": draw_aerosol_can,
    "medicine_blister": draw_medicine_blister,
    "ewaste_phone": draw_ewaste_phone,
}


def classify_case(case: Case, endpoint: str, timeout: int, retries: int, raw_dir: Path) -> dict[str, Any]:
    started = time.time()
    payload = {
        "image": image_to_data_url(case.image_path),
        "mimeType": "image/png",
        "fileName": f"{case.case_id}.png",
        "source": "test",
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
                time.sleep(1.3 * (attempt + 1))
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
        "item_type": case.item_type,
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
    return f"data:image/png;base64,{encoded}"


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


def build_summary(results: list[dict[str, Any]], elapsed: float, args: argparse.Namespace) -> dict[str, Any]:
    total = len(results)
    errors = [item for item in results if item.get("error")]
    valid = [item for item in results if not item.get("error")]
    correct = [item for item in valid if item.get("correct")]
    confidence_values = [float(item["confidence"]) for item in valid if isinstance(item.get("confidence"), (int, float))]
    latencies = [int(item["latency_ms"]) for item in results if isinstance(item.get("latency_ms"), int)]

    by_expected: dict[str, dict[str, Any]] = {}
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

    confusion = {
        expected: {predicted: 0 for predicted in BIN_ORDER + ["error", "missing"]}
        for expected in BIN_ORDER
    }
    for item in results:
        expected = item["expected_bin"]
        if item.get("error"):
            predicted = "error"
        else:
            predicted = item.get("predicted_bin") or "missing"
        confusion[expected][predicted] += 1

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "endpoint": args.endpoint,
        "seed": args.seed,
        "total_cases": total,
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
        "confusion_matrix": confusion,
    }


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator * 100, 2)


def percentile(values: list[int], pct: int) -> int:
    if not values:
        return 0
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
            "cases": [
                {
                    "case_id": case.case_id,
                    "expected_bin": case.expected_bin,
                    "item_type": case.item_type,
                    "image_path": str(case.image_path),
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
        "item_type",
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


def write_report(
    path: Path,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    cases: list[Case],
    args: argparse.Namespace,
) -> None:
    wrong = [item for item in results if not item.get("correct")]
    low_conf = [
        item for item in results
        if isinstance(item.get("confidence"), (int, float)) and float(item["confidence"]) < 80
    ]
    manual = [item for item in results if item.get("needs_manual_check") is True]

    lines = [
        "# OpenAI Direct Waste Classification Benchmark",
        "",
        "## Summary",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Endpoint: `{summary['endpoint']}`",
        f"- Test type: synthetic functional benchmark, 4 bins x {args.cases_per_bin} cases",
        f"- Total cases: **{summary['total_cases']}**",
        f"- Valid responses: **{summary['valid_responses']}**",
        f"- Errors: **{summary['error_count']}**",
        f"- Correct: **{summary['correct_count']}**",
        f"- Accuracy: **{summary['accuracy']}%**",
        f"- Accuracy excluding transport/API errors: **{summary['accuracy_excluding_errors']}%**",
        f"- Average confidence: **{summary['avg_confidence']}**",
        f"- Median confidence: **{summary['median_confidence']}**",
        f"- Average latency: **{summary['avg_latency_ms']} ms**",
        f"- P95 latency: **{summary['p95_latency_ms']} ms**",
        f"- Wall time: **{summary['elapsed_seconds']} s** with concurrency `{summary['concurrency']}`",
        "",
        "Important: this benchmark uses generated images, so it is good for regression testing the API path and obvious visual categories. It does not prove real-world camera/photo accuracy.",
        "",
        "## Per-Bin Accuracy",
        "",
        "| Expected bin | Cases | Correct | Accuracy | Avg confidence | Manual checks |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for bin_key in BIN_ORDER:
        row = summary["by_expected"][bin_key]
        lines.append(
            f"| {bin_key} ({BIN_LABELS[bin_key]}) | {row['total']} | {row['correct']} | "
            f"{row['accuracy']}% | {row['avg_confidence']} | {row['manual_check_count']} |"
        )

    lines.extend([
        "",
        "## Confusion Matrix",
        "",
        "| Expected \\ Predicted | green | yellow | blue | red | error | missing |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for expected in BIN_ORDER:
        row = summary["confusion_matrix"][expected]
        lines.append(
            f"| {expected} | {row['green']} | {row['yellow']} | {row['blue']} | "
            f"{row['red']} | {row['error']} | {row['missing']} |"
        )

    lines.extend([
        "",
        "## Wrong Or Error Cases",
        "",
    ])
    if wrong:
        lines.extend([
            "| Case | Item | Expected | Predicted | Confidence | Error | Evidence |",
            "| --- | --- | --- | --- | ---: | --- | --- |",
        ])
        for item in wrong:
            lines.append(
                f"| `{item['case_id']}` | {item['item_type']} | {item['expected_bin']} | "
                f"{item.get('predicted_bin') or '-'} | {item.get('confidence') or '-'} | "
                f"{safe_md(item.get('error') or '')} | {safe_md(shorten(item.get('evidence') or '', 220))} |"
            )
    else:
        lines.append("No wrong or error cases.")

    lines.extend([
        "",
        "## Low Confidence Cases (< 80)",
        "",
    ])
    if low_conf:
        lines.extend([
            "| Case | Item | Expected | Predicted | Confidence | Manual check |",
            "| --- | --- | --- | --- | ---: | --- |",
        ])
        for item in low_conf:
            lines.append(
                f"| `{item['case_id']}` | {item['item_type']} | {item['expected_bin']} | "
                f"{item.get('predicted_bin') or '-'} | {item.get('confidence')} | "
                f"{item.get('needs_manual_check')} |"
            )
    else:
        lines.append("No cases below 80 confidence.")

    lines.extend([
        "",
        "## Manual Check Flags",
        "",
        f"- Manual check count: **{len(manual)}**",
        "",
        "## Artifacts",
        "",
        "- `cases.json`: generated test cases and expected labels.",
        "- `images/`: generated PNG inputs.",
        "- `raw/`: raw JSON response for every case.",
        "- `results.csv`: flat table for spreadsheet inspection.",
        "- `results.json`: full flat result list.",
        "- `summary.json`: machine-readable summary.",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def safe_md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
