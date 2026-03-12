import argparse
import csv
import re
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    import pytesseract
except ImportError as exc:
    raise ImportError(
        "pytesseract is required for OCR. Install with: pip install pytesseract"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect license plates and extract text with OCR."
    )
    parser.add_argument(
        "--weights",
        default="runs/train/license_plate_detector/weights/best.pt",
        help="Path to model weights",
    )
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument(
        "--output",
        default="runs/ocr/plate_text.csv",
        help="Output CSV path for OCR results",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default="",
        help="Optional explicit path to tesseract binary",
    )
    return parser.parse_args()


def image_files(source: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source.is_file() and source.suffix.lower() in exts:
        return [source]
    if source.is_dir():
        return sorted([p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    return []


def clean_text(text: str) -> str:
    # Keep common plate characters only.
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9-]", "", text)
    return text


def ocr_plate(crop_bgr) -> str:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    text = pytesseract.image_to_string(gray, config=config)
    return clean_text(text)


def main() -> None:
    args = parse_args()
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source path not found: {source}")

    images = image_files(source)
    if not images:
        raise ValueError("No supported image files found.")

    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    rows: list[tuple[str, int, str]] = []

    for image_path in images:
        result = model.predict(source=str(image_path), conf=args.conf, verbose=False)[0]

        if result.boxes is None or len(result.boxes) == 0:
            rows.append((str(image_path), -1, ""))
            continue

        boxes = result.boxes.xyxy.cpu().tolist()
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(max(0, v)) for v in box]
            crop = result.orig_img[y1:y2, x1:x2]
            text = ocr_plate(crop) if crop.size > 0 else ""
            rows.append((str(image_path), idx, text))

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "plate_index", "plate_text"])
        writer.writerows(rows)

    print(f"OCR complete. Processed images: {len(images)}")
    print(f"Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()
