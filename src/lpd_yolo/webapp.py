import base64
import json
import os
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
from flask import Flask, render_template, request
from ultralytics import YOLO

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

try:
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = Path(__file__).resolve().parent / "web" / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_METRICS_PATH = PROJECT_ROOT / "runs" / "eval" / "kaggle_lp_updated_47_metrics.json"
WEIGHTS_ENV_VAR = "LPD_WEIGHTS"
DEFAULT_WEIGHT_CANDIDATES = [
    PROJECT_ROOT / "models" / "best.pt",
    PROJECT_ROOT / "runs" / "train" / "license_plate_detector" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "train" / "kaggle_lp_full_50" / "weights" / "best.pt",
]

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
_model: YOLO | None = None
_ocr_reader: Any | None = None


def resolve_weights_path() -> Path:
    env_path = os.getenv(WEIGHTS_ENV_VAR)
    candidates = [Path(env_path).expanduser()] if env_path else []
    candidates.extend(DEFAULT_WEIGHT_CANDIDATES)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched_paths = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Model weights not found. Set LPD_WEIGHTS to a deployed .pt file or bundle one of these paths:\n"
        f"{searched_paths}"
    )


def load_metrics() -> dict[str, Any] | None:
    if not DEFAULT_METRICS_PATH.exists():
        return None
    return json.loads(DEFAULT_METRICS_PATH.read_text(encoding="utf-8"))


def get_model() -> YOLO:
    global _model
    if _model is None:
        weights_path = resolve_weights_path()
        _model = YOLO(str(weights_path))
    return _model


def get_ocr_reader() -> Any | None:
    global _ocr_reader
    if easyocr is None:
        return None
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def encode_image(image_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise ValueError("Failed to encode output image")
    return base64.b64encode(buffer).decode("utf-8")


PLATE_PATTERN_REGEXES = [
    re.compile(r"^[A-Z]{2}\d{1}[A-Z]{1}\d{4}$"),
    re.compile(r"^[A-Z]{2}\d{1}[A-Z]{2}\d{4}$"),
    re.compile(r"^[A-Z]{2}\d{1}[A-Z]{3}\d{4}$"),
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1}\d{4}$"),
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$"),
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{3}\d{4}$"),
]
PLATE_PATTERNS = [
    "LLDLDDDD",
    "LLDLLDDDD",
    "LLDLLLDDDD",
    "LLDDLDDDD",
    "LLDDLLDDDD",
    "LLDDLLLDDDD",
]
LETTER_FIXES = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
DIGIT_FIXES = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7", "A": "4"}


def _clean_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _normalize_to_pattern(text: str, pattern: str) -> str:
    if len(text) != len(pattern):
        return ""

    normalized: list[str] = []
    for char, expected in zip(text, pattern):
        if expected == "L":
            fixed = LETTER_FIXES.get(char, char)
            if not fixed.isalpha():
                return ""
            normalized.append(fixed)
        else:
            fixed = DIGIT_FIXES.get(char, char)
            if not fixed.isdigit():
                return ""
            normalized.append(fixed)
    return "".join(normalized)


def _score_plate_candidate(text: str, confidence: float) -> float:
    if not text:
        return -1.0

    score = confidence
    if 8 <= len(text) <= 11:
        score += 12.0
    if any(char.isalpha() for char in text) and any(char.isdigit() for char in text):
        score += 8.0
    if any(pattern.fullmatch(text) for pattern in PLATE_PATTERN_REGEXES):
        score += 40.0
    if text[:2].isalpha():
        score += 4.0
    if text[-4:].isdigit():
        score += 6.0
    return score


def _choose_best_plate_text(candidates: list[tuple[str, float]]) -> str:
    best_text = ""
    best_score = -1.0

    for raw_text, confidence in candidates:
        cleaned = _clean_plate_text(raw_text)
        if not cleaned:
            continue

        variants = {(cleaned, confidence)}
        for pattern in PLATE_PATTERNS:
            normalized = _normalize_to_pattern(cleaned, pattern)
            if normalized:
                variants.add((normalized, confidence + 5.0))

        for variant, variant_conf in variants:
            score = _score_plate_candidate(variant, variant_conf)
            if score > best_score:
                best_score = score
                best_text = variant

    if best_score < MIN_OCR_SCORE:
        return ""
    if len(best_text) < MIN_ACCEPTED_PLATE_LENGTH:
        return ""
    if not any(char.isalpha() for char in best_text):
        return ""
    if not any(char.isdigit() for char in best_text):
        return ""
    return best_text


def _ocr_candidates(image_gray: np.ndarray) -> list[tuple[str, float]]:
    if pytesseract is None:
        return []

    bordered = cv2.copyMakeBorder(image_gray, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    scale = max(2.5, 96.0 / max(1, bordered.shape[0]))
    enlarged = cv2.resize(bordered, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(enlarged)
    denoised = cv2.bilateralFilter(enhanced, 7, 50, 50)
    sharpened = cv2.filter2D(denoised, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32))
    otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    inverted = cv2.bitwise_not(otsu)

    variants = [enlarged, denoised, sharpened, otsu, adaptive, inverted]
    configs = [
        "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ]

    candidates: list[tuple[str, float]] = []
    for variant in variants:
        for config in configs:
            data = pytesseract.image_to_data(
                variant,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
            tokens = []
            confidences = []
            for text, conf in zip(data.get("text", []), data.get("conf", [])):
                cleaned = _clean_plate_text(text)
                if not cleaned:
                    continue
                tokens.append(cleaned)
                try:
                    conf_value = float(conf)
                except ValueError:
                    conf_value = -1.0
                if conf_value >= 0:
                    confidences.append(conf_value)

            candidate_text = "".join(tokens)
            average_conf = sum(confidences) / len(confidences) if confidences else 0.0
            if candidate_text:
                candidates.append((candidate_text, average_conf))

    return candidates


def _easyocr_candidates(image_bgr: np.ndarray) -> list[tuple[str, float]]:
    reader = get_ocr_reader()
    if reader is None:
        return []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    scale = max(2.5, 96.0 / max(1, gray.shape[0]))
    enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(enlarged)
    binary = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    candidates: list[tuple[str, float]] = []
    for variant in (enlarged, clahe, binary):
        height, width = variant.shape[:2]
        try:
            results = reader.recognize(
                variant,
                horizontal_list=[[0, width, 0, height]],
                free_list=[],
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
        except Exception:
            continue
        for item in results:
            if len(item) < 3:
                continue
            _, text, confidence = item
            cleaned = _clean_plate_text(text)
            if cleaned:
                candidates.append((cleaned, float(confidence) * 100.0))
    return candidates


def extract_plate_text(image_bgr: np.ndarray) -> str:
    if pytesseract is None and easyocr is None:
        return ""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    candidates = _easyocr_candidates(image_bgr)
    candidates.extend(_ocr_candidates(gray))
    return _choose_best_plate_text(candidates)


def draw_detections(image_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    rendered = image_bgr.copy()
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        score = item["confidence"]
        label = f"plate {score:.2f}"

        cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 200, 255), 3)
        cv2.rectangle(rendered, (x1, max(0, y1 - 28)), (x1 + 170, y1), (0, 200, 255), -1)
        cv2.putText(
            rendered,
            label,
            (x1 + 8, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    return rendered


MIN_CONFIDENCE = 0.15      # never show predictions below this, regardless of user slider
MIN_ASPECT_RATIO = 1.5    # width/height – real plates are always wider than tall
MAX_ASPECT_RATIO = 8.0    # discard very long thin strips too
NMS_IOU_THRESHOLD = 0.35  # suppress overlapping boxes above this IoU
MIN_RELATIVE_AREA = 0.003  # reject tiny logo-like detections
MAX_DETECTIONS = 1         # keep only the strongest plate for single-vehicle photos
AUTOMATIC_CONFIDENCE = 0.25
MIN_ACCEPTED_PLATE_LENGTH = 6
MIN_OCR_SCORE = 45.0
PRIMARY_IMGSZ = 1280
SECONDARY_IMGSZ = 960


def _iou(a: list[int], b: list[int]) -> float:
    """Return IoU of two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _nms(boxes_confs: list[tuple[list[int], float]]) -> list[tuple[list[int], float]]:
    """Greedy NMS: keep highest-confidence box, suppress overlapping ones."""
    sorted_items = sorted(boxes_confs, key=lambda x: x[1], reverse=True)
    kept: list[tuple[list[int], float]] = []
    suppressed = [False] * len(sorted_items)
    for i, (box_i, conf_i) in enumerate(sorted_items):
        if suppressed[i]:
            continue
        kept.append((box_i, conf_i))
        for j in range(i + 1, len(sorted_items)):
            if not suppressed[j] and _iou(box_i, sorted_items[j][0]) > NMS_IOU_THRESHOLD:
                suppressed[j] = True
    return kept


def _enhance_for_detection(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return cv2.detailEnhance(enhanced, sigma_s=8, sigma_r=0.15)


def _candidate_score(box: list[int], confidence: float, image_w: int, image_h: int) -> float:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    aspect = width / height
    area_ratio = (width * height) / max(1.0, float(image_w * image_h))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    dx = abs(center_x - (image_w / 2.0)) / max(1.0, image_w / 2.0)
    dy = abs(center_y - (image_h / 2.0)) / max(1.0, image_h / 2.0)
    center_bonus = max(0.0, 1.0 - (0.65 * dx + 0.35 * dy))
    aspect_bonus = max(0.0, 1.0 - abs(aspect - 3.6) / 3.6)
    size_bonus = min(area_ratio / 0.02, 1.0)
    return confidence + 0.12 * center_bonus + 0.08 * aspect_bonus + 0.05 * size_bonus


def _collect_candidate_boxes(model: YOLO, image_bgr: np.ndarray, confidence: float) -> list[tuple[list[int], float]]:
    passes = [
        (image_bgr, PRIMARY_IMGSZ, True),
        (image_bgr, SECONDARY_IMGSZ, False),
        (_enhance_for_detection(image_bgr), SECONDARY_IMGSZ, False),
    ]

    candidates: list[tuple[list[int], float]] = []
    for source_image, imgsz, augment in passes:
        result = model.predict(
            source=source_image,
            conf=confidence,
            imgsz=imgsz,
            augment=augment,
            verbose=False,
        )[0]
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        candidates.extend(([max(0, int(v)) for v in box], float(score)) for box, score in zip(boxes, confs))

    return candidates


def run_inference(image_bytes: bytes) -> dict[str, Any]:
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image")

    image_h, image_w = image.shape[:2]
    image_area = float(image_h * image_w)

    model = get_model()
    effective_conf = max(AUTOMATIC_CONFIDENCE, MIN_CONFIDENCE)
    raw_predictions = _collect_candidate_boxes(model, image, effective_conf)

    if not raw_predictions and effective_conf > MIN_CONFIDENCE:
        raw_predictions = _collect_candidate_boxes(model, image, MIN_CONFIDENCE)

    raw_boxes: list[tuple[list[int], float]] = []
    for box, score in raw_predictions:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if h == 0:
            continue
        aspect = w / h
        area_ratio = (w * h) / image_area if image_area > 0 else 0.0
        # Filter by aspect ratio to reject grilles, logos, square objects
        if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO):
            continue
        if area_ratio < MIN_RELATIVE_AREA:
            continue
        ranked_score = _candidate_score(box, float(score), image_w, image_h)
        raw_boxes.append((box, ranked_score))

    # Apply NMS to remove duplicate/overlapping predictions of the same plate
    kept = _nms(raw_boxes)
    if MAX_DETECTIONS > 0:
        kept = kept[:MAX_DETECTIONS]

    detections: list[dict[str, Any]] = []
    for idx, (box, score) in enumerate(kept, start=1):
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]
        plate_text = extract_plate_text(crop) if crop.size > 0 else ""
        crop_b64 = encode_image(crop) if crop.size > 0 else ""
        detections.append(
            {
                "id": idx,
                "confidence": round(score, 4),
                "box": [x1, y1, x2, y2],
                "plate_text": plate_text,
                "crop_b64": crop_b64,
            }
        )

    plotted = draw_detections(image, detections)

    return {
        "image_b64": encode_image(plotted),
        "detections": detections,
        "count": len(detections),
    }


@app.get("/")
def index():
    return render_template(
        "index.html",
        result=None,
        error=None,
        metrics=load_metrics(),
        model_ready=model_is_ready(),
        weights_path=str(get_current_weights_path()) if model_is_ready() else None,
    )


@app.get("/health")
def health() -> tuple[dict[str, Any], int]:
    return {
        "status": "ok",
        "model_ready": model_is_ready(),
    }, 200


def get_current_weights_path() -> Path | None:
    try:
        return resolve_weights_path()
    except FileNotFoundError:
        return None


def model_is_ready() -> bool:
    return get_current_weights_path() is not None


@app.post("/predict")
def predict():
    file = request.files.get("image")
    metrics = load_metrics()

    if not model_is_ready():
        return render_template(
            "index.html",
            result=None,
            error=(
                "Model weights are not deployed. Set the LPD_WEIGHTS environment variable or add a model file "
                "at models/best.pt before deploying to Vercel."
            ),
            metrics=metrics,
            model_ready=False,
            weights_path=None,
        )

    if file is None or not file.filename:
        return render_template(
            "index.html",
            result=None,
            error="Choose an image to analyze.",
            metrics=metrics,
            model_ready=True,
            weights_path=str(get_current_weights_path()),
        )

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            result=None,
            error="Unsupported file type. Upload JPG, PNG, BMP, or WEBP.",
            metrics=metrics,
            model_ready=True,
            weights_path=str(get_current_weights_path()),
        )

    try:
        result = run_inference(file.read())
    except Exception as exc:
        return render_template(
            "index.html",
            result=None,
            error=str(exc),
            metrics=metrics,
            model_ready=True,
            weights_path=str(get_current_weights_path()),
        )

    return render_template(
        "index.html",
        result=result,
        error=None,
        metrics=metrics,
        model_ready=True,
        weights_path=str(get_current_weights_path()),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
