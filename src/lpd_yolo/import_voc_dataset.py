import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC XML annotations to YOLO train/val dataset layout."
    )
    parser.add_argument("--images-dir", required=True, help="Directory containing source images")
    parser.add_argument("--xml-dir", required=True, help="Directory containing Pascal VOC XML files")
    parser.add_argument("--output", default="datasets/license_plate", help="YOLO dataset output directory")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--class-name", default="number_plate", help="VOC class name to keep")
    parser.add_argument("--reset", action="store_true", help="Reset output images/labels folders first")
    return parser.parse_args()


def ensure_dirs(out: Path) -> None:
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def voc_box_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, w: float, h: float) -> tuple[float, float, float, float]:
    xc = ((xmin + xmax) / 2.0) / w
    yc = ((ymin + ymax) / 2.0) / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return clamp(xc, 0.0, 1.0), clamp(yc, 0.0, 1.0), clamp(bw, 0.0, 1.0), clamp(bh, 0.0, 1.0)


def parse_xml(xml_path: Path, class_name: str) -> tuple[str, list[str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    if not filename:
        raise ValueError(f"Missing <filename> in {xml_path}")

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    width = float(size.findtext("width") or 0)
    height = float(size.findtext("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size in {xml_path}")

    labels: list[str] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        if name != class_name:
            continue

        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = float(box.findtext("xmin") or 0)
        ymin = float(box.findtext("ymin") or 0)
        xmax = float(box.findtext("xmax") or 0)
        ymax = float(box.findtext("ymax") or 0)
        if xmax <= xmin or ymax <= ymin:
            continue

        xc, yc, bw, bh = voc_box_to_yolo(xmin, ymin, xmax, ymax, width, height)
        if bw <= 0.0 or bh <= 0.0:
            continue

        labels.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return filename, labels


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir)
    xml_dir = Path(args.xml_dir)
    out_dir = Path(args.output)

    if not images_dir.exists() or not xml_dir.exists():
        raise FileNotFoundError("--images-dir or --xml-dir not found")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    if args.reset:
        for part in [out_dir / "images", out_dir / "labels"]:
            if part.exists():
                shutil.rmtree(part)

    ensure_dirs(out_dir)

    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        raise ValueError("No XML files found in --xml-dir")

    samples: list[tuple[Path, list[str]]] = []
    skipped = 0

    for xml in xml_files:
        try:
            filename, labels = parse_xml(xml, args.class_name)
        except Exception:
            skipped += 1
            continue

        if not labels:
            skipped += 1
            continue

        img_path = images_dir / filename
        if not img_path.exists():
            skipped += 1
            continue

        samples.append((img_path, labels))

    if not samples:
        raise ValueError("No valid image/annotation pairs found")

    random.seed(args.seed)
    random.shuffle(samples)

    val_count = max(1, int(len(samples) * args.val_ratio))
    val_names = {img_path.name for img_path, _ in samples[:val_count]}

    train_n = 0
    val_n = 0

    for img_path, labels in samples:
        split = "val" if img_path.name in val_names else "train"
        if split == "train":
            train_n += 1
        else:
            val_n += 1

        out_img = out_dir / "images" / split / img_path.name
        out_lbl = out_dir / "labels" / split / f"{img_path.stem}.txt"

        shutil.copy2(img_path, out_img)
        out_lbl.write_text("\n".join(labels) + "\n", encoding="utf-8")

    print(f"Imported samples: {len(samples)}")
    print(f"Train: {train_n}, Val: {val_n}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
