import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split raw images/labels into YOLO train/val directories."
    )
    parser.add_argument("--images", required=True, help="Raw images directory")
    parser.add_argument("--labels", required=True, help="Raw labels directory")
    parser.add_argument(
        "--output",
        default="datasets/license_plate",
        help="Dataset output directory",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def ensure_dirs(output: Path) -> None:
    (output / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output / "labels" / "val").mkdir(parents=True, exist_ok=True)


def find_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]


def copy_pair(image_path: Path, labels_dir: Path, output: Path, split: str) -> bool:
    label_path = labels_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        return False

    shutil.copy2(image_path, output / "images" / split / image_path.name)
    shutil.copy2(label_path, output / "labels" / split / label_path.name)
    return True


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.output)

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError("Input image/label directories do not exist.")
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")

    ensure_dirs(output_dir)
    images = find_images(images_dir)
    if not images:
        raise ValueError("No image files found in --images directory.")

    random.seed(args.seed)
    random.shuffle(images)
    val_count = int(len(images) * args.val_ratio)
    val_set = set(images[:val_count])

    copied = {"train": 0, "val": 0}
    missing_labels = 0

    for img in images:
        split = "val" if img in val_set else "train"
        ok = copy_pair(img, labels_dir, output_dir, split)
        if ok:
            copied[split] += 1
        else:
            missing_labels += 1

    print(f"Train images: {copied['train']}")
    print(f"Val images: {copied['val']}")
    print(f"Skipped images (missing labels): {missing_labels}")


if __name__ == "__main__":
    main()
