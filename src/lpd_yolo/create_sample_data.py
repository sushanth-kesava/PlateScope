from pathlib import Path

import cv2
import numpy as np


def write_sample(split: str, name: str, box: tuple[int, int, int, int]) -> None:
    root = Path("datasets/license_plate")
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    x1, y1, x2, y2 = box
    img = np.full((480, 640, 3), 35, dtype=np.uint8)
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), -1)
    cv2.putText(img, "KA01AB1234", (x1 + 8, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    cv2.imwrite(str(root / "images" / split / name), img)

    xc = ((x1 + x2) / 2) / 640
    yc = ((y1 + y2) / 2) / 480
    w = (x2 - x1) / 640
    h = (y2 - y1) / 480
    label = f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
    (root / "labels" / split / f"{Path(name).stem}.txt").write_text(label, encoding="utf-8")


def main() -> None:
    write_sample("train", "sample_train.jpg", (220, 90, 420, 150))
    write_sample("val", "sample_val.jpg", (200, 100, 400, 160))
    print("Sample dataset created under datasets/license_plate")


if __name__ == "__main__":
    main()
