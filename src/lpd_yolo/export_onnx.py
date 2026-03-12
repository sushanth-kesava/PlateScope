import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained license plate detector to ONNX.")
    parser.add_argument(
        "--weights",
        default="runs/train/license_plate_detector/weights/best.pt",
        help="Path to the trained .pt checkpoint",
    )
    parser.add_argument(
        "--output",
        default="models/best.onnx",
        help="Destination path for the exported ONNX model",
    )
    parser.add_argument("--imgsz", type=int, default=960, help="Export image size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    exported_path = Path(model.export(format="onnx", imgsz=args.imgsz, simplify=True))
    shutil.copy2(exported_path, output_path)
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()