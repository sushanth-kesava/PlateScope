import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLO license plate detector on validation data."
    )
    parser.add_argument(
        "--weights",
        default="runs/train/license_plate_detector/weights/best.pt",
        help="Path to trained model weights",
    )
    parser.add_argument("--data", default="datasets/license_plate/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--device", default="", help="Device like cpu, mps, 0")
    parser.add_argument(
        "--output",
        default="runs/eval/license_plate_metrics.json",
        help="Output JSON report path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )

    summary = {
        "weights": args.weights,
        "data": args.data,
        "split": args.split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "mAP75": float(metrics.box.map75),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Evaluation complete.")
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
