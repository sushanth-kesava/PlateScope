import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO license plate detection on images, videos, or webcam."
    )
    parser.add_argument(
        "--weights",
        default="runs/train/license_plate_detector/weights/best.pt",
        help="Path to trained model weights",
    )
    parser.add_argument("--source", required=True, help="Path, URL, video file, or webcam index (0)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default="", help="Device like cpu, mps, 0")
    parser.add_argument("--project", default="runs/detect", help="Output project directory")
    parser.add_argument("--name", default="license_plate_predictions", help="Run name")
    parser.add_argument("--save-txt", action="store_true", help="Save detections in YOLO txt format")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped plate detections")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    if isinstance(source, str) and not source.startswith(("http://", "https://")):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {source}")

    model = YOLO(args.weights)
    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_crop=args.save_crop,
        project=args.project,
        name=args.name,
        verbose=False,
    )

    total = sum(len(r.boxes) for r in results)
    print(f"Detection finished. Total license plates detected: {total}")


if __name__ == "__main__":
    main()
