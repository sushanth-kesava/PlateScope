import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


class LineCounter:
    """Counts unique tracked objects that cross a horizontal line."""

    def __init__(self, y_line: int) -> None:
        self.y_line = y_line
        self.last_y: dict[int, float] = {}
        self.crossed_ids: set[int] = set()

    def update(self, track_id: int, y_center: float) -> None:
        prev = self.last_y.get(track_id)
        if prev is not None:
            crossed = (prev < self.y_line <= y_center) or (prev > self.y_line >= y_center)
            if crossed:
                self.crossed_ids.add(track_id)
        self.last_y[track_id] = y_center

    @property
    def total(self) -> int:
        return len(self.crossed_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track license plates in video and count line crossings."
    )
    parser.add_argument(
        "--weights",
        default="runs/train/license_plate_detector/weights/best.pt",
        help="Path to model weights",
    )
    parser.add_argument("--source", required=True, help="Video file path or webcam index (0)")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--line-y", type=int, default=-1, help="Y-pixel for counting line")
    parser.add_argument(
        "--output",
        default="runs/track/license_plate_tracking.mp4",
        help="Output annotated video path",
    )
    parser.add_argument("--show", action="store_true", help="Show live window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    if isinstance(source, str):
        src_path = Path(source)
        if not src_path.exists():
            raise FileNotFoundError(f"Source not found: {source}")

    model = YOLO(args.weights)
    results = model.track(
        source=source,
        conf=args.conf,
        iou=args.iou,
        tracker=args.tracker,
        stream=True,
        persist=True,
        verbose=False,
    )

    writer = None
    counter = None

    for frame_idx, result in enumerate(results, start=1):
        frame = result.orig_img.copy()
        h, w = frame.shape[:2]

        if counter is None:
            y_line = args.line_y if args.line_y >= 0 else h // 2
            counter = LineCounter(y_line=y_line)

        if writer is None:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                25.0,
                (w, h),
            )

        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()

            for track_id, box in zip(ids, boxes):
                x1, y1, x2, y2 = map(int, box)
                y_center = (y1 + y2) / 2.0
                counter.update(track_id, y_center)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
                cv2.putText(
                    frame,
                    f"id={track_id}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.line(frame, (0, counter.y_line), (w, counter.y_line), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"crossings={counter.total}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"frame={frame_idx}",
            (16, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        if args.show:
            cv2.imshow("License Plate Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    total = counter.total if counter is not None else 0
    print(f"Tracking complete. Total unique crossings: {total}")
    print(f"Saved annotated output to: {args.output}")


if __name__ == "__main__":
    main()
