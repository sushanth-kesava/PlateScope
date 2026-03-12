import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate YOLO label files for license plate annotations."
    )
    parser.add_argument("--labels", required=True, help="Directory with YOLO .txt labels")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1,
        help="Expected number of classes",
    )
    return parser.parse_args()


def valid_row(parts: list[str], num_classes: int) -> bool:
    if len(parts) != 5:
        return False

    try:
        cls = int(parts[0])
        vals = [float(x) for x in parts[1:]]
    except ValueError:
        return False

    if cls < 0 or cls >= num_classes:
        return False

    x_center, y_center, width, height = vals
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        return False
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        return False

    return True


def main() -> None:
    args = parse_args()
    labels_dir = Path(args.labels)

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    files = sorted(labels_dir.rglob("*.txt"))
    if not files:
        raise ValueError("No .txt label files found.")

    bad_files = []
    bad_lines = 0

    for file_path in files:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            parts = line.strip().split()
            if not valid_row(parts, args.num_classes):
                bad_files.append(f"{file_path}:{line_no}")
                bad_lines += 1

    print(f"Checked files: {len(files)}")
    if bad_lines == 0:
        print("All labels look valid.")
    else:
        print(f"Invalid label rows: {bad_lines}")
        for item in bad_files[:50]:
            print(item)


if __name__ == "__main__":
    main()
