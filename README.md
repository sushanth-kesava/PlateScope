# License Plate Detection with YOLO

Python project for detecting vehicle number plates using YOLO bounding boxes for automated vehicle monitoring and smart city applications.

## Features

- Train YOLO detector on license plate annotations.
- Run inference on images, videos, and webcam streams.
- Prepare train/val dataset split in YOLO format.
- Validate label quality before training.
- Track plate detections in video streams and count line crossings.
- Extract license plate text with OCR.
- Evaluate trained models with precision/recall/mAP report.
- Use the trained model from a browser-based upload website.

## Project Structure

```text
.
|-- datasets/
|   `-- license_plate/
|       |-- data.yaml
|       |-- images/
|       |   |-- train/
|       |   `-- val/
|       `-- labels/
|           |-- train/
|           `-- val/
|-- src/
|   `-- lpd_yolo/
|       |-- __init__.py
|       |-- create_sample_data.py
|       |-- detect.py
|       |-- evaluate.py
|       |-- import_voc_dataset.py
|       |-- ocr_pipeline.py
|       |-- prepare_dataset.py
|       |-- predict_and_track.py
|       |-- train.py
|       |-- validate_labels.py
|       `-- webapp.py
|-- requirements.txt
`-- README.md
```

## 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Prepare Dataset (YOLO Format)

Expected YOLO label row format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates must be normalized to `[0, 1]`.

If you have raw images and matching labels:

```bash
python -m src.lpd_yolo.prepare_dataset \
  --images /path/to/raw/images \
  --labels /path/to/raw/labels \
  --output datasets/license_plate \
  --val-ratio 0.2
```

Validate annotations before training:

```bash
python -m src.lpd_yolo.validate_labels --labels datasets/license_plate/labels
```

## 3. Train the License Plate Detector

```bash
python -m src.lpd_yolo.train \
  --data datasets/license_plate/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

Best model path after training is typically:

```text
runs/train/license_plate_detector/weights/best.pt
```

## 4. Run Detection

Image or video file:

```bash
python -m src.lpd_yolo.detect \
  --weights runs/train/license_plate_detector/weights/best.pt \
  --source /path/to/image_or_video.jpg \
  --save-txt
```

Webcam stream:

```bash
python -m src.lpd_yolo.detect \
  --weights runs/train/license_plate_detector/weights/best.pt \
  --source 0
```

Predictions are saved under `runs/detect/license_plate_predictions`.

## Notes

- Start with `yolov8n.pt` for speed, then move to larger models (`yolov8s.pt`, `yolov8m.pt`) for better accuracy.
- Accuracy depends on data quality, camera angle diversity, and annotation consistency.
- For production smart-city pipelines, integrate this detector with OCR for plate text extraction.

## 5. Track Plates in Video (Flow Monitoring)

This script uses YOLO tracking and counts unique tracked license plates that cross a horizontal line.

```bash
python -m src.lpd_yolo.predict_and_track \
  --weights runs/train/license_plate_detector/weights/best.pt \
  --source /path/to/traffic_video.mp4 \
  --line-y 300 \
  --output runs/track/license_plate_tracking.mp4
```

Optional live preview:

```bash
python -m src.lpd_yolo.predict_and_track \
  --weights runs/train/license_plate_detector/weights/best.pt \
  --source 0 \
  --show
```

## 6. OCR for Plate Text Extraction

Install Tesseract binary on macOS:

```bash
brew install tesseract
```

Run OCR pipeline on an image or image directory:

```bash
python -m src.lpd_yolo.ocr_pipeline \
  --weights runs/train/license_plate_detector/weights/best.pt \
  --source /path/to/images \
  --output runs/ocr/plate_text.csv
```

If your Tesseract binary is in a custom location:

```bash
python -m src.lpd_yolo.ocr_pipeline \
  --weights runs/train/license_plate_detector/weights/best.pt \
  --source /path/to/images \
  --tesseract-cmd /opt/homebrew/bin/tesseract
```

## 7. Model Evaluation Report

Generate validation metrics report (precision, recall, mAP):

```bash
python -m src.lpd_yolo.evaluate \
  --weights runs/train/license_plate_detector/weights/best.pt \
  --data datasets/license_plate/data.yaml \
  --output runs/eval/license_plate_metrics.json
```

## 8. Website UI

Start the browser app with your trained weights:

```bash
python -m src.lpd_yolo.webapp
```

Then open:

```text
http://127.0.0.1:5000
```
Notes:

- The website uses `runs/train/kaggle_lp_full_50/weights/best.pt` by default.
- To use another checkpoint, set `LPD_WEIGHTS` before launch.
- OCR in the website is optional and requires the system `tesseract` binary.
# PlateScope
