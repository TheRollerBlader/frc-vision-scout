# Dataset Sample

A 72-frame sample from the synthetic validation split.

## Contents

```
dataset_sample/
├── images/           # 72 synthetic renders (1280×720 JPEG q80)
│   └── val_frame_XXXXXX.jpg
├── labels/           # YOLO format labels
│   └── val_frame_XXXXXX.txt
├── data.yaml         # YOLO dataset config
├── SAMPLE_INFO.txt   # Sampling methodology
└── README.md
```

## Sample Details

| Property | Value |
|----------|-------|
| Images | 72 (all validation split) |
| Resolution | 1280 × 720 |
| Format | JPEG quality 80 |
| Size | ~260 KB each |
| Seed | `random.Random(42)` |

## Label Format

YOLO normalized coordinates:
```
<class> <cx> <cy> <width> <height>
```

Classes:
- `0` — bumper_red
- `1` — bumper_blue
- `2` — fuel

## Full Dataset

This sample is a size-limited subset for the repository. The full dataset is available on Google Drive:

| Dataset | Size | Link |
|---------|------|------|
| Full synthetic | 4.45 GB | [frc_dataset.tar](https://drive.google.com/file/d/1nzFFk6rwvYjk1w75BH_ETWjsc_cy0sdU/view) |
| YOLO split | 3.90 GB | [yolo.zip](https://drive.google.com/file/d/1YgHrvRYdsSVqIAC-XK0UlefKavIEfXAx/view) |

### Full Dataset Stats

| Split | Images | Boxes |
|-------|--------|-------|
| Train | 7,847 | — |
| Val | 872 | 90,993 |
| **Total** | **8,721** | — |

## Usage

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolo11s.pt")

# Run inference on sample
results = model.predict("dataset_sample/images/", imgsz=1280)
```

## Visualization

Each label file contains bounding boxes for visible bumpers and fuel. To visualize:

```python
import cv2
import os

img_path = "dataset_sample/images/val_frame_100113.jpg"
lbl_path = "dataset_sample/labels/val_frame_100113.txt"

img = cv2.imread(img_path)
h, w = img.shape[:2]

with open(lbl_path) as f:
    for line in f:
        cls, cx, cy, bw, bh = map(float, line.split())
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        color = [(0,0,255), (255,0,0), (0,255,255)][int(cls)]
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)

cv2.imshow("Sample", img)
cv2.waitKey(0)
```
