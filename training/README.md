# Training

YOLO11s training configuration and notebook.

## Files

```
training/
├── train_frc.ipynb    # Training notebook (Runpod/Colab)
└── data.yaml          # YOLO dataset configuration
```

## Model Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | YOLO11s (Ultralytics) |
| **Image Size** | 1280 |
| **Batch** | AutoBatch (5 on 24GB VRAM) |
| **Epochs** | 100 (patience 20) |
| **Time Limit** | 6 hours |
| **Seed** | 42 |

## Classes

```yaml
names:
  0: bumper_red
  1: bumper_blue
  2: fuel
```

## Dataset Split

| Split | Images | Boxes |
|-------|--------|-------|
| Train | 7,847 | — |
| Val | 872 | 90,993 |

## Training Results

| Metric | Best (Epoch 99) | Final (Epoch 119) |
|--------|-----------------|-------------------|
| Precision | 0.982 | 0.981 |
| Recall | 0.941 | 0.941 |
| mAP@50 | 0.963 | 0.959 |
| mAP@50-95 | 0.801 | 0.798 |

### Per-Class mAP@50

| Class | mAP@50 |
|-------|--------|
| bumper_red | 0.989 |
| bumper_blue | 0.989 |
| fuel | 0.911 |

## Usage

### 1. Get the Dataset

Download from Google Drive:
- [YOLO split (3.90 GB)](https://drive.google.com/file/d/1YgHrvRYdsSVqIAC-XK0UlefKavIEfXAx/view)

```bash
# Extract
unzip yolo.zip -d /workspace/yolo
```

### 2. Update data.yaml

The provided `data.yaml` expects:
```yaml
path: /workspace/yolo
train: images/train
val: images/val
```

Adjust paths for your environment.

### 3. Train

```bash
# Install Ultralytics
pip install ultralytics

# Run training
yolo detect train \
    model=yolo11s.pt \
    data=data.yaml \
    imgsz=1280 \
    epochs=100 \
    patience=20 \
    batch=-1 \
    seed=42 \
    time=6
```

Or use the notebook:
```bash
jupyter notebook train_frc.ipynb
```

## Notes

- **Weights not included**: This repo is a portfolio showcase. Train your own using the dataset.
- Training was done on Runpod with RTX PRO 4000 (24GB VRAM)
- Wall time: ~5.8 hours to epoch 119
- AutoBatch selected batch size 5 for 24GB VRAM

## Outputs

Training artifacts are saved to `runs/frc/`:
- `weights/best.pt` — Best checkpoint (epoch 99)
- `weights/last.pt` — Final checkpoint
- `results.csv` — Per-epoch metrics
- Curves and confusion matrices

See `results/training/` for the saved outputs.
