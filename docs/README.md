# Documentation Assets

Media files used in the README.

## Directory Structure

```
docs/
└── media/
    ├── banner_annotated_slideshow.gif   # README banner (12s, 800px)
    ├── annotated_frames/                # Real match detections
    │   ├── 2026wasam_qm55_frame_*.jpg
    │   ├── 2026micmp1_qm53_frame_*.jpg
    │   ├── 2026necmp1_qm96_frame_*.jpg
    │   └── 2026utwv_f1m1_frame_*.jpg
    └── previews/                        # Synthetic render samples
        ├── arena_broadcast.jpg
        ├── sheet_broadcast.jpg
        ├── crowd_hockey_000101.jpg
        ├── fix_jmn_center_777005.jpg
        ├── fix_sheet.jpg
        └── people_preview.jpg
```

## Banner GIF

The `banner_annotated_slideshow.gif` is a 12-second slideshow of 8 annotated real-match keyframes. Full annotated videos are too large for the repository (169-327 MB each).

## Annotated Frames

Real FRC 2026 match keyframes with YOLO11s detections overlaid:
- Bounding boxes for red/blue bumpers and fuel
- Confidence scores
- Robot cluster envelopes

Events shown:
- `2026wasam` — Washington State
- `2026micmp1` — Michigan Championship
- `2026necmp1` — New England Championship
- `2026utwv` — Utah/West Virginia

## Synthetic Previews

Sample renders from the Blender pipeline:
- `arena_broadcast.jpg` — Broadcast camera view
- `sheet_broadcast.jpg` — Contact sheet across stadiums
- `crowd_hockey_000101.jpg` — Hockey arena with crowd
- `fix_jmn_center_777005.jpg` — JMN center stadium
- `people_preview.jpg` — Field staff avatars
