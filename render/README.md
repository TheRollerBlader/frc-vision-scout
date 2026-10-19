# Synthetic Rendering

Blender Cycles rendering pipeline for generating synthetic FRC training data.

## Overview

Every training image is a procedurally generated Blender (Cycles) frame designed to look like a FIRST YouTube broadcast at 1080p. Nothing is hand-labeled — bounding boxes are computed from the 3D scene.

### Key Stats
- **7 stadium environments** (arenas, hockey rinks, convention centers)
- **8,721 frames** rendered
- **~$25 of Runpod GPU time** (RTX PRO 4500, L4, RTX PRO 4000)
- **2-3 Blender instances per GPU** for throughput
- **17-25s per frame** (mostly CPU scene setup, ~5s GPU rendering)

## Directory Structure

```
render/
├── blender_render.py       # Main Blender scene builder + renderer
├── bumpers.py              # Procedural bumper generation
├── post.py                 # Post-processing & YOLO label export
├── stadium_prep.py         # One-time stadium environment prep
├── render.sh               # Multi-instance render driver
├── setup.sh                # Cloud GPU setup script
├── make_bundle.sh          # Pack render kit for upload
├── manifest.json           # Asset manifest (robots/stadiums)
├── assets/
│   ├── TEXTURES.md         # Texture credits (CC0)
│   ├── carpet/             # Carpet texture files
│   └── fonts/              # Bumper number fonts
├── crowd/
│   └── build_crowd_proxies.py  # Low-poly crowd builder
└── examples/
    ├── frame_000001.json   # Example scene layout
    └── frame_000001.png    # Layout visualization
```

## Quick Start

### Cloud GPU Setup

```bash
# On a Runpod instance (or similar)
bash setup.sh

# Test render (50 frames)
bash render.sh --test

# Full render
bash render.sh --full 10000
```

### render.sh Options

```
--seed S            First frame seed (default 100000)
--samples K         Cycles samples (default 64)
--res WxH           Render size (default 1920x1080)
--chunk C           Frames per Blender process (default 25)
--gpus G            Number of GPUs (default: auto-detect)
--per-gpu N         Blender instances per GPU (default 1)
--shard i/n         Distributed rendering (multiple machines)
--budget USD        Stop after estimated cost
--price USD_PER_HR  GPU hourly rate for budget calc
```

### Resuming

Renders are resumable. Frames with existing `labels/<id>.json` are skipped.

## Scene Randomization

Per-frame randomization (seeded for reproducibility):

| Element | Randomization |
|---------|--------------|
| **Robots** | Pose, velocity, team numbers, bumper colors |
| **Fuel** | Count, positions, airborne trajectories |
| **Camera** | Position, zoom, DOF, motion blur |
| **Lighting** | Intensity, color temperature, arena lights |
| **Crowd** | 15k-64k fans, occupancy 30-90%, shirt colors |
| **Textures** | Carpet, metal, concrete variations |

### Broadcast Camera Style

All cameras mimic FRC broadcast conventions:
- Head-on from midfield (x ±0.5m, yaw ±2.5°)
- 8.5-20.5m behind near guardrail
- 11-25° elevation
- Far guardrail fills 84-98% of frame width
- DOF in 60% of frames, pan blur in 35%
- All 6 robots in view in 99% of frames

## Stadium Prep

Per-process stadium preparation:
- Remove duplicate back-faces (shadow acne fix)
- Clear field area
- Add floor planes
- Detect and cache seat positions
- Set up crowd instancer
- Place stand lights and fill lights

## Label Generation

`post.py` handles:
- Noise/blur augmentation
- YOLO label export (normalized coordinates)
- Visible-portion-only bounding boxes
- Split bumper handling

## Textures

All textures are CC0 licensed:
- [ambientCG](https://ambientcg.com/)
- [Poly Haven](https://polyhaven.com/)

See `assets/TEXTURES.md` for full credits.

## Notes

- Requires Blender 4.5 with Cycles GPU support
- Watch host RAM (~2-4 GB per Blender process)
- The manifest.json references robot/stadium GLBs (not included — see assets/ for conversion)
