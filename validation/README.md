# Validation

Zero-label evaluation, robot identification, and gameplay analytics on real FRC match footage.

## Overview

This pipeline validates the synthetic-trained model on real FRC broadcasts without requiring manual labels. Instead, it uses FRC game rules as ground truth:

- **≤3 robots per alliance** — There can never be more than 3 red or 3 blue robots on the field
- **HSV color purity** — Red bumpers should be red, blue bumpers should be blue
- **In-field detection** — Robots should be inside the field perimeter

## Directory Structure

```
validation/
├── ANALYTICS_REPORT.md           # Full validation report
├── ROBOT_SCOUTING_ENCYCLOPEDIA.md # Per-robot scouting dossiers
├── code_snapshot/                # Analytics pipeline source
│   ├── main.py                   # CLI entry point
│   ├── evaluator.py              # Zero-label accuracy engine
│   ├── robot_identifier.py       # Closed-set robot ID (frcXXXX)
│   ├── frc_diagnostics.py        # Chart generation & clustering
│   ├── dashboard.py              # Per-robot scouting dashboard
│   ├── downloader.py             # YouTube video downloader
│   ├── frame_extractor.py        # 5-keyframe extraction
│   ├── sampler.py                # Geographic match sampling
│   ├── tba_client.py             # TBA API client
│   ├── config.py                 # Configuration
│   ├── run_vision_pipeline.py    # End-to-end orchestrator
│   └── requirements.txt          # Python dependencies
├── data/                         # Evaluation outputs
│   ├── frame_evaluation_results.csv/json
│   ├── per_robot_match_telemetry.csv/json
│   ├── aggregated_team_scouting_profiles.csv/json
│   ├── all_matches_every_robot_catalog.csv/json
│   ├── match_level_tba_vision_summary.csv
│   └── active_learning_hardest_50_frames.csv
└── robot_profiles/               # Individual robot folders
    └── frc4306/                  # Sample profile
```

## Key Metrics

| Metric | Value |
|--------|-------|
| Keyframes evaluated | 775 (155 matches, 130 events) |
| Bounding boxes analyzed | 103,816 |
| Red alliance ≤3 robots | 95.87% |
| Blue alliance ≤3 robots | 96.52% |
| In-field precision | 99.90% |
| HSV color purity | 97.83% |
| Cross-year generalization | 1.98 vs 3.09 bots/frame |
| Tracked robots | 48 (45 unique teams) |
| Playstyle clusters | k=6, silhouette 0.250 |

## Quick Start

### Setup

```bash
cd validation/code_snapshot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Set TBA API Key

```bash
export TBA_API_KEY="your-key"
# or
export TBA_AUTH_KEY="your-key"
```

### Run Full Pipeline

```bash
# Process frames + render annotated videos
python run_vision_pipeline.py --max-videos 15 --video-stride 5

# Process all downloaded videos
python run_vision_pipeline.py --max-videos 0 --video-stride 5
```

### Run Individual Modules

```bash
# Zero-label evaluation only
python evaluator.py --ssd-root /path/to/data

# Robot identification only
python robot_identifier.py --max-videos 20 --stride 4

# Regenerate charts from CSVs
python frc_diagnostics.py
```

## Pipeline Stages

### 1. Match Sampling (`sampler.py`)
Geographically balanced selection across states, provinces, and seasons.

### 2. Video Download (`downloader.py`)
Downloads match videos from YouTube via TBA video URLs.

### 3. Frame Extraction (`frame_extractor.py`)
Extracts 5 keyframes per match at 5%, 27.5%, 50%, 72.5%, 95% duration.

### 4. Zero-Label Evaluation (`evaluator.py`)
- Runs YOLO inference
- Merges bumper segments into robot clusters
- Checks ≤3 robots per alliance
- Validates HSV color purity
- Mines hard examples for active learning

### 5. Robot Identification (`robot_identifier.py`)
5-stage closed-set identification:
1. **TBA Prior**: Load known team numbers from match data
2. **Re-ID Tracking**: HSV appearance embeddings + Kalman filter
3. **Bumper OCR**: White digit recognition on bumpers
4. **Hungarian Assignment**: Optimal 1-to-1 team matching
5. **Video Annotation**: Render labeled videos with tactical radar

### 6. Analytics (`frc_diagnostics.py`)
Generates 13 diagnostic charts:
1. Training convergence
2. Zero-label rule accuracy
3. Cross-year generalization
4. Fuel depletion curves
5. Field spatial heatmaps
6. Alliance territory dynamics
7. Venue lighting benchmarks
8. TBA score correlations
9. Per-robot leaderboard
10. ML playstyle clustering
11. Match trajectory boards
12. Robot ID diagnostics
13. Individual scouting cards

## Outputs

### Reports
- `ANALYTICS_REPORT.md` — Full narrative analysis
- `ROBOT_SCOUTING_ENCYCLOPEDIA.md` — Per-robot profiles

### Charts
See `results/charts/` for all 12 visualization charts.

### Data Files
- `frame_evaluation_results` — Per-frame metrics
- `per_robot_match_telemetry` — Robot kinematics
- `aggregated_team_scouting_profiles` — Team aggregates
- `active_learning_hardest_50_frames` — Hard examples for labeling
