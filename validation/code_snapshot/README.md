# FRC Vision Testing: Match Downloader, Zero-Label YOLO11 Evaluator, Closed-Set Robot Identifier (`frcXXXX`) & Gameplay Diagnostics

An end-to-end computer vision and match analytics pipeline for **FIRST Robotics Competition (FRC)** broadcast videos. It combines automated multi-season video/keyframe sampling from **The Blue Alliance (TBA)** with a **YOLO11s (`imgsz=1280`)** detector (`{0: 'bumper_red', 1: 'bumper_blue', 2: 'fuel'}`), **Zero-Label Physical Invariant Accuracy Verification**, **Closed-Set TBA Robot Identification (`frcXXXX`)**, **2D Tactical Minimap Video Annotation**, and **Unsupervised ML Robot Role Clustering**.

---

## Key Capabilities

1. **Geographically Balanced Match Video & Keyframe Pipeline ([`main.py`](main.py), [`downloader.py`](downloader.py), [`frame_extractor.py`](frame_extractor.py))**:
   - Queries The Blue Alliance API v3 ([`tba_client.py`](tba_client.py)) and samples matches evenly across states, provinces, countries, and seasons:
     - `frc_this_year/` (**2026** season matches)
     - `frc_multi_year/` (**2021–2025** multi-season matches)
   - Extracts 5 temporal keyframes per video (`frame_1_start.jpg` @ 5%, `frame_2_middle_1.jpg` @ 27.5%, `frame_3_middle_2.jpg` @ 50%, `frame_4_middle_3.jpg` @ 72.5%, `frame_5_end.jpg` @ 95%).

2. **Zero-Label Model Accuracy & Physical Rule-Invariant Engine ([`evaluator.py`](evaluator.py))**:
   - Evaluates model accuracy on unlabeled broadcast frames using **FRC Physics & Game Rules**:
     - **Alliance Robot Count Cap ($\le 3$ Red, $\le 3$ Blue)**: Merges adjacent bumper segments on the same chassis via hierarchical clustering and checks compliance with the 3-robot-per-alliance rule.
     - **Cross-Year Negative Control (2021–2025)**: Uses pre-2026 matches (where 2026 `fuel` did not exist) as an exact, label-free False Positive benchmark for `fuel` and a domain-generalization test for `bumper_red` and `bumper_blue`.
     - **HSV Bumper Color Purity Audit**: Verifies that predicted red and blue bumper crops contain expected HSV hue distributions and flags arena spotlight glare or white-balance shifts.
     - **Active Learning Hard-Example Mining**: Automatically exports the top 50 highest-uncertainty frames (`active_learning_hardest_50_frames.csv`).

3. **5-Stage Closed-Set TBA Robot Identification (`frcXXXX`) & Annotated Video Renderer ([`robot_identifier.py`](robot_identifier.py))**:
   - Identifies **which specific team (`frcXXXX`) is which robot** throughout a match video without training a 10,000-class detector:
     - **Stage 1 (Closed-Set TBA Prior)**: Loads the exact 3 Red and 3 Blue team numbers (`alliances.red.team_keys`, `alliances.blue.team_keys`) from cached TBA match data.
     - **Stage 2 (Multi-Bumper Merging + Upper-Chassis Re-ID Tracking)**: Tracks 3 persistent robot slots per alliance using Kalman velocity prediction + 32-dim HSV appearance embeddings sampled $1.5\times$ above the bumper to capture unique upper-chassis mechanisms.
     - **Stage 3 (Sharp-Crop White-Digit Bumper OCR)**: Filters for high-sharpness (`Laplacian variance >= 80`) bumper crops, segments white bumper digits, runs Tesseract numeric OCR + digit-length structural priors, and computes fuzzy subsequence similarity strictly against the 3 known alliance team numbers.
     - **Stage 4 (Bayesian Evidence Accumulation + Hungarian 1-to-1 Bijection)**: Solves `scipy.optimize.linear_sum_assignment` over the $(3 \text{ slots} \times 3 \text{ teams})$ evidence matrix—automatically identifying the 3rd robot by process of elimination when 2 robots on an alliance are read.
     - **Stage 5 (Two-Pass Retroactive Video Annotation + 2D Tactical Radar)**: Renders annotated `.mp4` videos with locked `frcXXXX` labels, motion trails, a live **2D Top-Down Tactical Minimap Radar**, and a broadcast telemetry HUD.

4. **FRC Gameplay Diagnostics, ML Role Clustering & Scouting Encyclopedia ([`frc_diagnostics.py`](frc_diagnostics.py))**:
   - Generates **13 publication-quality charts**, joins vision metrics with official TBA `score_breakdown` (`hubScore`, tower climbs, fouls), classifies robot playstyles (`Primary Cycler`, `Dedicated Defender`, `Feeder / Hybrid Support`, `Impaired / Low Mobility`) using **StandardScaler + Silhouette-Optimized K-Means + 2D PCA**, and compiles [`analytics_outputs/ANALYTICS_REPORT.md`](analytics_outputs/ANALYTICS_REPORT.md) and [`analytics_outputs/ROBOT_SCOUTING_ENCYCLOPEDIA.md`](analytics_outputs/ROBOT_SCOUTING_ENCYCLOPEDIA.md).

---

## Generated Analytics & Diagnostic Charts Preview

All charts are automatically saved to both [`analytics_outputs/charts/`](analytics_outputs/charts/) (tracked in GitHub) and `<FRC_DATA_ROOT>/frc_vision_analytics/charts/`:

| Training & Zero-Label Accuracy | Cross-Year & Fuel Dynamics |
| :---: | :---: |
| ![Training Curves](analytics_outputs/charts/01_training_convergence_and_val_curves.png) | ![Zero-Label Rule Invariants](analytics_outputs/charts/02_zero_label_rule_invariant_accuracy.png) |
| ![Cross-Year Generalization](analytics_outputs/charts/03_cross_year_generalization_and_fuel_fp.png) | ![Fuel Depletion Curve](analytics_outputs/charts/04_fuel_depletion_and_scarcity_by_phase.png) |

| Field Heatmaps & Territory/Defense | Venue Lighting & TBA Correlation |
| :---: | :---: |
| ![2D Field Heatmaps](analytics_outputs/charts/05_field_spatial_heatmaps_2d.png) | ![Territory and Defense](analytics_outputs/charts/06_alliance_territory_and_defense_dynamics.png) |
| ![Venue Lighting Report](analytics_outputs/charts/07_venue_lighting_and_camera_weather_report.png) | ![TBA Score Correlation](analytics_outputs/charts/08_tba_score_vs_vision_activity_correlation.png) |

| Per-Robot (`frcXXXX`) Leaderboard & ML Role Clustering | 2D Trajectories & Scouting Cards |
| :---: | :---: |
| ![Per-Robot Leaderboard](analytics_outputs/charts/09_per_robot_telemetry_leaderboard.png) | ![ML Role Clustering PCA](analytics_outputs/charts/10_robot_role_clustering_pca_silhouette.png) |
| ![2D Match Trajectories](analytics_outputs/charts/11_match_trajectories_tactical_board.png) | ![Sample Individual Robot Scouting Dashboard (frc4150)](analytics_outputs/charts/13_individual_robot_scouting_cards/frc4150_scouting_dashboard.png) |

---

## Repository & Extreme SSD Output Structure

```text
FRC Vision Testing/
├── models/
│   ├── frc_yolo11s_best.pt                           # Trained YOLO11s weights (18 MB)
│   └── training_artifacts/runs/frc/                  # Training curves, confusion matrices & results.csv
├── analytics_outputs/                                # GitHub-ready reports, charts & telemetry CSVs
│   ├── ANALYTICS_REPORT.md                           # Full narrative analysis & diagnostic report
│   ├── ROBOT_SCOUTING_ENCYCLOPEDIA.md                # Per-robot scouting table for every tracked team
│   ├── charts/                                       # 13 high-resolution PNG charts
│   └── data/                                         # Structured CSV/JSON datasets
│       ├── frame_evaluation_results.csv              # Per-frame zero-label accuracy & FRC metrics
│       ├── all_bounding_box_detections.csv           # Every predicted box with HSV color ratios
│       ├── per_robot_match_telemetry.csv             # Per-robot (frcXXXX) speed, cycles, defense, PCA
│       ├── aggregated_team_scouting_profiles.csv     # Aggregated team profiles
│       ├── all_matches_every_robot_catalog.csv       # Every team across all downloaded matches
│       └── active_learning_hardest_50_frames.csv     # Top 50 uncertain frames for active learning
├── evaluator.py                                      # Zero-label accuracy & FRC rule-invariant engine
├── robot_identifier.py                               # 5-stage closed-set TBA robot ID & video renderer
├── frc_diagnostics.py                                # 13-chart suite, K-Means/PCA clustering & reports
├── run_vision_pipeline.py                            # Master end-to-end orchestrator + SSD code backup
├── main.py                                           # TBA video downloader & 5-frame extractor CLI
├── dashboard.py                                      # Localhost live telemetry web dashboard
└── tba_client.py                                     # Cached TBA API v3 client
```

On `<FRC_DATA_ROOT>/frc_vision_analytics/`:
- `annotated_frames/`: Every extracted keyframe rendered with bounding boxes, robot cluster envelopes, and rule-compliance HUD.
- `annotated_videos/`: Full match `.mp4` videos rendered with locked `frcXXXX` team labels, motion trails, and the **2D Top-Down Tactical Minimap Radar**.
- `charts/`, `data/`, `code_snapshot/`: Full mirror of all charts, CSV/JSON datasets, Markdown reports, and source code.

---

## Quickstart & CLI Usage

### 1. Environment Setup
```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Run the Full Vision Evaluation, Robot Identification & Diagnostics Pipeline
To evaluate all extracted frames, track and annotate match videos with `frcXXXX` team IDs, generate all 13 charts + Markdown reports, and back up everything to the Extreme SSD:
```bash
# Process all extracted frames + render 15 full annotated match videos
.venv/bin/python run_vision_pipeline.py --max-videos 15 --video-stride 5

# Process ALL downloaded match videos on the Extreme SSD
.venv/bin/python run_vision_pipeline.py --max-videos 0 --video-stride 5
```

### 3. Run Individual Modules
```bash
# Zero-label frame evaluation & annotated frame generation only
.venv/bin/python evaluator.py --ssd-root "<FRC_DATA_ROOT>"

# Closed-set TBA robot identification & annotated MP4 rendering only
.venv/bin/python robot_identifier.py --max-videos 20 --stride 4

# Regenerate all 13 charts, K-Means/PCA clusters & Markdown reports from CSVs
.venv/bin/python frc_diagnostics.py
```

### 4. Download Match Videos & Extract Keyframes from The Blue Alliance
```bash
python3 main.py --gdrive-dir "<FRC_DATA_ROOT>" --quality best
```
