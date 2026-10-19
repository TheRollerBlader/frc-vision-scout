# FRC YOLO11s Zero-Label Accuracy, Robot Identification (`frcXXXX`) & Gameplay Diagnostics Report

> **Dataset Location (Extreme SSD)**: `<FRC_DATA_ROOT>`  
> **Annotated Videos, Frames & Full Telemetry**: `<FRC_DATA_ROOT>/frc_vision_analytics/`  
> **GitHub Analytics Mirror**: `analytics_outputs/`  
> **Model Weights**: `models/frc_yolo11s_best.pt` (`yolo11s`, `imgsz=1280`, classes: `0: bumper_red`, `1: bumper_blue`, `2: fuel`)

---

## 1. Executive Summary & Key Metrics

| Metric Category | Measured Value | Diagnostic Interpretation |
| :--- | :--- | :--- |
| **Total Extracted Keyframes Evaluated** | **775** frames (155 matches across 130 events) | Spans `frc_this_year` (2026) and `frc_multi_year` (2021–2025) |
| **Total Bounding Boxes Analyzed** | **103,816** detections | Audited for field bounds, HSV color purity, and robot clustering |
| **Validation Peak mAP@50 / mAP@50-95** | **96.30% / 80.10%** | Precision = **98.19%**, Recall = **94.10%** at Epoch 99 |
| **Zero-Label Red Alliance Count Rule ($\le 3$)** | **95.87%** compliance | Physical FRC rule check: at most 3 distinct Red robot clusters per frame |
| **Zero-Label Blue Alliance Count Rule ($\le 3$)** | **96.52%** compliance | Physical FRC rule check: at most 3 distinct Blue robot clusters per frame |
| **In-Field Detection Precision Proxy** | **99.90%** | Fraction of detections strictly inside field carpet perimeter (excluding audience/scoreboard) |
| **HSV Bumper Color Purity Rate** | **97.83%** | Verifies `bumper_red` boxes contain red hue and `bumper_blue` boxes contain blue hue |
| **Cross-Year Generalization (2021–2025 Bots)** | **1.98** bots/frame (vs **3.09** in 2026) | Strong multi-season transfer of `bumper_red` and `bumper_blue` across 5 years of cameras |
| **Full-Video Tracked Robot Instances** | **48** robots (**45** unique teams) | Plus **789 unique FRC teams (930 robot-match appearances)** cataloged across all 155 matches |
| **Optimal ML Playstyle Clusters ($k$)** | **$k = 6$** (Silhouette = **0.250**) | PCA 2D projection explains **56.3%** of variance across 10 kinematic/tactical features |

---

## 2. Training Convergence & Zero-Label Model Accuracy Verification

### 2.1 Training & Validation Curves (`runs.zip` $\rightarrow$ `results.csv`)
![Training Convergence](charts/01_training_convergence_and_val_curves.png)

### 2.2 Zero-Label Physical Rule-Invariant Audit ($\le 3$ Robots/Alliance & HSV Color Purity)
By merging adjacent bumper segments into distinct robot footprints via hierarchical clustering (`evaluator.py`), we verify whether the model obeys the fundamental physical law of FRC: **there can never be more than 3 Red robots or 3 Blue robots on the carpet**.
![Zero-Label Rule Invariant Accuracy](charts/02_zero_label_rule_invariant_accuracy.png)

### 2.3 Cross-Year Generalization & Out-of-Distribution Negative Control (`frc_multi_year`)
Running the 2026 model on 2021–2025 matches tests (1) whether `bumper_red` and `bumper_blue` generalize to older seasons and venues, and (2) whether older yellow/orange game pieces (e.g., 2022/2023/2024) trigger False Positives on the 2026 `fuel` class.
![Cross-Year Generalization](charts/03_cross_year_generalization_and_fuel_fp.png)

---

## 3. FRC Gameplay & Field Diagnostics

### 3.1 2026 Fuel Depletion, Scarcity & Zone Accumulation Across Match Phases
Tracking `fuel` count from `frame_1_start` (Auto, 5% duration) through `frame_5_end` (Endgame, 95% duration) reveals how loose game pieces accumulate and clear across the Left (Red), Center (Neutral), and Right (Blue) zones.
![Fuel Depletion Curve](charts/04_fuel_depletion_and_scarcity_by_phase.png)

### 3.2 2D Top-Down FRC Field Spatial Heatmaps
![2D Field Spatial Heatmaps](charts/05_field_spatial_heatmaps_2d.png)

### 3.3 Alliance Territorial Dominance & Active Defense Engagements
![Alliance Territory and Defense](charts/06_alliance_territory_and_defense_dynamics.png)

### 3.4 FRC "Vision Weather Report": Venue Camera & Arena Lighting Benchmark
Some FRC venues suffer from harsh overhead spotlights that create white glare on polycarbonate bumpers or shift white balance. Below is the automated ranking of events by vision reliability:
![Venue Lighting Weather Report](charts/07_venue_lighting_and_camera_weather_report.png)

### 3.5 The Blue Alliance (TBA) Score Breakdown vs. Vision Activity Correlation
![TBA Score Correlation](charts/08_tba_score_vs_vision_activity_correlation.png)

---

## 4. Persistent Robot Identification (`frcXXXX`) & Individual Robot Analytics

### 4.1 Closed-Set TBA Robot Identification Pipeline (`robot_identifier.py`)
Each match video is paired with its cached TBA match record (`.cache/tba`) to constrain the 3 Red robots and 3 Blue robots to their exact official team numbers (`frcXXXX`). Combining **Upper-Chassis HSV Appearance Re-ID**, **Sharp-Crop White-Digit Bumper OCR**, **Starting-Station Spatial Priors**, and **Hungarian 1-to-1 Bijection (`scipy.optimize.linear_sum_assignment`)** locks in all 6 robot identities throughout the match.
![Closed-Set Robot ID Diagnostics](charts/12_closed_set_robot_id_diagnostics.png)

### 4.2 Per-Robot Kinematics, Distance, Speed & Role Leaderboard
![Per-Robot Telemetry Leaderboard](charts/09_per_robot_telemetry_leaderboard.png)

### 4.3 Unsupervised ML Playstyle Clustering (StandardScaler + K-Means Silhouette + 2D PCA)
Following ML best practices, we standardized 10 kinematic and spatial features per robot, swept K-Means across $k \in [2, 6]$ to maximize the Silhouette Score, and projected the robots onto 2D PCA space:
![ML Role Clustering PCA](charts/10_robot_role_clustering_pca_silhouette.png)

### 4.4 2D Top-Down Match Trajectory Tactical Boards
![2D Match Trajectories](charts/11_match_trajectories_tactical_board.png)

### 4.5 Individual Robot Scouting Folders & 4-Panel Dashboards
Every tracked robot has its own folder and full-resolution 4-panel dashboard in [`charts/13_individual_robot_scouting_cards/`](charts/13_individual_robot_scouting_cards/) and [`robot_profiles/`](robot_profiles/) (see [**`ROBOT_SCOUTING_ENCYCLOPEDIA.md`**](ROBOT_SCOUTING_ENCYCLOPEDIA.md) for all links!).
