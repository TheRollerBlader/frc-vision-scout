"""Closed-Set TBA Robot Identification, Multi-Object Re-ID Tracker & Annotated Video Renderer.

This module identifies and tracks all 6 individual FRC robots (`frcXXXX`) throughout
a match video without requiring manual training labels for team numbers.

5-Stage Architecture:
1. Closed-Set TBA Roster Prior:
   - Queries cached The Blue Alliance (TBA) match metadata (`.cache/tba`) to obtain
     the exact 3 Red team numbers and 3 Blue team numbers on the field.
2. Multi-Bumper Clustering & Upper-Chassis Re-ID Tracking:
   - Merges same-robot bumper boxes and tracks 3 persistent robot slots per alliance
     using motion prediction + upper-chassis HSV appearance embeddings.
3. Sharpness-Gated Bumper Digit Extraction & Closed-Set OCR:
   - Isolates white bumper text on sharp crops, runs Tesseract OCR (`0-9` whitelist)
     plus digit-width structural priors, and scores strictly against the 3 known
     alliance team numbers.
4. Bayesian Accumulation & Hungarian 1-to-1 Bijection:
   - Accumulates team-number evidence across the match and enforces a strict 1-to-1
     mapping via `scipy.optimize.linear_sum_assignment` (solving the 3rd robot by
     process of elimination even if its bumper numbers face away).
5. Two-Pass Retroactive Video Annotation & 2D Tactical Minimap Radar:
   - Applies the globally converged `frcXXXX` identities across the entire timeline,
     computes rich per-robot FRC gameplay diagnostics, and renders annotated `.mp4`
     videos with motion trails, a 2D top-down radar minimap, and a live telemetry HUD.
"""

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from evaluator import (
    BGR_COLORS,
    CLASS_NAMES,
    DetectionBox,
    RobotCluster,
    cluster_alliance_bumpers,
    compute_hsv_color_ratios,
    parse_video_stem_metadata,
)


@dataclass
class RobotTelemetrySummary:
    """Comprehensive match-level performance & diagnostic profile for a single identified robot."""
    match_key: str
    event_key: str
    year: int
    video_stem: str
    alliance: str                   # "red" or "blue"
    station_slot: int               # 1, 2, or 3
    team_key: str                   # e.g. "frc2495"
    team_number: str                # e.g. "2495"
    # Tracking & Identification confidence
    frames_visible: int
    total_sampled_frames: int
    visibility_rate: float          # fraction of match frames robot was detected
    mean_yolo_conf: float
    id_confidence: float            # closed-set Bayesian assignment confidence (0-1)
    ocr_reads_count: int
    best_ocr_raw_text: str
    id_method: str                  # "OCR+Station", "ProcessOfElimination+Station", etc.
    # Kinematics & Mobility
    total_distance_norm: float      # total normalized field widths traveled
    estimated_distance_feet: float  # scaled to 54ft FRC field length
    mean_speed_fps: float           # estimated feet per second
    max_speed_fps: float            # peak sprint speed (ft/s)
    stationary_seconds: float       # time spent near-zero velocity
    is_dead_or_impaired: bool       # stationary > 25% of match or < 25ft total movement
    # Spatial & Tactical Role Metrics
    mean_norm_x: float
    mean_norm_y: float
    std_norm_x: float
    std_norm_y: float
    time_in_own_zone_pct: float
    time_in_neutral_zone_pct: float
    time_in_enemy_zone_pct: float   # deep offensive cycling or counter-defense
    # FRC Gameplay Diagnostics
    estimated_cycles: int           # round-trips between fuel pickup/neutral and scoring zone
    mean_cycle_time_sec: float
    fuel_proximity_frames: int      # frames within intake distance of visible fuel
    defense_engagement_sec: float   # seconds spent within contact distance of opposing robot
    primary_opponent_engaged: str   # opposing team_key most frequently in contact
    classified_role: str            # "Primary Cycler", "Dedicated Defender", "Feeder / Hybrid", "Impaired / Low Mobility"
    # Official TBA Match Context
    alliance_Final_score: int
    alliance_won: bool
    tba_auto_tower_status: str
    tba_endgame_tower_status: str
    tba_energized_achieved: bool
    tba_supercharged_achieved: bool
    tba_traversal_achieved: bool
    annotated_video_path: str


def extract_upper_chassis_embedding(img_bgr: np.ndarray, cluster: RobotCluster) -> np.ndarray:
    """Computes a 32-dim L2-normalized HSV appearance vector from the bumper + upper chassis."""
    h, w = img_bgr.shape[:2]
    bw = max(8.0, cluster.x2 - cluster.x1)
    bh = max(8.0, cluster.y2 - cluster.y1)

    # Sample region extending 1.5x bumper height above the bumper to capture robot mechanisms
    x1 = int(max(0, cluster.x1))
    x2 = int(min(w, cluster.x2))
    y1 = int(max(0, cluster.y1 - 1.5 * bh))
    y2 = int(min(h, cluster.y2))

    crop = img_bgr[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return np.zeros(32, dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

    vec = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-6:
        vec /= norm
    return vec


def closed_set_string_similarity(ocr_digits: str, candidate_team_num: str) -> float:
    """Scores a raw OCR digit string (e.g. '249' or '424') against a known TBA team number ('2495')."""
    ocr_clean = re.sub(r"[^0-9]", "", ocr_digits)
    cand_clean = re.sub(r"[^0-9]", "", candidate_team_num)
    if not ocr_clean or not cand_clean:
        return 0.0

    if ocr_clean == cand_clean:
        return 1.0

    # Direct substring match (e.g. '249' inside '2495' or '428' inside '428')
    if len(ocr_clean) >= 2 and ocr_clean in cand_clean:
        return 0.75 + 0.20 * (len(ocr_clean) / len(cand_clean))

    # Character-level longest common subsequence in order
    dp = [[0] * (len(cand_clean) + 1) for _ in range(len(ocr_clean) + 1)]
    for i in range(1, len(ocr_clean) + 1):
        for j in range(1, len(cand_clean) + 1):
            if ocr_clean[i - 1] == cand_clean[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]

    # Penalize large length mismatches
    max_len = max(len(ocr_clean), len(cand_clean))
    score = float(lcs) / float(max_len)
    return round(score, 4)


class ClosedSetBumperOCR:
    """Reads bumper numbers on high-sharpness crops and scores against the 3 TBA alliance teams."""

    def __init__(self, tesseract_bin: Optional[str] = None):
        self.tesseract_bin = tesseract_bin or shutil.which("tesseract") or "/opt/homebrew/bin/tesseract"
        self.has_tesseract = bool(self.tesseract_bin and Path(self.tesseract_bin).is_file())

    def read_bumper_crop(
        self,
        crop_bgr: np.ndarray,
        candidate_numbers: List[str],
    ) -> Tuple[str, Dict[str, float]]:
        """Extracts white bumper text, runs OCR + structural digit-length priors, and scores candidates."""
        scores = {num: 0.05 for num in candidate_numbers}
        if crop_bgr is None or crop_bgr.size == 0:
            return "", scores

        h, w = crop_bgr.shape[:2]
        if w < 28 or h < 10:
            return "", scores

        # Upscale 2.5x for cleaner white-digit segmentation
        upscaled = cv2.resize(crop_bgr, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(upscaled, cv2.COLOR_BGR2HSV)
        # White bumper numbers have low saturation and high value (brightness)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 155]), np.array([180, 95, 255]))

        # Estimate digit count from connected white blobs inside the bumper interior
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask)
        uh, uw = white_mask.shape[:2]
        valid_digit_blobs = 0
        for i in range(1, num_labels):
            bx, by, bw_blob, bh_blob, area = stats[i]
            if 0.22 * uh <= bh_blob <= 0.88 * uh and 0.04 * uw <= bw_blob <= 0.35 * uw and area >= 18:
                valid_digit_blobs += 1

        # Apply structural digit-length prior (e.g. 3-digit vs 4-digit team number)
        if 2 <= valid_digit_blobs <= 5:
            for num in candidate_numbers:
                if len(num) == valid_digit_blobs:
                    scores[num] += 0.15

        raw_text = ""
        if self.has_tesseract:
            # Invert so digits are black on white background for Tesseract
            inv = cv2.bitwise_not(white_mask)
            inv = cv2.copyMakeBorder(inv, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
            try:
                cmd = [
                    self.tesseract_bin,
                    "stdin",
                    "stdout",
                    "--psm",
                    "7",
                    "-c",
                    "tessedit_char_whitelist=0123456789",
                ]
                ok, buf = cv2.imencode(".png", inv)
                if ok:
                    proc = subprocess.run(
                        cmd,
                        input=buf.tobytes(),
                        capture_output=True,
                        timeout=1.5,
                        check=False,
                    )
                    raw_text = re.sub(r"[^0-9]", "", proc.stdout.decode("utf-8", errors="ignore"))
            except Exception:
                raw_text = ""

        if raw_text:
            for num in candidate_numbers:
                sim = closed_set_string_similarity(raw_text, num)
                scores[num] += sim * 1.25

        return raw_text, scores


class AllianceRobotTracker:
    """Maintains 3 persistent robot tracks for an alliance and resolves their TBA team numbers."""

    def __init__(self, alliance: str, team_keys: List[str], ocr_engine: ClosedSetBumperOCR):
        self.alliance = alliance  # "red" or "blue"
        # Ensure we always have 3 team slots
        padded = list(team_keys) + [f"frc{alliance.upper()}{i}" for i in range(1, 4)]
        self.team_keys = padded[:3]
        self.team_numbers = [tk.replace("frc", "") for tk in self.team_keys]
        self.ocr_engine = ocr_engine

        # 3 persistent slots (0, 1, 2) initialized at typical starting line Y positions (top, mid, bottom)
        default_x = 0.18 if alliance == "red" else 0.82
        default_ys = [0.35, 0.55, 0.75]
        self.slot_positions = [[default_x, y] for y in default_ys]
        self.slot_velocities = [[0.0, 0.0] for _ in range(3)]
        self.slot_embeddings = [np.zeros(32, dtype=np.float32) for _ in range(3)]
        self.slot_last_seen_frame = [-100, -100, -100]

        # Cumulative log-evidence matrix: shape (3 slots, 3 candidate team_numbers)
        # Initialize with slight prior favoring station order (Slot 0 -> Team 0, Slot 1 -> Team 1, Slot 2 -> Team 2)
        self.evidence_matrix = np.array(
            [
                [0.35, 0.10, 0.05],
                [0.10, 0.35, 0.10],
                [0.05, 0.10, 0.35],
            ],
            dtype=np.float64,
        )
        self.slot_ocr_reads: List[List[str]] = [[], [], []]
        self.slot_history: Dict[int, List[Dict[str, Any]]] = {0: [], 1: [], 2: []}

    def update(
        self,
        frame_idx: int,
        timestamp_sec: float,
        img_bgr: np.ndarray,
        clusters: List[RobotCluster],
        run_ocr: bool = False,
    ) -> Dict[int, RobotCluster]:
        """Associates detected robot clusters in the current frame to the 3 persistent alliance slots."""
        if not clusters:
            return {}

        # Limit to top 3 highest-confidence clusters in field bounds
        active_clusters = clusters[:3]
        n_det = len(active_clusters)

        # Build (3 slots x n_det) cost matrix combining position + upper-chassis appearance
        cost_matrix = np.zeros((3, n_det), dtype=np.float64)
        det_embeddings = [extract_upper_chassis_embedding(img_bgr, c) for c in active_clusters]

        for s_idx in range(3):
            dt_frames = max(1, min(15, frame_idx - self.slot_last_seen_frame[s_idx]))
            pred_x = self.slot_positions[s_idx][0] + self.slot_velocities[s_idx][0] * dt_frames
            pred_y = self.slot_positions[s_idx][1] + self.slot_velocities[s_idx][1] * dt_frames

            for d_idx, c in enumerate(active_clusters):
                pos_dist = math.hypot(c.norm_cx - pred_x, (c.norm_cy - pred_y) * 1.25)
                # Cosine distance on upper-chassis appearance
                s_emb = self.slot_embeddings[s_idx]
                d_emb = det_embeddings[d_idx]
                if np.linalg.norm(s_emb) > 1e-5 and np.linalg.norm(d_emb) > 1e-5:
                    cos_sim = float(np.dot(s_emb, d_emb))
                    app_dist = max(0.0, 1.0 - cos_sim)
                else:
                    app_dist = 0.25

                cost_matrix[s_idx, d_idx] = 0.72 * pos_dist + 0.28 * app_dist

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned: Dict[int, RobotCluster] = {}
        img_h, img_w = img_bgr.shape[:2]

        for s_idx, d_idx in zip(row_ind, col_ind):
            c = active_clusters[d_idx]
            # Reject extreme teleport jumps if slot was just seen, unless it's early initialization
            if self.slot_last_seen_frame[s_idx] >= 0 and (frame_idx - self.slot_last_seen_frame[s_idx]) <= 3:
                dist = math.hypot(
                    c.norm_cx - self.slot_positions[s_idx][0],
                    c.norm_cy - self.slot_positions[s_idx][1],
                )
                if dist > 0.42:
                    continue

            # Update slot kinematics & appearance EMA
            if self.slot_last_seen_frame[s_idx] >= 0:
                dt = max(1, frame_idx - self.slot_last_seen_frame[s_idx])
                vx = (c.norm_cx - self.slot_positions[s_idx][0]) / dt
                vy = (c.norm_cy - self.slot_positions[s_idx][1]) / dt
                self.slot_velocities[s_idx] = [
                    0.6 * self.slot_velocities[s_idx][0] + 0.4 * vx,
                    0.6 * self.slot_velocities[s_idx][1] + 0.4 * vy,
                ]
            self.slot_positions[s_idx] = [c.norm_cx, c.norm_cy]
            if np.linalg.norm(self.slot_embeddings[s_idx]) < 1e-5:
                self.slot_embeddings[s_idx] = det_embeddings[d_idx]
            else:
                 blended = 0.85 * self.slot_embeddings[s_idx] + 0.15 * det_embeddings[d_idx]
                 norm = float(np.linalg.norm(blended))
                 if norm > 1e-5:
                     self.slot_embeddings[s_idx] = blended / norm

            self.slot_last_seen_frame[s_idx] = frame_idx
            assigned[s_idx] = c

            # Early match starting-station spatial anchor (robots start lined up top-to-bottom)
            if timestamp_sec <= 12.0:
                # Sort order along Y axis correlates with station 1, 2, 3
                if c.norm_cy < 0.44:
                    self.evidence_matrix[s_idx, 0] += 0.08
                elif c.norm_cy <= 0.68:
                    self.evidence_matrix[s_idx, 1] += 0.08
                else:
                    self.evidence_matrix[s_idx, 2] += 0.08

            # Run Closed-Set Bumper OCR on sharp, sufficiently wide bumper crops
            bw_px = c.x2 - c.x1
            bh_px = c.y2 - c.y1
            if run_ocr and bw_px >= 36 and bh_px >= 12:
                crop = img_bgr[max(0, int(c.y1)) : min(img_h, int(c.y2)), max(0, int(c.x1)) : min(img_w, int(c.x2))]
                if crop.size > 0:
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
                    if sharpness >= 80.0:
                        raw_txt, cand_scores = self.ocr_engine.read_bumper_crop(crop, self.team_numbers)
                        if raw_txt:
                            self.slot_ocr_reads[s_idx].append(raw_txt)
                        for t_idx, num in enumerate(self.team_numbers):
                            self.evidence_matrix[s_idx, t_idx] += cand_scores.get(num, 0.0)

            self.slot_history[s_idx].append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "x1": c.x1,
                    "y1": c.y1,
                    "x2": c.x2,
                    "y2": c.y2,
                    "norm_cx": c.norm_cx,
                    "norm_cy": c.norm_cy,
                    "conf": c.max_conf,
                }
            )

        return assigned

    def resolve_bijective_team_assignment(self) -> Dict[int, Dict[str, Any]]:
        """Solves the 1-to-1 assignment between the 3 tracked slots and 3 TBA alliance teams."""
        # Maximize evidence -> minimize negative evidence
        row_ind, col_ind = linear_sum_assignment(-self.evidence_matrix)
        mapping: Dict[int, Dict[str, Any]] = {}

        row_sums = self.evidence_matrix.sum(axis=1, keepdims=True)
        probs = self.evidence_matrix / np.maximum(1e-6, row_sums)

        ocr_resolved_count = sum(1 for s in range(3) if len(self.slot_ocr_reads[s]) > 0)

        for s_idx, t_idx in zip(row_ind, col_ind):
            team_key = self.team_keys[t_idx]
            team_num = self.team_numbers[t_idx]
            conf = float(probs[s_idx, t_idx])
            reads = self.slot_ocr_reads[s_idx]
            best_read = max(reads, key=lambda r: closed_set_string_similarity(r, team_num)) if reads else ""

            if best_read and closed_set_string_similarity(best_read, team_num) >= 0.5:
                method = "ClosedSet_OCR+ReID"
                conf = max(conf, 0.86)
            elif ocr_resolved_count >= 2:
                method = "Hungarian_Elimination+ReID"
                conf = max(conf, 0.78)
            else:
                method = "Station_Prior+Kalman_ReID"
                conf = max(conf, 0.65)

            mapping[int(s_idx)] = {
                "slot_idx": int(s_idx),
                "station_slot": int(t_idx) + 1,
                "team_key": team_key,
                "team_number": team_num,
                "id_confidence": round(min(0.99, conf), 3),
                "ocr_reads_count": len(reads),
                "best_ocr_raw_text": best_read,
                "id_method": method,
            }
        return mapping


def draw_tactical_minimap(
    canvas: np.ndarray,
    red_slots: Dict[int, RobotCluster],
    blue_slots: Dict[int, RobotCluster],
    red_map: Dict[int, Dict[str, Any]],
    blue_map: Dict[int, Dict[str, Any]],
    fuel_boxes: List[DetectionBox],
    red_trails: Dict[int, List[Tuple[float, float]]],
    blue_trails: Dict[int, List[Tuple[float, float]]],
) -> None:
    """Draws a 2D Top-Down FRC Field Tactical Minimap Radar in the bottom-right corner of the frame."""
    h, w = canvas.shape[:2]
    map_w = int(w * 0.28)
    map_h = int(map_w * 0.48)
    margin = 16
    x0 = w - map_w - margin
    y0 = h - map_h - margin

    # Semi-transparent dark field carpet background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + map_w, y0 + map_h), (24, 30, 38), -1)
    cv2.addWeighted(overlay, 0.86, canvas, 0.14, 0, canvas)

    # Field border, center line, and alliance zones
    cv2.rectangle(canvas, (x0, y0), (x0 + map_w, y0 + map_h), (180, 190, 205), 2)
    mid_x = x0 + map_w // 2
    red_zone_x = x0 + int(map_w * 0.30)
    blue_zone_x = x0 + int(map_w * 0.70)

    cv2.line(canvas, (mid_x, y0), (mid_x, y0 + map_h), (110, 120, 135), 1, cv2.LINE_AA)
    cv2.line(canvas, (red_zone_x, y0), (red_zone_x, y0 + map_h), (60, 60, 180), 1, cv2.LINE_AA)
    cv2.line(canvas, (blue_zone_x, y0), (blue_zone_x, y0 + map_h), (180, 90, 40), 1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        "2D TACTICAL RADAR (TOP-DOWN)",
        (x0 + 8, y0 + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (210, 220, 235),
        1,
        cv2.LINE_AA,
    )

    def to_map_pt(norm_x: float, norm_y: float) -> Tuple[int, int]:
        # Map broadcast field Y [0.15, 0.95] to top-down radar Y [0.12, 0.92]
        ny = np.clip((norm_y - 0.14) / 0.82, 0.06, 0.94)
        nx = np.clip(norm_x, 0.04, 0.96)
        return int(x0 + nx * map_w), int(y0 + ny * map_h)

    # Draw fuel dots on minimap
    for f in fuel_boxes:
        fx, fy = to_map_pt(f.norm_cx, f.norm_cy)
        cv2.circle(canvas, (fx, fy), 3, (0, 215, 255), -1, cv2.LINE_AA)

    # Draw trails & robot markers with team numbers
    for slots, id_map, trails, col_bgr in [
        (red_slots, red_map, red_trails, (50, 50, 245)),
        (blue_slots, blue_map, blue_trails, (245, 120, 40)),
    ]:
        for s_idx, trail in trails.items():
            if len(trail) >= 2:
                pts = [to_map_pt(px, py) for px, py in trail[-20:]]
                for i in range(1, len(pts)):
                    cv2.line(canvas, pts[i - 1], pts[i], col_bgr, 1, cv2.LINE_AA)

        for s_idx, cluster in slots.items():
            rx, ry = to_map_pt(cluster.norm_cx, cluster.norm_cy)
            team_num = id_map.get(s_idx, {}).get("team_number", f"S{s_idx+1}")
            cv2.circle(canvas, (rx, ry), 8, col_bgr, -1, cv2.LINE_AA)
            cv2.circle(canvas, (rx, ry), 8, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                team_num,
                (rx - 12, ry - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


class MatchVideoIdentifier:
    """Runs two-pass robot identification, telemetry extraction, and annotated video rendering."""

    def __init__(
        self,
        model_path: Path,
        tba_cache_dir: Path,
        ssd_output_dir: Path,
        repo_output_dir: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        imgsz: int = 1280,
    ):
        self.model_path = model_path
        self.tba_cache_dir = tba_cache_dir
        self.ssd_output_dir = ssd_output_dir
        self.repo_output_dir = repo_output_dir
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = YOLO(str(self.model_path))
        self.ocr_engine = ClosedSetBumperOCR()

        self.annotated_videos_dir = self.ssd_output_dir / "annotated_videos"
        self.ssd_data_dir = self.ssd_output_dir / "data"
        self.repo_data_dir = self.repo_output_dir / "data"
        self.trajectories_dir = self.ssd_output_dir / "trajectories"

        for d in [self.annotated_videos_dir, self.ssd_data_dir, self.repo_data_dir, self.trajectories_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def lookup_tba_match_metadata(self, event_key: str, match_key: str) -> Dict[str, Any]:
        """Loads cached TBA match metadata (team rosters, scores, tower climbs, hubScore)."""
        cache_file = self.tba_cache_dir / f"event_{event_key}_matches.json"
        default_meta = {
            "red_teams": ["frcRed1", "frcRed2", "frcRed3"],
            "blue_teams": ["frcBlue1", "frcBlue2", "frcBlue3"],
            "red_score": 0,
            "blue_score": 0,
            "winning_alliance": "",
            "red_breakdown": {},
            "blue_breakdown": {},
        }
        if not cache_file.is_file():
            return default_meta

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                matches = json.load(f)
            if not isinstance(matches, list):
                return default_meta
            for m in matches:
                if m.get("key") == match_key:
                    alliances = m.get("alliances") or {}
                    red_info = alliances.get("red") or {}
                    blue_info = alliances.get("blue") or {}
                    sb = m.get("score_breakdown") or {}
                    return {
                        "red_teams": red_info.get("team_keys") or default_meta["red_teams"],
                        "blue_teams": blue_info.get("team_keys") or default_meta["blue_teams"],
                        "red_score": int(red_info.get("score") or 0),
                        "blue_score": int(blue_info.get("score") or 0),
                        "winning_alliance": m.get("winning_alliance") or "",
                        "red_breakdown": sb.get("red") or {},
                        "blue_breakdown": sb.get("blue") or {},
                    }
        except Exception:
            pass
        return default_meta

    def process_video(
        self,
        video_path: Path,
        frame_stride: int = 3,
        render_video: bool = True,
    ) -> Tuple[List[RobotTelemetrySummary], List[Dict[str, Any]]]:
        """Processes a full match video, identifies all 6 robots (`frcXXXX`), and renders annotated MP4."""
        meta = parse_video_stem_metadata(video_path.stem)
        tba_info = self.lookup_tba_match_metadata(meta["event_key"], meta["match_key"])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[RobotID] Warning: Could not open video {video_path}")
            return [], []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        effective_fps = fps / max(1, frame_stride)

        red_tracker = AllianceRobotTracker("red", tba_info["red_teams"], self.ocr_engine)
        blue_tracker = AllianceRobotTracker("blue", tba_info["blue_teams"], self.ocr_engine)

        # Store per-sampled-frame detections for Pass 2 video rendering & telemetry
        sampled_records: List[Dict[str, Any]] = []
        frame_idx = 0
        sampled_idx = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                timestamp_sec = frame_idx / fps
                preds = self.model.predict(
                    source=frame_bgr,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
                pred = preds[0]
                boxes_tensor = pred.boxes
                detections: List[DetectionBox] = []

                if boxes_tensor is not None and len(boxes_tensor) > 0:
                    xyxy = boxes_tensor.xyxy.cpu().numpy()
                    confs = boxes_tensor.conf.cpu().numpy()
                    clss = boxes_tensor.cls.cpu().numpy().astype(int)

                    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
                        cx = 0.5 * (x1 + x2)
                        cy = 0.5 * (y1 + y2)
                        norm_cx = cx / img_w
                        norm_cy = cy / img_h
                        if norm_cy < 0.11:
                            continue
                        detections.append(
                            DetectionBox(
                                cls_id=int(cls_id),
                                cls_name=CLASS_NAMES.get(int(cls_id), f"cls_{cls_id}"),
                                conf=round(float(conf), 4),
                                x1=round(float(x1), 1),
                                y1=round(float(y1), 1),
                                x2=round(float(x2), 1),
                                y2=round(float(y2), 1),
                                cx=round(float(cx), 1),
                                cy=round(float(cy), 1),
                                w=round(float(x2 - x1), 1),
                                h=round(float(y2 - y1), 1),
                                norm_cx=round(float(norm_cx), 4),
                                norm_cy=round(float(norm_cy), 4),
                                norm_area=round(float((x2 - x1) * (y2 - y1) / (img_w * img_h)), 6),
                            )
                        )

                red_clusters = cluster_alliance_bumpers(detections, 0, img_w, img_h)
                blue_clusters = cluster_alliance_bumpers(detections, 1, img_w, img_h)
                fuel_boxes = [d for d in detections if d.cls_id == 2]

                # Run OCR periodically (every 5th sampled frame) to keep video processing fast
                do_ocr = (sampled_idx % 5 == 0)
                red_assigned = red_tracker.update(sampled_idx, timestamp_sec, frame_bgr, red_clusters, run_ocr=do_ocr)
                blue_assigned = blue_tracker.update(sampled_idx, timestamp_sec, frame_bgr, blue_clusters, run_ocr=do_ocr)

                sampled_records.append(
                    {
                        "frame_idx": frame_idx,
                        "sampled_idx": sampled_idx,
                        "timestamp_sec": round(timestamp_sec, 3),
                        "red_assigned": red_assigned,
                        "blue_assigned": blue_assigned,
                        "fuel_boxes": fuel_boxes,
                    }
                )
                sampled_idx += 1

            frame_idx += 1

        cap.release()

        # Solve global bijective team-number assignment for all 6 robots
        red_id_map = red_tracker.resolve_bijective_team_assignment()
        blue_id_map = blue_tracker.resolve_bijective_team_assignment()

        annotated_video_out = self.annotated_videos_dir / f"{video_path.stem}_annotated.mp4"
        if render_video and sampled_records:
            self._render_annotated_video(
                video_path=video_path,
                output_path=annotated_video_out,
                sampled_records=sampled_records,
                red_id_map=red_id_map,
                blue_id_map=blue_id_map,
                meta=meta,
                tba_info=tba_info,
                frame_stride=frame_stride,
                fps=fps,
            )

        # Compute comprehensive per-robot telemetry & trajectory logs
        robot_summaries, trajectory_rows = self._compute_per_robot_diagnostics(
            meta=meta,
            tba_info=tba_info,
            sampled_records=sampled_records,
            red_tracker=red_tracker,
            blue_tracker=blue_tracker,
            red_id_map=red_id_map,
            blue_id_map=blue_id_map,
            effective_fps=effective_fps,
            annotated_video_path=str(annotated_video_out) if render_video else "",
        )
        return robot_summaries, trajectory_rows

    def _render_annotated_video(
        self,
        video_path: Path,
        output_path: Path,
        sampled_records: List[Dict[str, Any]],
        red_id_map: Dict[int, Dict[str, Any]],
        blue_id_map: Dict[int, Dict[str, Any]],
        meta: Dict[str, Any],
        tba_info: Dict[str, Any],
        frame_stride: int,
        fps: float,
    ) -> None:
        """Pass 2: Renders the full annotated match video with locked `frcXXXX` IDs and 2D radar."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        out_fps = max(5.0, fps / max(1, frame_stride))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (w, h))

        rec_by_frame = {r["frame_idx"]: r for r in sampled_records}
        red_trails: Dict[int, List[Tuple[float, float]]] = {0: [], 1: [], 2: []}
        blue_trails: Dict[int, List[Tuple[float, float]]] = {0: [], 1: [], 2: []}

        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx in rec_by_frame:
                rec = rec_by_frame[frame_idx]
                canvas = frame_bgr.copy()
                red_slots: Dict[int, RobotCluster] = rec["red_assigned"]
                blue_slots: Dict[int, RobotCluster] = rec["blue_assigned"]
                fuel_boxes: List[DetectionBox] = rec["fuel_boxes"]

                # Draw fuel detections
                for f in fuel_boxes:
                    cv2.rectangle(
                        canvas,
                        (int(f.x1), int(f.y1)),
                        (int(f.x2), int(f.y2)),
                        BGR_COLORS[2],
                        2,
                    )

                # Update trails and draw robot bounding boxes with `frcXXXX` team labels
                for slots, id_map, trails, col_bgr in [
                    (red_slots, red_id_map, red_trails, BGR_COLORS[0]),
                    (blue_slots, blue_id_map, blue_trails, BGR_COLORS[1]),
                ]:
                    for s_idx, c in slots.items():
                        trails[s_idx].append((c.norm_cx, c.norm_cy))
                        if len(trails[s_idx]) >= 2:
                            pts = [(int(px * w), int(py * h)) for px, py in trails[s_idx][-18:]]
                            for i in range(1, len(pts)):
                                cv2.line(canvas, pts[i - 1], pts[i], col_bgr, 2, cv2.LINE_AA)

                        info = id_map.get(s_idx, {})
                        team_key = info.get("team_key", f"Slot{s_idx+1}")
                        id_conf = info.get("id_confidence", 0.75)

                        cv2.rectangle(
                            canvas,
                            (int(c.x1), int(c.y1)),
                            (int(c.x2), int(c.y2)),
                            col_bgr,
                            2,
                        )
                        label = f"{team_key} ({id_conf*100:.0f}%)"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)
                        ty = max(th + 8, int(c.y1) - 6)
                        cv2.rectangle(
                            canvas,
                            (int(c.x1), ty - th - 6),
                            (int(c.x1) + tw + 8, ty + 2),
                            col_bgr,
                            -1,
                        )
                        cv2.putText(
                            canvas,
                            label,
                            (int(c.x1) + 4, ty - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.54,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                # Highlight active defense engagements (opposing robots within contact range)
                defense_active = 0
                for rc in red_slots.values():
                    for bc in blue_slots.values():
                        dist = math.hypot(rc.norm_cx - bc.norm_cx, rc.norm_cy - bc.norm_cy)
                        if dist < 0.11:
                            defense_active += 1
                            cv2.line(
                                canvas,
                                (int(rc.cx), int(rc.cy)),
                                (int(bc.cx), int(bc.cy)),
                                (0, 140, 255),
                                2,
                                cv2.LINE_AA,
                            )

                # Draw 2D Top-Down Tactical Minimap Radar
                draw_tactical_minimap(
                    canvas=canvas,
                    red_slots=red_slots,
                    blue_slots=blue_slots,
                    red_map=red_id_map,
                    blue_map=blue_id_map,
                    fuel_boxes=fuel_boxes,
                    red_trails=red_trails,
                    blue_trails=blue_trails,
                )

                # Draw top broadcast telemetry HUD
                overlay = canvas.copy()
                cv2.rectangle(overlay, (0, 0), (w, 46), (15, 20, 32), -1)
                cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

                r_roster = "/".join(t.replace("frc", "") for t in tba_info["red_teams"])
                b_roster = "/".join(t.replace("frc", "") for t in tba_info["blue_teams"])
                hud_line = (
                    f"{meta['match_key']} | t={rec['timestamp_sec']:.1f}s | "
                    f"RED [{r_roster}] ({tba_info['red_score']} pts) vs "
                    f"BLUE [{b_roster}] ({tba_info['blue_score']} pts) | "
                    f"Fuel: {len(fuel_boxes)} | Defense Contacts: {defense_active}"
                )
                cv2.putText(
                    canvas,
                    hud_line,
                    (14, 29),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (245, 248, 255),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(canvas)

            frame_idx += 1

        cap.release()
        writer.release()

    def _compute_per_robot_diagnostics(
        self,
        meta: Dict[str, Any],
        tba_info: Dict[str, Any],
        sampled_records: List[Dict[str, Any]],
        red_tracker: AllianceRobotTracker,
        blue_tracker: AllianceRobotTracker,
        red_id_map: Dict[int, Dict[str, Any]],
        blue_id_map: Dict[int, Dict[str, Any]],
        effective_fps: float,
        annotated_video_path: str,
    ) -> Tuple[List[RobotTelemetrySummary], List[Dict[str, Any]]]:
        """Computes per-robot kinematics, cycles, defense attribution, and TBA correlations."""
        summaries: List[RobotTelemetrySummary] = []
        trajectory_rows: List[Dict[str, Any]] = []
        total_sampled = max(1, len(sampled_records))
        dt_sec = 1.0 / max(1.0, effective_fps)

        # Build frame-by-frame lookup for opposing alliance positions and fuel
        for tracker, opp_tracker, id_map, opp_id_map, alliance_str in [
            (red_tracker, blue_tracker, red_id_map, blue_id_map, "red"),
            (blue_tracker, red_tracker, blue_id_map, red_id_map, "blue"),
        ]:
            sb = tba_info["red_breakdown"] if alliance_str == "red" else tba_info["blue_breakdown"]
            alliance_score = tba_info["red_score"] if alliance_str == "red" else tba_info["blue_score"]
            alliance_won = tba_info["winning_alliance"] == alliance_str

            for s_idx in range(3):
                info = id_map[s_idx]
                team_key = info["team_key"]
                team_num = info["team_number"]
                station_num = info["station_slot"]
                hist = tracker.slot_history[s_idx]

                for pt in hist:
                    trajectory_rows.append(
                        {
                            "match_key": meta["match_key"],
                            "event_key": meta["event_key"],
                            "year": meta["year"],
                            "alliance": alliance_str,
                            "team_key": team_key,
                            "team_number": team_num,
                            "station_slot": station_num,
                            **pt,
                        }
                    )

                frames_vis = len(hist)
                vis_rate = frames_vis / float(total_sampled)
                mean_conf = float(np.mean([p["conf"] for p in hist])) if hist else 0.0

                # Kinematics & distance (FRC field is ~54 feet long)
                total_dist_norm = 0.0
                speeds_fps: List[float] = []
                stationary_frames = 0

                for i in range(1, len(hist)):
                    dx = hist[i]["norm_cx"] - hist[i - 1]["norm_cx"]
                    dy = hist[i]["norm_cy"] - hist[i - 1]["norm_cy"]
                    f_gap = max(1, hist[i]["frame_idx"] - hist[i - 1]["frame_idx"])
                    step_dist = math.hypot(dx, dy)
                    # Ignore camera cut jumps
                    if step_dist < 0.35:
                        total_dist_norm += step_dist
                        inst_speed_fps = (step_dist * 54.0) / (f_gap * dt_sec)
                        speeds_fps.append(inst_speed_fps)
                        if inst_speed_fps < 1.2:
                            stationary_frames += 1

                est_dist_feet = total_dist_norm * 54.0
                mean_speed = float(np.mean(speeds_fps)) if speeds_fps else 0.0
                max_speed = float(np.percentile(speeds_fps, 95)) if len(speeds_fps) >= 5 else (max(speeds_fps) if speeds_fps else 0.0)
                stat_sec = stationary_frames * dt_sec
                is_impaired = (est_dist_feet < 35.0) or (frames_vis > 10 and (stationary_frames / max(1, len(speeds_fps))) > 0.68)

                xs = [p["norm_cx"] for p in hist]
                ys = [p["norm_cy"] for p in hist]
                mean_x = float(np.mean(xs)) if xs else 0.5
                mean_y = float(np.mean(ys)) if ys else 0.5
                std_x = float(np.std(xs)) if xs else 0.0
                std_y = float(np.std(ys)) if ys else 0.0

                # Zone breakdown (Red own zone: x < 0.36, Neutral: 0.36..0.64, Blue own zone: x > 0.64)
                if alliance_str == "red":
                    own_pct = sum(1 for x in xs if x < 0.36) / max(1, len(xs))
                    enemy_pct = sum(1 for x in xs if x > 0.64) / max(1, len(xs))
                else:
                    own_pct = sum(1 for x in xs if x > 0.64) / max(1, len(xs))
                    enemy_pct = sum(1 for x in xs if x < 0.36) / max(1, len(xs))
                neutral_pct = max(0.0, 1.0 - own_pct - enemy_pct)

                # Cycle estimation: count transitions between neutral/pickup zone and alliance scoring zone
                cycles = 0
                in_pickup = False
                for x in xs:
                    in_neutral_or_far = (x > 0.45) if alliance_str == "red" else (x < 0.55)
                    in_scoring = (x < 0.32) if alliance_str == "red" else (x > 0.68)
                    if in_neutral_or_far:
                        in_pickup = True
                    elif in_pickup and in_scoring:
                        cycles += 1
                        in_pickup = False

                match_duration_tracked = max(15.0, frames_vis * dt_sec)
                mean_cycle_time = round(match_duration_tracked / cycles, 2) if cycles > 0 else 0.0

                # Fuel proximity & opposing robot defense attribution
                fuel_prox_frames = 0
                defense_frames = 0
                opp_contact_counts: Dict[str, int] = {}

                rec_by_sidx = {r["sampled_idx"]: r for r in sampled_records}
                for p in hist:
                    rec = rec_by_sidx.get(p["frame_idx"])
                    if not rec:
                        continue
                    # Fuel proximity
                    for fbox in rec["fuel_boxes"]:
                        if math.hypot(p["norm_cx"] - fbox.norm_cx, p["norm_cy"] - fbox.norm_cy) < 0.09:
                            fuel_prox_frames += 1
                            break
                    # Opposing robot proximity
                    opp_slots = rec["blue_assigned"] if alliance_str == "red" else rec["red_assigned"]
                    for opp_sidx, opp_c in opp_slots.items():
                        if math.hypot(p["norm_cx"] - opp_c.norm_cx, p["norm_cy"] - opp_c.norm_cy) < 0.11:
                            defense_frames += 1
                            opp_team = opp_id_map.get(opp_sidx, {}).get("team_key", "unknown")
                            opp_contact_counts[opp_team] = opp_contact_counts.get(opp_team, 0) + 1

                defense_sec = defense_frames * dt_sec
                primary_opp = max(opp_contact_counts, key=opp_contact_counts.get) if opp_contact_counts else "None"

                # Role classification
                if is_impaired:
                    role = "Impaired / Low Mobility"
                elif enemy_pct > 0.32 and defense_sec >= 4.0:
                    role = "Dedicated Defender"
                elif cycles >= 2 or (own_pct > 0.25 and neutral_pct > 0.25 and mean_speed >= 4.5):
                    role = "Primary Cycler"
                else:
                    role = "Feeder / Hybrid Support"

                summaries.append(
                    RobotTelemetrySummary(
                        match_key=meta["match_key"],
                        event_key=meta["event_key"],
                        year=meta["year"],
                        video_stem=meta.get("video_stem", meta["match_key"]),
                        alliance=alliance_str,
                        station_slot=station_num,
                        team_key=team_key,
                        team_number=team_num,
                        frames_visible=frames_vis,
                        total_sampled_frames=total_sampled,
                        visibility_rate=round(vis_rate, 4),
                        mean_yolo_conf=round(mean_conf, 4),
                        id_confidence=info["id_confidence"],
                        ocr_reads_count=info["ocr_reads_count"],
                        best_ocr_raw_text=info["best_ocr_raw_text"],
                        id_method=info["id_method"],
                        total_distance_norm=round(total_dist_norm, 3),
                        estimated_distance_feet=round(est_dist_feet, 1),
                        mean_speed_fps=round(mean_speed, 2),
                        max_speed_fps=round(max_speed, 2),
                        stationary_seconds=round(stat_sec, 1),
                        is_dead_or_impaired=is_impaired,
                        mean_norm_x=round(mean_x, 4),
                        mean_norm_y=round(mean_y, 4),
                        std_norm_x=round(std_x, 4),
                        std_norm_y=round(std_y, 4),
                        time_in_own_zone_pct=round(own_pct * 100, 1),
                        time_in_neutral_zone_pct=round(neutral_pct * 100, 1),
                        time_in_enemy_zone_pct=round(enemy_pct * 100, 1),
                        estimated_cycles=cycles,
                        mean_cycle_time_sec=mean_cycle_time,
                        fuel_proximity_frames=fuel_prox_frames,
                        defense_engagement_sec=round(defense_sec, 1),
                        primary_opponent_engaged=primary_opp,
                        classified_role=role,
                        alliance_Final_score=alliance_score,
                        alliance_won=alliance_won,
                        tba_auto_tower_status=str(sb.get(f"autoTowerRobot{station_num}", "None")),
                        tba_endgame_tower_status=str(sb.get(f"endGameTowerRobot{station_num}", "None")),
                        tba_energized_achieved=bool(sb.get("energizedAchieved", False)),
                        tba_supercharged_achieved=bool(sb.get("superchargedAchieved", False)),
                        tba_traversal_achieved=bool(sb.get("traversalAchieved", False)),
                        annotated_video_path=annotated_video_path,
                    )
                )

        return summaries, trajectory_rows

    def run_batch(
        self,
        videos_root: Path,
        max_videos: int = 12,
        frame_stride: int = 4,
        render_videos: bool = True,
    ) -> Tuple[List[RobotTelemetrySummary], List[Dict[str, Any]]]:
        """Runs full-match robot identification and video annotation across downloaded MP4s."""
        video_files: List[Path] = []
        for split in ["frc_this_year", "frc_multi_year"]:
            split_dir = videos_root / split
            if not split_dir.is_dir():
                continue
            mp4s = [
                p for p in sorted(split_dir.glob("*.mp4"))
                if not p.name.startswith("._") and not p.name.endswith(".part")
            ]
            video_files.extend(mp4s)

        if max_videos > 0:
            video_files = video_files[:max_videos]

        print(f"[RobotID] Processing {len(video_files)} full match videos (stride={frame_stride}, render={render_videos})...")
        all_summaries: List[RobotTelemetrySummary] = []
        all_trajectories: List[Dict[str, Any]] = []

        for idx, vpath in enumerate(video_files, start=1):
            t0 = time.time()
            print(f"[RobotID] ({idx}/{len(video_files)}) Tracking & Identifying 6 robots in {vpath.name}...")
            summaries, traj = self.process_video(
                video_path=vpath,
                frame_stride=frame_stride,
                render_video=render_videos,
            )
            all_summaries.extend(summaries)
            all_trajectories.extend(traj)
            elapsed = time.time() - t0
            teams_str = ", ".join(f"{s.team_key}({s.classified_role})" for s in summaries)
            print(f"          -> Completed in {elapsed:.1f}s | Identified: {teams_str}")

        self._save_batch_outputs(all_summaries, all_trajectories)
        return all_summaries, all_trajectories

    def _save_batch_outputs(
        self,
        summaries: List[RobotTelemetrySummary],
        trajectories: List[Dict[str, Any]],
    ) -> None:
        summary_dicts = [asdict(s) for s in summaries]
        for target_dir in [self.ssd_data_dir, self.repo_data_dir]:
            if summary_dicts:
                csv_path = target_dir / "per_robot_match_telemetry.csv"
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(summary_dicts[0].keys()))
                    writer.writeheader()
                    writer.writerows(summary_dicts)

                json_path = target_dir / "per_robot_match_telemetry.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(summary_dicts, f, indent=2)

            if trajectories:
                traj_csv = target_dir / "all_robot_trajectories.csv"
                with open(traj_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(trajectories[0].keys()))
                    writer.writeheader()
                    writer.writerows(trajectories)

        print(f"[RobotID] Saved telemetry for {len(summary_dicts)} individual robots and {len(trajectories)} trajectory points.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Closed-Set TBA Robot Identification & Video Annotation.")
    parser.add_argument("--model", type=str, default="models/frc_yolo11s_best.pt")
    parser.add_argument("--ssd-root", type=str, default="./data")
    parser.add_argument("--repo-out", type=str, default="analytics_outputs")
    parser.add_argument("--max-videos", type=int, default=12, help="Number of full match videos to process (0 = all)")
    parser.add_argument("--stride", type=int, default=4, help="Frame sampling stride for video tracking")
    parser.add_argument("--no-render", action="store_true", help="Skip writing annotated MP4 files")
    args = parser.parse_args()

    ssd_root = Path(args.ssd_root)
    identifier = MatchVideoIdentifier(
        model_path=Path(args.model),
        tba_cache_dir=Path(".cache/tba"),
        ssd_output_dir=ssd_root / "frc_vision_analytics",
        repo_output_dir=Path(args.repo_out),
    )
    identifier.run_batch(
        videos_root=ssd_root,
        max_videos=args.max_videos,
        frame_stride=args.stride,
        render_videos=not args.no_render,
    )


if __name__ == "__main__":
    main()
