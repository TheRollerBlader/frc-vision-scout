"""Zero-Label YOLO11 Model Accuracy & FRC Rule-Invariant Evaluation Engine.

This module evaluates the trained FRC YOLO11s object detection model
(`{0: 'bumper_red', 1: 'bumper_blue', 2: 'fuel'}`) on extracted match frames
without requiring manual bounding-box annotations.

Evaluation Pillars:
1. FRC Physical Rule Invariants:
   - Clusters adjacent same-alliance bumper detections into distinct Robot Footprints.
   - Enforces the physical FRC match limit (<= 3 Red robots, <= 3 Blue robots).
   - Flags over-count False Positives (> 3 robot clusters per alliance) and
     out-of-field detections (audience/scoreboard regions).
   - Estimates Full-Field Recall on mid-match broadcast frames.
2. Cross-Year Negative Control (`frc_multi_year` 2021-2025):
   - Tests 2026 `fuel` False Positive rates on 2021-2025 matches where 2026 fuel
     did not exist.
   - Evaluates `bumper_red` and `bumper_blue` domain generalization across 5 years
     of diverse FRC broadcast cameras and venue lighting.
3. HSV Bumper Color-Purity Audit:
   - Crops every predicted red and blue bumper bounding box and measures HSV
     red vs. blue pixel ratios to automatically catch class-swap errors or extreme
     venue white-balance shifts.
4. Active Learning Hard-Example Mining:
   - Ranks frames by model uncertainty and rule violations for rapid human review.
"""

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.cluster.hierarchy import fclusterdata
from ultralytics import YOLO


# Class definitions from trained YOLO11s model
CLASS_NAMES = {0: "bumper_red", 1: "bumper_blue", 2: "fuel"}
BGR_COLORS = {
    0: (40, 40, 235),    # Red bumper (BGR)
    1: (235, 110, 30),   # Blue bumper (BGR)
    2: (0, 215, 255),    # Fuel - Gold/Yellow (BGR)
}


@dataclass
class DetectionBox:
    """Single YOLO bounding box prediction with HSV color-purity diagnostics."""
    cls_id: int
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    w: float
    h: float
    norm_cx: float
    norm_cy: float
    norm_area: float
    red_hue_ratio: float = 0.0
    blue_hue_ratio: float = 0.0
    yellow_hue_ratio: float = 0.0
    color_purity_ok: bool = True
    in_field_bounds: bool = True
    robot_cluster_id: int = -1


@dataclass
class RobotCluster:
    """Merged robot footprint formed by 1 or more bumper detections on the same chassis."""
    cluster_id: int
    alliance: str  # "red" or "blue"
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    norm_cx: float
    norm_cy: float
    max_conf: float
    mean_conf: float
    num_bumper_segments: int
    color_purity_ok: bool


@dataclass
class FrameEvalResult:
    """Comprehensive zero-label evaluation and FRC diagnostic record for a single frame."""
    split: str                  # "frc_this_year" or "frc_multi_year"
    year: int
    event_key: str
    match_key: str
    comp_level: str
    match_number: int
    youtube_id: str
    video_stem: str
    frame_label: str            # "start", "middle_1", "middle_2", "middle_3", "end"
    frame_index: int
    image_path: str
    annotated_path: str
    img_w: int
    img_h: int
    # Raw detection counts
    raw_red_bumpers: int
    raw_blue_bumpers: int
    raw_fuel_count: int
    # Clustered robot counts
    red_robot_clusters: int
    blue_robot_clusters: int
    total_robot_clusters: int
    # Invariant checks
    red_count_valid: bool       # <= 3 red robot clusters
    blue_count_valid: bool      # <= 3 blue robot clusters
    red_excess_fp: int          # max(0, red_robot_clusters - 3)
    blue_excess_fp: int         # max(0, blue_robot_clusters - 3)
    out_of_field_fp: int        # detections in top scoreboard/audience margin
    color_swap_suspects: int    # red box with blue HSV or vice versa
    cross_year_fuel_fp: int     # fuel detected in pre-2026 match
    # Confidence & image quality metrics
    mean_bumper_conf: float
    mean_fuel_conf: float
    low_conf_detections: int    # detections with 0.25 <= conf < 0.45
    brightness_mean: float
    contrast_std: float
    glare_ratio: float          # fraction of overexposed pixels (V > 245, S < 30)
    sharpness_laplacian: float
    # Spatial FRC gameplay metrics
    red_mean_norm_x: float
    blue_mean_norm_x: float
    territory_incursion_red: int   # red robots on blue half (norm_x > 0.5)
    territory_incursion_blue: int  # blue robots on red half (norm_x < 0.5)
    min_opposing_robot_dist_norm: float
    defense_engagements: int       # pairs of opposing robots within proximity threshold
    fuel_left_half: int
    fuel_right_half: int
    fuel_center_zone: int
    # Active learning priority score
    uncertainty_score: float


def parse_video_stem_metadata(video_stem: str) -> Dict[str, Any]:
    """Extracts year, event_key, match_key, comp_level, match_number, youtube_id from filename stem."""
    # Example stem: 2026_2026njski_2026njski_qm24_xmbWoU1KwB4
    parts = video_stem.split("_")
    year = int(parts[0]) if parts and parts[0].isdigit() else 2026
    event_key = parts[1] if len(parts) > 1 else "unknown"
    match_token = parts[3] if len(parts) > 3 else "qm1"
    match_key = f"{event_key}_{match_token}"
    youtube_id = "_".join(parts[4:]) if len(parts) > 4 else ""

    m = re.match(r"([a-z]+)(\d+)", match_token)
    if m:
        comp_level = m.group(1)
        match_number = int(m.group(2))
    else:
        comp_level = "qm"
        match_number = 1

    return {
        "year": year,
        "event_key": event_key,
        "match_key": match_key,
        "comp_level": comp_level,
        "match_number": match_number,
        "youtube_id": youtube_id,
    }


def compute_hsv_color_ratios(crop_bgr: np.ndarray) -> Tuple[float, float, float]:
    """Computes fraction of saturated pixels in Red, Blue, and Yellow HSV hue bands."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0, 0.0, 0.0

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Filter out dark/black or washed-out white/gray pixels
    valid_mask = (s >= 45) & (v >= 40)
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count < 10:
        return 0.0, 0.0, 0.0

    # OpenCV Hue range is [0, 179]
    # Red wraps around 0: [0, 14] U [165, 179]
    red_mask = valid_mask & ((h <= 14) | (h >= 165))
    # Blue: [95, 138]
    blue_mask = valid_mask & (h >= 95) & (h <= 138)
    # Yellow/Gold (2026 Fuel): [16, 42]
    yellow_mask = valid_mask & (h >= 16) & (h <= 42)

    red_ratio = float(np.count_nonzero(red_mask)) / valid_count
    blue_ratio = float(np.count_nonzero(blue_mask)) / valid_count
    yellow_ratio = float(np.count_nonzero(yellow_mask)) / valid_count
    return round(red_ratio, 4), round(blue_ratio, 4), round(yellow_ratio, 4)


def cluster_alliance_bumpers(
    detections: List[DetectionBox],
    alliance_cls_id: int,
    img_w: int,
    img_h: int,
    max_cluster_dist_norm: float = 0.085,
) -> List[RobotCluster]:
    """Merges nearby same-alliance bumper detections into distinct Robot Clusters.

    Because an L-shaped or 4-sided robot bumper can produce 1-2 adjacent bounding boxes
    on the same robot chassis, hierarchical clustering groups boxes whose centers are
    within `max_cluster_dist_norm` (normalized by image width) into a single robot.
    """
    alliance_str = "red" if alliance_cls_id == 0 else "blue"
    boxes = [d for d in detections if d.cls_id == alliance_cls_id and d.in_field_bounds]
    if not boxes:
        return []

    if len(boxes) == 1:
        b = boxes[0]
        b.robot_cluster_id = 1
        return [
            RobotCluster(
                cluster_id=1,
                alliance=alliance_str,
                x1=b.x1,
                y1=b.y1,
                x2=b.x2,
                y2=b.y2,
                cx=b.cx,
                cy=b.cy,
                norm_cx=b.norm_cx,
                norm_cy=b.norm_cy,
                max_conf=b.conf,
                mean_conf=b.conf,
                num_bumper_segments=1,
                color_purity_ok=b.color_purity_ok,
            )
        ]

    # Normalized coordinates with vertical aspect weighting (perspective foreshortening)
    pts = np.array([[b.norm_cx, b.norm_cy * 1.35] for b in boxes], dtype=np.float64)
    labels = fclusterdata(pts, t=max_cluster_dist_norm, criterion="distance")

    clusters: List[RobotCluster] = []
    for cid in sorted(set(labels)):
        member_boxes = [b for b, lbl in zip(boxes, labels) if lbl == cid]
        for b in member_boxes:
            b.robot_cluster_id = int(cid)

        x1 = min(b.x1 for b in member_boxes)
        y1 = min(b.y1 for b in member_boxes)
        x2 = max(b.x2 for b in member_boxes)
        y2 = max(b.y2 for b in member_boxes)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        clusters.append(
            RobotCluster(
                cluster_id=int(cid),
                alliance=alliance_str,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                cx=cx,
                cy=cy,
                norm_cx=round(cx / max(1, img_w), 4),
                norm_cy=round(cy / max(1, img_h), 4),
                max_conf=round(max(b.conf for b in member_boxes), 4),
                mean_conf=round(float(np.mean([b.conf for b in member_boxes])), 4),
                num_bumper_segments=len(member_boxes),
                color_purity_ok=all(b.color_purity_ok for b in member_boxes),
            )
        )

    # Sort clusters by confidence descending
    clusters.sort(key=lambda c: c.max_conf, reverse=True)
    for idx, c in enumerate(clusters, start=1):
        c.cluster_id = idx
    return clusters


def draw_annotated_frame(
    img_bgr: np.ndarray,
    detections: List[DetectionBox],
    red_clusters: List[RobotCluster],
    blue_clusters: List[RobotCluster],
    meta: Dict[str, Any],
    frame_label: str,
) -> np.ndarray:
    """Renders bounding boxes, robot cluster envelopes, and FRC diagnostic HUD onto a frame."""
    canvas = img_bgr.copy()
    h, w = canvas.shape[:2]

    # Draw subtle field boundary line (top 12% is typically scoreboard/audience)
    field_top_y = int(h * 0.12)
    cv2.line(canvas, (0, field_top_y), (w, field_top_y), (90, 90, 90), 1, cv2.LINE_AA)

    # Draw fuel detections first
    for d in detections:
        if d.cls_id == 2:
            color = BGR_COLORS[2]
            cv2.rectangle(
                canvas,
                (int(d.x1), int(d.y1)),
                (int(d.x2), int(d.y2)),
                color,
                2,
            )

    # Draw raw bumper boxes (thin) and merged Robot Clusters (bold with label)
    for d in detections:
        if d.cls_id in (0, 1):
            color = BGR_COLORS[d.cls_id]
            if not d.in_field_bounds or not d.color_purity_ok:
                color = (0, 0, 255) if not d.color_purity_ok else (160, 160, 160)
            cv2.rectangle(
                canvas,
                (int(d.x1), int(d.y1)),
                (int(d.x2), int(d.y2)),
                color,
                1,
            )

    for cluster_list, bgr in [(red_clusters, BGR_COLORS[0]), (blue_clusters, BGR_COLORS[1])]:
        for c in cluster_list:
            is_excess = c.cluster_id > 3
            box_color = (0, 140, 255) if is_excess else bgr
            cv2.rectangle(
                canvas,
                (int(c.x1) - 3, int(c.y1) - 3),
                (int(c.x2) + 3, int(c.y2) + 3),
                box_color,
                2,
            )
            tag = f"{c.alliance.upper()} BOT #{c.cluster_id} ({c.max_conf:.2f})"
            if is_excess:
                tag += " [FP>3!]"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(th + 6, int(c.y1) - 6)
            cv2.rectangle(
                canvas,
                (int(c.x1) - 3, ty - th - 5),
                (int(c.x1) + tw + 5, ty + 3),
                box_color,
                -1,
            )
            cv2.putText(
                canvas,
                tag,
                (int(c.x1), ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Draw top HUD banner
    hud_h = 40
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (18, 22, 34), -1)
    cv2.addWeighted(overlay, 0.84, canvas, 0.16, 0, canvas)

    fuel_cnt = sum(1 for d in detections if d.cls_id == 2)
    rule_ok = (len(red_clusters) <= 3) and (len(blue_clusters) <= 3)
    status_str = "RULES OK (<=3/alliance)" if rule_ok else "INVARIANT VIOLATION (>3)"
    hud_text = (
        f"Match: {meta['match_key']} ({meta['year']}) | Phase: {frame_label} | "
        f"Red Bots: {len(red_clusters)}/3 | Blue Bots: {len(blue_clusters)}/3 | "
        f"Fuel: {fuel_cnt} | {status_str}"
    )
    cv2.putText(
        canvas,
        hud_text,
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 245, 255) if rule_ok else (80, 190, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


class ZeroLabelEvaluator:
    """Runs batch YOLO11s evaluation and FRC invariant diagnostics across extracted frames."""

    def __init__(
        self,
        model_path: Path,
        frames_root: Path,
        ssd_output_dir: Path,
        repo_output_dir: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        imgsz: int = 1280,
        save_annotated_frames: bool = True,
    ):
        self.model_path = model_path
        self.frames_root = frames_root
        self.ssd_output_dir = ssd_output_dir
        self.repo_output_dir = repo_output_dir
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.save_annotated_frames = save_annotated_frames

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[Evaluator] Loading YOLO model from {self.model_path} on device='{self.device}'...")
        self.model = YOLO(str(self.model_path))

        self.annotated_frames_dir = self.ssd_output_dir / "annotated_frames"
        self.ssd_data_dir = self.ssd_output_dir / "data"
        self.repo_data_dir = self.repo_output_dir / "data"

        for d in [self.annotated_frames_dir, self.ssd_data_dir, self.repo_data_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def discover_frames(self) -> List[Tuple[str, str, Path]]:
        """Discovers all extracted `.jpg` frames across `frc_this_year` and `frc_multi_year`."""
        items: List[Tuple[str, str, Path]] = []
        for split in ["frc_this_year", "frc_multi_year"]:
            split_dir = self.frames_root / split
            if not split_dir.is_dir():
                continue
            for video_dir in sorted(split_dir.iterdir()):
                if not video_dir.is_dir() or video_dir.name.startswith("."):
                    continue
                for jpg in sorted(video_dir.glob("frame_*.jpg")):
                    if jpg.name.startswith("._"):
                        continue
                    items.append((split, video_dir.name, jpg))
        return items

    def evaluate_all_frames(self, batch_size: int = 16) -> Tuple[List[FrameEvalResult], List[Dict[str, Any]]]:
        """Evaluates all discovered frames using batch inference and saves annotated images + CSV/JSON."""
        items = self.discover_frames()
        print(f"[Evaluator] Discovered {len(items)} extracted frames across splits.")
        if not items:
            return [], []

        frame_results: List[FrameEvalResult] = []
        all_detections_rows: List[Dict[str, Any]] = []
        t0 = time.time()

        for start_idx in range(0, len(items), batch_size):
            batch_items = items[start_idx : start_idx + batch_size]
            batch_paths = [str(it[2]) for it in batch_items]

            preds = self.model.predict(
                source=batch_paths,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )

            for (split, video_stem, img_path), pred in zip(batch_items, preds):
                res = self._process_single_prediction(
                    split=split,
                    video_stem=video_stem,
                    img_path=img_path,
                    pred=pred,
                    all_detections_rows=all_detections_rows,
                )
                if res is not None:
                    frame_results.append(res)

            done = min(start_idx + batch_size, len(items))
            elapsed = time.time() - t0
            fps = done / max(0.01, elapsed)
            if done % 64 == 0 or done == len(items):
                print(f"[Evaluator] Processed {done}/{len(items)} frames ({fps:.1f} frames/sec)")

        self._save_results(frame_results, all_detections_rows)
        return frame_results, all_detections_rows

    def _process_single_prediction(
        self,
        split: str,
        video_stem: str,
        img_path: Path,
        pred: Any,
        all_detections_rows: List[Dict[str, Any]],
    ) -> Optional[FrameEvalResult]:
        img_bgr = pred.orig_img
        if img_bgr is None:
            img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None

        img_h, img_w = img_bgr.shape[:2]
        meta = parse_video_stem_metadata(video_stem)

        # Parse frame index and label from e.g. frame_1_start.jpg
        stem_parts = img_path.stem.split("_", 2)
        frame_idx = int(stem_parts[1]) if len(stem_parts) > 1 and stem_parts[1].isdigit() else 1
        frame_label = stem_parts[2] if len(stem_parts) > 2 else "start"

        # Image quality / venue lighting metrics
        hsv_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        v_chan = hsv_full[:, :, 2]
        s_chan = hsv_full[:, :, 1]
        brightness_mean = float(np.mean(v_chan))
        contrast_std = float(np.std(v_chan))
        glare_ratio = float(np.count_nonzero((v_chan > 245) & (s_chan < 30))) / float(img_h * img_w)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sharpness_lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        boxes_tensor = pred.boxes
        detections: List[DetectionBox] = []

        if boxes_tensor is not None and len(boxes_tensor) > 0:
            xyxy = boxes_tensor.xyxy.cpu().numpy()
            confs = boxes_tensor.conf.cpu().numpy()
            clss = boxes_tensor.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                norm_cx = cx / img_w
                norm_cy = cy / img_h
                norm_area = (bw * bh) / (img_w * img_h)

                # Top 11% of broadcast frame is typically scoreboard overlay or upper bleachers
                in_field = norm_cy >= 0.11

                crop = img_bgr[max(0, int(y1)) : min(img_h, int(y2)), max(0, int(x1)) : min(img_w, int(x2))]
                r_ratio, b_ratio, y_ratio = compute_hsv_color_ratios(crop)

                # Check color purity for red/blue bumpers
                color_ok = True
                if cls_id == 0 and b_ratio > 0.35 and b_ratio > r_ratio * 1.5:
                    color_ok = False
                elif cls_id == 1 and r_ratio > 0.35 and r_ratio > b_ratio * 1.5:
                    color_ok = False

                det = DetectionBox(
                    cls_id=int(cls_id),
                    cls_name=CLASS_NAMES.get(int(cls_id), f"cls_{cls_id}"),
                    conf=round(float(conf), 4),
                    x1=round(float(x1), 1),
                    y1=round(float(y1), 1),
                    x2=round(float(x2), 1),
                    y2=round(float(y2), 1),
                    cx=round(float(cx), 1),
                    cy=round(float(cy), 1),
                    w=round(float(bw), 1),
                    h=round(float(bh), 1),
                    norm_cx=round(float(norm_cx), 4),
                    norm_cy=round(float(norm_cy), 4),
                    norm_area=round(float(norm_area), 6),
                    red_hue_ratio=r_ratio,
                    blue_hue_ratio=b_ratio,
                    yellow_hue_ratio=y_ratio,
                    color_purity_ok=color_ok,
                    in_field_bounds=in_field,
                )
                detections.append(det)

        # Cluster same-alliance bumpers into distinct robots
        red_clusters = cluster_alliance_bumpers(detections, 0, img_w, img_h)
        blue_clusters = cluster_alliance_bumpers(detections, 1, img_w, img_h)

        for d in detections:
            all_detections_rows.append(
                {
                    "split": split,
                    "year": meta["year"],
                    "event_key": meta["event_key"],
                    "match_key": meta["match_key"],
                    "frame_label": frame_label,
                    "frame_index": frame_idx,
                    **asdict(d),
                }
            )

        raw_red = sum(1 for d in detections if d.cls_id == 0)
        raw_blue = sum(1 for d in detections if d.cls_id == 1)
        raw_fuel = sum(1 for d in detections if d.cls_id == 2 and d.in_field_bounds)

        n_red = len(red_clusters)
        n_blue = len(blue_clusters)
        red_valid = n_red <= 3
        blue_valid = n_blue <= 3
        red_excess = max(0, n_red - 3)
        blue_excess = max(0, n_blue - 3)
        out_of_field_fp = sum(1 for d in detections if not d.in_field_bounds)
        color_swaps = sum(1 for d in detections if d.cls_id in (0, 1) and not d.color_purity_ok)
        cross_year_fuel_fp = raw_fuel if meta["year"] < 2026 else 0

        bumper_confs = [d.conf for d in detections if d.cls_id in (0, 1)]
        fuel_confs = [d.conf for d in detections if d.cls_id == 2]
        low_conf_cnt = sum(1 for d in detections if 0.25 <= d.conf < 0.45)

        # Spatial & FRC gameplay diagnostics
        red_xs = [c.norm_cx for c in red_clusters]
        blue_xs = [c.norm_cx for c in blue_clusters]
        red_mean_x = float(np.mean(red_xs)) if red_xs else 0.25
        blue_mean_x = float(np.mean(blue_xs)) if blue_xs else 0.75

        incursion_red = sum(1 for c in red_clusters if c.norm_cx > 0.52)
        incursion_blue = sum(1 for c in blue_clusters if c.norm_cx < 0.48)

        # Opposing robot proximity (defense / pinning indicator)
        min_opp_dist = 1.0
        defense_pairs = 0
        for rc in red_clusters:
            for bc in blue_clusters:
                dist = math.hypot(rc.norm_cx - bc.norm_cx, rc.norm_cy - bc.norm_cy)
                if dist < min_opp_dist:
                    min_opp_dist = dist
                if dist < 0.11:  # Within bumper-to-bumper contact range
                    defense_pairs += 1

        fuel_boxes = [d for d in detections if d.cls_id == 2 and d.in_field_bounds]
        fuel_left = sum(1 for d in fuel_boxes if d.norm_cx < 0.35)
        fuel_right = sum(1 for d in fuel_boxes if d.norm_cx > 0.65)
        fuel_center = sum(1 for d in fuel_boxes if 0.35 <= d.norm_cx <= 0.65)

        # Uncertainty / active learning priority score
        missing_robots_penalty = max(0, 6 - (n_red + n_blue)) * 0.15 if frame_label in ("middle_1", "middle_2", "middle_3") else 0.0
        uncertainty = (
            low_conf_cnt * 0.25
            + (red_excess + blue_excess) * 1.2
            + color_swaps * 1.0
            + missing_robots_penalty
        )

        # Save annotated frame to Extreme SSD
        annotated_rel = Path(split) / video_stem / img_path.name
        annotated_out_path = self.annotated_frames_dir / annotated_rel
        if self.save_annotated_frames:
            annotated_out_path.parent.mkdir(parents=True, exist_ok=True)
            annotated_img = draw_annotated_frame(
                img_bgr=img_bgr,
                detections=detections,
                red_clusters=red_clusters,
                blue_clusters=blue_clusters,
                meta=meta,
                frame_label=frame_label,
            )
            cv2.imwrite(str(annotated_out_path), annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

        return FrameEvalResult(
            split=split,
            year=meta["year"],
            event_key=meta["event_key"],
            match_key=meta["match_key"],
            comp_level=meta["comp_level"],
            match_number=meta["match_number"],
            youtube_id=meta["youtube_id"],
            video_stem=video_stem,
            frame_label=frame_label,
            frame_index=frame_idx,
            image_path=str(img_path),
            annotated_path=str(annotated_out_path),
            img_w=img_w,
            img_h=img_h,
            raw_red_bumpers=raw_red,
            raw_blue_bumpers=raw_blue,
            raw_fuel_count=raw_fuel,
            red_robot_clusters=n_red,
            blue_robot_clusters=n_blue,
            total_robot_clusters=n_red + n_blue,
            red_count_valid=red_valid,
            blue_count_valid=blue_valid,
            red_excess_fp=red_excess,
            blue_excess_fp=blue_excess,
            out_of_field_fp=out_of_field_fp,
            color_swap_suspects=color_swaps,
            cross_year_fuel_fp=cross_year_fuel_fp,
            mean_bumper_conf=round(float(np.mean(bumper_confs)), 4) if bumper_confs else 0.0,
            mean_fuel_conf=round(float(np.mean(fuel_confs)), 4) if fuel_confs else 0.0,
            low_conf_detections=low_conf_cnt,
            brightness_mean=round(brightness_mean, 2),
            contrast_std=round(contrast_std, 2),
            glare_ratio=round(glare_ratio, 5),
            sharpness_laplacian=round(sharpness_lap, 2),
            red_mean_norm_x=round(red_mean_x, 4),
            blue_mean_norm_x=round(blue_mean_x, 4),
            territory_incursion_red=incursion_red,
            territory_incursion_blue=incursion_blue,
            min_opposing_robot_dist_norm=round(min_opp_dist, 4),
            defense_engagements=defense_pairs,
            fuel_left_half=fuel_left,
            fuel_right_half=fuel_right,
            fuel_center_zone=fuel_center,
            uncertainty_score=round(uncertainty, 3),
        )

    def _save_results(
        self,
        frame_results: List[FrameEvalResult],
        all_detections_rows: List[Dict[str, Any]],
    ) -> None:
        frame_dicts = [asdict(r) for r in frame_results]

        for target_dir in [self.ssd_data_dir, self.repo_data_dir]:
            # 1. Frame-level CSV & JSON
            frames_csv = target_dir / "frame_evaluation_results.csv"
            if frame_dicts:
                with open(frames_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(frame_dicts[0].keys()))
                    writer.writeheader()
                    writer.writerows(frame_dicts)

            frames_json = target_dir / "frame_evaluation_results.json"
            with open(frames_json, "w", encoding="utf-8") as f:
                json.dump(frame_dicts, f, indent=2)

            # 2. Detection-level CSV
            dets_csv = target_dir / "all_bounding_box_detections.csv"
            if all_detections_rows:
                with open(dets_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(all_detections_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(all_detections_rows)

            # 3. Active Learning Top-50 Hardest Frames CSV
            hardest = sorted(frame_dicts, key=lambda x: x["uncertainty_score"], reverse=True)[:50]
            hard_csv = target_dir / "active_learning_hardest_50_frames.csv"
            if hardest:
                with open(hard_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(hardest[0].keys()))
                    writer.writeheader()
                    writer.writerows(hardest)

        print(f"[Evaluator] Saved {len(frame_dicts)} frame evaluations and {len(all_detections_rows)} box records.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Zero-Label FRC YOLO11s Accuracy & Invariant Evaluation.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/frc_yolo11s_best.pt",
        help="Path to trained YOLO11 weights (.pt)",
    )
    parser.add_argument(
        "--ssd-root",
        type=str,
        default="./data",
        help="Root directory on Extreme SSD containing frc_extracted_frames",
    )
    parser.add_argument(
        "--repo-out",
        type=str,
        default="analytics_outputs",
        help="Local repository directory for GitHub-ready analytics outputs",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--no-annotated-frames", action="store_true", help="Skip saving annotated JPGs")
    args = parser.parse_args()

    ssd_root = Path(args.ssd_root)
    evaluator = ZeroLabelEvaluator(
        model_path=Path(args.model),
        frames_root=ssd_root / "frc_extracted_frames",
        ssd_output_dir=ssd_root / "frc_vision_analytics",
        repo_output_dir=Path(args.repo_out),
        save_annotated_frames=not args.no_annotated_frames,
    )
    evaluator.evaluate_all_frames(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
