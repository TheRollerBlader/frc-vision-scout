"""
Unit tests for closed-set robot identification, bumper OCR scoring,
and upper-chassis appearance embedding in frc-vision-scout.
"""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from validation.code_snapshot.evaluator import DetectionBox, RobotCluster
from validation.code_snapshot.robot_identifier import (
    RobotTelemetrySummary,
    closed_set_string_similarity,
    extract_upper_chassis_embedding,
)


class TestClosedSetOCRAndMatching:
    def test_closed_set_similarity_exact_matches(self):
        assert closed_set_string_similarity("2495", "2495") == 1.0
        assert closed_set_string_similarity("5411", "5411") == 1.0
        assert closed_set_string_similarity("254", "254") == 1.0

    def test_closed_set_similarity_partial_substring(self):
        # 3 out of 4 digits visible under angle/wrinkle
        score = closed_set_string_similarity("249", "2495")
        assert score >= 0.85
        assert score < 1.0

        # 2 out of 4 digits
        score_short = closed_set_string_similarity("49", "2495")
        assert score_short >= 0.75
        assert score_short < score

    def test_closed_set_similarity_disjoint_numbers(self):
        assert closed_set_string_similarity("1678", "254") == 0.0
        assert closed_set_string_similarity("9999", "1111") == 0.0

    def test_closed_set_similarity_empty_and_noise(self):
        assert closed_set_string_similarity("", "254") == 0.0
        assert closed_set_string_similarity("abc", "254") == 0.0
        assert closed_set_string_similarity("  ", "") == 0.0

    def test_upper_chassis_embedding_shape_and_normalization(self):
        # Create a synthetic 100x100 BGR test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:80, 20:80] = [180, 100, 50]  # Non-zero colored block

        cluster = RobotCluster(
            cluster_id=1,
            alliance="red",
            x1=20.0,
            y1=50.0,
            x2=80.0,
            y2=80.0,
            cx=50.0,
            cy=65.0,
            norm_cx=0.5,
            norm_cy=0.65,
            max_conf=0.9,
            mean_conf=0.9,
            num_bumper_segments=1,
            color_purity_ok=True,
        )

        embedding = extract_upper_chassis_embedding(img, cluster)
        assert embedding.shape == (32,)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-4

    def test_upper_chassis_embedding_empty_crop(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cluster = RobotCluster(
            cluster_id=2,
            alliance="blue",
            x1=200.0,
            y1=200.0,
            x2=250.0,
            y2=250.0,
            cx=225.0,
            cy=225.0,
            norm_cx=0.9,
            norm_cy=0.9,
            max_conf=0.8,
            mean_conf=0.8,
            num_bumper_segments=1,
            color_purity_ok=True,
        )
        embedding = extract_upper_chassis_embedding(img, cluster)
        assert embedding.shape == (32,)
        assert np.all(embedding == 0.0)

    def test_hungarian_closed_set_process_of_elimination(self):
        """
        Verify that 3 robot slots are mapped to 3 TBA alliance teams via Hungarian
        matching, correctly solving the third robot by process of elimination.
        """
        # 3 candidate teams: ["frc5411", "frc254", "frc1678"]
        # Slot 0 has strong reads for 5411
        # Slot 1 has strong reads for 254
        # Slot 2 has zero/noisy reads (facing away), but process of elimination assigns frc1678
        cost_matrix = np.array([
            [0.05, 0.95, 0.95],  # Slot 0: low cost for 5411
            [0.95, 0.05, 0.95],  # Slot 1: low cost for 254
            [0.80, 0.80, 0.80],  # Slot 2: ambiguous, but 5411 & 254 are taken
        ])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assert list(row_ind) == [0, 1, 2]
        assert col_ind[0] == 0  # Slot 0 -> frc5411
        assert col_ind[1] == 1  # Slot 1 -> frc254
        assert col_ind[2] == 2  # Slot 2 -> frc1678 (process of elimination)

    def test_robot_telemetry_summary_dataclass(self):
        summary = RobotTelemetrySummary(
            match_key="2026txcmp_qm12",
            event_key="2026txcmp",
            year=2026,
            video_stem="2026txcmp_qm12_raw",
            alliance="red",
            station_slot=1,
            team_key="frc5411",
            team_number="5411",
            frames_visible=1200,
            total_sampled_frames=1350,
            visibility_rate=1200 / 1350,
            mean_yolo_conf=0.92,
            id_confidence=0.98,
            ocr_reads_count=14,
            best_ocr_raw_text="5411",
            id_method="OCR+Station",
            total_distance_norm=18.4,
            estimated_distance_feet=993.6,
            mean_speed_fps=7.36,
            max_speed_fps=16.2,
            stationary_seconds=12.5,
            is_dead_or_impaired=False,
            mean_norm_x=0.45,
            mean_norm_y=0.55,
            std_norm_x=0.22,
            std_norm_y=0.18,
            time_in_own_zone_pct=0.42,
            time_in_neutral_zone_pct=0.38,
            time_in_enemy_zone_pct=0.20,
            estimated_cycles=9,
            mean_cycle_time_sec=14.8,
            fuel_proximity_frames=340,
            defense_engagement_sec=18.5,
            primary_opponent_engaged="frc1678",
            classified_role="Primary Cycler",
            alliance_Final_score=114,
            alliance_won=True,
            tba_auto_tower_status="Completed",
            tba_endgame_tower_status="DeepClimb",
            tba_energized_achieved=True,
            tba_supercharged_achieved=True,
            tba_traversal_achieved=False,
            annotated_video_path="outputs/2026txcmp_qm12_annotated.mp4",
        )
        assert summary.team_number == "5411"
        assert summary.classified_role == "Primary Cycler"
        assert summary.visibility_rate > 0.85
        assert summary.estimated_cycles == 9
