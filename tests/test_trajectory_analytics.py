"""
Unit tests for the kinematic trajectory & cycle analytics engine in frc-vision-scout.
"""

import pytest
import math
from validation.code_snapshot.trajectory_analytics import (
    KinematicKalmanFilter,
    TrajectoryAnalyticsEngine,
    KinematicState,
    CycleEvent,
    TrajectorySummary,
)


class TestKinematicKalmanFilter:
    def test_kalman_filter_initialization_and_prediction(self):
        kf = KinematicKalmanFilter(initial_x=5.0, initial_y=3.0)
        assert kf.x[0] == 5.0
        assert kf.x[1] == 3.0
        assert kf.x[2] == 0.0  # vx
        assert kf.x[3] == 0.0  # vy

        # Predict ahead by 0.1s
        pred = kf.predict(0.1)
        assert pytest.approx(pred[0], abs=1e-4) == 5.0
        assert pytest.approx(pred[1], abs=1e-4) == 3.0

    def test_kalman_filter_velocity_estimation_and_smoothing(self):
        kf = KinematicKalmanFilter(initial_x=0.0, initial_y=0.0)

        # Simulate robot moving linearly along x at 2.0 m/s with small sensor jitter
        dt = 0.1
        t = 0.0
        for step in range(1, 15):
            t += dt
            true_x = 2.0 * t
            noisy_x = true_x + (0.05 if step % 2 == 0 else -0.05)
            kf.predict(dt)
            updated = kf.update(noisy_x, 0.0)

        # Velocity in x should converge close to 2.0 m/s
        est_vx = kf.x[2]
        assert pytest.approx(est_vx, abs=0.3) == 2.0
        assert pytest.approx(kf.x[3], abs=0.1) == 0.0  # vy ~ 0


class TestTrajectoryAnalyticsEngine:
    def test_zone_classification(self):
        engine = TrajectoryAnalyticsEngine(alliance_color="blue")

        # Loading station at ends
        assert engine.classify_zone(1.0, 4.0) == "LOADING_STATION"
        assert engine.classify_zone(15.5, 4.0) == "LOADING_STATION"

        # Reef in middle
        assert engine.classify_zone(6.0, 4.0) == "SCORING_REEF"

        # Opponent territory for blue alliance is x > 8.27 (excluding loading station)
        assert engine.classify_zone(13.0, 1.0) == "OPPONENT_TERRITORY"

    def test_outlier_rejection_for_unphysical_jumps(self):
        engine = TrajectoryAnalyticsEngine()

        # Step 1: Start at (2.0, 2.0)
        s1 = engine.add_point(track_id=1, timestamp=0.0, measured_x=2.0, measured_y=2.0)
        assert s1.is_outlier is False

        # Step 2: Normal motion 0.1s later (2.2, 2.0) -> speed 2.0 m/s
        s2 = engine.add_point(track_id=1, timestamp=0.1, measured_x=2.2, measured_y=2.0)
        assert s2.is_outlier is False

        # Step 3: Massive impossible camera glitch 0.1s later jumping to (12.0, 2.0) -> speed 98 m/s!
        s3 = engine.add_point(track_id=1, timestamp=0.2, measured_x=12.0, measured_y=2.0)
        assert s3.is_outlier is True
        # Estimated x should NOT jump to 12.0; it should rely on forward kinematic prediction
        assert s3.x < 5.0

    def test_scoring_cycle_detection(self):
        engine = TrajectoryAnalyticsEngine(alliance_color="blue")

        # 1. Robot starts at Loading Station (x=1.0)
        engine.add_point(track_id=1, timestamp=0.0, measured_x=1.0, measured_y=4.0)
        engine.add_point(track_id=1, timestamp=1.0, measured_x=1.5, measured_y=4.0)

        # 2. Transits to Reef (x=6.0) over 3 seconds
        engine.add_point(track_id=1, timestamp=2.0, measured_x=3.5, measured_y=4.0)
        engine.add_point(track_id=1, timestamp=3.0, measured_x=5.0, measured_y=4.0)
        engine.add_point(track_id=1, timestamp=4.0, measured_x=6.0, measured_y=4.0)

        summary = engine.get_summary(1)
        assert summary is not None
        assert len(summary.completed_cycles) == 1

        cycle = summary.completed_cycles[0]
        assert isinstance(cycle, CycleEvent)
        assert cycle.duration_sec == 3.0
        assert cycle.distance_traveled_m >= 4.0
        assert cycle.avg_speed_mps > 0.5

    def test_trajectory_summary_statistics(self):
        engine = TrajectoryAnalyticsEngine(alliance_color="blue")

        # Simulate 10 frames of motion
        for i in range(10):
            t = i * 0.5
            x = 1.0 + (i * 0.8)  # moves from 1.0 to 8.2m
            engine.add_point(track_id=42, timestamp=t, measured_x=x, measured_y=3.0)

        summary = engine.get_summary(42)
        assert summary is not None
        assert summary.track_id == 42
        assert summary.total_distance_m >= 6.0
        assert summary.peak_speed_mps > 1.0
        assert summary.mean_speed_mps > 0.5
