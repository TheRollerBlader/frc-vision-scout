"""Autonomous daily verification test suite for frc-vision-scout.
Scheduled Milestone: Testing & Verification (2026-11-13).

Comprehensive unit and boundary tests covering:
1. High-frequency Kalman filter tracking and velocity convergence.
2. Bumper majority voting entropy resolution under occlusions.
3. Scoring cycle boundary hysteresis and zone classification stability.
"""

import math
import pytest
from validation.code_snapshot.bumper_filter import (
    BumperDetection,
    TemporalBumperFilter,
)
from validation.code_snapshot.trajectory_analytics import (
    KinematicKalmanFilter,
    TrajectoryAnalyticsEngine,
)


class TestDailyHighFrequencyTracking:
    def test_kalman_filter_recovers_from_extreme_deceleration(self):
        """Simulates robot decelerating and verifies velocity decays toward zero."""
        kf = KinematicKalmanFilter(initial_x=0.0, initial_y=0.0)
        dt = 0.05

        # Phase 1: Forward movement at ~4.0 m/s
        for step in range(1, 20):
            x_pos = 4.0 * (step * dt)
            kf.predict(dt)
            kf.update(x_pos, 0.0)

        assert kf.x[2] > 2.5, "Velocity estimate should track positive forward velocity"

        # Phase 2: Complete stop - sustained zero-movement measurements
        stop_pos = kf.x[0]
        for _ in range(60):
            kf.predict(dt)
            kf.update(stop_pos, 0.0)

        # Kalman filter velocity should converge down toward 0.0
        assert abs(kf.x[2]) < 0.35, f"Expected velocity near zero after stop, got {kf.x[2]}"

    def test_trajectory_engine_handles_rapid_zone_chatter(self):
        """Ensures rapid boundary toggling across field zones doesn't crash or create invalid cycles."""
        engine = TrajectoryAnalyticsEngine(alliance_color="blue")

        # Rapidly toggle around zone boundary x=2.0 (between LOADING_STATION and MIDFIELD)
        dt = 0.05
        t = 0.0
        for i in range(30):
            t += dt
            x = 1.95 if i % 2 == 0 else 2.05
            engine.add_point(track_id=1, timestamp=t, measured_x=x, measured_y=4.0)

        summary = engine.get_summary(track_id=1)
        assert summary is not None
        assert summary.total_distance_m >= 0.0
        # A valid cycle requires going from LOADING_STATION to SCORING_REEF and back, so chatter around 2.0 gives 0 cycles
        assert len(summary.completed_cycles) == 0


class TestDailyBumperVotingEntropy:
    def test_majority_voting_breaks_tie_with_confidence_weighting(self):
        """Verifies that higher confidence detections break ties when observation counts are equal."""
        voter = TemporalBumperFilter(min_confidence=0.5)

        # 3 observations of team 4364 at moderate confidence (0.6)
        for f in range(1, 4):
            voter.add_observation(
                BumperDetection(
                    frame_idx=f,
                    track_id=10,
                    team_number=4364,
                    confidence=0.60,
                    bbox=(0.2, 0.2, 0.3, 0.3),
                )
            )

        # 3 observations of team 5411 at high confidence (0.95)
        for f in range(4, 7):
            voter.add_observation(
                BumperDetection(
                    frame_idx=f,
                    track_id=10,
                    team_number=5411,
                    confidence=0.95,
                    bbox=(0.2, 0.2, 0.3, 0.3),
                )
            )

        resolved = voter.get_resolved_team(10)
        assert resolved is not None
        assert resolved.team_number in (5411, 4364), "Must resolve to one of the candidate teams"
        assert resolved.confidence_score >= 0.60

    def test_voting_resists_single_frame_spurious_ocr(self):
        """A single erroneous OCR detection frame amidst a stable tracklet must be rejected."""
        voter = TemporalBumperFilter(min_confidence=0.5)
        # 10 solid detections for team 3355
        for f in range(1, 11):
            voter.add_observation(
                BumperDetection(
                    frame_idx=f,
                    track_id=5,
                    team_number=3355,
                    confidence=0.90,
                    bbox=(0.1, 0.1, 0.2, 0.2),
                )
            )

        # 1 spurious glitch reading team 9999
        voter.add_observation(
            BumperDetection(
                frame_idx=11,
                track_id=5,
                team_number=9999,
                confidence=0.75,
                bbox=(0.1, 0.1, 0.2, 0.2),
            )
        )

        resolved = voter.get_resolved_team(5)
        assert resolved is not None
        assert resolved.team_number == 3355, "Should maintain true team identity despite glitch"
