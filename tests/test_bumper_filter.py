"""
Unit tests for the temporal bumper majority voting filter in frc-vision-scout.
"""

import pytest
from validation.code_snapshot.bumper_filter import (
    BumperDetection,
    TemporalBumperFilter,
    TrackletHypothesis,
)


class TestTemporalBumperFilter:
    def test_rejects_sub_threshold_observations(self):
        voter = TemporalBumperFilter(min_confidence=0.60)
        low_conf = BumperDetection(
            frame_idx=1,
            track_id=1,
            team_number=5411,
            confidence=0.45,
            bbox=(0.1, 0.1, 0.2, 0.2),
        )
        assert voter.add_observation(low_conf) is False
        assert voter.get_resolved_team(1) is None

    def test_rejects_invalid_team_numbers(self):
        voter = TemporalBumperFilter()
        invalid_detection = BumperDetection(
            frame_idx=1,
            track_id=1,
            team_number=-5,
            confidence=0.95,
            bbox=(0.1, 0.1, 0.2, 0.2),
        )
        assert voter.add_observation(invalid_detection) is False

    def test_locks_identity_under_consistent_detections(self):
        voter = TemporalBumperFilter(majority_ratio_threshold=0.60)
        for f in range(10):
            det = BumperDetection(
                frame_idx=f,
                track_id=3,
                team_number=254,
                confidence=0.85,
                bbox=(0.2, 0.2, 0.3, 0.3),
            )
            voter.add_observation(det)

        hypothesis = voter.get_resolved_team(3)
        assert hypothesis is not None
        assert isinstance(hypothesis, TrackletHypothesis)
        assert hypothesis.team_number == 254
        assert hypothesis.consistency_ratio == 1.0
        assert hypothesis.total_observations == 10

    def test_overcomes_transient_ocr_wrinkles_and_misreads(self):
        voter = TemporalBumperFilter(majority_ratio_threshold=0.60)
        # 8 observations of team 1678
        for f in range(8):
            voter.add_observation(
                BumperDetection(
                    frame_idx=f,
                    track_id=5,
                    team_number=1678,
                    confidence=0.90,
                    bbox=(0.3, 0.3, 0.4, 0.4),
                )
            )
        # 2 corrupted glitch frames reading 1670 due to bumper cloth wrinkle
        for f in range(8, 10):
            voter.add_observation(
                BumperDetection(
                    frame_idx=f,
                    track_id=5,
                    team_number=1670,
                    confidence=0.55,
                    bbox=(0.3, 0.3, 0.4, 0.4),
                )
            )

        hypothesis = voter.get_resolved_team(5)
        assert hypothesis is not None
        assert hypothesis.team_number == 1678
        assert hypothesis.consistency_ratio > 0.80

    def test_pruning_cleans_stale_tracklets(self):
        voter = TemporalBumperFilter()
        voter.add_observation(
            BumperDetection(
                frame_idx=10,
                track_id=12,
                team_number=971,
                confidence=0.90,
                bbox=(0.4, 0.4, 0.5, 0.5),
            )
        )
        # Track 12 was last seen at frame 10; at frame 200 it exceeds max_age=150
        pruned_count = voter.prune_stale_tracks(current_frame=200, max_age=150)
        assert pruned_count == 1
        assert voter.get_resolved_team(12) is None
