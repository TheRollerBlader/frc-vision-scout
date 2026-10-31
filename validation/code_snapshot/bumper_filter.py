"""
Temporal Majority-Voting Filter for Robot Bumper OCR Identification.

Stabilizes noisy or wrinkled bumper number readings across multi-object tracking
trajectories (ByteTrack / BoT-SORT) on FIRST Robotics Competition match video.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass(frozen=True)
class BumperDetection:
    """Represents a single-frame bumper OCR detection for a tracked robot."""
    frame_idx: int
    track_id: int
    team_number: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized [0, 1]


@dataclass(frozen=True)
class TrackletHypothesis:
    """Consolidated bumper identity hypothesis for a tracked trajectory."""
    team_number: int
    confidence_score: float
    total_observations: int
    consistency_ratio: float


class TemporalBumperFilter:
    """
    Sliding-window temporal aggregator that filters OCR misreads using
    confidence-weighted voting and persistence gating.
    """

    def __init__(
        self,
        window_size: int = 30,
        min_confidence: float = 0.50,
        majority_ratio_threshold: float = 0.60,
    ) -> None:
        """
        Args:
            window_size: Number of recent observations retained per track ID.
            min_confidence: Minimum frame-level OCR confidence to record observation.
            majority_ratio_threshold: Fraction of weighted votes needed to lock identity.
        """
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.majority_ratio_threshold = majority_ratio_threshold
        # track_id -> list of BumperDetection
        self._history: Dict[int, List[BumperDetection]] = defaultdict(list)
        self._last_seen_frame: Dict[int, int] = {}

    def add_observation(self, detection: BumperDetection) -> bool:
        """
        Records a detection for a tracklet if it passes the confidence threshold.

        Returns:
            True if detection was accepted, False if rejected by confidence gate.
        """
        if detection.confidence < self.min_confidence:
            return False

        # Validate team number is within valid FRC range (1 to 99999)
        if detection.team_number <= 0 or detection.team_number > 99999:
            return False

        track_id = detection.track_id
        self._history[track_id].append(detection)
        self._last_seen_frame[track_id] = detection.frame_idx

        # Enforce sliding window capacity
        if len(self._history[track_id]) > self.window_size:
            self._history[track_id].pop(0)

        return True

    def get_resolved_team(self, track_id: int) -> Optional[TrackletHypothesis]:
        """
        Calculates the winning bumper identity for track_id based on weighted confidence votes.

        Returns:
            TrackletHypothesis if a winning team exceeds the majority threshold,
            otherwise None.
        """
        observations = self._history.get(track_id, [])
        if not observations:
            return None

        # Tally confidence sums per team candidate
        vote_sums: Dict[int, float] = defaultdict(float)
        counts: Dict[int, int] = defaultdict(int)
        total_weight = 0.0

        for obs in observations:
            weight = max(0.01, obs.confidence)
            vote_sums[obs.team_number] += weight
            counts[obs.team_number] += 1
            total_weight += weight

        if total_weight <= 0.0:
            return None

        # Find team candidate with maximum weighted score
        best_team, best_weight = max(vote_sums.items(), key=lambda item: item[1])
        consistency = best_weight / total_weight

        if consistency >= self.majority_ratio_threshold:
            mean_conf = best_weight / counts[best_team]
            return TrackletHypothesis(
                team_number=best_team,
                confidence_score=round(mean_conf, 4),
                total_observations=counts[best_team],
                consistency_ratio=round(consistency, 4),
            )

        return None

    def prune_stale_tracks(self, current_frame: int, max_age: int = 150) -> int:
        """
        Removes trajectories that haven't been observed for max_age frames.

        Returns:
            Number of pruned tracks.
        """
        stale_ids = [
            tid
            for tid, last_frame in self._last_seen_frame.items()
            if current_frame - last_frame > max_age
        ]
        for tid in stale_ids:
            self._history.pop(tid, None)
            self._last_seen_frame.pop(tid, None)

        return len(stale_ids)

    def reset(self) -> None:
        """Clears all stored tracking histories."""
        self._history.clear()
        self._last_seen_frame.clear()
