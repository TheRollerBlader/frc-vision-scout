"""Unit tests for multi-robot defensive engagement and pinning detection."""

import pytest
from validation.code_snapshot.defense_engagement import (
    detect_pair_engagements,
    summarize_match_defense,
)
from validation.code_snapshot.trajectory_analytics import KinematicState


def _make_state(
    t: float,
    x: float,
    y: float,
    vx: float = 0.0,
    vy: float = 0.0,
) -> KinematicState:
    speed = (vx**2 + vy**2) ** 0.5
    return KinematicState(
        timestamp=t,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        speed=speed,
        acceleration=0.0,
        heading_deg=0.0,
    )


class TestDefenseEngagement:
    def test_detects_sustained_engagement_without_pinning_in_open_field(self):
        # Defender 254 intercepts Attacker 1678 near center field for 1.5s
        defender = [
            _make_state(0.0, 5.0, 4.0, vx=2.5),
            _make_state(0.5, 7.5, 4.0, vx=2.0),
            _make_state(1.0, 8.2, 4.0, vx=0.8),
            _make_state(1.5, 8.3, 4.0, vx=0.5),
            _make_state(2.0, 8.4, 4.0, vx=0.5),
            _make_state(2.5, 11.0, 4.0, vx=2.0),
        ]
        attacker = [
            _make_state(0.0, 9.0, 4.0, vx=0.0),
            _make_state(0.5, 8.5, 4.0, vx=-0.5),
            _make_state(1.0, 8.6, 4.0, vx=0.1),
            _make_state(1.5, 8.7, 4.0, vx=0.1),
            _make_state(2.0, 8.8, 4.0, vx=0.1),
            _make_state(2.5, 9.2, 4.0, vx=0.5),
        ]

        events = detect_pair_engagements(254, defender, 1678, attacker)
        assert len(events) == 1
        ev = events[0]
        assert ev.defender_id == 254
        assert ev.attacker_id == 1678
        assert ev.duration_s == pytest.approx(1.5, abs=0.01)
        assert ev.max_closing_speed_mps == pytest.approx(2.5, abs=0.01)
        assert ev.pinning_risk is False

    def test_flags_pinning_warning_when_stalled_near_perimeter_over_threshold(self):
        # Attacker pinned at y=0.4m (within 0.85m wall margin) for 3.5s
        times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        defender = [_make_state(t, 6.0, 1.2, vx=0.0, vy=-0.1) for t in times]
        attacker = [_make_state(t, 6.0, 0.4, vx=0.05, vy=0.0) for t in times]

        events = detect_pair_engagements(118, defender, 2056, attacker)
        assert len(events) == 1
        assert events[0].duration_s == pytest.approx(3.5, abs=0.01)
        assert events[0].pinning_risk is True

    def test_ignores_brief_drive_by_below_min_duration(self):
        defender = [
            _make_state(0.0, 4.0, 4.0),
            _make_state(0.3, 4.5, 4.0),
            _make_state(0.6, 7.0, 4.0),
        ]
        attacker = [
            _make_state(0.0, 4.6, 4.0),
            _make_state(0.3, 4.7, 4.0),
            _make_state(0.6, 4.8, 4.0),
        ]
        events = detect_pair_engagements(1, defender, 2, attacker, min_duration_s=1.0)
        assert events == []

    def test_summarize_match_defense_excludes_same_alliance(self):
        times = [0.0, 0.5, 1.0, 1.5]
        trajectories = {
            101: [_make_state(t, 5.0, 4.0) for t in times],  # red
            102: [_make_state(t, 5.4, 4.0) for t in times],  # red teammate
            201: [_make_state(t, 5.6, 4.0) for t in times],  # blue opponent
        }
        alliances = {101: "red", 102: "red", 201: "blue"}

        summaries = summarize_match_defense(trajectories, alliances)
        assert summaries[101].engagement_count == 1
        assert summaries[101].events[0].attacker_id == 201
        assert summaries[101].total_defense_time_s == pytest.approx(1.5, abs=0.01)
        assert summaries[201].engagement_count == 2
