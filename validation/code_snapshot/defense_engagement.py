"""Multi-Robot Defensive Engagement & Pinning Analytics for FRC Vision Scout.

Analyzes paired 2D field-space KinematicState trajectories across opposing
alliances to detect sustained defensive contact intervals, closing velocities,
and FRC perimeter-pinning rule (G424-style) duration warnings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Sequence
import math

from validation.code_snapshot.homography import FieldHomography
from validation.code_snapshot.trajectory_analytics import KinematicState

AllianceColor = Literal["red", "blue"]


@dataclass(frozen=True)
class DefenseEngagementEvent:
    """Represents a sustained defensive interaction between two opposing robots."""

    defender_id: int
    attacker_id: int
    start_time: float
    end_time: float
    duration_s: float
    min_separation_m: float
    mean_separation_m: float
    max_closing_speed_mps: float
    pinning_risk: bool


@dataclass
class DefenseSummary:
    """Aggregated defensive telemetry for a single defending robot."""

    defender_id: int
    total_defense_time_s: float
    engagement_count: int
    pinning_warnings: int
    events: List[DefenseEngagementEvent] = field(default_factory=list)


def _is_near_field_perimeter(x: float, y: float, wall_margin_m: float) -> bool:
    return (
        x <= wall_margin_m
        or x >= (FieldHomography.FIELD_LENGTH_METERS - wall_margin_m)
        or y <= wall_margin_m
        or y >= (FieldHomography.FIELD_WIDTH_METERS - wall_margin_m)
    )


def _align_trajectories_by_timestamp(
    defender_states: Sequence[KinematicState],
    attacker_states: Sequence[KinematicState],
    max_time_skew_s: float = 0.08,
) -> List[tuple[KinematicState, KinematicState]]:
    """Pairs defender and attacker states that occur within max_time_skew_s."""
    if not defender_states or not attacker_states:
        return []

    def_sorted = sorted(defender_states, key=lambda s: s.timestamp)
    att_sorted = sorted(attacker_states, key=lambda s: s.timestamp)

    paired: List[tuple[KinematicState, KinematicState]] = []
    j = 0
    for d_state in def_sorted:
        while j + 1 < len(att_sorted) and (
            abs(att_sorted[j + 1].timestamp - d_state.timestamp)
            <= abs(att_sorted[j].timestamp - d_state.timestamp)
        ):
            j += 1
        a_state = att_sorted[j]
        if abs(a_state.timestamp - d_state.timestamp) <= max_time_skew_s:
            paired.append((d_state, a_state))
    return paired


def detect_pair_engagements(
    defender_id: int,
    defender_states: Sequence[KinematicState],
    attacker_id: int,
    attacker_states: Sequence[KinematicState],
    engagement_radius_m: float = 1.25,
    min_duration_s: float = 1.0,
    pinning_duration_s: float = 3.0,
    wall_margin_m: float = 0.85,
    stalled_speed_mps: float = 0.25,
) -> List[DefenseEngagementEvent]:
    """Detects sustained defensive engagements and pinning risks for a robot pair."""
    aligned = _align_trajectories_by_timestamp(defender_states, attacker_states)
    if len(aligned) < 2:
        return []

    events: List[DefenseEngagementEvent] = []
    active_window: List[tuple[KinematicState, KinematicState, float, float]] = []

    def flush_window() -> None:
        if len(active_window) < 2:
            active_window.clear()
            return
        start_t = active_window[0][0].timestamp
        end_t = active_window[-1][0].timestamp
        duration = end_t - start_t
        if duration < min_duration_s:
            active_window.clear()
            return

        separations = [item[2] for item in active_window]
        closing_speeds = [item[3] for item in active_window]

        # Measure longest contiguous interval where attacker is stalled near a wall
        max_pin_streak = 0.0
        pin_streak_start: float | None = None
        for _, a_st, _, _ in active_window:
            stalled_at_wall = (
                a_st.speed <= stalled_speed_mps
                and _is_near_field_perimeter(a_st.x, a_st.y, wall_margin_m)
            )
            if stalled_at_wall:
                if pin_streak_start is None:
                    pin_streak_start = a_st.timestamp
                max_pin_streak = max(max_pin_streak, a_st.timestamp - pin_streak_start)
            else:
                pin_streak_start = None

        events.append(
            DefenseEngagementEvent(
                defender_id=defender_id,
                attacker_id=attacker_id,
                start_time=round(start_t, 3),
                end_time=round(end_t, 3),
                duration_s=round(duration, 3),
                min_separation_m=round(min(separations), 3),
                mean_separation_m=round(sum(separations) / len(separations), 3),
                max_closing_speed_mps=round(max(closing_speeds), 3),
                pinning_risk=max_pin_streak >= pinning_duration_s,
            )
        )
        active_window.clear()

    for d_st, a_st in aligned:
        dx = a_st.x - d_st.x
        dy = a_st.y - d_st.y
        dist = math.hypot(dx, dy)

        if dist <= engagement_radius_m:
            # Closing speed = projection of relative velocity (defender - attacker) onto unit vector toward attacker
            if dist > 1e-6:
                ux, uy = dx / dist, dy / dist
                rel_vx = d_st.vx - a_st.vx
                rel_vy = d_st.vy - a_st.vy
                closing = max(0.0, rel_vx * ux + rel_vy * uy)
            else:
                closing = 0.0
            active_window.append((d_st, a_st, dist, closing))
        else:
            flush_window()

    flush_window()
    return events


def summarize_match_defense(
    trajectories: Dict[int, Sequence[KinematicState]],
    alliances: Dict[int, AllianceColor],
    engagement_radius_m: float = 1.25,
    min_duration_s: float = 1.0,
) -> Dict[int, DefenseSummary]:
    """Computes per-robot defensive engagement summaries across opposing alliances."""
    summaries: Dict[int, DefenseSummary] = {}

    for defender_id, def_alliance in alliances.items():
        def_states = trajectories.get(defender_id, [])
        if not def_states:
            continue

        all_events: List[DefenseEngagementEvent] = []
        for attacker_id, att_alliance in alliances.items():
            if att_alliance == def_alliance:
                continue
            att_states = trajectories.get(attacker_id, [])
            if not att_states:
                continue
            pair_events = detect_pair_engagements(
                defender_id=defender_id,
                defender_states=def_states,
                attacker_id=attacker_id,
                attacker_states=att_states,
                engagement_radius_m=engagement_radius_m,
                min_duration_s=min_duration_s,
            )
            all_events.extend(pair_events)

        all_events.sort(key=lambda e: e.start_time)
        summaries[defender_id] = DefenseSummary(
            defender_id=defender_id,
            total_defense_time_s=round(sum(e.duration_s for e in all_events), 3),
            engagement_count=len(all_events),
            pinning_warnings=sum(1 for e in all_events if e.pinning_risk),
            events=all_events,
        )

    return summaries
