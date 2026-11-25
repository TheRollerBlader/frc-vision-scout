"""2D FRC Field Trajectory & Defense Engagement SVG Visualizer.

Renders field-space KinematicState trajectories and DefenseEngagementEvent
markers onto an accurate 2D FRC field canvas (16.54 m x 8.21 m) for scouting
dashboards, match reports, and calibration review.
"""

from __future__ import annotations

from html import escape
from typing import Dict, Iterable, Literal, Mapping, Sequence

from validation.code_snapshot.defense_engagement import DefenseEngagementEvent
from validation.code_snapshot.homography import FieldHomography
from validation.code_snapshot.trajectory_analytics import KinematicState

AllianceColor = Literal["red", "blue"]

_ALLIANCE_STROKE: Dict[str, str] = {
    "blue": "#38bdf8",
    "red": "#f87171",
}


def field_to_svg_xy(
    x_m: float,
    y_m: float,
    width: int = 880,
    height: int = 440,
    padding: int = 24,
) -> tuple[float, float]:
    """Maps metric FRC field coordinates into clamped SVG pixel coordinates."""
    field_len = FieldHomography.FIELD_LENGTH_METERS
    field_wid = FieldHomography.FIELD_WIDTH_METERS

    clamped_x = min(field_len, max(0.0, float(x_m)))
    clamped_y = min(field_wid, max(0.0, float(y_m)))

    inner_w = max(1, width - 2 * padding)
    inner_h = max(1, height - 2 * padding)

    px = padding + (clamped_x / field_len) * inner_w
    # Flip Y so y=0 sits at the bottom edge of the field diagram
    py = padding + (1.0 - clamped_y / field_wid) * inner_h
    return (round(px, 2), round(py, 2))


def render_field_trajectories_svg(
    trajectories: Mapping[int, Sequence[KinematicState]],
    alliances: Mapping[int, AllianceColor] | None = None,
    engagements: Iterable[DefenseEngagementEvent] = (),
    width: int = 880,
    height: int = 440,
    title: str = "FRC Field Trajectory & Defense Map",
) -> str:
    """Generates a standalone SVG field diagram with trajectories and engagement markers."""
    padding = 24
    inner_w = max(1, width - 2 * padding)
    inner_h = max(1, height - 2 * padding)
    mid_x = padding + inner_w / 2.0
    alliance_zone_w = inner_w * 0.18

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{escape(title)}">',
        f'  <rect x="0" y="0" width="{width}" height="{height}" rx="12" fill="#0f172a" />',
        f'  <rect x="{padding}" y="{padding}" width="{inner_w}" height="{inner_h}" '
        f'fill="#1e293b" stroke="#475569" stroke-width="2" />',
        f'  <rect x="{padding}" y="{padding}" width="{alliance_zone_w:.2f}" height="{inner_h}" '
        f'fill="#1d4ed8" fill-opacity="0.14" />',
        f'  <rect x="{padding + inner_w - alliance_zone_w:.2f}" y="{padding}" '
        f'width="{alliance_zone_w:.2f}" height="{inner_h}" fill="#dc2626" fill-opacity="0.14" />',
        f'  <line x1="{mid_x:.2f}" y1="{padding}" x2="{mid_x:.2f}" y2="{padding + inner_h}" '
        f'stroke="#64748b" stroke-width="1.5" stroke-dasharray="6 4" />',
        f'  <text x="{padding + 8}" y="{padding + 16}" fill="#cbd5e1" '
        f'font-family="monospace" font-size="11">{escape(title)}</text>',
    ]

    alliance_lookup = alliances or {}
    for team_id, states in sorted(trajectories.items()):
        if not states:
            continue
        color_key = alliance_lookup.get(team_id, "blue")
        stroke = _ALLIANCE_STROKE.get(color_key, "#38bdf8")

        pts = [field_to_svg_xy(s.x, s.y, width=width, height=height, padding=padding) for s in states]
        point_str = " ".join(f"{px},{py}" for px, py in pts)
        lines.append(
            f'  <polyline data-team="{team_id}" fill="none" stroke="{stroke}" '
            f'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" '
            f'points="{point_str}" />'
        )
        end_x, end_y = pts[-1]
        lines.append(
            f'  <circle cx="{end_x}" cy="{end_y}" r="5" fill="{stroke}" '
            f'stroke="#0f172a" stroke-width="1.5" />'
        )
        lines.append(
            f'  <text x="{end_x + 7:.2f}" y="{end_y + 3:.2f}" fill="#f8fafc" '
            f'font-family="monospace" font-size="10">{team_id}</text>'
        )

    for event in engagements:
        def_states = trajectories.get(event.defender_id, ())
        match_state = next(
            (s for s in def_states if s.timestamp >= event.start_time),
            def_states[0] if def_states else None,
        )
        if match_state is None:
            continue
        ex, ey = field_to_svg_xy(
            match_state.x, match_state.y, width=width, height=height, padding=padding
        )
        ring_color = "#f59e0b" if event.pinning_risk else "#22d3ee"
        tag = "PIN" if event.pinning_risk else "DEF"
        lines.append(
            f'  <g data-engagement="{event.defender_id}-{event.attacker_id}">'
            f'<circle cx="{ex}" cy="{ey}" r="11" fill="{ring_color}" fill-opacity="0.2" '
            f'stroke="{ring_color}" stroke-width="2" />'
            f'<text x="{ex}" y="{ey - 14:.2f}" text-anchor="middle" fill="{ring_color}" '
            f'font-family="monospace" font-size="9">{tag} {event.duration_s:.1f}s</text>'
            f"</g>"
        )

    lines.append("</svg>")
    return "\n".join(lines)
