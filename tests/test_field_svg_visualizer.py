"""Unit tests for the 2D FRC field trajectory and defense SVG visualizer."""

from validation.code_snapshot.defense_engagement import DefenseEngagementEvent
from validation.code_snapshot.field_svg_visualizer import (
    field_to_svg_xy,
    render_field_trajectories_svg,
)
from validation.code_snapshot.homography import FieldHomography
from validation.code_snapshot.trajectory_analytics import KinematicState


def _state(t: float, x: float, y: float) -> KinematicState:
    return KinematicState(
        timestamp=t,
        x=x,
        y=y,
        vx=1.0,
        vy=0.0,
        speed=1.0,
        acceleration=0.0,
        heading_deg=0.0,
    )


class TestFieldSvgVisualizer:
    def test_field_to_svg_xy_clamps_out_of_bounds_coordinates(self):
        px_min, py_min = field_to_svg_xy(-4.0, -2.0, width=880, height=440, padding=24)
        px_max, py_max = field_to_svg_xy(99.0, 99.0, width=880, height=440, padding=24)

        assert px_min == 24.0
        assert py_min == 416.0  # y=0 maps to bottom inner edge
        assert px_max == 856.0
        assert py_max == 24.0  # y=FIELD_WIDTH maps to top inner edge

        mid_x, _ = field_to_svg_xy(
            FieldHomography.FIELD_LENGTH_METERS / 2.0, 4.0, width=880, height=440, padding=24
        )
        assert mid_x == 440.0

    def test_renders_trajectories_and_pinning_engagement_markers(self):
        trajectories = {
            254: [_state(0.0, 2.0, 2.0), _state(1.0, 6.0, 1.0)],
            1678: [_state(0.0, 14.0, 6.0), _state(1.0, 6.2, 0.5)],
        }
        engagements = [
            DefenseEngagementEvent(
                defender_id=254,
                attacker_id=1678,
                start_time=0.8,
                end_time=4.0,
                duration_s=3.2,
                min_separation_m=0.55,
                mean_separation_m=0.62,
                max_closing_speed_mps=2.4,
                pinning_risk=True,
            )
        ]

        svg = render_field_trajectories_svg(
            trajectories=trajectories,
            alliances={254: "blue", 1678: "red"},
            engagements=engagements,
            title="Qual 42 Replay",
        )

        assert svg.startswith("<svg ")
        assert svg.endswith("</svg>")
        assert "Qual 42 Replay" in svg
        assert 'data-team="254"' in svg
        assert 'stroke="#38bdf8"' in svg
        assert 'stroke="#f87171"' in svg
        assert 'data-engagement="254-1678"' in svg
        assert "PIN 3.2s" in svg

    def test_handles_empty_trajectories_cleanly(self):
        svg = render_field_trajectories_svg(trajectories={})
        assert "<svg " in svg
        assert "<polyline" not in svg
        assert svg.endswith("</svg>")
