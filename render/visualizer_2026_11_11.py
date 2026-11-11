"""Tactical rendering and visualization helpers for frc-vision-scout.
Scheduled Milestone: UI & Visualization Components (2026-11-11).
"""

from typing import Dict, Any, List, Tuple


def render_ascii_sparkline(values: List[float], width: int = 20) -> str:
    """Renders a lightweight CLI ASCII sparkline for time-series inspection."""
    if not values:
        return "[no data]"
    chars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    min_v, max_v = min(values), max(values)
    rng = max_v - min_v if max_v != min_v else 1.0
    return "".join(chars[min(len(chars) - 1, int((v - min_v) / rng * (len(chars) - 1)))] for v in values[-width:])


def format_status_badge(component: str, is_healthy: bool, latency_ms: float) -> str:
    """Formats standardized status indicator string for frc-vision-scout dashboards."""
    status = "OK" if is_healthy else "WARN"
    return f"[{status}] {component}: {latency_ms:.1f}ms latency"
