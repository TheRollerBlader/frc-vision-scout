"""High-precision numerical math utilities and moving window statistics for frc-vision-scout.
Scheduled Milestone: Core Logic & Computation (2026-11-10).
"""

from typing import List, Tuple
import math


def compute_running_statistics(values: List[float]) -> Tuple[float, float]:
    """Computes sample mean and standard deviation with numerical stability."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1 if n > 1 else 1)
    return mean, math.sqrt(variance)
