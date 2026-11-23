"""Strict type-safe homography coordinate bounds for frc-vision-scout.
Scheduled Milestone: Architecture & Scaffolding (2026-11-23).
"""

from typing import Tuple, List
from pydantic import BaseModel, Field, model_validator


class PlanarPoint2D(BaseModel):
    u: float = Field(..., ge=0.0, description="Image horizontal pixel")
    v: float = Field(..., ge=0.0, description="Image vertical pixel")


class FRCFieldCalibrationContract(BaseModel):
    """Guarantees 4 non-degenerate calibration points mapping camera to field space."""
    milestone_date: str = Field(default="2026-11-23")
    camera_id: str = Field(..., min_length=1)
    resolution: Tuple[int, int] = Field(default=(1920, 1080))
    image_corners: List[PlanarPoint2D] = Field(..., min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_non_degenerate(self) -> "FRCFieldCalibrationContract":
        u_vals = [p.u for p in self.image_corners]
        v_vals = [p.v for p in self.image_corners]
        if max(u_vals) - min(u_vals) < 50 or max(v_vals) - min(v_vals) < 50:
            raise ValueError("Calibration bounding area is degenerate or collapsed")
        return self
