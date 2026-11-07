"""Type-safe schemas, geometric validations, and condition number guardrails
for FRC camera-to-field homography calibration.

Implements Week 1 Milestone: Saturday - Type Safety, Linting & Polish.
"""

from typing import List, Tuple, Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class Point2D(BaseModel):
    """Represents a 2D coordinate in pixel space or metric field space."""
    x: float = Field(..., description="Horizontal coordinate")
    y: float = Field(..., description="Vertical coordinate")

    @field_validator("x", "y")
    @classmethod
    def validate_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("Coordinate must be a finite numerical value")
        return float(v)

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


class CalibrationPointPair(BaseModel):
    """Maps a 2D image coordinate (pixels) to a known 2D ground truth field coordinate (meters)."""
    image_pt: Point2D = Field(..., description="Camera pixel coordinate (u, v)")
    field_pt: Point2D = Field(..., description="Metric field coordinate (x, y)")
    landmark_name: Optional[str] = Field(None, description="Optional descriptor e.g. 'Blue Reef Apex'")


class FieldGeometrySpec(BaseModel):
    """FRC competition field perimeter dimensions."""
    length_meters: float = Field(default=16.54, ge=5.0, le=30.0, description="Field length along X-axis")
    width_meters: float = Field(default=8.21, ge=4.0, le=20.0, description="Field width along Y-axis")
    season_name: str = Field(default="Reefscape 2025/2026", description="FRC game name")


class HomographyCalibrationConfig(BaseModel):
    """Complete configuration specification for 4-point planar field calibration."""
    camera_id: str = Field(..., min_length=1, description="Camera stream identifier e.g. 'cam_driver_station_blue'")
    image_resolution: Tuple[int, int] = Field(..., description="Image resolution (width, height) in pixels")
    point_pairs: List[CalibrationPointPair] = Field(..., min_length=4, max_length=4, description="Exactly 4 point correspondences")
    geometry: FieldGeometrySpec = Field(default_factory=FieldGeometrySpec)

    @field_validator("image_resolution")
    @classmethod
    def validate_resolution(cls, res: Tuple[int, int]) -> Tuple[int, int]:
        w, h = res
        if w <= 0 or h <= 0:
            raise ValueError("Resolution dimensions must be positive integers")
        return (int(w), int(h))

    @model_validator(mode="after")
    def validate_non_collinear_and_conditioned(self) -> "HomographyCalibrationConfig":
        """Ensures that no 3 points are collinear in image or field space, preventing singular matrices."""
        img_pts = [p.image_pt.as_tuple() for p in self.point_pairs]
        field_pts = [p.field_pt.as_tuple() for p in self.point_pairs]

        for pts, space in [(img_pts, "image"), (field_pts, "field")]:
            for i in range(4):
                p1 = np.array(pts[i])
                p2 = np.array(pts[(i + 1) % 4])
                p3 = np.array(pts[(i + 2) % 4])
                # Triangle area formula: 0.5 * |(x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2))|
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area < 1e-4:
                    raise ValueError(f"Collinear points detected in {space} space: points {i}, {(i+1)%4}, {(i+2)%4} form triangle area {area:.6f}")

        return self


class HomographyQualityMetrics(BaseModel):
    """Mathematical health and precision metrics for a computed homography matrix."""
    condition_number: float = Field(..., description="Matrix condition number (norm-2)")
    max_reprojection_error_px: float = Field(..., description="Worst-case pixel reprojection delta")
    mean_reprojection_error_px: float = Field(..., description="Average pixel reprojection delta")
    is_well_conditioned: bool = Field(..., description="True if condition number < 1e5 and mean error < 5px")
    recommendation: str = Field(..., description="Guidance for field calibration operator")


def evaluate_homography_quality(
    H: np.ndarray,
    image_points: List[Tuple[float, float]],
    field_points: List[Tuple[float, float]]
) -> HomographyQualityMetrics:
    """Computes condition number and bidirectional reprojection residuals for homography H."""
    if H.shape != (3, 3):
        raise ValueError("Homography must be a 3x3 matrix")

    # Singular value ratio (condition number)
    cond = float(np.linalg.cond(H))

    errors: List[float] = []
    H_inv = np.linalg.inv(H)

    for (u, v), (x, y) in zip(image_points, field_points):
        # Reproject field point back into image space
        p_field = np.array([x, y, 1.0], dtype=np.float64)
        p_img_proj = H_inv @ p_field
        if abs(p_img_proj[2]) > 1e-12:
            u_proj = p_img_proj[0] / p_img_proj[2]
            v_proj = p_img_proj[1] / p_img_proj[2]
            err = float(np.sqrt((u - u_proj) ** 2 + (v - v_proj) ** 2))
            errors.append(err)
        else:
            errors.append(float("inf"))

    max_err = max(errors) if errors else float("inf")
    mean_err = float(np.mean(errors)) if errors else float("inf")
    well_conditioned = (cond < 1e5) and (mean_err < 5.0)

    if well_conditioned:
        rec = "Calibration is mathematically stable and accurate for match scouting."
    elif cond >= 1e5:
        rec = f"Ill-conditioned matrix (condition number {cond:.1e}). Select calibration points with wider spatial distribution."
    else:
        rec = f"High reprojection error ({mean_err:.2f} px). Check field landmark corner associations."

    return HomographyQualityMetrics(
        condition_number=cond,
        max_reprojection_error_px=max_err,
        mean_reprojection_error_px=mean_err,
        is_well_conditioned=well_conditioned,
        recommendation=rec,
    )
