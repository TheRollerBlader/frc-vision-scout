"""Bumper distortion guards, perspective shear validation, and type-safe OCR boundaries
for FRC robot tracking and scouting.

Enforces FRC bumper physical geometry invariants and filters out severe perspective
distortions that induce false positive OCR readings.
"""

from typing import Optional, Tuple, Dict, Any
import math
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class BumperBox(BaseModel):
    """Bounding box for a detected robot bumper in pixel space (u, v)."""
    x_min: float = Field(..., ge=0.0, description="Left pixel coordinate")
    y_min: float = Field(..., ge=0.0, description="Top pixel coordinate")
    x_max: float = Field(..., gt=0.0, description="Right pixel coordinate")
    y_max: float = Field(..., gt=0.0, description="Bottom pixel coordinate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="YOLO detection confidence")

    @model_validator(mode="after")
    def validate_spatial_ordering(self) -> "BumperBox":
        if self.x_max <= self.x_min:
            raise ValueError(f"x_max ({self.x_max}) must be strictly greater than x_min ({self.x_min})")
        if self.y_max <= self.y_min:
            raise ValueError(f"y_max ({self.y_max}) must be strictly greater than y_min ({self.y_min})")
        return self

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def aspect_ratio(self) -> float:
        """Width-to-height ratio."""
        return self.width / (self.height if self.height > 0 else 1e-6)

    @property
    def area(self) -> float:
        return self.width * self.height


class BumperDistortionMetrics(BaseModel):
    """Quantitative distortion metrics assessing optical readability of bumper digits."""
    aspect_ratio: float = Field(..., gt=0.0, description="Width-to-height ratio")
    is_aspect_valid: bool = Field(..., description="Whether aspect ratio conforms to FRC bumper dimensions")
    estimated_shear_deg: float = Field(..., ge=0.0, le=90.0, description="Estimated perspective shear in degrees")
    is_shear_acceptable: bool = Field(..., description="Whether shear is below the threshold for reliable OCR")
    overall_readability_score: float = Field(..., ge=0.0, le=1.0, description="Aggregate composite readability (0 to 1)")


class GatedBumperOCRResult(BaseModel):
    """Validated, type-safe representation of an FRC bumper OCR recognition event."""
    team_number: int = Field(..., ge=1, le=99999, description="Sanctioned FRC team number")
    alliance: str = Field(..., description="Alliance color ('red' or 'blue')")
    raw_text: str = Field(..., min_length=1, description="Raw OCR decoded string")
    ocr_confidence: float = Field(..., ge=0.0, le=1.0, description="OCR text recognition probability")
    distortion_metrics: BumperDistortionMetrics
    is_trusted: bool = Field(..., description="Whether detection passes all physical and optical filters")

    @field_validator("alliance")
    @classmethod
    def validate_alliance_color(cls, v: str) -> str:
        color = v.strip().lower()
        if color not in ("red", "blue"):
            raise ValueError(f"Alliance must be 'red' or 'blue', got '{v}'")
        return color


def evaluate_bumper_distortion(
    box: BumperBox,
    min_aspect: float = 1.2,
    max_aspect: float = 10.0,
    max_shear_deg: float = 40.0,
    min_area_pixels: float = 200.0,
) -> BumperDistortionMetrics:
    """Evaluates geometric bounding box characteristics to determine if a bumper detection
    is physically plausible and optically conditioned for OCR.
    """
    ar = box.aspect_ratio
    area = box.area

    # Physical FRC bumpers are horizontally elongated bands
    is_aspect_valid = (min_aspect <= ar <= max_aspect) and (area >= min_area_pixels)

    # Approximate perspective slant based on deviation from standard planar viewing ratio (~3.5:1)
    nominal_ar = 3.5
    shear_ratio = abs(ar - nominal_ar) / nominal_ar
    estimated_shear = float(np.clip(math.degrees(math.atan(shear_ratio)), 0.0, 90.0))
    is_shear_acceptable = estimated_shear <= max_shear_deg

    # Composite score penalized by extreme shear, small bounding boxes, or low YOLO confidence
    area_factor = min(1.0, area / 1500.0)
    shear_penalty = max(0.0, 1.0 - (estimated_shear / 60.0))
    aspect_penalty = 1.0 if is_aspect_valid else 0.2

    readability = float(np.clip(box.confidence * area_factor * shear_penalty * aspect_penalty, 0.0, 1.0))

    return BumperDistortionMetrics(
        aspect_ratio=round(ar, 3),
        is_aspect_valid=is_aspect_valid,
        estimated_shear_deg=round(estimated_shear, 1),
        is_shear_acceptable=is_shear_acceptable,
        overall_readability_score=round(readability, 3),
    )


def gate_bumper_ocr_detection(
    raw_digits: str,
    raw_confidence: float,
    alliance: str,
    box: BumperBox,
    min_readability_threshold: float = 0.25,
) -> Optional[GatedBumperOCRResult]:
    """Applies strict geometric and character filters to an unverified bumper detection.
    Returns GatedBumperOCRResult if valid, or None if filtered out by optical guards.
    """
    # 1. Clean string: remove non-digits
    cleaned = "".join(ch for ch in raw_digits if ch.isdigit())
    if not cleaned:
        return None

    try:
        team_num = int(cleaned)
    except ValueError:
        return None

    if team_num < 1 or team_num > 99999:
        return None

    # 2. Geometric distortion evaluation
    distortion = evaluate_bumper_distortion(box)

    # 3. Overall trust assessment
    is_trusted = (
        distortion.is_aspect_valid
        and distortion.is_shear_acceptable
        and distortion.overall_readability_score >= min_readability_threshold
        and raw_confidence >= 0.35
    )

    return GatedBumperOCRResult(
        team_number=team_num,
        alliance=alliance,
        raw_text=raw_digits,
        ocr_confidence=round(float(raw_confidence), 3),
        distortion_metrics=distortion,
        is_trusted=is_trusted,
    )
