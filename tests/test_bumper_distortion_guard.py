"""Test suite for FRC bumper distortion guards and OCR confidence gating.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validation", "code_snapshot"))

from bumper_distortion_guard import (
    BumperBox,
    BumperDistortionMetrics,
    GatedBumperOCRResult,
    evaluate_bumper_distortion,
    gate_bumper_ocr_detection,
)


class TestBumperDistortionGuards:
    def test_bumper_box_creation_and_properties(self):
        box = BumperBox(x_min=100.0, y_min=200.0, x_max=240.0, y_max=240.0, confidence=0.88)
        assert box.width == 140.0
        assert box.height == 40.0
        assert box.aspect_ratio == 3.5
        assert box.area == 5600.0

    def test_bumper_box_rejects_inverted_coordinates(self):
        with pytest.raises(ValueError, match="x_max .* must be strictly greater than x_min"):
            BumperBox(x_min=300.0, y_min=200.0, x_max=250.0, y_max=240.0, confidence=0.8)

        with pytest.raises(ValueError, match="y_max .* must be strictly greater than y_min"):
            BumperBox(x_min=100.0, y_min=240.0, x_max=250.0, y_max=200.0, confidence=0.8)

    def test_evaluate_distortion_nominal_case(self):
        # A typical bumper: 140x40 pixels, ratio 3.5, high confidence
        box = BumperBox(x_min=50.0, y_min=50.0, x_max=190.0, y_max=90.0, confidence=0.92)
        metrics = evaluate_bumper_distortion(box)
        assert metrics.is_aspect_valid is True
        assert metrics.aspect_ratio == 3.5
        assert metrics.estimated_shear_deg == 0.0
        assert metrics.is_shear_acceptable is True
        assert metrics.overall_readability_score > 0.8

    def test_evaluate_distortion_extreme_perspective_aspect(self):
        # Very narrow vertical box (e.g. side-on edge view 15x80 pixels)
        box = BumperBox(x_min=50.0, y_min=50.0, x_max=65.0, y_max=130.0, confidence=0.60)
        metrics = evaluate_bumper_distortion(box)
        assert metrics.is_aspect_valid is False
        assert metrics.is_shear_acceptable is False
        assert metrics.overall_readability_score < 0.2

    def test_gate_ocr_detection_accepts_valid_reading(self):
        box = BumperBox(x_min=100.0, y_min=100.0, x_max=240.0, y_max=140.0, confidence=0.90)
        result = gate_bumper_ocr_detection("5411", 0.85, "blue", box)
        assert result is not None
        assert result.team_number == 5411
        assert result.alliance == "blue"
        assert result.is_trusted is True

    def test_gate_ocr_detection_cleans_punctuation_and_whitespace(self):
        box = BumperBox(x_min=100.0, y_min=100.0, x_max=240.0, y_max=140.0, confidence=0.90)
        result = gate_bumper_ocr_detection(" #254 ", 0.92, "RED", box)
        assert result is not None
        assert result.team_number == 254
        assert result.alliance == "red"
        assert result.is_trusted is True

    def test_gate_ocr_detection_rejects_empty_or_non_numeric(self):
        box = BumperBox(x_min=100.0, y_min=100.0, x_max=240.0, y_max=140.0, confidence=0.90)
        assert gate_bumper_ocr_detection("???", 0.90, "blue", box) is None
        assert gate_bumper_ocr_detection("", 0.90, "blue", box) is None

    def test_gate_ocr_detection_flags_untrusted_under_heavy_distortion(self):
        # Extreme skewed bounding box
        box = BumperBox(x_min=100.0, y_min=100.0, x_max=115.0, y_max=200.0, confidence=0.40)
        result = gate_bumper_ocr_detection("3355", 0.50, "red", box)
        assert result is not None
        assert result.is_trusted is False
