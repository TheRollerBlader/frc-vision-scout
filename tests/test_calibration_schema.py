"""Tests for FRC Field Homography Calibration Schemas, Type Safety, and Condition Quality."""

import pytest
import numpy as np
from pydantic import ValidationError
from validation.code_snapshot.calibration_schema import (
    Point2D,
    CalibrationPointPair,
    FieldGeometrySpec,
    HomographyCalibrationConfig,
    evaluate_homography_quality,
)
from validation.code_snapshot.homography import FieldHomography


class TestCalibrationSchema:
    def test_point2d_valid_and_immutable_conversion(self):
        pt = Point2D(x=100.5, y=200.25)
        assert pt.as_tuple() == (100.5, 200.25)

    def test_point2d_rejects_nan_and_inf(self):
        with pytest.raises(ValidationError):
            Point2D(x=float("nan"), y=10.0)
        with pytest.raises(ValidationError):
            Point2D(x=10.0, y=float("inf"))

    def test_homography_calibration_config_valid(self):
        img_pts = [(100, 100), (1800, 100), (1800, 900), (100, 900)]
        field_pts = [(0.0, 0.0), (16.54, 0.0), (16.54, 8.21), (0.0, 8.21)]

        pairs = [
            CalibrationPointPair(
                image_pt=Point2D(x=u, y=v),
                field_pt=Point2D(x=x, y=y),
                landmark_name=f"Corner {i+1}",
            )
            for i, ((u, v), (x, y)) in enumerate(zip(img_pts, field_pts))
        ]

        config = HomographyCalibrationConfig(
            camera_id="arena_overhead_1",
            image_resolution=(1920, 1080),
            point_pairs=pairs,
        )
        assert config.camera_id == "arena_overhead_1"
        assert len(config.point_pairs) == 4
        assert config.geometry.length_meters == 16.54

    def test_homography_calibration_config_rejects_collinear_points(self):
        # 3 collinear points along horizontal line in image space
        img_pts = [(100, 100), (500, 100), (900, 100), (100, 900)]
        field_pts = [(0.0, 0.0), (8.0, 0.0), (16.54, 0.0), (0.0, 8.21)]

        pairs = [
            CalibrationPointPair(
                image_pt=Point2D(x=u, y=v),
                field_pt=Point2D(x=x, y=y),
            )
            for (u, v), (x, y) in zip(img_pts, field_pts)
        ]

        with pytest.raises(ValidationError) as excinfo:
            HomographyCalibrationConfig(
                camera_id="cam_side",
                image_resolution=(1920, 1080),
                point_pairs=pairs,
            )
        assert "Collinear points detected" in str(excinfo.value)

    def test_evaluate_homography_quality_well_conditioned(self):
        img_pts = [(100.0, 100.0), (1800.0, 100.0), (1800.0, 900.0), (100.0, 900.0)]
        field_pts = [(0.0, 0.0), (16.54, 0.0), (16.54, 8.21), (0.0, 8.21)]

        solver = FieldHomography.from_four_points(img_pts, field_pts)
        metrics = evaluate_homography_quality(solver.H, img_pts, field_pts)

        assert metrics.mean_reprojection_error_px < 1.0
        assert metrics.max_reprojection_error_px < 1.0
        assert metrics.is_well_conditioned is True
        assert "stable and accurate" in metrics.recommendation

    def test_evaluate_homography_quality_ill_conditioned(self):
        # Degenerate matrix with huge condition number
        degenerate_H = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.000001, 3.0],
            [0.0, 0.0, 1.0],
        ])
        img_pts = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]
        field_pts = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]

        metrics = evaluate_homography_quality(degenerate_H, img_pts, field_pts)
        assert metrics.is_well_conditioned is False
        assert metrics.condition_number > 1e5
