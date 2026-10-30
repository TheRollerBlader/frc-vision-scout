import pytest
import numpy as np
from homography import FieldHomography


class TestFieldHomography:
    def test_identity_projection(self):
        h = FieldHomography()
        fx, fy = h.project_to_field(10.0, 5.0)
        assert pytest.approx(fx, 1e-4) == 10.0
        assert pytest.approx(fy, 1e-4) == 5.0

    def test_four_point_planar_reconstruction(self):
        # 4 corners of a 1000x500 px camera viewport mapped to 16.54m x 8.21m field
        img_pts = [(0.0, 0.0), (1000.0, 0.0), (1000.0, 500.0), (0.0, 500.0)]
        field_pts = [(0.0, 0.0), (16.54, 0.0), (16.54, 8.21), (0.0, 8.21)]

        homography = FieldHomography.from_four_points(img_pts, field_pts)

        # Test forward projections on all 4 corners
        for (u, v), (exp_x, exp_y) in zip(img_pts, field_pts):
            proj_x, proj_y = homography.project_to_field(u, v)
            assert pytest.approx(proj_x, abs=1e-3) == exp_x
            assert pytest.approx(proj_y, abs=1e-3) == exp_y

        # Test center point projection (500, 250) -> (8.27, 4.105)
        cx, cy = homography.project_to_field(500.0, 250.0)
        assert pytest.approx(cx, abs=1e-2) == 16.54 / 2
        assert pytest.approx(cy, abs=1e-2) == 8.21 / 2

        # Test inverse projection back to image
        u_back, v_back = homography.project_to_image(cx, cy)
        assert pytest.approx(u_back, abs=1e-2) == 500.0
        assert pytest.approx(v_back, abs=1e-2) == 250.0

    def test_field_boundary_checking(self):
        h = FieldHomography()
        assert h.is_within_field(8.0, 4.0) is True
        assert h.is_within_field(0.0, 0.0) is True
        assert h.is_within_field(16.54, 8.21) is True
        assert h.is_within_field(-1.0, 4.0) is False  # Outside left boundary
        assert h.is_within_field(20.0, 4.0) is False  # Outside right boundary
        assert h.is_within_field(8.0, 10.0) is False  # Outside width boundary
