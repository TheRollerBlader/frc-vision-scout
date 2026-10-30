"""Field Homography and Coordinate Projection Utilities for FRC Vision Scout.

Maps 2D camera pixels to real-world FRC field surface coordinates (meters).
"""

from typing import List, Tuple, Optional
import numpy as np


class FieldHomography:
    """Manages 4-point planar homography calibration for FRC competition fields."""

    # Standard FRC field dimensions in meters (Reefscape 2025 / 2026 approx: 16.54m x 8.21m)
    FIELD_LENGTH_METERS = 16.54
    FIELD_WIDTH_METERS = 8.21

    def __init__(self, matrix: Optional[np.ndarray] = None):
        """Initializes with an optional 3x3 homography matrix."""
        if matrix is not None:
            if matrix.shape != (3, 3):
                raise ValueError("Homography matrix must be 3x3")
            self.H = matrix.astype(np.float64)
            self.H_inv = np.linalg.inv(self.H)
        else:
            self.H = np.eye(3, dtype=np.float64)
            self.H_inv = np.eye(3, dtype=np.float64)

    @classmethod
    def from_four_points(
        cls,
        image_points: List[Tuple[float, float]],
        field_points: List[Tuple[float, float]],
    ) -> "FieldHomography":
        """Computes homography matrix H mapping image points to field points using DLT / SVD.

        Args:
            image_points: 4 (x, y) coordinates in pixel space.
            field_points: 4 corresponding (x, y) coordinates in field space (meters).
        """
        if len(image_points) != 4 or len(field_points) != 4:
            raise ValueError("Exactly 4 point pairs are required for planar homography.")

        # Construct 8x9 Direct Linear Transform (DLT) matrix A
        A = []
        for (u, v), (x, y) in zip(image_points, field_points):
            A.append([-u, -v, -1, 0, 0, 0, u * x, v * x, x])
            A.append([0, 0, 0, -u, -v, -1, u * y, v * y, y])
        A = np.array(A, dtype=np.float64)

        # Solve Ah = 0 via Singular Value Decomposition
        _, _, Vh = np.linalg.svd(A)
        h = Vh[-1]
        H = h.reshape(3, 3)

        # Normalize so that H[2, 2] = 1 (if non-zero)
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]

        return cls(H)

    def project_to_field(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """Projects image coordinates (u, v) onto field coordinates (x, y) in meters."""
        p_img = np.array([pixel_x, pixel_y, 1.0], dtype=np.float64)
        p_field = self.H @ p_img

        if abs(p_field[2]) < 1e-12:
            raise ValueError("Point projects to infinity (singular point in perspective transform)")

        x = p_field[0] / p_field[2]
        y = p_field[1] / p_field[2]
        return float(x), float(y)

    def project_to_image(self, field_x: float, field_y: float) -> Tuple[float, float]:
        """Projects field coordinates (x, y) back onto image coordinates (u, v) in pixels."""
        p_field = np.array([field_x, field_y, 1.0], dtype=np.float64)
        p_img = self.H_inv @ p_field

        if abs(p_img[2]) < 1e-12:
            raise ValueError("Point projects to infinity")

        u = p_img[0] / p_img[2]
        v = p_img[1] / p_img[2]
        return float(u), float(v)

    def is_within_field(
        self,
        field_x: float,
        field_y: float,
        margin: float = 0.5,
    ) -> bool:
        """Checks if a field coordinate lies within valid FRC carpet boundaries."""
        return (
            -margin <= field_x <= self.FIELD_LENGTH_METERS + margin
            and -margin <= field_y <= self.FIELD_WIDTH_METERS + margin
        )
