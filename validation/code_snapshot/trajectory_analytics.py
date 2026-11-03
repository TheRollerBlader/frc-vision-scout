"""FRC Robot Kinematic Trajectory & Scoring Cycle Analytics Engine.

Applies discrete 2D Kalman state estimation (position + velocity) to filter noisy
field projections, rejects unphysical swerve drive velocity spikes, tracks zone occupancy,
and calculates autonomous/teleop scoring cycle performance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


@dataclass
class KinematicState:
    """Estimated physical robot state at a given timestamp."""
    timestamp: float
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    acceleration: float
    heading_deg: float
    is_outlier: bool = False


@dataclass
class CycleEvent:
    """Detected transit cycle between feeding/loading station and scoring reef."""
    start_time: float
    end_time: float
    duration_sec: float
    distance_traveled_m: float
    avg_speed_mps: float


@dataclass
class TrajectorySummary:
    """Aggregated match trajectory metrics for a single tracked robot."""
    track_id: int
    total_distance_m: float
    peak_speed_mps: float
    mean_speed_mps: float
    time_in_opponent_zone_pct: float
    completed_cycles: List[CycleEvent] = field(default_factory=list)


class KinematicKalmanFilter:
    """2D Constant-Velocity Kinematic Kalman Filter for FRC ground tracking.

    State vector: [x, y, vx, vy]^T
    Measurement vector: [x, y]^T
    """

    def __init__(
        self,
        initial_x: float,
        initial_y: float,
        process_noise_std: float = 0.5,
        measurement_noise_std: float = 0.15,
    ):
        # State: [x, y, vx, vy]
        self.x = np.array([initial_x, initial_y, 0.0, 0.0], dtype=np.float64)

        # State covariance
        self.P = np.diag([0.2, 0.2, 1.0, 1.0]).astype(np.float64)

        # Measurement matrix H
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

        # Measurement noise covariance R
        self.R = np.eye(2, dtype=np.float64) * (measurement_noise_std ** 2)

        # Process noise variance parameter
        self.q_var = process_noise_std ** 2

    def predict(self, dt: float) -> np.ndarray:
        """Predicts state ahead by dt seconds."""
        if dt <= 0:
            return self.x

        # State transition matrix F
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Continuous white noise acceleration model for Q
        dt2 = (dt ** 2) / 2.0
        dt3 = (dt ** 3) / 3.0
        dt4 = (dt ** 4) / 4.0

        Q = self.q_var * np.array(
            [
                [dt4, 0.0, dt3, 0.0],
                [0.0, dt4, 0.0, dt3],
                [dt3, 0.0, dt2, 0.0],
                [0.0, dt3, 0.0, dt2],
            ],
            dtype=np.float64,
        )

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x

    def update(self, measured_x: float, measured_y: float) -> np.ndarray:
        """Corrects state using camera-derived field observation."""
        z = np.array([measured_x, measured_y], dtype=np.float64)
        y = z - (self.H @ self.x)  # Measurement residual

        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float64) - K @ self.H) @ self.P
        return self.x


class TrajectoryAnalyticsEngine:
    """Manages multi-robot trajectory smoothing, kinematic filtering, and cycle timing."""

    # FRC 2025/2026 Reefscape field boundaries
    FIELD_LENGTH_M = 16.54
    FIELD_WIDTH_M = 8.21

    # Maximum physical robot speed for modern FRC swerve drives (L3/L4 gearing ~ 5.5 - 6.0 m/s)
    MAX_PHYSICAL_SPEED_MPS = 6.2

    def __init__(self, alliance_color: str = "blue"):
        self.alliance = alliance_color.lower()
        self.filters: Dict[int, KinematicKalmanFilter] = {}
        self.history: Dict[int, List[KinematicState]] = {}
        self.last_timestamps: Dict[int, float] = {}
        self.cycle_state: Dict[int, Optional[str]] = {}
        self.cycle_start_time: Dict[int, float] = {}
        self.cycle_start_dist: Dict[int, float] = {}
        self.completed_cycles: Dict[int, List[CycleEvent]] = {}
        self.cumulative_distance: Dict[int, float] = {}

    def classify_zone(self, x: float, y: float) -> str:
        """Classifies robot position into strategic FRC field areas."""
        # Source / Human player loading zone: ends of field (x < 2.0 or x > 14.54)
        if x < 2.0 or x > (self.FIELD_LENGTH_M - 2.0):
            return "LOADING_STATION"

        # Center scoring reefs (approx middle third)
        if 4.5 <= x <= 12.0 and 2.0 <= y <= 6.2:
            return "SCORING_REEF"

        # Opponent territory check based on half-field
        midfield = self.FIELD_LENGTH_M / 2.0
        if (self.alliance == "blue" and x > midfield) or (self.alliance == "red" and x < midfield):
            return "OPPONENT_TERRITORY"

        return "MIDFIELD"

    def add_point(
        self,
        track_id: int,
        timestamp: float,
        measured_x: float,
        measured_y: float,
    ) -> KinematicState:
        """Processes an incoming raw field coordinate for a robot tracklet."""
        if track_id not in self.filters:
            # Initialize tracklet
            self.filters[track_id] = KinematicKalmanFilter(measured_x, measured_y)
            self.history[track_id] = []
            self.last_timestamps[track_id] = timestamp
            self.cumulative_distance[track_id] = 0.0
            self.completed_cycles[track_id] = []
            self.cycle_state[track_id] = self.classify_zone(measured_x, measured_y)
            self.cycle_start_time[track_id] = timestamp
            self.cycle_start_dist[track_id] = 0.0

            state = KinematicState(
                timestamp=timestamp,
                x=measured_x,
                y=measured_y,
                vx=0.0,
                vy=0.0,
                speed=0.0,
                acceleration=0.0,
                heading_deg=0.0,
            )
            self.history[track_id].append(state)
            return state

        kf = self.filters[track_id]
        prev_time = self.last_timestamps[track_id]
        dt = max(1e-4, timestamp - prev_time)
        self.last_timestamps[track_id] = timestamp

        # Predict ahead
        predicted_x = kf.predict(dt)

        # Check physical speed limits (reject teleports / camera noise)
        last_state = self.history[track_id][-1]
        raw_displacement = math.hypot(measured_x - last_state.x, measured_y - last_state.y)
        implied_speed = raw_displacement / dt

        is_outlier = implied_speed > self.MAX_PHYSICAL_SPEED_MPS
        if not is_outlier:
            # Valid observation, perform measurement update
            updated_state = kf.update(measured_x, measured_y)
        else:
            # Reject anomalous measurement, rely on forward kinematic prediction
            updated_state = predicted_x

        x, y, vx, vy = (
            float(updated_state[0]),
            float(updated_state[1]),
            float(updated_state[2]),
            float(updated_state[3]),
        )
        current_speed = math.hypot(vx, vy)
        accel = (current_speed - last_state.speed) / dt
        heading = (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0 if current_speed > 0.05 else last_state.heading_deg

        # Update cumulative distance
        step_dist = math.hypot(x - last_state.x, y - last_state.y)
        self.cumulative_distance[track_id] += step_dist

        kin_state = KinematicState(
            timestamp=timestamp,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            speed=current_speed,
            acceleration=accel,
            heading_deg=heading,
            is_outlier=is_outlier,
        )
        self.history[track_id].append(kin_state)

        # Update cycle state machine
        current_zone = self.classify_zone(x, y)
        self._update_cycle_state(track_id, current_zone, timestamp)

        return kin_state

    def _update_cycle_state(self, track_id: int, current_zone: str, timestamp: float) -> None:
        """Tracks complete travel cycles between feeding loading station and scoring reef."""
        state = self.cycle_state.get(track_id)
        if state == "LOADING_STATION" and current_zone == "SCORING_REEF":
            # Completed transit from loading station to reef
            start_t = self.cycle_start_time.get(track_id, timestamp)
            duration = max(0.5, timestamp - start_t)
            start_dist = self.cycle_start_dist.get(track_id, 0.0)
            total_dist = self.cumulative_distance[track_id]
            cycle_dist = max(1.0, total_dist - start_dist)

            event = CycleEvent(
                start_time=start_t,
                end_time=timestamp,
                duration_sec=duration,
                distance_traveled_m=cycle_dist,
                avg_speed_mps=cycle_dist / duration,
            )
            self.completed_cycles[track_id].append(event)

            # Transition state to reef
            self.cycle_state[track_id] = "SCORING_REEF"
            self.cycle_start_time[track_id] = timestamp
            self.cycle_start_dist[track_id] = total_dist

        elif state == "SCORING_REEF" and current_zone == "LOADING_STATION":
            # Returned to loading station, start new cycle
            self.cycle_state[track_id] = "LOADING_STATION"
            self.cycle_start_time[track_id] = timestamp
            self.cycle_start_dist[track_id] = self.cumulative_distance[track_id]

    def get_summary(self, track_id: int) -> Optional[TrajectorySummary]:
        """Calculates final kinematic summary for a given tracklet."""
        hist = self.history.get(track_id, [])
        if not hist:
            return None

        speeds = [s.speed for s in hist]
        total_dist = self.cumulative_distance.get(track_id, 0.0)
        peak_speed = max(speeds) if speeds else 0.0
        mean_speed = float(np.mean(speeds)) if speeds else 0.0

        # Opponent territory occupancy percentage
        opp_count = sum(1 for s in hist if self.classify_zone(s.x, s.y) == "OPPONENT_TERRITORY")
        opp_pct = (opp_count / len(hist)) * 100.0 if hist else 0.0

        return TrajectorySummary(
            track_id=track_id,
            total_distance_m=total_dist,
            peak_speed_mps=peak_speed,
            mean_speed_mps=mean_speed,
            time_in_opponent_zone_pct=opp_pct,
            completed_cycles=self.completed_cycles.get(track_id, []),
        )
