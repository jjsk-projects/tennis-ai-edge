"""
6-state Linear Kalman Filter for high-velocity tennis ball tracking.

Stage 1 of the edge tennis coaching pipeline. Models parabolic ball
kinetics under gravity (simplified aerodynamic drag folded into the
constant-acceleration model) to bridge temporal dropouts, net
crossings, and camera jitter.

State vector: x_k = [x, y, vx, vy, ax, ay]^T
Measurement:   z_k = [x, y] (ball box centroid)

The observation noise covariance R scales dynamically with the
estimated velocity magnitude |v|_2, so blurred (fast) detections
are down-weighted instead of corrupting trajectory continuity.

Compliance: pure numpy, no third-party deps, permissive use.
Target latency: < 2.8 ms/frame budget (filter alone is ~microseconds).
"""

from __future__ import annotations

import numpy as np


class TennisBallKalmanFilter:
    """Discrete-time 6-state Kalman filter for tennis ball trajectories.

    Parameters
    ----------
    dt : float
        Inter-frame time delta in seconds. Defaults to 1/60 for 60 FPS.
    p_scale : float
        Initial state covariance scale (high prior uncertainty so the
        first observations pull the state quickly).
    q_scale : float
        Process noise covariance scale (aerodynamic uncertainty).
    r_scale : float
        Base measurement noise variance accounting for optical blur
        (px^2 / m^2 depending on the coordinate domain used).
    """

    def __init__(self, dt: float = 1.0 / 60.0, p_scale: float = 50.0,
                 q_scale: float = 0.5, r_scale: float = 0.25):
        self.dt = dt

        # State vector: [x, y, vx, vy, ax, ay]
        self.x = np.zeros((6, 1), dtype=np.float64)

        # State transition matrix (constant-acceleration kinematics)
        self.F = self._build_F()

        # Measurement matrix: observe spatial coordinates only
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float64)

        # State estimation error covariance
        self.P = np.eye(6, dtype=np.float64) * p_scale

        # Process noise covariance. Position/velocity inherit the state
        # acceleration through F, so the constant-accel assumption is only
        # enforced through excitation of the acceleration states (modeling
        # aerodynamic drag); position/velocity Q terms are ~0.
        self.Q = np.zeros((6, 6), dtype=np.float64)
        self.Q[4, 4] = q_scale
        self.Q[5, 5] = q_scale

        # Measurement noise covariance (optical blur)
        self.R_base = np.array([[r_scale, 0.0], [0.0, r_scale]], dtype=np.float64)
        self.R = self.R_base.copy()

    def _build_F(self) -> np.ndarray:
        """Constant-acceleration state-transition matrix for ``self.dt``.

        Rebuilt whenever the effective timestep changes (see ``predict(dt)``)
        so the filter's time base matches the real observation interval.
        """
        dt = self.dt
        d2 = 0.5 * dt * dt
        return np.array([
            [1.0, 0.0, dt,   0.0, d2,   0.0],
            [0.0, 1.0, 0.0,  dt,   0.0, d2],
            [0.0, 0.0, 1.0,  0.0,  dt,   0.0],
            [0.0, 0.0, 0.0,  1.0,  0.0,  dt],
            [0.0, 0.0, 0.0,  0.0,  1.0,  0.0],
            [0.0, 0.0, 0.0,  0.0,  0.0,  1.0],
        ], dtype=np.float64)

    def _adaptive_r(self) -> None:
        """Scale R with estimated velocity magnitude |v|_2.

        Fast ball => larger detection boxes from motion blur => less
        trustworthy centroids => larger measurement variance.
        """
        speed = float(np.hypot(self.x[2, 0], self.x[3, 0]))
        self.R = self.R_base * (1.0 + speed / 10.0)

    def predict(self, dt: float | None = None) -> np.ndarray:
        """Propagate state and covariance one step: x = F x, P = F P F^T + Q.

        If ``dt`` is given and differs from the filter's base timestep, the
        state-transition matrix ``F`` is rebuilt for that step. This is what
        lets the orchestrator advance the filter by the *real* elapsed time
        (e.g. ``pose_every / fps``) when observations arrive on a throttled
        cadence rather than every video frame.
        """
        if dt is not None and dt != self.dt:
            self.dt = dt
            self.F = self._build_F()  # was discarded before -> dt changes had no effect
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2, 0]

    def update(self, z: np.ndarray) -> np.ndarray:
        """Incorporate an observed position z = [x_obs, y_obs].

        Computes innovation, Kalman gain, and updates state/covariance
        (Levine's form for numerical stability of P). Returns the
        corrected 2D position [x, y].
        """
        z = np.asarray(z, dtype=np.float64).reshape(2, 1)
        self._adaptive_r()

        y_innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_innov
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        return self.x[:2, 0]
