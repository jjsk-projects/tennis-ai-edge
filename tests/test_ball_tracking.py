"""
TDD Test Suite: Stage 1 - Ball Trajectory Tracking Engine

Tests for TennisBallKalmanFilter as specified in the spec:
- 6-state linear Kalman Filter: [x, y, vx, vy, ax, ay]
- State transition matrix F with constant acceleration model
- Measurement matrix H observing spatial position (x, y)
- Adaptive measurement noise covariance R based on velocity magnitude

Target: 60 FPS (dt = 1/60.0), sub-16ms processing
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


class TestBallFilterInitialization:
    """Test TennisBallKalmanFilter initialization."""

    def test_default_dt_is_60fps(self):
        """Verify default dt matches 60 FPS target."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        assert np.isclose(f.dt, 1.0 / 60.0), f"Expected dt={1/60}, got dt={f.dt}"

    def test_custom_dt(self):
        """Verify custom dt can be set."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter(dt=1.0 / 30.0)
        assert np.isclose(f.dt, 1.0 / 30.0)

    def test_state_vector_is_zero_initial(self):
        """Initial state vector should be all zeros."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        np.testing.assert_array_equal(f.x, np.zeros((6, 1)))

    def test_state_transition_matrix_shape(self):
        """F matrix should be 6x6."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        assert f.F.shape == (6, 6), f"F shape is {f.F.shape}, expected (6, 6)"

    def test_measurement_matrix_shape(self):
        """H matrix should be 2x6 (observes x, y from 6 states)."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        assert f.H.shape == (2, 6), f"H shape is {f.H.shape}, expected (2, 6)"

    def test_h_matrix_observes_only_spatial_coordinates(self):
        """H matrix should only observe x, y positions (first 2 rows)."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        np.testing.assert_array_equal(f.H[0], [1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(f.H[1], [0, 1, 0, 0, 0, 0])

    def test_covariance_matrices_initialized(self):
        """P, Q, R should be properly initialized."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        assert f.P.shape == (6, 6), f"P shape is {f.P.shape}"
        assert f.Q.shape == (6, 6), f"Q shape is {f.Q.shape}"
        assert f.R.shape == (2, 2), f"R shape is {f.R.shape}"


class TestKalmanPredict:
    """Test Kalman Filter predict method."""

    def test_predict_returns_position(self):
        """Predict should return 2-element array of estimated position."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        result = f.predict()
        assert result.shape == (2,), f"Predict returned shape {result.shape}, expected (2,)"
        assert isinstance(result, np.ndarray)

    def test_predict_advances_state(self):
        """Predict should advance the state vector using F matrix."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        f.x = np.array([[0], [0], [10], [5], [0], [0]])  # velocity only
        f.predict()
        assert np.isclose(f.x[0, 0], 10 * (1/60), atol=1e-6), "Position x not updated correctly"
        assert np.isclose(f.x[1, 0], 5 * (1/60), atol=1e-6), "Position y not updated correctly"

    def test_predict_updates_velocity_from_acceleration(self):
        """Constant-accel model: v += a*dt each predict step."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        f.x = np.array([[0], [0], [50], [-30], [10], [-20]])
        f.predict()
        dt = f.dt
        assert np.isclose(f.x[2, 0], 50 + 10 * dt, atol=1e-9), "Velocity vx should integrate acceleration"
        assert np.isclose(f.x[3, 0], -30 + (-20) * dt, atol=1e-9), "Velocity vy should integrate acceleration"

    def test_predict_with_acceleration(self):
        """Verify second-order (acceleration) position update: x += 0.5*a*dt^2."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        f.x = np.array([[0], [0], [0], [0], [20], [-10]])  # acceleration only
        f.predict()
        expected_x = 0.5 * 20 * (1/60) ** 2
        expected_y = 0.5 * (-10) * (1/60) ** 2
        assert np.isclose(f.x[0, 0], expected_x, atol=1e-9)
        assert np.isclose(f.x[1, 0], expected_y, atol=1e-9)
        # Acceleration preserved in constant-accel model
        assert np.isclose(f.x[4, 0], 20, atol=1e-6)
        assert np.isclose(f.x[5, 0], -10, atol=1e-6)

    def test_predict_updates_covariance(self):
        """Predict should update P = F P F^T + Q."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        f.Q = np.eye(6) * 0.5  # reference Q for this assertion
        P_before = f.P.copy()
        f.predict()
        expected = f.F @ P_before @ f.F.T + f.Q
        np.testing.assert_allclose(f.P, expected, rtol=1e-8)


    def test_predict_with_new_dt_rebuilds_F(self):
        """Regression: predict(dt=...) used to discard the rebuilt F."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter(dt=1.0 / 30.0)
        f.x = np.array([[0.0], [0.0], [10.0], [0.0], [0.0], [0.0]])
        f.predict(dt=0.1)
        assert np.isclose(f.x[0, 0], 1.0), "x must advance by v*dt with the NEW dt"
        assert np.isclose(f.F[0, 2], 0.1)


class TestKalmanUpdate:
    """Test Kalman Filter update method."""

    def test_update_with_perfect_measurement(self):
        """Updating with the predicted position should keep position unchanged."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        f.x = np.array([[10], [10], [0], [0], [0], [0]])
        f.P = np.eye(6) * 1e-6  # Very confident in state
        pred = f.predict()
        result = f.update(np.array(pred))  # Observation == prediction
        np.testing.assert_allclose(result, pred, rtol=1e-4, atol=1e-5)

    def test_update_moves_toward_measurement(self):
        """Update should pull the estimate toward the measurement."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        f.x = np.array([[0], [0], [0], [0], [0], [0]])
        f.P = np.eye(6) * 50.0  # High uncertainty
        f.predict()
        result = f.update(np.array([10.0, 5.0]))
        assert result[0] > 0, "x should move toward measurement"
        assert result[1] > 0, "y should move toward measurement"
        assert result[0] < 10, "x should not fully jump to measurement"

    def test_update_returns_2d_position(self):
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        result = f.predict()
        result = f.update(np.array([1.0, 2.0]))
        assert result.shape == (2,)

    def test_update_reduces_position_uncertainty(self):
        """After update, covariance in position components should shrink."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        f.P = np.eye(6) * 10.0
        f.predict()
        P_after_pred = f.P.copy()
        f.update(np.array([1.0, 1.0]))
        assert f.P[0, 0] < P_after_pred[0, 0], "Position variance should decrease"
        assert f.P[1, 1] < P_after_pred[1, 1], "Position variance should decrease"

    def test_update_with_high_uncertainty_tracks_measurement_closely(self):
        """When P is large (distrust state), gain should be high and
        the state should track the measurement tightly."""
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        f.x = np.array([[0], [0], [0], [0], [0], [0]])
        f.P = np.eye(6) * 1e6  # Extremely uncertain
        f.predict()
        result = f.update(np.array([100.0, -50.0]))
        np.testing.assert_allclose(result, [100.0, -50.0], atol=0.5)


class TestKalmanTrackSimulation:
    """End-to-end simulation: filter must track a physically plausible
    parabolic (projectile) ball trajectory."""

    def test_tracks_projectile_trajectory(self):
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        dt = f.dt
        g = 9.81

        x_true, y_true, vx, vy = 0.0, 0.0, 30.0, 20.0
        max_error = 0.0
        for i in range(300):  # 5 seconds of 60fps
            # Integrate ground-truth projectile motion
            x_true += vx * dt
            y_true += vy * dt - 0.5 * g * dt * dt
            vy -= g * dt
            predicted = f.predict()
            obs = np.array([x_true, y_true])
            f.update(obs)
            if i >= 60:  # skip transient after init
                err = np.linalg.norm(f.x[:2, 0] - [x_true, y_true])
                max_error = max(max_error, err)

        # Filter must stay within 1 meter of the true position
        assert max_error < 1.0, f"Max tracking error {max_error:.3f}m exceeds 1.0m"

    def test_velocity_estimate_converges(self):
        from src.stages.ball_tracking.kalman_filter import TennisBallKalmanFilter

        f = TennisBallKalmanFilter()
        dt = f.dt
        g = 9.81

        x_true, y_true, vx, vy = 0.0, 0.0, 25.0, 15.0
        for i in range(600):  # 10 seconds
            x_true += vx * dt
            y_true += vy * dt
            vy -= g * dt
            f.predict()
            f.update(np.array([x_true, y_true]))

        assert abs(f.x[2, 0] - vx) < 1.0, f"vx estimate {f.x[2,0]:.2f}, true {vx}"
        assert abs(f.x[3, 0] - vy) < 1.0, f"vy estimate {f.x[3,0]:.2f}, true {vy:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
