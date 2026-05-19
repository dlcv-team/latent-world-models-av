"""Unit tests for evaluation metrics."""

import numpy as np
import pytest
from evaluation.metrics import compute_rmse


def test_compute_rmse_perfect_predictions():
    """Test RMSE with perfect predictions."""
    # Create identical predictions and targets
    predictions = np.array([[0.5, -0.3], [0.2, 0.1], [-0.1, 0.8]])
    targets = predictions.copy()

    steer_rmse, accel_rmse = compute_rmse(predictions, targets)

    assert steer_rmse == 0.0
    assert accel_rmse == 0.0


def test_compute_rmse_known_error():
    """Test RMSE with known error values in normalized space."""
    # Normalized values: steering off by 0.1, accel off by 0.2
    predictions = np.array([[0.0, 0.0], [0.0, 0.0]])
    targets = np.array([[0.1, 0.2], [-0.1, -0.2]])

    steer_rmse, accel_rmse = compute_rmse(predictions, targets)

    # Expected: normalized RMSE (use convert_steer_rmse_to_deg for physical units)
    assert np.isclose(steer_rmse, 0.1, atol=1e-6)
    assert np.isclose(accel_rmse, 0.2, atol=1e-6)


def test_compute_rmse_shape_validation():
    """Test that function handles different batch sizes."""
    for n in [1, 10, 100, 1000]:
        predictions = np.random.randn(n, 2) * 0.5
        targets = np.random.randn(n, 2) * 0.5

        steer_rmse, accel_rmse = compute_rmse(predictions, targets)

        assert steer_rmse >= 0
        assert accel_rmse >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
