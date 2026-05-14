"""Unit tests for evaluation metrics."""

import numpy as np
import pytest
from evaluation.metrics import compute_rmse, scenario_breakdown


def test_compute_rmse_perfect_predictions():
    """Test RMSE with perfect predictions."""
    # Create identical predictions and targets
    predictions = np.array([[0.5, -0.3], [0.2, 0.1], [-0.1, 0.8]])
    targets = predictions.copy()

    steer_rmse, accel_rmse = compute_rmse(predictions, targets)

    assert steer_rmse == 0.0
    assert accel_rmse == 0.0


def test_compute_rmse_known_error():
    """Test RMSE with known error values."""
    # Normalized values: steering off by 0.1, accel off by 0.2
    predictions = np.array([[0.0, 0.0], [0.0, 0.0]])
    targets = np.array([[0.1, 0.2], [-0.1, -0.2]])

    steer_rmse, accel_rmse = compute_rmse(predictions, targets)

    # Expected: steer error = 0.1 * 6.0 = 0.6 degrees
    # Expected: accel error = 0.2 * 10.0 = 2.0 m/s^2
    assert np.isclose(steer_rmse, 0.6, atol=1e-6)
    assert np.isclose(accel_rmse, 2.0, atol=1e-6)


def test_compute_rmse_shape_validation():
    """Test that function handles different batch sizes."""
    for n in [1, 10, 100, 1000]:
        predictions = np.random.randn(n, 2) * 0.5
        targets = np.random.randn(n, 2) * 0.5

        steer_rmse, accel_rmse = compute_rmse(predictions, targets)

        assert steer_rmse >= 0
        assert accel_rmse >= 0


def test_scenario_breakdown_mock():
    """Test scenario breakdown with mock NuScenes data."""
    # Create mock nuScenes object
    class MockNuScenes:
        def __init__(self):
            self.scene = [
                {'token': 'scene1', 'description': 'Driving on highway with light traffic'},
                {'token': 'scene2', 'description': 'Urban street in downtown Boston'},
                {'token': 'scene3', 'description': 'Making left turn at intersection'},
            ]

    nusc = MockNuScenes()

    # Create test data: 6 frames (2 from each scene)
    scene_tokens = ['scene1', 'scene1', 'scene2', 'scene2', 'scene3', 'scene3']
    rmse_by_frame = np.array([
        [1.0, 0.5],  # highway
        [1.2, 0.6],  # highway
        [2.0, 1.0],  # urban
        [2.2, 1.2],  # urban
        [3.0, 1.5],  # intersection
        [3.2, 1.7],  # intersection
    ])

    results = scenario_breakdown(nusc, scene_tokens, rmse_by_frame)

    # Check all categories present
    assert 'highway' in results
    assert 'urban' in results
    assert 'intersection' in results

    # Check highway stats
    assert results['highway']['count'] == 2
    assert np.isclose(results['highway']['steer_rmse'], 1.1)
    assert np.isclose(results['highway']['accel_rmse'], 0.55)

    # Check urban stats
    assert results['urban']['count'] == 2
    assert np.isclose(results['urban']['steer_rmse'], 2.1)
    assert np.isclose(results['urban']['accel_rmse'], 1.1)

    # Check intersection stats
    assert results['intersection']['count'] == 2
    assert np.isclose(results['intersection']['steer_rmse'], 3.1)
    assert np.isclose(results['intersection']['accel_rmse'], 1.6)


def test_scenario_breakdown_keywords():
    """Test that keyword matching works correctly."""
    class MockNuScenes:
        def __init__(self):
            self.scene = [
                {'token': 's1', 'description': 'freeway'},
                {'token': 's2', 'description': 'expressway traffic'},
                {'token': 's3', 'description': 'motorway'},
                {'token': 's4', 'description': 'at the junction'},
                {'token': 's5', 'description': 'making a turn'},
                {'token': 's6', 'description': 'residential street'},
                {'token': 's7', 'description': 'city center'},
            ]

    nusc = MockNuScenes()
    scene_tokens = ['s1', 's2', 's3', 's4', 's5', 's6', 's7']
    rmse_by_frame = np.ones((7, 2))

    results = scenario_breakdown(nusc, scene_tokens, rmse_by_frame)

    # s1, s2, s3 should be highway
    assert results['highway']['count'] == 3

    # s4, s5 should be intersection
    assert results['intersection']['count'] == 2

    # s6, s7 should be urban
    assert results['urban']['count'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
