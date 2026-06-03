"""Unit tests for evaluation metrics.

Verifies RMSE computation and normalization conversions.
"""

import numpy as np
import pytest

from evaluation.metrics import (
    compute_rmse,
    convert_steer_rmse_to_deg,
    classify_scenes_by_scenario,
)


def test_compute_rmse_normalized():
    """Test that compute_rmse works in normalized space."""
    # Create simple test data in normalized space
    predictions = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ])
    targets = np.array([
        [0.15, 0.25],
        [0.25, 0.35],
        [0.45, 0.55],
    ])

    steer_rmse, accel_rmse = compute_rmse(predictions, targets)

    # Manual calculation: sqrt(mean((0.1-0.15)^2 + (0.3-0.25)^2 + (0.5-0.45)^2))
    expected_steer = np.sqrt(np.mean([0.05**2, 0.05**2, 0.05**2]))
    expected_accel = np.sqrt(np.mean([0.05**2, 0.05**2, 0.05**2]))

    assert np.isclose(steer_rmse, expected_steer, atol=1e-6)
    assert np.isclose(accel_rmse, expected_accel, atol=1e-6)


def test_convert_steer_rmse_to_deg():
    """Test conversion from normalized RMSE to degrees.

    The canonical config sets eval_back_to_deg_factor = 34.37746770784939
    (which is 6 * 180 / pi).
    """
    # Test with explicit config dict
    cfg = {
        "normalization": {
            "steering": {
                "eval_back_to_deg_factor": 34.37746770784939
            }
        }
    }

    # Test conversion: 0.1 normalized → ~3.4377 degrees
    rmse_norm = 0.1
    rmse_deg = convert_steer_rmse_to_deg(rmse_norm, cfg=cfg)

    expected = 0.1 * 34.37746770784939
    assert np.isclose(rmse_deg, expected, atol=1e-3)
    assert np.isclose(rmse_deg, 3.4377, atol=1e-3)


def test_convert_steer_rmse_to_deg_with_canonical():
    """Test conversion using canonical config loaded from file."""
    # This test requires canonical.yaml to exist
    try:
        rmse_deg = convert_steer_rmse_to_deg(0.1, cfg=None)
        # Should be approximately 3.4377 degrees
        assert np.isclose(rmse_deg, 3.4377, atol=1e-3)
    except FileNotFoundError:
        pytest.skip("Canonical config not available in test environment")


def test_compute_rmse_shape_validation():
    """Test that compute_rmse handles correct shapes."""
    # Valid shapes
    predictions = np.random.randn(100, 2)
    targets = np.random.randn(100, 2)

    steer_rmse, accel_rmse = compute_rmse(predictions, targets)

    assert isinstance(steer_rmse, float)
    assert isinstance(accel_rmse, float)
    assert steer_rmse >= 0
    assert accel_rmse >= 0


def test_compute_rmse_perfect_prediction():
    """Test RMSE is zero when predictions match targets exactly."""
    predictions = np.array([[0.5, 0.3], [0.2, 0.1]])
    targets = predictions.copy()

    steer_rmse, accel_rmse = compute_rmse(predictions, targets)

    assert np.isclose(steer_rmse, 0.0, atol=1e-10)
    assert np.isclose(accel_rmse, 0.0, atol=1e-10)


def test_compute_per_scenario_rmse_with_conversion():
    """Integration test: compute_per_scenario_rmse returns normalized RMSE,
    then convert_steer_rmse_to_deg converts it to degrees.

    This verifies the full workflow of computing normalized RMSE and explicitly
    converting to degrees using the config factor.
    """
    import pandas as pd
    from evaluation.metrics import compute_per_scenario_rmse

    # Create test data in normalized space
    predictions_df = pd.DataFrame({
        "encoder": ["test_encoder"] * 6,
        "scene_name": ["scene1", "scene1", "scene1", "scene2", "scene2", "scene2"],
        "steer_pred": [0.1, 0.2, 0.15, 0.3, 0.25, 0.28],
        "accel_pred": [0.5, 0.6, 0.55, 0.7, 0.65, 0.68],
        "steer_true": [0.15, 0.25, 0.20, 0.25, 0.30, 0.23],
        "accel_true": [0.55, 0.65, 0.60, 0.65, 0.70, 0.63],
    })

    scene_to_bucket = {
        "scene1": "urban",
        "scene2": "highway",
    }

    cfg = {
        "evaluation": {
            "bootstrap": {
                "n_resamples": 100,  # Small for fast test
                "seed": 42,
                "confidence_level": 0.95,
            }
        },
        "normalization": {
            "steering": {
                "eval_back_to_deg_factor": 34.37746770784939
            }
        }
    }

    # Compute per-scenario RMSE (in normalized space)
    results_df = compute_per_scenario_rmse(predictions_df, scene_to_bucket, cfg)

    # Verify results are in normalized space
    assert "steer_rmse" in results_df["metric"].values
    assert "accel_rmse" in results_df["metric"].values

    # Extract a steering RMSE value (normalized)
    steer_row = results_df[results_df["metric"] == "steer_rmse"].iloc[0]
    steer_rmse_norm = steer_row["mean"]

    # Test conversion to degrees
    steer_rmse_deg = convert_steer_rmse_to_deg(steer_rmse_norm, cfg=cfg)

    # Verify manual conversion matches
    expected_deg = steer_rmse_norm * cfg["normalization"]["steering"]["eval_back_to_deg_factor"]
    assert np.isclose(steer_rmse_deg, expected_deg, atol=1e-6)


def test_classify_scenes_by_scenario():
    """Test scene classification into scenario buckets."""
    # Mock NuScenes object
    class MockNuScenes:
        def __init__(self):
            self.scene = [
                {"token": "scene1", "description": "Driving on a highway at sunset"},
                {"token": "scene2", "description": "Navigating an urban intersection during rush hour"},
                {"token": "scene3", "description": "City street with light traffic"},
                {"token": "scene4", "description": "Approaching a junction with pedestrians"},
                {"token": "scene5", "description": ""},  # Empty description
            ]

    nusc = MockNuScenes()
    scene_tokens = ["scene1", "scene2", "scene3", "scene4", "scene5", "scene6"]  # scene6 doesn't exist

    scene_to_bucket = classify_scenes_by_scenario(nusc, scene_tokens)

    # Verify classifications
    assert scene_to_bucket["scene1"] == "highway"
    assert scene_to_bucket["scene2"] == "intersection"
    assert scene_to_bucket["scene3"] == "urban"
    assert scene_to_bucket["scene4"] == "intersection"  # Contains "junction"
    assert scene_to_bucket["scene5"] == "other"  # Empty description
    assert scene_to_bucket["scene6"] == "other"  # Doesn't exist in nusc.scene


def test_classify_scenes_by_scenario_empty_input():
    """Test classify_scenes_by_scenario with empty input."""
    class MockNuScenes:
        def __init__(self):
            self.scene = []

    nusc = MockNuScenes()
    scene_tokens = []

    scene_to_bucket = classify_scenes_by_scenario(nusc, scene_tokens)

    assert scene_to_bucket == {}


def test_classify_scenes_by_scenario_deduplication():
    """Test that classify_scenes_by_scenario deduplicates scene tokens."""
    class MockNuScenes:
        def __init__(self):
            self.scene = [
                {"token": "scene1", "description": "Highway driving"},
            ]

    nusc = MockNuScenes()
    scene_tokens = ["scene1", "scene1", "scene1"]  # Duplicates

    scene_to_bucket = classify_scenes_by_scenario(nusc, scene_tokens)

    assert len(scene_to_bucket) == 1
    assert scene_to_bucket["scene1"] == "highway"
