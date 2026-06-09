"""Unit tests for evaluation metrics.

Verifies RMSE computation and normalization conversions.
"""

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import (
    compute_rmse,
    convert_steer_rmse_to_deg,
    classify_scenes_by_scenario,
    classify_scenes_by_environment,
    bootstrap_ratio_ci,
    compute_robustness_ratios,
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


def test_classify_scenes_by_environment_basic():
    """Test basic scene classification by environment with overlaps."""
    scene_names = ["scene-0001", "scene-0002", "scene-0003", "scene-0004"]
    night_scenes = {"scene-0001", "scene-0002"}  # scenes 1,2 are night
    rain_scenes = {"scene-0002", "scene-0003"}   # scenes 2,3 are rain

    result = classify_scenes_by_environment(scene_names, night_scenes, rain_scenes)

    # Verify independent subsets with overlap
    assert sorted(result["night"]) == ["scene-0001", "scene-0002"]
    assert sorted(result["rain"]) == ["scene-0002", "scene-0003"]
    assert sorted(result["day_clear"]) == ["scene-0004"]


def test_classify_scenes_by_environment_no_overlap():
    """Test environment classification with no overlap (disjoint sets)."""
    scene_names = ["scene-0001", "scene-0002", "scene-0003"]
    night_scenes = {"scene-0001"}
    rain_scenes = {"scene-0002"}

    result = classify_scenes_by_environment(scene_names, night_scenes, rain_scenes)

    assert result["night"] == ["scene-0001"]
    assert result["rain"] == ["scene-0002"]
    assert result["day_clear"] == ["scene-0003"]


def test_classify_scenes_by_environment_all_day_clear():
    """Test that empty night/rain sets produce all day_clear."""
    scene_names = ["scene-0001", "scene-0002", "scene-0003"]
    night_scenes = set()  # Empty
    rain_scenes = set()   # Empty

    result = classify_scenes_by_environment(scene_names, night_scenes, rain_scenes)

    assert result["night"] == []
    assert result["rain"] == []
    assert sorted(result["day_clear"]) == ["scene-0001", "scene-0002", "scene-0003"]


def test_classify_scenes_by_environment_format_mismatch_night():
    """Test that scene-name format mismatch in night_scenes raises ValueError."""
    # CSV uses "scene-0001" but YAML uses "scene_0001" (underscore vs hyphen)
    scene_names = ["scene-0001", "scene-0002", "scene-0003"]
    night_scenes = {"scene_0001", "scene_0002"}  # Wrong format (underscore)
    rain_scenes = set()

    with pytest.raises(ValueError, match="Scene-name format mismatch.*night scenes"):
        classify_scenes_by_environment(scene_names, night_scenes, rain_scenes)


def test_classify_scenes_by_environment_format_mismatch_rain():
    """Test that scene-name format mismatch in rain_scenes raises ValueError."""
    scene_names = ["scene-0001", "scene-0002", "scene-0003"]
    night_scenes = set()
    rain_scenes = {"SCENE-0001", "SCENE-0002"}  # Wrong format (uppercase)

    with pytest.raises(ValueError, match="Scene-name format mismatch.*rain scenes"):
        classify_scenes_by_environment(scene_names, night_scenes, rain_scenes)


def test_classify_scenes_by_environment_partial_mismatch_night():
    """Test that partial missing night scenes raises ValueError with helpful message."""
    scene_names = ["scene-0001", "scene-0002", "scene-0003"]
    night_scenes = {"scene-0001", "scene-0004", "scene-0005"}  # 0004, 0005 missing
    rain_scenes = set()

    with pytest.raises(ValueError, match="Incomplete probe data.*night scenes missing"):
        classify_scenes_by_environment(scene_names, night_scenes, rain_scenes)


def test_classify_scenes_by_environment_partial_mismatch_rain():
    """Test that partial missing rain scenes raises ValueError."""
    scene_names = ["scene-0001", "scene-0002"]
    night_scenes = set()
    rain_scenes = {"scene-0001", "scene-0002", "scene-0999"}  # scene-0999 missing

    with pytest.raises(ValueError, match="Incomplete probe data.*rain scenes missing"):
        classify_scenes_by_environment(scene_names, night_scenes, rain_scenes)


def test_bootstrap_ratio_ci_basic():
    """Test basic bootstrap ratio CI computation."""
    numerator = np.array([2.0, 4.0, 6.0])
    denominator = np.array([1.0, 1.0, 1.0])

    ratio, ci_lo, ci_hi = bootstrap_ratio_ci(
        numerator,
        denominator,
        n_resamples=1000,
        seed=42,
        confidence_level=0.95,
    )

    # Expected ratio: mean([2, 4, 6]) / mean([1, 1, 1]) = 4.0 / 1.0 = 4.0
    assert np.isclose(ratio, 4.0, atol=1e-6)

    # CI should bracket the ratio
    assert ci_lo < ratio < ci_hi

    # CI should be reasonable width (not degenerate)
    assert ci_hi - ci_lo > 0


def test_bootstrap_ratio_ci_reproducibility():
    """Test that same seed produces same results."""
    numerator = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    denominator = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

    ratio1, ci_lo1, ci_hi1 = bootstrap_ratio_ci(
        numerator, denominator, n_resamples=500, seed=123, confidence_level=0.95
    )
    ratio2, ci_lo2, ci_hi2 = bootstrap_ratio_ci(
        numerator, denominator, n_resamples=500, seed=123, confidence_level=0.95
    )

    # Same seed → identical results
    assert ratio1 == ratio2
    assert ci_lo1 == ci_lo2
    assert ci_hi1 == ci_hi2


def test_compute_robustness_ratios_happy_path():
    """Test robustness ratio computation with valid data."""
    from evaluation.metrics import compute_robustness_ratios

    # Create mock per-scene RMSE data (normalized [-1, 1] space)
    # Add variance so CIs are non-degenerate
    # Night RMSE ~0.20, Rain RMSE ~0.15, Day RMSE ~0.10
    # Expected ratios: night/day ~2.0, rain/day ~1.5
    encoder_df = pd.DataFrame({
        "scene_name": ["scene-N1", "scene-N2", "scene-N3", "scene-R1", "scene-R2", "scene-R3", "scene-D1", "scene-D2", "scene-D3"],
        "steer_rmse": [0.18, 0.20, 0.22, 0.14, 0.15, 0.16, 0.09, 0.10, 0.11],
        "accel_rmse": [0.38, 0.40, 0.42, 0.28, 0.30, 0.32, 0.18, 0.20, 0.22],
    })

    env_subsets = {
        "night": ["scene-N1", "scene-N2", "scene-N3"],
        "rain": ["scene-R1", "scene-R2", "scene-R3"],
        "day_clear": ["scene-D1", "scene-D2", "scene-D3"],
    }

    results = compute_robustness_ratios(
        encoder_df=encoder_df,
        env_subsets=env_subsets,
        steer_denorm_factor=6.0 * 180.0 / np.pi,  # Canonical: rad → deg
        accel_denorm_factor=10.0,  # Canonical: normalized → m/s²
        n_resamples=100,
        bootstrap_seed=42,
        confidence_level=0.95,
    )

    # Should return 2 rows (one per metric)
    assert len(results) == 2

    # Check steering metric
    steer_result = [r for r in results if r["metric"] == "steer_rmse_deg"][0]
    assert np.isclose(steer_result["ratio_night_day"], 2.0, atol=0.2)
    assert np.isclose(steer_result["ratio_rain_day"], 1.5, atol=0.2)
    assert steer_result["n_night"] == 3
    assert steer_result["n_rain"] == 3
    assert steer_result["n_day_clear"] == 3

    # CIs should bracket the ratio (or be equal for degenerate case)
    assert steer_result["ci_lo_night_day"] <= steer_result["ratio_night_day"] <= steer_result["ci_hi_night_day"]
    assert steer_result["ci_lo_rain_day"] <= steer_result["ratio_rain_day"] <= steer_result["ci_hi_rain_day"]

    # Check acceleration metric
    accel_result = [r for r in results if r["metric"] == "accel_rmse_mps2"][0]
    assert np.isclose(accel_result["ratio_night_day"], 2.0, atol=0.2)
    assert np.isclose(accel_result["ratio_rain_day"], 1.5, atol=0.2)


def test_compute_robustness_ratios_zero_baseline():
    """Test that zero day_clear scenes results in no output rows."""
    from evaluation.metrics import compute_robustness_ratios

    # Only night and rain scenes, no day_clear baseline
    encoder_df = pd.DataFrame({
        "scene_name": ["scene-N1", "scene-R1"],
        "steer_rmse": [0.20, 0.15],
        "accel_rmse": [0.40, 0.30],
    })

    env_subsets = {
        "night": ["scene-N1"],
        "rain": ["scene-R1"],
        "day_clear": [],  # Empty baseline
    }

    results = compute_robustness_ratios(
        encoder_df=encoder_df,
        env_subsets=env_subsets,
        steer_denorm_factor=343.77,
        accel_denorm_factor=10.0,
        n_resamples=100,
        bootstrap_seed=42,
        confidence_level=0.95,
    )

    # No baseline → no ratios can be computed
    assert len(results) == 0


def test_compute_robustness_ratios_missing_night():
    """Test that missing night scenes results in None for night ratios."""
    from evaluation.metrics import compute_robustness_ratios

    # Only rain and day_clear, no night
    encoder_df = pd.DataFrame({
        "scene_name": ["scene-R1", "scene-D1"],
        "steer_rmse": [0.15, 0.10],
        "accel_rmse": [0.30, 0.20],
    })

    env_subsets = {
        "night": [],  # No night scenes
        "rain": ["scene-R1"],
        "day_clear": ["scene-D1"],
    }

    results = compute_robustness_ratios(
        encoder_df=encoder_df,
        env_subsets=env_subsets,
        steer_denorm_factor=343.77,
        accel_denorm_factor=10.0,
        n_resamples=100,
        bootstrap_seed=42,
        confidence_level=0.95,
    )

    # Should return 2 rows (one per metric) but night ratios should be None
    assert len(results) == 2

    for result in results:
        assert result["ratio_night_day"] is None
        assert result["ci_lo_night_day"] is None
        assert result["ci_hi_night_day"] is None
        assert result["ratio_rain_day"] is not None  # Rain ratio should be valid
        assert result["n_night"] == 0
        assert result["n_rain"] == 1


def test_compute_robustness_ratios_missing_rain():
    """Test that missing rain scenes results in None for rain ratios."""
    from evaluation.metrics import compute_robustness_ratios

    # Only night and day_clear, no rain
    encoder_df = pd.DataFrame({
        "scene_name": ["scene-N1", "scene-D1"],
        "steer_rmse": [0.20, 0.10],
        "accel_rmse": [0.40, 0.20],
    })

    env_subsets = {
        "night": ["scene-N1"],
        "rain": [],  # No rain scenes
        "day_clear": ["scene-D1"],
    }

    results = compute_robustness_ratios(
        encoder_df=encoder_df,
        env_subsets=env_subsets,
        steer_denorm_factor=343.77,
        accel_denorm_factor=10.0,
        n_resamples=100,
        bootstrap_seed=42,
        confidence_level=0.95,
    )

    # Should return 2 rows (one per metric) but rain ratios should be None
    assert len(results) == 2

    for result in results:
        assert result["ratio_night_day"] is not None  # Night ratio should be valid
        assert result["ratio_rain_day"] is None
        assert result["ci_lo_rain_day"] is None
        assert result["ci_hi_rain_day"] is None
        assert result["n_rain"] == 0
        assert result["n_night"] == 1


def test_bootstrap_ratio_ci_single_value():
    """Test that single-value inputs return degenerate CI."""
    numerator = np.array([5.0])
    denominator = np.array([2.0])

    ratio, ci_lo, ci_hi = bootstrap_ratio_ci(
        numerator, denominator, n_resamples=100, seed=42, confidence_level=0.95
    )

    # Expected ratio: 5.0 / 2.0 = 2.5
    assert np.isclose(ratio, 2.5, atol=1e-6)

    # CI should be degenerate (all equal to ratio)
    assert ci_lo == ratio
    assert ci_hi == ratio


def test_bootstrap_ratio_ci_empty_numerator():
    """Test that empty numerator array raises ValueError."""
    numerator = np.array([])
    denominator = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="bootstrap_ratio_ci received zero-length input"):
        bootstrap_ratio_ci(numerator, denominator, n_resamples=100, seed=42, confidence_level=0.95)


def test_bootstrap_ratio_ci_empty_denominator():
    """Test that empty denominator array raises ValueError."""
    numerator = np.array([1.0, 2.0, 3.0])
    denominator = np.array([])

    with pytest.raises(ValueError, match="bootstrap_ratio_ci received zero-length input"):
        bootstrap_ratio_ci(numerator, denominator, n_resamples=100, seed=42, confidence_level=0.95)


def test_bootstrap_ratio_ci_different_lengths():
    """Test that different-length arrays work (independent resampling)."""
    numerator = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # n=6
    denominator = np.array([1.0, 1.0, 1.0])  # n=3

    ratio, ci_lo, ci_hi = bootstrap_ratio_ci(
        numerator, denominator, n_resamples=1000, seed=42, confidence_level=0.95
    )

    # Expected ratio: mean([1,2,3,4,5,6]) / mean([1,1,1]) = 3.5 / 1.0 = 3.5
    assert np.isclose(ratio, 3.5, atol=1e-6)
    assert ci_lo < ratio < ci_hi


def test_bootstrap_ratio_ci_small_sample():
    """Test bootstrap with very small sample (n=2) produces wide CI."""
    numerator = np.array([10.0, 20.0])
    denominator = np.array([5.0, 5.0])

    ratio, ci_lo, ci_hi = bootstrap_ratio_ci(
        numerator, denominator, n_resamples=1000, seed=42, confidence_level=0.95
    )

    # Expected ratio: 15.0 / 5.0 = 3.0
    assert np.isclose(ratio, 3.0, atol=1e-6)

    # Small sample → wide CI (variance is high due to resampling [10,10], [10,20], [20,20])
    assert ci_lo < ratio < ci_hi
    ci_width = ci_hi - ci_lo
    assert ci_width > 0.5  # Expect reasonably wide CI for n=2


# ========================================================================
# compute_robustness_ratios tests
# ========================================================================


def test_compute_robustness_ratios_basic():
    """Test basic ratio computation with all environments present."""
    import pandas as pd

    # Create synthetic per-scene RMSE data
    # Night: 3 scenes, Rain: 2 scenes, Day_clear: 5 scenes
    encoder_df = pd.DataFrame({
        "scene_name": [
            "night-01", "night-02", "night-03",  # night scenes
            "rain-01", "rain-02",                  # rain scenes
            "day-01", "day-02", "day-03", "day-04", "day-05",  # day_clear scenes
        ],
        "steer_rmse": [
            0.20, 0.22, 0.18,  # night (normalized)
            0.15, 0.13,        # rain (normalized)
            0.10, 0.12, 0.11, 0.09, 0.10,  # day_clear (normalized)
        ],
        "accel_rmse": [
            0.15, 0.17, 0.16,  # night (normalized)
            0.12, 0.10,        # rain (normalized)
            0.08, 0.09, 0.08, 0.07, 0.08,  # day_clear (normalized)
        ],
    })

    env_subsets = {
        "night": ["night-01", "night-02", "night-03"],
        "rain": ["rain-01", "rain-02"],
        "day_clear": ["day-01", "day-02", "day-03", "day-04", "day-05"],
    }

    # Standard denorm factors
    steer_denorm_factor = 34.37746770784939
    accel_denorm_factor = 9.81

    results = compute_robustness_ratios(
        encoder_df=encoder_df,
        env_subsets=env_subsets,
        steer_denorm_factor=steer_denorm_factor,
        accel_denorm_factor=accel_denorm_factor,
        n_resamples=1000,
        bootstrap_seed=42,
        confidence_level=0.95,
    )

    # Should return 2 rows (steer, accel)
    assert len(results) == 2

    # Check first row (steer_rmse_deg)
    steer_row = results[0]
    assert steer_row["metric"] == "steer_rmse_deg"
    assert steer_row["n_night"] == 3
    assert steer_row["n_rain"] == 2
    assert steer_row["n_day_clear"] == 5

    # Verify denormalized means
    expected_night_steer = np.mean([0.20, 0.22, 0.18]) * steer_denorm_factor
    expected_day_steer = np.mean([0.10, 0.12, 0.11, 0.09, 0.10]) * steer_denorm_factor
    assert np.isclose(steer_row["rmse_night"], expected_night_steer, atol=1e-6)
    assert np.isclose(steer_row["rmse_day_clear"], expected_day_steer, atol=1e-6)

    # Verify ratios exist and CIs bracket ratios
    assert steer_row["ratio_night_day"] is not None
    assert steer_row["ratio_rain_day"] is not None
    assert steer_row["ci_lo_night_day"] < steer_row["ratio_night_day"] < steer_row["ci_hi_night_day"]
    assert steer_row["ci_lo_rain_day"] < steer_row["ratio_rain_day"] < steer_row["ci_hi_rain_day"]

    # Ratios should be > 1.0 (night/rain worse than day_clear)
    assert steer_row["ratio_night_day"] > 1.0
    assert steer_row["ratio_rain_day"] > 1.0

    # Check second row (accel_rmse_mps2)
    accel_row = results[1]
    assert accel_row["metric"] == "accel_rmse_mps2"
    assert accel_row["ratio_night_day"] is not None
    assert accel_row["ratio_rain_day"] is not None


def test_compute_robustness_ratios_no_baseline():
    """Test skipping metrics when baseline is empty."""
    import pandas as pd

    # Create encoder_df with only night/rain scenes, no day_clear
    encoder_df = pd.DataFrame({
        "scene_name": ["night-01", "night-02", "rain-01"],
        "steer_rmse": [0.20, 0.22, 0.15],
        "accel_rmse": [0.15, 0.17, 0.12],
    })

    env_subsets = {
        "night": ["night-01", "night-02"],
        "rain": ["rain-01"],
        "day_clear": [],  # Empty baseline
    }

    results = compute_robustness_ratios(
        encoder_df=encoder_df,
        env_subsets=env_subsets,
        steer_denorm_factor=34.37746770784939,
        accel_denorm_factor=9.81,
        n_resamples=1000,
        bootstrap_seed=42,
        confidence_level=0.95,
    )

    # Should return empty list (both metrics skipped)
    assert len(results) == 0


def test_compute_robustness_ratios_no_night():
    """Test handling of missing night scenes."""
    import pandas as pd

    encoder_df = pd.DataFrame({
        "scene_name": ["rain-01", "rain-02", "day-01", "day-02", "day-03"],
        "steer_rmse": [0.15, 0.13, 0.10, 0.12, 0.11],
        "accel_rmse": [0.12, 0.10, 0.08, 0.09, 0.08],
    })

    env_subsets = {
        "night": [],  # No night scenes
        "rain": ["rain-01", "rain-02"],
        "day_clear": ["day-01", "day-02", "day-03"],
    }

    results = compute_robustness_ratios(
        encoder_df=encoder_df,
        env_subsets=env_subsets,
        steer_denorm_factor=34.37746770784939,
        accel_denorm_factor=9.81,
        n_resamples=1000,
        bootstrap_seed=42,
        confidence_level=0.95,
    )

    # Should return 2 rows, but night ratio columns are None
    assert len(results) == 2

    steer_row = results[0]
    assert steer_row["n_night"] == 0
    assert steer_row["rmse_night"] is None
    assert steer_row["ratio_night_day"] is None
    assert steer_row["ci_lo_night_day"] is None
    assert steer_row["ci_hi_night_day"] is None

    # Rain ratio should be computed
    assert steer_row["n_rain"] == 2
    assert steer_row["ratio_rain_day"] is not None
    assert steer_row["ci_lo_rain_day"] is not None
    assert steer_row["ci_hi_rain_day"] is not None


def test_compute_robustness_ratios_no_rain():
    """Test handling of missing rain scenes."""
    import pandas as pd

    encoder_df = pd.DataFrame({
        "scene_name": ["night-01", "night-02", "night-03", "day-01", "day-02"],
        "steer_rmse": [0.20, 0.22, 0.18, 0.10, 0.12],
        "accel_rmse": [0.15, 0.17, 0.16, 0.08, 0.09],
    })

    env_subsets = {
        "night": ["night-01", "night-02", "night-03"],
        "rain": [],  # No rain scenes
        "day_clear": ["day-01", "day-02"],
    }

    results = compute_robustness_ratios(
        encoder_df=encoder_df,
        env_subsets=env_subsets,
        steer_denorm_factor=34.37746770784939,
        accel_denorm_factor=9.81,
        n_resamples=1000,
        bootstrap_seed=42,
        confidence_level=0.95,
    )

    # Should return 2 rows, but rain ratio columns are None
    assert len(results) == 2

    steer_row = results[0]
    assert steer_row["n_rain"] == 0
    assert steer_row["rmse_rain"] is None
    assert steer_row["ratio_rain_day"] is None
    assert steer_row["ci_lo_rain_day"] is None
    assert steer_row["ci_hi_rain_day"] is None

    # Night ratio should be computed
    assert steer_row["n_night"] == 3
    assert steer_row["ratio_night_day"] is not None


def test_compute_robustness_ratios_single_scene():
    """Test single-scene subsets (CI = point estimate)."""
    import pandas as pd

    encoder_df = pd.DataFrame({
        "scene_name": ["night-01", "rain-01", "day-01"],
        "steer_rmse": [0.20, 0.15, 0.10],
        "accel_rmse": [0.15, 0.12, 0.08],
    })

    env_subsets = {
        "night": ["night-01"],
        "rain": ["rain-01"],
        "day_clear": ["day-01"],
    }

    results = compute_robustness_ratios(
        encoder_df=encoder_df,
        env_subsets=env_subsets,
        steer_denorm_factor=34.37746770784939,
        accel_denorm_factor=9.81,
        n_resamples=1000,
        bootstrap_seed=42,
        confidence_level=0.95,
    )

    assert len(results) == 2

    steer_row = results[0]
    assert steer_row["n_night"] == 1
    assert steer_row["n_rain"] == 1
    assert steer_row["n_day_clear"] == 1

    # Single-element arrays → CI bounds = point estimate
    assert steer_row["ratio_night_day"] == steer_row["ci_lo_night_day"]
    assert steer_row["ratio_night_day"] == steer_row["ci_hi_night_day"]
    assert steer_row["ratio_rain_day"] == steer_row["ci_lo_rain_day"]
    assert steer_row["ratio_rain_day"] == steer_row["ci_hi_rain_day"]


def test_compute_robustness_ratios_reproducibility():
    """Test that same seed yields identical results."""
    import pandas as pd

    # Use more scenes with higher variance to ensure non-degenerate CIs
    encoder_df = pd.DataFrame({
        "scene_name": [
            "night-01", "night-02", "night-03", "night-04",
            "rain-01", "rain-02",
            "day-01", "day-02", "day-03", "day-04", "day-05",
        ],
        "steer_rmse": [
            0.18, 0.24, 0.16, 0.28,  # night (higher variance)
            0.15, 0.13,              # rain
            0.10, 0.12, 0.11, 0.09, 0.10,  # day_clear
        ],
        "accel_rmse": [
            0.14, 0.19, 0.13, 0.20,  # night (higher variance)
            0.12, 0.10,              # rain
            0.08, 0.09, 0.08, 0.07, 0.08,  # day_clear
        ],
    })

    env_subsets = {
        "night": ["night-01", "night-02", "night-03", "night-04"],
        "rain": ["rain-01", "rain-02"],
        "day_clear": ["day-01", "day-02", "day-03", "day-04", "day-05"],
    }

    results1 = compute_robustness_ratios(
        encoder_df=encoder_df,
        env_subsets=env_subsets,
        steer_denorm_factor=34.37746770784939,
        accel_denorm_factor=9.81,
        n_resamples=1000,
        bootstrap_seed=42,
        confidence_level=0.95,
    )

    results2 = compute_robustness_ratios(
        encoder_df=encoder_df,
        env_subsets=env_subsets,
        steer_denorm_factor=34.37746770784939,
        accel_denorm_factor=9.81,
        n_resamples=1000,
        bootstrap_seed=42,
        confidence_level=0.95,
    )

    # Same seed → identical results
    assert len(results1) == len(results2)
    for row1, row2 in zip(results1, results2):
        assert row1["ratio_night_day"] == row2["ratio_night_day"]
        assert row1["ci_lo_night_day"] == row2["ci_lo_night_day"]
        assert row1["ci_hi_night_day"] == row2["ci_hi_night_day"]
        assert row1["ratio_rain_day"] == row2["ratio_rain_day"]

    # Different seed → different CIs (with higher variance data, CIs will differ)
    results3 = compute_robustness_ratios(
        encoder_df=encoder_df,
        env_subsets=env_subsets,
        steer_denorm_factor=34.37746770784939,
        accel_denorm_factor=9.81,
        n_resamples=1000,
        bootstrap_seed=999,  # Different seed
        confidence_level=0.95,
    )

    # Ratios should be same (point estimate), but CIs different
    assert results1[0]["ratio_night_day"] == results3[0]["ratio_night_day"]
    # With higher variance, CIs should differ with different seeds
    assert results1[0]["ci_lo_night_day"] != results3[0]["ci_lo_night_day"] or \
           results1[0]["ci_hi_night_day"] != results3[0]["ci_hi_night_day"]


def test_compute_robustness_ratios_denormalization():
    """Test that denormalization factors are applied correctly."""
    import pandas as pd

    encoder_df = pd.DataFrame({
        "scene_name": ["night-01", "day-01"],
        "steer_rmse": [0.20, 0.10],  # Normalized
        "accel_rmse": [0.15, 0.05],  # Normalized
    })

    env_subsets = {
        "night": ["night-01"],
        "rain": [],
        "day_clear": ["day-01"],
    }

    steer_factor = 34.37746770784939
    accel_factor = 9.81

    results = compute_robustness_ratios(
        encoder_df=encoder_df,
        env_subsets=env_subsets,
        steer_denorm_factor=steer_factor,
        accel_denorm_factor=accel_factor,
        n_resamples=1000,
        bootstrap_seed=42,
        confidence_level=0.95,
    )

    assert len(results) == 2

    # Check steering denormalization
    steer_row = results[0]
    expected_night_steer = 0.20 * steer_factor
    expected_day_steer = 0.10 * steer_factor
    assert np.isclose(steer_row["rmse_night"], expected_night_steer, atol=1e-6)
    assert np.isclose(steer_row["rmse_day_clear"], expected_day_steer, atol=1e-6)

    # Ratio should match denormalized values
    expected_ratio = expected_night_steer / expected_day_steer
    assert np.isclose(steer_row["ratio_night_day"], expected_ratio, atol=1e-6)
    assert np.isclose(steer_row["ratio_night_day"], 2.0, atol=1e-6)

    # Check acceleration denormalization
    accel_row = results[1]
    expected_night_accel = 0.15 * accel_factor
    expected_day_accel = 0.05 * accel_factor
    assert np.isclose(accel_row["rmse_night"], expected_night_accel, atol=1e-6)
    assert np.isclose(accel_row["rmse_day_clear"], expected_day_accel, atol=1e-6)

    # Ratio should be 3.0 (0.15 / 0.05)
    assert np.isclose(accel_row["ratio_night_day"], 3.0, atol=1e-6)


def test_compute_robustness_ratios_nan_handling():
    """Test behavior when RMSE contains NaN values."""
    import pandas as pd

    encoder_df = pd.DataFrame({
        "scene_name": ["night-01", "night-02", "day-01", "day-02"],
        "steer_rmse": [0.20, np.nan, 0.10, 0.12],  # One NaN in night subset
        "accel_rmse": [0.15, 0.17, 0.08, 0.09],
    })

    env_subsets = {
        "night": ["night-01", "night-02"],
        "rain": [],
        "day_clear": ["day-01", "day-02"],
    }

    results = compute_robustness_ratios(
        encoder_df=encoder_df,
        env_subsets=env_subsets,
        steer_denorm_factor=34.37746770784939,
        accel_denorm_factor=9.81,
        n_resamples=1000,
        bootstrap_seed=42,
        confidence_level=0.95,
    )

    # Should still return 2 rows
    assert len(results) == 2

    steer_row = results[0]
    # .mean() propagates NaN
    assert np.isnan(steer_row["rmse_night"])

    # Accel should be fine (no NaN)
    accel_row = results[1]
    assert not np.isnan(accel_row["rmse_night"])
    assert accel_row["ratio_night_day"] is not None


# --- Merged from main-tier2 ---

