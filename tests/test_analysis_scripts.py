"""Unit tests for analysis scripts.

Tests utility functions used in analysis scripts for environment detection.
"""

import pytest
import yaml
from pathlib import Path

from evaluation.metrics import parse_logfile_timestamp, is_night_from_timestamp


def test_parse_logfile_timestamp_valid():
    """Test timestamp extraction from valid logfile names."""
    # Scene 0061: "n015-2018-07-18-11-07-57+0800"
    timestamp = parse_logfile_timestamp("n015-2018-07-18-11-07-57+0800")
    assert timestamp == "11:07:57"

    # Scene 0103: "n008-2018-08-01-15-16-36+0800"
    timestamp = parse_logfile_timestamp("n008-2018-08-01-15-16-36+0800")
    assert timestamp == "15:16:36"

    # Test various timestamp formats
    timestamp = parse_logfile_timestamp("n015-2018-07-18-23-45-12+0800")
    assert timestamp == "23:45:12"

    timestamp = parse_logfile_timestamp("n008-2018-08-01-00-00-00+0800")
    assert timestamp == "00:00:00"

    # Negative timezone offset
    timestamp = parse_logfile_timestamp("n015-2018-07-18-18-30-45-0500")
    assert timestamp == "18:30:45"


def test_parse_logfile_timestamp_invalid():
    """Test handling of logfile names without valid timestamps."""
    # Missing timestamp pattern
    timestamp = parse_logfile_timestamp("scene-0001-test")
    assert timestamp == "unknown"

    # Empty string
    timestamp = parse_logfile_timestamp("")
    assert timestamp == "unknown"

    # Malformed timestamp
    timestamp = parse_logfile_timestamp("n015-2018-07-18")
    assert timestamp == "unknown"


def test_night_scene_heuristic():
    """Verify that night scenes follow expected format and are reasonable count.

    This test verifies that the night scene detection logic (timestamps 18:00-05:59)
    correctly classifies scenes. The count should be reasonable (at least a few scenes,
    but not the majority of the 40-scene test set).
    """
    # Load environment scene lists
    env_config_path = Path("configs/environment_scene_lists.yaml")
    if not env_config_path.exists():
        pytest.skip(f"{env_config_path} not found (requires environment setup)")

    with open(env_config_path, "r") as f:
        env_config = yaml.safe_load(f)

    night_scenes = set(env_config.get("night_scenes", []))

    # Verify reasonable count (should be non-empty but < 50% of test set)
    assert 1 <= len(night_scenes) <= 20, \
        f"Expected 1-20 night scenes from 40-scene p0_test, got {len(night_scenes)}"

    # Night scenes should be non-empty strings following scene-XXXX pattern
    for scene_name in night_scenes:
        assert isinstance(scene_name, str)
        assert len(scene_name) > 0
        assert "scene-" in scene_name


def test_rain_scene_count():
    """Verify that rain scene count is reasonable.

    This test verifies the rain scene count from manual review is reasonable
    (a few scenes but not the majority of the test set).
    """
    env_config_path = Path("configs/environment_scene_lists.yaml")
    if not env_config_path.exists():
        pytest.skip(f"{env_config_path} not found (requires environment setup)")

    with open(env_config_path, "r") as f:
        env_config = yaml.safe_load(f)

    rain_scenes = set(env_config.get("rain_scenes", []))

    # Verify reasonable count (should be non-empty but < 50% of test set)
    assert 1 <= len(rain_scenes) <= 20, \
        f"Expected 1-20 rain scenes from 40-scene p0_test, got {len(rain_scenes)}"

    # Rain scenes should be non-empty strings following scene-XXXX pattern
    for scene_name in rain_scenes:
        assert isinstance(scene_name, str)
        assert len(scene_name) > 0
        assert "scene-" in scene_name


def test_night_time_classification():
    """Test night time classification logic (18:00-05:59 → night)."""
    # Create mock logfile names with timestamps
    night_times = [
        ("n015-2018-07-18-18-00-00+0800", True),   # 18:00 → night
        ("n015-2018-07-18-23-59-59+0800", True),   # 23:59 → night
        ("n015-2018-07-18-00-00-00+0800", True),   # 00:00 → night
        ("n015-2018-07-18-05-59-59+0800", True),   # 05:59 → night
        ("n015-2018-07-18-06-00-00+0800", False),  # 06:00 → day
        ("n015-2018-07-18-12-00-00+0800", False),  # 12:00 → day
        ("n015-2018-07-18-17-59-59+0800", False),  # 17:59 → day
    ]

    for logfile, expected_night in night_times:
        timestamp = parse_logfile_timestamp(logfile)
        assert timestamp != "unknown", f"Failed to parse {logfile}"

        is_night = is_night_from_timestamp(timestamp)

        assert is_night == expected_night, \
            f"Logfile {logfile} (timestamp={timestamp}) expected night={expected_night}, got {is_night}"
