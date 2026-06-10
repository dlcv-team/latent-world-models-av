"""Reproducibility tests for environment_scene_lists.yaml.

Ensures the manually-entered night/rain scene lists match the automated
timestamp-based classification logic from list_test_scene_descriptions.py.
This prevents drift between hand-copied YAML and the derivation script.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from nuscenes.nuscenes import NuScenes

from config import load_canonical, manifest_split
from evaluation.metrics import parse_logfile_timestamp, is_night_from_timestamp


def derive_night_scenes_from_timestamps(nusc: NuScenes, test_scenes: list[str]) -> set[str]:
    """Classify scenes as night based on timestamp (6pm-6am).

    Args:
        nusc: NuScenes instance
        test_scenes: List of scene names to classify

    Returns:
        Set of scene names classified as night
    """
    night_scenes = set()
    for scene_name in test_scenes:
        scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
        if scene is None:
            continue

        log = nusc.get("log", scene["log_token"])
        logfile = log.get("logfile", "")
        timestamp = parse_logfile_timestamp(logfile)

        if is_night_from_timestamp(timestamp):
            night_scenes.add(scene_name)

    return night_scenes


@pytest.mark.skipif(
    not pytest.importorskip("nuscenes", reason="nuscenes-devkit not installed"),
    reason="nuscenes-devkit not installed",
)
def test_yaml_night_scenes_match_timestamp_classification():
    """Verify YAML night_scenes list matches timestamp-derived classification.

    The YAML night_scenes list is hand-copied from list_test_scene_descriptions.py
    output. This test ensures it stays synchronized with the automated
    timestamp-based classification logic (6pm-6am).
    """
    cfg = load_canonical()

    # Load YAML night scenes
    env_config_path = cfg.root / "configs" / "environment_scene_lists.yaml"
    if not env_config_path.exists():
        pytest.skip(f"environment_scene_lists.yaml not found at {env_config_path}")

    with open(env_config_path) as f:
        env_config = yaml.safe_load(f)

    yaml_night_scenes = set(env_config.get("night_scenes", []))
    if not yaml_night_scenes:
        pytest.skip("No night scenes in YAML config")

    # Derive night scenes from timestamps
    nuscenes_root = cfg.root / "data"
    version = cfg.raw["dataset"]["version"]
    if not (nuscenes_root / version).exists():
        pytest.skip(f"NuScenes dataset {version} not found at {nuscenes_root}")
    
    nusc = NuScenes(version=version, dataroot=str(nuscenes_root), verbose=False)

    test_scenes = manifest_split(cfg, "p0_test")
    derived_night_scenes = derive_night_scenes_from_timestamps(nusc, test_scenes)

    # Compare
    missing_from_yaml = derived_night_scenes - yaml_night_scenes
    extra_in_yaml = yaml_night_scenes - derived_night_scenes

    assert not missing_from_yaml, (
        f"Night scenes derived from timestamps but missing from YAML: {sorted(missing_from_yaml)}. "
        f"Re-run scripts/list_test_scene_descriptions.py and update "
        f"configs/environment_scene_lists.yaml"
    )

    assert not extra_in_yaml, (
        f"Night scenes in YAML but not derived from timestamps: {sorted(extra_in_yaml)}. "
        f"These scenes do not have 6pm-6am timestamps. Remove from YAML or "
        f"adjust classification logic in list_test_scene_descriptions.py"
    )


@pytest.mark.skipif(
    not pytest.importorskip("nuscenes", reason="nuscenes-devkit not installed"),
    reason="nuscenes-devkit not installed",
)
def test_yaml_rain_scenes_are_subset_of_test_split():
    """Verify all rain scenes in YAML exist in p0_test split.

    Rain scenes are manually identified from descriptions, so we can't
    auto-derive them, but we can verify they're valid scene names.
    """
    cfg = load_canonical()

    # Load YAML rain scenes
    env_config_path = cfg.root / "configs" / "environment_scene_lists.yaml"
    if not env_config_path.exists():
        pytest.skip(f"environment_scene_lists.yaml not found at {env_config_path}")

    with open(env_config_path) as f:
        env_config = yaml.safe_load(f)

    yaml_rain_scenes = set(env_config.get("rain_scenes", []))
    if not yaml_rain_scenes:
        pytest.skip("No rain scenes in YAML config")

    # Get p0_test scenes
    test_scenes = set(manifest_split(cfg, "p0_test"))

    # Verify all rain scenes are in test split
    invalid_rain_scenes = yaml_rain_scenes - test_scenes

    assert not invalid_rain_scenes, (
        f"Rain scenes in YAML but not in p0_test split: {sorted(invalid_rain_scenes)}. "
        f"Remove invalid scenes from configs/environment_scene_lists.yaml"
    )


def test_yaml_environment_scene_counts_are_documented():
    """Verify YAML notes section documents correct scene counts.

    Helps catch copy-paste errors when updating the YAML file.
    """
    cfg = load_canonical()

    env_config_path = cfg.root / "configs" / "environment_scene_lists.yaml"
    if not env_config_path.exists():
        pytest.skip(f"environment_scene_lists.yaml not found at {env_config_path}")

    with open(env_config_path) as f:
        env_config = yaml.safe_load(f)

    night_scenes = env_config.get("night_scenes", [])
    rain_scenes = env_config.get("rain_scenes", [])
    notes = env_config.get("notes", "")

    # Extract documented counts from notes (flexible parsing)
    night_count_match = re.search(r'Night scenes.*?(\d+)\s+scenes', notes)
    rain_count_match = re.search(r'Rain scenes.*?(\d+)\s+scenes', notes)

    if night_count_match:
        documented_night_count = int(night_count_match.group(1))
        actual_night_count = len(night_scenes)
        assert documented_night_count == actual_night_count, (
            f"YAML notes document {documented_night_count} night scenes, "
            f"but {actual_night_count} are listed. Update notes section."
        )

    if rain_count_match:
        documented_rain_count = int(rain_count_match.group(1))
        actual_rain_count = len(rain_scenes)
        assert documented_rain_count == actual_rain_count, (
            f"YAML notes document {documented_rain_count} rain scenes, "
            f"but {actual_rain_count} are listed. Update notes section."
        )
