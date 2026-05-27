"""Integration test for B6.5 sidecars with real dataset.

This test verifies the end-to-end workflow:
1. Load dataset (which tracks data_quality_stats)
2. Write data_quality_report.json
3. Simulate predictions and write per_scenario_rmse.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from config import load_canonical
from data.dataset import NuScenesFrameDataset
from evaluation.metrics import (
    classify_scenes_by_scenario,
    compute_per_scenario_rmse,
)
from evaluation.sidecars import write_data_quality_report, write_per_scenario_rmse


@pytest.mark.skipif(
    not Path("data/v1.0-trainval").exists(),
    reason="NuScenes data not available",
)
def test_data_quality_report_integration(tmp_path: Path) -> None:
    """Test data quality report generation with real dataset."""
    # Load a small split
    cfg = load_canonical()
    dataset = NuScenesFrameDataset(split="p0_val", mode="single_frame")

    # Write report
    output_path = tmp_path / "data_quality_report.json"
    write_data_quality_report(dataset, output_path)

    # Verify output
    assert output_path.exists()

    with open(output_path) as f:
        report = json.load(f)

    # Check all required fields exist
    required_fields = [
        "max_can_alignment_us",
        "blacklisted_scenes_dropped",
        "blacklisted_scene_ids",
        "samples_dropped_for_tolerance",
        "sample_retention_pct",
        "manifest_sha256",
        "total_keyframes",
        "retained_samples",
    ]
    for field in required_fields:
        assert field in report, f"Missing field: {field}"

    # Sanity checks
    assert report["max_can_alignment_us"] == 50000
    assert 0 <= report["sample_retention_pct"] <= 100
    assert report["retained_samples"] <= report["total_keyframes"]
    assert isinstance(report["blacklisted_scene_ids"], list)

    print(f"✓ Data quality report generated successfully")
    print(f"  Total keyframes: {report['total_keyframes']}")
    print(f"  Retained samples: {report['retained_samples']}")
    print(f"  Retention: {report['sample_retention_pct']}%")


@pytest.mark.skipif(
    not Path("data/v1.0-trainval").exists(),
    reason="NuScenes data not available",
)
def test_per_scenario_rmse_integration(tmp_path: Path) -> None:
    """Test per-scenario RMSE CSV generation with simulated predictions."""
    # Load dataset
    cfg = load_canonical()
    dataset = NuScenesFrameDataset(split="p0_val", mode="single_frame")

    # Get scene tokens from samples, actions from dataset items
    scene_tokens = [sample["scene_token"] for sample in dataset.samples]
    actions = [dataset[i]["actions"] for i in range(len(dataset))]

    # Simulate predictions for one encoder
    # In real evaluation, this would come from running inference
    predictions_df = pd.DataFrame({
        "encoder": ["vit_s16"] * len(dataset),
        "scene_token": scene_tokens,
        "steer_pred": [0.1] * len(dataset),  # Dummy predictions
        "accel_pred": [0.05] * len(dataset),
        "steer_true": [float(action[0]) for action in actions],
        "accel_true": [float(action[1]) for action in actions],
    })

    # Classify scenes
    scene_tokens = predictions_df["scene_token"].unique().tolist()
    scene_to_bucket = classify_scenes_by_scenario(dataset.nusc, scene_tokens)

    # Compute per-scenario RMSE
    results_df = compute_per_scenario_rmse(predictions_df, scene_to_bucket, cfg)

    # Write CSV
    output_path = tmp_path / "per_scenario_rmse.csv"
    write_per_scenario_rmse(results_df, predictions_df, scene_to_bucket, output_path)

    # Verify output
    assert output_path.exists()

    output_df = pd.read_csv(output_path)

    # Check schema
    expected_cols = ["encoder", "scenario", "metric", "n_scenes", "mean", "ci_lo", "ci_hi"]
    assert list(output_df.columns) == expected_cols

    # Check data
    assert len(output_df) > 0
    assert all(output_df["encoder"] == "vit_s16")
    assert set(output_df["scenario"].unique()).issubset(
        {"highway", "urban", "intersection", "other"}
    )
    assert set(output_df["metric"].unique()) == {"steer_rmse_deg", "accel_rmse_mps2"}
    assert all(output_df["n_scenes"] > 0)
    assert all(output_df["ci_lo"] <= output_df["mean"])
    assert all(output_df["mean"] <= output_df["ci_hi"])

    print(f"✓ Per-scenario RMSE CSV generated successfully")
    print(f"  Total rows: {len(output_df)}")
    print(f"  Scenarios: {output_df['scenario'].unique().tolist()}")
    print(f"  Unique scenes per scenario:")
    for scenario in output_df["scenario"].unique():
        n = output_df[output_df["scenario"] == scenario]["n_scenes"].iloc[0]
        print(f"    {scenario}: {n} scenes")
