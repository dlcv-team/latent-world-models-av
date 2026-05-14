"""Tests for evaluation sidecars module (B6.5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from evaluation.sidecars import write_data_quality_report, write_per_scenario_rmse


class TestWriteDataQualityReport:
    """Tests for write_data_quality_report()."""

    def test_write_data_quality_report(self, tmp_path: Path) -> None:
        """Test basic data quality report generation."""
        # Mock dataset with data_quality_stats
        dataset = MagicMock()
        dataset.data_quality_stats = {
            "total_keyframes": 7230,
            "dropped_blacklist": 0,
            "dropped_can_alignment": 0,
            "retained_samples": 7230,
            "blacklisted_scene_ids": [],
        }

        # Mock config
        cfg = MagicMock()
        cfg.raw = {
            "dataset": {
                "can_bus": {"max_alignment_us": 50000}
            }
        }
        cfg.manifest_sha256 = "15c7ddfc8806aa03ced91010031fd259425b8b4c3d8f21a1b9dc0c24a926fd79"

        output_path = tmp_path / "data_quality_report.json"
        write_data_quality_report(dataset, output_path, cfg)

        # Verify file exists and has correct structure
        assert output_path.exists()

        with open(output_path) as f:
            report = json.load(f)

        assert report["max_can_alignment_us"] == 50000
        assert report["blacklisted_scenes_dropped"] == 0
        assert report["blacklisted_scene_ids"] == []
        assert report["samples_dropped_for_tolerance"] == 0
        assert report["sample_retention_pct"] == 100.0
        assert report["manifest_sha256"] == "15c7ddfc8806aa03ced91010031fd259425b8b4c3d8f21a1b9dc0c24a926fd79"
        assert report["total_keyframes"] == 7230
        assert report["retained_samples"] == 7230

    def test_write_data_quality_report_retention_pct(self, tmp_path: Path) -> None:
        """Test retention percentage calculation with dropped samples."""
        dataset = MagicMock()
        dataset.data_quality_stats = {
            "total_keyframes": 1000,
            "dropped_blacklist": 50,
            "dropped_can_alignment": 100,
            "retained_samples": 800,
            "blacklisted_scene_ids": ["scene-0161", "scene-0162"],
        }

        cfg = MagicMock()
        cfg.raw = {"dataset": {"can_bus": {"max_alignment_us": 50000}}}
        cfg.manifest_sha256 = "abc123"

        output_path = tmp_path / "data_quality_report.json"
        write_data_quality_report(dataset, output_path, cfg)

        with open(output_path) as f:
            report = json.load(f)

        # 800 / 1000 = 80.0%
        assert report["sample_retention_pct"] == 80.0
        assert report["blacklisted_scenes_dropped"] == 50
        assert report["samples_dropped_for_tolerance"] == 100
        assert len(report["blacklisted_scene_ids"]) == 2

    def test_write_data_quality_report_missing_stats(self, tmp_path: Path) -> None:
        """Test error handling when data_quality_stats is missing."""
        dataset = MagicMock()
        del dataset.data_quality_stats  # No attribute

        output_path = tmp_path / "data_quality_report.json"

        with pytest.raises(ValueError, match="data_quality_stats attribute"):
            write_data_quality_report(dataset, output_path)

    def test_write_data_quality_report_missing_fields(self, tmp_path: Path) -> None:
        """Test error handling when required stats fields are missing."""
        dataset = MagicMock()
        dataset.data_quality_stats = {
            "total_keyframes": 7230,
            # Missing other required fields
        }

        cfg = MagicMock()
        cfg.raw = {"dataset": {"can_bus": {"max_alignment_us": 50000}}}
        cfg.manifest_sha256 = "abc123"

        output_path = tmp_path / "data_quality_report.json"

        with pytest.raises(ValueError, match="missing required fields"):
            write_data_quality_report(dataset, output_path, cfg)

    def test_write_data_quality_report_creates_directory(self, tmp_path: Path) -> None:
        """Test that output directory is created if it doesn't exist."""
        dataset = MagicMock()
        dataset.data_quality_stats = {
            "total_keyframes": 100,
            "dropped_blacklist": 0,
            "dropped_can_alignment": 0,
            "retained_samples": 100,
            "blacklisted_scene_ids": [],
        }

        cfg = MagicMock()
        cfg.raw = {"dataset": {"can_bus": {"max_alignment_us": 50000}}}
        cfg.manifest_sha256 = "abc123"

        # Deep nested path that doesn't exist
        output_path = tmp_path / "nested" / "dir" / "report.json"
        assert not output_path.parent.exists()

        write_data_quality_report(dataset, output_path, cfg)

        assert output_path.exists()


class TestWritePerScenarioRMSE:
    """Tests for write_per_scenario_rmse()."""

    def test_write_per_scenario_rmse(self, tmp_path: Path) -> None:
        """Test basic per-scenario RMSE CSV generation."""
        # Mock results from compute_per_scenario_rmse
        results_df = pd.DataFrame({
            "encoder": ["vit_s16", "vit_s16", "vit_s16", "vit_s16"],
            "scenario": ["highway", "highway", "urban", "urban"],
            "metric": ["steer_rmse_norm", "accel_rmse_norm", "steer_rmse_norm", "accel_rmse_norm"],
            "mean": [0.0831, 0.0523, 0.0912, 0.0634],
            "ci_lo": [0.0782, 0.0489, 0.0856, 0.0591],
            "ci_hi": [0.0881, 0.0558, 0.0968, 0.0677],
        })

        # Mock predictions with scenes
        predictions_df = pd.DataFrame({
            "encoder": ["vit_s16"] * 20,
            "scene_token": ["scene1"] * 5 + ["scene2"] * 5 + ["scene3"] * 5 + ["scene4"] * 5,
            "steer_pred": [0.1] * 20,
            "accel_pred": [0.1] * 20,
            "steer_true": [0.1] * 20,
            "accel_true": [0.1] * 20,
        })

        # Scene to bucket mapping
        scene_to_bucket = {
            "scene1": "highway",
            "scene2": "highway",
            "scene3": "urban",
            "scene4": "urban",
        }

        output_path = tmp_path / "per_scenario_rmse.csv"
        write_per_scenario_rmse(results_df, predictions_df, scene_to_bucket, output_path)

        # Verify CSV exists and has correct structure
        assert output_path.exists()

        output_df = pd.read_csv(output_path)

        # Check columns
        expected_cols = ["encoder", "scenario", "metric", "n_scenes", "mean", "ci_lo", "ci_hi"]
        assert list(output_df.columns) == expected_cols

        # Check n_scenes counts (2 scenes per scenario)
        assert all(output_df["n_scenes"] == 2)

        # Check sorting (encoder, scenario, metric)
        assert output_df["scenario"].tolist() == ["highway", "highway", "urban", "urban"]
        assert output_df["metric"].tolist() == [
            "accel_rmse_norm",
            "steer_rmse_norm",
            "accel_rmse_norm",
            "steer_rmse_norm",
        ]

    def test_write_per_scenario_rmse_n_scenes(self, tmp_path: Path) -> None:
        """Test unique scene counting per encoder × scenario."""
        results_df = pd.DataFrame({
            "encoder": ["enc1", "enc1", "enc2", "enc2"],
            "scenario": ["highway", "urban", "highway", "urban"],
            "metric": ["steer_rmse_norm", "steer_rmse_norm", "steer_rmse_norm", "steer_rmse_norm"],
            "mean": [0.1, 0.2, 0.15, 0.25],
            "ci_lo": [0.09, 0.18, 0.13, 0.23],
            "ci_hi": [0.11, 0.22, 0.17, 0.27],
        })

        # enc1 has 3 highway scenes, 1 urban scene (u1 repeated)
        # enc2 has 2 highway scenes, 1 urban scene
        predictions_df = pd.DataFrame({
            "encoder": ["enc1"] * 10 + ["enc2"] * 6,
            "scene_token": [
                # enc1: highway scenes h1, h2, h3 (repeated)
                "h1", "h2", "h3", "h1", "h2", "h3",
                # enc1: urban scene u1 (repeated 4 times)
                "u1", "u1", "u1", "u1",
                # enc2: highway scenes h4, h5 (repeated)
                "h4", "h5", "h4", "h5",
                # enc2: urban scene u2 (repeated)
                "u2", "u2"
            ],
        })

        scene_to_bucket = {
            "h1": "highway",
            "h2": "highway",
            "h3": "highway",
            "h4": "highway",
            "h5": "highway",
            "u1": "urban",
            "u2": "urban",
        }

        output_path = tmp_path / "per_scenario_rmse.csv"
        write_per_scenario_rmse(results_df, predictions_df, scene_to_bucket, output_path)

        output_df = pd.read_csv(output_path)

        # Verify scene counts
        enc1_highway = output_df[
            (output_df["encoder"] == "enc1") & (output_df["scenario"] == "highway")
        ]["n_scenes"].iloc[0]
        enc1_urban = output_df[
            (output_df["encoder"] == "enc1") & (output_df["scenario"] == "urban")
        ]["n_scenes"].iloc[0]
        enc2_highway = output_df[
            (output_df["encoder"] == "enc2") & (output_df["scenario"] == "highway")
        ]["n_scenes"].iloc[0]
        enc2_urban = output_df[
            (output_df["encoder"] == "enc2") & (output_df["scenario"] == "urban")
        ]["n_scenes"].iloc[0]

        assert enc1_highway == 3
        assert enc1_urban == 1  # u1 repeated but counted once
        assert enc2_highway == 2
        assert enc2_urban == 1

    def test_write_per_scenario_rmse_multiple_encoders(self, tmp_path: Path) -> None:
        """Test with multiple encoders."""
        results_df = pd.DataFrame({
            "encoder": ["vit_s16", "vit_s16", "dinov2_s14", "dinov2_s14"],
            "scenario": ["highway", "urban", "highway", "urban"],
            "metric": ["steer_rmse_norm"] * 4,
            "mean": [0.1, 0.2, 0.15, 0.25],
            "ci_lo": [0.09, 0.18, 0.13, 0.23],
            "ci_hi": [0.11, 0.22, 0.17, 0.27],
        })

        predictions_df = pd.DataFrame({
            "encoder": ["vit_s16"] * 4 + ["dinov2_s14"] * 4,
            "scene_token": ["s1", "s2"] * 2 + ["s3", "s4"] * 2,
        })

        scene_to_bucket = {
            "s1": "highway",
            "s2": "urban",
            "s3": "highway",
            "s4": "urban",
        }

        output_path = tmp_path / "per_scenario_rmse.csv"
        write_per_scenario_rmse(results_df, predictions_df, scene_to_bucket, output_path)

        output_df = pd.read_csv(output_path)

        assert len(output_df) == 4
        assert set(output_df["encoder"].unique()) == {"vit_s16", "dinov2_s14"}
        assert all(output_df["n_scenes"] == 1)  # Each encoder×scenario has 1 scene

    def test_write_per_scenario_rmse_empty_results(self, tmp_path: Path) -> None:
        """Test handling of empty results DataFrame."""
        results_df = pd.DataFrame()
        predictions_df = pd.DataFrame()
        scene_to_bucket = {}

        output_path = tmp_path / "per_scenario_rmse.csv"
        write_per_scenario_rmse(results_df, predictions_df, scene_to_bucket, output_path)

        assert output_path.exists()

        output_df = pd.read_csv(output_path)
        assert len(output_df) == 0
        assert list(output_df.columns) == [
            "encoder",
            "scenario",
            "metric",
            "n_scenes",
            "mean",
            "ci_lo",
            "ci_hi",
        ]

    def test_write_per_scenario_rmse_missing_columns(self, tmp_path: Path) -> None:
        """Test error handling for missing columns in results_df."""
        results_df = pd.DataFrame({
            "encoder": ["vit_s16"],
            "scenario": ["highway"],
            # Missing metric, mean, ci_lo, ci_hi
        })

        predictions_df = pd.DataFrame({"encoder": ["vit_s16"], "scene_token": ["s1"]})
        scene_to_bucket = {"s1": "highway"}

        output_path = tmp_path / "per_scenario_rmse.csv"

        with pytest.raises(ValueError, match="missing required columns"):
            write_per_scenario_rmse(results_df, predictions_df, scene_to_bucket, output_path)

    def test_write_per_scenario_rmse_missing_scene_column(self, tmp_path: Path) -> None:
        """Test error handling when predictions_df has no scene column."""
        results_df = pd.DataFrame({
            "encoder": ["vit_s16"],
            "scenario": ["highway"],
            "metric": ["steer_rmse_norm"],
            "mean": [0.1],
            "ci_lo": [0.09],
            "ci_hi": [0.11],
        })

        predictions_df = pd.DataFrame({
            "encoder": ["vit_s16"],
            # No scene_token or scene_name column
        })
        scene_to_bucket = {"s1": "highway"}

        output_path = tmp_path / "per_scenario_rmse.csv"

        with pytest.raises(ValueError, match="scene_token.*scene_name"):
            write_per_scenario_rmse(results_df, predictions_df, scene_to_bucket, output_path)

    def test_write_per_scenario_rmse_empty_scene_mapping(self, tmp_path: Path) -> None:
        """Test error handling when scene_to_bucket is empty."""
        results_df = pd.DataFrame({
            "encoder": ["vit_s16"],
            "scenario": ["highway"],
            "metric": ["steer_rmse_norm"],
            "mean": [0.1],
            "ci_lo": [0.09],
            "ci_hi": [0.11],
        })

        predictions_df = pd.DataFrame({"encoder": ["vit_s16"], "scene_token": ["s1"]})
        scene_to_bucket = {}  # Empty mapping

        output_path = tmp_path / "per_scenario_rmse.csv"

        with pytest.raises(ValueError, match="scene_to_bucket mapping is empty"):
            write_per_scenario_rmse(results_df, predictions_df, scene_to_bucket, output_path)

    def test_write_per_scenario_rmse_scene_name_column(self, tmp_path: Path) -> None:
        """Test using scene_name instead of scene_token."""
        results_df = pd.DataFrame({
            "encoder": ["vit_s16"],
            "scenario": ["highway"],
            "metric": ["steer_rmse_norm"],
            "mean": [0.1],
            "ci_lo": [0.09],
            "ci_hi": [0.11],
        })

        # Use scene_name instead of scene_token
        predictions_df = pd.DataFrame({
            "encoder": ["vit_s16", "vit_s16"],
            "scene_name": ["scene-0001", "scene-0002"],
        })

        scene_to_bucket = {
            "scene-0001": "highway",
            "scene-0002": "highway",
        }

        output_path = tmp_path / "per_scenario_rmse.csv"
        write_per_scenario_rmse(results_df, predictions_df, scene_to_bucket, output_path)

        output_df = pd.read_csv(output_path)
        assert len(output_df) == 1
        assert output_df["n_scenes"].iloc[0] == 2

    def test_write_per_scenario_rmse_creates_directory(self, tmp_path: Path) -> None:
        """Test that output directory is created if it doesn't exist."""
        results_df = pd.DataFrame({
            "encoder": ["vit_s16"],
            "scenario": ["highway"],
            "metric": ["steer_rmse_norm"],
            "mean": [0.1],
            "ci_lo": [0.09],
            "ci_hi": [0.11],
        })

        predictions_df = pd.DataFrame({"encoder": ["vit_s16"], "scene_token": ["s1"]})
        scene_to_bucket = {"s1": "highway"}

        output_path = tmp_path / "nested" / "dir" / "rmse.csv"
        assert not output_path.parent.exists()

        write_per_scenario_rmse(results_df, predictions_df, scene_to_bucket, output_path)

        assert output_path.exists()
