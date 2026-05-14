"""Sidecar output functions for evaluation results.

Generates JSON and CSV files that serve as single source of truth for
downstream reporting and figure generation (B8, Methods section).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from data.dataset import NuScenesFrameDataset
    from config import CanonicalConfig

logger = logging.getLogger(__name__)


def write_data_quality_report(
    dataset: NuScenesFrameDataset,
    output_path: Path | str,
    cfg: CanonicalConfig | None = None,
) -> None:
    """Write data quality summary to JSON.

    Emits data_quality_report.json with dataset statistics for Methods section
    documentation. Reports CAN alignment policy, blacklist drops, sample
    retention, and manifest verification.

    Parameters
    ----------
    dataset
        NuScenesFrameDataset instance with data_quality_stats populated.
    output_path
        Path to write JSON file (typically outputs/data_quality_report.json).
    cfg
        Configuration object with manifest SHA256 and CAN alignment threshold.
        If None, loads canonical config.

    Raises
    ------
    ValueError
        If dataset.data_quality_stats is missing or incomplete.
    FileNotFoundError
        If output directory does not exist.

    Notes
    -----
    Output schema matches B6.5 PRD requirements:
    - max_can_alignment_us: CAN tolerance threshold from config
    - blacklisted_scenes_dropped: Count of scenes dropped due to blacklist
    - blacklisted_scene_ids: List of scene names that were blacklisted
    - samples_dropped_for_tolerance: Samples exceeding CAN alignment threshold
    - sample_retention_pct: Percentage of candidate samples retained
    - manifest_sha256: SHA256 of active subset manifest
    - total_keyframes: Total keyframes scanned
    - retained_samples: Samples passing all filters

    Examples
    --------
    >>> dataset = NuScenesFrameDataset(split="p0_train")
    >>> write_data_quality_report(dataset, "outputs/data_quality_report.json")
    """
    if cfg is None:
        from config import load_canonical
        cfg = load_canonical()

    # Validate dataset has quality stats
    if not hasattr(dataset, "data_quality_stats"):
        raise ValueError(
            "Dataset must have data_quality_stats attribute. "
            "Ensure _build_sample_index() has been called."
        )

    stats = dataset.data_quality_stats

    # Validate required fields
    required_fields = [
        "total_keyframes",
        "dropped_blacklist",
        "dropped_can_alignment",
        "retained_samples",
        "blacklisted_scene_ids",
    ]
    missing = [f for f in required_fields if f not in stats]
    if missing:
        raise ValueError(
            f"data_quality_stats missing required fields: {missing}. "
            "Update data/dataset.py to track these fields."
        )

    # Extract config values
    max_alignment_us = cfg.raw["dataset"]["can_bus"]["max_alignment_us"]
    manifest_sha256 = cfg.manifest_sha256

    # Compute retention percentage
    total = stats["total_keyframes"]
    retained = stats["retained_samples"]
    retention_pct = (retained / total * 100.0) if total > 0 else 0.0

    # Build output report
    report = {
        "max_can_alignment_us": max_alignment_us,
        "blacklisted_scenes_dropped": stats["dropped_blacklist"],
        "blacklisted_scene_ids": stats["blacklisted_scene_ids"],
        "samples_dropped_for_tolerance": stats["dropped_can_alignment"],
        "sample_retention_pct": round(retention_pct, 2),
        "manifest_sha256": manifest_sha256,
        "total_keyframes": total,
        "retained_samples": retained,
    }

    # Write JSON with pretty formatting
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Wrote data quality report to %s (retention: %.1f%%)",
        output_path,
        retention_pct,
    )


def write_per_scenario_rmse(
    results_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    scene_to_bucket: dict[str, str],
    output_path: Path | str,
) -> None:
    """Write per-scenario RMSE with confidence intervals to CSV.

    Emits per_scenario_rmse.csv for B8 heatmap generation. Extends the output
    of compute_per_scenario_rmse() with n_scenes column showing unique scene
    counts per (encoder × scenario) combination.

    Parameters
    ----------
    results_df
        DataFrame from compute_per_scenario_rmse() with columns:
        encoder, scenario, metric, mean, ci_lo, ci_hi.
    predictions_df
        Raw predictions DataFrame with columns:
        encoder, scene_token (or scene_name), steer_pred, accel_pred,
        steer_true, accel_true. Used to count unique scenes per group.
    scene_to_bucket
        Mapping from scene_token to scenario category (output of
        classify_scenes_by_scenario).
    output_path
        Path to write CSV file (typically outputs/per_scenario_rmse.csv).

    Raises
    ------
    ValueError
        If required columns are missing from DataFrames.
    FileNotFoundError
        If output directory does not exist.

    Notes
    -----
    Output CSV has 7 columns:
    - encoder: Encoder name (vit_s16, dinov2_s14, clip_b32, vqvae, vjepa2)
    - scenario: Scenario bucket (highway, urban, intersection, other)
    - metric: Either "steer_rmse_norm" or "accel_rmse_norm"
    - n_scenes: Number of unique scenes in this (encoder, scenario) group
    - mean: Mean RMSE (normalized space)
    - ci_lo: Lower 95% confidence interval bound
    - ci_hi: Upper 95% confidence interval bound

    Rows are sorted by (encoder, scenario, metric) for deterministic output.

    Examples
    --------
    >>> results_df = compute_per_scenario_rmse(predictions_df, scene_to_bucket, cfg)
    >>> write_per_scenario_rmse(
    ...     results_df, predictions_df, scene_to_bucket,
    ...     "outputs/per_scenario_rmse.csv"
    ... )
    """
    # Validate inputs
    if results_df.empty:
        logger.warning("results_df is empty, writing empty CSV with schema")
        empty_df = pd.DataFrame(
            columns=["encoder", "scenario", "metric", "n_scenes", "mean", "ci_lo", "ci_hi"]
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(output_path, index=False)
        return

    required_results_cols = ["encoder", "scenario", "metric", "mean", "ci_lo", "ci_hi"]
    missing = [c for c in required_results_cols if c not in results_df.columns]
    if missing:
        raise ValueError(
            f"results_df missing required columns: {missing}. "
            "Ensure it is output from compute_per_scenario_rmse()."
        )

    # Determine scene column name
    if "scene_token" in predictions_df.columns:
        scene_col = "scene_token"
    elif "scene_name" in predictions_df.columns:
        scene_col = "scene_name"
    else:
        raise ValueError(
            "predictions_df must have 'scene_token' or 'scene_name' column"
        )

    required_pred_cols = ["encoder", scene_col]
    missing = [c for c in required_pred_cols if c not in predictions_df.columns]
    if missing:
        raise ValueError(
            f"predictions_df missing required columns: {missing}. "
            "Ensure it has encoder and scene identifier columns."
        )

    if not scene_to_bucket:
        raise ValueError("scene_to_bucket mapping is empty")

    # Map predictions to scenarios
    predictions_with_scenario = predictions_df.copy()
    predictions_with_scenario["scenario"] = predictions_with_scenario[scene_col].map(
        scene_to_bucket
    )

    # Count unique scenes per encoder × scenario
    scene_counts = (
        predictions_with_scenario.dropna(subset=["scenario"])
        .groupby(["encoder", "scenario"])[scene_col]
        .nunique()
        .reset_index(name="n_scenes")
    )

    # Merge n_scenes into results
    output_df = results_df.merge(scene_counts, on=["encoder", "scenario"], how="left")

    # Fill missing n_scenes with 0 (shouldn't happen, but defensive)
    output_df["n_scenes"] = output_df["n_scenes"].fillna(0).astype(int)

    # Reorder columns per PRD spec
    output_df = output_df[["encoder", "scenario", "metric", "n_scenes", "mean", "ci_lo", "ci_hi"]]

    # Sort for deterministic output
    output_df = output_df.sort_values(["encoder", "scenario", "metric"]).reset_index(
        drop=True
    )

    # Write CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    logger.info(
        "Wrote per-scenario RMSE to %s (%d rows, %d encoders × %d scenarios)",
        output_path,
        len(output_df),
        output_df["encoder"].nunique(),
        output_df["scenario"].nunique(),
    )
