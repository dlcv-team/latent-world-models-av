#!/usr/bin/env python3
"""Generate environmental_robustness.csv with RMSE ratios (B12).

Reads per-scene RMSE from outputs/probes/{encoder}/per_scene_rmse.csv and computes
robustness ratios with bootstrap CIs:
- night/day: Performance degradation in low-light conditions
- rain/day: Performance degradation in wet conditions

Ratios > 1.0 indicate worse performance in challenging conditions.
Ratios near 1.0 indicate robust performance across environments.

Writes to outputs/analysis/environmental_robustness.csv.

Usage:
    python scripts/generate_environmental_robustness_table.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

import numpy as np

from config import load_canonical
from evaluation.metrics import classify_scenes_by_environment, compute_robustness_ratios


def main() -> int:
    print("Generating environmental_robustness.csv with bootstrap CIs...")

    # Load canonical config for bootstrap params
    cfg = load_canonical()
    bootstrap_cfg = cfg.raw["evaluation"]["bootstrap"]
    n_resamples = bootstrap_cfg["n_resamples"]
    bootstrap_seed = bootstrap_cfg["seed"]
    confidence_level = bootstrap_cfg["confidence_level"]

    print(f"  Bootstrap config: {n_resamples} resamples, seed={bootstrap_seed}, CI={confidence_level}")

    # Load environment assignments
    env_config_path = Path("configs/environment_scene_lists.yaml")
    if not env_config_path.exists():
        print(f"\n❌ Error: {env_config_path} not found")
        print("Run generate_per_environment_rmse.py to create it")
        return 1

    with open(env_config_path, "r") as f:
        env_config = yaml.safe_load(f)

    night_scenes = set(env_config.get("night_scenes", []))
    rain_scenes = set(env_config.get("rain_scenes", []))

    print(f"  Environment lists: {len(night_scenes)} night, {len(rain_scenes)} rain")

    # Load per-scene RMSE for all encoders
    probe_dir = Path("outputs/probes")
    if not probe_dir.exists():
        print(f"\n❌ Error: {probe_dir} not found")
        print("Run probe training first to generate per_scene_rmse.csv files")
        return 1

    # Collect all per-scene RMSE files
    per_scene_files = list(probe_dir.glob("*/per_scene_rmse.csv"))
    if not per_scene_files:
        print(f"\n❌ Error: No per_scene_rmse.csv files found in {probe_dir}")
        return 1

    print(f"\nFound {len(per_scene_files)} encoder probe results")

    # Load and concatenate all per-scene data
    scene_dfs = []
    for file_path in per_scene_files:
        encoder_name = file_path.parent.name
        df = pd.read_csv(file_path)
        df["encoder"] = encoder_name
        scene_dfs.append(df)

    all_scenes_df = pd.concat(scene_dfs, ignore_index=True)
    print(f"  Loaded {len(all_scenes_df)} rows (encoder × scene × metric)")
    print(f"  Encoders: {sorted(all_scenes_df['encoder'].unique())}")

    # Get unique scene names
    scene_names = sorted(all_scenes_df["scene_name"].unique())
    print(f"  Total scenes: {len(scene_names)}")

    # Classify scenes by environment
    env_subsets = classify_scenes_by_environment(scene_names, night_scenes, rain_scenes)
    print(f"  Environment subsets: {len(env_subsets['night'])} night, {len(env_subsets['rain'])} rain, {len(env_subsets['day_clear'])} day_clear")

    # Validate environment subsets
    if len(env_subsets["day_clear"]) == 0:
        print("\n❌ Error: No day_clear scenes found (baseline required for ratios)")
        return 1

    if len(env_subsets["night"]) == 0 and len(env_subsets["rain"]) == 0:
        print("\n⚠️  Warning: No night or rain scenes found")
        print("Populate configs/environment_scene_lists.yaml with night/rain scenes")

    # Get denormalization factors from config
    steer_norm_cfg = cfg.normalization("steering")
    accel_norm_cfg = cfg.normalization("acceleration")
    steer_denorm_factor = steer_norm_cfg["eval_back_to_deg_factor"]
    accel_denorm_factor = accel_norm_cfg["divisor"]

    print(f"  Denormalization: steer × {steer_denorm_factor:.2f} → deg, accel × {accel_denorm_factor:.2f} → m/s²")

    # Compute ratios with bootstrap CIs
    results = []
    encoders = sorted(all_scenes_df["encoder"].unique())

    for encoder in encoders:
        encoder_df = all_scenes_df[all_scenes_df["encoder"] == encoder]

        encoder_results = compute_robustness_ratios(
            encoder_df=encoder_df,
            env_subsets=env_subsets,
            steer_denorm_factor=steer_denorm_factor,
            accel_denorm_factor=accel_denorm_factor,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        )

        # Add encoder name to each row and check for warnings
        for row in encoder_results:
            row["encoder"] = encoder

            # Check small sample warnings
            if row["n_day_clear"] < 5:
                print(f"  ⚠️  Warning: {encoder}/{row['metric']} has only {row['n_day_clear']} day_clear scenes (recommend ≥5)")
            if row["n_night"] > 0 and row["n_night"] < 5:
                print(f"  ⚠️  Warning: {encoder}/{row['metric']} has only {row['n_night']} night scenes (small sample → wide CI)")
            if row["n_rain"] > 0 and row["n_rain"] < 10:
                print(f"  ⚠️  Warning: {encoder}/{row['metric']} has only {row['n_rain']} rain scenes (small sample → wide CI)")

        results.extend(encoder_results)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["encoder", "metric"]).reset_index(drop=True)

    print(f"\nComputed {len(results_df)} rows (encoder × metric)")

    # Log scene count summary
    if len(results_df) > 0:
        print("\nScene count summary (independent subsets, may overlap):")
        print(f"  Night scenes:     {results_df['n_night'].iloc[0]}")
        print(f"  Rain scenes:      {results_df['n_rain'].iloc[0]}")
        print(f"  Day_clear scenes: {results_df['n_day_clear'].iloc[0]}")
        overlap = results_df['n_night'].iloc[0] + results_df['n_rain'].iloc[0] - len(night_scenes.intersection(rain_scenes))
        if overlap < len(scene_names):
            print(f"  Total unique: {len(scene_names)} (p0_test)")

    # Write output
    output_path = Path("outputs/analysis/environmental_robustness.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n✓ Wrote {output_path}")
    print(f"  Total rows: {len(results_df)}")
    print(f"  Columns: {list(results_df.columns)}")

    # Show sample with CIs
    print("\nSample output (with bootstrap CIs):")
    display_cols = [
        "encoder", "metric",
        "ratio_night_day", "ci_lo_night_day", "ci_hi_night_day",
        "ratio_rain_day", "ci_lo_rain_day", "ci_hi_rain_day",
        "n_night", "n_rain", "n_day_clear"
    ]
    if len(results_df) > 0:
        sample_df = results_df[display_cols].head(6)
        # Format for readability
        pd.options.display.float_format = "{:.3f}".format
        print(sample_df.to_string(index=False))
        pd.options.display.float_format = None

        # Summary statistics
        print("\nRatio summary (mean ± CI width across encoders):")
        if results_df["ratio_night_day"].notna().any():
            night_ratios = results_df["ratio_night_day"].dropna()
            night_ci_widths = (results_df["ci_hi_night_day"] - results_df["ci_lo_night_day"]).dropna()
            print(f"  Night/Day: {night_ratios.mean():.3f} (mean CI width: {night_ci_widths.mean():.3f})")
        if results_df["ratio_rain_day"].notna().any():
            rain_ratios = results_df["ratio_rain_day"].dropna()
            rain_ci_widths = (results_df["ci_hi_rain_day"] - results_df["ci_lo_rain_day"]).dropna()
            print(f"  Rain/Day:  {rain_ratios.mean():.3f} (mean CI width: {rain_ci_widths.mean():.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
