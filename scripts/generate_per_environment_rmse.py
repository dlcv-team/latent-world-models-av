#!/usr/bin/env python3
"""Generate per_environment_rmse.csv from probe results (B12).

Reads outputs/probes/*/per_scene_rmse.csv (columns: steer_rmse, accel_rmse
with normalized values in [-1, 1] space), classifies scenes by environment
(night/rain/day) using manual scene lists, and computes per-environment RMSE
with bootstrap confidence intervals. Converts to physical units (degrees, m/s²)
when writing output CSV.

Writes to outputs/analysis/per_environment_rmse.csv.

Usage:
    python scripts/generate_per_environment_rmse.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from config import load_canonical
from evaluation.metrics import bootstrap_mean_ci, classify_scenes_by_environment, denormalize_rmse_dataframe


def load_environment_scene_lists(config_path: Path) -> tuple[set[str], set[str]]:
    """Load manually-identified night and rain scene lists from config.

    Args:
        config_path: Path to environment_scene_lists.yaml

    Returns:
        Tuple of (night_scenes, rain_scenes) as sets

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is malformed
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Environment scene lists not found at {config_path}. "
            "Please populate configs/environment_scene_lists.yaml with "
            "manually-identified night and rain scenes from p0_test."
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    night_scenes = set(config.get("night_scenes", []))
    rain_scenes = set(config.get("rain_scenes", []))

    return night_scenes, rain_scenes


def main():
    print("Generating per_environment_rmse.csv from probe results...")

    # Load config
    cfg = load_canonical()

    # Load environment scene lists
    env_config_path = cfg.root / "configs" / "environment_scene_lists.yaml"
    print(f"\nLoading environment scene lists from {env_config_path}...")

    try:
        night_scenes, rain_scenes = load_environment_scene_lists(env_config_path)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nTo use this script:")
        print("1. Manually inspect p0_test scenes in nuScenes")
        print("2. Populate configs/environment_scene_lists.yaml with night/rain scene names")
        print("3. Re-run this script")
        return

    print(f"  Night scenes: {len(night_scenes)}")
    print(f"  Rain scenes: {len(rain_scenes)}")

    if len(night_scenes) == 0 and len(rain_scenes) == 0:
        print("\n⚠️  Warning: No night or rain scenes found in config.")
        print("All scenes will be classified as 'day'.")
        print("Populate configs/environment_scene_lists.yaml to enable environment analysis.")

    # Load probe results
    probe_root = Path("outputs/probes")
    if not probe_root.exists():
        print(f"\n❌ Probe results not found at {probe_root}")
        print("Run probe training first: python training/train_probe.py")
        return

    encoders = sorted([d.name for d in probe_root.iterdir() if d.is_dir()])
    print(f"\nFound {len(encoders)} encoders: {encoders}")

    # Collect all per-scene RMSE values
    all_scene_rmse = []

    for encoder in encoders:
        csv_path = probe_root / encoder / "per_scene_rmse.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path)

        # Validate canonical column names (without _norm suffix per origin/main)
        required_cols = {"scene_name", "steer_rmse", "accel_rmse"}
        if not required_cols.issubset(df.columns):
            print(f"  ⚠️  {encoder}: Skipping (uses old column names with _norm suffix)")
            print(f"      Expected: {required_cols}, Found: {set(df.columns)}")
            print(f"      Re-run probe training for this encoder to use canonical format")
            continue

        print(f"  {encoder}: {len(df)} scenes")

        # Values are normalized in [-1, 1] space
        for _, row in df.iterrows():
            all_scene_rmse.append({
                "encoder": encoder,
                "scene_name": row["scene_name"],
                "steer_rmse": row["steer_rmse"],
                "accel_rmse": row["accel_rmse"],
            })

    if not all_scene_rmse:
        print("\n❌ No per_scene_rmse.csv files found")
        return

    scene_rmse_df = pd.DataFrame(all_scene_rmse)
    print(f"\nTotal scene-encoder pairs: {len(scene_rmse_df)}")

    # Classify scenes by environment (returns independent subsets)
    print("\nClassifying scenes by environment...")
    unique_scene_names = scene_rmse_df["scene_name"].unique().tolist()
    env_subsets = classify_scenes_by_environment(
        unique_scene_names, night_scenes, rain_scenes
    )

    # Compute per-environment RMSE with bootstrap CIs
    print("\nComputing per-environment aggregates with bootstrap CIs...")

    bootstrap_cfg = cfg.raw["evaluation"]["bootstrap"]
    n_resamples = bootstrap_cfg["n_resamples"]
    bootstrap_seed = bootstrap_cfg["seed"]
    confidence_level = bootstrap_cfg["confidence_level"]

    results = []

    for encoder in encoders:
        encoder_df = scene_rmse_df[scene_rmse_df["encoder"] == encoder]

        # Aggregate by environment (independent subsets)
        for metric_name, col_name in [("steer_rmse", "steer_rmse"),
                                       ("accel_rmse", "accel_rmse")]:
            for environment in ["night", "rain", "day_clear"]:
                # Get scenes for this environment subset
                env_scenes = env_subsets[environment]
                env_df = encoder_df[encoder_df["scene_name"].isin(env_scenes)]

                if len(env_df) == 0:
                    continue

                rmse_values = env_df[col_name].values
                n_scenes = len(rmse_values)

                mean, ci_lo, ci_hi = bootstrap_mean_ci(
                    rmse_values,
                    n_resamples=n_resamples,
                    seed=bootstrap_seed,
                    confidence_level=confidence_level,
                )

                results.append({
                    "encoder": encoder,
                    "environment": environment,
                    "metric": metric_name,
                    "n_scenes": n_scenes,
                    "mean": mean,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["encoder", "environment", "metric"]).reset_index(drop=True)

    print("\nConverting to physical units for output CSV...")
    results_df = denormalize_rmse_dataframe(results_df, cfg)

    print(f"\nComputed {len(results_df)} rows:")
    print(f"  Encoders: {sorted(results_df['encoder'].unique())}")
    print(f"  Environments: {sorted(results_df['environment'].unique())}")
    print(f"  Metrics: {sorted(results_df['metric'].unique())}")

    # Write output
    output_path = Path("outputs/analysis/per_environment_rmse.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n✓ Wrote {output_path}")
    print(f"  Total rows: {len(results_df)}")

    # Show sample
    print("\nSample (first 12 rows):")
    print(results_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
