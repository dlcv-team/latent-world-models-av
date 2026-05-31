#!/usr/bin/env python3
"""Generate environmental_robustness.csv with RMSE ratios (B12).

Reads outputs/analysis/per_environment_rmse.csv and computes robustness ratios:
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


def main():
    print("Generating environmental_robustness.csv from per-environment RMSE...")

    # Load per-environment RMSE
    input_path = Path("outputs/analysis/per_environment_rmse.csv")
    if not input_path.exists():
        print(f"\n❌ Input file not found: {input_path}")
        print("Run generate_per_environment_rmse.py first")
        return

    df = pd.read_csv(input_path)
    print(f"\nLoaded {len(df)} rows from {input_path}")
    print(f"  Encoders: {sorted(df['encoder'].unique())}")
    print(f"  Environments: {sorted(df['environment'].unique())}")
    print(f"  Metrics: {sorted(df['metric'].unique())}")

    # Validate required environments
    available_envs = set(df["environment"].unique())
    if "day" not in available_envs:
        print("\n❌ Error: 'day' environment not found in input data")
        print("Cannot compute ratios without baseline day performance")
        return

    if "night" not in available_envs and "rain" not in available_envs:
        print("\n⚠️  Warning: Neither 'night' nor 'rain' environments found")
        print("Populate configs/environment_scene_lists.yaml with night/rain scenes")

    # Pivot to get environment columns
    # Columns after pivot: encoder, metric, n_scenes_day, mean_day, ci_lo_day, ci_hi_day, ...
    pivot_df = df.pivot_table(
        index=["encoder", "metric"],
        columns="environment",
        values=["mean", "n_scenes"],
        aggfunc="first",
    )

    # Flatten multi-index columns
    pivot_df.columns = [f"{val}_{env}" for val, env in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    print(f"\nPivoted data shape: {pivot_df.shape}")

    # Compute ratios
    results = []

    for _, row in pivot_df.iterrows():
        encoder = row["encoder"]
        metric = row["metric"]

        # Extract values (handle missing environments gracefully)
        mean_day = row.get("mean_day")
        n_day = row.get("n_scenes_day", 0)

        mean_night = row.get("mean_night")
        n_night = row.get("n_scenes_night", 0)

        mean_rain = row.get("mean_rain")
        n_rain = row.get("n_scenes_rain", 0)

        # Validate day baseline
        if pd.isna(mean_day) or mean_day == 0:
            print(f"  Warning: {encoder}/{metric} has invalid day baseline (mean={mean_day}), skipping")
            continue

        if n_day < 5:
            print(f"  Warning: {encoder}/{metric} has only {n_day} day scenes (recommend ≥5)")

        # Compute ratios
        ratio_night_day = mean_night / mean_day if not pd.isna(mean_night) else None
        ratio_rain_day = mean_rain / mean_day if not pd.isna(mean_rain) else None

        results.append({
            "encoder": encoder,
            "metric": metric,
            "rmse_night": mean_night if not pd.isna(mean_night) else None,
            "rmse_rain": mean_rain if not pd.isna(mean_rain) else None,
            "rmse_day": mean_day,
            "ratio_night_day": ratio_night_day,
            "ratio_rain_day": ratio_rain_day,
            "n_night": int(n_night) if not pd.isna(n_night) else 0,
            "n_rain": int(n_rain) if not pd.isna(n_rain) else 0,
            "n_day": int(n_day),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["encoder", "metric"]).reset_index(drop=True)

    print(f"\nComputed {len(results_df)} rows")

    # Log scene count distribution
    if len(results_df) > 0:
        print("\nScene count summary:")
        print(f"  Night scenes: {results_df['n_night'].iloc[0]}")
        print(f"  Rain scenes: {results_df['n_rain'].iloc[0]}")
        print(f"  Day scenes: {results_df['n_day'].iloc[0]}")
        total = results_df['n_night'].iloc[0] + results_df['n_rain'].iloc[0] + results_df['n_day'].iloc[0]
        print(f"  Total: {total} (should be 40 for p0_test)")

    # Write output
    output_path = Path("outputs/analysis/environmental_robustness.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n✓ Wrote {output_path}")
    print(f"  Total rows: {len(results_df)}")

    # Show sample
    print("\nSample output:")
    display_cols = ["encoder", "metric", "ratio_night_day", "ratio_rain_day", "n_night", "n_rain", "n_day"]
    if len(results_df) > 0:
        print(results_df[display_cols].head(10).to_string(index=False))

        # Summary statistics
        print("\nRatio summary (mean across encoders):")
        if "ratio_night_day" in results_df.columns and results_df["ratio_night_day"].notna().any():
            print(f"  Night/Day: {results_df['ratio_night_day'].mean():.3f} (higher = worse in night)")
        if "ratio_rain_day" in results_df.columns and results_df["ratio_rain_day"].notna().any():
            print(f"  Rain/Day:  {results_df['ratio_rain_day'].mean():.3f} (higher = worse in rain)")


if __name__ == "__main__":
    main()
