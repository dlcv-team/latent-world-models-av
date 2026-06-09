#!/usr/bin/env python3
"""Generate per_scenario_rmse.csv from probe results (B6.5).

Reads outputs/probes/*/per_scene_rmse.csv (columns: steer_rmse, accel_rmse
with normalized values in [-1, 1] space), classifies scenes by scenario
(highway/urban/intersection/other), and computes per-scenario RMSE with bootstrap
confidence intervals. Converts to physical units (degrees, m/s²) when writing output CSV.

Writes to outputs/analysis/per_scenario_rmse.csv to match analysis.paired_tests output location.

Usage:
    python scripts/generate_per_scenario_from_probes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from nuscenes.nuscenes import NuScenes

from config import load_canonical
from evaluation.metrics import bootstrap_mean_ci, classify_scenes_by_scenario, denormalize_rmse_dataframe


def main() -> int:
    print("Generating per_scenario_rmse.csv from probe results...")

    # Load config
    cfg = load_canonical()

    # Load NuScenes for scene classification
    nuscenes_root = cfg.root / "data"
    version = cfg.raw["dataset"]["version"]
    print(f"\nLoading NuScenes {version}...")
    nusc = NuScenes(version=version, dataroot=str(nuscenes_root), verbose=False)

    # Load probe results
    probe_root = Path("outputs/probes")
    if not probe_root.exists():
        print(f"ERROR: {probe_root} does not exist", file=sys.stderr)
        return 1

    encoders = sorted([d.name for d in probe_root.iterdir() if d.is_dir()])
    if not encoders:
        print(f"ERROR: No encoder directories found in {probe_root}", file=sys.stderr)
        return 1

    print(f"Found {len(encoders)} encoders: {encoders}")

    # Collect all predictions
    all_predictions = []
    missing_csvs = []

    for encoder in encoders:
        csv_path = probe_root / encoder / "per_scene_rmse.csv"
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found", file=sys.stderr)
            missing_csvs.append(encoder)
            continue

        df = pd.read_csv(csv_path)
        print(f"  {encoder}: {len(df)} scenes")

        # Convert to predictions format (values are normalized)
        for _, row in df.iterrows():
            # CSV values are normalized in [-1, 1] space
            all_predictions.append({
                "encoder": encoder,
                "scene_name": row["scene_name"],
                "steer_rmse": row["steer_rmse"],
                "accel_rmse": row["accel_rmse"],
            })

    if missing_csvs:
        print(f"ERROR: Missing per_scene_rmse.csv for encoders: {missing_csvs}", file=sys.stderr)
        return 1

    if not all_predictions:
        print("ERROR: No predictions collected from any encoder", file=sys.stderr)
        return 1

    predictions_df = pd.DataFrame(all_predictions)
    print(f"\nTotal prediction rows: {len(predictions_df)}")

    # Classify scenes by scenario
    print("\nClassifying scenes by scenario...")
    unique_scene_names = predictions_df["scene_name"].unique().tolist()

    # Convert scene names to tokens
    scene_name_to_token = {}
    for scene in nusc.scene:
        scene_name_to_token[scene["name"]] = scene["token"]

    scene_tokens = [scene_name_to_token[name] for name in unique_scene_names if name in scene_name_to_token]
    print(f"Mapped {len(scene_tokens)} scene names to tokens")

    token_to_bucket = classify_scenes_by_scenario(nusc, scene_tokens)

    # Convert back to scene_name → bucket mapping
    token_to_name = {v: k for k, v in scene_name_to_token.items()}
    scene_to_bucket = {token_to_name[token]: bucket for token, bucket in token_to_bucket.items()}

    scenario_counts = {}
    for bucket in scene_to_bucket.values():
        scenario_counts[bucket] = scenario_counts.get(bucket, 0) + 1
    print(f"Scenario distribution: {scenario_counts}")

    # Compute per-scenario RMSE with bootstrap CIs (read params from canonical config)
    # Note: Values are normalized in [-1, 1] space; will convert to physical
    # units (degrees, m/s²) before writing output CSV
    print("\nComputing per-scenario aggregates with bootstrap CIs...")

    bootstrap_cfg = cfg.raw["evaluation"]["bootstrap"]
    n_resamples = bootstrap_cfg["n_resamples"]
    bootstrap_seed = bootstrap_cfg["seed"]
    confidence_level = bootstrap_cfg["confidence_level"]

    results = []
    for encoder in encoders:
        encoder_df = predictions_df[predictions_df["encoder"] == encoder]

        # Aggregate RMSE values by scenario (values in normalized space)
        for metric_name, col_name in [("steer_rmse", "steer_rmse"),
                                        ("accel_rmse", "accel_rmse")]:
            # Group by scenario
            for scenario in set(scene_to_bucket.values()):
                # Get scenes in this scenario
                scenario_scenes = [name for name, bucket in scene_to_bucket.items() if bucket == scenario]
                scenario_df = encoder_df[encoder_df["scene_name"].isin(scenario_scenes)]

                if len(scenario_df) == 0:
                    continue

                rmse_values = scenario_df[col_name].values
                n_scenes = len(rmse_values)

                mean, ci_lo, ci_hi = bootstrap_mean_ci(
                    rmse_values,
                    n_resamples=n_resamples,
                    seed=bootstrap_seed,
                    confidence_level=confidence_level,
                )

                results.append({
                    "encoder": encoder,
                    "scenario": scenario,
                    "metric": metric_name,
                    "n_scenes": n_scenes,
                    "mean": mean,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                })

    if not results:
        print("ERROR: No per-scenario results computed", file=sys.stderr)
        return 1

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["encoder", "scenario", "metric"]).reset_index(drop=True)

    print("\nConverting to physical units for output CSV...")
    results_df = denormalize_rmse_dataframe(results_df, cfg)

    # Validate results_df is not empty and contains valid data
    if len(results_df) == 0:
        print("ERROR: results_df is empty after denormalization", file=sys.stderr)
        return 1

    # Check for all-None/all-NaN in critical columns
    if results_df["mean"].isna().all():
        print("ERROR: All mean values are NaN in results_df", file=sys.stderr)
        return 1

    print(f"Computed {len(results_df)} rows:")
    print(f"  Encoders: {sorted(results_df['encoder'].unique())}")
    print(f"  Scenarios: {sorted(results_df['scenario'].unique())}")
    print(f"  Metrics: {sorted(results_df['metric'].unique())}")

    # Write output
    output_path = Path("outputs/analysis/per_scenario_rmse.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n✓ Wrote {output_path}")
    print(f"  Total rows: {len(results_df)}")

    # Show sample
    print("\nSample (first 10 rows):")
    print(results_df.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
